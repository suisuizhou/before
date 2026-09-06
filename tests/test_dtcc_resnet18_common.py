import torch
import torch.nn as nn

from Lib.dtcc_resnet18_common import (
    DtCCMemoryBank,
    balance_probabilities,
    configure_bn_only,
    dtcc_ncl_loss,
    dtcc_pcl_loss,
    dynamic_data_division,
    spectral_entropy,
)


def test_spectral_entropy_ranks_flat_spectrum_as_more_complex():
    peaked = torch.zeros(2, 1, 8)
    peaked[:, :, 2] = 1.0
    flat = torch.ones(2, 1, 8)
    assert torch.all(spectral_entropy(flat) > spectral_entropy(peaked))


def test_dynamic_data_division_uses_batch_mean_confidence_and_entropy():
    probs = torch.tensor([
        [0.90, 0.10],
        [0.60, 0.40],
        [0.55, 0.45],
        [0.95, 0.05],
    ])
    entropy = torch.tensor([0.10, 0.20, 0.80, 0.90])
    certain, uncertain = dynamic_data_division(probs, entropy)
    assert certain.tolist() == [True, False, False, False]
    assert torch.equal(uncertain, ~certain)


def test_memory_bank_slims_each_class_by_confidence():
    classifier_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    classifier = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(classifier_weight)
    memory = DtCCMemoryBank.from_classifier(classifier, num_classes=2)

    features = torch.tensor([[2.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    probs = torch.tensor([[0.7, 0.3], [0.9, 0.1], [0.2, 0.8]])
    certain = torch.tensor([True, True, True])
    memory.update(features, probs, certain)
    memory.slim(max_per_class=2)

    labels = memory.labels.argmax(dim=1)
    assert int((labels == 0).sum()) == 2
    assert int((labels == 1).sum()) == 2
    class0_scores = memory.scores[labels == 0].max(dim=1).values
    assert torch.all(class0_scores[:-1] >= class0_scores[1:])


def test_balance_probabilities_matches_dtcc_frequency_rule():
    probs = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
    certain = torch.tensor([True, True, False])
    balanced = balance_probabilities(probs, certain)
    expected = probs / torch.tensor([[3.0, 1.0]])
    assert torch.allclose(balanced, expected)


def test_pcl_and_ncl_losses_are_finite_and_differentiable():
    certain_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    uncertain_features = torch.tensor([[0.7, 0.3], [0.3, 0.7]], requires_grad=True)
    prototypes = torch.eye(2)
    labels = torch.tensor([0, 1])
    pcl = dtcc_pcl_loss(certain_features, uncertain_features, prototypes, labels)

    supports = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    uncertain_probs = torch.softmax(torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True), dim=1)
    ncl = dtcc_ncl_loss(
        uncertain_features,
        supports,
        scores,
        neighbor_k=1,
        probs_uncertain=uncertain_probs,
        temperature=0.1,
    )
    total = pcl + ncl
    assert torch.isfinite(total)
    total.backward()
    assert certain_features.grad is not None


def test_configure_bn_only_freezes_non_bn_parameters_and_discards_running_stats():
    model = nn.Sequential(
        nn.Conv1d(1, 4, 3),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.AdaptiveAvgPool1d(1),
    )
    params = configure_bn_only(model)
    assert len(params) == 2
    assert model[0].weight.requires_grad is False
    assert model[1].weight.requires_grad is True
    assert model[1].running_mean is None
    assert model[1].running_var is None
    assert model[3].training is False


class _ClassifierWithMisalignedSelfPrediction(nn.Module):
    """Expose class weights but deliberately swap their predicted classes."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            requires_grad=False,
        )

    def forward(self, x):
        assert x.shape == (2, 2)
        return x.new_tensor([[0.0, 5.0], [5.0, 0.0]])


def test_memory_initialization_assigns_classifier_weight_k_to_class_k():
    classifier = _ClassifierWithMisalignedSelfPrediction()
    memory = DtCCMemoryBank.from_classifier(classifier, num_classes=2)

    assert torch.equal(memory.labels, torch.eye(2))
    expected_scores = torch.softmax(classifier(memory.supports), dim=1)
    assert torch.allclose(memory.scores, expected_scores)
    assert memory.covered_classes() == 2
