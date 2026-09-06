from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .augment import cfg_get


class EMATeacherPrior:
    def __init__(self, num_classes, momentum=0.9):
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self._value = torch.full((self.num_classes,), 1.0 / self.num_classes)

    @property
    def value(self):
        return self._value

    @torch.no_grad()
    def update(self, q):
        batch_prior = q.detach().mean(dim=0)
        batch_prior = batch_prior / batch_prior.sum().clamp_min(1e-8)
        self._value = self._value.to(batch_prior)
        self._value.mul_(self.momentum).add_(batch_prior, alpha=1.0 - self.momentum)
        self._value.div_(self._value.sum().clamp_min(1e-8))
        return self._value


@dataclass
class TargetLosses:
    total: torch.Tensor
    mt: torch.Tensor
    sem: torch.Tensor
    diversity: torch.Tensor
    pcl: torch.Tensor
    ncl: torch.Tensor
    warp_reg: torch.Tensor
    spectral_reg: torch.Tensor
    feature_anchor: torch.Tensor

    def detached(self):
        return {name: float(value.detach()) for name, value in self.__dict__.items()}


def mean_teacher_kl(student_logits, teacher_q, reliability):
    each = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_q.detach(),
        reduction="none",
    ).sum(dim=1)
    weights = reliability.detach()
    return (weights * each).sum() / weights.sum().clamp_min(1e-8)


def weighted_tsallis(student_logits, reliability, eta=0.1, alpha=2.0):
    probability = student_logits.softmax(dim=1).clamp_min(1e-8)
    weights = float(eta) + (1.0 - float(eta)) * reliability.detach()
    if abs(float(alpha) - 1.0) < 1e-6:
        sample = -(probability * probability.log()).sum(dim=1)
    else:
        sample = (1.0 - probability.pow(float(alpha)).sum(dim=1)) / (float(alpha) - 1.0)
    return (weights * sample).sum() / weights.sum().clamp_min(1e-8)


def prior_diversity(student_logits, prior):
    mean_probability = student_logits.softmax(dim=1).mean(dim=0).clamp_min(1e-8)
    target = prior.detach().to(mean_probability).clamp_min(1e-8)
    target = target / target.sum()
    return (target * (target.log() - mean_probability.log())).sum()


def prototype_contrastive(features, pseudo, certain, memory, temperature=0.1):
    zero = features.sum() * 0.0
    if not bool(certain.any()):
        return zero
    prototypes, classes = memory.prototypes()
    if not len(prototypes):
        return zero
    class_to_index = {int(label): index for index, label in enumerate(classes.tolist())}
    rows = [
        int(row)
        for row in certain.nonzero(as_tuple=False).flatten()
        if int(pseudo[row]) in class_to_index
    ]
    if not rows:
        return zero
    row_tensor = torch.tensor(rows, dtype=torch.long, device=features.device)
    target = torch.tensor(
        [class_to_index[int(pseudo[row])] for row in rows],
        dtype=torch.long,
        device=features.device,
    )
    scores = F.normalize(features[row_tensor], dim=1) @ F.normalize(
        prototypes.to(features), dim=1
    ).T
    return F.cross_entropy(scores / float(temperature), target)


def uncertain_neighborhood_kl(
    logits,
    features,
    uncertain,
    memory,
    neighbors=5,
    similarity_threshold=0.3,
):
    zero = logits.sum() * 0.0
    if not bool(uncertain.any()):
        return zero
    rows = uncertain.nonzero(as_tuple=False).flatten()
    similarity, q, reliability = memory.neighbors(features[rows].detach(), neighbors)
    if not similarity.shape[1]:
        return zero
    valid = similarity.mean(dim=1) >= float(similarity_threshold)
    if not bool(valid.any()):
        return zero
    similarity = similarity[valid]
    q = q[valid]
    reliability = reliability[valid]
    weights = similarity.clamp_min(0.0) * reliability
    usable = weights.sum(dim=1) > 0
    if not bool(usable.any()):
        return zero
    weights = weights[usable]
    q = q[usable]
    target = (weights[..., None] * q).sum(dim=1) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    selected_rows = rows[valid][usable]
    return F.kl_div(
        F.log_softmax(logits[selected_rows], dim=1),
        target.detach(),
        reduction="batchmean",
    )


def _adapter_regularizers(model, feature_anchor):
    displacement = model.frequency_warp.displacement()
    maximum = max(float(model.frequency_warp.max_warp), 1e-8)
    warp = (displacement / maximum).square().mean()
    if displacement.shape[-1] > 1:
        warp = warp + (
            (displacement[..., 1:] - displacement[..., :-1]) / maximum
        ).square().mean()
    spectral = model.spectral_adapter.band_scale.square().mean()
    spectral = spectral + model.spectral_adapter.band_bias.square().mean()
    feature = sum(parameter.sum() * 0.0 for parameter in model.feature_adapter.parameters())
    if feature_anchor is not None:
        feature = sum(
            (parameter - feature_anchor[name].to(parameter)).square().sum()
            for name, parameter in model.feature_adapter.named_parameters()
        )
    return warp, spectral, feature


def target_objective(
    model,
    student_logits,
    student_features,
    teacher_q,
    reliability,
    certain,
    uncertain,
    memory,
    prior,
    cfg,
    feature_anchor=None,
):
    zero = student_logits.sum() * 0.0
    mt = (
        mean_teacher_kl(student_logits, teacher_q, reliability)
        if bool(cfg_get(cfg, "enable_mt", True))
        else zero
    )
    sem = weighted_tsallis(
        student_logits,
        reliability,
        eta=float(cfg_get(cfg, "loss.uncertain_floor", 0.1)),
        alpha=float(cfg_get(cfg, "loss.tsallis_alpha", 2.0)),
    )
    diversity = prior_diversity(student_logits, prior.value)
    pseudo = teacher_q.detach().argmax(dim=1)
    pcl = (
        prototype_contrastive(
            student_features,
            pseudo,
            certain,
            memory,
            temperature=float(cfg_get(cfg, "loss.pcl_temperature", 0.1)),
        )
        if bool(cfg_get(cfg, "enable_pcl", True))
        else zero
    )
    ncl = (
        uncertain_neighborhood_kl(
            student_logits,
            student_features,
            uncertain,
            memory,
            neighbors=int(cfg_get(cfg, "memory.neighbor_num", 5)),
            similarity_threshold=float(
                cfg_get(cfg, "memory.neighbor_similarity_threshold", 0.3)
            ),
        )
        if bool(cfg_get(cfg, "enable_ncl", True))
        else zero
    )
    warp_reg, spectral_reg, feature_reg = _adapter_regularizers(model, feature_anchor)
    sem_total = sem + float(cfg_get(cfg, "loss.lambda_div", 1.0)) * diversity
    total = (
        float(cfg_get(cfg, "loss.lambda_mt", 1.0)) * mt
        + float(cfg_get(cfg, "loss.lambda_sem", 0.2)) * sem_total
        + float(cfg_get(cfg, "loss.lambda_pcl", 0.1)) * pcl
        + float(cfg_get(cfg, "loss.lambda_ncl", 0.2)) * ncl
        + float(cfg_get(cfg, "loss.lambda_warp_reg", 0.0002)) * warp_reg
        + float(cfg_get(cfg, "loss.lambda_adapter_reg", 0.0001)) * spectral_reg
        + float(cfg_get(cfg, "loss.lambda_feature_anchor", 0.001)) * feature_reg
    )
    return TargetLosses(
        total=total,
        mt=mt,
        sem=sem,
        diversity=diversity,
        pcl=pcl,
        ncl=ncl,
        warp_reg=warp_reg,
        spectral_reg=spectral_reg,
        feature_anchor=feature_reg,
    )
