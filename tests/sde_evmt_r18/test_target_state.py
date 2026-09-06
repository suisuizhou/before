import unittest

import torch
import torch.nn.functional as F


class TargetStateTests(unittest.TestCase):
    def test_memory_is_class_balanced_fifo_and_prototypes_are_reliability_weighted(self):
        from sde_evmt_r18.memory import ClassBalancedMemory

        memory = ClassBalancedMemory(num_classes=2, capacity=2, feature_dim=2)
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        q = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        reliability = torch.tensor([0.1, 0.25, 0.75])
        memory.add(features, q, reliability, torch.ones(3, dtype=torch.bool), step=0)
        memory.add(
            torch.tensor([[-1.0, 0.0]]),
            torch.tensor([[0.1, 0.9]]),
            torch.tensor([1.0]),
            torch.tensor([True]),
            step=1,
        )

        self.assertEqual(memory.stats().sizes, (2, 1))
        prototypes, classes = memory.prototypes()
        class_zero = prototypes[(classes == 0).nonzero().item()]
        expected = F.normalize(
            0.25 * F.normalize(features[1], dim=0)
            + 0.75 * F.normalize(features[2], dim=0),
            dim=0,
        )
        self.assertTrue(torch.allclose(class_zero, expected, atol=1e-6))

    def test_teacher_prior_tracks_nonuniform_predictions(self):
        from sde_evmt_r18.target_losses import EMATeacherPrior

        prior = EMATeacherPrior(num_classes=3, momentum=0.5)
        prior.update(torch.tensor([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]))

        self.assertAlmostEqual(float(prior.value.sum()), 1.0, places=6)
        self.assertGreater(float(prior.value[0]), float(prior.value[1]))

    def test_scheduler_has_exact_five_batch_warmup_and_monotonic_stages(self):
        from sde_evmt_r18.scheduler import OnlineScheduler, TargetStage

        scheduler = OnlineScheduler(warmup_batches=5)
        empty = {"covered_classes": 0, "total_entries": 0}
        for step in range(5):
            self.assertEqual(
                scheduler.update(step, empty, empty, [0.4]),
                TargetStage.WARMUP,
            )
        self.assertEqual(
            scheduler.update(5, empty, empty, [0.4, 0.41]),
            TargetStage.EVIDENCE_BUILD,
        )
        ready = {"covered_classes": 2, "total_entries": 4}
        self.assertEqual(
            scheduler.update(6, ready, empty, [0.40, 0.40, 0.41]),
            TargetStage.FULL,
        )
        self.assertEqual(
            scheduler.update(7, empty, empty, [0.4, 0.3, 0.2]),
            TargetStage.FULL,
        )

    def test_neighborhood_loss_skips_low_similarity_and_uses_close_neighbors(self):
        from sde_evmt_r18.memory import ClassBalancedMemory
        from sde_evmt_r18.target_losses import uncertain_neighborhood_kl

        memory = ClassBalancedMemory(num_classes=2, capacity=2, feature_dim=2)
        memory.add(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1.0]),
            torch.tensor([True]),
            step=0,
        )
        logits = torch.zeros(1, 2, requires_grad=True)
        uncertain = torch.tensor([True])
        skipped = uncertain_neighborhood_kl(
            logits,
            torch.tensor([[-1.0, 0.0]]),
            uncertain,
            memory,
            neighbors=1,
            similarity_threshold=0.3,
        )
        self.assertEqual(float(skipped.detach()), 0.0)
        skipped.backward()
        self.assertIsNotNone(logits.grad)

        used = uncertain_neighborhood_kl(
            torch.zeros(1, 2, requires_grad=True),
            torch.tensor([[1.0, 0.0]]),
            uncertain,
            memory,
            neighbors=1,
            similarity_threshold=0.3,
        )
        self.assertGreater(float(used.detach()), 0.0)

    def test_target_objective_empty_branches_are_finite_and_backward_safe(self):
        from sde_evmt_r18.memory import ClassBalancedMemory
        from sde_evmt_r18.model import SDEEVMTResNet18
        from sde_evmt_r18.target_losses import EMATeacherPrior, target_objective

        model = SDEEVMTResNet18(num_classes=3)
        logits = torch.randn(2, 3, requires_grad=True)
        features = torch.randn(2, 256, requires_grad=True)
        teacher_q = torch.softmax(torch.randn(2, 3), dim=1)
        reliability = torch.tensor([0.4, 0.8])
        losses = target_objective(
            model=model,
            student_logits=logits,
            student_features=features,
            teacher_q=teacher_q,
            reliability=reliability,
            certain=torch.zeros(2, dtype=torch.bool),
            uncertain=torch.ones(2, dtype=torch.bool),
            memory=ClassBalancedMemory(3, 2, 256),
            prior=EMATeacherPrior(3),
            cfg={},
            feature_anchor=None,
        )

        self.assertTrue(torch.isfinite(losses.total))
        self.assertEqual(float(losses.pcl.detach()), 0.0)
        self.assertEqual(float(losses.ncl.detach()), 0.0)
        losses.total.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
