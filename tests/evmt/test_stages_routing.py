import unittest

import torch

from evmt.memory import ClassBalancedMemory
from evmt.reliability import ReliabilityRouter
from evmt.stages import AdaptationStage, StageController


class StageRoutingTests(unittest.TestCase):
    def test_stage_never_forces_full_without_memory(self):
        stages = StageController(
            warmup_batches=20,
            coverage_window=5,
            min_pred_classes=24,
            min_memory_classes=16,
            min_entries_per_class=2,
        )

        self.assertEqual(
            stages.update(20, 23, 0, 0), AdaptationStage.BN_WARMUP
        )
        self.assertEqual(
            stages.update(21, 24, 0, 0), AdaptationStage.EVIDENCE_BUILD
        )
        self.assertEqual(
            stages.update(100, 32, 15, 32), AdaptationStage.EVIDENCE_BUILD
        )
        self.assertEqual(stages.update(101, 32, 16, 2), AdaptationStage.FULL)

    def test_stage_transitions_are_monotonic(self):
        stages = StageController(1, 2, 2, 1, 1)
        self.assertEqual(stages.update(2, 2, 0, 0), AdaptationStage.EVIDENCE_BUILD)
        self.assertEqual(stages.update(3, 0, 1, 1), AdaptationStage.FULL)
        self.assertEqual(stages.update(4, 0, 0, 0), AdaptationStage.FULL)

    def test_route_without_evidence_uses_confidence_and_agreement(self):
        router = ReliabilityRouter(3, min_conf=0.2, max_js=0.2)
        q = torch.tensor([[0.9, 0.05, 0.05], [0.6, 0.3, 0.1]])
        js = torch.tensor([0.0, 0.1])

        out = router.route(q, js, margin_drop=None)

        self.assertTrue(
            torch.allclose(out.reliability, out.confidence * out.agreement)
        )
        self.assertTrue(torch.equal(out.evidence, torch.ones_like(out.confidence)))

    def test_memory_stats_report_only_queue_occupancy(self):
        memory = ClassBalancedMemory(3, 4, 2)
        memory.add(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.0, 0.1, 0.9]]),
            torch.ones(3),
            torch.ones(3, dtype=torch.bool),
            step=1,
        )

        stats = memory.stats()

        self.assertEqual(stats.total_entries, 3)
        self.assertEqual(stats.covered_classes, 2)
        self.assertEqual(stats.min_positive_size, 1)
        self.assertEqual(stats.max_size, 2)


if __name__ == "__main__":
    unittest.main()
