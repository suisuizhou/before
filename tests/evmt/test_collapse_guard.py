import unittest

import torch

from evmt.guards import CollapseGuard


class CollapseGuardTests(unittest.TestCase):
    def test_guard_rejects_single_class_collapse(self):
        logits = torch.full((64, 32), -20.0)
        logits[:, 0] = 20.0

        decision = CollapseGuard(32).check(logits)

        self.assertFalse(decision.allow_update)
        self.assertIn("max_class_share", decision.reasons)
        self.assertIn("effective_classes", decision.reasons)

    def test_guard_accepts_balanced_finite_logits(self):
        logits = torch.eye(32).repeat(2, 1) * 5.0

        decision = CollapseGuard(32).check(logits)

        self.assertTrue(decision.allow_update)
        self.assertEqual(decision.reasons, ())

    def test_guard_rejects_relative_effective_class_drop(self):
        guard = CollapseGuard(4, min_effective_classes=1, max_class_share=1.0)
        self.assertTrue(guard.check(torch.eye(4).repeat(4, 1) * 8.0).allow_update)
        collapsed = torch.tensor([[8.0, 0.0, 0.0, 0.0]]).repeat(16, 1)

        decision = guard.check(collapsed)

        self.assertIn("relative_effective_drop", decision.reasons)


if __name__ == "__main__":
    unittest.main()
