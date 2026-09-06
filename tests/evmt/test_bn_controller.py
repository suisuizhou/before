import unittest

import torch
import torch.nn as nn

from evmt.bn import TargetBNController


class BNControllerTests(unittest.TestCase):
    def test_online_moments_match_concatenated_tensor(self):
        model = nn.Sequential(nn.BatchNorm1d(2))
        controller = TargetBNController(
            model, blend_batches=1, max_target_weight=1.0
        )
        x1 = torch.tensor([[[1.0, 3.0], [2.0, 4.0]],
                           [[3.0, 5.0], [4.0, 6.0]]])
        x2 = torch.tensor([[[5.0, 7.0], [6.0, 8.0]],
                           [[7.0, 9.0], [8.0, 10.0]]])

        controller.observe_batch(model, x1, lambda m, x: m(x))
        controller.observe_batch(model, x2, lambda m, x: m(x))

        expected = torch.cat([x1, x2], dim=0).permute(1, 0, 2).reshape(2, -1)
        state = controller.layer_state("0")
        self.assertEqual(state.target_count, expected.shape[1])
        self.assertTrue(torch.allclose(state.target_mean, expected.mean(dim=1)))
        self.assertTrue(torch.allclose(
            state.target_m2 / state.target_count,
            expected.var(dim=1, unbiased=False),
        ))

    def test_prediction_context_does_not_observe_current_batch(self):
        model = nn.Sequential(nn.BatchNorm1d(2))
        controller = TargetBNController(model)
        source_mean = controller.source_state("0").mean.clone()
        source_var = controller.source_state("0").var.clone()

        with controller.prediction_stats():
            _ = model(torch.randn(8, 2, 4))

        self.assertEqual(controller.seen_batches, 0)
        self.assertTrue(torch.equal(source_mean, controller.source_state("0").mean))
        self.assertTrue(torch.equal(source_var, controller.source_state("0").var))
        self.assertTrue(torch.equal(model[0].running_mean, source_mean))
        self.assertTrue(torch.equal(model[0].running_var, source_var))

    def test_fused_stats_use_only_previously_observed_batches(self):
        model = nn.Sequential(nn.BatchNorm1d(1))
        controller = TargetBNController(
            model, blend_batches=2, max_target_weight=1.0
        )
        controller.observe_batch(
            model, torch.full((4, 1, 3), 4.0), lambda m, x: m(x)
        )

        fused = controller.fused_state("0")
        self.assertEqual(controller.seen_batches, 1)
        self.assertAlmostEqual(controller.target_weight, 0.5)
        self.assertTrue(torch.allclose(fused.mean, torch.tensor([2.0])))
        self.assertTrue(torch.isfinite(fused.var).all())


if __name__ == "__main__":
    unittest.main()
