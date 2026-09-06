import unittest

import torch
import torch.nn as nn

from evmt.bn import TargetBNController


class BNBackwardTests(unittest.TestCase):
    def test_prediction_context_preserves_backward_saved_tensors(self):
        model = nn.Sequential(nn.BatchNorm1d(2), nn.Flatten(), nn.Linear(8, 1))
        controller = TargetBNController(model)
        x = torch.randn(4, 2, 4)

        with controller.prediction_stats():
            output = model(x).sum()
        output.backward()

        self.assertTrue(torch.isfinite(model[0].weight.grad).all())


if __name__ == "__main__":
    unittest.main()
