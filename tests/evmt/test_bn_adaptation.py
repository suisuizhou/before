import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from evmt.ema import adaptable_state, ema_update_
from evmt.regularization import (
    adapter_reg_loss,
    bn_affine_anchor,
    bn_anchor_loss,
    warp_reg_loss,
)
from main_tta_evmt import freeze_bn_adapter_warp, build_bnfirst_optimizer


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_scale = nn.Parameter(torch.zeros(1, 1, 4))
        self.band_bias = nn.Parameter(torch.zeros(1, 1, 4))
        self.warp_ctrl = nn.Parameter(torch.zeros(1, 1, 3))
        self.adapter_delta = 0.1
        self.input_len = 8
        self.max_warp = 2.0
        self.conv = nn.Conv1d(1, 2, 1, bias=False)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(self.conv(x)).mean(dim=-1)


def toy_model():
    return nn.Sequential(
        ToyBackbone(),
        nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU()),
        nn.Linear(2, 3),
    )


class BNAdaptationTests(unittest.TestCase):
    def test_only_bn_affine_adapter_and_warp_are_trainable(self):
        model = toy_model()
        names = freeze_bn_adapter_warp(model)

        self.assertIn("0.bn.weight", names)
        self.assertIn("0.bn.bias", names)
        self.assertIn("1.1.weight", names)
        self.assertIn("0.band_scale", names)
        self.assertIn("0.band_bias", names)
        self.assertIn("0.warp_ctrl", names)
        self.assertFalse(model[0].conv.weight.requires_grad)
        self.assertFalse(model[2].weight.requires_grad)
        self.assertEqual(set(adaptable_state(model)), set(names))

    def test_ema_updates_bn_affine_but_not_running_stats(self):
        student = toy_model()
        teacher = copy.deepcopy(student)
        before_running = teacher[0].bn.running_mean.clone()
        teacher_weight_before = teacher[0].bn.weight.clone()
        student[0].bn.weight.data.add_(1.0)

        ema_update_(teacher, student, beta=0.5, include_bn_affine=True)

        self.assertTrue(torch.equal(before_running, teacher[0].bn.running_mean))
        expected = teacher_weight_before + 0.5
        self.assertTrue(torch.allclose(teacher[0].bn.weight, expected))

    def test_optimizer_has_scaled_bn_adapter_and_warp_groups(self):
        model = toy_model()
        freeze_bn_adapter_warp(model)
        cfg = SimpleNamespace(
            Opt=SimpleNamespace(lr_tar=1e-3, weight_decay_tar=1e-4),
            TTA=SimpleNamespace(
                bn_lr_scale=0.1,
                adapter_lr_scale=1.0,
                warp_lr_scale=0.2,
            ),
        )

        optimizer = build_bnfirst_optimizer(model, cfg)

        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["bn_affine", "adapter", "warp"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1e-4, 1e-3, 2e-4],
        )

    def test_regularizers_are_finite_and_anchor_bn_affine(self):
        model = toy_model()
        anchor = bn_affine_anchor(model)
        model[0].bn.weight.data.add_(0.25)
        model[0].band_scale.data.add_(0.5)
        model[0].band_bias.data.add_(0.2)
        model[0].warp_ctrl.data.add_(0.1)

        losses = [
            bn_anchor_loss(model, anchor),
            adapter_reg_loss(model),
            warp_reg_loss(model),
        ]

        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        self.assertTrue(all(float(loss) > 0 for loss in losses))
        sum(losses).backward()
        self.assertIsNotNone(model[0].bn.weight.grad)
        self.assertIsNotNone(model[0].band_scale.grad)
        self.assertIsNotNone(model[0].warp_ctrl.grad)


if __name__ == "__main__":
    unittest.main()
