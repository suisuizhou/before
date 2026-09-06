import unittest

import torch
import torch.nn as nn

from sde_evmt_r18.augment import SourceAugmenter
from sde_evmt_r18.source import source_objective


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(512, 32), nn.ReLU(), nn.Linear(32, 8))
        self.classifier = nn.Linear(8, 4)

    def forward_parts(self, x, use_feature_adapter=True):
        if x.ndim == 3:
            x = x.squeeze(1)
        feature = self.encoder(x)
        return feature, self.classifier(feature)


def config(variant="R1"):
    return {
        "variant": variant,
        "augmentation": {
            "style_prob": 1.0,
            "style_strength_min": 0.05,
            "style_strength_max": 0.2,
            "style_knots": 8,
            "warp_prob": 1.0,
            "warp_knots": 8,
            "warp_max_min": 0.3,
            "warp_max_max": 2.0,
            "noise_prob": 1.0,
            "gaussian_snr_min": 25.0,
            "gaussian_snr_max": 40.0,
            "uniform_scale_min": 0.02,
            "uniform_scale_max": 0.08,
            "impulse_prob_min": 0.01,
            "impulse_prob_max": 0.05,
        },
        "source": {
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2,
            "mixup_prob": 0.0,
        },
        "loss": {
            "lambda_style": 0.5,
            "lambda_warp": 0.5,
            "lambda_noise": 0.5,
            "lambda_pred_cons": 0.2,
            "lambda_feat_cons": 0.05,
        },
    }


class SourceObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.x = torch.rand(6, 512) * 10.0
        self.y = torch.arange(6) % 4

    def test_views_are_reproducible_and_shape_preserving(self):
        aug1 = SourceAugmenter(config(), torch.Generator().manual_seed(9))
        aug2 = SourceAugmenter(config(), torch.Generator().manual_seed(9))
        views1 = aug1.views(self.x)
        views2 = aug2.views(self.x)
        self.assertEqual(set(views1), {"clean", "style", "warp", "noise"})
        for name in views1:
            self.assertEqual(tuple(views1[name].shape), tuple(self.x.shape))
            self.assertTrue(torch.allclose(views1[name], views2[name]))
            self.assertTrue(torch.isfinite(views1[name]).all())

    def test_r0_uses_only_clean_classification(self):
        model = TinyModel()
        aug = SourceAugmenter(config("R0"), torch.Generator().manual_seed(2))
        losses = source_objective(model, self.x, self.y, aug, config("R0"))
        self.assertEqual(float(losses.style_cls), 0.0)
        self.assertEqual(float(losses.pred_cons), 0.0)
        self.assertTrue(torch.allclose(losses.total, losses.clean_cls))

    def test_r1_objective_is_finite_and_backward_safe(self):
        model = TinyModel()
        aug = SourceAugmenter(config(), torch.Generator().manual_seed(3))
        losses = source_objective(model, self.x, self.y, aug, config())
        losses.total.backward()
        self.assertTrue(torch.isfinite(losses.total))
        self.assertGreater(float(losses.pred_cons), 0.0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

