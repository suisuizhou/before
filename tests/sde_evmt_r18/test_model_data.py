import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from sde_evmt_r18.data import load_pu4d_domain
from sde_evmt_r18.model import (
    GatedFrequencyWarp,
    GatedSpectralAdapter,
    RobustSpectrumNorm,
    SDEEVMTResNet18,
)


class ModelDataTests(unittest.TestCase):
    def test_robust_norm_is_finite_and_sample_centered(self):
        norm = RobustSpectrumNorm()
        y = norm(torch.rand(4, 512) * 100.0)

        self.assertEqual(tuple(y.shape), (4, 1, 512))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(
            torch.allclose(y.median(-1).values, torch.zeros(4, 1), atol=1e-5)
        )

    def test_gated_input_modules_start_at_exact_identity(self):
        x = torch.rand(2, 1, 512)
        warp = GatedFrequencyWarp(initial_gate=0.05)
        adapter = GatedSpectralAdapter(initial_gate=0.1)

        self.assertTrue(torch.allclose(warp(x), x, atol=1e-6))
        self.assertTrue(torch.allclose(adapter(x), x, atol=1e-6))
        self.assertAlmostEqual(warp.gate_value, 0.05, places=5)
        self.assertAlmostEqual(adapter.gate_value, 0.1, places=5)

    def test_model_uses_groupnorm_and_returns_expected_shapes(self):
        model = SDEEVMTResNet18(num_classes=32)

        feature, logits = model.forward_parts(torch.rand(3, 512))

        self.assertEqual(tuple(feature.shape), (3, 256))
        self.assertEqual(tuple(logits.shape), (3, 32))
        self.assertTrue(any(isinstance(module, nn.GroupNorm) for module in model.modules()))
        self.assertFalse(any(isinstance(module, nn.BatchNorm1d) for module in model.modules()))

    def test_cache_loader_keeps_raw_fft_and_validates_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = torch.rand(5, 512)
            y = torch.tensor([0, 1, 2, 3, 31])
            torch.save(
                {"x": x, "y": y, "domain": torch.zeros(5, dtype=torch.long)},
                root / "D1_1500_0.7_1000_fft.pt",
            )

            dataset = load_pu4d_domain(root, 0)

            self.assertEqual(len(dataset), 5)
            self.assertTrue(torch.equal(dataset.tensors[0], x))
            self.assertTrue(torch.equal(dataset.tensors[1], y))
            with self.assertRaisesRegex(ValueError, "domain must be one of"):
                load_pu4d_domain(root, 4)


if __name__ == "__main__":
    unittest.main()
