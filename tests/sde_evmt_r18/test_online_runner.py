import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


class TinyWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_warp = 2.0
        self.warp_ctrl = nn.Parameter(torch.zeros(1, 1, 4))
        self.warp_gate_logit = nn.Parameter(torch.tensor(-2.0))

    def displacement(self):
        return self.warp_gate_logit.sigmoid() * self.warp_ctrl

    def forward(self, x):
        return x + self.displacement().mean()


class TinySpectral(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_scale = nn.Parameter(torch.zeros(1, 1, 4))
        self.band_bias = nn.Parameter(torch.zeros(1, 1, 4))
        self.adapter_gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        gate = self.adapter_gate_logit.sigmoid()
        return x * (1.0 + gate * self.band_scale.mean()) + gate * self.band_bias.mean()


class TinyFeatureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.down = nn.Linear(8, 4)
        self.up = nn.Linear(4, 8)
        self.feature_gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x):
        return x + self.feature_gate_logit.sigmoid() * self.up(
            torch.relu(self.down(self.norm(x)))
        )


class TinySDEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frequency_warp = TinyWarp()
        self.spectral_adapter = TinySpectral()
        self.backbone = nn.Linear(16, 8)
        self.feature_adapter = TinyFeatureAdapter()
        self.classifier = nn.Linear(8, 3)

    def prepare_spectrum(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        return self.spectral_adapter(self.frequency_warp(x))

    def forward_from_spectrum(self, spectrum, use_feature_adapter=True):
        feature = torch.relu(self.backbone(spectrum.flatten(1)))
        if use_feature_adapter:
            feature = self.feature_adapter(feature)
        return feature, self.classifier(feature)

    def forward_parts(self, x, use_feature_adapter=True):
        return self.forward_from_spectrum(
            self.prepare_spectrum(x), use_feature_adapter=use_feature_adapter
        )

    def forward(self, x):
        return self.forward_parts(x)[1]

    def gate_summary(self):
        return {
            "warp_gate": float(self.frequency_warp.warp_gate_logit.sigmoid().detach()),
            "spectral_gate": float(self.spectral_adapter.adapter_gate_logit.sigmoid().detach()),
            "feature_gate": float(self.feature_adapter.feature_gate_logit.sigmoid().detach()),
        }


def config():
    return {
        "warmup_batches": 5,
        "teacher": {"ema_beta": 0.9},
        "teacher_views": {
            "style_strength": 0.01,
            "warp_max": 0.1,
            "noise_snr_min": 40.0,
            "noise_snr_max": 45.0,
        },
        "reliability": {
            "confidence_min": 0.0,
            "gamma": 5.0,
            "min_view_agreement": 0.0,
            "min_class_samples": 1,
        },
        "memory": {"capacity_per_class": 4},
        "optimizer": {
            "lr_warp_adapter": 0.001,
            "lr_feature_adapter": 0.0005,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
    }


class OnlineRunnerTests(unittest.TestCase):
    def test_configure_adaptation_has_exact_variant_and_stage_parameter_sets(self):
        from sde_evmt_r18.model import SDEEVMTResNet18
        from sde_evmt_r18.runner import configure_adaptation
        from sde_evmt_r18.scheduler import TargetStage

        model = SDEEVMTResNet18(num_classes=3)
        input_side = {
            "frequency_warp.warp_ctrl",
            "frequency_warp.warp_gate_logit",
            "spectral_adapter.band_scale",
            "spectral_adapter.band_bias",
            "spectral_adapter.adapter_gate_logit",
        }
        for variant in ("R2", "R3", "R4", "R5", "R6"):
            self.assertEqual(
                set(configure_adaptation(model, variant, TargetStage.WARMUP)),
                input_side,
            )
            expected = input_side
            if variant == "R6":
                expected = input_side | {
                    name for name, _ in model.named_parameters()
                    if name.startswith("feature_adapter.")
                }
            self.assertEqual(
                set(configure_adaptation(model, variant, TargetStage.FULL)),
                expected,
            )
        self.assertFalse(model.classifier.weight.requires_grad)
        with self.assertRaises(ValueError):
            configure_adaptation(model, "R7", TargetStage.FULL)

    def test_optimizer_uses_separate_input_and_feature_learning_rates(self):
        from sde_evmt_r18.runner import build_target_optimizer, configure_adaptation
        from sde_evmt_r18.scheduler import TargetStage

        model = TinySDEModel()
        configure_adaptation(model, "R6", TargetStage.FULL)
        optimizer = build_target_optimizer(model, config())
        self.assertEqual(
            {group["name"]: group["lr"] for group in optimizer.param_groups},
            {"warp_spectral": 0.001, "feature_adapter": 0.0005},
        )

    def test_step_scores_before_update_and_optimizer_steps_at_most_once(self):
        from sde_evmt_r18.runner import SDEEVMTOnlineRunner

        model = TinySDEModel()
        runner = SDEEVMTOnlineRunner(model, variant="R3", cfg=config(), seed=7)
        x = torch.rand(4, 16)
        with torch.no_grad():
            expected = model(x).detach()
        calls = []
        original_step = runner.optimizer.step

        def counted_step(*args, **kwargs):
            calls.append(1)
            return original_step(*args, **kwargs)

        runner.optimizer.step = counted_step
        classifier_before = runner.teacher.classifier.weight.detach().clone()
        metrics = runner.step(x, torch.tensor([0, 1, 2, 0]))

        self.assertTrue(torch.equal(runner.last_preupdate_logits, expected))
        self.assertLessEqual(len(calls), 1)
        self.assertEqual(metrics.samples, 4)
        self.assertEqual(sum(metrics.pred_histogram), 4)
        self.assertTrue(torch.equal(runner.teacher.classifier.weight, classifier_before))

    def test_metric_labels_do_not_change_adaptation_state(self):
        from sde_evmt_r18.runner import SDEEVMTOnlineRunner

        first_model = TinySDEModel()
        second_model = copy.deepcopy(first_model)
        first = SDEEVMTOnlineRunner(first_model, variant="R3", cfg=config(), seed=11)
        second = SDEEVMTOnlineRunner(second_model, variant="R3", cfg=config(), seed=11)
        x = torch.rand(4, 16)
        first.step(x, torch.tensor([0, 0, 0, 0]))
        second.step(x, torch.tensor([1, 2, 1, 2]))

        for left, right in zip(first.student.state_dict().values(), second.student.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(first.teacher.state_dict().values(), second.teacher.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(first.prior.value, second.prior.value))
        self.assertEqual(first.memory.stats(), second.memory.stats())

    def test_nonfinite_loss_skips_optimizer_memory_teacher_and_prior_writes(self):
        from sde_evmt_r18.runner import SDEEVMTOnlineRunner

        runner = SDEEVMTOnlineRunner(TinySDEModel(), variant="R6", cfg=config(), seed=3)
        student_before = copy.deepcopy(runner.student.state_dict())
        teacher_before = copy.deepcopy(runner.teacher.state_dict())
        prior_before = runner.prior.value.clone()
        calls = []
        original_step = runner.optimizer.step

        def counted_step(*args, **kwargs):
            calls.append(1)
            return original_step(*args, **kwargs)

        runner.optimizer.step = counted_step
        bad = SimpleNamespace(
            total=torch.tensor(float("inf"), requires_grad=True),
            mt=torch.tensor(0.0),
            sem=torch.tensor(0.0),
            diversity=torch.tensor(0.0),
            pcl=torch.tensor(0.0),
            ncl=torch.tensor(0.0),
            warp_reg=torch.tensor(0.0),
            spectral_reg=torch.tensor(0.0),
            feature_anchor=torch.tensor(0.0),
        )
        with patch("sde_evmt_r18.runner.target_objective", return_value=bad):
            metrics = runner.step(torch.rand(4, 16))

        self.assertFalse(metrics.updated)
        self.assertIn("non_finite_loss", metrics.skip_reasons)
        self.assertEqual(calls, [])
        self.assertEqual(runner.memory.stats().total_entries, 0)
        self.assertTrue(torch.equal(runner.prior.value, prior_before))
        for name, value in runner.student.state_dict().items():
            self.assertTrue(torch.equal(value, student_before[name]))
        for name, value in runner.teacher.state_dict().items():
            self.assertTrue(torch.equal(value, teacher_before[name]))


if __name__ == "__main__":
    unittest.main()
