import copy
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn

from evmt.runner import EVMTOnlineRunner


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_scale = nn.Parameter(torch.zeros(1))
        self.band_bias = nn.Parameter(torch.zeros(1))
        self.warp_ctrl = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(x) * (1 + self.band_scale) + self.band_bias


class SpyBNController:
    def __init__(self):
        self.seen_batches = 0
        self.target_weight = 0.0
        self.runner = None
        self.scored_before_observe = False

    @contextmanager
    def prediction_stats(self, model=None):
        yield

    def observe_batch(self, model, x, forward_fn):
        self.scored_before_observe = self.runner.last_preupdate_logits is not None
        self.seen_batches += 1
        self.target_weight = 0.5


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def config(**overrides):
    values = dict(
        min_class_support=1, min_conf=0.0, max_js=1.0,
        reliability_gamma=5.0, memory_per_class=4,
        style_strength=0.01, view_warp_max=0.1,
        evidence_bands=1, evidence_width=2, warmup_batches=1,
        coverage_window=2, min_pred_classes=1, min_memory_classes=1,
        min_entries_per_class=1, use_evidence=False, use_pcl=True,
        use_ncl=True, use_mt=True, eta=0.1, alpha=2.0, tau=0.1,
        neighbor_k=2, lambda_pcl=0.2, lambda_ncl=0.2,
        lambda_mt=0.5, lambda_bn=0.001, lambda_reg=0.001,
        lambda_warp=0.0002, warp_smooth_weight=5.0, grad_clip=5.0,
        ema_beta=0.9, min_effective_classes=1, max_class_share=1.0,
        max_effective_drop=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_runner(cfg):
    student = nn.Sequential(TinyBackbone(), nn.Identity(), nn.Linear(8, 3))
    teacher = copy.deepcopy(student)
    for model in (student, teacher):
        adaptable_ids = {
            id(model[0].band_scale), id(model[0].band_bias), id(model[0].warp_ctrl)
        }
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in adaptable_ids
    optimizer = CountingSGD(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    bn = SpyBNController()
    runner = EVMTOnlineRunner(
        student, teacher, optimizer, cfg, 3, 8, bn_controller=bn
    )
    bn.runner = runner
    return runner, optimizer, bn


class StrictRunnerTests(unittest.TestCase):
    def test_step_scores_before_observation_and_updates_at_most_once(self):
        runner, optimizer, bn = make_runner(config())
        x, y = torch.rand(4, 16), torch.tensor([0, 1, 2, 0])

        metrics = runner.step(x, y)

        self.assertTrue(bn.scored_before_observe)
        self.assertEqual(bn.seen_batches, 1)
        self.assertEqual(runner.seen_samples, 4)
        self.assertEqual(metrics.samples, 4)
        self.assertLessEqual(optimizer.step_calls, 1)
        self.assertEqual(sum(metrics.pred_histogram), 4)

    def test_guard_rejection_skips_optimizer_memory_and_ema(self):
        runner, optimizer, _ = make_runner(
            config(min_effective_classes=2, max_class_share=0.5)
        )
        with torch.no_grad():
            runner.student[2].weight.zero_()
            runner.student[2].bias.copy_(torch.tensor([20.0, -20.0, -20.0]))
            runner.teacher.load_state_dict(runner.student.state_dict())
        teacher_before = {
            name: value.detach().clone()
            for name, value in runner.teacher.named_parameters()
        }

        metrics = runner.step(torch.rand(8, 16))

        self.assertFalse(metrics.updated)
        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(runner.memory.stats().total_entries, 0)
        self.assertTrue(metrics.skip_reasons)
        for name, value in runner.teacher.named_parameters():
            self.assertTrue(torch.equal(value, teacher_before[name]))


if __name__ == "__main__":
    unittest.main()
