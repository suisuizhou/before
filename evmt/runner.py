from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import time

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .bn import TargetBNController
from .ema import adaptable_state, ema_update_
from .evidence import verify_evidence
from .guards import CollapseGuard
from .losses import (
    mean_teacher_kl,
    neighborhood_kl,
    prototype_contrastive,
    weighted_sem,
)
from .memory import ClassBalancedMemory
from .regularization import (
    adapter_reg_loss,
    bn_affine_anchor,
    bn_anchor_loss,
    warp_reg_loss,
)
from .reliability import ReliabilityRouter
from .stages import AdaptationStage, StageController
from .views import make_teacher_views


@dataclass
class BatchMetrics:
    step: int
    samples: int
    correct: int
    loss: float
    sem: float
    pcl: float
    ncl: float
    mt: float
    reg_bn: float
    reg_adapter: float
    reg_warp: float
    certain_ratio: float
    confidence: float
    view_js: float
    plpd: float
    margin_drop: float
    reliability: float
    stage: str
    pred_histogram: list[int]
    pred_coverage: int
    recent_pred_coverage: int
    effective_classes: float
    max_class_share: float
    relative_effective_drop: float
    memory_entries: int
    memory_coverage: int
    memory_min_support: int
    bn_target_weight: float
    bn_seen_batches: int
    gradient_norms: dict[str, float]
    displacement_norms: dict[str, float]
    skip_reasons: list[str]
    updated: bool
    elapsed: float

    def as_dict(self):
        return asdict(self)


def forward_parts(model, x):
    feature = model[1](model[0](x))
    return feature, model[2](feature)


class _IdentityBNController:
    def __init__(self):
        self.seen_batches = 0
        self.target_weight = 0.0

    @contextmanager
    def prediction_stats(self, model=None):
        yield

    def observe_batch(self, model, x, forward_fn):
        self.seen_batches += 1


def _cfg(cfg, name, default):
    return getattr(cfg, name, default)


class EVMTOnlineRunner:
    def __init__(
        self,
        student,
        teacher,
        optimizer,
        cfg,
        num_classes,
        feature_dim=128,
        seed=2025,
        bn_controller=None,
        source_anchors=None,
    ):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.cfg = cfg
        self.num_classes = int(num_classes)
        self.router = ReliabilityRouter(
            num_classes,
            _cfg(cfg, "min_class_support", 2),
            _cfg(cfg, "min_conf", 0.5),
            _cfg(cfg, "max_js", 0.2),
            _cfg(cfg, "reliability_gamma", 5.0),
        )
        self.memory = ClassBalancedMemory(
            num_classes, _cfg(cfg, "memory_per_class", 32), feature_dim
        )
        self.stages = StageController(
            _cfg(cfg, "warmup_batches", 20),
            _cfg(cfg, "coverage_window", 5),
            _cfg(cfg, "min_pred_classes", min(24, num_classes)),
            _cfg(cfg, "min_memory_classes", min(16, num_classes)),
            _cfg(cfg, "min_entries_per_class", 2),
        )
        self.guard = CollapseGuard(
            num_classes,
            _cfg(cfg, "min_effective_classes", min(8, num_classes)),
            _cfg(cfg, "max_class_share", 0.5),
            _cfg(cfg, "max_effective_drop", 0.5),
        )
        self.generator = torch.Generator(
            device=next(student.parameters()).device
        ).manual_seed(seed)
        if bn_controller is None:
            try:
                bn_controller = TargetBNController(student)
            except ValueError:
                bn_controller = _IdentityBNController()
        self.bn = bn_controller
        self.source_anchors = (
            bn_affine_anchor(student) if source_anchors is None else source_anchors
        )
        self._initial_adaptable = {
            name: parameter.detach().clone()
            for name, parameter in adaptable_state(student).items()
        }
        self._bn_parameter_ids = {
            id(parameter)
            for module in student.modules()
            if isinstance(module, _BatchNorm) and module.affine
            for parameter in (module.weight, module.bias)
        }
        self._recent_histograms = deque(
            maxlen=max(1, int(_cfg(cfg, "coverage_window", 5)))
        )
        self.step_index = 0
        self.seen_samples = 0
        self.last_preupdate_logits = None
        self.student.eval()
        self.teacher.eval()

    def _teacher_route(self, x, stage):
        use_evidence = bool(_cfg(self.cfg, "use_evidence", True)) and (
            stage is not AdaptationStage.BN_WARMUP
        )
        teacher_x = x.detach().requires_grad_(use_evidence)
        with self.bn.prediction_stats(self.teacher):
            if use_evidence:
                teacher_feature, teacher_clean = forward_parts(self.teacher, teacher_x)
            else:
                with torch.no_grad():
                    teacher_feature, teacher_clean = forward_parts(
                        self.teacher, teacher_x
                    )
            views = make_teacher_views(
                x.detach(),
                _cfg(self.cfg, "style_strength", 0.03),
                _cfg(self.cfg, "view_warp_max", 0.5),
                self.generator,
            )
            teacher_probs = [teacher_clean.detach().softmax(1)]
            with torch.no_grad():
                for view in views[1:]:
                    teacher_probs.append(
                        forward_parts(self.teacher, view)[1].softmax(1)
                    )
            q = torch.stack(teacher_probs).mean(0)
            view_js = torch.stack(
                [
                    0.5
                    * (
                        F.kl_div(
                            q.clamp_min(1e-8).log(), p,
                            reduction="none",
                        ).sum(1)
                        + F.kl_div(
                            p.clamp_min(1e-8).log(), q,
                            reduction="none",
                        ).sum(1)
                    )
                    for p in teacher_probs
                ]
            ).mean(0)
            if use_evidence:
                evidence = verify_evidence(
                    lambda z: forward_parts(self.teacher, z)[1],
                    teacher_x,
                    teacher_clean,
                    q.argmax(1),
                    _cfg(self.cfg, "evidence_bands", 2),
                    _cfg(self.cfg, "evidence_width", 8),
                )
                margin_drop = evidence.margin_drop
                plpd = evidence.plpd
            else:
                margin_drop = None
                plpd = q.new_zeros(len(q))
        route = self.router.route(q, view_js, margin_drop)
        logged_margin = q.new_zeros(len(q)) if margin_drop is None else margin_drop
        return teacher_feature.detach(), q, view_js, plpd, logged_margin, route

    def _losses(self, feature, logits, q, route, stage):
        sem = weighted_sem(
            logits,
            route.reliability,
            _cfg(self.cfg, "eta", 0.1),
            _cfg(self.cfg, "alpha", 2.0),
        )
        zero = sem * 0.0
        pcl = zero
        ncl = zero
        if stage is AdaptationStage.FULL:
            prototypes, classes = self.memory.prototypes()
            if bool(_cfg(self.cfg, "use_pcl", True)):
                pcl = prototype_contrastive(
                    feature,
                    route.pseudo,
                    route.certain,
                    prototypes,
                    classes,
                    _cfg(self.cfg, "tau", 0.1),
                )
            if bool(_cfg(self.cfg, "use_ncl", True)):
                ncl = neighborhood_kl(
                    logits,
                    feature,
                    route.uncertain,
                    self.memory,
                    _cfg(self.cfg, "neighbor_k", 5),
                )
        mt = (
            mean_teacher_kl(logits, q, route.reliability)
            if bool(_cfg(self.cfg, "use_mt", True)) else zero
        )
        reg_bn = bn_anchor_loss(self.student, self.source_anchors)
        reg_adapter = adapter_reg_loss(self.student)
        reg_warp = warp_reg_loss(
            self.student, _cfg(self.cfg, "warp_smooth_weight", 5.0)
        )
        total = (
            sem
            + _cfg(self.cfg, "lambda_pcl", 0.2) * pcl
            + _cfg(self.cfg, "lambda_ncl", 0.2) * ncl
            + _cfg(self.cfg, "lambda_mt", 0.5) * mt
            + _cfg(self.cfg, "lambda_bn", 0.001) * reg_bn
            + _cfg(self.cfg, "lambda_reg", 0.001) * reg_adapter
            + _cfg(self.cfg, "lambda_warp", 0.0002) * reg_warp
        )
        return total, sem, pcl, ncl, mt, reg_bn, reg_adapter, reg_warp

    def _group_name(self, name, parameter):
        if id(parameter) in self._bn_parameter_ids:
            return "bn_affine"
        if name.endswith(("band_scale", "band_bias")):
            return "adapter"
        if name.endswith("warp_ctrl"):
            return "warp"
        return "other"

    def _gradient_norms(self):
        sums = {"bn_affine": 0.0, "adapter": 0.0, "warp": 0.0}
        for name, parameter in adaptable_state(self.student).items():
            if parameter.grad is None:
                continue
            group = self._group_name(name, parameter)
            if group in sums:
                sums[group] += float(parameter.grad.detach().square().sum())
        return {name: math.sqrt(value) for name, value in sums.items()}

    def _displacement_norms(self):
        sums = {"bn_affine": 0.0, "adapter": 0.0, "warp": 0.0}
        parameters = adaptable_state(self.student)
        for name, anchor in self._initial_adaptable.items():
            parameter = parameters[name]
            group = self._group_name(name, parameter)
            if group in sums:
                delta = parameter.detach() - anchor.to(parameter)
                sums[group] += float(delta.square().sum())
        return {name: math.sqrt(value) for name, value in sums.items()}

    def step(self, x, y_for_metrics=None):
        start = time.perf_counter()
        self.step_index += 1
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), self.bn.prediction_stats(self.student):
            _, scored_logits = forward_parts(self.student, x)
        pre_logits = scored_logits.detach()
        self.last_preupdate_logits = pre_logits
        correct = (
            0 if y_for_metrics is None
            else int((pre_logits.argmax(1) == y_for_metrics).sum())
        )
        histogram = torch.bincount(
            pre_logits.argmax(1), minlength=self.num_classes
        ).detach().cpu()
        self._recent_histograms.append(histogram)
        recent_coverage = int(
            (torch.stack(list(self._recent_histograms)).sum(0) > 0).sum()
        )

        self.bn.observe_batch(self.student, x, forward_parts)
        memory_before = self.memory.stats()
        stage = self.stages.update(
            self.step_index,
            recent_coverage,
            memory_before.covered_classes,
            memory_before.min_positive_size,
        )

        with self.bn.prediction_stats(self.student):
            student_feature, student_logits = forward_parts(self.student, x)
        teacher_feature, q, view_js, plpd, margin_drop, route = (
            self._teacher_route(x, stage)
        )
        losses = self._losses(
            student_feature, student_logits, q, route, stage
        )
        total, sem, pcl, ncl, mt, reg_bn, reg_adapter, reg_warp = losses
        decision = self.guard.check(pre_logits, total)
        skip_reasons = list(decision.reasons)
        updated = False
        gradient_norms = {"bn_affine": 0.0, "adapter": 0.0, "warp": 0.0}
        trainable = adaptable_state(self.student)

        if decision.allow_update and self.optimizer is not None and trainable:
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(trainable.values()), _cfg(self.cfg, "grad_clip", 5.0)
            )
            gradient_norms = self._gradient_norms()
            gradients_finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable.values()
            )
            if gradients_finite:
                self.optimizer.step()
                if stage is not AdaptationStage.BN_WARMUP:
                    self.memory.add(
                        teacher_feature,
                        q,
                        route.reliability,
                        route.certain,
                        self.step_index,
                    )
                ema_update_(
                    self.teacher,
                    self.student,
                    _cfg(self.cfg, "ema_beta", 0.99),
                )
                updated = True
            else:
                skip_reasons.append("non_finite_gradients")

        memory_after = self.memory.stats()
        self.seen_samples += len(x)
        return BatchMetrics(
            step=self.step_index,
            samples=len(x),
            correct=correct,
            loss=float(total.detach()),
            sem=float(sem.detach()),
            pcl=float(pcl.detach()),
            ncl=float(ncl.detach()),
            mt=float(mt.detach()),
            reg_bn=float(reg_bn.detach()),
            reg_adapter=float(reg_adapter.detach()),
            reg_warp=float(reg_warp.detach()),
            certain_ratio=float(route.certain.float().mean()),
            confidence=float(route.confidence.mean()),
            view_js=float(view_js.mean()),
            plpd=float(plpd.mean()),
            margin_drop=float(margin_drop.mean()),
            reliability=float(route.reliability.mean()),
            stage=stage.value,
            pred_histogram=[int(value) for value in histogram.tolist()],
            pred_coverage=int((histogram > 0).sum()),
            recent_pred_coverage=recent_coverage,
            effective_classes=decision.effective_classes,
            max_class_share=decision.max_class_share,
            relative_effective_drop=decision.relative_effective_drop,
            memory_entries=memory_after.total_entries,
            memory_coverage=memory_after.covered_classes,
            memory_min_support=memory_after.min_positive_size,
            bn_target_weight=float(self.bn.target_weight),
            bn_seen_batches=int(self.bn.seen_batches),
            gradient_norms=gradient_norms,
            displacement_norms=self._displacement_norms(),
            skip_reasons=skip_reasons,
            updated=updated,
            elapsed=time.perf_counter() - start,
        )
