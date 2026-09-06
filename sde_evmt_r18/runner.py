import copy
import time
from dataclasses import asdict, dataclass

import torch

from .augment import cfg_get
from .evidence import verify_margin_evidence
from .memory import ClassBalancedMemory
from .reliability import EvidenceReliabilityRouter
from .scheduler import OnlineScheduler, TargetStage
from .target_losses import EMATeacherPrior, target_objective
from .target_views import make_teacher_views


_INPUT_PARAMETERS = {
    "frequency_warp.warp_ctrl",
    "frequency_warp.warp_gate_logit",
    "spectral_adapter.band_scale",
    "spectral_adapter.band_bias",
    "spectral_adapter.adapter_gate_logit",
}


def _variant_number(variant):
    variant = str(variant).upper()
    if variant not in {"R2", "R3", "R4", "R5", "R6"}:
        raise ValueError(f"target variant must be R2-R6, got {variant}")
    return int(variant[1:])


def configure_adaptation(model, variant, stage):
    number = _variant_number(variant)
    stage = TargetStage(stage)
    names = {name for name, _ in model.named_parameters()}
    missing = _INPUT_PARAMETERS - names
    if missing:
        raise ValueError(f"model is missing adaptable parameters: {sorted(missing)}")
    allowed = set(_INPUT_PARAMETERS)
    if number >= 6 and stage is TargetStage.FULL:
        allowed.update(name for name in names if name.startswith("feature_adapter."))
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in allowed
        if parameter.requires_grad:
            selected.append(name)
    return selected


def build_target_optimizer(model, cfg):
    input_parameters = []
    feature_parameters = []
    gn_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in _INPUT_PARAMETERS:
            input_parameters.append(parameter)
        elif name.startswith("feature_adapter."):
            feature_parameters.append(parameter)
        elif name.startswith("layer4.") and name.endswith(("weight", "bias")):
            gn_parameters.append(parameter)
    groups = []
    if input_parameters:
        groups.append({
            "params": input_parameters,
            "lr": float(cfg_get(cfg, "optimizer.lr_warp_adapter", 5e-4)),
            "name": "warp_spectral",
        })
    if feature_parameters:
        groups.append({
            "params": feature_parameters,
            "lr": float(cfg_get(cfg, "optimizer.lr_feature_adapter", 1e-4)),
            "name": "feature_adapter",
        })
    if gn_parameters:
        groups.append({
            "params": gn_parameters,
            "lr": float(cfg_get(cfg, "optimizer.lr_gn_affine", 5e-5)),
            "name": "layer4_gn",
        })
    if not groups:
        raise ValueError("no target-adaptable parameters were selected")
    return torch.optim.AdamW(
        groups,
        weight_decay=float(cfg_get(cfg, "optimizer.weight_decay", 1e-5)),
    )


@dataclass
class BatchMetrics:
    step: int
    samples: int
    correct: int
    loss: float
    mt: float
    sem: float
    diversity: float
    pcl: float
    ncl: float
    warp_reg: float
    spectral_reg: float
    feature_anchor: float
    confidence: float
    view_js: float
    plpd: float
    margin_drop: float
    reliability: float
    certain_ratio: float
    stage: str
    pred_histogram: list
    pred_coverage: int
    memory_entries: int
    memory_coverage: int
    memory_min_support: int
    gate_values: dict
    gradient_norm: float
    skip_reasons: list
    updated: bool
    elapsed: float

    def as_dict(self):
        return asdict(self)


class SDEEVMTOnlineRunner:
    def __init__(
        self,
        student,
        teacher=None,
        optimizer=None,
        cfg=None,
        variant="R6",
        num_classes=None,
        feature_dim=None,
        seed=2025,
    ):
        if cfg is None and isinstance(teacher, dict):
            cfg, teacher = teacher, None
        self.cfg = {} if cfg is None else cfg
        self.variant = str(variant).upper()
        self.variant_number = _variant_number(self.variant)
        self.student = student
        self.teacher = copy.deepcopy(student) if teacher is None else teacher
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        self.student.eval()
        self.num_classes = int(
            num_classes if num_classes is not None
            else self.student.classifier.out_features
        )
        self.feature_dim = int(
            feature_dim if feature_dim is not None
            else getattr(self.student, "output_num", self.student.classifier.in_features)
        )
        configure_adaptation(self.student, self.variant, TargetStage.FULL)
        self.optimizer = build_target_optimizer(self.student, self.cfg) if optimizer is None else optimizer
        self.stage = TargetStage.WARMUP
        configure_adaptation(self.student, self.variant, self.stage)
        self.router = EvidenceReliabilityRouter(
            confidence_min=float(cfg_get(self.cfg, "reliability.confidence_min", 0.3)),
            gamma=float(cfg_get(self.cfg, "reliability.gamma", 5.0)),
            min_view_agreement=float(cfg_get(self.cfg, "reliability.min_view_agreement", 0.75)),
            min_class_samples=int(cfg_get(self.cfg, "reliability.min_class_samples", 4)),
        )
        self.memory = ClassBalancedMemory(
            self.num_classes,
            int(cfg_get(self.cfg, "memory.capacity_per_class", 64)),
            self.feature_dim,
        )
        self.prior = EMATeacherPrior(
            self.num_classes,
            float(cfg_get(self.cfg, "prior.momentum", 0.9)),
        )
        self.scheduler = OnlineScheduler(int(cfg_get(self.cfg, "warmup_batches", 5)))
        device = next(self.student.parameters()).device
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.feature_anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.student.feature_adapter.named_parameters()
        }
        self.candidate_counts = torch.zeros(self.num_classes, dtype=torch.long)
        self.reliability_history = []
        self.step_index = 0
        self.seen_samples = 0
        self.last_preupdate_logits = None

    def _candidate_stats(self):
        positive = self.candidate_counts > 0
        return {
            "covered_classes": int(positive.sum()),
            "total_entries": int(self.candidate_counts.sum()),
        }

    def _objective_cfg(self, stage):
        defaults = {
            "lambda_mt": 1.0,
            "lambda_sem": 0.2,
            "lambda_pcl": 0.1,
            "lambda_ncl": 0.2,
            "lambda_div": 1.0,
            "lambda_warp_reg": 0.0002,
            "lambda_adapter_reg": 0.0001,
            "lambda_feature_anchor": 0.001,
            "tsallis_alpha": 2.0,
            "uncertain_floor": 0.1,
            "pcl_temperature": 0.1,
        }
        loss = {
            name: cfg_get(self.cfg, f"loss.{name}", default)
            for name, default in defaults.items()
        }
        return {
            "enable_mt": self.variant_number >= 3,
            "enable_pcl": self.variant_number >= 5 and stage is TargetStage.FULL,
            "enable_ncl": self.variant_number >= 6 and stage is TargetStage.FULL,
            "loss": loss,
            "memory": {
                "neighbor_num": int(cfg_get(self.cfg, "memory.neighbor_num", 5)),
                "neighbor_similarity_threshold": float(
                    cfg_get(self.cfg, "memory.neighbor_similarity_threshold", 0.3)
                ),
            },
        }

    def _teacher_route(self, x, stage):
        use_views = self.variant_number >= 3
        use_evidence = self.variant_number >= 4 and stage is not TargetStage.WARMUP
        views = make_teacher_views(x.detach(), self.cfg, self.generator) if use_views else [x]
        with torch.no_grad():
            clean_spectrum = self.teacher.prepare_spectrum(views[0])
            teacher_feature, clean_logits = self.teacher.forward_from_spectrum(clean_spectrum)
            probabilities = [clean_logits.softmax(dim=1)]
            for view in views[1:]:
                probabilities.append(self.teacher(view).softmax(dim=1))
        stacked = torch.stack(probabilities)
        margin_drop = None
        plpd = clean_logits.new_zeros(len(x))
        if use_evidence:
            pseudo = stacked.mean(dim=0).argmax(dim=1)
            evidence = verify_margin_evidence(
                lambda spectrum: self.teacher.forward_from_spectrum(spectrum)[1],
                clean_spectrum.detach(),
                clean_logits.detach(),
                pseudo,
                self.cfg,
            )
            margin_drop = evidence.margin_drop
            plpd = evidence.plpd
        route = self.router.route(stacked, margin_drop)
        logged_margin = route.q.new_zeros(len(x)) if margin_drop is None else margin_drop
        return teacher_feature.detach(), route, plpd, logged_margin

    def _ema_update(self):
        beta = float(cfg_get(self.cfg, "teacher.ema_beta", 0.99))
        student_parameters = dict(self.student.named_parameters())
        with torch.no_grad():
            for name, teacher_parameter in self.teacher.named_parameters():
                student_parameter = student_parameters[name]
                if student_parameter.requires_grad:
                    teacher_parameter.mul_(beta).add_(student_parameter, alpha=1.0 - beta)

    def _record_successful_state(self, teacher_feature, route, stage):
        self.prior.update(route.q)
        self.reliability_history.append(float(route.reliability.mean()))
        labels = route.pseudo[route.memory_eligible].detach().cpu()
        if len(labels):
            self.candidate_counts += torch.bincount(labels, minlength=self.num_classes)
        if self.variant_number >= 5 and stage is TargetStage.FULL:
            self.memory.add(
                teacher_feature,
                route.q,
                route.reliability,
                route.memory_eligible,
                step=self.step_index,
            )
        self._ema_update()

    def step(self, x, y_for_metrics=None):
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scored_logits = self.student(x)
        preupdate = scored_logits.detach().clone()
        self.last_preupdate_logits = preupdate
        prediction = preupdate.argmax(dim=1)
        correct = 0
        if y_for_metrics is not None:
            labels = y_for_metrics.detach().to(prediction.device)
            correct = int((prediction == labels).sum())
        histogram = torch.bincount(prediction, minlength=self.num_classes).detach().cpu()

        previous_stage = self.stage
        generator_state = self.generator.get_state()
        self.stage = self.scheduler.update(
            self.step_index,
            self._candidate_stats(),
            self.memory.stats(),
            self.reliability_history,
        )
        configure_adaptation(self.student, self.variant, self.stage)
        student_feature, student_logits = self.student.forward_parts(x)
        teacher_feature, route, plpd, margin_drop = self._teacher_route(x, self.stage)
        losses = target_objective(
            model=self.student,
            student_logits=student_logits,
            student_features=student_feature,
            teacher_q=route.q,
            reliability=route.reliability,
            certain=route.certain,
            uncertain=route.uncertain,
            memory=self.memory,
            prior=self.prior,
            cfg=self._objective_cfg(self.stage),
            feature_anchor=self.feature_anchor,
        )
        skip_reasons = []
        updated = False
        gradient_norm = 0.0
        if not bool(torch.isfinite(losses.total)):
            skip_reasons.append("non_finite_loss")
        else:
            losses.total.backward()
            trainable = [parameter for parameter in self.student.parameters() if parameter.requires_grad]
            finite_gradients = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            )
            if not finite_gradients:
                skip_reasons.append("non_finite_gradients")
            else:
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    trainable,
                    float(cfg_get(self.cfg, "optimizer.grad_clip", 1.0)),
                ))
                self.optimizer.step()
                self._record_successful_state(teacher_feature, route, self.stage)
                updated = True
        if not updated:
            self.generator.set_state(generator_state)
            self.scheduler.stage = previous_stage
            self.stage = previous_stage
            configure_adaptation(self.student, self.variant, self.stage)

        memory_stats = self.memory.stats()
        self.step_index += 1
        self.seen_samples += len(x)
        gate_values = self.student.gate_summary() if hasattr(self.student, "gate_summary") else {}
        return BatchMetrics(
            step=self.step_index,
            samples=len(x),
            correct=correct,
            loss=float(losses.total.detach()),
            mt=float(losses.mt.detach()),
            sem=float(losses.sem.detach()),
            diversity=float(losses.diversity.detach()),
            pcl=float(losses.pcl.detach()),
            ncl=float(losses.ncl.detach()),
            warp_reg=float(losses.warp_reg.detach()),
            spectral_reg=float(losses.spectral_reg.detach()),
            feature_anchor=float(losses.feature_anchor.detach()),
            confidence=float(route.confidence.mean()),
            view_js=float(route.view_js.mean()),
            plpd=float(plpd.mean()),
            margin_drop=float(margin_drop.mean()),
            reliability=float(route.reliability.mean()),
            certain_ratio=float(route.certain.float().mean()),
            stage=self.stage.name,
            pred_histogram=[int(value) for value in histogram.tolist()],
            pred_coverage=int((histogram > 0).sum()),
            memory_entries=memory_stats.total_entries,
            memory_coverage=memory_stats.covered_classes,
            memory_min_support=memory_stats.min_positive_size,
            gate_values=gate_values,
            gradient_norm=gradient_norm,
            skip_reasons=skip_reasons,
            updated=updated,
            elapsed=time.perf_counter() - started,
        )
