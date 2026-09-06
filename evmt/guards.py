from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GuardDecision:
    allow_update: bool
    reasons: tuple[str, ...]
    effective_classes: float
    max_class_share: float
    relative_effective_drop: float


class CollapseGuard:
    def __init__(
        self,
        num_classes,
        min_effective_classes=8,
        max_class_share=0.5,
        max_relative_drop=0.5,
        reference_momentum=0.9,
    ):
        self.num_classes = int(num_classes)
        self.min_effective_classes = float(min_effective_classes)
        self.max_class_share = float(max_class_share)
        self.max_relative_drop = float(max_relative_drop)
        self.reference_momentum = float(reference_momentum)
        self.stable_effective_classes = None

    def check(self, logits, loss=None):
        reasons = []
        if not bool(torch.isfinite(logits).all()):
            return GuardDecision(False, ("non_finite_logits",), math.nan, math.nan, math.nan)
        if loss is not None and not bool(torch.isfinite(loss).all()):
            reasons.append("non_finite_loss")

        marginal = logits.detach().softmax(1).mean(0).clamp_min(1e-12)
        entropy = -(marginal * marginal.log()).sum()
        effective = float(entropy.exp())
        largest = float(marginal.max())
        reference = self.stable_effective_classes
        relative_drop = (
            0.0 if reference is None
            else max(0.0, (reference - effective) / max(reference, 1e-12))
        )

        if effective < self.min_effective_classes:
            reasons.append("effective_classes")
        if largest > self.max_class_share:
            reasons.append("max_class_share")
        if relative_drop > self.max_relative_drop:
            reasons.append("relative_effective_drop")

        allow = not reasons
        if allow:
            if reference is None:
                self.stable_effective_classes = effective
            else:
                momentum = self.reference_momentum
                self.stable_effective_classes = (
                    momentum * reference + (1.0 - momentum) * effective
                )
        return GuardDecision(allow, tuple(reasons), effective, largest, relative_drop)
