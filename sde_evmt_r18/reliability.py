import math
from dataclasses import dataclass

import torch


@dataclass
class RouteResult:
    q: torch.Tensor
    pseudo: torch.Tensor
    confidence: torch.Tensor
    view_js: torch.Tensor
    agreement: torch.Tensor
    evidence: torch.Tensor
    reliability: torch.Tensor
    view_vote_fraction: torch.Tensor
    certain: torch.Tensor
    uncertain: torch.Tensor
    memory_eligible: torch.Tensor


class EvidenceReliabilityRouter:
    def __init__(
        self,
        confidence_min=0.3,
        gamma=5.0,
        min_view_agreement=0.75,
        min_class_samples=4,
        eps=1e-8,
    ):
        self.confidence_min = float(confidence_min)
        self.gamma = float(gamma)
        self.min_view_agreement = float(min_view_agreement)
        self.min_class_samples = int(min_class_samples)
        self.eps = float(eps)

    def _evidence_score(self, margin_drop):
        value = margin_drop.detach()
        median = value.median()
        mad = (value - median).abs().median()
        scale = (1.4826 * mad).clamp_min(self.eps)
        return torch.sigmoid((value - median) / scale)

    def route(self, q_views, margin_drop):
        if q_views.ndim != 3:
            raise ValueError(f"expected [V,B,C] probabilities, got {tuple(q_views.shape)}")
        if q_views.shape[0] < 1 or q_views.shape[-1] < 2:
            raise ValueError("at least one view and two classes are required")
        probability = q_views.detach().clamp_min(self.eps)
        probability = probability / probability.sum(dim=-1, keepdim=True)
        q = probability.mean(dim=0)
        q = q / q.sum(dim=-1, keepdim=True)
        pseudo = q.argmax(dim=1)
        entropy = -(q * q.clamp_min(self.eps).log()).sum(dim=1)
        confidence = 1.0 - entropy / math.log(q.shape[1])

        mean = q.unsqueeze(0).expand_as(probability)
        midpoint = 0.5 * (probability + mean)
        js = 0.5 * (
            (probability * (probability.log() - midpoint.log())).sum(dim=-1)
            + (mean * (mean.log() - midpoint.log())).sum(dim=-1)
        )
        view_js = js.mean(dim=0)
        agreement = torch.exp(-self.gamma * view_js)
        evidence = (
            torch.ones_like(confidence)
            if margin_drop is None
            else self._evidence_score(margin_drop)
        )
        reliability = confidence * agreement * evidence
        view_vote_fraction = (probability.argmax(dim=-1) == pseudo[None]).float().mean(dim=0)

        safe = (
            (confidence >= self.confidence_min)
            & (view_vote_fraction >= self.min_view_agreement)
        )
        if margin_drop is not None:
            safe &= margin_drop.detach() > 0
        certain = torch.zeros_like(safe)
        memory_eligible = torch.zeros_like(safe)
        for label in pseudo.unique():
            members = pseudo == label
            valid = members & safe
            if bool(valid.any()):
                threshold = torch.quantile(reliability[members], 0.5)
                selected = valid & (reliability >= threshold)
                certain |= selected
                if int(members.sum()) >= self.min_class_samples:
                    memory_eligible |= selected
        return RouteResult(
            q=q,
            pseudo=pseudo,
            confidence=confidence,
            view_js=view_js,
            agreement=agreement,
            evidence=evidence,
            reliability=reliability,
            view_vote_fraction=view_vote_fraction,
            certain=certain,
            uncertain=~certain,
            memory_eligible=memory_eligible,
        )
