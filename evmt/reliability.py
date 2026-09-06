from dataclasses import dataclass
import math
import torch


@dataclass
class RouteResult:
    reliability: torch.Tensor
    certain: torch.Tensor
    uncertain: torch.Tensor
    confidence: torch.Tensor
    agreement: torch.Tensor
    evidence: torch.Tensor
    pseudo: torch.Tensor


class ReliabilityRouter:
    def __init__(self, num_classes, min_class_support=2, min_conf=0.5,
                 max_js=0.2, gamma=5.0, stats_momentum=0.9):
        self.num_classes = int(num_classes)
        self.min_class_support = int(min_class_support)
        self.min_conf = float(min_conf)
        self.max_js = float(max_js)
        self.gamma = float(gamma)
        self.stats_momentum = float(stats_momentum)
        self.running_median = None
        self.running_mad = None

    def _evidence(self, values):
        values = values.detach()
        median = values.median()
        mad = (values - median).abs().median()
        if mad <= 1e-8:
            center = median if self.running_median is None else self.running_median
            scale = values.new_tensor(1.0) if self.running_mad is None else self.running_mad
        else:
            center, scale = median, mad
        if self.running_median is None:
            self.running_median, self.running_mad = median, mad.clamp_min(1e-3)
        else:
            m = self.stats_momentum
            self.running_median = m*self.running_median + (1-m)*median
            self.running_mad = m*self.running_mad + (1-m)*mad.clamp_min(1e-3)
        return torch.sigmoid((values-center)/(1.4826*scale+1e-8))

    def route(self, q, view_js, margin_drop=None):
        q = q.detach().clamp_min(1e-8)
        pseudo = q.argmax(1)
        confidence = 1.0 + (q*q.log()).sum(1)/math.log(self.num_classes)
        agreement = torch.exp(-self.gamma*view_js.detach())
        evidence = (torch.ones_like(confidence) if margin_drop is None
                    else self._evidence(margin_drop))
        reliability = confidence*agreement*evidence
        safe = (confidence >= self.min_conf) & (view_js.detach() <= self.max_js)
        if margin_drop is not None:
            safe &= margin_drop.detach() > 0
        certain = torch.zeros_like(safe)
        global_t = reliability[safe].mean() if safe.any() else reliability.new_tensor(float("inf"))
        for cls in range(self.num_classes):
            members = safe & (pseudo == cls)
            threshold = reliability[members].mean() if int(members.sum()) >= self.min_class_support else global_t
            certain |= members & (reliability >= threshold)
        return RouteResult(reliability, certain, ~certain, confidence,
                           agreement, evidence, pseudo)
