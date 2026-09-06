from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .augment import cfg_get


@dataclass
class SourceLosses:
    total: torch.Tensor
    clean_cls: torch.Tensor
    style_cls: torch.Tensor
    warp_cls: torch.Tensor
    noise_cls: torch.Tensor
    pred_cons: torch.Tensor
    feat_cons: torch.Tensor

    def detached(self):
        return {
            name: float(value.detach())
            for name, value in self.__dict__.items()
        }


def symmetric_kl(clean_logits, view_logits):
    clean_log = F.log_softmax(clean_logits, dim=-1)
    view_log = F.log_softmax(view_logits, dim=-1)
    clean_prob = clean_log.exp()
    view_prob = view_log.exp()
    forward = F.kl_div(view_log, clean_prob, reduction="batchmean")
    reverse = F.kl_div(clean_log, view_prob, reduction="batchmean")
    return 0.5 * (forward + reverse)


def feature_consistency(clean_feature, view_feature):
    clean_feature = F.normalize(clean_feature, dim=-1)
    view_feature = F.normalize(view_feature, dim=-1)
    return (clean_feature - view_feature).square().sum(dim=-1).mean()


def _mixup(views, labels, cfg, generator):
    probability = float(cfg_get(cfg, "source.mixup_prob", 0.5))
    alpha = float(cfg_get(cfg, "source.mixup_alpha", 0.2))
    trigger = torch.rand((), device=labels.device, generator=generator)
    if alpha <= 0.0 or float(trigger) >= probability:
        return views, labels, labels, 1.0
    concentration = torch.full((1,), alpha, device=labels.device)
    first = torch._standard_gamma(concentration, generator=generator)
    second = torch._standard_gamma(concentration, generator=generator)
    lam = float((first / (first + second).clamp_min(1e-12)).item())
    lam = max(lam, 1.0 - lam)
    permutation = torch.randperm(len(labels), device=labels.device, generator=generator)
    mixed = {
        name: lam * value + (1.0 - lam) * value[permutation]
        for name, value in views.items()
    }
    return mixed, labels, labels[permutation], lam


def _classification(logits, first_labels, second_labels, lam, smoothing):
    first = F.cross_entropy(logits, first_labels, label_smoothing=smoothing)
    if lam >= 1.0:
        return first
    second = F.cross_entropy(logits, second_labels, label_smoothing=smoothing)
    return lam * first + (1.0 - lam) * second


def source_objective(model, x, y, augmenter, cfg):
    views = augmenter.views(x)
    views, labels_a, labels_b, lam = _mixup(
        views, y, cfg, augmenter.generator
    )
    smoothing = float(cfg_get(cfg, "source.label_smoothing", 0.1))
    clean_feature, clean_logits = model.forward_parts(views["clean"])
    clean_cls = _classification(clean_logits, labels_a, labels_b, lam, smoothing)
    zero = clean_cls * 0.0
    if str(cfg_get(cfg, "variant", "R1")).upper() == "R0":
        return SourceLosses(clean_cls, clean_cls, zero, zero, zero, zero, zero)

    outputs = {}
    for name in ("style", "warp", "noise"):
        outputs[name] = model.forward_parts(views[name])
    style_cls = _classification(outputs["style"][1], labels_a, labels_b, lam, smoothing)
    warp_cls = _classification(outputs["warp"][1], labels_a, labels_b, lam, smoothing)
    noise_cls = _classification(outputs["noise"][1], labels_a, labels_b, lam, smoothing)
    pred_cons = torch.stack(
        [symmetric_kl(clean_logits, outputs[name][1]) for name in outputs]
    ).mean()
    feat_cons = torch.stack(
        [feature_consistency(clean_feature, outputs[name][0]) for name in outputs]
    ).mean()
    total = (
        clean_cls
        + float(cfg_get(cfg, "loss.lambda_style", 0.5)) * style_cls
        + float(cfg_get(cfg, "loss.lambda_warp", 0.5)) * warp_cls
        + float(cfg_get(cfg, "loss.lambda_noise", 0.5)) * noise_cls
        + float(cfg_get(cfg, "loss.lambda_pred_cons", 0.2)) * pred_cons
        + float(cfg_get(cfg, "loss.lambda_feat_cons", 0.05)) * feat_cons
    )
    return SourceLosses(
        total, clean_cls, style_cls, warp_cls, noise_cls, pred_cons, feat_cons
    )
