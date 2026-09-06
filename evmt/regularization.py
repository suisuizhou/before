import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def _zero(model):
    return next(model.parameters()).new_zeros(())


def bn_affine_anchor(model):
    anchor = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, _BatchNorm) or not module.affine:
            continue
        prefix = f"{module_name}." if module_name else ""
        anchor[prefix + "weight"] = module.weight.detach().clone()
        anchor[prefix + "bias"] = module.bias.detach().clone()
    return anchor


def bn_anchor_loss(model, anchor):
    parameters = dict(model.named_parameters())
    terms = [
        (parameters[name] - value.to(parameters[name])).square().mean()
        for name, value in anchor.items()
    ]
    return torch.stack(terms).mean() if terms else _zero(model)


def adapter_reg_loss(model):
    terms = []
    for module in model.modules():
        scale_param = getattr(module, "band_scale", None)
        bias_param = getattr(module, "band_bias", None)
        if scale_param is None or bias_param is None:
            continue
        delta = float(getattr(module, "adapter_delta", 0.1))
        scale = 1.0 + delta * torch.tanh(scale_param)
        bias = delta * bias_param
        terms.append((scale - 1.0).square().mean() + bias.square().mean())
    return torch.stack(terms).mean() if terms else _zero(model)


def warp_reg_loss(model, smooth_weight=5.0):
    terms = []
    for module in model.modules():
        control = getattr(module, "warp_ctrl", None)
        if control is None:
            continue
        length = int(getattr(module, "input_len", control.shape[-1]))
        max_warp = float(getattr(module, "max_warp", 2.0))
        control_line = control if control.ndim == 3 else control.reshape(1, 1, -1)
        delta = F.interpolate(
            torch.tanh(control_line), size=length, mode="linear", align_corners=True
        ) * max_warp
        denom = max(max_warp * max_warp, 1e-6)
        l2 = delta.square().mean() / denom
        smooth = (
            (delta[..., 1:] - delta[..., :-1]).square().mean() / denom
            if length > 1 else delta.new_zeros(())
        )
        terms.append(l2 + float(smooth_weight) * smooth)
    return torch.stack(terms).mean() if terms else _zero(model)
