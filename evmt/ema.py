import torch
from torch.nn.modules.batchnorm import _BatchNorm

ADAPTABLE_SUFFIXES = ("band_scale", "band_bias", "warp_ctrl")


def _bn_affine_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, _BatchNorm) or not module.affine:
            continue
        prefix = f"{module_name}." if module_name else ""
        names.add(prefix + "weight")
        names.add(prefix + "bias")
    return names


def adaptable_state(model, include_bn_affine=True):
    bn_names = _bn_affine_names(model) if include_bn_affine else set()
    return {
        name: param
        for name, param in model.named_parameters()
        if name.endswith(ADAPTABLE_SUFFIXES) or name in bn_names
    }


@torch.no_grad()
def ema_update_(teacher, student, beta, include_bn_affine=True):
    teacher_state = adaptable_state(teacher, include_bn_affine)
    student_state = adaptable_state(student, include_bn_affine)
    if teacher_state.keys() != student_state.keys():
        raise ValueError("teacher/student adaptable states differ")
    for name, target in teacher_state.items():
        target.mul_(beta).add_(student_state[name], alpha=1.0 - beta)
