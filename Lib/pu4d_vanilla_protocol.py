#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn

ADAPTATION_TOKENS = ("band_scale", "band_bias", "warp_ctrl")


def vanilla_checkpoint_dir(root: Path | str, source: int, seed: int) -> Path:
    return Path(root) / "PU4D" / f"source_{int(source)}" / f"seed_{int(seed)}"


def choose_dummy_target(source: int, domains: Iterable[int]) -> int:
    source = int(source)
    for domain in domains:
        domain = int(domain)
        if domain != source:
            return domain
    raise ValueError("At least two domains are required")


@torch.no_grad()
def reset_and_freeze_adaptation_carrier(model: nn.Module) -> List[str]:
    """Keep target-only adapter/warp parameters at identity during source training."""
    frozen: List[str] = []
    for name, parameter in model.named_parameters():
        if any(token in name for token in ADAPTATION_TOKENS):
            parameter.zero_()
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def resolve_vanilla_checkpoint(
    root: Path | str,
    source: int,
    seed: int,
    model_name: str,
) -> Path:
    directory = vanilla_checkpoint_dir(root, source, seed)
    best = directory / f"best_source_{model_name}"
    final = directory / model_name
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(f"Vanilla checkpoint not found: {best} or {final}")
