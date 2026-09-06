"""Full-model exponential moving average for the documented 0711 teacher."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
import torch.nn as nn


class FullModelEMA:
    """Keep an EMA shadow of every parameter and buffer in a model.

    Frozen parameters are included deliberately: this follows the mathematical
    formulation in the 0711 system document rather than the strict-online
    lightweight optimization shortcut. Integer buffers are copied exactly;
    floating tensors use the standard EMA update.
    """

    def __init__(self, model: nn.Module):
        self.shadow: Dict[str, torch.Tensor] = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        if not self.shadow:
            raise RuntimeError("cannot create EMA for an empty model")

    @property
    def names(self):
        return tuple(self.shadow.keys())

    @property
    def numel(self) -> int:
        return sum(value.numel() for value in self.shadow.values())

    @torch.no_grad()
    def update(self, model: nn.Module, beta: float) -> None:
        beta = float(beta)
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0,1)")
        state = model.state_dict()
        missing = [name for name in self.shadow if name not in state]
        if missing:
            raise KeyError(f"EMA state missing from model: {missing}")
        for name, shadow in self.shadow.items():
            current = state[name].detach()
            if shadow.is_floating_point() or shadow.is_complex():
                shadow.mul_(beta).add_(current, alpha=1.0 - beta)
            else:
                shadow.copy_(current)

    @contextmanager
    def applied_to(self, model: nn.Module) -> Iterator[None]:
        state = model.state_dict()
        current = {name: value.detach().clone() for name, value in state.items()}
        with torch.no_grad():
            for name, value in state.items():
                value.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in state.items():
                    value.copy_(current[name])

    def state_dict(self):
        return {name: value.detach().clone() for name, value in self.shadow.items()}
