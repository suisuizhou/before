#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA storage for only the light-weight test-time adaptation parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, Tuple

import torch
import torch.nn as nn


class AdaptationEMA:
    """Maintain EMA copies of band_scale, band_bias and warp_ctrl only."""

    def __init__(
        self,
        model: nn.Module,
        name_tokens: Tuple[str, ...] = ("band_scale", "band_bias", "warp_ctrl"),
    ) -> None:
        self.name_tokens = tuple(name_tokens)
        selected = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if any(token in name for token in self.name_tokens)
        }
        if not selected:
            raise RuntimeError(
                "No adaptation parameters found; expected band_scale, band_bias or warp_ctrl"
            )
        self.shadow: Dict[str, torch.Tensor] = selected

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.shadow.keys())

    @property
    def numel(self) -> int:
        return sum(tensor.numel() for tensor in self.shadow.values())

    def _selected_parameters(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        params = dict(model.named_parameters())
        missing = [name for name in self.shadow if name not in params]
        if missing:
            raise KeyError(f"EMA parameters missing from model: {missing}")
        return {name: params[name] for name in self.shadow}

    @torch.no_grad()
    def update(self, model: nn.Module, beta: float) -> None:
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("beta must be in [0,1)")
        params = self._selected_parameters(model)
        for name, shadow_value in self.shadow.items():
            shadow_value.mul_(float(beta)).add_(
                params[name].detach(), alpha=1.0 - float(beta)
            )

    @contextmanager
    def applied_to(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap current adaptation parameters for their EMA values."""
        params = self._selected_parameters(model)
        current = {
            name: parameter.detach().clone()
            for name, parameter in params.items()
        }
        with torch.no_grad():
            for name, parameter in params.items():
                parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in params.items():
                    parameter.copy_(current[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}
