#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small target-side helpers with testable online-order semantics."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from Lib.wu_resnet18_common import RecentBatchMemory, feature_aggregation_loss


@torch.no_grad()
def preupdate_predict(
    model: nn.Module,
    x: torch.Tensor,
    logits_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return hard predictions before any optimizer update for the batch."""
    was_training = model.training
    model.eval()
    logits = logits_fn(model, x)
    model.train(was_training)
    return logits.argmax(dim=1)


def wu_historical_feature_loss(
    current_features: torch.Tensor,
    pseudo: torch.Tensor,
    memory: RecentBatchMemory,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or len(memory) == 0:
        return current_features.new_tensor(0.0)
    return feature_aggregation_loss(current_features, pseudo, memory)
