#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic one-pass random target-stream construction."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def make_fixed_random_stream_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """Build a reproducibly shuffled loader that visits each sample once.

    A fresh ``torch.Generator`` is seeded once when the loader is created.
    With a single pass over the loader, this produces one fixed random
    permutation without replacement. It removes dataset-order class blocks
    while preserving strict online semantics: no replay and no second pass.
    """
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
        pin_memory=bool(pin_memory),
        **kwargs,
    )
