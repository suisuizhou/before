#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testable source-training primitives for PU4D Protocol B."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from Lib.protocol_b_common import label_smoothed_cross_entropy
from Lib.wu_resnet18_common import gaussian_noise, impulse_noise, uniform_noise


def forward_parts(model: nn.Module, x: torch.Tensor):
    feature = model[0](x)
    bottleneck = model[1](feature)
    logits = model[2](bottleneck)
    return bottleneck, logits


def _macro_metrics(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> Dict[str, float]:
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    accuracy = (labels == predictions).float().mean().item() * 100.0
    f1s = []
    precisions = []
    recalls = []
    for class_id in range(int(num_classes)):
        tp = ((labels == class_id) & (predictions == class_id)).sum().item()
        fp = ((labels != class_id) & (predictions == class_id)).sum().item()
        fn = ((labels == class_id) & (predictions != class_id)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "macro_precision": 100.0 * sum(precisions) / len(precisions),
        "macro_recall": 100.0 * sum(recalls) / len(recalls),
        "macro_f1": 100.0 * sum(f1s) / len(f1s),
    }


def train_dtcc_source_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epsilon: float = 0.1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y, *_ in loader:
        if x.size(0) <= 1:
            continue
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, logits = forward_parts(model, x)
        loss = label_smoothed_cross_entropy(logits, y, epsilon, num_classes)
        loss.backward()
        optimizer.step()
        batch = y.numel()
        total_loss += float(loss.detach()) * batch
        total += batch
        correct += int((logits.detach().argmax(dim=1) == y).sum().item())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


def train_wu_source_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gaussian_std: float = 0.2,
    uniform_amplitude: float = 0.1,
    impulse_probability: float = 0.2,
    impulse_intensity: float = 0.2,
) -> Dict[str, float]:
    model.train()
    sums = {"loss": 0.0, "clean_ce": 0.0, "gaussian_ce": 0.0, "uniform_ce": 0.0, "impulse_ce": 0.0}
    total = 0
    correct = 0
    for x, y, *_ in loader:
        if x.size(0) <= 1:
            continue
        x = x.to(device)
        y = y.to(device)
        views = [
            x,
            gaussian_noise(x, gaussian_std),
            uniform_noise(x, uniform_amplitude),
            impulse_noise(x, impulse_probability, impulse_intensity),
        ]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        clean_logits = None
        for index, view in enumerate(views):
            _, logits = forward_parts(model, view)
            if index == 0:
                clean_logits = logits
            losses.append(F.cross_entropy(logits, y))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        batch = y.numel()
        total += batch
        sums["loss"] += float(loss.detach()) * batch
        for key, value in zip(("clean_ce", "gaussian_ce", "uniform_ce", "impulse_ce"), losses):
            sums[key] += float(value.detach()) * batch
        correct += int((clean_logits.detach().argmax(dim=1) == y).sum().item())
    result = {key: value / max(total, 1) for key, value in sums.items()}
    result["accuracy"] = 100.0 * correct / max(total, 1)
    return result


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader, device: torch.device, num_classes: int) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    labels = []
    predictions = []
    for x, y, *_ in loader:
        x = x.to(device)
        _, logits = forward_parts(model, x)
        labels.append(y.detach().cpu())
        predictions.append(logits.argmax(dim=1).detach().cpu())
    model.train(was_training)
    return _macro_metrics(torch.cat(labels), torch.cat(predictions), num_classes)
