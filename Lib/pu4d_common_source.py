#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared PU4D experiment plumbing for common-source TTA comparisons."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import omegaconf
import torch
import torch.nn as nn
from omegaconf import open_dict
from torch.utils.data import DataLoader

import Dataset
from Lib.fixed_random_stream import make_fixed_random_stream_loader
from Lib.model import get_model
from Lib.train_utils import seed_torch


def _plain(value):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def parse_only_task(value):
    if value is None:
        return None
    value = ast.literal_eval(value) if isinstance(value, str) else _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"only_task must be [source,target], got {value}")
    return int(value[0]), int(value[1])


def parse_seed_runs(value) -> List[int]:
    if value is None:
        return [2025]
    value = ast.literal_eval(value) if isinstance(value, str) else _plain(value)
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise ValueError(f"Unsupported seed_runs: {value}")


def build_model_name(cfg) -> str:
    suffix = {
        "linear": "_Linear.pt",
        "wn": "_WN.pt",
    }.get(str(cfg.Model.model_type), "_Pro.pt")
    return (
        f"{cfg.Model.model_name}{cfg.seed_run}"
        f"{cfg.Dataset.input_kind}{suffix}"
    )


def _get(section, name: str, default):
    return getattr(section, name, default)


def load_checkpoint_compatible(model: nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state)}")
    current = model.state_dict()
    compatible = {}
    mismatched = []
    unexpected = []
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if key not in current:
            unexpected.append(key)
        elif current[key].shape != value.shape:
            mismatched.append((key, tuple(value.shape), tuple(current[key].shape)))
        else:
            compatible[key] = value
    result = model.load_state_dict(compatible, strict=False)
    print(f"[COMMON SOURCE] checkpoint={path}")
    print(f"  loaded={len(compatible)} missing={len(result.missing_keys)}")
    if mismatched:
        print("  mismatched:", mismatched[:10])
    if unexpected:
        print("  unexpected:", unexpected[:10])


def forward_parts(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    feature = model[0](x)
    bottleneck = model[1](feature)
    logits = model[2](bottleneck)
    return bottleneck, logits


def classification_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    accuracy = (labels == predictions).float().mean().item() * 100.0
    precisions = []
    recalls = []
    f1s = []
    for class_id in range(int(num_classes)):
        true_pos = ((labels == class_id) & (predictions == class_id)).sum().item()
        false_pos = ((labels != class_id) & (predictions == class_id)).sum().item()
        false_neg = ((labels == class_id) & (predictions != class_id)).sum().item()
        precision = true_pos / max(true_pos + false_pos, 1)
        recall = true_pos / max(true_pos + false_neg, 1)
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


class PredictionAccumulator:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self._labels: List[torch.Tensor] = []
        self._predictions: List[torch.Tensor] = []

    def update(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        self._labels.append(labels.detach().cpu())
        self._predictions.append(predictions.detach().cpu())

    def metrics(self) -> Dict[str, float]:
        if not self._labels:
            return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
        return classification_metrics(
            torch.cat(self._labels), torch.cat(self._predictions), self.num_classes
        )


class CommonSourcePU4DBase:
    def __init__(self, cfg: omegaconf.DictConfig, run_obj=None):
        seed_torch(cfg.seed_run)
        with open_dict(cfg):
            cfg.model_name = build_model_name(cfg)
            cfg.save_model_path = "./TTA_Model"
        self.cfg = cfg
        self.run = run_obj
        self.device: torch.device | None = None
        self.num_classes = 0
        self.datasets = {}
        self.eval_loader = None
        self.stream_loader = None
        self.model = None
        self.stream_seed = 0

    def setup(self) -> None:
        cfg = self.cfg
        seed_torch(cfg.seed_run)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("using %s GPUs", torch.cuda.device_count())
        else:
            self.device = torch.device("cpu")
            logging.warning("GPU unavailable; using CPU")

        dataset_cls = getattr(Dataset, cfg.Dataset.data_name)
        self.num_classes = int(dataset_cls.num_classes)
        source_data, target_data = dataset_cls(**cfg.Dataset).data_generator()
        self.datasets = {"source_data": source_data, "target_data": target_data}
        self.eval_loader = DataLoader(
            target_data,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )
        self.stream_seed = int(getattr(cfg, "stream_seed", cfg.seed_run))
        self.stream_loader = make_fixed_random_stream_loader(
            dataset=target_data,
            batch_size=int(cfg.batch_size),
            seed=self.stream_seed,
            num_workers=int(cfg.num_workers),
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )
        self.model = get_model(
            num_classes=self.num_classes,
            cfg=cfg,
            **cfg.Model,
        ).to(self.device)

    def checkpoint_path(self) -> Path:
        source = int(self.cfg.Dataset.TL_Task[0])
        directory = (
            Path(self.cfg.save_model_path)
            / f"{self.cfg.Dataset.data_name}{self.cfg.Opt.lr_src}"
            / f"source_{source}"
            / f"seed_{int(self.cfg.seed_run)}"
        )
        best = directory / f"best_source_{self.cfg.model_name}"
        final = directory / self.cfg.model_name
        if best.exists():
            return best
        if final.exists():
            return final
        raise FileNotFoundError(f"Common source checkpoint not found: {best} or {final}")

    def initialize_common_source(self) -> None:
        load_checkpoint_compatible(self.model, self.checkpoint_path(), self.device)

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        was_training = model.training
        model.eval()
        labels = []
        predictions = []
        for x, y, _ in self.eval_loader:
            x = x.to(self.device)
            _, logits = forward_parts(model, x)
            labels.append(y.cpu())
            predictions.append(logits.argmax(dim=1).cpu())
        model.train(was_training)
        return classification_metrics(
            torch.cat(labels), torch.cat(predictions), self.num_classes
        )
