#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-facing PU4D plumbing for Protocol-B experiments."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List

import omegaconf
import torch
import torch.nn as nn
from omegaconf import open_dict
from torch.utils.data import DataLoader

import Dataset
from Lib.fixed_random_stream import make_fixed_random_stream_loader
from Lib.model import get_model
from Lib.protocol_b_common import method_checkpoint_dir, strict_load_state_dict
from Lib.protocol_b_source import evaluate_classifier, forward_parts
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
    suffix = {"linear": "_Linear.pt", "wn": "_WN.pt"}.get(str(cfg.Model.model_type), "_Pro.pt")
    return f"{cfg.Model.model_name}{cfg.seed_run}{cfg.Dataset.input_kind}{suffix}"


def _get(section, name: str, default):
    if isinstance(section, dict):
        return section.get(name, default)
    return getattr(section, name, default)


def classification_metrics(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> Dict[str, float]:
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    accuracy = (labels == predictions).float().mean().item() * 100.0
    precisions, recalls, f1s = [], [], []
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


class PredictionAccumulator:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.labels = []
        self.predictions = []

    def update(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        self.labels.append(labels.detach().cpu())
        self.predictions.append(predictions.detach().cpu())

    def metrics(self):
        if not self.labels:
            return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
        return classification_metrics(torch.cat(self.labels), torch.cat(self.predictions), self.num_classes)


class ProtocolBPU4DBase:
    method = ""

    def __init__(self, cfg: omegaconf.DictConfig, run_obj=None):
        seed_torch(cfg.seed_run)
        self.cfg = cfg
        self.run = run_obj
        self.device = None
        self.num_classes = 0
        self.datasets = {}
        self.eval_loader = None
        self.stream_loader = None
        self.model = None
        self.stream_seed = 0

    def setup(self):
        cfg = self.cfg
        seed_torch(cfg.seed_run)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logging.info("using %s GPUs", torch.cuda.device_count())
        else:
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
            target_data,
            batch_size=int(cfg.batch_size),
            seed=self.stream_seed,
            num_workers=int(cfg.num_workers),
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )
        self.model = get_model(num_classes=self.num_classes, cfg=cfg, **cfg.Model).to(self.device)

    def checkpoint_path(self) -> Path:
        root = Path(str(_get(getattr(self.cfg, "ProtocolB", {}), "checkpoint_root", "TTA_Model_B")))
        source = int(self.cfg.Dataset.TL_Task[0])
        directory = method_checkpoint_dir(root, self.method, self.cfg.Dataset.data_name, source, self.cfg.seed_run)
        model_name = build_model_name(self.cfg)
        best = directory / f"best_source_{model_name}"
        final = directory / model_name
        if best.is_file():
            return best
        if final.is_file():
            return final
        raise FileNotFoundError(f"Protocol-B {self.method} checkpoint missing: {best} or {final}")

    def initialize_method_source(self):
        path = self.checkpoint_path()
        strict_load_state_dict(self.model, path, self.device)
        print(f"[PROTOCOL-B SOURCE] method={self.method} source={self.cfg.Dataset.TL_Task[0]} checkpoint={path}")

    @torch.no_grad()
    def evaluate(self, model: nn.Module):
        was_training = model.training
        model.eval()
        labels, predictions = [], []
        for x, y, *_ in self.eval_loader:
            x = x.to(self.device)
            _, logits = forward_parts(model, x)
            labels.append(y.cpu())
            predictions.append(logits.argmax(dim=1).cpu())
        model.train(was_training)
        return classification_metrics(torch.cat(labels), torch.cat(predictions), self.num_classes)


def prepare_common_cfg(cfg, seed: int, task):
    with open_dict(cfg):
        cfg.seed_run = int(seed)
        cfg.Dataset.TL_Task = task
        cfg.Dataset.input_kind = "fft"
        cfg.Model.bottleneck_num = 128
        cfg.Model.use_spectral_adapter = False
        cfg.batch_size = int(getattr(cfg, "batch_size", 128))
        cfg.num_workers = int(getattr(cfg, "num_workers", 4))
        cfg.stream_seed = int(getattr(cfg, "stream_seed", 2025))
        cfg.model_name = build_model_name(cfg)
    return cfg
