#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
from torch.utils.data import Dataset as TorchDataset


class TripleTensorDataset(TorchDataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.long()

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx


def _load_cache_file(path):
    try:
        obj = torch.load(path, map_location="cpu")
    except TypeError:
        obj = torch.load(path, map_location="cpu", weights_only=False)

    if "x" in obj and "y" in obj:
        x, y = obj["x"], obj["y"]
    elif "data" in obj and "label" in obj:
        x, y = obj["data"], obj["label"]
    else:
        raise KeyError(f"Cannot find x/y or data/label in {path}")

    return x, y


def _normalize_task_node(node):
    if isinstance(node, (list, tuple)):
        return tuple(_normalize_task_node(x) for x in node)
    return int(node)


def _parse_task(task):
    if task is None:
        return (0, 1)

    # Hydra/OmegaConf may pass ListConfig, which prints like [0, 1]
    # but is not a native Python list.
    try:
        from omegaconf import OmegaConf, ListConfig
        if isinstance(task, ListConfig):
            task = OmegaConf.to_container(task, resolve=True)
    except Exception:
        pass

    if isinstance(task, str):
        import ast
        task = ast.literal_eval(task)

    try:
        task = list(task)
    except TypeError:
        raise ValueError(f"TL_Task must be like [src, tar], got: {task}")

    if len(task) != 2:
        raise ValueError(f"TL_Task must be like [src, tar], got: {task}")

    return _normalize_task_node(task[0]), _normalize_task_node(task[1])


class CachedDomainDataset:
    num_classes = None
    cache_name = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data_path = Path(kwargs.get("data_path"))
        self.TL_Task = kwargs.get("TL_Task", [0, 1])

        if not self.data_path.exists():
            raise FileNotFoundError(f"data_path not found: {self.data_path}")

        self.domain_files = sorted(self.data_path.glob("D*_fft.pt"))
        if not self.domain_files:
            raise FileNotFoundError(f"No D*_fft.pt found in {self.data_path}")

    def _domain_file(self, domain_id):
        domain_id = int(domain_id)
        prefix = f"D{domain_id + 1}_"
        matched = [p for p in self.domain_files if p.name.startswith(prefix)]
        if len(matched) != 1:
            raise RuntimeError(
                f"Expected one cache file for domain {domain_id} with prefix {prefix}, "
                f"got {matched}"
            )
        return matched[0]

    def data_generator(self):
        src, tar = _parse_task(self.TL_Task)

        src_file = self._domain_file(src)
        tar_file = self._domain_file(tar)

        x_s, y_s = _load_cache_file(src_file)
        x_t, y_t = _load_cache_file(tar_file)

        print(f"[DATASET] {self.cache_name} source domain {src}: {src_file.name}, x={tuple(x_s.shape)}, y={tuple(y_s.shape)}")
        print(f"[DATASET] {self.cache_name} target domain {tar}: {tar_file.name}, x={tuple(x_t.shape)}, y={tuple(y_t.shape)}")
        print(f"[DATASET] source classes={sorted(y_s.unique().tolist())}")
        print(f"[DATASET] target classes={sorted(y_t.unique().tolist())}")

        return TripleTensorDataset(x_s, y_s), TripleTensorDataset(x_t, y_t)


class HUSTBAL(CachedDomainDataset):
    num_classes = 7
    cache_name = "HUSTBAL"


class UO(CachedDomainDataset):
    num_classes = 5
    cache_name = "UO"

class UOSPEED(CachedDomainDataset):
    num_classes = 5
    cache_name = "UOSPEED"
