#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""DtCC-ViT-Prompt for PU4D Protocol A.

DtCC algorithmic components are preserved (dynamic division, class-balanced
memory, PCL, NCL, SEM).  Because ViT1D has no ResNet-style BN adaptation
carrier, Protocol A freezes the entire network and updates only prompt_embed.
The arriving target batch is predicted before adaptation; that prediction is
reported as the strict online metric.  Each arriving batch is then optimized
for two steps, matching DtCC's target update count.
"""

from __future__ import annotations

import ast
import os
import time
from itertools import permutations
from pathlib import Path

import hydra
import omegaconf
from omegaconf import open_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import Dataset
from Lib.fixed_random_stream import make_fixed_random_stream_loader
from Lib.model import get_model
from Lib.train_utils import seed_torch
from Lib.vit_protocol_a import (
    DtCCMemory,
    apply_vit_cfg,
    compatible_load,
    confusion_update,
    dtcc_dynamic_divide,
    forward_parts,
    macro_f1_from_confusion,
    method_checkpoint_dir,
    select_trainable_parameters,
)


def _plain(value):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def parse_task(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Task must look like [src,tar], got {value!r}")
    return int(value[0]), int(value[1])


def _model_name(seed: int) -> str:
    return f"ViT1D{int(seed)}fft_Linear.pt"


def _classifier(model: nn.Module):
    return model[2]


def sem_loss(x: torch.Tensor, logits: torch.Tensor, alpha: float = 2.0):
    certain, _, probability, pseudo, _, _ = dtcc_dynamic_divide(x, logits)
    k = probability.shape[1]
    if certain.any():
        frequency = torch.bincount(pseudo[certain], minlength=k).float().to(probability.device)
    else:
        frequency = torch.zeros(k, device=probability.device)

    weighted = probability / (frequency.unsqueeze(0) + 1.0)
    weighted = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if abs(float(alpha) - 1.0) < 1e-8:
        l_te = -(weighted * weighted.clamp_min(1e-12).log()).sum(dim=1).mean()
    else:
        l_te = (1.0 - weighted.clamp_min(1e-12).pow(float(alpha)).sum(dim=1)).mean()
        l_te = l_te / (float(alpha) - 1.0)
    p_bar = weighted.mean(dim=0)
    l_div = (p_bar * p_bar.clamp_min(1e-12).log()).sum()
    return l_te + l_div


def pcl_loss(features, pseudo, certain, uncertain, memory: DtCCMemory):
    zero = features.sum() * 0.0
    if not certain.any():
        return zero
    prototypes, classes = memory.prototypes(features.device)
    if prototypes is None or not classes:
        return zero

    prototypes_n = F.normalize(prototypes, dim=1)
    class_to_row = {int(k): i for i, k in enumerate(classes)}
    uncertain_features = F.normalize(features[uncertain], dim=1) if uncertain.any() else None
    losses = []
    for idx in torch.nonzero(certain, as_tuple=False).flatten():
        cls = int(pseudo[idx].item())
        if cls not in class_to_row:
            continue
        query = F.normalize(features[idx : idx + 1], dim=1)
        proto_scores = torch.exp(query @ prototypes_n.t()).flatten()
        positive = proto_scores[class_to_row[cls]]
        denominator = proto_scores.sum()
        if uncertain_features is not None and uncertain_features.numel() > 0:
            denominator = denominator + torch.exp(query @ uncertain_features.t()).sum()
        losses.append(-torch.log(positive / denominator.clamp_min(1e-12)))
    return torch.stack(losses).mean() if losses else zero


def ncl_loss(features, logits, uncertain, memory: DtCCMemory, neighbors: int = 5):
    zero = logits.sum() * 0.0
    uncertain_indices = torch.nonzero(uncertain, as_tuple=False).flatten()
    if uncertain_indices.numel() == 0:
        return zero
    mem_feat, mem_prob, _ = memory.all_entries(features.device)
    if mem_feat is None:
        return zero

    mem_feat = F.normalize(mem_feat, dim=1)
    current_prob = F.softmax(logits, dim=1)
    losses = []
    for idx in uncertain_indices:
        query_feat = F.normalize(features[idx : idx + 1], dim=1)
        sim = (query_feat @ mem_feat.t()).flatten()
        top_k = min(max(1, int(neighbors)), int(mem_feat.shape[0]))
        top_idx = sim.topk(top_k).indices

        p_i = current_prob[idx : idx + 1]
        positive_prob = mem_prob[top_idx]
        positive_h = torch.exp(F.cosine_similarity(p_i.expand_as(positive_prob), positive_prob, dim=1))
        numerator = positive_h.mean()

        other_mask = uncertain_indices != idx
        other_idx = uncertain_indices[other_mask]
        if other_idx.numel() > 0:
            negative_prob = current_prob[other_idx]
            negative_h = torch.exp(F.cosine_similarity(p_i.expand_as(negative_prob), negative_prob, dim=1))
            denominator = positive_h.sum() + negative_h.sum()
        else:
            denominator = positive_h.sum()
        losses.append(-torch.log(numerator / denominator.clamp_min(1e-12)))
    return torch.stack(losses).mean() if losses else zero


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y, *_ in loader:
        x = x.to(device)
        y = y.to(device)
        _, logits = forward_parts(model, x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        confusion_update(confusion, y, pred)
    acc = 100.0 * correct / max(1, total)
    return acc, macro_f1_from_confusion(confusion)


def update_memory_from_batch(memory, features, probability, pseudo, confidence, certain):
    for idx in torch.nonzero(certain, as_tuple=False).flatten():
        i = int(idx.item())
        memory.add(
            int(pseudo[i].item()),
            features[i],
            probability[i],
            float(confidence[i].item()),
        )


def run_one(cfg, task):
    src, tar = task
    with open_dict(cfg):
        cfg.Dataset.TL_Task = [src, tar]
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
        cfg.seed_run = int(getattr(cfg, "seed_run", 2025))
        cfg.batch_size = int(getattr(cfg, "batch_size", 128))
        cfg.num_workers = int(getattr(cfg, "num_workers", 4))
        cfg.stream_seed = int(getattr(cfg, "stream_seed", 2025))
        cfg.process_wandb = False
        cfg.Opt.lr_tar = float(getattr(cfg.Opt, "lr_tar", 1e-2))
        cfg.Opt.weight_decay_tar = float(getattr(cfg.Opt, "weight_decay_tar", 1e-3))
        apply_vit_cfg(cfg, use_spectral_adapter=False)

    seed_torch(cfg.seed_run)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id).strip("\"'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls = getattr(Dataset, cfg.Dataset.data_name)
    num_classes = int(dataset_cls.num_classes)
    _, target_dataset = dataset_cls(**cfg.Dataset).data_generator()

    eval_loader = DataLoader(
        target_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    stream_loader = make_fixed_random_stream_loader(
        target_dataset,
        batch_size=cfg.batch_size,
        seed=cfg.stream_seed,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = get_model(num_classes=num_classes, cfg=cfg, **cfg.Model).to(device)
    ckpt_dir = Path(
        getattr(
            cfg,
            "source_ckpt_dir",
            str(method_checkpoint_dir("DTCC_VIT", "PU4D", src, cfg.seed_run)),
        )
    )
    ckpt = ckpt_dir / ("best_source_" + _model_name(cfg.seed_run))
    if not ckpt.exists():
        ckpt = ckpt_dir / _model_name(cfg.seed_run)
    if not ckpt.exists():
        raise FileNotFoundError(f"DtCC-ViT source checkpoint not found: {ckpt_dir}")
    audit = compatible_load(model, ckpt, device)
    print(f"[SOURCE CKPT] {ckpt} loaded={audit['loaded']} missing={audit['missing']}")

    selected = select_trainable_parameters(model, method="dtcc")
    print("[TRAINABLE]", selected)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.Opt.lr_tar,
        weight_decay=cfg.Opt.weight_decay_tar,
    )
    optim_steps = 2
    memory = DtCCMemory(num_classes=num_classes, capacity_per_class=50)
    memory.initialize_from_classifier(_classifier(model), device)

    before, _ = evaluate(model, eval_loader, device, num_classes)
    print(f"Task: [{src},{tar}]: Before TTA = {before:.2f}%")

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    online_correct = 0
    online_total = 0
    batch_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()  # keep dropout deterministic; prompt gradients still propagate
    for batch_index, batch in enumerate(stream_loader, start=1):
        x, y = batch[0].to(device), batch[1].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        # Strict online prediction BEFORE the target update.
        with torch.no_grad():
            initial_features, initial_logits = forward_parts(model, x)
            initial_prob = F.softmax(initial_logits, dim=1)
            initial_conf, initial_pseudo = initial_prob.max(dim=1)
            certain0, _, _, _, _, _ = dtcc_dynamic_divide(x, initial_logits)
            pred = initial_logits.argmax(dim=1)
            online_correct += int((pred == y).sum().item())
            online_total += int(y.numel())
            confusion_update(confusion, y, pred)

        # DtCC Algorithm 1 updates the class-balanced memory with the current
        # certain set before PCL/NCL. Target labels are never used here.
        update_memory_from_batch(
            memory,
            initial_features,
            initial_prob,
            initial_pseudo,
            initial_conf,
            certain0,
        )

        for _ in range(optim_steps):
            optimizer.zero_grad(set_to_none=True)
            features, logits = forward_parts(model, x)
            certain, uncertain, _, pseudo, _, _ = dtcc_dynamic_divide(x, logits)
            loss_sem = sem_loss(x, logits, alpha=2.0)
            loss_pcl = pcl_loss(features, pseudo, certain, uncertain, memory)
            loss_ncl = ncl_loss(features, logits, uncertain, memory, neighbors=5)
            loss = loss_sem + loss_pcl + loss_ncl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DtCC loss at batch {batch_index}: {loss.item()}")
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_times.append((time.perf_counter() - start) * 1000.0)

    online = 100.0 * online_correct / max(1, online_total)
    online_f1 = macro_f1_from_confusion(confusion)
    post, post_f1 = evaluate(model, eval_loader, device, num_classes)
    mean_ms = sum(batch_times) / max(1, len(batch_times))
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    print(f"Task: [{src},{tar}]: Online Acc = {online:.2f}%;")
    print(f"Task: [{src},{tar}]: Final Full-Target Acc = {post:.2f}%")
    print(
        f"[RESULT] method=DTCC_VIT task=[{src},{tar}] before={before:.4f} "
        f"online={online:.4f} post={post:.4f} online_f1={online_f1:.4f} "
        f"post_f1={post_f1:.4f} batch_ms={mean_ms:.4f} peak_mb={peak_mb:.4f}"
    )


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    only = getattr(cfg, "only_task", None)
    tasks = [parse_task(only)] if only is not None else list(permutations(cfg.Dataset.TL_list, 2))
    for task in tasks:
        run_one(cfg, task)


if __name__ == "__main__":
    run()
