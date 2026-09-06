#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one vanilla PU4D ResNet18 source model without source robustness augmentation."""
from __future__ import annotations

from pathlib import Path
import os
import shutil

import hydra
import omegaconf
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader

import Dataset
from Lib.model import get_model
from Lib.optimizer import get_lr_scheduler, get_optimizer
from Lib.train_utils import seed_torch
from Lib.pu4d_common_source import build_model_name, forward_parts, classification_metrics
from Lib.pu4d_vanilla_protocol import (
    choose_dummy_target,
    reset_and_freeze_adaptation_carrier,
    vanilla_checkpoint_dir,
)


def _evaluate(model, loader, device, num_classes):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            _, logits = forward_parts(model, x)
            ys.append(y.cpu())
            ps.append(logits.argmax(dim=1).cpu())
    return classification_metrics(torch.cat(ys), torch.cat(ps), num_classes)


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    source = int(getattr(cfg, "only_source"))
    seed = int(getattr(cfg, "seed_run", 2025))
    domains = [int(v) for v in cfg.Dataset.TL_list]
    dummy_target = choose_dummy_target(source, domains)
    vanilla_root = Path(str(getattr(cfg, "vanilla_source_root", "TTA_Model_VANILLA")))

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id).strip("\"'")

    with open_dict(cfg):
        cfg.seed_run = seed
        cfg.Dataset.TL_Task = (source, dummy_target)
        cfg.Dataset.input_kind = "fft"
        cfg.Model.bottleneck_num = 128
        cfg.Model.use_spectral_adapter = True
        cfg.Model.band_num = 256
        cfg.Model.adapter_delta = 0.1
        cfg.Opt.lr_src = 1e-3
        cfg.Opt.weight_decay_src = 1e-4
        cfg.Opt.lr_scheduler = 'designed'
        cfg.src_epoch = 50
        cfg.batch_size = int(getattr(cfg, "batch_size", 128))
        cfg.num_workers = int(getattr(cfg, "num_workers", 4))
        cfg.model_name = build_model_name(cfg)

    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls = getattr(Dataset, cfg.Dataset.data_name)
    num_classes = int(dataset_cls.num_classes)
    source_data, _ = dataset_cls(**cfg.Dataset).data_generator()

    train_loader = DataLoader(
        source_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    eval_loader = DataLoader(
        source_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = get_model(num_classes=num_classes, cfg=cfg, **cfg.Model).to(device)
    frozen = reset_and_freeze_adaptation_carrier(model)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Vanilla source has no trainable parameters")

    optimizer = get_optimizer(model, args=cfg.Opt, kind="src")
    lr_scheduler = get_lr_scheduler(
        optimizer, args=cfg.Opt, epoch_train=cfg.src_epoch, kind="src"
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"[VANILLA SOURCE] source={source} dummy_target={dummy_target} seed={seed}")
    print(f"[VANILLA SOURCE] samples={len(source_data)} batch={cfg.batch_size} epochs={cfg.src_epoch}")
    print(f"[VANILLA SOURCE] frozen target-only params={frozen}")
    print("[VANILLA SOURCE] source augmentation=NONE")

    for epoch in range(1, cfg.src_epoch + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for x, y, _ in train_loader:
            if x.size(0) <= 1:
                continue
            x = x.to(device)
            y = y.to(device)
            _, logits = forward_parts(model, x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * x.size(0)
            total_seen += x.size(0)
        if lr_scheduler is not None:
            lr_scheduler.step()
        metrics = _evaluate(model, eval_loader, device, num_classes)
        print(
            f"[VANILLA SOURCE] epoch={epoch:02d}/{cfg.src_epoch} "
            f"loss={total_loss/max(total_seen,1):.6f} source_acc={metrics['accuracy']:.2f}% "
            f"source_f1={metrics['macro_f1']:.2f}%"
        )

    directory = vanilla_checkpoint_dir(vanilla_root, source, seed)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / cfg.model_name
    best_alias = directory / f"best_source_{cfg.model_name}"
    torch.save(model.state_dict(), final)
    shutil.copy2(final, best_alias)
    print(f"[SAVE] final -> {final}")
    print(f"[SAVE] compatibility best_source alias -> {best_alias}")
    print(f"[RESULT] method=VANILLA_SOURCE source={source} source_acc={metrics['accuracy']:.2f} source_f1={metrics['macro_f1']:.2f}")


if __name__ == "__main__":
    run()
