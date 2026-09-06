import hashlib
import json
import random
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sde_evmt_r18.augment import SourceAugmenter, cfg_get
from sde_evmt_r18.data import load_pu4d_domain
from sde_evmt_r18.model import SDEEVMTResNet18
from sde_evmt_r18.source import source_objective


def _plain(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return {key: _plain(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_plain(value) for value in cfg]
    return cfg


def _setting(cfg, name, default):
    value = cfg_get(cfg, name, None)
    if value is not None:
        return value
    return cfg_get(cfg, f"training.{name}", default)


def source_output_dir(root, source, seed, variant):
    return Path(root) / f"source_{int(source)}" / f"seed_{int(seed)}" / str(variant).upper()


def build_model(cfg):
    model_cfg = cfg_get(cfg, "model", {})
    return SDEEVMTResNet18(
        num_classes=int(cfg_get(cfg, "num_classes", 32)),
        input_len=int(cfg_get(model_cfg, "input_len", 512)),
        bottleneck_dim=int(cfg_get(model_cfg, "bottleneck_dim", 256)),
        feature_hidden_dim=int(cfg_get(model_cfg, "feature_hidden_dim", 64)),
        gn_groups=int(cfg_get(model_cfg, "gn_groups", 8)),
        warp_knots=int(cfg_get(model_cfg, "warp_knots", 16)),
        max_warp=float(cfg_get(model_cfg, "max_warp", 2.0)),
        spectral_bands=int(cfg_get(model_cfg, "spectral_bands", 64)),
        adapter_delta=float(cfg_get(model_cfg, "adapter_delta", 0.1)),
    )


def model_signature(model):
    shapes = {
        name: list(value.shape)
        for name, value in sorted(model.state_dict().items())
    }
    encoded = json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()
    return {
        "architecture": "SDEEVMTResNet18-1D-GN",
        "state_shapes_sha256": hashlib.sha256(encoded).hexdigest(),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_source(model, loader, device):
    model.eval()
    correct = 0
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prediction = model(x).argmax(dim=-1)
        correct += int((prediction == y).sum())
        seen += len(y)
    return 100.0 * correct / max(seen, 1)


def train_source(cfg):
    variant = str(cfg_get(cfg, "variant", "R1")).upper()
    if variant not in {"R0", "R1"}:
        raise ValueError(f"source variant must be R0 or R1, got {variant}")
    source = int(cfg_get(cfg, "source", 0))
    seed = int(cfg_get(cfg, "seed", 1))
    epochs = int(_setting(cfg, "epochs", 80))
    smoke_batches = cfg_get(cfg, "smoke_batches", None)
    smoke_batches = None if smoke_batches is None else int(smoke_batches)
    requested_device = str(cfg_get(cfg, "device", "cuda"))
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    dataset = load_pu4d_domain(
        cfg_get(cfg, "dataset.cache_root", "Dataset/PU4D_CACHE"),
        source,
        cfg_get(cfg, "dataset.input_kind", "fft"),
    )
    batch_size = int(_setting(cfg, "batch_size", 64))
    eval_batch_size = int(_setting(cfg, "eval_batch_size", 512))
    workers = int(_setting(cfg, "num_workers", 2))
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=loader_generator,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_setting(cfg, "lr", 0.001)),
        weight_decay=float(_setting(cfg, "weight_decay", 0.00001)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    augment_generator = torch.Generator(device=device).manual_seed(seed + 104729)
    augmenter = SourceAugmenter(cfg, augment_generator)

    output = source_output_dir(
        cfg_get(cfg, "output_root", "TTA_Model/SDE_EVMT_R18/PU4D"),
        source,
        seed,
        variant,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    history_path = output / "history.json"
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary_path.unlink()

    history = []
    best_accuracy = -1.0
    best_epoch = -1
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {}
        processed = 0
        for batch_index, (x, y) in enumerate(train_loader):
            if smoke_batches is not None and batch_index >= smoke_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = source_objective(model, x, y, augmenter, cfg)
            if not bool(torch.isfinite(losses.total)):
                raise FloatingPointError(f"non-finite source loss at epoch {epoch} batch {batch_index}")
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values = losses.detached()
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value * len(y)
            processed += len(y)
        scheduler.step()
        accuracy = evaluate_source(model, eval_loader, device)
        row = {
            "epoch": epoch,
            "trained_samples": processed,
            "source_accuracy": accuracy,
            "lr": optimizer.param_groups[0]["lr"],
            "gates": model.gate_summary(),
        }
        row.update({name: value / max(processed, 1) for name, value in totals.items()})
        history.append(row)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "source_accuracy": accuracy,
                "config": _plain(cfg),
                "model_signature": model_signature(model),
            }
            torch.save(checkpoint, checkpoint_path)
        print(
            f"[Source {source} {variant}] epoch={epoch}/{epochs} "
            f"loss={row.get('total', float('nan')):.5f} acc={accuracy:.4f}% "
            f"gates={row['gates']}",
            flush=True,
        )

    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if reloaded["source_accuracy"] != best_accuracy:
        raise RuntimeError("checkpoint reload verification failed")
    summary = {
        "variant": variant,
        "source_domain": source,
        "seed": seed,
        "epochs": epochs,
        "smoke_batches": smoke_batches,
        "dataset_samples": len(dataset),
        "best_epoch": best_epoch,
        "best_source_accuracy": best_accuracy,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_signature": reloaded["model_signature"],
        "elapsed_seconds": time.time() - started,
        "output_dir": str(output.resolve()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


@hydra.main(version_base=None, config_path="Configs/SDE_EVMT_R18", config_name="source")
def main(cfg: DictConfig):
    train_source(cfg)


if __name__ == "__main__":
    main()
