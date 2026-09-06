import hashlib
import json
import random
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from main_src_sde_evmt_r18 import build_model, model_signature
from sde_evmt_r18.augment import cfg_get
from sde_evmt_r18.data import load_pu4d_domain
from sde_evmt_r18.runner import SDEEVMTOnlineRunner


def variant_switches(variant):
    variant = str(variant).upper()
    if variant not in {"R2", "R3", "R4", "R5", "R6"}:
        raise ValueError(f"target variant must be R2-R6, got {variant}")
    number = int(variant[1:])
    return {
        "multiview_teacher": number >= 3,
        "mean_teacher": number >= 3,
        "evidence": number >= 4,
        "memory": number >= 5,
        "pcl": number >= 5,
        "ncl": number >= 6,
        "feature_adapter": number >= 6,
    }


def target_output_dir(root, source, target, seed, variant):
    return (
        Path(root)
        / f"{int(source)}_to_{int(target)}"
        / f"seed_{int(seed)}"
        / str(variant).upper()
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_variant(checkpoint):
    config = checkpoint.get("config", {})
    if OmegaConf.is_config(config):
        return str(cfg_get(config, "variant", "")).upper()
    return str(config.get("variant", "")).upper() if isinstance(config, dict) else ""


def load_source_checkpoint(model, path):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if _checkpoint_variant(checkpoint) != "R1":
        raise ValueError("SDE-EVMT target adaptation requires an R1 source checkpoint")
    expected = model_signature(model)
    actual = checkpoint.get("model_signature")
    if actual != expected:
        raise ValueError(f"source checkpoint model signature mismatch: {actual} != {expected}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "checkpoint": checkpoint,
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def _plain(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return {key: _plain(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_plain(value) for value in cfg]
    return cfg


def _task(cfg):
    task = cfg_get(cfg, "only_task", None)
    if task is None:
        task = [cfg_get(cfg, "source", 0), cfg_get(cfg, "target", 1)]
    if isinstance(task, str):
        task = json.loads(task)
    if len(task) != 2:
        raise ValueError(f"only_task must contain [source,target], got {task}")
    source, target = int(task[0]), int(task[1])
    if source == target:
        raise ValueError("source and target domains must differ")
    return source, target


def _macro_f1(confusion):
    confusion = confusion.float()
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1.0)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return 100.0 * float(f1.mean())


def _default_checkpoint(cfg, source, seed):
    root = Path(cfg_get(cfg, "checkpoint_root", "TTA_Model/SDE_EVMT_R18/PU4D"))
    return root / f"source_{source}" / f"seed_{seed}" / "R1" / "best.pt"


def run_target(cfg):
    variant = str(cfg_get(cfg, "variant", "R6")).upper()
    switches = variant_switches(variant)
    source, target = _task(cfg)
    seed = int(cfg_get(cfg, "seed", 1))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested_device = str(cfg_get(cfg, "device", "cuda"))
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = build_model(cfg).to(device)
    checkpoint_path = cfg_get(cfg, "source_checkpoint", None)
    if checkpoint_path in (None, "null", ""):
        checkpoint_path = _default_checkpoint(cfg, source, seed)
    loaded = load_source_checkpoint(model, checkpoint_path)
    source_checkpoint = loaded["checkpoint"]

    dataset = load_pu4d_domain(
        cfg_get(cfg, "dataset.cache_root", "Dataset/PU4D_CACHE"),
        target,
        cfg_get(cfg, "dataset.input_kind", "fft"),
    )
    batch_size = int(cfg_get(cfg, "data.batch_size", 512))
    workers = int(cfg_get(cfg, "data.num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    smoke_batches = cfg_get(cfg, "smoke_batches", None)
    smoke_batches = None if smoke_batches is None else int(smoke_batches)
    output = target_output_dir(
        cfg_get(cfg, "output_root", "outputs/sde_evmt_r18"),
        source,
        target,
        seed,
        variant,
    )
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.yaml"
    batches_path = output / "batches.jsonl"
    summary_path = output / "summary.json"
    log_path = output / "run.log"
    for stale in (batches_path, summary_path, log_path):
        if stale.exists():
            stale.unlink()
    config_path.write_text(OmegaConf.to_yaml(OmegaConf.create(_plain(cfg)), resolve=True))

    runner = SDEEVMTOnlineRunner(
        model,
        cfg=cfg,
        variant=variant,
        num_classes=int(cfg_get(cfg, "num_classes", 32)),
        seed=seed,
    )
    confusion = torch.zeros(runner.num_classes, runner.num_classes, dtype=torch.long)
    seen = 0
    correct = 0
    updated = 0
    skipped = 0
    rows = 0
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with log_path.open("w") as log_handle, batches_path.open("w") as batch_handle:
        log_handle.write(
            f"start variant={variant} task={source}->{target} checkpoint={loaded['path']}\n"
        )
        for batch_index, (x, y) in enumerate(loader):
            if smoke_batches is not None and batch_index >= smoke_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            metrics = runner.step(x, y_for_metrics=y)
            prediction = runner.last_preupdate_logits.argmax(dim=1)
            pair = (y.detach().cpu() * runner.num_classes + prediction.detach().cpu())
            confusion += torch.bincount(
                pair, minlength=runner.num_classes * runner.num_classes
            ).reshape(runner.num_classes, runner.num_classes)
            row = metrics.as_dict()
            row["batch_index"] = batch_index
            batch_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            batch_handle.flush()
            seen += len(y)
            correct += metrics.correct
            updated += int(metrics.updated)
            skipped += int(not metrics.updated)
            rows += 1
            message = (
                f"batch={batch_index + 1} samples={seen} loss={metrics.loss:.6f} "
                f"stage={metrics.stage} updated={metrics.updated} "
                f"reliability={metrics.reliability:.6f}\n"
            )
            log_handle.write(message)
            log_handle.flush()
            if batch_index == 0 or (batch_index + 1) % 10 == 0:
                print(message.rstrip(), flush=True)

    elapsed = time.time() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    summary = {
        "variant": variant,
        "switches": switches,
        "source_domain": source,
        "target_domain": target,
        "seed": seed,
        "passes": 1,
        "dataset_samples": len(dataset),
        "seen_samples": seen,
        "batch_rows": rows,
        "expected_full_batches": (len(dataset) + batch_size - 1) // batch_size,
        "smoke_batches": smoke_batches,
        "updated_batches": updated,
        "skipped_batches": skipped,
        "accuracy": 100.0 * correct / max(seen, 1),
        "macro_f1": _macro_f1(confusion),
        "final_stage": runner.stage.name,
        "final_gate_values": model.gate_summary(),
        "final_memory_entries": runner.memory.stats().total_entries,
        "final_memory_coverage": runner.memory.stats().covered_classes,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "source_checkpoint": loaded["path"],
        "source_checkpoint_sha256": loaded["sha256"],
        "source_accuracy": float(source_checkpoint["source_accuracy"]),
        "model_signature": source_checkpoint["model_signature"],
        "output_dir": str(output.resolve()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


@hydra.main(version_base=None, config_path="Configs/SDE_EVMT_R18", config_name="target")
def main(cfg: DictConfig):
    run_target(cfg)


if __name__ == "__main__":
    main()
