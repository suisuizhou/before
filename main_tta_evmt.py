import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.nn.modules.batchnorm import _BatchNorm

from Lib.prompt_ckpt import load_vit_prompt_compatible_ckpt
from Lib.train_utils import seed_torch
from evmt.bn import TargetBNController
from evmt.runner import EVMTOnlineRunner
from main_tta_SDE import SDEFwarpTTATrainer, build_model_name, parse_only_task


def configure_visible_device(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)


def method_overrides(name):
    common = dict(
        use_bn_stats=True,
        train_bn_affine=False,
        train_adapter=False,
        train_warp=False,
        use_mt=False,
        use_evidence=False,
        use_pcl=False,
        use_ncl=False,
    )
    variants = {
        "bn_stat": {},
        "bn_affine": {"train_bn_affine": True},
        "bn_mt_adapter": {
            "train_bn_affine": True,
            "train_adapter": True,
            "train_warp": True,
            "use_mt": True,
        },
        "bnfirst_full": {
            "train_bn_affine": True,
            "train_adapter": True,
            "train_warp": True,
            "use_mt": True,
            "use_evidence": True,
            "use_pcl": True,
            "use_ncl": True,
        },
    }
    if name not in variants:
        raise ValueError(f"unknown BN-first method: {name}")
    values = dict(common)
    values.update(variants[name])
    return SimpleNamespace(**values)


def apply_method_overrides(cfg):
    method = str(cfg.TTA.method)
    overrides = method_overrides(method)
    with open_dict(cfg.TTA):
        for name, value in vars(overrides).items():
            cfg.TTA[name] = value
    return method


def freeze_bn_adapter_warp(
    model, train_bn_affine=True, train_adapter=True, train_warp=True
):
    for parameter in model.parameters():
        parameter.requires_grad = False
    bn_names = set()
    if train_bn_affine:
        for module_name, module in model.named_modules():
            if not isinstance(module, _BatchNorm) or not module.affine:
                continue
            prefix = f"{module_name}." if module_name else ""
            bn_names.update({prefix + "weight", prefix + "bias"})
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = (
            name in bn_names
            or (train_adapter and name.endswith(("band_scale", "band_bias")))
            or (train_warp and name.endswith("warp_ctrl"))
        )
        parameter.requires_grad = enabled
        if enabled:
            trainable.append(name)
    return trainable


def build_bnfirst_optimizer(model, cfg):
    bn_ids = set()
    for module in model.modules():
        if isinstance(module, _BatchNorm) and module.affine:
            bn_ids.update({id(module.weight), id(module.bias)})
    bn_params, adapter_params, warp_params = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in bn_ids:
            bn_params.append(parameter)
        elif name.endswith(("band_scale", "band_bias")):
            adapter_params.append(parameter)
        elif name.endswith("warp_ctrl"):
            warp_params.append(parameter)
    base_lr = float(cfg.Opt.lr_tar)
    weight_decay = float(cfg.Opt.weight_decay_tar)
    groups = [
        {
            "name": "bn_affine",
            "params": bn_params,
            "lr": base_lr * float(cfg.TTA.bn_lr_scale),
            "weight_decay": weight_decay,
        },
        {
            "name": "adapter",
            "params": adapter_params,
            "lr": base_lr * float(cfg.TTA.adapter_lr_scale),
            "weight_decay": weight_decay,
        },
        {
            "name": "warp",
            "params": warp_params,
            "lr": base_lr * float(cfg.TTA.warp_lr_scale),
            "weight_decay": weight_decay,
        },
    ]
    active = [group for group in groups if group["params"]]
    if not active:
        raise RuntimeError("no BN-first adaptation parameters were found")
    return torch.optim.AdamW(active)


def source_checkpoint(cfg, source):
    return (
        Path("TTA_Model")
        / f"{cfg.Dataset.data_name}{cfg.Opt.lr_src}"
        / f"source_{source}"
        / f"seed_{int(cfg.seed_run)}"
        / ("best_source_" + build_model_name(cfg))
    )


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(confusion):
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    false_positive = confusion.sum(0) - true_positive
    false_negative = confusion.sum(1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    scores = torch.where(
        denominator > 0,
        2 * true_positive / denominator.clamp_min(1),
        torch.zeros_like(denominator),
    )
    return 100.0 * float(scores.mean())


def run_task(cfg):
    configure_visible_device(cfg)
    task = parse_only_task(cfg.only_task)
    if task is None:
        raise ValueError("strict EVMT runs require only_task=[source,target]")
    method = apply_method_overrides(cfg)
    with open_dict(cfg):
        cfg.Dataset.TL_Task = task
        cfg.seed_run = int(getattr(cfg, "seed_run", 2025))
        cfg.model_name = build_model_name(cfg)
        cfg.batch_size = int(getattr(cfg, "batch_size", 128))
        cfg.num_workers = int(getattr(cfg, "num_workers", 4))
        cfg.Dataset.input_kind = "fft"

    seed_torch(cfg.seed_run)
    setup = SDEFwarpTTATrainer(cfg, None)
    setup.setup()
    checkpoint = source_checkpoint(cfg, int(task[0]))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = checkpoint_sha256(checkpoint)

    student = setup.model.to(setup.device)
    load_vit_prompt_compatible_ckpt(student, checkpoint, map_location=setup.device)
    teacher = copy.deepcopy(student)
    setup.enable_frequency_warp(student[0])
    setup.enable_frequency_warp(teacher[0])
    trainable_names = freeze_bn_adapter_warp(
        student,
        train_bn_affine=bool(cfg.TTA.train_bn_affine),
        train_adapter=bool(cfg.TTA.train_adapter),
        train_warp=bool(cfg.TTA.train_warp),
    )
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = build_bnfirst_optimizer(student, cfg) if trainable_names else None
    if not bool(cfg.TTA.use_bn_stats):
        raise ValueError("all declared BN-first methods require use_bn_stats=true")
    bn_controller = TargetBNController(
        student,
        blend_batches=int(cfg.TTA.bn_blend_batches),
        max_target_weight=float(cfg.TTA.bn_target_weight_max),
        eps=float(cfg.TTA.bn_eps),
    )
    runner = EVMTOnlineRunner(
        student,
        teacher,
        optimizer,
        cfg.TTA,
        setup.num_classes,
        cfg.Model.bottleneck_num,
        cfg.seed_run,
        bn_controller=bn_controller,
    )

    output = Path(
        getattr(cfg, "output_dir", "outputs/evmt/bnfirst_calibration")
    )
    run_dir = (
        output / f"{task[0]}-{task[1]}" / f"seed_{cfg.seed_run}" / method
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.unlink(missing_ok=True)
    (run_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    device = torch.device(setup.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_correct = 0
    total = 0
    updated_batches = 0
    skipped_batches = 0
    finite_batches = 0
    confusion = torch.zeros(
        setup.num_classes, setup.num_classes, dtype=torch.int64
    )
    stage_counts = {}
    last_metrics = None
    jsonl = run_dir / "batches.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for x, y, _ in setup.target_dataloader:
            x, y = x.to(setup.device), y.to(setup.device)
            metrics = runner.step(x, y)
            last_metrics = metrics
            total_correct += metrics.correct
            total += metrics.samples
            updated_batches += int(metrics.updated)
            skipped_batches += int(bool(metrics.skip_reasons))
            finite_batches += int(math.isfinite(metrics.loss))
            stage_counts[metrics.stage] = stage_counts.get(metrics.stage, 0) + 1
            predicted = runner.last_preupdate_logits.argmax(1)
            pairs = (y.detach() * setup.num_classes + predicted).to("cpu")
            confusion += torch.bincount(
                pairs, minlength=setup.num_classes * setup.num_classes
            ).reshape(setup.num_classes, setup.num_classes)
            handle.write(json.dumps(metrics.as_dict(), ensure_ascii=False) + "\n")
            print(
                f"step={metrics.step} stage={metrics.stage} "
                f"acc={100 * total_correct / total:.2f} loss={metrics.loss:.4f} "
                f"coverage={metrics.pred_coverage}/{setup.num_classes} "
                f"updated={int(metrics.updated)}"
            )

    if last_metrics is None:
        raise RuntimeError("target dataloader produced no batches")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_gpu_memory_mb = 0.0
    elapsed = time.perf_counter() - started
    summary = {
        "task": [int(task[0]), int(task[1])],
        "seed": int(cfg.seed_run),
        "method": method,
        "samples": total,
        "seen_samples": runner.seen_samples,
        "passes": 1,
        "batches": last_metrics.step,
        "updated_batches": updated_batches,
        "skipped_batches": skipped_batches,
        "finite_batches": finite_batches,
        "online_accuracy": 100.0 * total_correct / max(total, 1),
        "online_macro_f1": macro_f1(confusion),
        "final_stage": runner.stages.stage.value,
        "stage_counts": stage_counts,
        "final_pred_coverage": last_metrics.pred_coverage,
        "final_recent_pred_coverage": last_metrics.recent_pred_coverage,
        "final_effective_classes": last_metrics.effective_classes,
        "final_memory_coverage": last_metrics.memory_coverage,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "trainable_parameters": trainable_names,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EVMT_SUMMARY", json.dumps(summary, ensure_ascii=False))
    return summary


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def main(cfg):
    run_task(cfg)


if __name__ == "__main__":
    main()
