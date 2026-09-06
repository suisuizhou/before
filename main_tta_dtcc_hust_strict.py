#!/usr/bin/env python3
"""Strict single-pass DtCC target adaptation for the HUST protocol."""

from __future__ import annotations

from itertools import permutations
import json
import math
import os
from pathlib import Path
from pprint import pprint
import time

import hydra
import omegaconf
import torch
import wandb
from omegaconf import open_dict

from Lib.dtcc_resnet18_common import (
    DtCCMemoryBank,
    configure_bn_only,
    dtcc_ncl_loss,
    dtcc_pcl_loss,
    dtcc_sem_loss,
    dynamic_data_division,
    spectral_entropy,
)
from Lib.hust_strict_protocol import (
    apply_hust_protocol_split,
    resolve_hust_checkpoint,
    strict_load_hust_checkpoint,
)
from Lib.pu4d_common_source import (
    CommonSourcePU4DBase,
    PredictionAccumulator,
    _get,
    forward_parts,
    parse_only_task,
    parse_seed_runs,
)


def _metrics_from_confusion(confusion_matrix) -> dict[str, float]:
    matrix = torch.as_tensor(confusion_matrix, dtype=torch.long).cpu()
    if matrix.shape != (7, 7) or bool((matrix < 0).any()):
        raise ValueError("HUST result confusion matrix must be non-negative 7x7")
    precision, recall, f1 = [], [], []
    for class_id in range(7):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum()) - tp
        fn = int(matrix[class_id, :].sum()) - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(p + r, 1e-12))
    return {
        "macro_precision": 100.0 * sum(precision) / 7,
        "macro_recall": 100.0 * sum(recall) / 7,
        "macro_f1": 100.0 * sum(f1) / 7,
    }


def build_hust_result_record(
    *, result_kind: str, route: str, variant: str, task, source_checkpoint_sha256: str,
    config_sha256: str, candidate_id: str, source_seed: int, stream_seed: int,
    beginning: float, strict_online: float | None, post_stream: float | None,
    confusion_matrix, samples: int, batches: int, passes: int, finite_losses: bool,
    trainable_parameters, pre_update_scoring: bool, metadata_evidence_used: bool,
    runtime_seconds: float, peak_memory_mb: float, **diagnostics,
) -> dict:
    """Build the canonical cross-route result contract consumed by formal orchestration."""
    if result_kind not in {"beginning", "target"} or route not in {"dtcc_ordinary", "0711_robust", "0711_common"}:
        raise ValueError("invalid HUST result identity")
    source, target = int(task[0]), int(task[1])
    if source == target or variant not in {"ordinary", "robust"}:
        raise ValueError("invalid HUST route task/variant")
    target_diagnostics = {
        "memory_size", "memory_class_coverage", "certain_ratio", "uncertain_ratio",
        "offline_purity", "certain_purity", "evidence_applicable", "evidence_active_ratio",
    }
    if result_kind == "target" and (strict_online is None or confusion_matrix is None or target_diagnostics - set(diagnostics)):
        raise ValueError(f"formal target result requires strict diagnostics: {sorted(target_diagnostics - set(diagnostics))}")
    metrics = _metrics_from_confusion(confusion_matrix) if confusion_matrix is not None else {
        "macro_precision": None, "macro_recall": None, "macro_f1": None,
    }
    record = {
        "schema_version": 1, "result_kind": result_kind, "route": route,
        "variant": variant, "source": source, "target": target, "task": [source, target],
        "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "config_sha256": str(config_sha256), "candidate_id": str(candidate_id),
        "source_seed": int(source_seed), "stream_seed": int(stream_seed),
        "beginning": float(beginning),
        "strict_online": None if strict_online is None else float(strict_online),
        "post_stream": None if post_stream is None else float(post_stream),
        **metrics, "confusion_matrix": confusion_matrix, "samples": int(samples),
        "batches": int(batches), "passes": int(passes),
        "class_coverage": sum(sum(int(value) for value in row) > 0 for row in confusion_matrix) if confusion_matrix is not None else None,
        "finite_losses": bool(finite_losses),
        "trainable_parameters": sorted(str(name) for name in trainable_parameters),
        "trainable_allowlist": sorted(str(name) for name in trainable_parameters),
        "pre_update_scoring": bool(pre_update_scoring),
        "metadata_evidence_used": bool(metadata_evidence_used),
        "runtime_seconds": float(runtime_seconds), "peak_memory_mb": float(peak_memory_mb),
    }
    record.update(diagnostics)
    return record


def print_hust_result_record(record: dict) -> None:
    print("HUST_RESULT_JSON=" + json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))


class OfflineDiagnosticsAccumulator:
    """Detached target-label diagnostics that cannot feed adaptation."""

    def __init__(self, num_classes: int = 7):
        self.num_classes = int(num_classes)
        self._truth: list[torch.Tensor] = []
        self._pseudo: list[torch.Tensor] = []
        self._certain: list[torch.Tensor] = []

    def update(
        self,
        truth: torch.Tensor,
        pseudo: torch.Tensor,
        certain_mask: torch.Tensor,
    ) -> None:
        self._truth.append(truth.detach().cpu().long().clone())
        self._pseudo.append(pseudo.detach().cpu().long().clone())
        self._certain.append(certain_mask.detach().cpu().bool().clone())
        return None

    def metrics(self) -> dict:
        if not self._truth:
            return {
                "pseudo_purity": 0.0,
                "certain_purity": 0.0,
                "class_coverage": 0,
                "sample_count": 0,
                "confusion_matrix": torch.zeros(
                    self.num_classes, self.num_classes, dtype=torch.long
                ),
            }
        truth = torch.cat(self._truth)
        pseudo = torch.cat(self._pseudo)
        certain = torch.cat(self._certain)
        valid = (
            (truth >= 0)
            & (truth < self.num_classes)
            & (pseudo >= 0)
            & (pseudo < self.num_classes)
        )
        encoded = truth[valid] * self.num_classes + pseudo[valid]
        confusion = torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        certain_valid = certain & valid
        certain_count = int(certain_valid.sum().item())
        return {
            "pseudo_purity": 100.0 * float((truth == pseudo).float().mean().item()),
            "certain_purity": (
                100.0
                * float((truth[certain_valid] == pseudo[certain_valid]).float().mean().item())
                if certain_count
                else 0.0
            ),
            "class_coverage": int(torch.unique(truth[valid]).numel()),
            "sample_count": int(truth.numel()),
            "confusion_matrix": confusion,
        }


def _trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _print_offline_diagnostics(metrics: dict) -> None:
    print(
        "[HUST OFFLINE DIAGNOSTICS] "
        f"pseudo_purity={metrics['pseudo_purity']:.2f} "
        f"certain_purity={metrics['certain_purity']:.2f} "
        f"class_coverage={metrics['class_coverage']}/7 "
        f"sample_count={metrics['sample_count']}"
    )
    print("[HUST OFFLINE DIAGNOSTICS] confusion_matrix_7x7=")
    for row in metrics["confusion_matrix"].tolist():
        print(" ".join(str(int(value)) for value in row))


class HUSTDtCCTrainer(CommonSourcePU4DBase):
    """DtCC with a strictly verified ordinary HUST source checkpoint."""

    def checkpoint_path(self) -> Path:
        return resolve_hust_checkpoint(
            root=Path(
                str(getattr(self.cfg, "hust_checkpoint_root", "TTA_Model_HUST_STRICT_V2"))
            ),
            variant="ordinary",
            source=int(self.cfg.Dataset.TL_Task[0]),
            seed=int(self.cfg.seed_run),
            model_name=self.cfg.model_name,
        )

    def initialize_common_source(self) -> None:
        checkpoint = self.checkpoint_path()
        self.checkpoint_metadata = strict_load_hust_checkpoint(self.model, checkpoint)
        print(
            "[HUST STRICT LOAD] "
            f"route={self.checkpoint_metadata['route']} checkpoint={checkpoint} "
            f"checkpoint_sha256={self.checkpoint_metadata['checkpoint_sha256']}"
        )

    def configure_adaptation(self):
        dcfg = getattr(self.cfg, "DtCC", {})
        bn_params = configure_bn_only(self.model)
        if not bn_params:
            raise RuntimeError("DtCC found no BN affine parameters")
        optimizer = torch.optim.AdamW(
            bn_params,
            lr=float(_get(dcfg, "lr", 1e-2)),
            weight_decay=float(_get(dcfg, "weight_decay", 1e-3)),
        )
        memory = DtCCMemoryBank.from_classifier(self.model[2], self.num_classes)
        return optimizer, memory

    def emit_result(self, **kwargs) -> None:
        """Dataset wrappers may override the serialized result contract."""
        print_hust_result_record(build_hust_result_record(**kwargs))

    def adapt(self):
        task_started = time.perf_counter()
        cfg = self.cfg
        dcfg = getattr(cfg, "DtCC", {})
        self.initialize_common_source()
        before = self.evaluate(self.model)
        print(
            f"Task: {cfg.Dataset.TL_Task}: "
            f"Beginning Acc T = {before['accuracy']:.2f}%;"
        )
        if bool(getattr(cfg, "beginning_only", False)):
            if hasattr(self, "checkpoint_metadata") and hasattr(cfg, "seed_run") and hasattr(self, "stream_seed") and hasattr(self, "stream_loader"):
                self.emit_result(
                result_kind="beginning", route="dtcc_ordinary", variant="ordinary",
                task=cfg.Dataset.TL_Task,
                source_checkpoint_sha256=self.checkpoint_metadata["checkpoint_sha256"],
                config_sha256=str(getattr(cfg, "hust_config_sha256", "unmanaged")),
                candidate_id=str(getattr(cfg, "hust_candidate_id", "beginning-ordinary")),
                source_seed=int(cfg.seed_run), stream_seed=int(self.stream_seed),
                beginning=before["accuracy"], strict_online=None, post_stream=None,
                confusion_matrix=None, samples=len(self.stream_loader.dataset),
                batches=0, passes=0, finite_losses=True, trainable_parameters=[],
                pre_update_scoring=True, metadata_evidence_used=False,
                runtime_seconds=time.perf_counter() - task_started,
                peak_memory_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0,
                )
            return {"before": before}

        # DtCC uses target-batch statistics during strict online adaptation.
        # configure_bn_only() therefore disables BN running statistics.  Keep
        # the source buffers so Post-stream evaluation can restore a valid,
        # deterministic eval-mode state; otherwise the score is artificially
        # low and depends on the evaluation batch partition even with no
        # parameter updates.
        bn_snapshot = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                bn_snapshot.append(
                    (
                        module,
                        bool(module.track_running_stats),
                        None if module.running_mean is None else module.running_mean.detach().clone(),
                        None if module.running_var is None else module.running_var.detach().clone(),
                        None if module.num_batches_tracked is None else module.num_batches_tracked.detach().clone(),
                    )
                )

        optimizer, memory = self.configure_adaptation()
        trainable_names = _trainable_parameter_names(self.model)
        optim_steps = int(_get(dcfg, "optim_steps", 2))
        filter_k = int(_get(dcfg, "filter_k", 50))
        neighbor_k = int(_get(dcfg, "neighbor_k", 5))
        alpha = float(_get(dcfg, "alpha", 2.0))
        ncl_temp = float(_get(dcfg, "ncl_temperature", 0.1))
        log_interval = max(1, int(_get(dcfg, "log_interval", 25)))
        online = PredictionAccumulator(self.num_classes)
        diagnostics = OfflineDiagnosticsAccumulator(self.num_classes)
        finite_losses = True
        batch_times = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        print(
            "[HUST DTCC TARGET] "
            f"stream_seed={self.stream_seed} passes=1 batch={cfg.batch_size} "
            f"optim_steps={optim_steps} BN_only=1"
        )

        for batch_id, (x, y, *_) in enumerate(self.stream_loader, start=1):
            start = time.perf_counter()
            x = x.to(self.device)
            y = y.to(self.device)

            scoring_x = torch.cat((x, x), dim=0) if x.size(0) == 1 else x
            with torch.no_grad():
                _, online_logits = forward_parts(self.model, scoring_x)
                online_logits = online_logits[: x.size(0)]
                online.update(y, online_logits.argmax(dim=1))

            if x.size(0) <= 1:
                probabilities = torch.softmax(online_logits, dim=1)
                certain, _ = dynamic_data_division(
                    probabilities, spectral_entropy(x)
                )
                diagnostics.update(y, probabilities.argmax(dim=1), certain)
                batch_times.append(time.perf_counter() - start)
                continue

            base_snapshot = memory.snapshot()
            values = (0.0, 0.0, 0.0, 0.0, 0.0)
            for optim_step in range(optim_steps):
                optimizer.zero_grad(set_to_none=True)
                features, logits = forward_parts(self.model, x)
                probabilities = torch.softmax(logits, dim=1)
                certain, uncertain = dynamic_data_division(
                    probabilities, spectral_entropy(x)
                )
                pseudo = probabilities.argmax(dim=1)
                if optim_step == 0:
                    diagnostics.update(y, pseudo, certain)
                memory.update(
                    features,
                    probabilities,
                    certain,
                    base_snapshot=base_snapshot,
                )
                prototypes = memory.prototypes()
                loss_sem = dtcc_sem_loss(probabilities, certain, alpha=alpha)
                loss_pcl = dtcc_pcl_loss(
                    features[certain],
                    features[uncertain],
                    prototypes,
                    probabilities[certain].argmax(dim=1),
                    temperature=1.0,
                )
                loss_ncl = dtcc_ncl_loss(
                    features[uncertain],
                    memory.supports.detach(),
                    memory.scores.detach(),
                    neighbor_k=neighbor_k,
                    probs_uncertain=probabilities[uncertain],
                    temperature=ncl_temp,
                )
                loss = loss_sem + loss_pcl + loss_ncl
                finite_losses = finite_losses and all(
                    math.isfinite(float(value.detach().item()))
                    for value in (loss, loss_sem, loss_pcl, loss_ncl)
                )
                loss.backward()
                optimizer.step()
                values = (
                    float(loss.detach().item()),
                    float(loss_sem.detach().item()),
                    float(loss_pcl.detach().item()),
                    float(loss_ncl.detach().item()),
                    float(certain.float().mean().item()),
                )

            memory.slim(filter_k)
            batch_times.append(time.perf_counter() - start)
            if (
                batch_id == 1
                or batch_id % log_interval == 0
                or batch_id == len(self.stream_loader)
            ):
                print(
                    f"iter {batch_id}/{len(self.stream_loader)} | method=DtCC-HUST | "
                    f"loss={values[0]:.6f} sem={values[1]:.6f} "
                    f"pcl={values[2]:.6f} ncl={values[3]:.6f} "
                    f"certain={values[4]:.3f} bank={len(memory)} "
                    f"bank_cls={memory.covered_classes()}/{self.num_classes}"
                )

        online_metrics = online.metrics()
        diagnostic_metrics = diagnostics.metrics()
        _print_offline_diagnostics(diagnostic_metrics)
        print(
            "[HUST OFFLINE PROTOCOL] "
            f"checkpoint_sha256={self.checkpoint_metadata['checkpoint_sha256']} "
            f"checkpoint_route={self.checkpoint_metadata['route']} "
            f"trainable_parameters={','.join(trainable_names)} "
            "passes=1 pre_update_scoring=True "
            f"stream_seed={self.stream_seed} "
            f"class_coverage={diagnostic_metrics['class_coverage']}/7 "
            f"finite_losses={finite_losses} metadata_evidence_used=False"
        )
        print(
            f"Task: {cfg.Dataset.TL_Task}: "
            f"Strict Online Acc = {online_metrics['accuracy']:.2f}%;"
        )
        for module, track, running_mean, running_var, nbt in bn_snapshot:
            module.track_running_stats = track
            module.running_mean = (
                None if running_mean is None else running_mean.to(module.weight.device)
            )
            module.running_var = (
                None if running_var is None else running_var.to(module.weight.device)
            )
            module.num_batches_tracked = (
                None if nbt is None else nbt.to(module.weight.device)
            )
        post = self.evaluate(self.model)
        print(
            f"Task: {cfg.Dataset.TL_Task}: "
            f"Post-stream Full-Target Acc = {post['accuracy']:.2f}%"
        )
        mean_ms = 1000.0 * sum(batch_times) / max(len(batch_times), 1)
        peak_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )
        print(
            f"[DtCC DIAGNOSTICS] mean_batch_ms={mean_ms:.2f} "
            f"peak_memory_mb={peak_mb:.2f}"
        )
        labels = torch.cat(online._labels).long()
        predictions = torch.cat(online._predictions).long()
        encoded = labels * self.num_classes + predictions
        confusion = torch.bincount(encoded, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        certain_count = sum(int(value.sum()) for value in diagnostics._certain)
        if hasattr(cfg, "seed_run"):
            self.emit_result(
            result_kind="target", route="dtcc_ordinary", variant="ordinary",
            task=cfg.Dataset.TL_Task,
            source_checkpoint_sha256=self.checkpoint_metadata["checkpoint_sha256"],
            config_sha256=str(getattr(cfg, "hust_config_sha256", "unmanaged")),
            candidate_id=str(getattr(cfg, "hust_candidate_id", "dtcc_ordinary")),
            source_seed=int(cfg.seed_run), stream_seed=int(self.stream_seed),
            beginning=before["accuracy"], strict_online=online_metrics["accuracy"],
            post_stream=post["accuracy"], confusion_matrix=confusion.tolist(),
            samples=int(labels.numel()), batches=len(batch_times), passes=1,
            finite_losses=finite_losses, trainable_parameters=trainable_names,
            pre_update_scoring=True, metadata_evidence_used=False,
            runtime_seconds=sum(batch_times), peak_memory_mb=peak_mb,
            post_macro_precision=post["macro_precision"], post_macro_recall=post["macro_recall"],
            post_macro_f1=post["macro_f1"], offline_purity=diagnostic_metrics["pseudo_purity"],
            certain_ratio=certain_count / max(int(labels.numel()), 1),
            uncertain_ratio=1.0 - certain_count / max(int(labels.numel()), 1),
            memory_size=len(memory), memory_class_coverage=memory.covered_classes(),
            certain_purity=diagnostic_metrics["certain_purity"],
            evidence_applicable=False, evidence_active_ratio="not_applicable",
            mean_batch_ms=mean_ms,
            )
        return {"before": before, "online": online_metrics, "post": post}


def prepare_hust_config(cfg: omegaconf.DictConfig) -> None:
    """Apply the non-tunable strict-HUST DtCC target constraints in place."""

    with open_dict(cfg):
        cfg.Dataset.data_name = "HUSTStrict"
        apply_hust_protocol_split(cfg)
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
        cfg.Model.model_name = "ResNet18_1D_SDE"
        cfg.Model.use_spectral_adapter = True
        cfg.Model.band_num = 256
        cfg.Model.input_len = 512
        cfg.batch_size = 128
        cfg.num_workers = 4
        if not hasattr(cfg, "stream_seed"):
            cfg.stream_seed = 2025
        if not hasattr(cfg, "DtCC"):
            cfg.DtCC = {}
        # Keep the formal strict defaults, but do not overwrite an explicit
        # tuning value supplied on the command line.  This makes target-side
        # tuning reproducible while preserving the original behavior when no
        # DtCC overrides are given.
        for key, value in {
            "optim_steps": 2,
            "filter_k": 50,
            "neighbor_k": 5,
            "alpha": 2.0,
            "ncl_temperature": 0.1,
        }.items():
            if key not in cfg.DtCC:
                cfg.DtCC[key] = value
        if not hasattr(cfg, "hust_checkpoint_root"):
            cfg.hust_checkpoint_root = "TTA_Model_HUST_STRICT_V2"


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_hust_config(cfg)
    only_task = parse_only_task(getattr(cfg, "only_task", None))
    tasks = (
        [only_task]
        if only_task is not None
        else list(permutations(cfg.Dataset.TL_list, 2))
    )
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id).strip("\"'")
    seeds = parse_seed_runs(getattr(cfg, "seed_runs", None))
    final_cfg = None
    for seed in seeds:
        for task in tasks:
            with open_dict(cfg):
                cfg.seed_run = int(seed)
                cfg.Dataset.TL_Task = task
                cfg.Model.bottleneck_num = 128
                cfg.num_workers = 4
                cfg.model_name = (
                    f"{cfg.Model.model_name}{cfg.seed_run}"
                    f"{cfg.Dataset.input_kind}_Linear.pt"
                )
                final_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            print("=" * 80)
            print(f"DtCC-HUST Strict | task={task} seed={seed}")
            print("=" * 80)
            if cfg.process_wandb:
                run_obj = wandb.init(config=final_cfg, **cfg.wandb.setup)
                with run_obj:
                    trainer = HUSTDtCCTrainer(cfg, run_obj)
                    trainer.setup()
                    trainer.adapt()
            else:
                trainer = HUSTDtCCTrainer(cfg, None)
                trainer.setup()
                trainer.adapt()
    pprint(final_cfg)


if __name__ == "__main__":
    run()
