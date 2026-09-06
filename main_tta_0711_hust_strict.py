#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict single-pass 0711 target adaptation for the HUST protocol."""

from __future__ import annotations

from pathlib import Path
import time

import hydra
import omegaconf
import torch
from omegaconf import open_dict

import main_tta_0711_strict_randomstream as base
from Lib.hust_physical_fault_evidence import (
    HUSTPhysicalEvidenceConfig,
    build_hust_physical_masks,
)
from Lib.hust_strict_protocol import (
    apply_hust_protocol_split,
    resolve_hust_checkpoint,
    strict_load_hust_checkpoint,
)
from Lib.pu4d_common_source import classification_metrics
from main_tta_dtcc_hust_strict import (
    OfflineDiagnosticsAccumulator,
    _print_offline_diagnostics,
    _trainable_parameter_names,
    build_hust_result_record,
    print_hust_result_record,
)


class HUST0711Trainer(base.Strict0711ResNetTrainer):
    def _checkpoint_path(self):
        source_variant = str(getattr(self.cfg, "source_variant", "robust"))
        if source_variant not in {"ordinary", "robust"}:
            raise ValueError(f"invalid source_variant: {source_variant}")
        return resolve_hust_checkpoint(
            root=Path(
                str(
                    getattr(
                        self.cfg,
                        "hust_checkpoint_root",
                        "TTA_Model_HUST_STRICT_V2",
                    )
                )
            ),
            variant=source_variant,
            source=int(self.cfg.Dataset.TL_Task[0]),
            seed=int(self.cfg.seed_run),
            model_name=self.cfg.model_name,
        )

    def load_source_checkpoint(self, model, checkpoint_path):
        self.checkpoint_metadata = strict_load_hust_checkpoint(
            model, Path(checkpoint_path)
        )
        print(
            "[HUST STRICT LOAD] "
            f"route={self.checkpoint_metadata['route']} "
            f"checkpoint={checkpoint_path} "
            f"checkpoint_sha256={self.checkpoint_metadata['checkpoint_sha256']}"
        )

    def _build_physical_config(self):
        return HUSTPhysicalEvidenceConfig(
            sampling_rate_hz=51200.0,
            fft_size=2048,
            spectrum_length=512,
            harmonics=int(base._get(self.tcfg, "physical_harmonics", 8)),
            outer_sideband_orders=tuple(
                int(value)
                for value in base._get(
                    self.tcfg, "outer_sideband_orders", [0, 1]
                )
            ),
            inner_sideband_orders=tuple(
                int(value)
                for value in base._get(
                    self.tcfg, "inner_sideband_orders", [0, 1, 2]
                )
            ),
            ball_sideband_orders=tuple(
                int(value)
                for value in base._get(
                    self.tcfg, "ball_sideband_orders", [0, 1, 2]
                )
            ),
            mask_sigma_bins=float(base._get(self.tcfg, "mask_sigma_bins", 1.0)),
            background_width_bins=int(
                base._get(self.tcfg, "physical_background_width", 7)
            ),
            max_mask_ratio=float(base._get(self.tcfg, "max_mask_ratio", 0.18)),
            mask_activity_threshold=float(
                base._get(self.tcfg, "mask_activity_threshold", 0.10)
            ),
            exclude_dc=True,
        )

    def evidence_batch_metadata(self, sample_indices):
        indices = torch.as_tensor(sample_indices).detach().cpu().long()
        dataset = self.datasets["target_data"]
        return {"shaft_hz": dataset.shaft_hz[indices].to(self.device)}

    def build_evidence_masks(self, pseudo_labels, batch_metadata=None):
        if not batch_metadata or "shaft_hz" not in batch_metadata:
            raise RuntimeError("HUST shaft metadata missing")
        return build_hust_physical_masks(
            pseudo_labels,
            self.target_domain,
            batch_metadata["shaft_hz"],
            self.physical_config,
        )

    def offline_diagnostics_update(self, labels, reliability_info):
        if not hasattr(self, "_offline_diagnostics"):
            self._offline_diagnostics = OfflineDiagnosticsAccumulator(
                self.num_classes
            )
            self._offline_metadata_evidence_used = False
        self._offline_metadata_evidence_used = (
            self._offline_metadata_evidence_used
            or "evidence_applicable" in reliability_info
        )
        self._offline_diagnostics.update(
            labels,
            reliability_info["pseudo"],
            reliability_info["certain"],
        )
        if not hasattr(self, "_routing_samples"):
            self._routing_samples = self._certain_samples = self._evidence_active_samples = 0
        self._routing_samples += int(labels.numel())
        self._certain_samples += int(reliability_info["certain"].sum())
        self._evidence_active_samples += int(bool(reliability_info.get("evidence_active", False))) * int(labels.numel())
        return None

    def strict_online_diagnostics_update(self, labels, predictions):
        if not hasattr(self, "_online_diagnostics"):
            self._online_diagnostics = OfflineDiagnosticsAccumulator(
                self.num_classes
            )
        self._online_diagnostics.update(
            labels,
            predictions,
            torch.ones_like(labels, dtype=torch.bool),
        )
        return None

    def offline_diagnostics_finalize(self):
        if not hasattr(self, "_offline_diagnostics"):
            self._offline_diagnostics = OfflineDiagnosticsAccumulator(
                self.num_classes
            )
            self._offline_metadata_evidence_used = False
        if not hasattr(self, "_online_diagnostics"):
            self._online_diagnostics = OfflineDiagnosticsAccumulator(
                self.num_classes
            )
        offline_metrics = self._offline_diagnostics.metrics()
        online_metrics = self._online_diagnostics.metrics()
        self._final_offline_metrics = offline_metrics
        self._final_online_metrics = online_metrics
        _print_offline_diagnostics(offline_metrics)
        print("[HUST STRICT ONLINE DIAGNOSTICS] confusion_matrix_7x7=")
        for row in online_metrics["confusion_matrix"].tolist():
            print(" ".join(str(int(value)) for value in row))
        print(
            "[HUST OFFLINE PROTOCOL] "
            f"checkpoint_sha256={self.checkpoint_metadata['checkpoint_sha256']} "
            f"checkpoint_route={self.checkpoint_metadata['route']} "
            f"trainable_parameters={','.join(_trainable_parameter_names(self.student))} "
            "passes=1 pre_update_scoring=True "
            f"stream_seed={self.stream_seed} "
            f"class_coverage={online_metrics['class_coverage']}/7 "
            f"finite_losses={self._offline_finite_losses} "
            f"losses_seen={self._offline_losses_seen} "
            f"metadata_evidence_used={self._offline_metadata_evidence_used}"
        )
        return None

    def post_stream_diagnostics(self, evaluation):
        if len(evaluation) < 5:
            return {}
        metrics = classification_metrics(
            evaluation[3], evaluation[4], self.num_classes
        )
        return {
            "post_macro_precision": metrics["macro_precision"],
            "post_macro_recall": metrics["macro_recall"],
            "post_macro_f1": metrics["macro_f1"],
        }

    def offline_loss_diagnostics_update(self, **losses):
        detached_losses = tuple(loss.detach() for loss in losses.values())
        self._offline_losses_seen += len(detached_losses)
        self._offline_finite_losses = (
            self._offline_finite_losses
            and all(bool(torch.isfinite(loss).all()) for loss in detached_losses)
        )
        return None

    def adapt(self):
        task_started = time.perf_counter()
        if bool(getattr(self.cfg, "beginning_only", False)):
            self.initialize_models()
            before_acc = base.cal_acc(
                self.dataloaders["target_data"], self.student
            )[0]
            print(
                f"Task: {self.cfg.Dataset.TL_Task}: "
                f"Beginning Acc T = {before_acc:.2f}%;"
            )
            variant = str(getattr(self.cfg, "source_variant", "robust"))
            route = "0711_robust" if variant == "robust" else "0711_common"
            if hasattr(self, "checkpoint_metadata") and hasattr(self.cfg, "seed_run") and hasattr(self, "stream_seed") and hasattr(self, "target_dataloader"):
                print_hust_result_record(build_hust_result_record(
                result_kind="beginning", route=route, variant=variant,
                task=self.cfg.Dataset.TL_Task,
                source_checkpoint_sha256=self.checkpoint_metadata["checkpoint_sha256"],
                config_sha256=str(getattr(self.cfg, "hust_config_sha256", "unmanaged")),
                candidate_id=str(getattr(self.cfg, "hust_candidate_id", f"beginning-{variant}")),
                source_seed=int(self.cfg.seed_run), stream_seed=int(self.stream_seed),
                beginning=before_acc, strict_online=None, post_stream=None,
                confusion_matrix=None, samples=len(self.target_dataloader.dataset),
                batches=0, passes=0, finite_losses=True, trainable_parameters=[],
                pre_update_scoring=True, metadata_evidence_used=False,
                runtime_seconds=time.perf_counter() - task_started,
                peak_memory_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0,
                ))
            return {"before": before_acc}
        self._offline_finite_losses = True
        self._offline_losses_seen = 0
        self._routing_samples = 0
        self._certain_samples = 0
        self._evidence_active_samples = 0
        result = super().adapt()
        offline_metrics = self._final_offline_metrics
        online_metrics = self._final_online_metrics
        variant = str(getattr(self.cfg, "source_variant", "robust"))
        route = "0711_robust" if variant == "robust" else "0711_common"
        elapsed = time.perf_counter() - task_started
        batches = len(self.target_dataloader)
        peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
        if hasattr(self.cfg, "seed_run"):
            print_hust_result_record(build_hust_result_record(
            result_kind="target", route=route, variant=variant,
            task=self.cfg.Dataset.TL_Task,
            source_checkpoint_sha256=self.checkpoint_metadata["checkpoint_sha256"],
            config_sha256=str(getattr(self.cfg, "hust_config_sha256", "unmanaged")),
            candidate_id=str(getattr(self.cfg, "hust_candidate_id", variant)),
            source_seed=int(self.cfg.seed_run), stream_seed=int(self.stream_seed),
            beginning=result["before"], strict_online=result["online"], post_stream=result["post_stream"],
            confusion_matrix=online_metrics["confusion_matrix"].tolist(), samples=online_metrics["sample_count"],
            batches=batches, passes=1, finite_losses=self._offline_finite_losses,
            trainable_parameters=_trainable_parameter_names(self.student),
            pre_update_scoring=True, metadata_evidence_used=self._offline_metadata_evidence_used,
            runtime_seconds=elapsed, peak_memory_mb=peak_mb,
            post_macro_precision=result["post_macro_precision"],
            post_macro_recall=result["post_macro_recall"],
            post_macro_f1=result["post_macro_f1"],
            offline_confusion_matrix=offline_metrics["confusion_matrix"].tolist(),
            offline_purity=offline_metrics["pseudo_purity"],
            certain_purity=offline_metrics["certain_purity"],
            certain_ratio=self._certain_samples / max(self._routing_samples, 1),
            uncertain_ratio=1.0 - self._certain_samples / max(self._routing_samples, 1),
            evidence_applicable=True,
            evidence_active_ratio=self._evidence_active_samples / max(self._routing_samples, 1),
            memory_size=len(self.memory), memory_class_coverage=self.memory.covered_classes(),
            mean_batch_ms=1000.0 * elapsed / max(batches, 1),
            ))
        return result


def prepare_hust_config(cfg: omegaconf.DictConfig) -> None:
    """Apply the non-tunable HUST strict-protocol constraints in place."""

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
        if not hasattr(cfg, "TTA0711"):
            cfg.TTA0711 = {}
        cfg.TTA0711.passes = 1
        if not hasattr(cfg.TTA0711, "stream_seed"):
            cfg.TTA0711.stream_seed = 2025
        cfg.TTA0711.min_pcl_classes = 3
        cfg.TTA0711.min_ncl_classes = 5
        cfg.TTA0711.min_ncl_entries = 20
        if not hasattr(cfg, "hust_checkpoint_root"):
            cfg.hust_checkpoint_root = "TTA_Model_HUST_STRICT_V2"
        if not hasattr(cfg, "source_variant"):
            cfg.source_variant = "robust"


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_hust_config(cfg)
    original_cls = base.Strict0711ResNetTrainer
    base.Strict0711ResNetTrainer = HUST0711Trainer
    try:
        return base.run.__wrapped__(cfg)
    finally:
        base.Strict0711ResNetTrainer = original_cls


if __name__ == "__main__":
    run()
