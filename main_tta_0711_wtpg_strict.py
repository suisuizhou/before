#!/usr/bin/env python3
"""0711-Full strict single-pass adaptation for WTPG speed domains."""

from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import open_dict

from Lib.physical_fault_evidence import PhysicalEvidenceConfig
from Lib.wtpg_results import build_wtpg_result_record, print_wtpg_result
from Lib.wtpg_saliency_evidence import contiguous_saliency_masks
from Lib.wtpg_strict_protocol import SPEEDS, resolve_checkpoint, strict_load_checkpoint
import main_tta_0711_strict_randomstream as base
from main_tta_0711_hust_strict import HUST0711Trainer


class WTPG0711Trainer(HUST0711Trainer):
    def _checkpoint_path(self):
        variant = str(getattr(self.cfg, "source_variant", "robust"))
        return resolve_checkpoint(
            Path(str(self.cfg.wtpg_checkpoint_root)), variant,
            int(self.cfg.Dataset.TL_Task[0]), int(self.cfg.seed_run), self.cfg.model_name,
        )

    def load_source_checkpoint(self, model, checkpoint_path):
        self.checkpoint_metadata = strict_load_checkpoint(model, Path(checkpoint_path))
        print(f"[WTPG STRICT LOAD] route={self.checkpoint_metadata['route']} checkpoint={checkpoint_path} checkpoint_sha256={self.checkpoint_metadata['checkpoint_sha256']}")

    def _build_physical_config(self):
        return PhysicalEvidenceConfig(
            sampling_rate_hz=48_000.0, fft_size=2_048, spectrum_length=512,
            harmonics=8, outer_sideband_orders=(0, 1), inner_sideband_orders=(0, 1, 2),
            mask_sigma_bins=1.0,
            background_width_bins=int(base._get(self.tcfg, "physical_background_width", 7)),
            max_mask_ratio=float(base._get(self.tcfg, "max_mask_ratio", 0.18)),
            mask_activity_threshold=float(base._get(self.tcfg, "mask_activity_threshold", 0.10)),
            exclude_dc=True,
        )

    def evidence_batch_metadata(self, sample_indices):
        indices = torch.as_tensor(sample_indices).detach().cpu().long()
        return {"speed_hz": self.datasets["target_data"].speed_hz[indices].to(self.device)}

    @torch.no_grad()
    def teacher_reliability(self, x, iter_num, batch_metadata=None):
        self._wtpg_evidence_input = x.detach()
        return super().teacher_reliability(x, iter_num, batch_metadata)

    def build_evidence_masks(self, pseudo_labels, batch_metadata=None):
        if not hasattr(self, "_wtpg_evidence_input"):
            raise RuntimeError("WTPG evidence input is unavailable")
        with torch.enable_grad():
            evidence_input = self._wtpg_evidence_input.detach().clone().requires_grad_(True)
            with self.adaptation_ema.applied_to(self.student):
                self.student.eval()
                _, logits = self.forward_parts(self.student, evidence_input)
                selected = logits.gather(1, pseudo_labels.view(-1, 1)).sum()
                gradient = torch.autograd.grad(selected, evidence_input, only_inputs=True)[0]
        saliency = (gradient.abs() * evidence_input.detach().abs()).detach()
        return contiguous_saliency_masks(
            saliency, pseudo_labels,
            bands=int(base._get(self.tcfg, "saliency_bands", 8)),
            half_width=int(base._get(self.tcfg, "saliency_half_width", 3)),
            max_mask_ratio=float(base._get(self.tcfg, "max_mask_ratio", 0.18)),
        )

    def emit_result(self, **kwargs):
        print_wtpg_result(build_wtpg_result_record(**kwargs))

    def offline_diagnostics_finalize(self):
        offline = self._offline_diagnostics.metrics()
        online = self._online_diagnostics.metrics()
        self._final_offline_metrics = offline
        self._final_online_metrics = online
        print(
            "[WTPG OFFLINE DIAGNOSTICS] "
            f"pseudo_purity={offline['pseudo_purity']:.2f} "
            f"certain_purity={offline['certain_purity']:.2f} "
            f"class_coverage={offline['class_coverage']}/5 "
            f"sample_count={offline['sample_count']}"
        )
        print("[WTPG OFFLINE DIAGNOSTICS] confusion_matrix_5x5=")
        for row in offline["confusion_matrix"].tolist():
            print(" ".join(str(int(value)) for value in row))
        print("[WTPG STRICT ONLINE DIAGNOSTICS] confusion_matrix_5x5=")
        for row in online["confusion_matrix"].tolist():
            print(" ".join(str(int(value)) for value in row))
        print(
            "[WTPG OFFLINE PROTOCOL] "
            f"checkpoint_sha256={self.checkpoint_metadata['checkpoint_sha256']} "
            f"checkpoint_route={self.checkpoint_metadata['route']} "
            "passes=1 pre_update_scoring=True "
            f"stream_seed={self.stream_seed} class_coverage={online['class_coverage']}/5 "
            f"finite_losses={self._offline_finite_losses} "
            f"metadata_evidence_used={self._offline_metadata_evidence_used}"
        )


def prepare_wtpg_config(cfg):
    with open_dict(cfg):
        cfg.Dataset.data_name = "WTPGStrict"
        cfg.Dataset.data_path = "Dataset/WTPG_STRICT_CACHE_V1"
        cfg.Dataset.TL_list = list(SPEEDS)
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "pre_normalized"
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
        cfg.TTA0711.sampling_rate_hz = 48_000
        cfg.TTA0711.fft_size = 2_048
        cfg.TTA0711.spectrum_length = 512
        cfg.TTA0711.min_pcl_classes = 2
        cfg.TTA0711.min_ncl_classes = 4
        cfg.TTA0711.min_ncl_entries = 20
        if not hasattr(cfg, "wtpg_checkpoint_root"):
            cfg.wtpg_checkpoint_root = "TTA_Model_WTPG_STRICT_V1"
        if not hasattr(cfg, "source_variant"):
            cfg.source_variant = "robust"
        if not hasattr(cfg, "wtpg_config_sha256"):
            cfg.wtpg_config_sha256 = "unmanaged"
        if not hasattr(cfg, "wtpg_candidate_id"):
            cfg.wtpg_candidate_id = "0711_baseline"
        cfg.hust_config_sha256 = cfg.wtpg_config_sha256
        cfg.hust_candidate_id = cfg.wtpg_candidate_id


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_wtpg_config(cfg)
    original = base.Strict0711ResNetTrainer
    base.Strict0711ResNetTrainer = WTPG0711Trainer
    try:
        return base.run.__wrapped__(cfg)
    finally:
        base.Strict0711ResNetTrainer = original


if __name__ == "__main__":
    run()
