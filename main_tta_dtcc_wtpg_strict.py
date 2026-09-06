#!/usr/bin/env python3
"""Strict single-pass DtCC for WTPG speed-domain tasks."""

from itertools import permutations
import os
from pathlib import Path
from pprint import pprint

import hydra
import omegaconf
from omegaconf import open_dict

from Lib.pu4d_common_source import parse_only_task, parse_seed_runs
from Lib.wtpg_results import build_wtpg_result_record, print_wtpg_result
from Lib.wtpg_strict_protocol import SPEEDS, resolve_checkpoint, strict_load_checkpoint
from main_tta_dtcc_hust_strict import HUSTDtCCTrainer


class WTPGDtCCTrainer(HUSTDtCCTrainer):
    def checkpoint_path(self):
        return resolve_checkpoint(
            Path(str(self.cfg.wtpg_checkpoint_root)), "ordinary",
            int(self.cfg.Dataset.TL_Task[0]), int(self.cfg.seed_run), self.cfg.model_name,
        )

    def initialize_common_source(self):
        checkpoint = self.checkpoint_path()
        self.checkpoint_metadata = strict_load_checkpoint(self.model, checkpoint)
        print(f"[WTPG STRICT LOAD] route=ordinary checkpoint={checkpoint} checkpoint_sha256={self.checkpoint_metadata['checkpoint_sha256']}")

    def emit_result(self, **kwargs):
        print_wtpg_result(build_wtpg_result_record(**kwargs))


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
        if not hasattr(cfg, "stream_seed"):
            cfg.stream_seed = 2025
        if not hasattr(cfg, "DtCC"):
            cfg.DtCC = {}
        cfg.DtCC.optim_steps = 2
        cfg.DtCC.filter_k = 50
        cfg.DtCC.neighbor_k = 5
        cfg.DtCC.alpha = 2.0
        cfg.DtCC.ncl_temperature = 0.1
        if not hasattr(cfg, "wtpg_checkpoint_root"):
            cfg.wtpg_checkpoint_root = "TTA_Model_WTPG_STRICT_V1"
        if not hasattr(cfg, "wtpg_config_sha256"):
            cfg.wtpg_config_sha256 = "unmanaged"
        if not hasattr(cfg, "wtpg_candidate_id"):
            cfg.wtpg_candidate_id = "dtcc_baseline"
        # Parent serialization names are accepted by the WTPG result contract.
        cfg.hust_config_sha256 = cfg.wtpg_config_sha256
        cfg.hust_candidate_id = cfg.wtpg_candidate_id


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_wtpg_config(cfg)
    only_task = parse_only_task(getattr(cfg, "only_task", None))
    tasks = [only_task] if only_task is not None else list(permutations(cfg.Dataset.TL_list, 2))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id).strip("\"'")
    final_cfg = None
    for seed in parse_seed_runs(getattr(cfg, "seed_runs", None)):
        for task in tasks:
            with open_dict(cfg):
                cfg.seed_run = int(seed)
                cfg.Dataset.TL_Task = task
                cfg.Model.bottleneck_num = 128
                cfg.model_name = f"ResNet18_1D_SDE{cfg.seed_run}fft_Linear.pt"
                final_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            trainer = WTPGDtCCTrainer(cfg, None)
            trainer.setup()
            trainer.adapt()
    pprint(final_cfg)


if __name__ == "__main__":
    run()

