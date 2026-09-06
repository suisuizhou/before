#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DtCC on PU4D using the vanilla common-source ResNet18 checkpoint."""

from __future__ import annotations

from itertools import permutations
from pprint import pprint
import os
import time
from pathlib import Path

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
from Lib.pu4d_common_source import (
    CommonSourcePU4DBase,
    PredictionAccumulator,
    _get,
    forward_parts,
    parse_only_task,
    parse_seed_runs,
)
from Lib.pu4d_vanilla_protocol import resolve_vanilla_checkpoint


class DtCCCommonSourceTrainer(CommonSourcePU4DBase):
    def checkpoint_path(self):
        root = Path(str(getattr(self.cfg, "vanilla_source_root", "TTA_Model_VANILLA")))
        return resolve_vanilla_checkpoint(
            root,
            int(self.cfg.Dataset.TL_Task[0]),
            int(self.cfg.seed_run),
            self.cfg.model_name,
        )

    def adapt(self):
        cfg = self.cfg
        dcfg = getattr(cfg, "DtCC", {})
        self.initialize_common_source()
        before = self.evaluate(self.model)
        print(f"Task: {cfg.Dataset.TL_Task}: Beginning Acc T = {before['accuracy']:.2f}%;")

        bn_params = configure_bn_only(self.model)
        if not bn_params:
            raise RuntimeError("DtCC found no BN affine parameters")
        optimizer = torch.optim.AdamW(
            bn_params,
            lr=float(_get(dcfg, "lr", 1e-2)),
            weight_decay=float(_get(dcfg, "weight_decay", 1e-3)),
        )
        memory = DtCCMemoryBank.from_classifier(self.model[2], self.num_classes)
        optim_steps = int(_get(dcfg, "optim_steps", 2))
        filter_k = int(_get(dcfg, "filter_k", 50))
        neighbor_k = int(_get(dcfg, "neighbor_k", 5))
        alpha = float(_get(dcfg, "alpha", 2.0))
        ncl_temp = float(_get(dcfg, "ncl_temperature", 1.0))
        log_interval = max(1, int(_get(dcfg, "log_interval", 25)))
        online = PredictionAccumulator(self.num_classes)
        batch_times = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        print(
            f"[DtCC-R18 VanillaSource] stream_seed={self.stream_seed} passes=1 "
            f"batch={cfg.batch_size} optim_steps={optim_steps} BN_only=1"
        )

        for batch_id, (x, y, _) in enumerate(self.stream_loader, start=1):
            if x.size(0) <= 1:
                continue
            start = time.perf_counter()
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                _, online_logits = forward_parts(self.model, x)
                online.update(y, online_logits.argmax(dim=1))

            base_snapshot = memory.snapshot()
            loss_value = 0.0
            certain_ratio = 0.0
            loss_sem_value = 0.0
            loss_pcl_value = 0.0
            loss_ncl_value = 0.0

            for _ in range(optim_steps):
                optimizer.zero_grad(set_to_none=True)
                features, logits = forward_parts(self.model, x)
                probabilities = torch.softmax(logits, dim=1)
                certain, uncertain = dynamic_data_division(
                    probabilities, spectral_entropy(x)
                )
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
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().item())
                certain_ratio = float(certain.float().mean().item())
                loss_sem_value = float(loss_sem.detach().item())
                loss_pcl_value = float(loss_pcl.detach().item())
                loss_ncl_value = float(loss_ncl.detach().item())

            memory.slim(filter_k)
            batch_times.append(time.perf_counter() - start)
            if batch_id == 1 or batch_id % log_interval == 0 or batch_id == len(self.stream_loader):
                print(
                    f"iter {batch_id}/{len(self.stream_loader)} | method=DtCC | "
                    f"loss={loss_value:.6f} sem={loss_sem_value:.6f} "
                    f"pcl={loss_pcl_value:.6f} ncl={loss_ncl_value:.6f} "
                    f"certain={certain_ratio:.3f} bank={len(memory)} "
                    f"bank_cls={memory.covered_classes()}/{self.num_classes}"
                )

        online_metrics = online.metrics()
        print(f"Task: {cfg.Dataset.TL_Task}: Strict Online Acc = {online_metrics['accuracy']:.2f}%;")
        print(
            f"Task: {cfg.Dataset.TL_Task}: Strict Online Macro P/R/F1 = "
            f"{online_metrics['macro_precision']:.2f}/"
            f"{online_metrics['macro_recall']:.2f}/"
            f"{online_metrics['macro_f1']:.2f}%;"
        )
        post = self.evaluate(self.model)
        print(f"Task: {cfg.Dataset.TL_Task}: Post-stream Full-Target Acc = {post['accuracy']:.2f}%")
        print(
            f"Task: {cfg.Dataset.TL_Task}: Post-stream Macro P/R/F1 = "
            f"{post['macro_precision']:.2f}/{post['macro_recall']:.2f}/{post['macro_f1']:.2f}%;"
        )
        mean_ms = 1000.0 * sum(batch_times) / max(len(batch_times), 1)
        peak_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda" else 0.0
        )
        print(f"[DtCC DIAGNOSTICS] mean_batch_ms={mean_ms:.2f} peak_memory_mb={peak_mb:.2f}")
        return {"before": before, "online": online_metrics, "post": post}


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    only_task = parse_only_task(getattr(cfg, "only_task", None))
    tasks = [only_task] if only_task is not None else list(permutations(cfg.Dataset.TL_list, 2))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id).strip("\"'")
    seeds = parse_seed_runs(getattr(cfg, "seed_runs", None))
    final_cfg = None
    for seed in seeds:
        for task in tasks:
            with open_dict(cfg):
                cfg.seed_run = int(seed)
                cfg.Dataset.TL_Task = task
                cfg.Dataset.input_kind = "fft"
                cfg.Model.bottleneck_num = 128
                cfg.batch_size = int(getattr(cfg, "batch_size", 128))
                cfg.num_workers = int(getattr(cfg, "num_workers", 4))
                cfg.Opt.lr_src = float(getattr(cfg.Opt, "lr_src", 1e-3))
                cfg.stream_seed = int(getattr(cfg, "stream_seed", 2025))
                if not hasattr(cfg, "DtCC"):
                    cfg.DtCC = {}
                final_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            print("=" * 80)
            print(f"DtCC-R18 VanillaSource | task={task} seed={seed}")
            print("=" * 80)
            if cfg.process_wandb:
                run_obj = wandb.init(config=final_cfg, **cfg.wandb.setup)
                with run_obj:
                    trainer = DtCCCommonSourceTrainer(cfg, run_obj)
                    trainer.setup()
                    trainer.adapt()
            else:
                trainer = DtCCCommonSourceTrainer(cfg, None)
                trainer.setup()
                trainer.adapt()
    pprint(final_cfg)


if __name__ == "__main__":
    run()
