#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pathlib import Path
from pprint import pprint
from itertools import permutations
import os
import time
import logging
import ast
import random

import hydra
import torch
import torch.nn.functional as F
import omegaconf
from omegaconf import open_dict
from torch.utils.data import DataLoader

import Dataset
from Lib.model import get_model
from Lib.train_utils import cal_acc, AverageMeter, seed_torch


def build_model_name(cfg):
    if cfg.Model.model_type == 'linear':
        return f"{cfg.Model.model_name}{cfg.seed_run}{cfg.Dataset.input_kind}_Linear.pt"
    elif cfg.Model.model_type == 'wn':
        return f"{cfg.Model.model_name}{cfg.seed_run}{cfg.Dataset.input_kind}_WN.pt"
    else:
        return f"{cfg.Model.model_name}{cfg.seed_run}{cfg.Dataset.input_kind}_Pro.pt"


def smooth_one_hot(targets, num_classes, eps=0.1):
    with torch.no_grad():
        true_dist = torch.zeros(targets.size(0), num_classes, device=targets.device)
        true_dist.fill_(eps / num_classes)
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - eps + eps / num_classes)
    return true_dist


def soft_target_ce(logits, soft_targets):
    log_prob = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()


def label_smoothing_ce(logits, targets, num_classes, eps=0.1):
    soft_targets = smooth_one_hot(targets, num_classes, eps)
    return soft_target_ce(logits, soft_targets)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample().item()
    index = torch.randperm(x.size(0), device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_label_smoothing_ce(logits, y_a, y_b, lam, num_classes, eps=0.1):
    soft_a = smooth_one_hot(y_a, num_classes, eps)
    soft_b = smooth_one_hot(y_b, num_classes, eps)
    soft_targets = lam * soft_a + (1 - lam) * soft_b
    return soft_target_ce(logits, soft_targets)


def _to_channel_first_1d(x):
    squeezed = False
    transposed = False

    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeezed = True
    elif x.dim() == 3 and x.shape[-1] == 1:
        x = x.transpose(1, 2)
        transposed = True
    elif x.dim() != 3:
        raise ValueError(f"Unexpected x shape: {x.shape}")

    return x, squeezed, transposed


def _restore_1d_shape(x, squeezed, transposed):
    if transposed:
        x = x.transpose(1, 2)
    if squeezed:
        x = x.squeeze(1)
    return x


def spectral_style_augment(x, strength=0.15, knots=8, prob=0.7):
    if random.random() > prob:
        return x

    x_cf, squeezed, transposed = _to_channel_first_1d(x)
    B, C, L = x_cf.shape
    device = x_cf.device

    knots = max(2, int(knots))
    ctrl = torch.empty(B, 1, knots, device=device).uniform_(-strength, strength)

    # 低维光滑谱形：全局斜率 + 控制点插值
    pos = torch.linspace(-1.0, 1.0, steps=knots, device=device).view(1, 1, knots)
    tilt = torch.empty(B, 1, 1, device=device).uniform_(-0.5 * strength, 0.5 * strength)
    ctrl = ctrl + tilt * pos

    mask = F.interpolate(ctrl, size=L, mode="linear", align_corners=True)
    mask = torch.exp(mask)

    x_aug = x_cf * mask
    x_aug = _restore_1d_shape(x_aug, squeezed, transposed)
    return x_aug


def symmetric_kl(logits_a, logits_b, T=1.0):
    log_pa = F.log_softmax(logits_a / T, dim=1)
    log_pb = F.log_softmax(logits_b / T, dim=1)
    pa = log_pa.exp()
    pb = log_pb.exp()

    loss = 0.5 * (
        F.kl_div(log_pa, pb, reduction="batchmean") +
        F.kl_div(log_pb, pa, reduction="batchmean")
    ) * (T * T)
    return loss


def _normalize_task_node(node):
    if isinstance(node, (list, tuple)):
        return tuple(_normalize_task_node(x) for x in node)
    return int(node)


def parse_only_task(s):
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        obj = s
    else:
        obj = ast.literal_eval(str(s))

    if not isinstance(obj, (list, tuple)) or len(obj) != 2:
        raise ValueError(f"only_task must be like [src, tar], got: {obj}")

    src = _normalize_task_node(obj[0])
    tar = _normalize_task_node(obj[1])
    return (src, tar)


class SourceTrainer:
    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        self.device = None
        self.device_count = 1
        self.dataset_cls = None
        self.num_classes = None
        self.datasets = {}
        self.dataloaders = {}
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def setup(self):
        cfg = self.cfg
        seed_torch(cfg.seed_run)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info(f'using {self.device_count} gpus')
            assert cfg.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            import warnings
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info(f'using {self.device_count} cpu')

        dataset_cls = getattr(Dataset, cfg.Dataset.data_name)
        self.dataset_cls = dataset_cls
        self.num_classes = dataset_cls.num_classes

        self.datasets['source_data'], self.datasets['target_data'] = dataset_cls(**cfg.Dataset).data_generator()

        self.dataloaders['source_train'] = DataLoader(
            self.datasets['source_data'],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=True,
        )

        self.dataloaders['source_eval'] = DataLoader(
            self.datasets['source_data'],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False,
        )

        self.dataloaders['target_eval'] = DataLoader(
            self.datasets['target_data'],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False,
        )

        with open_dict(cfg):
            if cfg.Model.model_name == "ViT1D":
                cfg.Model.prompt_len = int(getattr(cfg, "prompt_len_src", getattr(cfg.Model, "prompt_len", 0)))

        self.model = get_model(num_classes=self.num_classes, cfg=cfg, **cfg.Model).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.Opt.lr_src,
            weight_decay=cfg.Opt.weight_decay_src
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.src_epoch,
            eta_min=cfg.Opt.lr_src * 0.01
        )

    def train_one_epoch(self, epoch):
        cfg = self.cfg
        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        style_meter = AverageMeter()
        cons_meter = AverageMeter()

        use_ssp_lite = bool(getattr(cfg, "use_ssp_lite", False))
        ssp_style_prob = float(getattr(cfg, "ssp_style_prob", 0.7))
        ssp_style_strength = float(getattr(cfg, "ssp_style_strength", 0.15))
        ssp_style_knots = int(getattr(cfg, "ssp_style_knots", 8))
        ssp_lambda_style = float(getattr(cfg, "ssp_lambda_style", 0.5))
        ssp_lambda_cons = float(getattr(cfg, "ssp_lambda_cons", 0.05))

        for x, y, _ in self.dataloaders['source_train']:
            if x.size(0) <= 1:
                continue

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            do_mixup = (cfg.mixup_alpha > 0) and (random.random() < cfg.mixup_prob)

            zero = x.new_tensor(0.0)

            if do_mixup:
                x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=cfg.mixup_alpha)
                logits = self.model(x_mix)
                loss_cls = mixup_label_smoothing_ce(
                    logits, y_a, y_b, lam,
                    num_classes=self.num_classes,
                    eps=cfg.label_smoothing
                )

                if use_ssp_lite:
                    x_style = spectral_style_augment(
                        x_mix,
                        strength=ssp_style_strength,
                        knots=ssp_style_knots,
                        prob=ssp_style_prob
                    )
                    logits_style = self.model(x_style)
                    loss_style = mixup_label_smoothing_ce(
                        logits_style, y_a, y_b, lam,
                        num_classes=self.num_classes,
                        eps=cfg.label_smoothing
                    )
                    loss_cons = symmetric_kl(logits, logits_style)
                    loss = loss_cls + ssp_lambda_style * loss_style + ssp_lambda_cons * loss_cons
                else:
                    loss_style = zero
                    loss_cons = zero
                    loss = loss_cls

                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    acc = (
                        lam * (pred == y_a).float() +
                        (1 - lam) * (pred == y_b).float()
                    ).mean().item() * 100.0
            else:
                logits = self.model(x)
                loss_cls = label_smoothing_ce(
                    logits, y,
                    num_classes=self.num_classes,
                    eps=cfg.label_smoothing
                )

                if use_ssp_lite:
                    x_style = spectral_style_augment(
                        x,
                        strength=ssp_style_strength,
                        knots=ssp_style_knots,
                        prob=ssp_style_prob
                    )
                    logits_style = self.model(x_style)
                    loss_style = label_smoothing_ce(
                        logits_style, y,
                        num_classes=self.num_classes,
                        eps=cfg.label_smoothing
                    )
                    loss_cons = symmetric_kl(logits, logits_style)
                    loss = loss_cls + ssp_lambda_style * loss_style + ssp_lambda_cons * loss_cons
                else:
                    loss_style = zero
                    loss_cons = zero
                    loss = loss_cls

                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    acc = (pred == y).float().mean().item() * 100.0

            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))
            style_meter.update(float(loss_style.item()), x.size(0))
            cons_meter.update(float(loss_cons.item()), x.size(0))

        self.scheduler.step()

        if use_ssp_lite:
            print(
                f"Epoch [{epoch}/{cfg.src_epoch}] | "
                f"train_loss={loss_meter.avg:.6f} | "
                f"style_loss={style_meter.avg:.6f} | "
                f"cons_loss={cons_meter.avg:.6f} | "
                f"train_acc={acc_meter.avg:.2f}% | "
                f"lr={self.optimizer.param_groups[0]['lr']:.8f}"
            )
        else:
            print(
                f"Epoch [{epoch}/{cfg.src_epoch}] | "
                f"train_loss={loss_meter.avg:.6f} | "
                f"train_acc={acc_meter.avg:.2f}% | "
                f"lr={self.optimizer.param_groups[0]['lr']:.8f}"
            )

        return loss_meter.avg, acc_meter.avg

    def train(self):
        cfg = self.cfg

        save_dir = Path(cfg.save_model_path) / (str(cfg.Dataset.data_name) + str(cfg.Opt.lr_src)) / (str(cfg.Dataset.TL_Task) + "_Task")
        save_dir.mkdir(parents=True, exist_ok=True)

        final_ckpt_path = save_dir / cfg.model_name
        best_src_ckpt_path = save_dir / ("best_source_" + cfg.model_name)

        best_src_acc = -1.0
        best_epoch = -1

        for epoch in range(1, cfg.src_epoch + 1):
            self.train_one_epoch(epoch)

            src_acc = cal_acc(self.dataloaders["source_eval"], self.model, self.device)[0]
            tar_acc = cal_acc(self.dataloaders["target_eval"], self.model, self.device)[0]

            print(
                f"Task: {cfg.Dataset.TL_Task} | "
                f"Epoch {epoch}/{cfg.src_epoch} | "
                f"Source Acc = {src_acc:.2f}% | "
                f"Target Acc = {tar_acc:.2f}%"
            )

            if src_acc >= best_src_acc:
                best_src_acc = src_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), best_src_ckpt_path)
                print(f"[SAVE] best source ckpt -> {best_src_ckpt_path}")

            if cfg.save_every > 0 and ((epoch % cfg.save_every == 0) or (epoch == cfg.src_epoch)):
                epoch_ckpt_path = save_dir / f"epoch_{epoch}_{cfg.model_name}"
                torch.save(self.model.state_dict(), epoch_ckpt_path)
                print(f"[SAVE] epoch ckpt -> {epoch_ckpt_path}")

        torch.save(self.model.state_dict(), final_ckpt_path)
        print(f"[SAVE] final ckpt -> {final_ckpt_path}")
        print(f"[BEST] epoch={best_epoch}, best_source_acc={best_src_acc:.2f}%")

        return {
            "best_source_acc": best_src_acc,
            "best_epoch": best_epoch,
            "final_ckpt": str(final_ckpt_path),
            "best_source_ckpt": str(best_src_ckpt_path),
        }


@hydra.main(version_base=None, config_path='./Configs', config_name='defaults')
def run(cfg: omegaconf.DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id).strip('"\'')
    train_time = time.strftime('%m-%d %H:%M', time.localtime(time.time()))
    seed_list = list(range(2025, 2026))

    only_task = parse_only_task(getattr(cfg, "only_task", None))
    if only_task is not None:
        TL_Task_list = [only_task]
    else:
        TL_Task_list = list(permutations(cfg.Dataset.TL_list, 2))

    for seed_run in seed_list:
        for TL_Task in TL_Task_list:
            with open_dict(cfg):
                cfg.train_time = train_time
                cfg.seed_run = seed_run
                cfg.Dataset.TL_Task = TL_Task

                cfg.src_epoch = int(getattr(cfg, "src_epoch", 100))
                cfg.batch_size = int(getattr(cfg, "batch_size", 128))
                cfg.num_workers = int(getattr(cfg, "num_workers", 4))
                cfg.Dataset.input_kind = 'fft'
                cfg.Model.bottleneck_num = 128

                cfg.label_smoothing = float(getattr(cfg, "label_smoothing", 0.1))
                cfg.mixup_alpha = float(getattr(cfg, "mixup_alpha", 0.2))
                cfg.mixup_prob = float(getattr(cfg, "mixup_prob", 0.5))

                cfg.prompt_len_src = int(getattr(cfg, "prompt_len_src", 3))
                cfg.use_ssp_lite = bool(getattr(cfg, "use_ssp_lite", False))
                cfg.ssp_style_prob = float(getattr(cfg, "ssp_style_prob", 0.7))
                cfg.ssp_style_strength = float(getattr(cfg, "ssp_style_strength", 0.15))
                cfg.ssp_style_knots = int(getattr(cfg, "ssp_style_knots", 8))
                cfg.ssp_lambda_style = float(getattr(cfg, "ssp_lambda_style", 0.5))
                cfg.ssp_lambda_cons = float(getattr(cfg, "ssp_lambda_cons", 0.05))

                cfg.save_every = int(getattr(cfg, "save_every", 0))
                cfg.PR = 0

                if cfg.Dataset.data_name in ['PU', 'PU4D', 'CWRU', 'Gear']:
                    cfg.Opt.lr_src = float(getattr(cfg.Opt, "lr_src", 1e-3))
                else:
                    cfg.Opt.lr_src = float(getattr(cfg.Opt, "lr_src", 1e-3))

                cfg.Opt.weight_decay_src = float(getattr(cfg.Opt, "weight_decay_src", 1e-4))

                cfg.save_model_path = "./TTA_Model"
                cfg.model_name = build_model_name(cfg)

                cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

            trainer = SourceTrainer(cfg)
            trainer.setup()

            print(f"-------- {cfg.Dataset.data_name} Dataset: {TL_Task} Source Training --------")
            print(f"input kind: {cfg.Dataset.input_kind}, model: {cfg.Model.model_name}")
            print(f"source opt: lr_src={cfg.Opt.lr_src}, wd_src={cfg.Opt.weight_decay_src}")
            print(f"src_epoch={cfg.src_epoch}, batch_size={cfg.batch_size}")
            print(f"label_smoothing={cfg.label_smoothing}, mixup_alpha={cfg.mixup_alpha}, mixup_prob={cfg.mixup_prob}")
            print(
                f"prompt_len_src={cfg.prompt_len_src}, use_ssp_lite={cfg.use_ssp_lite}, "
                f"style_prob={cfg.ssp_style_prob}, style_strength={cfg.ssp_style_strength}, "
                f"style_knots={cfg.ssp_style_knots}, lambda_style={cfg.ssp_lambda_style}, "
                f"lambda_cons={cfg.ssp_lambda_cons}"
            )
            print(f"save_every={cfg.save_every}")
            print("config information:")
            pprint(cfg_dict)

            trainer.train()

    print(f"the TL task is {TL_Task_list}")


if __name__ == '__main__':
    run()