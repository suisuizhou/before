#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import hydra
import omegaconf

import main_Src_stronger_SSP_lite_STABLE as base


def spectral_warp_augment(x, max_warp=2.0, knots=16, prob=0.7):
    """
    Source-side random smooth frequency deformation:
        x(f) -> x(f + delta(f))
    delta is random, low-dimensional, and smoothly interpolated.
    """
    if base.random.random() > prob:
        return x

    x_cf, squeezed, transposed = base._to_channel_first_1d(x)
    B, C, L = x_cf.shape
    device = x_cf.device
    dtype = x_cf.dtype

    knots = max(2, int(knots))
    ctrl = torch.empty(B, 1, knots, device=device, dtype=dtype).uniform_(-max_warp, max_warp)

    # add a weak global tilt so the deformation can include mild compression/expansion
    pos = torch.linspace(-1.0, 1.0, steps=knots, device=device, dtype=dtype).view(1, 1, knots)
    tilt = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(-0.25 * max_warp, 0.25 * max_warp)
    ctrl = ctrl + tilt * pos

    delta = F.interpolate(ctrl, size=L, mode="linear", align_corners=True)

    base_pos = torch.arange(L, device=device, dtype=dtype).view(1, 1, L)
    sample_pos = (base_pos + delta).clamp(0.0, float(L - 1))

    x_norm = 2.0 * sample_pos / float(L - 1) - 1.0
    x_norm = x_norm.squeeze(1)
    y_norm = torch.zeros_like(x_norm)

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)  # [B, 1, L, 2]
    x_4d = x_cf.unsqueeze(2)  # [B, C, 1, L]

    y_4d = F.grid_sample(
        x_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    x_aug = y_4d.squeeze(2)
    x_aug = base._restore_1d_shape(x_aug, squeezed, transposed)
    return x_aug


def feature_consistency_loss(feat_clean, feat_aug):
    feat_clean = F.normalize(feat_clean.detach(), dim=1)
    feat_aug = F.normalize(feat_aug, dim=1)
    return (feat_clean - feat_aug).pow(2).sum(dim=1).mean()


class SDE_SourceTrainer(base.SourceTrainer):
    def forward_parts(self, x):
        feat = self.model[0](x)
        feat_b = self.model[1](feat)
        logits = self.model[2](feat_b)
        return logits, feat_b

    def train_one_epoch(self, epoch):
        cfg = self.cfg
        self.model.train()

        loss_meter = base.AverageMeter()
        acc_meter = base.AverageMeter()
        style_meter = base.AverageMeter()
        warp_meter = base.AverageMeter()
        cons_meter = base.AverageMeter()
        feat_meter = base.AverageMeter()

        use_ssp_lite = bool(getattr(cfg, "use_ssp_lite", True))
        use_sde_lite = bool(getattr(cfg, "use_sde_lite", True))

        ssp_style_prob = float(getattr(cfg, "ssp_style_prob", 0.7))
        ssp_style_strength = float(getattr(cfg, "ssp_style_strength", 0.15))
        ssp_style_knots = int(getattr(cfg, "ssp_style_knots", 8))
        ssp_lambda_style = float(getattr(cfg, "ssp_lambda_style", 0.5))

        sde_warp_prob = float(getattr(cfg, "sde_warp_prob", 0.7))
        sde_warp_knots = int(getattr(cfg, "sde_warp_knots", 16))
        sde_warp_max = float(getattr(cfg, "sde_warp_max", 2.0))
        sde_lambda_warp = float(getattr(cfg, "sde_lambda_warp", 0.5))
        sde_lambda_style_warp = float(getattr(cfg, "sde_lambda_style_warp", 0.25))
        sde_lambda_cons = float(getattr(cfg, "sde_lambda_cons", 0.03))
        sde_lambda_feat = float(getattr(cfg, "sde_lambda_feat", 0.02))
        sde_use_style_warp = bool(getattr(cfg, "sde_use_style_warp", True))

        for x, y, _ in self.dataloaders["source_train"]:
            if x.size(0) <= 1:
                continue

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            zero = x.new_tensor(0.0)

            do_mixup = (cfg.mixup_alpha > 0) and (base.random.random() < cfg.mixup_prob)

            loss_style = zero
            loss_warp = zero
            loss_style_warp = zero
            loss_cons = zero
            loss_feat = zero
            cons_count = 0
            feat_count = 0

            if do_mixup:
                x_main, y_a, y_b, lam = base.mixup_data(x, y, alpha=cfg.mixup_alpha)
                logits, feat_b = self.forward_parts(x_main)

                loss_cls = base.mixup_label_smoothing_ce(
                    logits, y_a, y_b, lam,
                    num_classes=self.num_classes,
                    eps=cfg.label_smoothing
                )

                if use_ssp_lite:
                    x_style = base.spectral_style_augment(
                        x_main,
                        strength=ssp_style_strength,
                        knots=ssp_style_knots,
                        prob=ssp_style_prob
                    )
                    logits_style, feat_style = self.forward_parts(x_style)
                    loss_style = base.mixup_label_smoothing_ce(
                        logits_style, y_a, y_b, lam,
                        num_classes=self.num_classes,
                        eps=cfg.label_smoothing
                    )
                    loss_cons = loss_cons + base.symmetric_kl(logits, logits_style)
                    loss_feat = loss_feat + feature_consistency_loss(feat_b, feat_style)
                    cons_count += 1
                    feat_count += 1

                if use_sde_lite:
                    x_warp = spectral_warp_augment(
                        x_main,
                        max_warp=sde_warp_max,
                        knots=sde_warp_knots,
                        prob=sde_warp_prob
                    )
                    logits_warp, feat_warp = self.forward_parts(x_warp)
                    loss_warp = base.mixup_label_smoothing_ce(
                        logits_warp, y_a, y_b, lam,
                        num_classes=self.num_classes,
                        eps=cfg.label_smoothing
                    )
                    loss_cons = loss_cons + base.symmetric_kl(logits, logits_warp)
                    loss_feat = loss_feat + feature_consistency_loss(feat_b, feat_warp)
                    cons_count += 1
                    feat_count += 1

                    if sde_use_style_warp:
                        x_style_warp = base.spectral_style_augment(
                            x_warp,
                            strength=ssp_style_strength,
                            knots=ssp_style_knots,
                            prob=ssp_style_prob
                        )
                        logits_style_warp, feat_style_warp = self.forward_parts(x_style_warp)
                        loss_style_warp = base.mixup_label_smoothing_ce(
                            logits_style_warp, y_a, y_b, lam,
                            num_classes=self.num_classes,
                            eps=cfg.label_smoothing
                        )
                        loss_cons = loss_cons + base.symmetric_kl(logits, logits_style_warp)
                        loss_feat = loss_feat + feature_consistency_loss(feat_b, feat_style_warp)
                        cons_count += 1
                        feat_count += 1

                if cons_count > 0:
                    loss_cons = loss_cons / cons_count
                if feat_count > 0:
                    loss_feat = loss_feat / feat_count

                loss = (
                    loss_cls
                    + ssp_lambda_style * loss_style
                    + sde_lambda_warp * loss_warp
                    + sde_lambda_style_warp * loss_style_warp
                    + sde_lambda_cons * loss_cons
                    + sde_lambda_feat * loss_feat
                )

                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    acc = (
                        lam * (pred == y_a).float()
                        + (1.0 - lam) * (pred == y_b).float()
                    ).mean().item() * 100.0

            else:
                logits, feat_b = self.forward_parts(x)

                loss_cls = base.label_smoothing_ce(
                    logits, y,
                    num_classes=self.num_classes,
                    eps=cfg.label_smoothing
                )

                if use_ssp_lite:
                    x_style = base.spectral_style_augment(
                        x,
                        strength=ssp_style_strength,
                        knots=ssp_style_knots,
                        prob=ssp_style_prob
                    )
                    logits_style, feat_style = self.forward_parts(x_style)
                    loss_style = base.label_smoothing_ce(
                        logits_style, y,
                        num_classes=self.num_classes,
                        eps=cfg.label_smoothing
                    )
                    loss_cons = loss_cons + base.symmetric_kl(logits, logits_style)
                    loss_feat = loss_feat + feature_consistency_loss(feat_b, feat_style)
                    cons_count += 1
                    feat_count += 1

                if use_sde_lite:
                    x_warp = spectral_warp_augment(
                        x,
                        max_warp=sde_warp_max,
                        knots=sde_warp_knots,
                        prob=sde_warp_prob
                    )
                    logits_warp, feat_warp = self.forward_parts(x_warp)
                    loss_warp = base.label_smoothing_ce(
                        logits_warp, y,
                        num_classes=self.num_classes,
                        eps=cfg.label_smoothing
                    )
                    loss_cons = loss_cons + base.symmetric_kl(logits, logits_warp)
                    loss_feat = loss_feat + feature_consistency_loss(feat_b, feat_warp)
                    cons_count += 1
                    feat_count += 1

                    if sde_use_style_warp:
                        x_style_warp = base.spectral_style_augment(
                            x_warp,
                            strength=ssp_style_strength,
                            knots=ssp_style_knots,
                            prob=ssp_style_prob
                        )
                        logits_style_warp, feat_style_warp = self.forward_parts(x_style_warp)
                        loss_style_warp = base.label_smoothing_ce(
                            logits_style_warp, y,
                            num_classes=self.num_classes,
                            eps=cfg.label_smoothing
                        )
                        loss_cons = loss_cons + base.symmetric_kl(logits, logits_style_warp)
                        loss_feat = loss_feat + feature_consistency_loss(feat_b, feat_style_warp)
                        cons_count += 1
                        feat_count += 1

                if cons_count > 0:
                    loss_cons = loss_cons / cons_count
                if feat_count > 0:
                    loss_feat = loss_feat / feat_count

                loss = (
                    loss_cls
                    + ssp_lambda_style * loss_style
                    + sde_lambda_warp * loss_warp
                    + sde_lambda_style_warp * loss_style_warp
                    + sde_lambda_cons * loss_cons
                    + sde_lambda_feat * loss_feat
                )

                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    acc = (pred == y).float().mean().item() * 100.0

            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))
            style_meter.update(float(loss_style.item()), x.size(0))
            warp_meter.update(float((loss_warp + loss_style_warp).item()), x.size(0))
            cons_meter.update(float(loss_cons.item()), x.size(0))
            feat_meter.update(float(loss_feat.item()), x.size(0))

        self.scheduler.step()

        print(
            f"Epoch [{epoch}/{cfg.src_epoch}] | "
            f"train_loss={loss_meter.avg:.6f} | "
            f"style_loss={style_meter.avg:.6f} | "
            f"warp_loss={warp_meter.avg:.6f} | "
            f"cons_loss={cons_meter.avg:.6f} | "
            f"feat_loss={feat_meter.avg:.6f} | "
            f"train_acc={acc_meter.avg:.2f}% | "
            f"lr={self.optimizer.param_groups[0]['lr']:.8f}"
        )

        return loss_meter.avg, acc_meter.avg


base.SourceTrainer = SDE_SourceTrainer


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    base.run.__wrapped__(cfg)


if __name__ == "__main__":
    run()
