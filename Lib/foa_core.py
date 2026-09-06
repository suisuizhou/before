#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

def entropy_from_logits(logits):
    p = torch.softmax(logits, dim=1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)
    return ent.mean()

def batch_stats(feat):
    mu = feat.mean(dim=0)
    std = feat.std(dim=0, unbiased=False)
    return mu, std

def update_ema(mu_batch, mu_ema, alpha=0.1):
    if mu_ema is None:
        return mu_batch.detach()
    return alpha * mu_batch.detach() + (1.0 - alpha) * mu_ema.detach()

def shift_features(feat, mu_s, mu_t_ema, gamma=1.0):
    if mu_t_ema is None:
        return feat
    direction = (mu_s - mu_t_ema).detach()
    feat_shift = feat + gamma * direction.unsqueeze(0)
    return feat_shift

@torch.no_grad()
def evaluate_candidate(model, x, prompt_candidate, mu_s, std_s, mu_t_ema,
                       lambda_stat=0.4, gamma=1.0, alpha=0.1):
    # 写入候选 prompt
    model[0].prompt_embed.copy_(prompt_candidate)

    # backbone feature
    feat = model[0](x)  # [B, D]

    # 当前 batch 统计
    mu_t, std_t = batch_stats(feat)

    # 更新 EMA center
    mu_t_ema_new = update_ema(mu_t, mu_t_ema, alpha=alpha)

    # activation shifting
    feat_shift = shift_features(feat, mu_s, mu_t_ema_new, gamma=gamma)

    # head
    feat_b = model[1](feat_shift)
    logits = model[2](feat_b)

    # fitness = entropy + stat discrepancy
    ent = entropy_from_logits(logits)
    disc = F.mse_loss(mu_t, mu_s) + F.mse_loss(std_t, std_s)
    fitness = ent + lambda_stat * disc

    pred = logits.argmax(dim=1)

    return {
        "fitness": float(fitness.item()),
        "entropy": float(ent.item()),
        "disc": float(disc.item()),
        "logits": logits.detach(),
        "pred": pred.detach(),
        "mu_t_ema_new": mu_t_ema_new.detach(),
        "mu_t": mu_t.detach(),
        "std_t": std_t.detach(),
    }
