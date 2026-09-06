#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CWRU-specific physical fault-spectrum masks for 0711-Full.

Class order:
0 Normal
1/4/7 inner-race faults (7/14/21 mil)
2/5/8 ball faults
3/6/9 outer-race faults at 6 o'clock
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
import math
import torch
import torch.nn.functional as F

DOMAIN_SPEED_RPM: Dict[int, float] = {0:1797.0, 1:1772.0, 2:1750.0, 3:1730.0}
BPFI_ORDER = 5.4152
BPFO_ORDER = 3.5848
BSF_ORDER = 4.7135
FTF_ORDER = 0.39828
INNER_LABELS = frozenset({1,4,7})
BALL_LABELS = frozenset({2,5,8})
OUTER_LABELS = frozenset({3,6,9})

@dataclass(frozen=True)
class PhysicalEvidenceConfig:
    sampling_rate_hz: float = 12000.0
    fft_size: int = 1024
    spectrum_length: int = 512
    harmonics: int = 8
    outer_sideband_orders: Tuple[int,...] = (0,1)
    inner_sideband_orders: Tuple[int,...] = (0,1,2)
    ball_sideband_orders: Tuple[int,...] = (0,1,2)
    mask_sigma_bins: float = 1.0
    background_width_bins: int = 7
    max_mask_ratio: float = 0.18
    mask_activity_threshold: float = 0.10
    exclude_dc: bool = True
    def validate(self):
        if self.sampling_rate_hz <= 0: raise ValueError("sampling_rate_hz must be positive")
        if self.fft_size < 2 or self.fft_size % 2: raise ValueError("fft_size must be even")
        if not 1 < self.spectrum_length <= self.fft_size//2: raise ValueError("bad spectrum_length")
        if self.harmonics < 1: raise ValueError("harmonics must be >=1")
        if self.mask_sigma_bins <= 0: raise ValueError("mask_sigma_bins must be positive")
        if not 0 < self.max_mask_ratio <= 1: raise ValueError("max_mask_ratio must be in (0,1]")


def fault_group_from_label(label:int)->str:
    label=int(label)
    if label==0: return "healthy"
    if label in INNER_LABELS: return "inner"
    if label in BALL_LABELS: return "ball"
    if label in OUTER_LABELS: return "outer"
    raise ValueError(f"Unsupported CWRU label: {label}")


def characteristic_frequencies(speed_rpm: float) -> Dict[str,float]:
    if speed_rpm <= 0: raise ValueError("speed_rpm must be positive")
    shaft=float(speed_rpm)/60.0
    return {"shaft":shaft, "bpfi":BPFI_ORDER*shaft, "bpfo":BPFO_ORDER*shaft, "bsf":BSF_ORDER*shaft, "ftf":FTF_ORDER*shaft}


def _candidate_frequencies(base:float, sideband:float, harmonics:int, orders:Sequence[int], max_hz:float):
    values=set()
    for h in range(1,int(harmonics)+1):
        center=h*float(base)
        for order in orders:
            o=int(order)
            offsets=(0.0,) if o==0 else (-o*float(sideband), o*float(sideband))
            for off in offsets:
                f=center+off
                if 0.0 < f <= max_hz: values.add(round(float(f),10))
    return tuple(sorted(values))


def _soft_mask(freqs, cfg:PhysicalEvidenceConfig, device, dtype):
    if not freqs: return torch.zeros(cfg.spectrum_length,device=device,dtype=dtype)
    resolution=cfg.sampling_rate_hz/float(cfg.fft_size)
    axis=torch.arange(cfg.spectrum_length,device=device,dtype=dtype)
    centers=torch.tensor(freqs,device=device,dtype=dtype)/resolution
    distance=axis.unsqueeze(0)-centers.unsqueeze(1)
    mask=torch.exp(-0.5*(distance/float(cfg.mask_sigma_bins)).pow(2)).amax(dim=0).clamp(0,1)
    if cfg.exclude_dc: mask[0]=0
    active=mask>=float(cfg.mask_activity_threshold)
    max_active=max(1,int(math.floor(cfg.max_mask_ratio*cfg.spectrum_length)))
    if int(active.sum())>max_active:
        idx=torch.topk(mask,k=max_active).indices
        clipped=torch.zeros_like(mask); clipped[idx]=mask[idx]; mask=clipped
    return mask


def build_physical_masks(pseudo_labels:torch.Tensor,target_domain:int,config:PhysicalEvidenceConfig=PhysicalEvidenceConfig()):
    config.validate()
    if pseudo_labels.ndim != 1: raise ValueError("pseudo_labels must be 1-D")
    if int(target_domain) not in DOMAIN_SPEED_RPM: raise ValueError(f"Unsupported CWRU target domain: {target_domain}")
    labels=pseudo_labels.long(); device=labels.device; dtype=torch.float32
    freq=characteristic_frequencies(DOMAIN_SPEED_RPM[int(target_domain)])
    resolution=config.sampling_rate_hz/float(config.fft_size)
    max_hz=(config.spectrum_length-1)*resolution
    inner=_soft_mask(_candidate_frequencies(freq['bpfi'],freq['shaft'],config.harmonics,config.inner_sideband_orders,max_hz),config,device,dtype)
    outer=_soft_mask(_candidate_frequencies(freq['bpfo'],freq['shaft'],config.harmonics,config.outer_sideband_orders,max_hz),config,device,dtype)
    ball=_soft_mask(_candidate_frequencies(freq['bsf'],freq['ftf'],config.harmonics,config.ball_sideband_orders,max_hz),config,device,dtype)
    masks=torch.zeros(labels.numel(),config.spectrum_length,device=device,dtype=dtype)
    inner_sel=torch.zeros_like(labels,dtype=torch.bool); ball_sel=torch.zeros_like(labels,dtype=torch.bool); outer_sel=torch.zeros_like(labels,dtype=torch.bool)
    for k in INNER_LABELS: inner_sel |= labels.eq(k)
    for k in BALL_LABELS: ball_sel |= labels.eq(k)
    for k in OUTER_LABELS: outer_sel |= labels.eq(k)
    masks[inner_sel]=inner; masks[ball_sel]=ball; masks[outer_sel]=outer
    return masks, inner_sel|ball_sel|outer_sel


def _to_cf(x):
    squeezed=transposed=False
    if x.ndim==2: x=x.unsqueeze(1); squeezed=True
    elif x.ndim==3 and x.shape[-1]==1: x=x.transpose(1,2); transposed=True
    elif x.ndim!=3: raise ValueError(f"Expected [B,L]/[B,C,L]/[B,L,1], got {tuple(x.shape)}")
    return x,squeezed,transposed

def _restore(x,squeezed,transposed):
    if transposed: x=x.transpose(1,2)
    if squeezed: x=x.squeeze(1)
    return x

def construct_physical_destructive_view(x,mask,background_width_bins=7):
    xcf,sq,tr=_to_cf(x)
    if mask.shape != (xcf.shape[0],xcf.shape[-1]): raise ValueError("mask shape mismatch")
    width=max(3,int(background_width_bins)); width += int(width%2==0); pad=width//2
    bg=F.avg_pool1d(F.pad(xcf,(pad,pad),mode='replicate'),kernel_size=width,stride=1)
    m=mask.to(device=xcf.device,dtype=xcf.dtype).unsqueeze(1)
    return _restore((1-m)*xcf+m*bg,sq,tr)

def class_margin(probability,pseudo_labels):
    labels=pseudo_labels.long().to(probability.device)
    selected=probability.gather(1,labels.unsqueeze(1)).squeeze(1)
    other=probability.clone(); other.scatter_(1,labels.unsqueeze(1),float('-inf'))
    return selected-other.max(dim=1).values

def robust_fault_evidence_score(evidence_drop,applicable,eps=1e-6):
    score=torch.ones_like(evidence_drop); values=evidence_drop[applicable]
    if values.numel()==0: return score
    med=values.median(); mad=(values-med).abs().median(); scale=1.4826*mad+float(eps)
    score[applicable]=torch.sigmoid((values-med)/scale)
    return score.clamp(0,1)

def mask_active_ratio(mask,threshold=0.10):
    return (mask>=float(threshold)).to(mask.dtype).mean(dim=1)
