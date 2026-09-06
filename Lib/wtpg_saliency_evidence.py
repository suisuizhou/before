"""Teacher-gradient contiguous spectral evidence masks for WTPG gear faults."""

import torch
import torch.nn.functional as F


def contiguous_saliency_masks(saliency: torch.Tensor, pseudo_labels: torch.Tensor,
                              bands: int = 8, half_width: int = 3,
                              max_mask_ratio: float = 0.18):
    if saliency.ndim != 3 or saliency.shape[1] != 1:
        raise ValueError("WTPG saliency must have shape [B,1,L]")
    if pseudo_labels.shape != (saliency.shape[0],):
        raise ValueError("WTPG saliency/pseudo-label shape mismatch")
    if not 0 < max_mask_ratio <= 1:
        raise ValueError("invalid WTPG maximum mask ratio")
    length = saliency.shape[-1]
    smooth = F.avg_pool1d(saliency.abs(), kernel_size=9, stride=1, padding=4).squeeze(1)
    masks = torch.zeros_like(smooth)
    applicable = pseudo_labels != 0
    max_bins = max(1, int(length * max_mask_ratio))
    radius = max(1, int(half_width))
    for row in range(saliency.shape[0]):
        if not bool(applicable[row]):
            continue
        scores = smooth[row].clone()
        used = 0
        for _ in range(max(1, int(bands))):
            center = int(scores.argmax())
            left, right = max(1, center - radius), min(length, center + radius + 1)
            available = min(right - left, max_bins - used)
            if available <= 0 or float(scores[center]) <= 0:
                break
            right = left + available
            masks[row, left:right] = 1.0
            used += available
            scores[max(0, left - radius):min(length, right + radius)] = -1
    return masks, applicable
