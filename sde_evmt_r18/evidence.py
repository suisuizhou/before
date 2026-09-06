from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .augment import cfg_get


@dataclass
class EvidenceResult:
    mask: torch.Tensor
    destroyed: torch.Tensor
    plpd: torch.Tensor
    margin_drop: torch.Tensor


def _probability_margin(probability, pseudo):
    selected = probability.gather(1, pseudo[:, None]).squeeze(1)
    competitors = probability.clone()
    competitors.scatter_(1, pseudo[:, None], float("-inf"))
    return selected - competitors.max(dim=1).values


def _contiguous_mask(saliency, num_bands, total_width):
    batch, length = saliency.shape
    num_bands = max(1, min(int(num_bands), int(total_width)))
    widths = [total_width // num_bands] * num_bands
    for index in range(total_width % num_bands):
        widths[index] += 1
    mask = torch.zeros_like(saliency, dtype=torch.bool)
    for width in widths:
        score = F.avg_pool1d(
            saliency[:, None], kernel_size=width, stride=1
        ).squeeze(1)
        for row in range(batch):
            blocked = F.max_pool1d(
                mask[row][None, None].float(),
                kernel_size=width,
                stride=1,
            ).squeeze()
            candidate = score[row].masked_fill(blocked.bool(), float("-inf"))
            if bool(torch.isfinite(candidate).any()):
                start = int(candidate.argmax())
                mask[row, start : start + width] = True
    return mask


def _local_median_replacement(flat, mask):
    destroyed = flat.clone()
    batch, length = flat.shape
    for row in range(batch):
        indices = mask[row].nonzero(as_tuple=False).flatten()
        if not len(indices):
            continue
        splits = torch.where(indices[1:] != indices[:-1] + 1)[0] + 1
        for segment in torch.tensor_split(indices, splits.cpu().tolist()):
            start = int(segment[0])
            end = int(segment[-1]) + 1
            width = end - start
            left = flat[row, max(0, start - width) : start]
            right = flat[row, end : min(length, end + width)]
            neighbors = torch.cat((left, right))
            replacement = (
                neighbors.median()
                if len(neighbors)
                else flat[row].median()
            )
            destroyed[row, start:end] = replacement
    return destroyed


def verify_margin_evidence(forward_fn, spectrum, logits, pseudo, cfg):
    """Destroy salient contiguous bands and measure pseudo-class margin loss."""
    if spectrum.ndim not in (2, 3):
        raise ValueError(f"expected [B,L] or [B,1,L], got {tuple(spectrum.shape)}")
    shape = spectrum.shape
    with torch.enable_grad():
        probe = spectrum.detach().requires_grad_(True)
        probe_logits = forward_fn(probe)
        selected = probe_logits.gather(1, pseudo[:, None]).sum()
        gradient = torch.autograd.grad(selected, probe, create_graph=False)[0]
    saliency = (probe.detach() * gradient.detach()).abs().reshape(len(spectrum), -1)
    kernel = int(cfg_get(cfg, "evidence.smooth_kernel", 9))
    kernel = max(1, min(kernel, saliency.shape[-1]))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    saliency = F.avg_pool1d(
        saliency[:, None], kernel, stride=1, padding=kernel // 2
    ).squeeze(1)
    ratio = float(cfg_get(cfg, "evidence.mask_ratio_max", 0.10))
    ratio = max(float(cfg_get(cfg, "evidence.mask_ratio_min", 0.05)), ratio)
    total_width = max(1, min(saliency.shape[-1], round(saliency.shape[-1] * ratio)))
    mask = _contiguous_mask(
        saliency,
        int(cfg_get(cfg, "evidence.num_bands", 2)),
        total_width,
    )
    flat = spectrum.detach().reshape(len(spectrum), -1)
    destroyed = _local_median_replacement(flat, mask).reshape(shape)
    with torch.no_grad():
        destroyed_logits = forward_fn(destroyed)
        original_probability = logits.detach().softmax(dim=1)
        destroyed_probability = destroyed_logits.softmax(dim=1)
    selected_original = original_probability.gather(1, pseudo[:, None]).squeeze(1)
    selected_destroyed = destroyed_probability.gather(1, pseudo[:, None]).squeeze(1)
    plpd = selected_original - selected_destroyed
    margin_drop = _probability_margin(
        original_probability, pseudo
    ) - _probability_margin(destroyed_probability, pseudo)
    return EvidenceResult(mask, destroyed, plpd, margin_drop)
