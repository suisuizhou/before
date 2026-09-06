from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def configure_vit_cfg_dict(cfg: Dict | None = None, *, use_spectral_adapter: bool) -> Dict:
    """Return the fixed ViT1D architecture used by Protocol A.

    The caller may pass an existing plain dict; agreed architecture fields are
    overwritten so DtCC and 0711 use the same backbone capacity.
    """
    out = dict(cfg or {})
    out.update(
        {
            "model_name": "ViT1D",
            "input_len": 512,
            "patch_size": 16,
            "in_chans": 1,
            "embed_dim": 256,
            "depth": 4,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "drop_rate": 0.1,
            "prompt_len": 3,
            "bottleneck": True,
            "bottleneck_num": 128,
            "temp": 1,
            "model_type": "linear",
            "Dropout": True,
            "use_spectral_adapter": bool(use_spectral_adapter),
            "band_num": 256,
            "adapter_delta": 0.1,
        }
    )
    return out


def apply_vit_cfg(cfg, *, use_spectral_adapter: bool) -> None:
    """Mutate an OmegaConf/Hydra config to the agreed ViT architecture."""
    fixed = configure_vit_cfg_dict({}, use_spectral_adapter=use_spectral_adapter)
    for key, value in fixed.items():
        setattr(cfg.Model, key, value)


def select_trainable_parameters(model: nn.Module, *, method: str) -> List[str]:
    """Freeze the model and enable exactly the Protocol-A PEFT parameters."""
    method = str(method).strip().lower()
    if method not in {"dtcc", "0711_full"}:
        raise ValueError(f"Unsupported method: {method}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    allowed = ("prompt_embed",)
    if method == "0711_full":
        allowed = ("prompt_embed", "band_scale", "band_bias", "warp_ctrl")

    selected: List[str] = []
    for name, parameter in model.named_parameters():
        if any(token in name for token in allowed):
            parameter.requires_grad = True
            selected.append(name)

    if not selected:
        raise RuntimeError(
            f"No Protocol-A trainable parameters were found for method={method}. "
            "Check that ViT prompt_len=3 and, for 0711_full, Adapter/F-Warp are enabled."
        )
    return selected


def method_checkpoint_dir(
    method: str,
    dataset: str,
    source_domain: int,
    seed: int,
    *,
    root: str | Path = "TTA_Model_ViT_A",
) -> Path:
    return (
        Path(root)
        / str(method)
        / str(dataset)
        / f"source_{int(source_domain)}"
        / f"seed_{int(seed)}"
    )


def strict_online_batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy from the pre-update logits for the arriving batch."""
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets batch dimensions must match")
    if targets.numel() == 0:
        return 0.0
    predictions = logits.detach().argmax(dim=1)
    correct = int((predictions == targets.detach()).sum().item())
    return 100.0 * correct / int(targets.numel())


def forward_parts(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project-specific sequential model forward with tuple-safe ViT output."""
    feature = model[0](x)
    if isinstance(feature, (tuple, list)):
        feature = feature[0]
    bottleneck = model[1](feature)
    logits = model[2](bottleneck)
    return bottleneck, logits


def confusion_update(confusion: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    n = confusion.shape[0]
    y_true = y_true.detach().to("cpu", dtype=torch.long).view(-1)
    y_pred = y_pred.detach().to("cpu", dtype=torch.long).view(-1)
    valid = (y_true >= 0) & (y_true < n) & (y_pred >= 0) & (y_pred < n)
    encoded = y_true[valid] * n + y_pred[valid]
    counts = torch.bincount(encoded, minlength=n * n).reshape(n, n)
    confusion += counts


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    cm = confusion.to(dtype=torch.float64)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
    return float(f1.mean().item() * 100.0)


def compatible_load(model: nn.Module, checkpoint: str | Path, device=None):
    """Load all shape-compatible tensors; return audit information."""
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state)}")

    current = model.state_dict()
    compatible = {}
    unexpected = []
    mismatched = []
    for key, value in state.items():
        key2 = key[7:] if str(key).startswith("module.") else key
        if key2 not in current:
            unexpected.append(key2)
        elif tuple(current[key2].shape) != tuple(value.shape):
            mismatched.append((key2, tuple(value.shape), tuple(current[key2].shape)))
        else:
            compatible[key2] = value
    message = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "missing": list(message.missing_keys),
        "unexpected": unexpected,
        "mismatched": mismatched,
    }

from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class DtCCMemoryEntry:
    feature: torch.Tensor
    probability: torch.Tensor
    confidence: float


class DtCCMemory:
    """Class-balanced DtCC memory with per-class top-confidence retention."""

    def __init__(self, num_classes: int, capacity_per_class: int = 50):
        self.num_classes = int(num_classes)
        self.capacity_per_class = int(capacity_per_class)
        self.by_class = {k: [] for k in range(self.num_classes)}

    def add(self, class_id: int, feature: torch.Tensor, probability: torch.Tensor, confidence: float):
        class_id = int(class_id)
        if class_id not in self.by_class:
            raise ValueError(f"Invalid class id {class_id}")
        entry = DtCCMemoryEntry(
            feature=feature.detach().to("cpu").clone(),
            probability=probability.detach().to("cpu").clone(),
            confidence=float(confidence),
        )
        bucket = self.by_class[class_id]
        bucket.append(entry)
        bucket.sort(key=lambda e: e.confidence, reverse=True)
        del bucket[self.capacity_per_class :]

    def initialize_from_classifier(self, classifier: nn.Module, device: torch.device | str):
        """Initialize M0^k from the k-th classifier weight vector, as in DtCC."""
        fc = getattr(classifier, "fc", classifier)
        if not hasattr(fc, "weight"):
            raise AttributeError("Classifier must expose .fc.weight or .weight")
        weights = fc.weight.detach().to(device)
        with torch.no_grad():
            try:
                logits = classifier(weights)
            except Exception:
                logits = F.linear(weights, fc.weight, getattr(fc, "bias", None))
            probabilities = F.softmax(logits, dim=1)
        if weights.shape[0] != self.num_classes:
            raise ValueError(
                f"Classifier has {weights.shape[0]} rows but memory expects {self.num_classes} classes"
            )
        for k in range(self.num_classes):
            self.add(k, weights[k], probabilities[k], probabilities[k, k].item())

    def prototypes(self, device: torch.device | str):
        protos = []
        valid_classes = []
        for k in range(self.num_classes):
            bucket = self.by_class[k]
            if not bucket:
                continue
            feat = torch.stack([entry.feature for entry in bucket], dim=0).to(device)
            protos.append(feat.mean(dim=0))
            valid_classes.append(k)
        if not protos:
            return None, []
        return torch.stack(protos, dim=0), valid_classes

    def all_entries(self, device: torch.device | str):
        features = []
        probabilities = []
        classes = []
        for k in range(self.num_classes):
            for entry in self.by_class[k]:
                features.append(entry.feature)
                probabilities.append(entry.probability)
                classes.append(k)
        if not features:
            return None, None, None
        return (
            torch.stack(features, dim=0).to(device),
            torch.stack(probabilities, dim=0).to(device),
            torch.tensor(classes, dtype=torch.long, device=device),
        )


def spectral_entropy(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        if x.shape[1] == 1:
            x = x[:, 0]
        elif x.shape[-1] == 1:
            x = x[..., 0]
        else:
            x = x.flatten(1)
    elif x.dim() != 2:
        x = x.flatten(1)
    power = x.abs().pow(2)
    distribution = power / power.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(distribution * distribution.clamp_min(1e-12).log()).sum(dim=1)


def dtcc_dynamic_divide(x: torch.Tensor, logits: torch.Tensor):
    probability = F.softmax(logits, dim=1)
    confidence, pseudo = probability.detach().max(dim=1)
    entropy = spectral_entropy(x).detach()
    certain = (confidence >= confidence.mean()) & (entropy <= entropy.mean())
    uncertain = ~certain
    return certain, uncertain, probability, pseudo, confidence, entropy


def source_checkpoint_candidates(
    source_domain: int,
    representative_target: int,
    seed: int = 2025,
    *,
    root: str | Path = "TTA_Model",
    dataset_lr_dir: str = "PU4D0.001",
) -> List[Path]:
    """Candidate source paths produced by existing project source trainers.

    Historical scripts have serialized TL_Task as either a Python tuple or a
    list.  Staging accepts both spellings, then copies the selected source
    checkpoint into a method-specific Protocol-A directory.
    """
    src = int(source_domain)
    tar = int(representative_target)
    model_name = f"ViT1D{int(seed)}fft_Linear.pt"
    task_spellings = [f"({src}, {tar})_Task", f"[{src}, {tar}]_Task"]
    return [Path(root) / dataset_lr_dir / task / ("best_source_" + model_name) for task in task_spellings]
