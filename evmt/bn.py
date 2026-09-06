from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm


@dataclass(frozen=True)
class BNStats:
    mean: Tensor
    var: Tensor


@dataclass
class BNLayerState:
    source_mean: Tensor
    source_var: Tensor
    target_mean: Tensor | None = None
    target_m2: Tensor | None = None
    target_count: int = 0


def _batch_channel_moments(x: Tensor) -> tuple[Tensor, Tensor, int]:
    if x.ndim < 2:
        raise ValueError(f"batch-norm input must have a channel dimension: {x.shape}")
    channel_first = x.detach().movedim(1, 0).reshape(x.shape[1], -1)
    return (
        channel_first.mean(dim=1),
        channel_first.var(dim=1, unbiased=False),
        int(channel_first.shape[1]),
    )


def _merge_moments(
    mean: Tensor | None,
    m2: Tensor | None,
    count: int,
    batch_mean: Tensor,
    batch_var: Tensor,
    batch_count: int,
) -> tuple[Tensor, Tensor, int]:
    if count == 0 or mean is None or m2 is None:
        return batch_mean.clone(), batch_var * batch_count, int(batch_count)
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = (
        m2
        + batch_var * batch_count
        + delta.square() * (count * batch_count / total)
    )
    return merged_mean, merged_m2, int(total)


class TargetBNController:
    def __init__(
        self,
        model,
        blend_batches: int = 5,
        max_target_weight: float = 0.9,
        eps: float = 1e-5,
    ):
        if blend_batches < 1:
            raise ValueError("blend_batches must be at least one")
        if not 0.0 <= max_target_weight <= 1.0:
            raise ValueError("max_target_weight must be in [0, 1]")
        self.model = model
        self.blend_batches = int(blend_batches)
        self.max_target_weight = float(max_target_weight)
        self.eps = float(eps)
        self.seen_batches = 0
        self._states: dict[str, BNLayerState] = {}
        for name, module in self._bn_modules(model).items():
            if module.running_mean is None or module.running_var is None:
                raise ValueError(f"BN layer {name!r} does not track running statistics")
            self._states[name] = BNLayerState(
                source_mean=module.running_mean.detach().clone(),
                source_var=module.running_var.detach().clone(),
            )
        if not self._states:
            raise ValueError("model contains no batch-normalization layers")

    @staticmethod
    def _bn_modules(model) -> dict[str, _BatchNorm]:
        return {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, _BatchNorm)
        }

    def _checked_modules(self, model) -> dict[str, _BatchNorm]:
        modules = self._bn_modules(model)
        if modules.keys() != self._states.keys():
            raise ValueError("model BN layout differs from controller layout")
        return modules

    @property
    def target_weight(self) -> float:
        progress = min(1.0, self.seen_batches / self.blend_batches)
        return self.max_target_weight * progress

    def layer_state(self, name: str) -> BNLayerState:
        return self._states[name]

    def source_state(self, name: str) -> BNStats:
        state = self._states[name]
        return BNStats(state.source_mean.clone(), state.source_var.clone())

    def fused_state(self, name: str) -> BNStats:
        state = self._states[name]
        if state.target_count == 0:
            return self.source_state(name)
        weight = self.target_weight
        target_var = state.target_m2 / max(state.target_count, 1)
        mixed_mean = (
            (1.0 - weight) * state.source_mean + weight * state.target_mean
        )
        mixed_var = (
            (1.0 - weight)
            * (state.source_var + (state.source_mean - mixed_mean).square())
            + weight
            * (target_var + (state.target_mean - mixed_mean).square())
        ).clamp_min(self.eps)
        return BNStats(mixed_mean, mixed_var)

    @contextmanager
    def prediction_stats(self, model=None):
        model = self.model if model is None else model
        modules = self._checked_modules(model)
        snapshots = {
            name: (
                module.running_mean,
                module.running_var,
                bool(module.training),
            )
            for name, module in modules.items()
        }
        try:
            for name, module in modules.items():
                fused = self.fused_state(name)
                module.running_mean = fused.mean.to(
                    device=module.running_mean.device,
                    dtype=module.running_mean.dtype,
                ).clone()
                module.running_var = fused.var.to(
                    device=module.running_var.device,
                    dtype=module.running_var.dtype,
                ).clone()
                module.eval()
            yield
        finally:
            for name, module in modules.items():
                mean, var, training = snapshots[name]
                module.running_mean = mean
                module.running_var = var
                module.train(training)

    @torch.no_grad()
    def observe_batch(self, model, x: Tensor, forward_fn) -> None:
        modules = self._checked_modules(model)
        captured: dict[str, list[tuple[Tensor, Tensor, int]]] = {
            name: [] for name in modules
        }
        handles = []

        def make_hook(name):
            def hook(_module, inputs):
                captured[name].append(_batch_channel_moments(inputs[0]))
            return hook

        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(make_hook(name)))
        try:
            with self.prediction_stats(model):
                forward_fn(model, x)
        finally:
            for handle in handles:
                handle.remove()

        for name, observations in captured.items():
            if len(observations) != 1:
                raise RuntimeError(
                    f"expected one observation for BN layer {name!r}, "
                    f"got {len(observations)}"
                )
            batch_mean, batch_var, batch_count = observations[0]
            state = self._states[name]
            state.target_mean, state.target_m2, state.target_count = _merge_moments(
                state.target_mean,
                state.target_m2,
                state.target_count,
                batch_mean,
                batch_var,
                batch_count,
            )
        self.seen_batches += 1

    def summary(self) -> dict[str, float | int]:
        return {
            "bn_layers": len(self._states),
            "bn_seen_batches": self.seen_batches,
            "bn_target_weight": self.target_weight,
        }
