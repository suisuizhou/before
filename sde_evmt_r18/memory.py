from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    feature: torch.Tensor
    q: torch.Tensor
    reliability: float
    step: int


@dataclass(frozen=True)
class MemoryStats:
    total_entries: int
    covered_classes: int
    min_positive_size: int
    max_size: int
    sizes: tuple


class ClassBalancedMemory:
    def __init__(self, num_classes, capacity, feature_dim):
        self.num_classes = int(num_classes)
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.queues = [deque(maxlen=self.capacity) for _ in range(self.num_classes)]

    def add(self, features, q, reliability, mask, step=0):
        if len(features) != len(q) or len(q) != len(reliability) or len(mask) != len(q):
            raise ValueError("memory inputs must have matching batch dimensions")
        normalized = F.normalize(features.detach(), dim=1)
        pseudo = q.detach().argmax(dim=1)
        for row in mask.detach().nonzero(as_tuple=False).flatten().tolist():
            label = int(pseudo[row])
            self.queues[label].append(
                MemoryEntry(
                    feature=normalized[row].cpu(),
                    q=q[row].detach().cpu(),
                    reliability=float(reliability[row]),
                    step=int(step),
                )
            )

    def stats(self):
        sizes = tuple(len(queue) for queue in self.queues)
        positive = [size for size in sizes if size]
        return MemoryStats(
            total_entries=sum(sizes),
            covered_classes=len(positive),
            min_positive_size=min(positive, default=0),
            max_size=max(sizes, default=0),
            sizes=sizes,
        )

    def _stack(self):
        entries = [
            (label, entry)
            for label, queue in enumerate(self.queues)
            for entry in queue
        ]
        if not entries:
            return (
                torch.empty(0, self.feature_dim),
                torch.empty(0, self.num_classes),
                torch.empty(0),
                torch.empty(0, dtype=torch.long),
            )
        return (
            torch.stack([entry.feature for _, entry in entries]),
            torch.stack([entry.q for _, entry in entries]),
            torch.tensor([entry.reliability for _, entry in entries]),
            torch.tensor([label for label, _ in entries], dtype=torch.long),
        )

    def prototypes(self):
        features, _, reliability, classes = self._stack()
        values = []
        present = []
        for label in range(self.num_classes):
            selected = classes == label
            if bool(selected.any()):
                weight = reliability[selected].clamp_min(1e-8)
                value = (features[selected] * weight[:, None]).sum(dim=0) / weight.sum()
                values.append(F.normalize(value, dim=0))
                present.append(label)
        if not values:
            return torch.empty(0, self.feature_dim), torch.empty(0, dtype=torch.long)
        return torch.stack(values), torch.tensor(present, dtype=torch.long)

    def neighbors(self, features, k):
        bank, q, reliability, _ = self._stack()
        if not len(bank):
            return (
                features.new_empty((len(features), 0)),
                features.new_empty((len(features), 0, self.num_classes)),
                features.new_empty((len(features), 0)),
            )
        bank = bank.to(features)
        q = q.to(features)
        reliability = reliability.to(features)
        similarity = F.normalize(features, dim=1) @ bank.T
        values, indices = similarity.topk(min(int(k), len(bank)), dim=1)
        return values, q[indices], reliability[indices]
