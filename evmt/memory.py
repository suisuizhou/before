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


class ClassBalancedMemory:
    def __init__(self, num_classes, capacity_per_class, feature_dim):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.queues = [deque(maxlen=int(capacity_per_class))
                       for _ in range(self.num_classes)]

    def add(self, features, q, reliability, mask, step):
        features = F.normalize(features.detach(), dim=1)
        pseudo = q.detach().argmax(1)
        for i in mask.detach().nonzero(as_tuple=False).flatten().tolist():
            cls = int(pseudo[i])
            self.queues[cls].append(MemoryEntry(
                features[i].cpu(), q[i].detach().cpu(),
                float(reliability[i]), int(step)))

    def stats(self):
        sizes = [len(queue) for queue in self.queues]
        positive = [size for size in sizes if size > 0]
        return MemoryStats(
            total_entries=sum(sizes),
            covered_classes=len(positive),
            min_positive_size=min(positive, default=0),
            max_size=max(sizes, default=0),
        )

    def _stack(self):
        items = [(cls, entry) for cls, queue in enumerate(self.queues)
                 for entry in queue]
        if not items:
            return (torch.empty(0, self.feature_dim),
                    torch.empty(0, self.num_classes), torch.empty(0),
                    torch.empty(0, dtype=torch.long))
        return (torch.stack([e.feature for _, e in items]),
                torch.stack([e.q for _, e in items]),
                torch.tensor([e.reliability for _, e in items]),
                torch.tensor([cls for cls, _ in items]))

    def prototypes(self):
        features, _, reliability, classes = self._stack()
        prototypes, present = [], []
        for cls in range(self.num_classes):
            selected = classes == cls
            if selected.any():
                w = reliability[selected].clamp_min(1e-8)
                value = (features[selected]*w[:, None]).sum(0)/w.sum()
                prototypes.append(F.normalize(value, dim=0))
                present.append(cls)
        if not prototypes:
            return torch.empty(0, self.feature_dim), torch.empty(0, dtype=torch.long)
        return torch.stack(prototypes), torch.tensor(present)

    def neighbors(self, features, k):
        bank, q, reliability, _ = self._stack()
        if len(bank) == 0:
            return (features.new_empty((len(features), 0)),
                    features.new_empty((len(features), 0, self.num_classes)),
                    features.new_empty((len(features), 0)))
        bank, q, reliability = bank.to(features.device), q.to(features.device), reliability.to(features.device)
        similarity = F.normalize(features, dim=1) @ bank.T
        values, indices = similarity.topk(min(int(k), len(bank)), dim=1)
        return values, q[indices], reliability[indices]
