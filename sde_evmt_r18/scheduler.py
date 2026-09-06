from enum import IntEnum


class TargetStage(IntEnum):
    WARMUP = 0
    EVIDENCE_BUILD = 1
    FULL = 2


def _value(stats, name, default=0):
    if stats is None:
        return default
    if isinstance(stats, dict):
        return stats.get(name, default)
    return getattr(stats, name, default)


class OnlineScheduler:
    def __init__(self, warmup_batches=5):
        self.warmup_batches = int(warmup_batches)
        self.stage = TargetStage.WARMUP

    @staticmethod
    def _reliability_not_declining(history):
        if history is None or len(history) < 3:
            return True
        first, second, third = history[-3:]
        return not (first > second > third)

    def update(self, step, candidate_stats, memory_stats, reliability_history):
        del memory_stats
        if self.stage == TargetStage.FULL:
            return self.stage
        if int(step) < self.warmup_batches:
            return self.stage
        self.stage = max(self.stage, TargetStage.EVIDENCE_BUILD)
        covered = int(_value(candidate_stats, "covered_classes", 0))
        total = int(_value(candidate_stats, "total_entries", 0))
        ready = (
            covered >= 2
            and total >= 2 * covered
            and self._reliability_not_declining(reliability_history)
        )
        if ready:
            self.stage = TargetStage.FULL
        return self.stage
