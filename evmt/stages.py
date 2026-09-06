from enum import Enum


class AdaptationStage(str, Enum):
    BN_WARMUP = "bn_warmup"
    EVIDENCE_BUILD = "evidence_build"
    FULL = "full"


class StageController:
    def __init__(
        self,
        warmup_batches,
        coverage_window,
        min_pred_classes,
        min_memory_classes,
        min_entries_per_class,
    ):
        self.warmup_batches = int(warmup_batches)
        self.coverage_window = int(coverage_window)
        self.min_pred_classes = int(min_pred_classes)
        self.min_memory_classes = int(min_memory_classes)
        self.min_entries_per_class = int(min_entries_per_class)
        self.stage = AdaptationStage.BN_WARMUP

    def update(self, step, recent_coverage, memory_coverage, min_support):
        if self.stage is AdaptationStage.BN_WARMUP:
            if (
                int(step) > self.warmup_batches
                and int(recent_coverage) >= self.min_pred_classes
            ):
                self.stage = AdaptationStage.EVIDENCE_BUILD
        elif self.stage is AdaptationStage.EVIDENCE_BUILD:
            if (
                int(memory_coverage) >= self.min_memory_classes
                and int(min_support) >= self.min_entries_per_class
            ):
                self.stage = AdaptationStage.FULL
        return self.stage
