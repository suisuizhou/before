"""Canonical result records for strict WTPG source-only and TTA runs."""

import json
import math

import torch


def build_wtpg_result_record(*, result_kind, route, variant, task,
                             source_checkpoint_sha256, config_sha256, candidate_id,
                             source_seed, stream_seed, beginning, strict_online,
                             post_stream, confusion_matrix, samples, batches, passes,
                             finite_losses, trainable_parameters, pre_update_scoring,
                             metadata_evidence_used, runtime_seconds, peak_memory_mb,
                             **diagnostics):
    if result_kind not in {"beginning", "target"}:
        raise ValueError("invalid WTPG result kind")
    if route not in {"dtcc_ordinary", "0711_robust", "0711_common"}:
        raise ValueError("invalid WTPG route")
    if variant not in {"ordinary", "robust"}:
        raise ValueError("invalid WTPG source variant")
    source, target = map(int, task)
    if source == target or source not in range(8) or target not in range(8):
        raise ValueError("invalid WTPG task")
    matrix = None if confusion_matrix is None else torch.as_tensor(confusion_matrix, dtype=torch.long)
    if matrix is not None and (matrix.shape != (5, 5) or bool((matrix < 0).any())):
        raise ValueError("WTPG confusion matrix must be non-negative 5x5")
    metrics = {"macro_precision": None, "macro_recall": None, "macro_f1": None}
    if matrix is not None:
        ps, rs, fs = [], [], []
        for class_id in range(5):
            tp = int(matrix[class_id, class_id])
            fp = int(matrix[:, class_id].sum()) - tp
            fn = int(matrix[class_id, :].sum()) - tp
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            ps.append(p); rs.append(r); fs.append(2 * p * r / max(p + r, 1e-12))
        metrics = {
            "macro_precision": 100.0 * sum(ps) / 5,
            "macro_recall": 100.0 * sum(rs) / 5,
            "macro_f1": 100.0 * sum(fs) / 5,
        }
    record = {
        "schema_version": 1, "dataset": "WTPGStrict", "result_kind": result_kind,
        "route": route, "variant": variant, "source": source, "target": target,
        "task": [source, target], "source_checkpoint_sha256": str(source_checkpoint_sha256),
        "config_sha256": str(config_sha256), "candidate_id": str(candidate_id),
        "source_seed": int(source_seed), "stream_seed": int(stream_seed),
        "beginning": float(beginning),
        "strict_online": None if strict_online is None else float(strict_online),
        "post_stream": None if post_stream is None else float(post_stream),
        **metrics, "confusion_matrix": None if matrix is None else matrix.tolist(),
        "samples": int(samples), "batches": int(batches), "passes": int(passes),
        "finite_losses": bool(finite_losses),
        "trainable_parameters": sorted(str(name) for name in trainable_parameters),
        "trainable_allowlist": sorted(str(name) for name in trainable_parameters),
        "pre_update_scoring": bool(pre_update_scoring),
        "metadata_evidence_used": bool(metadata_evidence_used),
        "runtime_seconds": float(runtime_seconds), "peak_memory_mb": float(peak_memory_mb),
    }
    record.update(diagnostics)
    if not all(math.isfinite(value) for value in (record["beginning"], record["runtime_seconds"], record["peak_memory_mb"])):
        raise ValueError("non-finite WTPG result")
    return record


def print_wtpg_result(record):
    print("WTPG_RESULT_JSON=" + json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))

