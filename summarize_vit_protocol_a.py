#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""Summarize Protocol-A target logs into CSV and a compact text table."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

RESULT_RE = re.compile(
    r"^\[RESULT\]\s+method=(?P<method>\S+)\s+task=(?P<task>\[[^\]]+\])\s+"
    r"before=(?P<before>\S+)\s+online=(?P<online>\S+)\s+post=(?P<post>\S+)\s+"
    r"online_f1=(?P<online_f1>\S+)\s+post_f1=(?P<post_f1>\S+)\s+"
    r"batch_ms=(?P<batch_ms>\S+)\s+peak_mb=(?P<peak_mb>\S+)"
)

FIELDS = ["method", "task", "before", "online", "post", "online_f1", "post_f1", "batch_ms", "peak_mb"]
NUMERIC = FIELDS[2:]
DIAG_RE = re.compile(r"mean_batch_ms=(?P<batch_ms>[0-9.]+)\s*\|\s*peak_memory_mb=(?P<peak_mb>[0-9.]+)")


def _number(token: str):
    if token.upper() == "NA":
        return None
    return float(token)


def parse_result_line(line: str):
    match = RESULT_RE.match(line.strip())
    if not match:
        return None
    row = match.groupdict()
    for key in NUMERIC:
        row[key] = _number(row[key])
    return row


def collect(run_dir: Path):
    rows = []
    seen = set()
    for path in sorted(Path(run_dir).rglob("*.log")):
        latest_diag = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            diag = DIAG_RE.search(line)
            if diag:
                latest_diag = {k: float(v) for k, v in diag.groupdict().items()}
            row = parse_result_line(line)
            if row is None:
                continue
            if latest_diag is not None:
                if row["batch_ms"] is None:
                    row["batch_ms"] = latest_diag["batch_ms"]
                if row["peak_mb"] is None:
                    row["peak_mb"] = latest_diag["peak_mb"]
            key = (row["method"], row["task"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda r: (r["method"], r["task"]))
    return rows


def mean_available(rows, key):
    values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
    return sum(values) / len(values) if values else None


def fmt(v, width=10):
    return ("NA" if v is None else f"{v:.2f}").rjust(width)


def write_outputs(run_dir: Path, rows):
    csv_path = Path(run_dir) / "summary_vit_protocol_a.csv"
    txt_path = Path(run_dir) / "summary_vit_protocol_a.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("NA" if row[k] is None else row[k]) for k in FIELDS})

    lines = []
    header = f"{'Method':<16} {'Task':<8} {'Before':>10} {'Online':>10} {'Post':>10} {'OnlineF1':>10} {'PostF1':>10}"
    lines.append(header)
    for r in rows:
        lines.append(
            f"{r['method']:<16} {r['task']:<8} {fmt(r['before'])} {fmt(r['online'])} {fmt(r['post'])} "
            f"{fmt(r['online_f1'])} {fmt(r['post_f1'])}"
        )
    lines.append("")
    lines.append(f"{'Method':<16} {'MeanBefore':>12} {'MeanOnline':>12} {'MeanPost':>12}")
    for method in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        lines.append(
            f"{method:<16} {fmt(mean_available(group,'before'),12)} {fmt(mean_available(group,'online'),12)} "
            f"{fmt(mean_available(group,'post'),12)}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, txt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = collect(args.run_dir)
    if not rows:
        raise SystemExit(f"No [RESULT] lines found below {args.run_dir}")
    csv_path, txt_path = write_outputs(args.run_dir, rows)
    print(txt_path.read_text(encoding="utf-8"))
    print(f"[SUMMARY] CSV={csv_path}")


if __name__ == "__main__":
    main()
