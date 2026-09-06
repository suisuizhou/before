#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean

TASKS = [(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)]


def grab(text: str, pattern: str):
    m = re.findall(pattern, text)
    return float(m[-1]) if m else None


def parse_log(path: Path):
    text = path.read_text(errors='replace')
    return {
        'before': grab(text, r'Beginning Acc T\s*=\s*([0-9.]+)%'),
        'online': grab(text, r'Strict Online Acc\s*=\s*([0-9.]+)%'),
        'post': grab(text, r'Post-stream Full-Target Acc\s*=\s*([0-9.]+)%'),
        'online_f1': grab(text, r'Strict Online Macro P/R/F1\s*=\s*[0-9.]+/[0-9.]+/([0-9.]+)%'),
        'post_f1': grab(text, r'Post-stream Macro P/R/F1\s*=\s*[0-9.]+/[0-9.]+/([0-9.]+)%'),
    }


def fmt(v):
    return 'NA' if v is None else f'{v:.2f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--before-tolerance', dest='before_tolerance', type=float, default=0.01)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    before_tolerance = float(args.before_tolerance)
    rows = []
    mismatch = []
    for s,t in TASKS:
        d = parse_log(run_dir / f'dtcc_{s}to{t}.log')
        q = parse_log(run_dir / f'0711_{s}to{t}.log')
        if d['before'] is None or q['before'] is None:
            raise RuntimeError(f'Missing Before metric for task [{s},{t}]')
        delta = abs(d['before'] - q['before'])
        if delta > before_tolerance:
            mismatch.append((s,t,d['before'],q['before'],delta))
        rows.append(('DTCC',s,t,d))
        rows.append(('0711_FULL',s,t,q))
    if mismatch:
        for item in mismatch:
            print('[BEFORE_MISMATCH]', item)
        raise RuntimeError(f'BEFORE_MISMATCH: {len(mismatch)} tasks exceed tolerance {before_tolerance}')

    header = f"{'Method':<14} {'Task':<8} {'Before':>9} {'Online':>9} {'Post':>9} {'OnlineF1':>10} {'PostF1':>9}"
    lines = [header]
    for method in ('DTCC','0711_FULL'):
        method_rows = [r for r in rows if r[0] == method]
        for _,s,t,m in method_rows:
            lines.append(f"{method:<14} {'['+str(s)+','+str(t)+']':<8} {fmt(m['before']):>9} {fmt(m['online']):>9} {fmt(m['post']):>9} {fmt(m['online_f1']):>10} {fmt(m['post_f1']):>9}")
        vals = lambda k: [r[3][k] for r in method_rows if r[3][k] is not None]
        lines.append(f"{method:<14} {'MEAN':<8} {mean(vals('before')):>9.2f} {mean(vals('online')):>9.2f} {mean(vals('post')):>9.2f} {fmt(mean(vals('online_f1')) if vals('online_f1') else None):>10} {fmt(mean(vals('post_f1')) if vals('post_f1') else None):>9}")
    vanilla_before = mean([r[3]['before'] for r in rows if r[0] == 'DTCC'])
    dtcc_online = mean([r[3]['online'] for r in rows if r[0] == 'DTCC'])
    q_online = mean([r[3]['online'] for r in rows if r[0] == '0711_FULL'])
    lines += [
        '',
        f'VANILLA_SOURCE MeanBefore = {vanilla_before:.2f}',
        f'DTCC on Vanilla Source MeanOnline = {dtcc_online:.2f}  Gain={dtcc_online-vanilla_before:+.2f}',
        f'0711 on Vanilla Source MeanOnline = {q_online:.2f}  Gain={q_online-vanilla_before:+.2f}',
        f'0711 - DtCC MeanOnline = {q_online-dtcc_online:+.2f}',
        '[BEFORE CHECK PASS] DtCC and 0711 use identical per-task Before accuracies.',
    ]
    out = '\n'.join(lines)
    print(out)
    (run_dir / 'summary_vanilla_common.txt').write_text(out + '\n')


if __name__ == '__main__':
    main()
