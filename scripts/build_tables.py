#!/usr/bin/env python3
"""
scripts/build_tables.py

Journal-format versions of generated tables (BMC: no thousands separators, no shading,
real table objects). Table 1 is derived from results_clean/table1_cohort.md, which
generate_cohort_table.py writes directly from the curated event table.
"""

import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITES = {'p1': 'Left posterior (p1)', 'p2': 'Left lateral (p2)',
         'p3': 'Right posterior (p3)', 'p4': 'Right lateral (p4)'}


def table1(src, out):
    rows = [ln for ln in src.read_text().splitlines() if ln.startswith('|')]
    cells = [[c.strip() for c in ln.strip('|').split('|')] for ln in rows]
    keep = [r[:6] for r in cells if 'id-less' not in r[0]]
    keep[0][5] = 'p (train vs test)'
    body = []
    for i, r in enumerate(keep):
        if i == 1:
            body.append('|' + '|'.join([':---'] + ['---:'] * 5) + '|')
            continue
        r = [re.sub(r'(?<=\d),(?=\d{3})', '', c) for c in r]
        r[0] = SITES.get(r[0], r[0]).replace('Recording location', 'Auscultation site')
        body.append('| ' + ' | '.join(r) + ' |')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(body) + '\n')
    print(f"table 1 -> {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    table1(ROOT / 'results_clean' / 'table1_cohort.md', Path(a.out_dir) / 'table1.md')
