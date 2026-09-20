#!/usr/bin/env python3
"""
scripts/render_manuscript.py

Fill {{key}} placeholders in a Markdown source from results_clean/manuscript_numbers.json
and verify that no hand-typed result number slipped in.

  {{key}}          value, 2 decimals (integers printed as integers)
  {{key|3}}        value, 3 decimals
  {{key|ci}}       "0.96 (95% CI 0.93 to 0.98)"
  {{key|ci3}}      same, 3 decimals
  {{key|t}}        compact table form "0.96 (0.93–0.98)"; {{key|t3}} with 3 decimals
  {{key|pct}}      value x 100, 1 decimal, with %
  {{key|p}}        p value ("<0.001" floor)

--check  exit 1 if, after removing placeholders, citations, headings, table/figure/file
         numbering and lines marked <!-- numbers-ok -->, any decimal number or any integer
         > 12 remains that is not listed in numbers_whitelist.txt next to the source.
"""

import re
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PH = re.compile(r'\{\{\s*([A-Za-z0-9_.]+)\s*(?:\|\s*([a-z0-9]+))?\s*\}\}')


def fmt(entry, style):
    v = entry['value']
    style = style or ''
    if style == 'pct':
        return f'{v * 100:.1f}%'
    if style == 'p':
        return '<0.001' if v < 0.001 else f'{v:.3f}'
    if isinstance(v, str):
        return v
    d = int(style[-1]) if style and style[-1].isdigit() else 2
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) > 1):
        return str(int(v))
    if style.startswith('t') and 'ci' in entry:          # compact form for table cells
        lo, hi = entry['ci']
        return f'{v:.{d}f} ({lo:.{d}f}\u2013{hi:.{d}f})'
    if style.startswith('ci'):
        lo, hi = entry['ci']
        return f'{v:.{d}f} (95% CI {lo:.{d}f} to {hi:.{d}f})'
    return f'{v:.{d}f}'


def stray_numbers(src_text, whitelist):
    bad = []
    for n, line in enumerate(src_text.splitlines(), 1):
        if '<!-- numbers-ok -->' in line or line.lstrip().startswith(('#', '<!--', '![')):
            continue
        s = re.sub(r'\{\{include:[^}]+\}\}', '', PH.sub('', line))
        s = re.sub(r'\[@[^\]]*\]|\[\d+(?:[,–-]\s*\d+)*\]', '', s)            # citations
        s = re.sub(r'(?i)\b(table|figure|fig\.|additional file|fold|phase|stage|layers?|'
                   r'model|arm l|p)\s*\d+[a-z]?', '', s)
        s = re.sub(r'\b(19|20)\d{2}\b', '', s)                               # years
        for tok in re.findall(r'(?<![\w.\-])\d+(?:[.,]\d+)?(?![\w])', s):
            val = float(tok.replace(',', '.'))
            if tok in whitelist or (tok.isdigit() and val <= 12):
                continue
            bad.append((n, tok, line.strip()[:90]))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('-o', '--output')
    ap.add_argument('--registry', default=str(ROOT / 'results_clean' / 'manuscript_numbers.json'))
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    src = Path(args.source)
    text = src.read_text()
    reg = json.loads(Path(args.registry).read_text())['numbers']

    missing = sorted({m.group(1) for m in PH.finditer(text) if m.group(1) not in reg})
    if missing:
        sys.exit("Placeholders with no registry entry:\n  " + "\n  ".join(missing))

    wl_file = src.with_name('numbers_whitelist.txt')
    whitelist = set(wl_file.read_text().split()) if wl_file.exists() else set()
    bad = stray_numbers(text, whitelist)
    for n, tok, line in bad:
        print(f"  line {n}: '{tok}' not from registry  | {line}")
    if bad and args.check:
        sys.exit(f"{len(bad)} hand-typed numbers found")

    rendered = PH.sub(lambda m: fmt(reg[m.group(1)], m.group(2)).replace('-', '\u2212'), text)
    # {{include:relative/path.md}} - generated tables (numbers come from result files)
    rendered = re.sub(r'\{\{include:([^}]+)\}\}',
                      lambda m: (src.parent / m.group(1).strip()).read_text(), rendered)
    out = Path(args.output) if args.output else src.with_name('build') / (src.stem + '_rendered.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    print(f"rendered {len(PH.findall(text))} placeholders -> {out}")


if __name__ == '__main__':
    main()
