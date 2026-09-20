#!/usr/bin/env python3
"""
scripts/build_submission_package.py

Assembles the complete BMC Medical Informatics and Decision Making submission folder and the
arXiv replacement package from the manuscript source and the result files.

    SUBMISSION_PACKAGE/
      01_Manuscript/      main document (double spaced, line + page numbers, tables inside)
      02_Cover_letter/
      03_Figures/         Figure_N.tiff (300 dpi, LZW) + Figure_N.pdf (vector)
      04_Tables/          Table_N.docx, one editable table object per file
      05_Additional_files/
      arXiv/              single-spaced PDF with figures in place + metadata + ancillary files
      SUBMISSION_CHECKLIST.md

Run Manuscript/Submission/BMC_MIDM/build.sh first (it renders the numbers and the main DOCX).
"""

import re
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / 'results_clean'
PANDOC = str(Path.home() / '.local' / 'bin' / 'pandoc')
Image.MAX_IMAGE_PIXELS = None


def run(*cmd, cwd=None):
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def render(src, out, strict):
    cmd = [sys.executable, ROOT / 'scripts' / 'render_manuscript.py', src, '-o', out]
    run(*(cmd + ['--check'] if strict else cmd))


def pandoc(md, out, ms, reference=None, resource=None):
    cmd = [PANDOC, md, '--from', 'markdown+pipe_tables', '--citeproc', '--bibliography',
           ms / 'references.bib', '--csl', ms / 'biomed-central.csl', '-o', out]
    if reference:
        cmd += ['--reference-doc', reference]
    if resource:
        cmd += ['--resource-path', resource]
    run(*cmd)


def md_table(df, floatfmt='{:.3f}'):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: floatfmt.format(v))
    head = '| ' + ' | '.join(df.columns) + ' |'
    sep = '|' + '|'.join(':---' if df[c].dtype == object and not df[c].str.match(r'^-?[\d.]+$').all()
                         else '---:' for c in df.columns) + '|'
    return '\n'.join([head, sep] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.values])


def split_tables(rendered):
    """-> [(n, title, table_markdown, note)] from the rendered manuscript."""
    out = []
    lines = rendered.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'Table (\d) (.+)', lines[i])
        if m:
            j = i + 1
            while not lines[j].startswith('|'):
                j += 1
            k = j
            while k < len(lines) and lines[k].startswith('|'):
                k += 1
            n = k
            while n < len(lines) and not lines[n].strip():
                n += 1
            note = lines[n] if n < len(lines) and not lines[n].startswith('#') else ''
            out.append((int(m.group(1)), lines[i], '\n'.join(lines[j:k]), note))
            i = k
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manuscript-dir', required=True)
    a = ap.parse_args()
    ms = Path(a.manuscript_dir).resolve()
    build, pkg = ms / 'build', ms / 'SUBMISSION_PACKAGE'
    if pkg.exists():
        shutil.rmtree(pkg)
    d = {k: pkg / v for k, v in dict(ms='01_Manuscript', cl='02_Cover_letter', fig='03_Figures',
                                     tab='04_Tables', af='05_Additional_files',
                                     arx='arXiv').items()}
    for p in d.values():
        p.mkdir(parents=True)
    rendered = (build / 'manuscript_rendered.md').read_text()
    reg = json.loads((RC / 'manuscript_numbers.json').read_text())['numbers']

    # ---- 01 manuscript, 02 cover letter ------------------------------------------------
    shutil.copy(build / 'PulmoVec_BMC_MIDM.docx', d['ms'] / 'Manuscript.docx')
    pandoc(ms / 'cover_letter.md', d['cl'] / 'Cover_letter.docx', ms)

    # ---- 03 figures --------------------------------------------------------------------
    for n in range(1, 7):
        shutil.copy(ms / 'figures' / f'Fig{n}.tiff', d['fig'] / f'Figure_{n}.tiff')
        shutil.copy(ms / 'figures' / f'Fig{n}.pdf', d['fig'] / f'Figure_{n}.pdf')

    # ---- 04 tables, one file each -------------------------------------------------------
    tables = split_tables(rendered[:rendered.index('# Discussion')])
    for n, title, tab, note in tables:
        tmp = build / f'_table{n}.md'
        tmp.write_text(f'**{title}**\n\n{tab}\n\n{note}\n')
        pandoc(tmp, d['tab'] / f'Table_{n}.docx', ms, reference=ms / 'reference.docx')
        tmp.unlink()

    # ---- 05 additional files ------------------------------------------------------------
    src = ms / 'additional_files' / 'src'
    render(src / 'additional_file_1.md', build / '_af1.md', strict=False)
    pandoc(build / '_af1.md', d['af'] / 'Additional file 1.docx', ms)
    pandoc(ms / 'TRIPOD_AI.md', d['af'] / 'Additional file 2.docx', ms)

    shap_dir = RC / 'arm_L0_clean' / 'shap'
    ft = pd.read_csv(shap_dir / 'shap_feature_table.csv')
    ft['share'] = (ft['share'] * 100).round(1)
    ft = ft.rename(columns={'task': 'Outcome', 'feature': 'Feature', 'group': 'Group',
                            'mean_abs_shap': 'Mean |SHAP|', 'share': 'Share (%)'})
    share = lambda t: f"{reg[f'shap.l0.{t}.demographic_share']['value'] * 100:.1f}%"  # noqa: E731
    af4 = f"""---
title: "Additional file 4. SHAP attribution of the second-stage learners"
---

SHAP values were computed with TreeExplainer for the full-stack LightGBM models of the locked
hold-out arm (released encoder, patient-level partitioning) on its test events. Demographic
variables (age, sex, auscultation site) carried {share('screen')} of the mean absolute
attribution for screening, {share('pattern')} for sound pattern and {share('disease')} for
disease group.

![Figure S3. Share of mean absolute SHAP attribution by feature, coloured by feature group, for the three outcomes.]({shap_dir / 'shap_group_share.png'}){{width=100%}}

Table S8 Mean absolute SHAP value and share of attribution per feature

{md_table(ft)}

![Figure S4. SHAP summary plot, screening (class: adventitious).]({shap_dir / 'beeswarm_model2_label.png'}){{width=75%}}

![Figure S5. SHAP summary plot, sound pattern (class: wheeze/rhonchi).]({shap_dir / 'beeswarm_model1_label.png'}){{width=75%}}

![Figure S6. SHAP summary plot, disease group (class: pneumonia).]({shap_dir / 'beeswarm_model3_label.png'}){{width=75%}}
"""
    (build / '_af4.md').write_text(af4)
    pandoc(build / '_af4.md', d['af'] / 'Additional file 6.docx', ms)

    img = src / 'img'
    af5_src = src / 'additional_file_5.md'
    af5_src.write_text(f"""---
title: "Additional file 5. Model-derived attribution: further maps and faithfulness checks"
---

Events were sampled from the locked test partition by a fixed seeded rule (up to 150 per true
class), not selected by hand: {{{{sal.screen.n_events}}}} events for screening
({{{{sal.screen.n_correct}}}} correctly classified) and {{{{sal.pattern.n_events}}}} for sound
pattern ({{{{sal.pattern.n_correct}}}} correctly classified). <!-- numbers-ok -->

![Figure S1. Class-mean attribution on the 12 x 8 (time x mel band) patch grid of the encoder input for correctly classified events, by three methods. The dashed line marks the cut-off of the band-pass filter applied during preprocessing; bands above it contain only attenuated residual energy. Occlusion and the attention-pooling weights agree on a low band around the centre of the clip for adventitious events, whereas integrated gradients concentrates on single patches. For normal events, occlusion attributes most of the effect to bands above the filter cut-off.]({img / 'FigS1_attribution_methods.png'}){{width=100%}}

![Figure S2. Per-event faithfulness statistics for correctly classified events. a Share of positive occlusion attribution inside the annotated event against the share of the clip that the event occupies (dotted line: equality). b Correlation between occlusion maps of the trained model and of a model whose classification head was re-initialised. c Correlation between occlusion and integrated-gradients maps.]({img / 'FigS2_faithfulness.png'}){{width=100%}}

Table S9 Summary of faithfulness checks (correctly classified events)

| Measure | Screening | Sound pattern |
|:---|---:|---:|
| Area under deletion curve, attribution-ordered | {{{{sal.screen.deletion_auc_attr_mean|3}}}} | {{{{sal.pattern.deletion_auc_attr_mean|3}}}} |
| Area under deletion curve, random order | {{{{sal.screen.deletion_auc_rand_mean|3}}}} | {{{{sal.pattern.deletion_auc_rand_mean|3}}}} |
| Difference (random minus attribution-ordered) | {{{{sal.screen.deletion_gap_mean|3}}}} | {{{{sal.pattern.deletion_gap_mean|3}}}} |
| Share of attribution inside the annotated event | {{{{sal.screen.mass_in_event_mean}}}} | {{{{sal.pattern.mass_in_event_mean}}}} |
| Share of the clip occupied by the event | {{{{sal.screen.event_time_share_mean}}}} | {{{{sal.pattern.event_time_share_mean}}}} |
| Median correlation with re-initialised head | {{{{sal.screen.occ_randhead_spearman_median}}}} | {{{{sal.pattern.occ_randhead_spearman_median}}}} |
| Median correlation, occlusion vs integrated gradients | {{{{sal.screen.occ_ig_spearman_median}}}} | {{{{sal.pattern.occ_ig_spearman_median}}}} |
""")
    render(af5_src, build / '_af5.md', strict=False)   # design constants (150 per class, 12 x 8 grid)
    pandoc(build / '_af5.md', d['af'] / 'Additional file 7.docx', ms)

    af = ms / 'additional_files'
    # built as 2/3/6/8 by build_additional_files.py, renumbered here to citation order
    for src_n, new_n, ext in ((2, 3, 'csv'), (6, 4, 'xlsx'), (3, 5, 'csv'), (8, 8, 'csv')):
        shutil.copy(af / f'Additional_file_{src_n}.{ext}', d['af'] / f'Additional file {new_n}.{ext}')
    conflicts = ROOT / 'data' / 'SPRSound_Event_Level_Dataset_CLEAN.diagnosis_conflicts.csv'
    for p in build.glob('_af*.md'):
        p.unlink()

    # ---- arXiv --------------------------------------------------------------------------
    body = rendered[:rendered.index('# Figure legends')]
    legends = {m[0]: (m[1], m[2]) for m in re.findall(r'\*\*Fig\. (\d) (.+?)\*\* (.+)', rendered)}
    (d['arx'] / 'figures').mkdir()
    paras = body.split('\n\n')
    for n in map(str, range(1, 7)):
        Image.open(ms / 'figures' / f'Fig{n}.tiff').save(d['arx'] / 'figures' / f'Figure_{n}.png')
        title, text = legends[n]
        fig = f"![**Fig. {n} {title}** {text}](figures/Figure_{n}.png){{width=100%}}"
        k = next(i for i, p in enumerate(paras) if re.search(rf'Figs?\. (\d[–-])?{n}\b', p)
                 and not p.startswith(('|', 'Table')))
        paras.insert(k + 1, fig)
    arx_md = '\n\n'.join(paras) + '\n\n# References\n'
    arx_md = arx_md.replace('link-citations: true', 'link-citations: true\ndate: "Version 2 — '
                            'supersedes arXiv:2603.15688v1"')
    (d['arx'] / 'manuscript_arxiv.md').write_text(arx_md)
    pandoc(d['arx'] / 'manuscript_arxiv.md', d['arx'] / 'PulmoVec_arXiv_v2.docx', ms,
           resource=d['arx'])
    run('soffice', '--headless', '--convert-to', 'pdf', '--outdir', d['arx'],
        d['arx'] / 'PulmoVec_arXiv_v2.docx')
    anc = d['arx'] / 'anc'
    anc.mkdir()
    for f in d['af'].iterdir():
        shutil.copy(f, anc / f.name.replace(' ', '_'))
    abstract = re.sub(r'\*\*|\n+', ' ', rendered[rendered.index('# Abstract') + 10:
                                                  rendered.index('**Keywords')]).strip()
    abstract = re.sub(r'\s+', ' ', abstract)
    title = re.search(r'title: "(.+)"', rendered).group(1)
    render(ms / 'arxiv_abstract.md', build / '_arxiv_abstract.md', strict=True)
    abstract = re.sub(r'\s+', ' ', (build / '_arxiv_abstract.md').read_text()).strip()
    (build / '_arxiv_abstract.md').unlink()
    assert len(abstract) <= 1920, f'arXiv abstract too long: {len(abstract)}'
    (d['arx'] / 'arXiv_metadata.txt').write_text(f"""arXiv REPLACEMENT (v2) of arXiv:2603.15688
Use "Replace" on the existing submission - do NOT create a new submission.

Upload:      PulmoVec_arXiv_v2.pdf   (arXiv does not accept DOCX; PDF made from the DOCX here)
Ancillary:   contents of anc/ (upload as ancillary files)

Title:
{title}

Authors:
Izzet Turkalp Akbasli, Oguzhan Serin

Categories:  cs.SD (primary, as v1); cross-lists: eess.AS, cs.LG

Comments:
v2: complete re-analysis with patient-level validation; supersedes v1, whose results were obtained with an event-level split and are not valid for disease-group prediction (see Results, Leakage). 6 figures, 4 tables, 8 ancillary files. Code: https://github.com/turkalpmd/PulmoVec

Abstract ({len(abstract)} characters; arXiv limit 1920):
{abstract}
""")
    print(f"arXiv abstract length: {len(abstract)} characters")

    # ---- checklist ----------------------------------------------------------------------
    todo = sorted(set(re.findall(r'\*\*\[([^\]]+)\]\*\*', rendered + (ms / 'cover_letter.md').read_text())))
    words = None
    (pkg / 'SUBMISSION_CHECKLIST.md').write_text(f"""# Submission checklist - BMC Medical Informatics and Decision Making

Article type: Research article. Upload in this order.

| # | File | Item type in the submission system |
|---|---|---|
| 1 | 02_Cover_letter/Cover_letter.docx | Cover letter |
| 2 | 01_Manuscript/Manuscript.docx | Manuscript (double spaced, line and page numbers, tables 1-4 inside, figure legends at the end, no figures embedded) |
| 3 | 03_Figures/Figure_1.tiff ... Figure_6.tiff | Figure (one file each, in order; 170 mm, 300 dpi, LZW, all < 1 MB). Figure_N.pdf are vector alternatives |
| 4 | 05_Additional_files/Additional file 1-8 | Additional file (numbered in order of first citation) |

04_Tables/ holds each table as its own editable Word table object. BMC wants tables inside
the manuscript (they are); upload the separate files only if the editorial office asks.

## Guideline compliance
- Abstract <= 350 words, four headings; 10 keywords (3-10 allowed)
- Sections: Background, Methods, Results, Discussion, Conclusions, List of abbreviations,
  Declarations (all sub-headings present), Additional files, Figure legends, References
- References numbered in order of citation, BioMed Central style; web resources are numbered
  references with access dates; the dataset is cited in the reference list
- Tables: real table objects, no colour or shading, no thousands separators
- Figure titles <= 15 words and legends <= 300 words are in the manuscript, not in the graphics
- Every figure and additional file is cited in sequence in the text
- TRIPOD+AI checklist: Additional file 7

## Still to do by the authors before pressing submit
""" + '\n'.join(f'- [ ] {t}' for t in todo) + """
- [ ] Withdraw the earlier version from the previous journal and enter the date in the cover letter
- [ ] Tag a GitHub release, archive it on Zenodo and enter the DOI under "Availability of data and materials"
- [ ] Read the generative-AI statement and confirm it describes what was done
- [ ] Suggested reviewers: names with institutional e-mail addresses (must be real and verifiable)
- [ ] Choose the licence at acceptance (CC BY or CC BY-NC-ND) and check APC / institutional agreement
- [ ] After submission: replace arXiv:2603.15688 with arXiv/PulmoVec_arXiv_v2.pdf
""")
    print('package ->', pkg)


if __name__ == '__main__':
    main()
