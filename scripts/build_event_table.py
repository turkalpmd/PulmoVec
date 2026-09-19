#!/usr/bin/env python3
"""
scripts/build_event_table.py

Builds the event-level table (data/SPRSound_Event_Level_Dataset_CLEAN.csv) from a local copy
of the SPRSound repository - one row per annotated respiratory event.

Sources (json dir -> wav dir), as released by the SPRSound authors:
  Classification/train, Classification/valid 2022 (inter / intra) and 2023,
  Detection/test2024, BioCAS2025/test2025.
The Detection train/valid material duplicates the Classification recordings and is not read.

File names encode  <patient>_<age>_<gender 0=male,1=female>_<site p1-p4>_<recording no>.
Recording-level diagnosis comes from the three "Patient Summary" tables. Where they
disagree, Grand Challenge'24 takes precedence, then the TBioCAS summary, then Grand
Challenge'23; every disagreement is written to <out>.diagnosis_conflicts.csv. Participants
found in no summary get disease = 'Unknown' (excluded downstream by the curation step).
Recordings with an empty event list yield a single 'No Event' row (also excluded downstream).
"""

import json
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SOURCES = [  # (dataset name, json sub-dir, wav sub-dir)
    ('Classification-Train', 'Classification/train_classification_json',
     'Classification/train_classification_wav'),
    ('Classification-Valid-2022-Inter', 'Classification/valid_classification_json/2022/inter_test_json',
     'Classification/valid_classification_wav/2022'),
    ('Classification-Valid-2022-Intra', 'Classification/valid_classification_json/2022/intra_test_json',
     'Classification/valid_classification_wav/2022'),
    ('Classification-Valid-2023', 'Classification/valid_classification_json/2023',
     'Classification/valid_classification_wav/2023'),
    ('Detection-Test-2024', 'Detection/test2024_detection_json',
     'Detection/test2024_detection_wav'),
    ('BioCAS2025-Test', 'BioCAS2025/test2025_json', 'BioCAS2025/test2025_wav'),
]
SUMMARIES = ["Grand_Challenge'24_patient_summary.csv", 'SPRSound_patient_summary.csv',
             "Grand_Challenge'23_patient_summary.csv"]          # precedence order
SITE_NAMES = {'p1': 'Left Posterior', 'p2': 'Left Lateral', 'p3': 'Right Posterior',
              'p4': 'Right Lateral'}
DISEASE_ALIASES = {'control group': 'Control Group',
                   'other respiratory': 'Other respiratory diseases', '-': None}


def norm_disease(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return DISEASE_ALIASES.get(x.lower(), x)


def load_diagnoses(summary_dir, conflicts_path):
    rows = []
    for rank, name in enumerate(SUMMARIES):
        t = pd.read_csv(summary_dir / name, dtype={'patient_num': str})
        t['disease'] = t['disease'].map(norm_disease)
        t = t.dropna(subset=['disease'])
        t['pid'] = t['patient_num'].str.lstrip('0')
        rows.append(t[['pid', 'disease']].assign(source=name, rank=rank))
    allrows = pd.concat(rows, ignore_index=True)
    n_labels = allrows.groupby('pid')['disease'].nunique()
    allrows[allrows['pid'].isin(n_labels[n_labels > 1].index)].sort_values(
        ['pid', 'rank']).to_csv(conflicts_path, index=False)
    print(f"  diagnoses: {allrows['pid'].nunique()} participants, "
          f"{int((n_labels > 1).sum())} with conflicting labels -> {conflicts_path.name}")
    return allrows.sort_values('rank').drop_duplicates('pid').set_index('pid')['disease']


def find_dir(roots, sub):
    for r in roots:
        if (r / sub).is_dir():
            return r / sub
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sprsound-root', nargs='+',
                    default=[str(ROOT / 'SPRSound-main'), str(ROOT / 'SPRSound')],
                    help='one or more local copies of the SPRSound repository')
    ap.add_argument('--out', default=str(ROOT / 'data' / 'SPRSound_Event_Level_Dataset_CLEAN.csv'))
    args = ap.parse_args()
    roots = [Path(r) for r in args.sprsound_root]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary_dir = find_dir(roots, 'Patient Summary')
    assert summary_dir, "Patient Summary directory not found"
    diagnosis = load_diagnoses(summary_dir, out.with_suffix('.diagnosis_conflicts.csv'))

    rows = []
    for dataset, jsub, wsub in SOURCES:
        jdir, wdir = find_dir(roots, jsub), find_dir(roots, wsub)
        if jdir is None:
            print(f"  [missing] {dataset}: {jsub}")
            continue
        files = sorted(jdir.glob('*.json'))
        for jf in files:
            pid, age, sex, site, rec = jf.stem.split('_')
            ann = json.loads(jf.read_text())
            events = ann.get('event_annotation') or []
            wav = (wdir / f'{jf.stem}.wav') if wdir else None
            base = {
                'dataset': dataset, 'file_path': str(jf), 'filename': jf.name,
                'patient_number': float(pid) if pid else float('nan'),
                'age': float(age), 'gender': 'Female' if sex == '1' else 'Male',
                'gender_code': int(sex), 'recording_location': site,
                'recording_location_name': SITE_NAMES.get(
                    site, f'Additional Location {site[1:]}'),
                'recording_number': int(rec),
                'record_annotation': ann.get('record_annotation', 'Unknown'),
                'disease': diagnosis.get(pid.lstrip('0'), 'Unknown') if pid else 'Unknown',
                'total_events_in_file': len(events),
                'wav_path': str(wav) if wav else '',
                'wav_exists': 'yes' if wav is not None and wav.exists() else 'no',
            }
            if not events:
                rows.append({**base, 'event_start_ms': float('nan'), 'event_end_ms': float('nan'),
                             'event_duration_ms': float('nan'), 'event_type': 'No Event',
                             'event_index': 0})
            for i, ev in enumerate(events, start=1):
                s, e = float(ev['start']), float(ev['end'])
                rows.append({**base, 'event_start_ms': s, 'event_end_ms': e,
                             'event_duration_ms': e - s, 'event_type': ev['type'],
                             'event_index': i})
        print(f"  {dataset:32s} {len(files):5d} recordings")

    cols = ['dataset', 'file_path', 'filename', 'patient_number', 'age', 'gender', 'gender_code',
            'recording_location', 'recording_location_name', 'recording_number',
            'record_annotation', 'disease', 'event_start_ms', 'event_end_ms',
            'event_duration_ms', 'event_type', 'event_index', 'total_events_in_file',
            'wav_path', 'wav_exists']
    df = pd.DataFrame(rows)[cols]
    df.to_csv(out, index=False)
    print(f"{len(df)} events, {df['filename'].nunique()} recordings, "
          f"{df['patient_number'].nunique()} participants with an id -> {out}")


if __name__ == '__main__':
    main()
