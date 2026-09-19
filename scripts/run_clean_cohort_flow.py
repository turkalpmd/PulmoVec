#!/usr/bin/env python3
"""
scripts/run_clean_cohort_flow.py

Counts for the participant / event flow diagram, derived from the same tables and the same
curation rules as the analysis (src/sprsound_dataset._load_curated_frame).
"""

import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import config  # noqa: E402
from sprsound_dataset import build_patient_group_key  # noqa: E402


def count(df):
    key = build_patient_group_key(df)
    return {'events': int(len(df)), 'recordings': int(df['filename'].nunique()),
            'patients_with_id': int(key[key.str.startswith('pid:')].nunique()),
            'events_without_patient_id': int(key.str.startswith('unknown:').sum()),
            'recordings_without_patient_id': int(
                df.loc[key.str.startswith('unknown:'), 'filename'].nunique())}


def main():
    flow = {}
    raw = pd.read_csv(config.CSV_PATH)
    flow['1_annotated_events_with_audio'] = count(raw[raw['wav_exists'] == 'yes'])
    flow['1_event_types'] = raw['event_type'].value_counts().to_dict()

    ens = pd.read_csv(config.ENSEMBLE_CSV_PATH)
    ens = ens[ens['wav_exists'] == 'yes']
    flow['2_after_label_mapping'] = count(ens)
    flow['2_excluded_no_task_label'] = (flow['1_annotated_events_with_audio']['events']
                                        - flow['2_after_label_mapping']['events'])
    valid = ens[ens['model3_label'] >= 0]
    unknown = valid[valid['disease'] == 'Unknown']
    flow['3_excluded_undocumented_diagnosis'] = count(unknown)
    documented = valid[valid['disease'] != 'Unknown']
    implausible = pd.to_numeric(documented['age'], errors='coerce') > 18
    flow['3b_excluded_implausible_age'] = count(documented[implausible])
    cohort = documented[~implausible]
    flow['4_analysis_cohort'] = count(cohort)
    flow['4_by_source_partition'] = cohort['dataset'].value_counts().to_dict()
    flow['4_events_per_patient'] = (cohort.groupby(build_patient_group_key(cohort)).size()
                                    .describe()[['min', '25%', '50%', '75%', 'max']]
                                    .round(1).to_dict())
    flow['4_recordings_per_patient'] = (cohort.groupby(build_patient_group_key(cohort))['filename']
                                        .nunique().describe()[['min', '25%', '50%', '75%', 'max']]
                                        .round(1).to_dict())

    out = ROOT / 'results_clean' / 'cohort_flow.json'
    out.write_text(json.dumps(flow, indent=2))
    print(json.dumps(flow, indent=2))


if __name__ == '__main__':
    main()
