"""
SPRSound PyTorch Dataset for event-level respiratory sound classification.

Features:
- Event-centered 2.0-second window (32,000 samples @ 16kHz).
- Zero neighbor contamination: isolates target event with smooth 10ms cosine tapering.
- Physiological vesicular breath background noise in margins (Option B).
- Bounded temporal jitter data augmentation for training.
- Zero-leakage StratifiedGroupKFold splitting by recording/patient.
"""

import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedGroupKFold
import warnings
warnings.filterwarnings('ignore')

import config


def generate_vesicular_breath_noise(
    length_samples: int,
    sr: int = 16000,
    f_low: float = 100.0,
    f_high: float = 600.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate soft physiological vesicular breath noise.
    Simulates acoustic air movement through pulmonary parenchyma
    by bandpass filtering Gaussian noise between 100 Hz and 600 Hz.
    """
    if length_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    gen_len = max(64, length_samples)
    if rng is None:
        noise = np.random.randn(gen_len).astype(np.float32)
    else:
        noise = rng.standard_normal(gen_len).astype(np.float32)
        
    nyq = 0.5 * sr
    low = max(0.01, f_low / nyq)
    high = min(0.99, f_high / nyq)
    b, a = butter(2, [low, high], btype='band')
    filtered = filtfilt(b, a, noise).astype(np.float32)
    return filtered[:length_samples]


def extract_isolated_centered_clip(
    audio: np.ndarray,
    sr: int,
    event_start_ms: float,
    event_end_ms: float,
    clip_duration_s: float = 2.0,
    is_training: bool = False,
    max_jitter_ms: float = 200.0,
    add_breath_noise: bool = True,
    noise_ratio: float = 0.08,
    filter_bandpass: bool = True,
    normalize_peak: bool = True,
    target_peak: float = 0.90,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Isolate the target event, center it inside a 2.0s buffer (32,000 samples @ 16kHz),
    apply smooth cosine boundary tapering, and fill empty margins with subtle vesicular
    breath noise (Option B).
    
    Guarantees:
    - 0% contamination from neighboring events.
    - Symmetric centering at t=1.0s (with bounded temporal jitter during training).
    - Zero dead-silence discontinuities.
    """
    clip_samples = int(clip_duration_s * sr) # 32,000 samples
    
    if rng is None:
        if not is_training:
            # Deterministic noise for validation/testing based on event timestamps
            seed = int(abs(event_start_ms * 1000.0 + event_end_ms)) % (2**31 - 1)
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
    
    start_sample = max(0, int(event_start_ms * sr / 1000.0))
    end_sample = min(len(audio), int(event_end_ms * sr / 1000.0))
    
    # Extract isolated target event
    if end_sample <= start_sample:
        end_sample = min(len(audio), start_sample + int(0.1 * sr))
    
    raw_event = audio[start_sample:end_sample].copy()
    event_len = len(raw_event)
    
    # 10ms smooth cosine boundary taper
    taper_len = min(event_len // 4, int(0.010 * sr)) # 160 samples @ 16kHz
    if taper_len > 4:
        taper_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_len)))
        taper_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, taper_len)))
        raw_event[:taper_len] *= taper_in
        raw_event[-taper_len:] *= taper_out
        
    out_clip = np.zeros(clip_samples, dtype=np.float32)
    
    if event_len <= clip_samples:
        # Event fits inside 2.0-second window
        ideal_left_pad = (clip_samples - event_len) // 2
        slack_left = ideal_left_pad
        slack_right = clip_samples - event_len - ideal_left_pad
        
        if is_training and (slack_left > 0 or slack_right > 0):
            # Apply temporal jitter bounded by available slack and max_jitter_ms
            max_jitter_samples = int(max_jitter_ms * sr / 1000.0)
            lim_left = min(slack_left, max_jitter_samples)
            lim_right = min(slack_right, max_jitter_samples)
            if rng is not None:
                jitter = rng.integers(-lim_left, lim_right + 1)
            else:
                jitter = np.random.randint(-lim_left, lim_right + 1)
            left_pad = ideal_left_pad + jitter
        else:
            left_pad = ideal_left_pad
            
        right_pad = clip_samples - event_len - left_pad
        out_clip[left_pad:left_pad + event_len] = raw_event
        
        # Fill margins with soft physiological breath noise
        if add_breath_noise:
            event_rms = np.sqrt(np.mean(raw_event**2)) if event_len > 0 else 0.05
            event_rms = max(1e-4, event_rms)
            
            # Left margin
            if left_pad > 0:
                noise_left = generate_vesicular_breath_noise(left_pad, sr=sr, rng=rng)
                n_rms = np.sqrt(np.mean(noise_left**2)) + 1e-6
                out_clip[:left_pad] = noise_left * (noise_ratio * event_rms / n_rms)
                
            # Right margin
            if right_pad > 0:
                noise_right = generate_vesicular_breath_noise(right_pad, sr=sr, rng=rng)
                n_rms = np.sqrt(np.mean(noise_right**2)) + 1e-6
                out_clip[left_pad + event_len:] = noise_right * (noise_ratio * event_rms / n_rms)
                
    else:
        # Event is longer than 2.0s (e.g. 2.5s breath cycle)
        excess = event_len - clip_samples
        if is_training and excess > 0:
            start_offset = rng.integers(0, excess + 1) if rng else np.random.randint(0, excess + 1)
        else:
            start_offset = excess // 2
        out_clip = raw_event[start_offset:start_offset + clip_samples]
        
    # 4th-Order Butterworth Bandpass (100 Hz - 1800 Hz) to eliminate cardiac thump & friction
    if filter_bandpass:
        nyq = 0.5 * sr
        low = max(0.01, 100.0 / nyq)
        high = min(0.99, 1800.0 / nyq)
        b, a = butter(4, [low, high], btype='band')
        out_clip = filtfilt(b, a, out_clip).astype(np.float32)
        
    # Target Peak Normalization (0.90) to prevent faint sounds from vanishing in PCEN
    if normalize_peak:
        pk = float(np.max(np.abs(out_clip)))
        if pk > 1e-5:
            out_clip = (out_clip / pk) * target_peak
            out_clip = np.clip(out_clip, -0.99, 0.99)
            
    return out_clip


class SPRSoundDataset(Dataset):
    """
    PyTorch Dataset for SPRSound event-level data with event-centering
    and Option B physiological breath margin padding.
    """
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        transform=None,
        is_training: bool = False
    ):
        if csv_path is None:
            csv_path = config.CSV_PATH
        
        self.df = pd.read_csv(csv_path)
        if dataset_filter:
            self.df = self.df[self.df['dataset'] == dataset_filter].reset_index(drop=True)
        
        self.df = self.df[self.df['wav_exists'] == 'yes'].reset_index(drop=True)
        self.class_names = config.CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        self.df['label'] = self.df['event_type'].map(self.class_to_idx)
        self.df = self.df.dropna(subset=['label']).reset_index(drop=True)
        self.transform = transform
        self.is_training = is_training
        self.rng = np.random.default_rng(42) if is_training else None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_wav_path(self, raw_wav: str) -> Path:
        wav_path = Path(raw_wav)
        if wav_path.exists():
            return wav_path
        candidates = [
            config.DATA_DIR / "wavfiles" / wav_path.name,
            config.DATA_DIR / raw_wav,
            config.PROJECT_ROOT / raw_wav,
            config.WAV_ROOT_DIR / wav_path.name,
            Path("/home/izzet/Desktop/PulmoVec_Workspace/data/wavfiles") / wav_path.name,
            Path("/home/izzet/Desktop/PulmoVec_Workspace/SPRSound") / wav_path.name
        ]
        for c in candidates:
            if c.exists():
                return c
        return config.DATA_DIR / "wavfiles" / wav_path.name

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        row = self.df.iloc[idx]
        wav_path = self._resolve_wav_path(str(row['wav_path']))
        
        event_start_ms = float(row['event_start_ms'])
        event_end_ms = float(row['event_end_ms'])
        label = int(row['label'])
        
        # Load audio (resample to 16 kHz)
        audio, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
        
        # Extract isolated, centered, breath-noise padded clip (Option B)
        event_clip = extract_isolated_centered_clip(
            audio=audio,
            sr=config.SAMPLE_RATE,
            event_start_ms=event_start_ms,
            event_end_ms=event_end_ms,
            clip_duration_s=config.CLIP_DURATION,
            is_training=self.is_training,
            max_jitter_ms=200.0,
            add_breath_noise=True,
            noise_ratio=0.08,
            rng=self.rng
        )
        
        audio_tensor = torch.from_numpy(event_clip).float()
        if self.transform:
            audio_tensor = self.transform(audio_tensor)
            
        metadata = {
            'filename': row['filename'],
            'patient_number': row.get('patient_number', ''),
            'event_type': row.get('event_type', ''),
            'event_start_ms': event_start_ms,
            'event_end_ms': event_end_ms,
            'event_duration_ms': event_end_ms - event_start_ms,
            'dataset': row.get('dataset', '')
        }
        
        return audio_tensor, label, metadata

    def get_class_weights(self) -> torch.Tensor:
        class_counts = self.df['label'].value_counts().sort_index().values
        weights = 1.0 / (class_counts + 1e-5)
        weights = weights / weights.sum() * len(class_counts)
        return torch.from_numpy(weights).float()


class SPRSoundDatasetFromDF(SPRSoundDataset):
    """
    SPRSound Dataset constructed directly from a DataFrame.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str = 'label',
        transform=None,
        is_training: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.label_column = label_column
        self.is_training = is_training
        self.transform = transform
        self.rng = np.random.default_rng(42) if is_training else None
        
        TASK_CLASS_NAMES = {
            'model1_label': ['Normal', 'Crackles', 'Rhonchi_Wheezes'],
            'model2_label': ['Normal', 'Abnormal'],
            'model3_label': ['Pneumonia', 'Bronchitis_Asthma_Bronchiolitis', 'Normal_Other'],
            'model4_label': ['Normal', 'Pneumonia', 'Bronchitis', 'Bronchiolitis']
        }
        if label_column in self.df.columns:
            self.df['label'] = self.df[label_column].astype(int)
            unique_labels = sorted(self.df['label'].unique())
            if label_column in TASK_CLASS_NAMES and len(TASK_CLASS_NAMES[label_column]) == len(unique_labels):
                self.class_names = TASK_CLASS_NAMES[label_column]
            else:
                self.class_names = [f"Class_{i}" for i in unique_labels]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        else:
            self.class_names = config.CLASS_NAMES
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            self.df['label'] = self.df['event_type'].map(self.class_to_idx)
            self.df = self.df.dropna(subset=['label']).reset_index(drop=True)



def build_patient_group_key(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic patient-level grouping key.

    385 SPRSound recordings ship with an empty patient-id field (filenames such as
    '_3.9_1_p2_18144.json'), so patient_number is NaN for 5.6% of events. Those rows
    cannot be assigned to a real patient; each such recording therefore becomes its
    own 'unknown:<filename>' group and is kept out of the evaluation partitions by
    the split functions.
    """
    pn = df['patient_number']
    known = pn.notna()
    key = pd.Series(index=df.index, dtype=object)
    key[known] = 'pid:' + pn[known].astype('int64').astype(str)
    key[~known] = 'unknown:' + df.loc[~known, 'filename'].astype(str)
    return key


def _assert_no_patient_leakage(splits: dict, df_cols) -> None:
    """
    Hard guarantee against patient-level leakage, independent of the grouping
    column used for the split. Grouping by 'filename' silently allows the same
    patient to appear in several partitions (SPRSound stores p1-p4 auscultation
    locations of one patient as separate recordings), so the patient identity is
    always re-checked here.
    """
    names = list(splits)
    for name, part in splits.items():
        if '_group_key' not in part.columns:
            raise KeyError(f"_group_key column missing in '{name}'")
        if part['_group_key'].isna().any():
            raise ValueError(f"NaN group key in '{name}' - cannot verify leakage")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = set(splits[a]['_group_key']) & set(splits[b]['_group_key'])
            assert not shared, (
                f"CRITICAL PATIENT LEAKAGE: {len(shared)} patients shared between "
                f"{a} and {b}. Split must be grouped by patient_number."
            )
    sizes = {k: len(set(v['_group_key'])) for k, v in splits.items()}
    print(f"  Patient-level leakage check: PASSED {sizes}")


MAX_AGE_YEARS = 18.0


def apply_age_exclusion(df):
    """Paediatric cohort: drop events of participants older than MAX_AGE_YEARS."""
    age = pd.to_numeric(df['age'], errors='coerce')
    over = age > MAX_AGE_YEARS
    if over.any():
        print(f"  Curation: dropped {int(over.sum())} events from "
              f"{df.loc[over, 'patient_number'].nunique()} participant(s) older than "
              f"{MAX_AGE_YEARS:.0f} years")
    return df[~over].reset_index(drop=True)


def _load_curated_frame(csv_path, label_column, drop_undiagnosed=True):
    """Load the event table, drop unusable rows, and attach the patient group key."""
    if csv_path is None:
        csv_path = config.ENSEMBLE_CSV_PATH if config.ENSEMBLE_CSV_PATH.exists() else config.CSV_PATH
    df = pd.read_csv(csv_path)
    df = df[df['wav_exists'] == 'yes'].reset_index(drop=True)
    if label_column in df.columns:
        df = df[df[label_column] >= 0].reset_index(drop=True)

    # SPRSound stores 'Unknown' where no diagnosis was recorded. The upstream label
    # map folded these into the Normal/Other disease group, so ~20% of the disease
    # target was 'diagnosis not documented' rather than a clinical category. They are
    # excluded study-wide to keep one consistent, diagnosable cohort.
    if drop_undiagnosed and 'disease' in df.columns:
        n_before = len(df)
        df = df[df['disease'] != 'Unknown'].reset_index(drop=True)
        dropped = n_before - len(df)
        if dropped:
            print(f"  Curation: dropped {dropped} events with disease='Unknown' "
                  f"({dropped / n_before * 100:.1f}%)")

    df = apply_age_exclusion(df)
    df['_group_key'] = build_patient_group_key(df)
    return df


def _split_known_unknown(df):
    """Recordings without a patient id may only be used for training."""
    unknown = df['_group_key'].str.startswith('unknown:')
    return df[~unknown].reset_index(drop=True), df[unknown].reset_index(drop=True)


def _patient_level_split(known, label_column, fractions, random_seed):
    """
    Split PATIENTS (not events) so the disease-group mix stays balanced across
    partitions. Stratifying on event rows while grouping by patient lets a few
    patients with many events dominate a stratum, which skews the patient-level
    class mix - the level clinical conclusions are actually drawn at.
    """
    from sklearn.model_selection import train_test_split

    pat = (known.groupby('_group_key')[label_column].first().reset_index()
           .rename(columns={label_column: 'y'}))

    counts = pat['y'].value_counts()
    rare = counts[counts < 3].index
    strat = pat['y'].where(~pat['y'].isin(rare), -1)

    train_frac, val_frac, test_frac = fractions
    holdout = val_frac + test_frac
    tr_pat, rest_pat = train_test_split(
        pat['_group_key'], test_size=holdout, random_state=random_seed, stratify=strat)

    if test_frac == 0:
        return set(tr_pat), set(rest_pat), set()

    rest_strat = pat.set_index('_group_key').loc[rest_pat, 'y']
    rc = rest_strat.value_counts()
    rest_strat = rest_strat.where(~rest_strat.isin(rc[rc < 2].index), -1)
    va_pat, te_pat = train_test_split(
        rest_pat, test_size=test_frac / holdout, random_state=random_seed + 1,
        stratify=rest_strat)
    return set(tr_pat), set(va_pat), set(te_pat)


def stratified_train_val_split(
    csv_path: Optional[str] = None,
    label_column: str = 'label',
    train_ratio: float = 0.8,
    random_seed: int = 42,
    group_column: str = 'patient_number'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Patient-grouped stratified split. All events of one patient stay in a single
    partition; recordings with no patient id are training-only.
    """
    df = _load_curated_frame(csv_path, label_column)
    known, unknown = _split_known_unknown(df)

    tr_pat, va_pat, _ = _patient_level_split(
        known, label_column, (train_ratio, 1.0 - train_ratio, 0.0), random_seed)

    train_df = pd.concat([known[known['_group_key'].isin(tr_pat)], unknown], ignore_index=True)
    val_df = known[known['_group_key'].isin(va_pat)].reset_index(drop=True)

    _assert_no_patient_leakage({'train': train_df, 'val': val_df}, df.columns)

    print(f"\nPatient-Grouped Stratified Split (by {group_column}):")
    print(f"  Training:   {len(train_df)} events ({len(train_df)/len(df)*100:.1f}%) "
          f"across {train_df['_group_key'].nunique()} groups "
          f"(+{len(unknown)} events from {unknown['filename'].nunique()} id-less recordings)")
    print(f"  Validation: {len(val_df)} events ({len(val_df)/len(df)*100:.1f}%) "
          f"across {val_df['_group_key'].nunique()} patients")

    return train_df, val_df


def stratified_train_val_test_split(
    csv_path: Optional[str] = None,
    label_column: str = 'label',
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
    group_column: str = 'patient_number'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3-Way Zero-Leakage Group Stratified Split (Train / Val / Test = 80:10:10).
    Guarantees that all events from the same patient/recording remain strictly in
    one partition, preventing any data leakage.
    """
    df = _load_curated_frame(csv_path, label_column)
    known, unknown = _split_known_unknown(df)

    tr_pat, va_pat, te_pat = _patient_level_split(
        known, label_column, (train_ratio, val_ratio, test_ratio), random_seed)

    # Id-less recordings are training-only so evaluation stays patient-clean
    train_df = pd.concat([known[known['_group_key'].isin(tr_pat)], unknown], ignore_index=True)
    val_df = known[known['_group_key'].isin(va_pat)].reset_index(drop=True)
    test_df = known[known['_group_key'].isin(te_pat)].reset_index(drop=True)

    _assert_no_patient_leakage({'train': train_df, 'val': val_df, 'test': test_df}, df.columns)

    print(f"\n3-Way Patient-Grouped Stratified Split (80:10:10):")
    print(f"  Training:   {len(train_df)} events ({len(train_df)/len(df)*100:.1f}%) "
          f"across {train_df['_group_key'].nunique()} groups "
          f"(+{len(unknown)} events from {unknown['filename'].nunique()} id-less recordings)")
    print(f"  Validation: {len(val_df)} events ({len(val_df)/len(df)*100:.1f}%) "
          f"across {val_df['_group_key'].nunique()} patients")
    print(f"  Testing:    {len(test_df)} events ({len(test_df)/len(df)*100:.1f}%) "
          f"across {test_df['_group_key'].nunique()} patients")

    return train_df, val_df, test_df


def event_level_train_val_test_split(
    csv_path: Optional[str] = None,
    label_column: str = 'label',
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    DELIBERATELY LEAKY comparator: events are split at random (stratified on the
    label) with no regard for patient identity, reproducing the record-wise protocol
    common in the lung-sound literature. Same curated cohort as the patient-grouped
    split. Used only to quantify leakage inflation; never for headline results.

    Returns the three partitions plus a patient-overlap summary.
    """
    from sklearn.model_selection import train_test_split

    df = _load_curated_frame(csv_path, label_column)
    holdout = val_ratio + test_ratio
    train_df, rest = train_test_split(df, test_size=holdout, random_state=random_seed,
                                      stratify=df[label_column])
    val_df, test_df = train_test_split(rest, test_size=test_ratio / holdout,
                                       random_state=random_seed + 1,
                                       stratify=rest[label_column])
    train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))

    tr_pat, tr_rec = set(train_df['_group_key']), set(train_df['filename'])
    overlap = {}
    for name, part in [('val', val_df), ('test', test_df)]:
        overlap[name] = {
            'events': int(len(part)),
            'patients': int(part['_group_key'].nunique()),
            'patients_also_in_train': int(len(set(part['_group_key']) & tr_pat)),
            'events_with_patient_in_train': int(part['_group_key'].isin(tr_pat).sum()),
            'events_with_recording_in_train': int(part['filename'].isin(tr_rec).sum()),
        }
    print(f"\nEVENT-LEVEL split (leaky comparator): "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"  test events whose patient is also in train: "
          f"{overlap['test']['events_with_patient_in_train']}/{len(test_df)}")
    return train_df, val_df, test_df, overlap


if __name__ == "__main__":
    train_df, val_df = stratified_train_val_split()
    train_dataset = SPRSoundDatasetFromDF(train_df, is_training=True)
    val_dataset = SPRSoundDatasetFromDF(val_df, is_training=False)
    
    t_audio, t_label, t_meta = train_dataset[0]
    v_audio, v_label, v_meta = val_dataset[0]
    
    print(f"\nTrain sample shape: {t_audio.shape}, label: {t_label}")
    print(f"Val sample shape:   {v_audio.shape}, label: {v_label}")
    print("Dataset module tests passed successfully!")
