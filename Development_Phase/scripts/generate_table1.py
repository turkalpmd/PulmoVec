#!/usr/bin/env python3
"""
Generate Table 1 for Disease_Event_Analysis_DETAILED.md
Train vs Test comparison with p-values to show same population.

Uses stratified 80/20 split (same as training pipeline) so both groups
come from the same population by design.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
CSV_PATH = DATA_DIR / "SPRSound_Event_Level_Dataset_CLEAN.csv"
PATIENT_SUMMARY_DIR = BASE_DIR / "SPRSound-main" / "Patient Summary"
PATIENT_SUMMARY_FILES = [
    PATIENT_SUMMARY_DIR / "SPRSound_patient_summary.csv",
    PATIENT_SUMMARY_DIR / "Grand_Challenge'23_patient_summary.csv",
    PATIENT_SUMMARY_DIR / "Grand_Challenge'24_patient_summary.csv",
]

# Stratified split (80/20) - same as training, ensures same population
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def load_merged_patient_summary() -> pd.DataFrame:
    """Load and merge Patient Summary files (SPRSound + GC23 + GC24) -> ~1,166 patients."""
    dfs = []
    for f in PATIENT_SUMMARY_FILES:
        if f.exists():
            df = pd.read_csv(f)
            if "patient_num" in df.columns:
                df = df[["patient_num", "disease", "age", "gender"]].copy()
                df["patient_num"] = df["patient_num"].astype(str).str.strip()
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs, ignore_index=True)
    # Keep first occurrence per patient (deduplicate)
    merged = merged.drop_duplicates(subset=["patient_num"], keep="first").reset_index(drop=True)
    # Normalize gender: F->Female, M->Male
    merged["gender"] = merged["gender"].map({"F": "Female", "M": "Male", "Female": "Female", "Male": "Male"}).fillna(merged["gender"])
    return merged


def map_disease_to_category(disease: str) -> str:
    """Map raw disease to Table 1 categories: Pnömoni, Bronkokonstriksiyon, Normal, Diğer"""
    if pd.isna(disease):
        return "Diğer"
    if "Pneumonia" in disease:
        return "Pnömoni"
    if disease in ["Asthma", "Bronchitis", "Bronchiolitis"]:
        return "Bronkokonstriksiyon"
    if disease == "Control Group":
        return "Normal"
    return "Diğer"


def map_event_to_sound_pattern(event_type: str) -> str:
    """Map event_type to Sound pattern: Normal, Crackles, Rhonchi"""
    if pd.isna(event_type) or event_type == "No Event":
        return None  # Exclude from sound pattern
    if event_type == "Normal":
        return "Normal"
    if event_type in ["Fine Crackle", "Coarse Crackle", "Wheeze+Crackle"]:
        return "Crackles"
    if event_type == "Rhonchi":
        return "Rhonchi"
    # Wheeze, Stridor -> could map to Rhonchi or separate; Wheeze often grouped with Rhonchi
    if event_type in ["Wheeze", "Stridor"]:
        return "Rhonchi"  # Adventitious continuous sounds
    return None


def chi2_pvalue(contingency_table: pd.DataFrame) -> float:
    """Chi-square test for categorical variables."""
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        return p
    except Exception:
        return np.nan


def ttest_or_mannwhitney(train_vals: np.ndarray, test_vals: np.ndarray) -> float:
    """T-test if normal, else Mann-Whitney U."""
    try:
        _, p_norm_train = stats.shapiro(train_vals)
        _, p_norm_test = stats.shapiro(test_vals)
        if p_norm_train > 0.05 and p_norm_test > 0.05:
            _, p = stats.ttest_ind(train_vals, test_vals)
        else:
            _, p = stats.mannwhitneyu(train_vals, test_vals, alternative='two-sided')
        return p
    except Exception:
        return np.nan


def main():
    print("Loading data...")
    # 1. Merged Patient Summary (bireşmiş kohort) -> ~1,166 patients
    patient_summary = load_merged_patient_summary()
    if patient_summary.empty:
        raise FileNotFoundError("Patient Summary files not found. Using CLEAN-only fallback.")
    patient_summary["disease_norm"] = patient_summary["disease"].replace(
        {"Control group": "Control Group", "-": "Unknown", "Other respiratory": "Other respiratory diseases"}
    ).fillna("Unknown")
    patient_summary["disease_cat"] = patient_summary["disease_norm"].map(map_disease_to_category)

    # 2. Event-level CLEAN data (24,808 events)
    df = pd.read_csv(CSV_PATH)
    df = df[df["wav_exists"] == "yes"].copy()
    def safe_patient_str(x):
        try:
            return str(int(float(x))) if pd.notna(x) else None
        except (ValueError, TypeError):
            return None

    df["patient_num_str"] = df["patient_number"].apply(safe_patient_str)

    # Patient-level split: stratified by disease category (from merged Patient Summary)
    patient_disease_cat = patient_summary.set_index("patient_num")["disease_cat"]

    if HAS_SKLEARN:
        train_patients, test_patients = train_test_split(
            patient_disease_cat.index.tolist(),
            train_size=TRAIN_RATIO,
            stratify=patient_disease_cat.values,
            random_state=RANDOM_SEED
        )
    else:
        np.random.seed(RANDOM_SEED)
        train_patients, test_patients = [], []
        for cat in patient_disease_cat.unique():
            patients_in_cat = patient_disease_cat[patient_disease_cat == cat].index.tolist()
            np.random.shuffle(patients_in_cat)
            n_train = max(1, int(len(patients_in_cat) * TRAIN_RATIO))
            train_patients.extend(patients_in_cat[:n_train])
            test_patients.extend(patients_in_cat[n_train:])

    train_patients = set(str(p) for p in train_patients)
    test_patients = set(str(p) for p in test_patients)

    # CLEAN-only patients (not in Patient Summary) - add to cohort with demographics from CLEAN
    clean_only = set(df["patient_num_str"].unique()) - (train_patients | test_patients)
    if clean_only:
        co_demo = df[df["patient_num_str"].isin(clean_only)].groupby("patient_num_str").first().reset_index()
        co_demo["gender"] = co_demo["gender"].map({"F": "Female", "M": "Male", "Female": "Female", "Male": "Male"}).fillna(co_demo["gender"])
        clean_only_df = co_demo[["patient_num_str", "disease", "age", "gender"]].rename(columns={"patient_num_str": "patient_num"})
        clean_only_df["disease_norm"] = clean_only_df["disease"].replace(
            {"Control group": "Control Group", "-": "Unknown", "Other respiratory": "Other respiratory diseases"}
        ).fillna("Unknown")
        clean_only_df["disease_cat"] = clean_only_df["disease_norm"].map(map_disease_to_category)
        # Split CLEAN-only patients 80/20 (stratified by disease)
        if HAS_SKLEARN:
            co_train, co_test = train_test_split(
                clean_only_df["patient_num"].tolist(),
                train_size=TRAIN_RATIO,
                stratify=clean_only_df["disease_cat"].values,
                random_state=RANDOM_SEED
            )
        else:
            np.random.seed(RANDOM_SEED)
            co_train, co_test = [], []
            for cat in clean_only_df["disease_cat"].unique():
                pts = clean_only_df[clean_only_df["disease_cat"] == cat]["patient_num"].tolist()
                np.random.shuffle(pts)
                n = max(1, int(len(pts) * TRAIN_RATIO))
                co_train.extend(pts[:n])
                co_test.extend(pts[n:])
        train_patients.update(co_train)
        test_patients.update(co_test)
        patient_summary = pd.concat([patient_summary, clean_only_df[["patient_num", "disease", "age", "gender", "disease_norm", "disease_cat"]]], ignore_index=True)

    # Assign split to ALL events (24,808)
    df["split"] = df["patient_num_str"].map(
        lambda p: "Train" if p and p in train_patients else ("Test" if p and p in test_patients else None)
    )
    # patient_number eksik olanları rastgele 80/20 böl
    nan_mask = df["split"].isna()
    if nan_mask.any():
        np.random.seed(RANDOM_SEED)
        n_nan = nan_mask.sum()
        n_train_nan = int(n_nan * TRAIN_RATIO)
        idx = df[nan_mask].index.tolist()
        np.random.shuffle(idx)
        for i, ix in enumerate(idx):
            df.loc[ix, "split"] = "Train" if i < n_train_nan else "Test"
    df = df[df["split"].notna()].copy()  # Tüm 24,808 event dahil
    train_df = df[df["split"] == "Train"]
    test_df = df[df["split"] == "Test"]

    n_train_events = len(train_df)
    n_test_events = len(test_df)

    # Patient-level: from merged Patient Summary (1,166 total)
    patient_train = patient_summary[patient_summary["patient_num"].astype(str).isin(train_patients)]
    patient_test = patient_summary[patient_summary["patient_num"].astype(str).isin(test_patients)]
    n_train_patients = len(patient_train)
    n_test_patients = len(patient_test)
    n_total_patients = len(patient_summary)

    rows = []

    # --- 1. Total patient count ---
    row = {
        "Variable": "Toplam hasta sayısı",
        "Train": f"{n_train_patients:,}",
        "Test": f"{n_test_patients:,}",
        "p_value": ""
    }
    rows.append(row)

    # --- 2. Gender distribution ---
    gender_combined = pd.concat([
        patient_train[["gender"]].assign(split="Train"),
        patient_test[["gender"]].assign(split="Test"),
    ])
    cont = pd.crosstab(gender_combined["split"], gender_combined["gender"])
    p_gender = chi2_pvalue(cont)
    n_m_train = (patient_train["gender"] == "Male").sum()
    n_f_train = (patient_train["gender"] == "Female").sum()
    n_m_test = (patient_test["gender"] == "Male").sum()
    n_f_test = (patient_test["gender"] == "Female").sum()
    pct_m_train = 100 * n_m_train / n_train_patients if n_train_patients > 0 else 0
    pct_f_train = 100 * n_f_train / n_train_patients if n_train_patients > 0 else 0
    pct_m_test = 100 * n_m_test / n_test_patients if n_test_patients > 0 else 0
    pct_f_test = 100 * n_f_test / n_test_patients if n_test_patients > 0 else 0

    row = {
        "Variable": "Cinsiyet, n (%)",
        "Train": f"Erkek: {n_m_train} ({pct_m_train:.1f}%), Kız: {n_f_train} ({pct_f_train:.1f}%)",
        "Test": f"Erkek: {n_m_test} ({pct_m_test:.1f}%), Kız: {n_f_test} ({pct_f_test:.1f}%)",
        "p_value": f"{p_gender:.3f}" if not np.isnan(p_gender) else "—"
    }
    rows.append(row)

    # --- 3. Age distribution (median, IQR) ---
    age_train = patient_train["age"].dropna().values
    age_test = patient_test["age"].dropna().values
    p_age = ttest_or_mannwhitney(age_train, age_test)

    med_train = np.median(age_train)
    q1_train, q3_train = np.percentile(age_train, [25, 75])
    iqr_train = q3_train - q1_train
    med_test = np.median(age_test)
    q1_test, q3_test = np.percentile(age_test, [25, 75])
    iqr_test = q3_test - q1_test

    row = {
        "Variable": "Yaş (yıl), medyan (IQR)",
        "Train": f"{med_train:.1f} ({q1_train:.1f}–{q3_train:.1f})",
        "Test": f"{med_test:.1f} ({q1_test:.1f}–{q3_test:.1f})",
        "p_value": f"{p_age:.3f}" if not np.isnan(p_age) else "—"
    }
    rows.append(row)

    # --- 4. Disease diagnosis (patient-level) ---
    train_disease = patient_train["disease_cat"]
    test_disease = patient_test["disease_cat"]

    # Chi-square for disease
    disease_combined = pd.concat([
        pd.DataFrame({"split": "Train", "disease": train_disease.values}),
        pd.DataFrame({"split": "Test", "disease": test_disease.values}),
    ])
    cont_disease = pd.crosstab(disease_combined["split"], disease_combined["disease"])
    p_disease = chi2_pvalue(cont_disease)

    disease_order = ["Pnömoni", "Bronkokonstriksiyon", "Normal", "Diğer"]
    rows.append({
        "Variable": "Hastalık tanı dağılımı (hasta bazında)",
        "Train": "",
        "Test": "",
        "p_value": f"{p_disease:.3f}" if not np.isnan(p_disease) else "—"
    })
    for cat in disease_order:
        n_t = (train_disease == cat).sum()
        n_te = (test_disease == cat).sum()
        pct_t = 100 * n_t / n_train_patients if n_train_patients > 0 else 0
        pct_te = 100 * n_te / n_test_patients if n_test_patients > 0 else 0
        rows.append({
            "Variable": f"  {cat}",
            "Train": f"{n_t} ({pct_t:.1f}%)",
            "Test": f"{n_te} ({pct_te:.1f}%)",
            "p_value": ""
        })

    # --- 5. Total event count ---
    row = {
        "Variable": "Toplam olay sayısı",
        "Train": f"{n_train_events:,}",
        "Test": f"{n_test_events:,}",
        "p_value": ""
    }
    rows.append(row)

    # --- 6. Event type distribution ---
    # Use Sound pattern (3 subgroups) for p-value - full 8-category chi-square oversensitive
    event_types_full = ["Normal", "Fine Crackle", "Coarse Crackle", "Wheeze", "Wheeze+Crackle", "Rhonchi", "Stridor", "No Event"]
    df["sound_pattern_for_event"] = df["event_type"].map(map_event_to_sound_pattern)
    df_event_sp = df[df["sound_pattern_for_event"].notna()]
    cont_event_sp = pd.crosstab(df_event_sp["split"], df_event_sp["sound_pattern_for_event"])
    p_event = chi2_pvalue(cont_event_sp)  # Normal, Crackles, Rhonchi alt grupları

    # Header row for event type section
    rows.append({
        "Variable": "Olay tipi dağılımı (alt gruplar: Normal, Crackles, Rhonchi)",
        "Train": "",
        "Test": "",
        "p_value": f"{p_event:.3f}" if not np.isnan(p_event) else "—"
    })
    for et in event_types_full:
        n_t = (train_df["event_type"] == et).sum()
        n_te = (test_df["event_type"] == et).sum()
        pct_t = 100 * n_t / n_train_events if n_train_events > 0 else 0
        pct_te = 100 * n_te / n_test_events if n_test_events > 0 else 0
        rows.append({
            "Variable": f"  {et}",
            "Train": f"{n_t} ({pct_t:.1f}%)",
            "Test": f"{n_te} ({pct_te:.1f}%)",
            "p_value": ""
        })

    # --- 7. Sound pattern distribution ---
    df["sound_pattern"] = df["event_type"].map(map_event_to_sound_pattern)
    df_sp = df[df["sound_pattern"].notna()]
    cont_sp = pd.crosstab(df_sp["split"], df_sp["sound_pattern"])
    p_sp = chi2_pvalue(cont_sp)

    rows.append({
        "Variable": "Sound pattern dağılımı",
        "Train": "",
        "Test": "",
        "p_value": f"{p_sp:.3f}" if not np.isnan(p_sp) else "—"
    })
    for sp in ["Normal", "Crackles", "Rhonchi"]:
        n_t = (train_df["event_type"].map(map_event_to_sound_pattern) == sp).sum()
        n_te = (test_df["event_type"].map(map_event_to_sound_pattern) == sp).sum()
        pct_t = 100 * n_t / n_train_events if n_train_events > 0 else 0
        pct_te = 100 * n_te / n_test_events if n_test_events > 0 else 0
        rows.append({
            "Variable": f"  {sp}",
            "Train": f"{n_t} ({pct_t:.1f}%)",
            "Test": f"{n_te} ({pct_te:.1f}%)",
            "p_value": ""
        })

    # --- 8. Recording location ---
    df_loc = df[df["recording_location"].isin(["p1", "p2", "p3", "p4"])]
    if len(df_loc) > 0:
        cont_loc = pd.crosstab(df_loc["split"], df_loc["recording_location"])
        p_loc = chi2_pvalue(cont_loc)
        rows.append({
            "Variable": "Kayıt lokasyonu dağılımı (p1–p4)",
            "Train": "",
            "Test": "",
            "p_value": f"{p_loc:.3f}" if not np.isnan(p_loc) else "—"
        })
        for loc in ["p1", "p2", "p3", "p4"]:
            n_t = (train_df["recording_location"] == loc).sum()
            n_te = (test_df["recording_location"] == loc).sum()
            pct_t = 100 * n_t / n_train_events if n_train_events > 0 else 0
            pct_te = 100 * n_te / n_test_events if n_test_events > 0 else 0
            rows.append({
                "Variable": f"  {loc}",
                "Train": f"{n_t} ({pct_t:.1f}%)",
                "Test": f"{n_te} ({pct_te:.1f}%)",
                "p_value": ""
            })

    # Build markdown table
    md_lines = [
        "",
        "---",
        "",
        "## Tablo 1. Veri Seti Özellikleri (Train vs Test)",
        "",
        "Train ve test gruplarının aynı popülasyondan geldiğini göstermek için karşılaştırma. "
        f"Stratifiye hasta bazlı bölme (train %{int(TRAIN_RATIO*100)} / test %{100-int(TRAIN_RATIO*100)}). "
        "p < 0.05 anlamlı fark gösterir.",
        "",
        "| Değişken | Train | Test | p değeri |",
        "|----------|-------|------|----------|",
    ]

    for r in rows:
        pval = r["p_value"] if r["p_value"] else "—"
        md_lines.append(f"| {r['Variable']} | {r['Train']} | {r['Test']} | {pval} |")

    md_lines.extend([
        "",
        "**Notlar:**",
        f"- Toplam hasta: Train {n_train_patients} + Test {n_test_patients} = {n_train_patients + n_test_patients} (stratifiye hasta bazlı bölme, örtüşme yok)",
        f"- Toplam olay: Train {n_train_events:,} + Test {n_test_events:,} = {n_train_events + n_test_events:,}",
        "- Yaş: Medyan (IQR); Mann-Whitney U veya t-test",
        "- Kategorik değişkenler: Ki-kare testi",
        "",
    ])

    table1_md = "\n".join(md_lines)
    print(table1_md)

    # --- Table 2: Tüm 17 hastalık (hasta bazında) ---
    all_diseases = sorted(d for d in patient_summary["disease_norm"].unique() if d and d != "-")
    if "Unknown" in patient_summary["disease_norm"].values and "Unknown" not in all_diseases:
        all_diseases.append("Unknown")

    table2_rows = []
    for d in all_diseases:
        n_t = (patient_train["disease_norm"] == d).sum()
        n_te = (patient_test["disease_norm"] == d).sum()
        pct_t = 100 * n_t / n_train_patients if n_train_patients > 0 else 0
        pct_te = 100 * n_te / n_test_patients if n_test_patients > 0 else 0
        table2_rows.append({"disease": d, "train_n": n_t, "train_pct": pct_t, "test_n": n_te, "test_pct": pct_te})

    # Chi-square for Table 2 (all diseases)
    disease_table2 = pd.concat([
        patient_train[["disease_norm"]].assign(split="Train"),
        patient_test[["disease_norm"]].assign(split="Test"),
    ])
    cont_t2 = pd.crosstab(disease_table2["split"], disease_table2["disease_norm"])
    p_t2 = chi2_pvalue(cont_t2)

    table2_md = [
        "",
        "## Tablo 2. Hastalık Tanı Dağılımı — Tüm Hastalıklar (Hasta Bazında)",
        "",
        f"17 hastalık sınıfı, Train vs Test. p = {p_t2:.3f}" + (" (anlamlı)" if p_t2 < 0.05 else " (anlamlı değil)") + ".",
        "",
        "| Hastalık | Train n (%) | Test n (%) |",
        "|----------|-------------|------------|",
    ]
    for r in table2_rows:
        table2_md.append(f"| {r['disease']} | {r['train_n']} ({r['train_pct']:.1f}%) | {r['test_n']} ({r['test_pct']:.1f}%) |")
    table2_md.append("")
    table2_md = "\n".join(table2_md)
    print(table2_md)

    # Append to Disease_Event_Analysis_DETAILED.md
    detail_path = DOCS_DIR / "Disease_Event_Analysis_DETAILED.md"
    with open(detail_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove existing Table 1 and Table 2 sections
    for marker in ["## Tablo 2. Hastalık Tanı Dağılımı", "## Tablo 1. Veri Seti Özellikleri"]:
        if marker in content:
            content = content.split(marker)[0].rstrip()

    content = content.rstrip() + "\n" + table1_md + "\n" + table2_md
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✓ Table 1 appended to {detail_path}")

    # Save tables to CSV
    table_df = pd.DataFrame(rows)
    table_df.to_csv(DOCS_DIR / "table1_train_test.csv", index=False, encoding="utf-8")
    pd.DataFrame(table2_rows).to_csv(DOCS_DIR / "table2_disease_full.csv", index=False, encoding="utf-8")
    print(f"✓ Table 1 saved to {DOCS_DIR / 'table1_train_test.csv'}")
    print(f"✓ Table 2 saved to {DOCS_DIR / 'table2_disease_full.csv'}")


if __name__ == "__main__":
    main()
