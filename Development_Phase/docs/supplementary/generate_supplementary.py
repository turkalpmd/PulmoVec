#!/usr/bin/env python3
"""
Supplementary Materials Generator
Creates all supplementary tables and figures for the manuscript.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
SUPP_DIR = SCRIPT_DIR
FIGURES_DIR = SUPP_DIR / "figures"
TABLES_DIR = SUPP_DIR / "tables"
DOCS_DIR = SCRIPT_DIR.parent
DEV_DIR = SCRIPT_DIR.parent.parent
DATA_PATH = DEV_DIR / "data" / "SPRSound_Event_Level_Dataset_CLEAN.csv"
META_RESULTS = DEV_DIR / "Project" / "Results" / "meta_model"
PATIENT_SUMMARY_DIR = DEV_DIR / "SPRSound-main" / "Patient Summary"
PATIENT_SUMMARY_FILES = [
    PATIENT_SUMMARY_DIR / "SPRSound_patient_summary.csv",
    PATIENT_SUMMARY_DIR / "Grand_Challenge'23_patient_summary.csv",
    PATIENT_SUMMARY_DIR / "Grand_Challenge'24_patient_summary.csv",
]
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

EVENT_COLS = ["Coarse Crackle", "Fine Crackle", "No Event", "Normal", "Rhonchi", "Stridor", "Wheeze", "Wheeze+Crackle"]


def map_disease_to_category(disease):
    """Map raw disease to 4 categories."""
    if pd.isna(disease):
        return "Others"
    if "Pneumonia" in disease:
        return "Pneumonia"
    if disease in ["Asthma", "Bronchitis", "Bronchiolitis"]:
        return "Bronchial Diseases"
    if disease == "Control Group":
        return "Normal"
    return "Others"


def load_data_with_split():
    """Load CLEAN data and assign train/test split (same logic as generate_table1)."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["wav_exists"] == "yes"].copy()

    def safe_patient_str(x):
        try:
            return str(int(float(x))) if pd.notna(x) else None
        except (ValueError, TypeError):
            return None

    df["patient_num_str"] = df["patient_number"].apply(safe_patient_str)

    # Load merged patient summary
    dfs = []
    for f in PATIENT_SUMMARY_FILES:
        if f.exists():
            d = pd.read_csv(f)
            if "patient_num" in d.columns:
                d = d[["patient_num", "disease", "age", "gender"]].copy()
                d["patient_num"] = d["patient_num"].astype(str).str.strip()
                dfs.append(d)
    if not dfs:
        # Fallback: random split by patient
        patients = df["patient_num_str"].dropna().unique()
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(patients)
        n_train = max(1, int(len(patients) * TRAIN_RATIO))
        train_patients = set(patients[:n_train])
        test_patients = set(patients[n_train:])
    else:
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["patient_num"], keep="first")
        merged["disease_norm"] = merged["disease"].replace(
            {"Control group": "Control Group", "-": "Unknown", "Other respiratory": "Other respiratory diseases"}
        ).fillna("Unknown")
        merged["disease_cat"] = merged["disease_norm"].map(map_disease_to_category)

        try:
            from sklearn.model_selection import train_test_split
            train_pts, test_pts = train_test_split(
                merged["patient_num"].tolist(),
                train_size=TRAIN_RATIO,
                stratify=merged["disease_cat"].values,
                random_state=RANDOM_SEED
            )
            train_patients = set(str(p) for p in train_pts)
            test_patients = set(str(p) for p in test_pts)
        except ImportError:
            np.random.seed(RANDOM_SEED)
            train_patients, test_patients = set(), set()
            for cat in merged["disease_cat"].unique():
                pts = merged[merged["disease_cat"] == cat]["patient_num"].tolist()
                np.random.shuffle(pts)
                n = max(1, int(len(pts) * TRAIN_RATIO))
                train_patients.update(pts[:n])
                test_patients.update(pts[n:])

        # CLEAN-only patients
        all_pts = train_patients | test_patients
        clean_only = set(df["patient_num_str"].dropna().unique()) - all_pts
        if clean_only:
            np.random.seed(RANDOM_SEED)
            co_list = list(clean_only)
            np.random.shuffle(co_list)
            n_co_train = max(1, int(len(co_list) * TRAIN_RATIO))
            train_patients.update(co_list[:n_co_train])
            test_patients.update(co_list[n_co_train:])

    df["split"] = df["patient_num_str"].map(
        lambda p: "Train" if p and p in train_patients else ("Test" if p and p in test_patients else None)
    )
    nan_mask = df["split"].isna()
    if nan_mask.any():
        np.random.seed(RANDOM_SEED)
        idx = df[nan_mask].index.tolist()
        np.random.shuffle(idx)
        n_train_nan = int(len(idx) * TRAIN_RATIO)
        for i, ix in enumerate(idx):
            df.loc[ix, "split"] = "Train" if i < n_train_nan else "Test"
    df = df[df["split"].notna()].copy()
    return df


def create_s1_sixteen_disease_distribution(df):
    """Supplementary Table S1: Sixteen-disease distribution in the SPRSound cohort."""
    train_df = df[df["split"] == "Train"]
    test_df = df[df["split"] == "Test"]

    # Patient counts per disease (from unique patients)
    train_pat_per_disease = train_df.groupby("disease")["patient_num_str"].nunique()
    test_pat_per_disease = test_df.groupby("disease")["patient_num_str"].nunique()

    # Event counts per disease
    train_ev_per_disease = train_df.groupby("disease").size()
    test_ev_per_disease = test_df.groupby("disease").size()

    all_diseases = sorted(set(train_pat_per_disease.index) | set(test_pat_per_disease.index))
    rows = []
    for d in all_diseases:
        tp = train_pat_per_disease.get(d, 0)
        te = test_pat_per_disease.get(d, 0)
        ep = train_ev_per_disease.get(d, 0)
        ee = test_ev_per_disease.get(d, 0)
        rows.append({
            "Disease": d,
            "Train patients": int(tp),
            "Test patients": int(te),
            "Train events": int(ep),
            "Test events": int(ee),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("Train events", ascending=False)
    out.to_csv(TABLES_DIR / "s1_sixteen_disease_distribution.csv", index=False)
    print(f"✓ Saved {TABLES_DIR / 's1_sixteen_disease_distribution.csv'}")
    return out


def create_s2_label_mapping():
    """Supplementary Table S2: Label mapping from original annotations to study targets."""
    # Event type -> Sound pattern
    event_to_sp = [
        {"Original event label": "Normal", "Sound pattern": "Normal"},
        {"Original event label": "Fine Crackle", "Sound pattern": "Crackles"},
        {"Original event label": "Coarse Crackle", "Sound pattern": "Crackles"},
        {"Original event label": "Wheeze+Crackle", "Sound pattern": "Crackles"},
        {"Original event label": "Wheeze", "Sound pattern": "Rhonchi"},
        {"Original event label": "Stridor", "Sound pattern": "Rhonchi"},
        {"Original event label": "Rhonchi", "Sound pattern": "Rhonchi"},
        {"Original event label": "No Event", "Sound pattern": "Excluded"},
    ]
    df_sp = pd.DataFrame(event_to_sp)
    df_sp.to_csv(TABLES_DIR / "s2a_event_to_sound_pattern.csv", index=False)

    # Diagnosis -> 4 disease group
    diag_to_group = [
        {"Original diagnosis": "Pneumonia (severe)", "4 Disease group": "Pneumonia"},
        {"Original diagnosis": "Pneumonia (non-severe)", "4 Disease group": "Pneumonia"},
        {"Original diagnosis": "Asthma", "4 Disease group": "Bronchial Diseases"},
        {"Original diagnosis": "Bronchitis", "4 Disease group": "Bronchial Diseases"},
        {"Original diagnosis": "Bronchiolitis", "4 Disease group": "Bronchial Diseases"},
        {"Original diagnosis": "Control Group", "4 Disease group": "Normal"},
        {"Original diagnosis": "Acute upper respiratory infection", "4 Disease group": "Others"},
        {"Original diagnosis": "Airway foreign body", "4 Disease group": "Others"},
        {"Original diagnosis": "Bronchiectasia", "4 Disease group": "Others"},
        {"Original diagnosis": "Chronic cough", "4 Disease group": "Others"},
        {"Original diagnosis": "Hemoptysis", "4 Disease group": "Others"},
        {"Original diagnosis": "Kawasaki disease", "4 Disease group": "Others"},
        {"Original diagnosis": "Other respiratory diseases", "4 Disease group": "Others"},
        {"Original diagnosis": "Protracted bacterial bronchitis", "4 Disease group": "Others"},
        {"Original diagnosis": "Pulmonary hemosiderosis", "4 Disease group": "Others"},
        {"Original diagnosis": "Unknown", "4 Disease group": "Others"},
    ]
    df_dg = pd.DataFrame(diag_to_group)
    df_dg.to_csv(TABLES_DIR / "s2b_diagnosis_to_four_group.csv", index=False)
    print(f"✓ Saved s2a_event_to_sound_pattern.csv, s2b_diagnosis_to_four_group.csv")
    return df_sp, df_dg


def create_s3_additional_model_results():
    """Supplementary Table S3: Additional classification results for 7-class event type and 16-disease models."""
    rows = []
    # Event type (6-class, excludes Stridor/No Event)
    et_path = META_RESULTS / "event_type" / "metrics.json"
    if et_path.exists():
        with open(et_path) as f:
            m = json.load(f)
        rows.append({
            "Model": "Event type (6-class)",
            "Classes": "Coarse Crackle, Fine Crackle, Normal, Rhonchi, Wheeze, Wheeze+Crackle",
            "Accuracy": f"{m['accuracy']['value']:.4f}",
            "Macro F1": f"{m['f1_macro']['value']:.4f}",
            "MCC": f"{m['mcc']['value']:.4f}",
            "ROC-AUC (macro)": f"{m['roc_auc_macro']['value']:.4f}",
        })
    # Disease (16-class)
    d_path = META_RESULTS / "disease" / "metrics.json"
    if d_path.exists():
        with open(d_path) as f:
            m = json.load(f)
        rows.append({
            "Model": "Disease (16-class)",
            "Classes": "All 16 diagnoses (see Table S1)",
            "Accuracy": f"{m['accuracy']['value']:.4f}",
            "Macro F1": f"{m['f1_macro']['value']:.4f}",
            "MCC": f"{m['mcc']['value']:.4f}",
            "ROC-AUC (macro)": f"{m['roc_auc_macro']['value']:.4f}",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "s3_additional_model_results.csv", index=False)
    print(f"✓ Saved {TABLES_DIR / 's3_additional_model_results.csv'}")
    return out


def create_s4_hyperparameters():
    """Supplementary Table S4: Final hyperparameters for base models and LightGBM meta-models."""
    # Base models (HeAR fine-tuned) - from config
    base_rows = [
        {"Component": "HeAR encoder", "Parameter": "Model", "Value": "google/hear-pytorch"},
        {"Component": "HeAR encoder", "Parameter": "Embedding dim", "Value": "512"},
        {"Component": "Classification head", "Parameter": "Hidden dim", "Value": "256"},
        {"Component": "Classification head", "Parameter": "Dropout", "Value": "0.3"},
        {"Component": "Training", "Parameter": "Phase 1 epochs", "Value": "10 (frozen encoder)"},
        {"Component": "Training", "Parameter": "Phase 2 epochs", "Value": "40 (fine-tune)"},
        {"Component": "Training", "Parameter": "Batch size", "Value": "32"},
        {"Component": "Training", "Parameter": "Learning rate (Phase 1)", "Value": "1e-4"},
        {"Component": "Training", "Parameter": "Learning rate (Phase 2)", "Value": "5e-7"},
    ]
    # LightGBM meta-model - Optuna search space (typical final values)
    lgb_rows = [
        {"Component": "LightGBM", "Parameter": "n_estimators", "Value": "50–500 (Optuna)"},
        {"Component": "LightGBM", "Parameter": "max_depth", "Value": "3–15"},
        {"Component": "LightGBM", "Parameter": "learning_rate", "Value": "0.01–0.3 (log)"},
        {"Component": "LightGBM", "Parameter": "num_leaves", "Value": "15–300"},
        {"Component": "LightGBM", "Parameter": "min_child_samples", "Value": "5–100"},
        {"Component": "LightGBM", "Parameter": "subsample", "Value": "0.6–1.0"},
        {"Component": "LightGBM", "Parameter": "colsample_bytree", "Value": "0.6–1.0"},
        {"Component": "LightGBM", "Parameter": "early_stopping_rounds", "Value": "20"},
        {"Component": "LightGBM", "Parameter": "Optimization", "Value": "Optuna TPE, 100 trials"},
    ]
    out = pd.DataFrame(base_rows + lgb_rows)
    out.to_csv(TABLES_DIR / "s4_hyperparameters.csv", index=False)
    print(f"✓ Saved {TABLES_DIR / 's4_hyperparameters.csv'}")
    return out


def create_fig_s1_confusion_matrices():
    """Supplementary Figure S1: Confusion matrices for event-level tasks."""
    src_dirs = [
        ("model1_label", "Sound pattern (3-class)"),
        ("model2_label", "Binary (Normal/Abnormal)"),
        ("event_type", "Event type (6-class)"),
        ("model4_label", "Disease group (4-class)"),
    ]
    for folder, label in src_dirs:
        src = META_RESULTS / folder / "confusion_matrix.png"
        if src.exists():
            dst = FIGURES_DIR / f"fig_s1a_{folder}_confusion.png"
            shutil.copy(src, dst)
            print(f"✓ Copied {folder} confusion matrix -> {dst.name}")
    # ROC/PR curves for disease-group and disease (16-class)
    for folder, name in [("model4_label", "disease_group"), ("disease", "disease_16class")]:
        src = META_RESULTS / folder / "roc_auprc_curves.png"
        if src.exists():
            dst = FIGURES_DIR / f"fig_s1b_{name}_roc_pr.png"
            shutil.copy(src, dst)
            print(f"✓ Copied {folder} ROC/PR -> {dst.name}")


def main():
    print("Loading data...")
    df = load_data_with_split()
    print(f"Total events: {len(df):,}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"Unique patients: {df['patient_num_str'].nunique()}")

    # Pivot tables (existing)
    pivot = pd.pivot_table(df, values="filename", index="disease", columns="event_type", aggfunc="count", fill_value=0)
    for col in EVENT_COLS:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[[c for c in EVENT_COLS if c in pivot.columns]]
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOTAL", ascending=False)
    event_cols = [c for c in EVENT_COLS if c in pivot.columns]
    pivot_pct = pivot.copy()
    pivot_pct[event_cols] = pivot_pct[event_cols].div(pivot_pct["TOTAL"], axis=0) * 100

    pivot.to_csv(TABLES_DIR / "disease_event_pivot_counts.csv")
    pivot_pct.to_csv(TABLES_DIR / "disease_event_pivot_percentages.csv")

    # Heatmap
    pivot_heatmap = pivot_pct.loc[pivot.index, event_cols]
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_heatmap, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Percentage (%)"},
                linewidths=0.5, vmin=0, vmax=100)
    plt.title("Disease vs Event Type Heatmap (All Diseases)\nPercentage Distribution", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Event Type", fontsize=12, fontweight="bold")
    plt.ylabel("Disease", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_disease_event_heatmap_all.png", dpi=300, bbox_inches="tight")
    plt.close()

    # New supplementary tables and figures
    create_s1_sixteen_disease_distribution(df)
    create_s2_label_mapping()
    create_s3_additional_model_results()
    create_s4_hyperparameters()
    create_fig_s1_confusion_matrices()

    # Copy Table 1 and Table 2
    if (DOCS_DIR / "table1_train_test.csv").exists():
        shutil.copy(DOCS_DIR / "table1_train_test.csv", TABLES_DIR / "table1_train_test.csv")
    if (DOCS_DIR / "table2_disease_full.csv").exists():
        shutil.copy(DOCS_DIR / "table2_disease_full.csv", TABLES_DIR / "table2_disease_full.csv")

    # Build SUPPLEMENTARY.md
    md = build_supplementary_md(pivot, pivot_pct, event_cols, len(df))
    with open(SUPP_DIR / "SUPPLEMENTARY.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ Saved SUPPLEMENTARY.md")

    print("\n" + "=" * 60)
    print("SUPPLEMENTARY MATERIALS GENERATED")
    print("=" * 60)


def build_supplementary_md(pivot, pivot_pct, event_cols, total_events):
    """Build full SUPPLEMENTARY.md content."""
    s1 = pd.read_csv(TABLES_DIR / "s1_sixteen_disease_distribution.csv")
    s2a = pd.read_csv(TABLES_DIR / "s2a_event_to_sound_pattern.csv")
    s2b = pd.read_csv(TABLES_DIR / "s2b_diagnosis_to_four_group.csv")
    s3 = pd.read_csv(TABLES_DIR / "s3_additional_model_results.csv")
    s4 = pd.read_csv(TABLES_DIR / "s4_hyperparameters.csv")

    def df_to_md(df):
        h = "| " + " | ".join(df.columns) + " |"
        sep = "|" + "|".join(["-" * (len(c) + 2) for c in df.columns]) + "|"
        rows = [h, sep]
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(str(v) for v in r.values) + " |")
        return "\n".join(rows)

    lines = [
        "# Supplementary Materials",
        "",
        "This document contains all supplementary tables and figures referenced in the manuscript.",
        "",
        "---",
        "",
        "## Supplementary Table S1. Sixteen-disease distribution in the SPRSound cohort",
        "",
        "Full distribution of 16 diseases with train/test patient counts and event counts.",
        "",
        df_to_md(s1),
        "",
        "---",
        "",
        "## Supplementary Table S2. Label mapping from original annotations to study targets",
        "",
        "**Part A – Event type → Sound pattern:**",
        "",
        df_to_md(s2a),
        "",
        "**Part B – Original diagnosis → 4 disease group:**",
        "",
        df_to_md(s2b),
        "",
        "---",
        "",
        "## Supplementary Table S3. Additional classification results for 6-class event type and 16-disease models",
        "",
        df_to_md(s3),
        "",
        "---",
        "",
        "## Supplementary Table S4. Final hyperparameters for base models and LightGBM meta-models",
        "",
        df_to_md(s4),
        "",
        "---",
        "",
        "## Supplementary Figure S1. Confusion matrices for event-level tasks",
        "",
        "**Sound pattern (3-class)**",
        "",
        "![model1](figures/fig_s1a_model1_label_confusion.png)",
        "",
        "**Binary (Normal/Abnormal)**",
        "",
        "![model2](figures/fig_s1a_model2_label_confusion.png)",
        "",
        "**Event type (6-class)**",
        "",
        "![event](figures/fig_s1a_event_type_confusion.png)",
        "",
        "**Disease group (4-class)**",
        "",
        "![model4](figures/fig_s1a_model4_label_confusion.png)",
        "",
        "**ROC and Precision-Recall curves (Disease group, 4-class)**",
        "",
        "![roc4](figures/fig_s1b_disease_group_roc_pr.png)",
        "",
        "**ROC and Precision-Recall curves (Disease, 16-class)**",
        "",
        "![roc16](figures/fig_s1b_disease_16class_roc_pr.png)",
        "",
        "---",
        "",
        "## Disease vs Event Type Heatmap",
        "",
        "![Heatmap](figures/fig1_disease_event_heatmap_all.png)",
        "",
        "---",
        "",
        "## File structure",
        "",
        "```",
        "supplementary/",
        "├── SUPPLEMENTARY.md",
        "├── figures/",
        "│   ├── fig1_disease_event_heatmap_all.png",
        "│   ├── fig_s1a_model1_label_confusion.png",
        "│   ├── fig_s1a_model2_label_confusion.png",
        "│   ├── fig_s1a_event_type_confusion.png",
        "│   ├── fig_s1a_model4_label_confusion.png",
        "│   ├── fig_s1b_disease_group_roc_pr.png",
        "│   └── fig_s1b_disease_16class_roc_pr.png",
        "└── tables/",
        "    ├── s1_sixteen_disease_distribution.csv",
        "    ├── s2a_event_to_sound_pattern.csv",
        "    ├── s2b_diagnosis_to_four_group.csv",
        "    ├── s3_additional_model_results.csv",
        "    ├── s4_hyperparameters.csv",
        "    ├── disease_event_pivot_counts.csv",
        "    ├── disease_event_pivot_percentages.csv",
        "    ├── table1_train_test.csv",
        "    └── table2_disease_full.csv",
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
