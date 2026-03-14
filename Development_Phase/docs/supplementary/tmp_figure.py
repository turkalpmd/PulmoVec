from pathlib import Path
import pickle
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import label_binarize


"""
Temporary script to regenerate the disease-group ROC + PR curves
for the meta-model (Model 4) and copy them into the supplementary folder.

Output:
- Project/Results/meta_model/model4_label/roc_auprc_curves.png
- docs/supplementary/figures/fig_s1b_disease_group_roc_pr.png
"""

base = Path("/Users/turkalpmd/Desktop/PROJECTS/PulmoVec/Development_Phase")
model_dir = base / "Project" / "Results" / "meta_model" / "model4_label"
supp_fig = base / "docs" / "supplementary" / "figures"
val_csv = base / "data" / "ensemble_probabilities_val.csv"
clean_csv = base / "data" / "SPRSound_Event_Level_Dataset_CLEAN.csv"
model_pkl = model_dir / "model.pkl"

# Load validation probabilities and trained LightGBM model
df = pd.read_csv(val_csv)
with open(model_pkl, "rb") as f:
    model = pickle.load(f)

# Demographic encoding (match training pipeline)
df["gender_encoded"] = df["gender"].map({"Female": 0, "Male": 1})
if df["age"].isna().any():
    df["age"] = df["age"].fillna(df["age"].median())

# recording_location may not be present in the probabilities CSV;
# if missing, merge from CLEAN dataset.
if "recording_location" not in df.columns:
    clean = pd.read_csv(clean_csv)
    clean = clean[["patient_number", "age", "gender", "disease", "event_type", "recording_location"]]
    clean = clean.drop_duplicates(
        subset=["patient_number", "age", "gender", "disease", "event_type"], keep="first"
    )
    df = df.merge(clean, on=["patient_number", "age", "gender", "disease", "event_type"], how="left")

if "recording_location" in df.columns:
    if df["recording_location"].isna().any():
        df["recording_location"] = df["recording_location"].fillna(df["recording_location"].mode()[0])
    loc_map = {v: i for i, v in enumerate(sorted(df["recording_location"].unique()))}
    df["recording_location_encoded"] = df["recording_location"].map(loc_map)

# Feature matrix for the meta-model
feature_cols = [
    "Model1_Normalpp",
    "Model1_Cracklespp",
    "Model1_Rhonchipp",
    "Model2_Normalpp",
    "Model2_Abnormalpp",
    "Model3_Normalpp",
    "Model3_Pneumoniapp",
    "Model3_Bronchiolitispp",
    "age",
    "gender_encoded",
]
if "recording_location_encoded" in df.columns:
    feature_cols.append("recording_location_encoded")

X = df[feature_cols[: model.n_features_in_]]

# Reconstruct model4_label if not present
if "model4_label" not in df.columns:

    def to_model4(d):
        if isinstance(d, str) and "Pneumonia" in d:
            return 0
        if d in ["Asthma", "Bronchitis", "Bronchiolitis"]:
            return 1
        if d == "Control Group":
            return 2
        return 3

    df["model4_label"] = df["disease"].map(to_model4)

y_true = df["model4_label"].astype(int).values
y_proba = model.predict_proba(X)
class_names = ["Pneumonia", "Bronchial Diseases", "Normal", "Others"]


def plot_curves(display_name: str, out_path: Path) -> None:
    """Helper to plot ROC + PR curves with a shared title."""
    plt.style.use("ggplot")
    # Taller figure as requested (width=10, height=14)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    colors = ["#E24A33", "#348ABD", "#988ED5", "#777777"]

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Macro averages (one-vs-rest, multi-class)
    macro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    macro_ap = average_precision_score(y_bin, y_proba, average="macro")

    # ROC panel
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=2, color=colors[i], label=f"{cname} (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
    axes[0].set_title("ROC (One-vs-Rest)", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=16)
    axes[0].set_ylabel("True Positive Rate", fontsize=16)
    axes[0].tick_params(axis="both", labelsize=16)
    axes[0].legend(fontsize=16, loc="lower right", framealpha=0.95)
    axes[0].grid(alpha=0.3)

    # PR panel
    for i, cname in enumerate(class_names):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        axes[1].plot(r, p, lw=2, color=colors[i], label=f"{cname} (AP={ap:.3f})")
    axes[1].set_title("Precision-Recall (One-vs-Rest)", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=16)
    axes[1].set_ylabel("Precision", fontsize=16)
    axes[1].tick_params(axis="both", labelsize=16)
    axes[1].legend(fontsize=16, loc="lower left", framealpha=0.95)
    axes[1].grid(alpha=0.3)

    subtitle = f"Macro AUC = {macro_auc:.3f}, Macro AP = {macro_ap:.3f}"
    # Single-line title with model name + macro scores
    fig.suptitle(
        f"{display_name} – Multi-Class Performance Curves ({subtitle})",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    # Reduce spacing between title and plots a bit more
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# 1) Project-level: Disease Group Prediction Model
out_project = model_dir / "roc_auprc_curves.png"
plot_curves("Disease Group Prediction Model", out_project)

# 2) Supplement-level: Patient-level Disease Group Prediction Model
supp_fig.mkdir(exist_ok=True)
out_supp = supp_fig / "fig_s1b_disease_group_roc_pr.png"
plot_curves("Patient-level Disease Group Prediction Model", out_supp)

print("tmp_figure.py finished.")
