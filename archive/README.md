# Archive Folder

This folder contains **old, obsolete, or superseded files** that are kept for reference but are no longer used in the active system.

---

## 📦 Contents

### Old Data Preparation Scripts

**`create_event_level_dataset.py`**
- First version of event-level dataset creation
- Had duplication issues between SPRSound and SPRSound-main
- **Superseded by**: `create_event_level_dataset_CLEAN.py`

**`create_event_level_dataset_CLEAN.py`**
- Second version that fixed duplication issues
- **Superseded by**: `scripts/prepare_ensemble_labels.py`

**`add_wav_paths_to_csv.py`**
- Helper script to add WAV file paths to CSV
- Used once during initial data preparation
- No longer needed (integrated into main data prep)

---

### Old Dataset Files

**`SPRSound_Event_Level_Dataset_CLEAN_no_wav_paths.csv`**
- Intermediate version without WAV paths
- **Superseded by**: `data/SPRSound_Event_Level_Dataset_CLEAN.csv`

**`SPRSound_Event_Level_Dataset_OLD_DUPLICATE.csv`**
- Version with duplicate files
- **Superseded by**: `data/SPRSound_Event_Level_Dataset_CLEAN.csv`

---

### Old Analysis Files

**`SPRSound_HASTALIK_ANALIZI.md`**
- Early disease analysis (RSV-focused)
- **Superseded by**: `docs/Disease_Event_Analysis_DETAILED.md`

---

### Exploration Scripts

**`evaluate_hear_detector.py`**
- Experimental script for testing HeAR's built-in event detector
- Used for initial exploration only
- Not part of final system

---

## 🗑️ Why Archived?

These files are **not deleted** because:
1. **Historical Reference**: Shows evolution of the project
2. **Debugging**: May be useful if issues arise
3. **Learning**: Documents what approaches were tried
4. **Audit Trail**: Complete project history

---

## ⚠️ Important Notes

- **Do NOT use these files** for new work
- **Use the current versions** in `data/`, `src/`, `scripts/` instead
- These files may have bugs, inefficiencies, or outdated logic
- Kept for reference only

---

## 📅 Archive Date

These files were archived on **2026-01-14** during project reorganization.

---

## 🔄 Migration Guide

If you need functionality from archived files:

| Archived File | Current Replacement |
|---------------|---------------------|
| `create_event_level_dataset*.py` | `scripts/prepare_ensemble_labels.py` |
| `add_wav_paths_to_csv.py` | Integrated in `sprsound_dataset.py` |
| `SPRSound_HASTALIK_ANALIZI.md` | `docs/Disease_Event_Analysis_DETAILED.md` |
| `evaluate_hear_detector.py` | Not needed in ensemble system |

---

**If in doubt, use the files in the main project folders, not this archive!**
