# PulmoVec

Patient-level validation of a foundation-model pipeline for paediatric lung-sound
classification on the public [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
database.

This repository contains the **training and evaluation code** behind the manuscript. It does
not contain audio, derived data tables, model weights or the manuscript itself.

> **Status.** An earlier preprint of this project (arXiv:2603.15688) reported results obtained
> with an event-level train/test split. Children in SPRSound contribute many recordings and
> events each, so that split leaks patient identity. The code here implements the patient-level
> re-analysis that supersedes those results; final numbers will be added when the runs listed
> in `results_clean/ANALYSIS_PLAN.md` are complete.

## What the pipeline does

1. **Events → clips.** Each annotated event is cut at its boundaries, centred in a 2 s buffer,
   band-pass filtered (100–1800 Hz) and padded with low-level band-limited noise
   (`src/sprsound_dataset.py`).
2. **Base models.** Frozen HeAR ViT-L encoder + LoRA adapters (last six blocks) + attention
   pooling; one model per task (`src/models_hear_lora.py`, `scripts/train_hear_lora.py`).
   Tasks: screening (normal / adventitious), sound pattern (normal / crackles / wheeze-rhonchi),
   disease group (pneumonia / bronchial / normal-other).
3. **Stacking.** Out-of-fold base-model probabilities + age, sex and auscultation site →
   LightGBM, tuned with Optuna on validation patients (`scripts/run_clean_meta_v2.py`).
4. **Evaluation.** Patient-cluster bootstrap CIs, paired Holm-adjusted contrasts, event /
   recording / patient-level aggregation, calibration, subgroups, SHAP, occlusion saliency
   with faithfulness checks (`scripts/run_clean_metrics.py` and neighbours).

## Leakage safeguards

* The unit of every split is the **patient** (`_group_key`); `_assert_no_patient_leakage`
  runs before every fit and every evaluation.
* The encoder weights are passed explicitly and their SHA-256 is stored with every run.
  `scripts/run_clean_fetch_backbone.py` downloads the released `google/hear-pytorch` weights
  and audits any local checkpoint against them tensor by tensor
  (`results_clean/backbone_audit.json`).
* Leakage is measured rather than assumed — four arms with identical architecture and seed:

  | Arm | Encoder | Split |
  |---|---|---|
  | L0 | released HeAR | patient-level everywhere |
  | L1 | checkpoint previously fine-tuned on an event-level split | patient-level heads and meta-learner |
  | L2 | released HeAR | event-level (deliberately leaky) |
  | L3 | — | numbers of the earlier preprint, context only |

* Comparators include a **duration-only** model (annotated event length is a potential
  shortcut) and a **demographics-only** model.

## Reproducing

```bash
pip install -r requirements.txt
git clone https://github.com/Google-Health/hear hear         # HeAR preprocessing code
cp .env.example .env                                          # add a Hugging Face token with
                                                              # access to google/hear-pytorch
python scripts/run_clean_fetch_backbone.py                    # released encoder + audit
python scripts/prepare_ensemble_labels.py                     # task labels from the event table
python scripts/run_clean_pipeline.py --results-dir results_clean/arm_L0_clean      # hold-out
python scripts/run_clean_nested_cv.py                         # primary analysis (5 x 4 nested CV)
python scripts/run_clean_pipeline.py --split event --results-dir results_clean/arm_L2_event_split
scripts/run_clean_all.sh                                      # or: queue everything
```

One RTX 3090 (24 GB): ≈ 3–4 h for a hold-out arm, ≈ 12 h for the nested cross-validation.
HeAR weights are distributed by Google under the Health AI Developer Foundations terms and
are not redistributed here.

## Layout

```
src/                 dataset, clip extraction, models, seeding
scripts/run_clean_*  the analysis reported in the manuscript
scripts/train_hear_classifier.py   legacy end-to-end fine-tuning; kept because it produced
                                   the contaminated encoder studied as arm L1
results_clean/       pre-specified analysis plan, audits, aggregate metrics (no per-event data)
```

## Licence

Code: MIT (see `LICENSE`). SPRSound data and HeAR weights are governed by their own terms.
