# PulmoVec — pre-specified analysis plan

Written 2026-09-19, before any of the runs below were started. Deviations are recorded at
the end of this file with a date and a reason.

## Cohort
SPRSound event table, curated by `src/sprsound_dataset._load_curated_frame`: events with an
audio file, a valid task label and a documented diagnosis (`disease != 'Unknown'`).
Patient identity = `_group_key`. Expected size: 19 702 events, 736 patients.

## Tasks
| Key | Classes |
|---|---|
| `model2_label` screening | Normal / Abnormal |
| `model1_label` sound pattern | Normal / Crackles / Wheeze-Rhonchi |
| `model3_label` disease group | Pneumonia / Bronchial / Normal-Other (recording-level diagnosis broadcast to events) |

## Evaluation designs
1. **Primary — nested patient-grouped cross-validation** (`scripts/run_clean_nested_cv.py`).
   Outer 5-fold `StratifiedGroupKFold` on patients (every patient tested exactly once); within
   each outer fold a 10 %-of-cohort validation partition and 4-fold grouped out-of-fold stacking;
   inner models early-stop on the outer validation partition, never on the fold they score.
   Reported as pooled out-of-fold predictions with a patient-level bootstrap CI **and** the
   per-fold mean, SD and range (pooled-fold bootstrap ignores training-set overlap and is mildly
   anti-conservative).
2. **Secondary — locked hold-out** (588 / 74 / 74 patients, seed 42). Used for model-specific
   analyses (calibration plots, SHAP, saliency, leakage arms). It was inspected on 2026-09-14
   and is therefore not strictly single-use; this is stated in the manuscript.

Rationale for the ordering: the hold-out contains 12 Normal-Other and 18 Bronchial patients,
too few to carry a negative finding; nested CV evaluates the procedure, which is what the
paper claims.

## Leakage arms (identical architecture, hyper-parameters, seed)
| Arm | Backbone | Split |
|---|---|---|
| L0 | released `google/hear-pytorch` | patient-grouped everywhere |
| L1 | `hear_sprsound_best.pth` (fine-tuned in Jan 2026 on an event-level split) | patient-grouped heads + meta (the 2026-09-14 run) |
| L2 | released | event-level split for base, OOF folds and meta |
| L3 | — | numbers of the withdrawn submission; uncontrolled, context only |

Inflation = arm − L0, reported as ΔAUC and Δaccuracy. L2 is additionally scored on L0-test
events absent from its own training partition.

## Baselines and ablation ladder
majority class → duration only → demographics only (age, sex, site) → own-task base model
→ acoustic only (8 probabilities) → full stack (8 probabilities + demographics).

## Statistics
* Uncertainty: cluster bootstrap over patients, B = 2000, percentile 95 % CI, seed 42.
* Paired contrasts (same resampled patients for both models), two-sided bootstrap p,
  Holm-adjusted within task: full vs acoustic-only; full vs demographics-only; acoustic-only
  vs demographics-only; full vs own-task base; acoustic-only vs duration-only.
* DeLong is not used: events are clustered within patients.
* Metrics: accuracy, balanced accuracy, macro-F1, MCC, (macro one-vs-rest) ROC-AUC, AUPRC,
  per-class sensitivity / specificity / PPV / NPV / F1, Brier, ECE (10 bins, top-1),
  calibration intercept and slope.
* Aggregation: event, recording and patient level; confidence-weighted soft vote (primary),
  mean probability (sensitivity). Patient/recording labels for the two acoustic tasks are
  *derived* ("any adventitious event") and described as such.
* Subgroups (exploratory, no hypothesis tests): age < 3, 3–<6, 6–<12, ≥ 12 y (sparse bands
  merged); sex; auscultation site p1–p4. Cells with < 10 patients or one class are suppressed.

## Decision rules fixed in advance
* Acoustic phenotyping may be called **robust** only if, under L0 in the primary analysis,
  the lower 95 % CI bound of the AUC exceeds the point estimate of the duration-only
  baseline for that task. Otherwise the wording is "classification of annotated events" and
  the title is rebuilt around the leakage contribution.
* Disease-group prediction is reported as **not demonstrated** unless the full stack beats
  demographics-only with a Holm-adjusted p < 0.05 **and** patient-level accuracy exceeds the
  majority-class rate.
* No analysis is dropped for being unfavourable; analyses that could not be run are listed
  as such.

## Deviations
* 2026-09-19 — Released HeAR weights could not be downloaded (HF token invalid). CPU analysis
  code is developed and dry-run against the arm-L1 probability tables (same schema); no L1
  number is a headline result.
* 2026-09-19, later the same day — a valid token was supplied; the released encoder was
  downloaded, audited against the January checkpoint and used for every reported arm. The
  earlier entry is retained as a record of the sequence.
* 2026-09-21 — the pre-specified comparison of the two split designs on the events both held
  out was run (`scripts/run_clean_common_events.py`) and is reported in the Results. No
  pre-specified analysis remains un-run.
* 2026-09-21 (logged 2026-09-22) — two preprocessing controls added after internal review,
  not pre-specified and reported as such: retraining with the mel bands above the band-pass
  cut-off masked, and with the event embedded in real surrounding audio instead of isolated
  and noise-padded (`scripts/run_clean_ablations.py`; run 2026-09-21 01:45-02:30).
* Timing note — arm L1 is the run of 2026-09-14, which predates this plan; its base models
  were not re-trained. All other arms and analyses were run after the plan was written.
* Cohort — nine events of one child whose age was encoded as 55 years in three recordings
  (2.5 years in all others) were excluded (user decision, 2026-09-19); the cohort is therefore
  19 693 rather than the expected 19 702 events, with the same 736 patients.
* Subgroups — annotated-event-duration tertiles were added to the pre-specified subgroups
  (age band, sex, site) as a shortcut check; exploratory, like the others.
* 2026-09-22 — child-level aggregation rules (at least one / at least two events predicted
  adventitious) added after review, not pre-specified (`scripts/run_clean_patient_rules.py`).

* 2026-09-23 — recording- and patient-level disease-group models on aggregated outer-test event
  probabilities, and pneumonia versus other diagnoses as a target, added after the primary
  results were known; not pre-specified (`scripts/run_clean_disease_posthoc.py`, output in
  `results_clean/posthoc_disease/`). The pre-specified disease-group conclusion is unchanged.
* 2026-09-23 — sensitivity and specificity of the acoustic outcomes by event-duration tertile,
  at the model's own decision and at a tertile-specific threshold matched to the overall
  specificity, added after internal review; not pre-specified
  (`scripts/run_clean_duration_operating_point.py`, output in
  `results_clean/duration_operating_point/`).
* 2026-09-23 (later) — after a further internal review, not pre-specified: pneumonia versus
  healthy controls as an additional post hoc disease target, age by disease group and severe
  versus non-severe pneumonia (`scripts/run_clean_disease_posthoc.py`; earlier rows unchanged),
  and positive predictive values at other prevalences (`scripts/run_clean_ppv_prevalence.py`).
* 2026-09-24 — revision requests, all post hoc (`scripts/run_clean_revision_requests.py`, output in
  `results_clean/revision_requests/`): child-level confusion counts and PPV/NPV at prevalences
  0.10, 0.20 and 0.327 (A1); inventory of the 42 post hoc disease-group configurations, six of
  which are unfitted mean predicted probabilities (A2); child-level decision curve (C1); release
  inventory for a cross-release validation, no model trained (B1); and the nested CV re-run with
  an unweighted second stage for the full stack, demographics-only and duration-only sets
  (`run_clean_meta_v2.py --class-weight none`, same folds, features and 50-trial tuning; B2).
  A re-run of the weighted full stack in fold 1 reproduced the original AUCs to four decimals.
