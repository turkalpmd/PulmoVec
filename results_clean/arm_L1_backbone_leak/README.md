# Arm L1 — backbone-level leakage

The 2026-09-14 run of `run_clean_pipeline.py` + `run_clean_meta.py`, moved here unchanged on 2026-09-19.
Heads, LoRA adapters, OOF folds and the meta-learner are patient-grouped, but the frozen encoder
is `models/hear_sprsound_best.pth` (January 2026, epoch 50, encoder unfrozen, event-level split);
see `../backbone_audit.json`. Splits are the shared `../split_{train,val,test}.csv`.
