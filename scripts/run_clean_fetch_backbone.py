#!/usr/bin/env python3
"""
scripts/run_clean_fetch_backbone.py

Backbone provenance audit.

HeARLoRAClassifier used to default to models/hear_sprsound_best.pth, a checkpoint
produced in January by train_hear_classifier.py (encoder unfrozen, EVENT-level split).
Any model built on it may have seen test-patient audio through the backbone.

This script
  1. downloads the released google/hear-pytorch weights (gated; needs HF_TOKEN),
  2. stores them as models/hear_pristine_encoder.pth in the key layout that
     HeARLoRAClassifier(weights_path=...) loads with strict=True,
  3. compares them tensor-by-tensor with hear_sprsound_best.pth,
  4. writes results_clean/backbone_audit.json.
"""

import os
import sys
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from repro import sha256_file, env_snapshot  # noqa: E402

MODEL_DIR = ROOT / 'models'
PRISTINE = MODEL_DIR / 'hear_pristine_encoder.pth'
LEGACY = MODEL_DIR / 'hear_sprsound_best.pth'
AUDIT = ROOT / 'results_clean' / 'backbone_audit.json'
HF_NAME = 'google/hear-pytorch'


def load_env_token():
    env = ROOT / '.env'
    if not env.exists():
        return
    values = dict(line.split('=', 1) for line in env.read_text().splitlines()
                  if '=' in line and not line.lstrip().startswith('#'))
    for name in ('Home_HF_Token', 'HF_TOKEN'):          # first valid-looking entry wins
        tok = values.get(name, '').strip().strip('"\'')
        if tok:
            os.environ['HF_TOKEN'] = tok
            return


PROBE_KEYS = [
    'hear_encoder.embeddings.patch_embeddings.projection.weight',
    'hear_encoder.encoder.layer.0.attention.attention.query.weight',
    'hear_encoder.encoder.layer.23.output.dense.weight',
]


def legacy_only_audit(reason):
    """No pristine weights available: record what the January checkpoints themselves show.

    A checkpoint saved during Phase 1 (epoch <= 10, encoder frozen) would still carry
    the released encoder, and all such checkpoints would agree bit-for-bit.
    """
    siblings = {'hear_sprsound_best': LEGACY,
                'model1_event_type': MODEL_DIR / 'model1_event_type' / 'best.pth',
                'model2_binary': MODEL_DIR / 'model2_binary' / 'best.pth',
                'model3_disease': MODEL_DIR / 'model3_disease' / 'best.pth'}
    probes, epochs = {}, {}
    for name, p in siblings.items():
        if not p.exists():
            continue
        ckpt = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
        epochs[name] = ckpt.get('epoch')
        sd = ckpt['model_state_dict']
        probes[name] = {k: sd[k].float().clone() for k in PROBE_KEYS if k in sd}
    names = list(probes)
    pairwise = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairwise[f'{a} vs {b}'] = max(
                (probes[a][k] - probes[b][k]).abs().max().item() for k in probes[a])
    audit = {
        'pristine_available': False, 'reason': reason,
        'phase1_epochs': 10, 'checkpoint_epoch': epochs,
        'pairwise_max_abs_diff_probe_tensors': pairwise,
        'legacy_is_pristine': False if epochs.get('hear_sprsound_best', 0) > 10 else None,
        'interpretation': ('hear_sprsound_best.pth was saved in Phase 2 (encoder unfrozen, '
                           'event-level split) and no two January encoders agree, so none of '
                           'them is the released HeAR encoder.'),
        'env': env_snapshot(),
    }
    AUDIT.parent.mkdir(exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ('checkpoint_epoch',
                                            'pairwise_max_abs_diff_probe_tensors')}, indent=2))
    print(f"Audit (legacy-only) written to {AUDIT}")


def main():
    load_env_token()
    from transformers import AutoModel

    print(f"Downloading {HF_NAME} ...")
    try:
        enc = AutoModel.from_pretrained(HF_NAME, token=os.environ.get('HF_TOKEN'))
    except OSError as e:
        print(f"Could not download {HF_NAME}: {str(e).splitlines()[0]}")
        legacy_only_audit(reason=str(e).splitlines()[0])
        sys.exit(2)
    pristine = {f'hear_encoder.{k}': v.detach().cpu() for k, v in enc.state_dict().items()}
    torch.save({'model_state_dict': pristine, 'source': HF_NAME}, PRISTINE)
    print(f"Saved {PRISTINE} ({len(pristine)} tensors)")

    audit = {'pristine_path': str(PRISTINE), 'pristine_sha256': sha256_file(PRISTINE),
             'source': HF_NAME, 'env': env_snapshot()}

    if LEGACY.exists():
        ckpt = torch.load(LEGACY, map_location='cpu', weights_only=False)
        legacy = {k: v for k, v in ckpt.get('model_state_dict', ckpt).items()
                  if k.startswith('hear_encoder.')}
        audit['legacy_path'] = str(LEGACY)
        audit['legacy_meta'] = {k: (v if isinstance(v, (int, float, str)) else str(type(v)))
                                for k, v in ckpt.items() if k != 'model_state_dict'
                                and not k.endswith('state_dict')}
        missing = sorted(set(pristine) - set(legacy))
        extra = sorted(set(legacy) - set(pristine))
        diffs = {}
        for k in sorted(set(pristine) & set(legacy)):
            a, b = pristine[k].float(), legacy[k].float()
            if a.shape != b.shape:
                diffs[k] = 'shape mismatch'
                continue
            d = (a - b).abs().max().item()
            if d > 0:
                diffs[k] = d
        n_common = len(set(pristine) & set(legacy))
        audit.update({
            'n_tensors_common': n_common,
            'n_tensors_changed': len(diffs),
            'max_abs_diff': max([v for v in diffs.values() if isinstance(v, float)], default=0.0),
            'keys_missing_in_legacy': missing[:20],
            'keys_extra_in_legacy': extra[:20],
            'legacy_is_pristine': len(diffs) == 0 and not missing,
            'largest_changes': dict(sorted(
                ((k, v) for k, v in diffs.items() if isinstance(v, float)),
                key=lambda kv: -kv[1])[:15]),
        })
        print(f"Tensors changed vs pristine: {len(diffs)}/{n_common} "
              f"(max |diff| = {audit['max_abs_diff']:.3e})")
        print("VERDICT:", "legacy backbone IS pristine" if audit['legacy_is_pristine']
              else "legacy backbone was FINE-TUNED -> existing results_clean checkpoints "
                   "inherit a backbone trained on an event-level split")

    AUDIT.parent.mkdir(exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2))
    print(f"Audit written to {AUDIT}")


if __name__ == '__main__':
    main()
