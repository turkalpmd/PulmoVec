#!/usr/bin/env python3
"""
scripts/export_adapter_weights.py

Export ONLY what this project trained - LoRA matrices, the attention-pooling layer and the
classification head - from a HeAR+LoRA checkpoint. The frozen HeAR encoder is NOT included:
it is distributed by Google under the Health AI Developer Foundations terms and must be
obtained from google/hear-pytorch. load_adapter() rebuilds the full model from the two parts
and verifies that no encoder tensor was shipped.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

TRAINED_PREFIXES = ('patch_pooler.', 'classifier.')


def is_trained(key):
    return key.startswith(TRAINED_PREFIXES) or '.lora_' in key


def export(ckpt_path, out_dir, task, classes):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    keep = {k: v.contiguous() for k, v in sd.items() if is_trained(k)}
    assert keep and not any(k.startswith('hear_encoder.') and '.lora_' not in k for k in keep)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(keep, str(out_dir / f'{task}.safetensors'))
    meta = {'task': task, 'classes': classes, 'n_tensors': len(keep),
            'n_parameters': int(sum(v.numel() for v in keep.values())),
            'lora': {'r': 8, 'alpha': 16.0, 'layers': 6, 'targets': ['query', 'value']},
            'base_model': 'google/hear-pytorch', 'best_epoch': ckpt.get('epoch'),
            'val_macro_f1': ckpt.get('val_macro_f1')}
    (out_dir / f'{task}.json').write_text(json.dumps(meta, indent=2))
    return meta


def load_adapter(adapter_file, backbone_path, num_classes):
    from models_hear_lora import HeARLoRAClassifier
    model = HeARLoRAClassifier(num_classes=num_classes, weights_path=str(backbone_path))
    missing, unexpected = model.load_state_dict(load_file(str(adapter_file)), strict=False)
    assert not unexpected and all(not is_trained(k) for k in missing)
    return model.eval()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='arm directory containing checkpoints/')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from run_clean_pipeline import TASKS  # noqa: E402
    for task, cfg in TASKS.items():
        m = export(Path(a.run_dir) / 'checkpoints' / f'{task}_full.pth', Path(a.out_dir), task,
                   cfg['names'])
        print(task, m['n_tensors'], 'tensors', f"{m['n_parameters']:,} parameters")
