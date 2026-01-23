# Training Tamamlandı - Sonraki Adımlar

## 1. Model Kontrolü

Training tamamlandı. En iyi model epoch 49'da (val_auprc: 0.433).

**Mevcut dosyalar:**
- `models/hear_bilstm_checkpoints/last.pth` - Son epoch modeli
- `models/hear_bilstm_checkpoints/training_history.json` - Training geçmişi

**Not:** Best model otomatik kaydedilmemiş. Epoch 49'daki modeli kullanabilirsiniz veya `last.pth` kullanabilirsiniz.

## 2. Inference (Tahmin Yapma)

Tek bir audio dosyası için tahmin yapmak:

```bash
cd EventDetect
python scripts/inference_hear_bilstm.py \
    --audio <audio_dosyasi_yolu> \
    --checkpoint models/hear_bilstm_checkpoints/last.pth \
    --output results/prediction.json
```

**Örnek:**
```bash
python scripts/inference_hear_bilstm.py \
    --audio ../SPRSound/BioCAS2022/train/wav/3246_1.0_0_p1_1.wav \
    --checkpoint models/hear_bilstm_checkpoints/last.pth \
    --output results/sample_prediction.json
```

## 3. Evaluation (Değerlendirme)

Sample audio dosyaları üzerinde değerlendirme yapmak:

```bash
python scripts/evaluate_hear_bilstm_samples.py \
    --checkpoint models/hear_bilstm_checkpoints/last.pth \
    --output-dir results/hear_bilstm_evaluation
```

Bu script:
- Timeline görselleştirmeleri oluşturur
- Probability plot'ları oluşturur
- Detaylı metrikler hesaplar
- Summary report oluşturur

## 4. Training History Analizi

Training geçmişini görselleştirmek için:

```python
import json
import matplotlib.pyplot as plt

with open('models/hear_bilstm_checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

train_loss = [e['loss'] for e in history['train']]
val_loss = [e['loss'] for e in history['val']]
train_f1 = [e['f1'] for e in history['train']]
val_f1 = [e['f1'] for e in history['val']]
val_auprc = [e['auprc'] for e in history['val']]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(train_f1, label='Train')
plt.plot(val_f1, label='Val')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()
plt.title('F1 Score')

plt.subplot(1, 3, 3)
plt.plot(val_auprc, label='Val')
plt.xlabel('Epoch')
plt.ylabel('AUPRC')
plt.legend()
plt.title('Validation AUPRC')

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=150)
plt.show()
```

## 5. Model Performansı

Son epoch metrikleri:
- **Train**: Loss: 0.9661, AUROC: 0.7619, AUPRC: 0.4385, F1: 0.3808
- **Val**: Loss: 0.9217, AUROC: 0.7552, AUPRC: 0.4130, F1: 0.3573

En iyi validation AUPRC: **0.433** (Epoch 49)

## 6. Öneriler

1. **Hyperparameter Tuning**: 
   - Learning rate, batch size, BiLSTM hidden_dim ayarlanabilir
   - Overlap ratio threshold denenebilir

2. **Model İyileştirme**:
   - HeAR encoder'ı fine-tune edilebilir (config'de `HEAR_ENCODER_FROZEN = False`)
   - Daha fazla epoch ile training yapılabilir

3. **Post-processing**:
   - Threshold değeri optimize edilebilir
   - Min duration filtresi ayarlanabilir

4. **Evaluation**:
   - Test seti üzerinde değerlendirme yapılmalı
   - Segment-level metrikler detaylı incelenmeli
