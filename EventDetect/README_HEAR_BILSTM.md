# HeAR + BiLSTM Temporal Event Segmentation

Bu modül, solunum seslerinde temporal event segmentation için HeAR encoder ve BiLSTM tabanlı bir derin öğrenme modeli içerir. Model, ses dosyalarındaki patolojik event'lerin (crackles, wheezing, rhonchi vb.) zaman içindeki konumlarını tespit eder.

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [HeAR Encoder](#hear-encoder)
- [Veri İşleme](#veri-işleme)
- [Model Yapısı](#model-yapısı)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Dosya Yapısı](#dosya-yapısı)
- [Kullanım Örnekleri](#kullanım-örnekleri)

## 🎯 Genel Bakış

### Problem

Solunum seslerinde event detection için mevcut clip-based yaklaşımlar agresif segment birleştirme yapıyor ve büyük bloklar oluşturuyor. Bu, gerçek event'lerin zaman içindeki konumlarını doğru bir şekilde tespit etmeyi zorlaştırıyor.

### Çözüm

HeAR encoder + BiLSTM tabanlı temporal segmentation modeli:
- **Pre-trained Encoder**: HeAR (HuggingFace) ile güçlü audio embeddings
- **Temporal Modeling**: BiLSTM ile sequence-level context öğrenimi
- **Frame-level Prediction**: Her window için event probability
- **Binary Classification**: Sadece "event var mı yok mu?" (event tipi önemli değil)
- **Sliding Window**: 2s windows ile temporal resolution

### Pipeline

```
Raw Audio Signal (16kHz, mono)
    ↓
Sliding Window (2s, hop_sec)
    ↓
HeAR Encoder (frozen/fine-tunable)
    ↓
Frame-level Embeddings (512-d per window)
    ↓
BiLSTM Sequence Tagger
    ↓
Window-level Binary Labels (event var/yok)
    ↓
Post-processing (threshold, merge, min_duration)
    ↓
Event Segments (start_sec, end_sec)
```

## 🏗️ Mimari

### HeAR Encoder

HeAR (Holistic Evaluation of Audio Representations) encoder, HuggingFace'den yüklenen pre-trained bir modeldir:
- **Model**: `google/hear-pytorch`
- **Input**: 2 saniye audio (16kHz, mono, 32000 samples)
- **Output**: 512-dimensional embedding
- **Mode**: Frozen (default) veya fine-tunable

**Not**: HeAR modelini kullanmak için HuggingFace'de terms/conditions kabul etmeniz gerekebilir:
1. https://huggingface.co/google/hear-pytorch adresine gidin
2. Terms'i kabul edin
3. `huggingface-cli login` ile giriş yapın

### BiLSTM Model

BiLSTM sequence tagger:
- **Input**: (batch, T, 512) - HeAR embeddings
- **Architecture**: 
  - BiLSTM: hidden_dim=256, num_layers=2, dropout=0.2
  - Linear projection: (hidden_dim*2) → 1
- **Output**: (batch, T) - Binary logits per window
- **Variable-length**: `pack_padded_sequence` ile handling

### Model Özellikleri

- **Input**: HeAR embeddings `(batch, T_windows, 512)`
- **Output**: Binary logits `(batch, T_windows)` - Event probability per window
- **Parameters**: ~1-2M (BiLSTM only, encoder frozen)
- **Activation**: Sigmoid (binary probabilities)

## 📊 Veri İşleme

### Dataset Sınıfı

`HeAREmbeddingDataset` sınıfı:

1. **Audio Loading**: WAV dosyalarını yükler (16kHz, mono)
2. **Embedding Extraction**: HeAR encoder ile sliding window embeddings
   - Window: 2.0s (HeAR requirement)
   - Hop: 0.25s (configurable)
   - Caching: Embeddings otomatik olarak cache'lenir
3. **Ground Truth Labels**: JSON annotations'dan window-level binary labels
   - Her window için event segmentleriyle overlap ratio hesaplanır
   - Overlap ratio >= threshold → label=1, else 0
   - Default threshold: 0.2

### Ground Truth Format

Her window için binary label:
- `0` = Normal (event yok)
- `1` = Event (herhangi bir patolojik event var)

### Embedding Caching

Embeddings otomatik olarak cache'lenir:
- **Cache Key**: audio_path + encoder_version + window_sec + hop_sec + sample_rate
- **Format**: `.npz` (compressed numpy)
- **Location**: `models/hear_embeddings_cache/`
- **Override**: `--recompute-embeddings` flag ile yeniden hesaplanabilir

## 🎓 Training

### Komut

```bash
cd EventDetect
python scripts/train_hear_bilstm.py [OPTIONS]
```

### Seçenekler

- `--recompute-embeddings`: Embeddings'leri yeniden hesapla (cache'i atla)
- `--device`: Device seçimi (cuda/cpu)

### Örnek

```bash
# Normal training (cached embeddings kullan)
python scripts/train_hear_bilstm.py

# Embeddings'leri yeniden hesapla
python scripts/train_hear_bilstm.py --recompute-embeddings

# CPU'da çalıştır
python scripts/train_hear_bilstm.py --device cpu
```

### Training Parametreleri

Config dosyasından (`config_hear_bilstm.py`):
- `BATCH_SIZE`: 16 (embedding sequences)
- `LEARNING_RATE`: 1e-3
- `NUM_EPOCHS`: 50
- `BEST_BY`: "val_auprc" veya "val_f1"
- `POS_WEIGHT_MODE`: "auto" (class imbalance için)

### Checkpoints

- `models/hear_bilstm_checkpoints/best.pth`: Best model (val AUPRC/F1)
- `models/hear_bilstm_checkpoints/last.pth`: Last epoch
- `models/hear_bilstm_checkpoints/training_history.json`: Training history

## 🔮 Inference

### Komut

```bash
python scripts/inference_hear_bilstm.py --audio <audio_path> [OPTIONS]
```

### Seçenekler

- `--audio`: Audio dosyası path (required)
- `--checkpoint`: Model checkpoint path (default: best.pth)
- `--threshold`: Probability threshold (default: 0.5)
- `--output`: Output JSON path
- `--min-duration`: Minimum segment duration (default: 0.1s)
- `--device`: Device seçimi

### Örnek

```bash
# Basit inference
python scripts/inference_hear_bilstm.py --audio data/sample.wav

# Custom threshold ve output
python scripts/inference_hear_bilstm.py \
    --audio data/sample.wav \
    --threshold 0.6 \
    --output results/prediction.json
```

### Output Format

```json
{
  "audio_path": "data/sample.wav",
  "duration_sec": 10.5,
  "window_sec": 2.0,
  "hop_sec": 0.25,
  "num_windows": 35,
  "threshold": 0.5,
  "num_segments": 3,
  "segments": [[1.2, 2.5], [4.0, 5.8], [8.1, 9.3]],
  "window_probs": [0.1, 0.2, 0.8, 0.9, ...],
  "mean_event_prob": 0.45,
  "max_event_prob": 0.95,
  "min_event_prob": 0.05
}
```

## 📈 Evaluation

### Komut

```bash
python scripts/evaluate_hear_bilstm_samples.py [OPTIONS]
```

### Çıktılar

Her sample için:
- `timeline.png`: Waveform + GT segments + predicted segments
- `prob_plot.png`: Window probabilities over time
- `results.json`: Tüm sonuçlar
- `summary_report.md`: Özet rapor

### Örnek

```bash
python scripts/evaluate_hear_bilstm_samples.py \
    --checkpoint models/hear_bilstm_checkpoints/best.pth \
    --output-dir results/hear_bilstm_evaluation
```

## 🔧 Offline Embedding Extraction

Embeddings'leri önceden hesaplayarak training'i hızlandırabilirsiniz:

```bash
python scripts/extract_hear_embeddings.py
```

Seçenekler:
- `--recompute`: Cache'i atla, tüm embeddings'leri yeniden hesapla
- `--device`: Device seçimi

## 📁 Dosya Yapısı

```
EventDetect/
  config_hear_bilstm.py              # Configuration
  models/
    hear_encoder.py                  # HeAR encoder wrapper
    bilstm_event_detector.py         # BiLSTM model
    hear_bilstm_checkpoints/          # Model checkpoints
    hear_embeddings_cache/           # Cached embeddings
  data_processing/
    hear_embedding_dataset.py         # Dataset class
  utils/
    collate_seq.py                   # Sequence collate function
    metrics_seq.py                   # Sequence metrics
    postprocess_segments.py          # Post-processing
    seed.py                          # Seed utility
  scripts/
    train_hear_bilstm.py             # Training script
    inference_hear_bilstm.py          # Inference script
    evaluate_hear_bilstm_samples.py   # Evaluation script
    extract_hear_embeddings.py       # Embedding extraction
  results/
    hear_bilstm_evaluation/         # Evaluation results
  README_HEAR_BILSTM.md              # This file
```

## 🎯 Önemli Notlar

### Event Tipi Değil, Event Var/Yok

**Kritik**: Bu model sadece "event var mı yok mu?" sorusunu cevaplar. Event tipleri (crackle/wheeze/rhonchi) ile ilgilenmez. Event bulunduktan sonra downstream model ayrı olarak event tipini sınıflandırabilir.

### HeAR Model Gereksinimleri

HeAR encoder 2 saniye audio window gerektirir:
- **Window Size**: 2.0s (sabit, HeAR requirement)
- **Hop Size**: 0.25s (configurable, default)
- **Sample Rate**: 16kHz (HeAR requirement)

### Embedding Caching

Embeddings otomatik olarak cache'lenir. İlk çalıştırmada tüm embeddings hesaplanır, sonraki çalıştırmalarda cache'den yüklenir. Bu training süresini önemli ölçüde kısaltır.

### Variable-Length Sequences

Her audio farklı uzunlukta olduğu için:
- Custom collate function ile padding
- `pack_padded_sequence` ile efficient LSTM processing
- Attention masks ile valid positions tracking

## 🔄 U-Net ile Karşılaştırma

| Özellik | U-Net | HeAR+BiLSTM |
|---------|-------|-------------|
| Encoder | Mel-Spectrogram | HeAR (pre-trained) |
| Temporal Model | U-Net (CNN) | BiLSTM |
| Input | Spectrogram | Embeddings |
| Resolution | Frame-level | Window-level |
| Pre-training | None | HeAR (audio) |
| Parameters | ~17M | ~1-2M (BiLSTM) |

Her iki yaklaşım da binary event detection yapar. HeAR+BiLSTM, pre-trained encoder avantajı sağlar ve daha az parametre kullanır.

## 📚 Referanslar

- HeAR: https://github.com/google-research/hear
- HuggingFace HeAR: https://huggingface.co/google/hear-pytorch
- BiLSTM: Standard bidirectional LSTM for sequence tagging
