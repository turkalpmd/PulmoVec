# U-Net Temporal Event Segmentation

Bu proje, solunum seslerinde (respiratory sounds) temporal event segmentation için U-Net tabanlı bir derin öğrenme modeli içerir. Model, ses dosyalarındaki patolojik event'lerin (crackles, wheezing, rhonchi vb.) zaman içindeki konumlarını tespit eder.

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
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

U-Net tabanlı temporal segmentation modeli:
- **Temporal Resolution**: Her zaman noktası için event probability prediction
- **No Aggressive Merging**: Segment birleştirme sorunu yok
- **Context Awareness**: U-Net temporal context'i öğrenir
- **End-to-End**: Raw signal → STFT → U-Net → segmentation mask
- **Binary Segmentation**: Event var/yok (event tipi önemli değil)

### Pipeline

```
Raw Audio Signal (16kHz, mono)
    ↓
Mel-Spectrogram (time x frequency)
    ↓
U-Net Encoder-Decoder
    ↓
Binary Temporal Mask (time x 1)
    ↓
Event Segments (start_sec, end_sec)
```

## 🏗️ Mimari

### U-Net Modeli

U-Net, encoder-decoder mimarisine sahip bir convolutional neural network'tür:

```
Encoder (Downsampling):
  - Conv2D blocks with MaxPooling
  - Feature extraction from spectrogram
  - 4 downsampling stages

Bottleneck:
  - Dense feature representation
  - 1024 channels

Decoder (Upsampling):
  - Transposed Conv2D blocks
  - Skip connections from encoder
  - Temporal resolution restoration
  - 4 upsampling stages

Output Head:
  - Conv2D to 1 channel
  - Binary temporal mask (Event probability)
```

### Model Özellikleri

- **Input**: Mel-Spectrogram `(batch, 1, time_steps, 128)`
- **Output**: Binary mask `(batch, 1, time_steps, 1)` - Event probability per time step
- **Parameters**: ~17.2M
- **Activation**: Sigmoid (binary probabilities)

## 📊 Veri İşleme

### Dataset Sınıfı

`TemporalSegmentationDataset` sınıfı:

1. **Audio Loading**: WAV dosyalarını yükler (16kHz, mono)
2. **Spectrogram Extraction**: Mel-Spectrogram hesaplar
   - `n_fft=2048`, `hop_length=512`, `n_mels=128`
3. **Ground Truth Mask**: JSON annotation'lardan binary temporal mask oluşturur
   - Normal events → 0
   - Pathological events (Crackles, Wheeze, Rhonchi, vb.) → 1

### Ground Truth Format

Her zaman noktası için binary label:
- `0` = Normal
- `1` = Event (herhangi bir patolojik event)

### Variable-Length Handling

Farklı uzunluktaki audio dosyaları için:
- **Custom Collate Function**: Batch içinde padding
- Padding değeri: `0.0` (Normal class)
- Her batch, batch içindeki en uzun spectrogram'a göre pad edilir

## 🧠 Model Yapısı

### Loss Function

**Combined Loss**: Binary Cross-Entropy + Dice Loss

```python
Loss = BCE_WEIGHT * BCE + DICE_WEIGHT * DiceLoss
```

- **BCE**: Pixel-level classification için iyi
- **Dice Loss**: Class imbalance için iyi
- **Weights**: `BCE_WEIGHT=1.0`, `DICE_WEIGHT=1.0`

### Metrics

- **Temporal IoU**: Intersection over Union per time step
- **Precision/Recall/F1**: Segment-level performance
- **Accuracy**: Overall classification accuracy

## 🚀 Training

### Hyperparameters

```python
BATCH_SIZE = 8          # Smaller batch for full audio files
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_RATIO = 0.8
```

### Training Process

1. **Data Loading**: 
   - 6,567 unique audio files
   - Train: 5,253 samples
   - Val: 1,314 samples

2. **Training Loop**:
   - Forward pass: Spectrogram → U-Net → Binary mask
   - Loss calculation: Combined BCE + Dice
   - Backward pass: Gradient update
   - Metrics: IoU, Precision, Recall, F1

3. **Learning Rate Scheduling**:
   - `ReduceLROnPlateau`: Validation loss'a göre LR azaltma
   - Factor: 0.5, Patience: 5 epochs

4. **Checkpointing**:
   - Best model: Highest validation IoU
   - Last model: Final epoch checkpoint
   - Training history: JSON format

### Training Script

```bash
python EventDetect/scripts/train_unet.py
```

**Output**:
- `EventDetect/models/unet_checkpoints/best.pth` - Best model
- `EventDetect/models/unet_checkpoints/last.pth` - Last model
- `EventDetect/models/unet_checkpoints/training_history.json` - Training metrics

## 🔮 Inference

### Inference Script

```bash
python EventDetect/scripts/inference_unet.py \
    --audio <path_to_audio.wav> \
    --checkpoint <path_to_checkpoint.pth> \
    --threshold 0.5 \
    --output results.json \
    --min-duration 0.1
```

### Inference Process

1. **Preprocessing**: Audio → Mel-Spectrogram
2. **Prediction**: U-Net → Temporal mask (probabilities)
3. **Post-processing**: 
   - Thresholding (default: 0.5)
   - Segment extraction
   - Minimum duration filtering (default: 0.1s)

### Output Format

```json
{
  "audio_path": "...",
  "duration_sec": 15.36,
  "num_segments": 3,
  "segments": [
    [2.5, 4.2],
    [7.1, 9.8],
    [12.3, 14.5]
  ],
  "temporal_mask": [0, 0, 1, 1, ...],
  "temporal_probs": [0.1, 0.2, 0.8, 0.9, ...],
  "mean_event_prob": 0.45
}
```

## 📈 Evaluation

### Evaluation Script

3 örnek audio dosyası için (normal, crackles, wheezing) performans değerlendirmesi:

```bash
python EventDetect/scripts/evaluate_unet_samples.py
```

### Evaluation Metrics

- **Temporal IoU**: Mean, median IoU
- **Precision/Recall/F1 @ IoU thresholds**: 0.3, 0.5, 0.7
- **Boundary Metrics**: Onset/offset errors, boundary F1
- **False Positives per Hour**: FP rate

### Output

Her örnek için:
- `results.json`: Metrics ve predictions
- `timeline.png`: Waveform + GT vs Predicted segments
- `probabilities.png`: Temporal event probabilities

Summary report:
- `summary_report.md`: Average metrics across all samples

## 📁 Dosya Yapısı

```
EventDetect/
├── README.md                          # Bu dosya
├── UNET_SEGMENTATION_PLAN.md          # İlk plan dokümantasyonu
├── config.py                          # Genel config (HeAR için)
├── config_unet.py                     # U-Net training config
│
├── models/
│   ├── __init__.py
│   ├── unet_segmentation.py          # U-Net model tanımı
│   └── unet_checkpoints/              # Model checkpoints
│       ├── best.pth
│       ├── last.pth
│       └── training_history.json
│
├── data_processing/
│   ├── __init__.py
│   └── spectrogram_dataset.py        # Dataset sınıfı
│
├── utils/
│   ├── __init__.py
│   ├── losses.py                      # BCE + Dice Loss
│   ├── metrics.py                     # Temporal metrics
│   └── collate_fn.py                  # Variable-length padding
│
├── scripts/
│   ├── train_unet.py                  # Training script
│   ├── inference_unet.py              # Inference script
│   └── evaluate_unet_samples.py      # Evaluation script
│
├── evaluation/
│   └── temporal_metrics.py            # Segmentation metrics
│
├── samples/
│   └── selected_samples.json          # 3 örnek dosya bilgisi
│
└── results/
    ├── unet_evaluation/               # Evaluation results
    │   ├── sample_normal/
    │   ├── sample_crackles/
    │   ├── sample_wheezing/
    │   └── summary_report.md
    └── ...
```

## 💡 Kullanım Örnekleri

### 1. Model Training

```bash
# Training başlat
python EventDetect/scripts/train_unet.py

# Training log'larını takip et
tail -f training.log
```

### 2. Tek Dosya Inference

```bash
# Audio dosyası için event detection
python EventDetect/scripts/inference_unet.py \
    --audio data/sample.wav \
    --output results/sample_results.json \
    --threshold 0.5
```

### 3. Evaluation

```bash
# 3 örnek için evaluation
python EventDetect/scripts/evaluate_unet_samples.py

# Sonuçlar:
# - EventDetect/results/unet_evaluation/sample_*/timeline.png
# - EventDetect/results/unet_evaluation/summary_report.md
```

### 4. Python API

```python
import torch
from EventDetect.models.unet_segmentation import UNetTemporalSegmentation
from EventDetect.scripts.inference_unet import predict_events

# Model yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('EventDetect/models/unet_checkpoints/best.pth', device)

# Event detection
results = predict_events(
    audio_path='data/sample.wav',
    model=model,
    device=device,
    threshold=0.5
)

# Segments
for start, end in results['segments']:
    print(f"Event: {start:.2f}s - {end:.2f}s")
```

## 🔧 Teknik Detaylar

### Spectrogram Parameters

- **Sample Rate**: 16,000 Hz
- **N_FFT**: 2048
- **Hop Length**: 512
- **N_Mels**: 128
- **Window**: Hann window

### Model Architecture Details

- **Encoder Channels**: 64 → 128 → 256 → 512 → 1024
- **Decoder Channels**: 1024 → 512 → 256 → 128 → 64 → 1
- **Skip Connections**: U-Net style (concatenation)
- **Upsampling**: Bilinear interpolation

### Training Details

- **Optimizer**: Adam
- **Learning Rate**: 1e-4 (initial)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Loss**: Combined BCE + Dice
- **Batch Size**: 8 (variable-length padding)

## 📊 Performans Beklentileri

Training sonrası beklenen metrikler:
- **Mean IoU**: > 0.60
- **Precision @ IoU 0.5**: > 0.70
- **Recall @ IoU 0.5**: > 0.65
- **F1 @ IoU 0.5**: > 0.67

## 🐛 Troubleshooting

### 1. CUDA Out of Memory

```python
# config_unet.py'de batch size'ı azalt
BATCH_SIZE = 4  # veya 2
```

### 2. Variable-Length Error

Custom collate function zaten eklendi. Eğer hala hata alıyorsanız:
- `NUM_WORKERS = 0` yapın (single-threaded loading)

### 3. JSON Not Found

Dataset, JSON dosyalarını otomatik bulmaya çalışır. Eğer bulamazsa:
- CSV'de `file_path` kolonunun doğru olduğundan emin olun
- JSON dosyalarının WAV dosyalarıyla aynı dizinde olduğundan emin olun

## 📚 Referanslar

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Dice Loss: Binary segmentation için yaygın kullanım
- Temporal Segmentation: Audio event detection için uygun yaklaşım

## 🔄 HeAR+BiLSTM Alternatifi

Bu repo'da U-Net'e alternatif olarak **HeAR + BiLSTM** tabanlı bir temporal segmentation modülü de bulunmaktadır.

### Özellikler

- **Pre-trained Encoder**: HuggingFace HeAR encoder ile güçlü audio embeddings
- **Temporal Modeling**: BiLSTM ile sequence-level context öğrenimi
- **Sliding Window**: 2s windows ile temporal resolution
- **Embedding Caching**: Performans için otomatik caching

### Kullanım

```bash
# Training
python scripts/train_hear_bilstm.py

# Inference
python scripts/inference_hear_bilstm.py --audio <audio_path>

# Evaluation
python scripts/evaluate_hear_bilstm_samples.py
```

Detaylı dokümantasyon için: [README_HEAR_BILSTM.md](README_HEAR_BILSTM.md)

### Karşılaştırma

| Özellik | U-Net | HeAR+BiLSTM |
|---------|-------|-------------|
| Encoder | Mel-Spectrogram | HeAR (pre-trained) |
| Temporal Model | U-Net (CNN) | BiLSTM |
| Pre-training | None | HeAR (audio) |
| Parameters | ~17M | ~1-2M (BiLSTM) |

Her iki yaklaşım da binary event detection yapar. HeAR+BiLSTM, pre-trained encoder avantajı sağlar.

## 📝 Notlar

- Model binary segmentation yapar (event var/yok)
- Event tipi (Crackles, Wheeze, vb.) önemli değil
- Temporal resolution: Her zaman noktası için prediction
- Variable-length audio dosyaları desteklenir (padding ile)

## 🤝 Katkıda Bulunma

Bu proje, solunum seslerinde temporal event detection için geliştirilmiştir. Sorularınız veya önerileriniz için issue açabilirsiniz.

---

**Son Güncelleme**: 2024
**Versiyon**: 1.0
