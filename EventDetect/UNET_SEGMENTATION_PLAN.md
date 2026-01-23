# U-Net Temporal Segmentation Model Plan

## Problem
Mevcut clip-based yaklaşım çok agresif segment birleştirme yapıyor ve büyük bloklar oluşturuyor. U-Net tabanlı temporal segmentation daha iyi sonuçlar verebilir.

## Architecture

### Input Pipeline
```
Raw Audio Signal (16kHz, mono)
    ↓
STFT (Short-Time Fourier Transform)
    ↓
Mel-Spectrogram (time x frequency)
    ↓
U-Net Encoder-Decoder
    ↓
Segmentation Mask (time x num_classes)
```

### U-Net Architecture
- **Input**: STFT/Mel-Spectrogram (time_steps x freq_bins)
- **Output**: Segmentation mask (time_steps x num_classes)
  - Binary: (time_steps x 1) - Event var/yok
  - Multi-class: (time_steps x 3) - Normal, Crackles, Rhonchi

### Model Structure
```
Encoder (Downsampling):
  - Conv2D blocks with MaxPooling
  - Feature extraction from spectrogram

Bottleneck:
  - Dense feature representation

Decoder (Upsampling):
  - Transposed Conv2D blocks
  - Skip connections from encoder
  - Temporal resolution restoration

Output Head:
  - Conv2D to num_classes
  - Temporal mask prediction
```

## Data Preparation

### Ground Truth Format
Her zaman noktası için label:
- Binary: 0=Normal, 1=Event
- Multi-class: 0=Normal, 1=Crackles, 2=Rhonchi

### Data Loading
1. Audio dosyasını yükle (16kHz)
2. STFT/Mel-Spectrogram hesapla
3. Ground truth mask oluştur (JSON'dan event timings)
4. Time alignment (STFT time steps ↔ event timings)

## Training Strategy

### Loss Function
- Binary: Binary Cross-Entropy + Dice Loss
- Multi-class: Categorical Cross-Entropy + Dice Loss

### Metrics
- IoU (Intersection over Union) per time step
- Temporal F1 score
- Boundary accuracy

### Data Augmentation
- Time stretching
- Frequency masking
- Noise injection

## Implementation Steps

1. **Data Preprocessing Module**
   - STFT/Mel-Spectrogram extraction
   - Ground truth mask generation from JSON
   - Dataset class for PyTorch

2. **U-Net Model**
   - Encoder-Decoder architecture
   - Skip connections
   - Temporal segmentation head

3. **Training Script**
   - Loss function
   - Metrics calculation
   - Checkpointing

4. **Inference Script**
   - Model loading
   - Temporal mask prediction
   - Post-processing (threshold, smoothing)

5. **Evaluation**
   - Temporal IoU
   - Boundary metrics
   - Comparison with clip-based approach

## Advantages

1. **Temporal Resolution**: Her zaman noktası için prediction
2. **No Aggressive Merging**: Segment birleştirme sorunu yok
3. **Context Awareness**: U-Net temporal context'i öğrenir
4. **End-to-End**: Raw signal → segmentation mask
5. **Flexible**: Binary veya multi-class segmentation

## Next Steps

1. Segment birleştirme mantığını düzelt (şimdilik)
2. U-Net model architecture'ı tasarla
3. Data preprocessing pipeline'ı oluştur
4. Training script'i yaz
5. Inference ve evaluation
