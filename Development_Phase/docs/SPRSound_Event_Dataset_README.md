# SPRSound Event-Level Dataset

## 📋 Genel Bakış

Bu CSV dosyası, SPRSound veri setindeki **tüm event'leri ayrı satırlara** dönüştürerek hazırlanmıştır. Her event için hasta bilgileri, hastalık tanısı, kayıt konumu ve zaman damgaları içerir.

**Dosya**: `SPRSound_Event_Level_Dataset.csv`
**Toplam Event**: 57,123
**Boyut**: 13.24 MB
**Format**: CSV (UTF-8)

---

## 🎯 Kullanım Amaçları

### 1. **Data Augmentation**
Her event ayrı satırda olduğu için:
- Event bazlı augmentation kolaylaşır
- Spectrogram'lar event'lere göre kesilebilir
- Her event için ayrı özellik çıkarımı yapılabilir

### 2. **Event-Level Classification**
- Fine Crackle, Wheeze, Rhonchi vb. tespiti
- Multi-label classification
- Temporal segmentation

### 3. **Disease-Sound Correlation Analysis**
- Hangi hastalıkta hangi sesler var?
- Hastalık-event ilişkileri
- Klinik pattern discovery

### 4. **Time-Series Analysis**
- Event start/end timestamps
- Event duration analysis
- Temporal patterns

---

## 📊 CSV Yapısı (18 Kolon)

| # | Kolon Adı | Açıklama | Örnek |
|---|-----------|----------|-------|
| 1 | `dataset` | Hangi dataset'ten geldiği | `Classification-Train` |
| 2 | `file_path` | JSON dosyasının tam yolu | `/home/.../41251473_2.7_1_p1_2856.json` |
| 3 | `filename` | JSON dosya adı | `41251473_2.7_1_p1_2856.json` |
| 4 | `patient_number` | Hasta numarası | `41251473` |
| 5 | `age` | Yaş (yıl) | `2.7` |
| 6 | `gender` | Cinsiyet (açık) | `Female` / `Male` |
| 7 | `gender_code` | Cinsiyet kodu | `0` (Erkek) / `1` (Kadın) |
| 8 | `recording_location` | Kayıt konumu kodu | `p1`, `p2`, `p3`, `p4` |
| 9 | `recording_location_name` | Kayıt konumu (açık) | `Left Posterior` |
| 10 | `recording_number` | Kayıt numarası | `2856` |
| 11 | `record_annotation` | Kayıt seviyesi etiket | `Normal`, `DAS`, `CAS`, `CAS & DAS` |
| 12 | `disease` | Hastalık tanısı | `Pneumonia (non-severe)` |
| 13 | `event_start_ms` | Event başlangıç (ms) | `790` |
| 14 | `event_end_ms` | Event bitiş (ms) | `2057` |
| 15 | `event_duration_ms` | Event süresi (ms) | `1267` |
| 16 | `event_type` | Event tipi | `Normal`, `Fine Crackle`, `Wheeze` |
| 17 | `event_index` | Dosyadaki kaçıncı event | `1`, `2`, `3` ... |
| 18 | `total_events_in_file` | Dosyadaki toplam event | `6` |

---

## 📖 Kullanım Örnekleri

### Python ile Yükleme

```python
import pandas as pd

# CSV'yi yükle
df = pd.read_csv('SPRSound_Event_Level_Dataset.csv')

print(f"Toplam event: {len(df):,}")
print(f"Kolonlar: {list(df.columns)}")
```

### 1. Training Seti Filtrele

```python
# Classification training eventleri
train_events = df[df['dataset'] == 'Classification-Train']
print(f"Training events: {len(train_events):,}")

# Detection training eventleri
detection_train = df[df['dataset'] == 'Detection-Train']
```

### 2. Event Type'a Göre Filtrele

```python
# Fine Crackle eventleri
fine_crackle = df[df['event_type'] == 'Fine Crackle']
print(f"Fine Crackle events: {len(fine_crackle):,}")

# Wheeze eventleri
wheeze = df[df['event_type'] == 'Wheeze']
print(f"Wheeze events: {len(wheeze):,}")

# Normal olmayan eventler
adventitious = df[df['event_type'] != 'Normal']
```

### 3. Hastalığa Göre Filtrele

```python
# Pneumonia eventleri
pneumonia = df[df['disease'] == 'Pneumonia (non-severe)']

# Asthma eventleri
asthma = df[df['disease'] == 'Asthma']

# Bronchiolitis (RSV!) eventleri
bronchiolitis = df[df['disease'] == 'Bronchiolitis']
print(f"Bronchiolitis events: {len(bronchiolitis):,}")
```

### 4. Yaş Grubu Filtrele

```python
# Yaşı numeric'e çevir
df['age_numeric'] = pd.to_numeric(df['age'], errors='coerce')

# 0-2 yaş bebek eventleri (RSV risk group)
infant_events = df[df['age_numeric'] <= 2]

# 3-6 yaş çocuk eventleri
preschool_events = df[(df['age_numeric'] > 2) & (df['age_numeric'] <= 6)]
```

### 5. Event Süresi Analizi

```python
# Event sürelerini numeric'e çevir
df['duration_sec'] = pd.to_numeric(df['event_duration_ms'], errors='coerce') / 1000

# Ortalama event süreleri (event type'a göre)
avg_duration = df.groupby('event_type')['duration_sec'].mean()
print(avg_duration)

# Uzun eventleri filtrele (>2 saniye)
long_events = df[df['duration_sec'] > 2]
```

### 6. Augmentation İçin Event Çıkar

```python
# Bir WAV dosyasından event'leri çıkar
import librosa
import numpy as np

# Örnek: Fine Crackle event'i al
event = fine_crackle.iloc[0]

# WAV dosya yolu (JSON yerine WAV)
wav_path = event['file_path'].replace('.json', '.wav').replace('_json', '_wav')

# Audio yükle
audio, sr = librosa.load(wav_path, sr=None)

# Event'i kes
start_sample = int(float(event['event_start_ms']) / 1000 * sr)
end_sample = int(float(event['event_end_ms']) / 1000 * sr)
event_audio = audio[start_sample:end_sample]

# Event'i kaydet veya işle
# librosa.output.write_wav(f"event_{event['event_type']}.wav", event_audio, sr)
```

### 7. Hastalık-Ses Korelasyonu

```python
# Hastalık bazında event type dağılımı
disease_event_matrix = pd.crosstab(
    df['disease'], 
    df['event_type'], 
    normalize='index'
) * 100

print(disease_event_matrix)

# Örnek: Bronchiolitis'te hangi sesler var?
bronch_sounds = df[df['disease'] == 'Bronchiolitis']['event_type'].value_counts()
print("\nBronchiolitis Sound Distribution:")
print(bronch_sounds)
```

### 8. Bir Hastanın Tüm Eventleri

```python
# Belirli bir hastaya ait tüm eventler
patient_id = '41251473'
patient_events = df[df['patient_number'] == patient_id]

print(f"Hasta {patient_id}:")
print(f"  Toplam event: {len(patient_events)}")
print(f"  Hastalık: {patient_events.iloc[0]['disease']}")
print(f"  Yaş: {patient_events.iloc[0]['age']}")
print("\nEvent Types:")
print(patient_events['event_type'].value_counts())
```

### 9. Dataset Split (Train/Val/Test)

```python
from sklearn.model_selection import train_test_split

# Training eventleri al
train_df = df[df['dataset'] == 'Classification-Train']

# Event type'a göre stratified split
X = train_df.drop(['event_type'], axis=1)
y = train_df['event_type']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)
```

### 10. RSV Proxy Detection

```python
# RSV için proxy: Bronchiolitis + Yaş <2
rsv_proxy = df[
    (df['disease'] == 'Bronchiolitis') & 
    (df['age_numeric'] < 2)
]

print(f"Muhtemel RSV events: {len(rsv_proxy):,}")

# RSV karakteristik ses: Wheeze + Crackle
rsv_sounds = rsv_proxy[
    (rsv_proxy['event_type'] == 'Wheeze') | 
    (rsv_proxy['event_type'] == 'Fine Crackle') |
    (rsv_proxy['event_type'] == 'Wheeze+Crackle')
]
```

---

## 📊 Dataset İstatistikleri

### Dataset Dağılımı

| Dataset | Event Sayısı |
|---------|--------------|
| Detection-Train | 9,785 |
| Detection-Test-2024 | 7,659 |
| BioCAS2024-Test | 7,659 |
| BioCAS2022-Train | 6,833 |
| Classification-Train | 6,833 |
| BioCAS2025-Test | 4,706 |
| Classification-Valid-2023 | 3,167 |
| BioCAS2023-Test | 3,167 |
| Detection-Valid | 2,428 |
| BioCAS2022-Test-Inter | 1,437 |
| Classification-Valid-2022-Inter | 1,437 |
| BioCAS2022-Test-Intra | 1,006 |
| Classification-Valid-2022-Intra | 1,006 |

### Event Type Dağılımı

| Event Type | Sayı | Yüzde |
|------------|------|-------|
| Normal | 44,714 | 78.91% |
| Fine Crackle | 6,651 | 11.74% |
| Wheeze | 3,764 | 6.64% |
| Rhonchi | 575 | 1.01% |
| Coarse Crackle | 389 | 0.69% |
| Wheeze+Crackle | 381 | 0.67% |
| Stridor | 189 | 0.33% |

### Top 10 Hastalık

| Hastalık | Event Sayısı |
|----------|--------------|
| Pneumonia (non-severe) | 29,126 |
| Bronchitis | 5,376 |
| Control Group | 4,950 |
| Asthma | 4,255 |
| Pneumonia (severe) | 3,356 |
| Other respiratory diseases | 1,014 |
| Bronchiolitis | 909 |
| Bronchiectasia | 891 |
| Hemoptysis | 687 |
| Acute upper respiratory infection | 511 |

### Record Annotation Dağılımı

| Record Annotation | Sayı | Yüzde |
|-------------------|------|-------|
| Unknown | 32,237 | 56.43% |
| Normal | 16,562 | 28.99% |
| CAS | 3,168 | 5.55% |
| DAS | 3,114 | 5.45% |
| CAS & DAS | 1,582 | 2.77% |
| Poor Quality | 460 | 0.81% |

---

## 🎯 Augmentation Stratejileri

### 1. Event-Based Segmentation

```python
# Her event için ayrı audio segment
def extract_event_segments(csv_file):
    df = pd.read_csv(csv_file)
    
    for idx, row in df.iterrows():
        wav_path = row['file_path'].replace('.json', '.wav').replace('_json', '_wav')
        
        if os.path.exists(wav_path):
            audio, sr = librosa.load(wav_path, sr=None)
            
            start = int(float(row['event_start_ms']) / 1000 * sr)
            end = int(float(row['event_end_ms']) / 1000 * sr)
            
            event_audio = audio[start:end]
            
            # Kaydet veya işle
            output_path = f"events/{row['event_type']}_{idx}.wav"
            # librosa.output.write_wav(output_path, event_audio, sr)
```

### 2. Class Balancing

```python
from imblearn.over_sampling import SMOTE

# Rare class'ları oversample et
rare_classes = ['Stridor', 'Wheeze+Crackle', 'Coarse Crackle']
rare_events = df[df['event_type'].isin(rare_classes)]

# Bu event'leri daha fazla kullan
```

### 3. Time-Stretching & Pitch-Shifting

```python
import librosa.effects as effects

# Event audio'yu augment et
def augment_event_audio(audio, sr):
    # Time stretch
    audio_stretched = effects.time_stretch(audio, rate=1.1)
    
    # Pitch shift
    audio_pitched = effects.pitch_shift(audio, sr=sr, n_steps=2)
    
    return audio_stretched, audio_pitched
```

---

## ⚠️ Önemli Notlar

### 1. **Unknown Disease**
- Bazı hastalar patient summary CSV'de yok → `disease = 'Unknown'`
- Toplam 32,237 event'te hastalık bilgisi yok

### 2. **No Event**
- Bazı JSON dosyalarında `event_annotation` boş
- Bu kayıtlar `event_type = 'No Event'` olarak işaretlendi

### 3. **WAV Dosya Yolları**
- CSV'de JSON yolu var
- WAV yolu: `.json` → `.wav` ve `_json` → `_wav` değiştir

### 4. **Event Index**
- Aynı dosyadaki event'ler `event_index` ile sıralı
- `total_events_in_file` ile dosyadaki toplam event sayısı

### 5. **Sınıf Dengesizliği**
- Normal eventler %79 → Ağırlıklı loss kullan
- Rare events (<1%) → Augmentation şart

---

## 🔬 Araştırma Kullanımları

### 1. **RSV Detection**
- Bronchiolitis + Yaş <2 filtrele
- Wheeze + Fine Crackle paterni ara
- Temporal features kullan

### 2. **Disease Classification**
- Event-level features → Disease prediction
- Aggregate event'leri → Record-level prediction

### 3. **Sound Pattern Analysis**
- Hastalık-Ses korelasyonu
- Event süre analizi
- Temporal patterns

### 4. **Multi-task Learning**
- Record + Event level etiketler birlikte
- Hierarchical classification

---

## 📚 İlgili Dosyalar

- `create_event_level_dataset.py` - CSV oluşturma scripti
- `SPRSound_TUM_HASTALIKLAR_DETAYLI_RAPOR.md` - Hastalık analizleri
- `SPRSound_VERI_SETI_ANALIZ_RAPORU.md` - Veri seti genel rapor

---

## 🚀 Hızlı Başlangıç

```python
import pandas as pd

# 1. CSV'yi yükle
df = pd.read_csv('SPRSound_Event_Level_Dataset.csv')

# 2. Training eventleri al
train = df[df['dataset'] == 'Classification-Train']

# 3. Fine Crackle eventleri filtrele
crackles = train[train['event_type'] == 'Fine Crackle']

# 4. WAV dosyalarını işle
for idx, event in crackles.head(10).iterrows():
    wav_path = event['file_path'].replace('.json', '.wav').replace('_json', '_wav')
    print(f"Event: {event['event_type']}, Duration: {event['event_duration_ms']} ms")
    print(f"WAV: {wav_path}")
```

---

**Hazırlayan**: AI Assistant  
**Tarih**: 2026-01-14  
**Veri Seti**: SPRSound (2022-2025)  
**Toplam Event**: 57,123  
**Script**: create_event_level_dataset.py
