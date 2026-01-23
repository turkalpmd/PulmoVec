# SPRSound VERİ SETİ - DETAYLI ANALİZ RAPORU

## 📋 Genel Bakış

**SPRSound (SJTU Paediatric Respiratory Sound Database)**, çocuk popülasyonunda kaydedilmiş ilk açık erişimli solunum sesi veri setidir. Bu veri seti, Shanghai Children's Medical Center (SCMC) çocuk solunum hastalıkları bölümünde Yunting model II stetoskoplar kullanılarak toplanmıştır.

- **Yaş Aralığı**: 1 ay - 18 yaş
- **Kayıt Cihazı**: Yunting model II Stetoskop
- **Format**: WAV (ses) + JSON (etiketler)
- **İki Seviye Etiketleme**: Kayıt seviyesi + Olay seviyesi

---

## 📊 VERİ SETİ İSTATİSTİKLERİ

### Training Set (Eğitim Seti)
- **Toplam Kayıt**: 1,949 kayıt
- **Toplam Event**: 6,656 olay
- **Kayıt Başına Ortalama Event**: 3.42

### Validation/Test Setleri
- **2022 Inter-Test**: 355 kayıt (farklı hastalar arası)
- **2022 Intra-Test**: 379 kayıt (aynı hastalar içinde)
- **2023 Test**: 871 kayıt
- **2024 Test**: 1,704 kayıt
- **2025 Test**: 1,309 kayıt

### Detection Set
- **Training**: 2,660 kayıt
- **Validation**: 664 kayıt
- **Test 2024**: 1,704 kayıt

### Compression Set
- **Training**: 2,844 kayıt
- **Validation**: 710 kayıt
- **Test 2024**: 1,704 kayıt

---

## 🏷️ ETİKET SINIFLANDIRMASI

### 1️⃣ KAYIT SEVİYESİ ETİKETLER (Record Level Annotations)

Kayıt seviyesinde her ses kaydı genel olarak şu kategorilerden birine atanır:

#### **Normal** (66.85% - 1,303 kayıt)
- Yüksek kaliteli kayıt
- Normal solunum sesleri
- Adventitious (anormal) ses yok

#### **DAS - Discontinuous Adventitious Sounds** (12.72% - 248 kayıt)
- **Kesikli Anormal Sesler**
- Crackle (çatırtı) türü sesler içerir
- Akciğer hastalıklarında yaygın (pnömoni, bronşit, fibrozis)

#### **CAS - Continuous Adventitious Sounds** (6.46% - 126 kayıt)
- **Sürekli Anormal Sesler**
- Wheeze (hırıltı), Rhonchi, Stridor türü sesler
- Hava yolu daralması belirtisi (astım, bronşit)

#### **CAS & DAS** (4.87% - 95 kayıt)
- **Hem Sürekli Hem Kesikli Anormal Sesler**
- Kompleks solunum hastalıkları
- Birden fazla patoloji birlikte

#### **Poor Quality** (9.08% - 177 kayıt)
- Düşük sinyal kalitesi
- Gürültülü veya bozuk kayıtlar
- Analiz için uygun değil

---

### 2️⃣ OLAY SEVİYESİ ETİKETLER (Event Level Annotations)

Her kayıt, milisaniye hassasiyetinde zaman damgalı olaylardan oluşur:

#### **Normal** (77.51% - 5,159 event)
- Normal solunum sesleri
- Anormallik yok
- Sağlıklı akciğer sesleri

#### **Fine Crackle** (13.70% - 912 event)
- **İnce/Tiz Çatırtı**
- Kısa süreli, yüksek frekanslı sesler
- **İlişkili Hastalıklar**:
  - Interstisyel Akciğer Hastalığı
  - Pulmoner Fibrozis
  - Pnömoni (erken evre)
  - Kalp Yetmezliği

#### **Wheeze** (6.79% - 452 event)
- **Hırıltı/Vızıltı**
- Yüksek frekanslı, sürekli ses
- **İlişkili Hastalıklar**:
  - Astım
  - Bronşit
  - Kronik Obstrüktif Akciğer Hastalığı (KOAH)
  - Hava yolu daralması

#### **Coarse Crackle** (0.74% - 49 event)
- **Kaba/Pes Çatırtı**
- Düşük frekanslı, daha uzun süreli
- **İlişkili Hastalıklar**:
  - Kronik Bronşit
  - KOAH
  - Bronşektazi
  - Büyük hava yollarında sekresyon

#### **Rhonchi** (0.59% - 39 event)
- **Ronküs**
- Düşük frekanslı, sürekli, gürültülü ses
- **İlişkili Hastalıklar**:
  - Akut/Kronik Bronşit
  - Mukus birikimi
  - Hava yolu tıkanıklığı

#### **Wheeze+Crackle** (0.45% - 30 event)
- **Kombinasyon**
- Hem hırıltı hem çatırtı birlikte
- Kompleks patolojiler

#### **Stridor** (0.23% - 15 event)
- **Stridor**
- Yüksek frekanslı inspiratuar ses
- **İlişkili Hastalıklar**:
  - Üst hava yolu tıkanıklığı
  - Krup
  - Epiglottit
  - Trakeal stenoz
  - **ACİL DURUM** göstergesi olabilir

---

## 📁 DOSYA ADLANDIRMA KURALLARI

Her dosya ismi 5 elementten oluşur (underscore ile ayrılmış):

```
PatientNumber_Age_Gender_Location_RecordingNumber
```

### Örnek: `41186340_6.6_0_p2_3004.wav`

1. **Hasta Numarası**: `41186340` (Benzersiz hasta kimliği)
2. **Yaş**: `6.6` (6.6 yaş)
3. **Cinsiyet**: `0` (Erkek=0, Kadın=1)
4. **Kayıt Konumu**: `p2` (Sol lateral)
   - `p1`: Sol posterior
   - `p2`: Sol lateral
   - `p3`: Sağ posterior
   - `p4`: Sağ lateral
5. **Kayıt Numarası**: `3004` (Benzersiz kayıt ID)

---

## 📝 JSON DOSYA YAPISI

```json
{
    "record_annotation": "DAS",
    "event_annotation": [
        {
            "start": "1829",
            "end": "2771",
            "type": "Fine Crackle"
        },
        {
            "start": "3436",
            "end": "4716",
            "type": "Normal"
        }
    ]
}
```

- **record_annotation**: Kayıt düzeyinde genel sınıf
- **event_annotation**: Zaman damgalı olay listesi
  - **start/end**: Milisaniye cinsinden başlangıç/bitiş
  - **type**: Olay tipi

---

## 🎯 CHALLENGE GÖREVLERİ

### Challenge 2022 & 2023: Classification (Sınıflandırma)

#### **Task 1: Event Level Classification**
- **Task 1-1**: Binary (Normal vs Adventitious)
- **Task 1-2**: Multi-class (7 sınıf: N, R, W, S, CC, FC, WC)

#### **Task 2: Record Level Classification**
- **Task 2-1**: Ternary (Normal, Adventitious, Poor Quality)
- **Task 2-2**: Multi-class (5 sınıf: Normal, CAS, DAS, CAS&DAS, PQ)

### Challenge 2024 & 2025

#### **Track 1: Compression (Sıkıştırma)**
- Solunum seslerinin sıkıştırılması
- Compressive sensing yöntemleri

#### **Track 2: Detection (Tespit)**
- Solunum olaylarının başlangıç/bitiş tespiti
- Event label assignment

---

## 📊 VERİ SETİ DAĞILIMI

### Training Set Kayıt Seviyesi Dağılımı

```
Normal        : ████████████████████████████████████ 66.85%
DAS           : ██████                               12.72%
Poor Quality  : ████                                  9.08%
CAS           : ███                                   6.46%
CAS & DAS     : ██                                    4.87%
```

### Training Set Event Seviyesi Dağılımı

```
Normal         : ████████████████████████████████████████ 77.51%
Fine Crackle   : ██████                                   13.70%
Wheeze         : ███                                       6.79%
Coarse Crackle : █                                         0.74%
Rhonchi        : █                                         0.59%
Wheeze+Crackle : █                                         0.45%
Stridor        : █                                         0.23%
```

---

## 💡 VERİ SETİNİ NASIL DEĞERLENDİREBİLİRSİNİZ?

### 1. **Sınıf Dengesizliği (Class Imbalance)**
- Normal olaylar %77.5 ile dominant
- Nadir sınıflar (Stridor, Wheeze+Crackle) çok az
- **Öneri**: 
  - Weighted loss functions kullanın
  - Data augmentation uygulayın
  - SMOTE veya oversampling teknikleri
  - Focal Loss kullanabilirsiniz

### 2. **Hiyerarşik Yapı**
- İki seviye etiketleme var (record + event)
- **Öneri**: 
  - Multi-task learning yaklaşımı
  - Hierarchical classification
  - Her iki seviyeyi birlikte modelleyin

### 3. **Temporal/Zamansal Bilgi**
- Olaylar milisaniye hassasiyetinde
- **Öneri**:
  - RNN, LSTM, GRU gibi sequential modeller
  - Temporal CNN
  - Attention mechanisms
  - Transformer architectures

### 4. **Transfer Learning**
- Pediatrik ses verisi nadir
- **Öneri**:
  - AudioSet, ESC-50 gibi pre-trained modeller
  - VGGish, YAMNet embeddings
  - Fine-tuning stratejileri

### 5. **Feature Extraction**
- **Önerilen Özellikler**:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Mel-Spectrogram
  - Chroma features
  - Spectral features (centroid, rolloff, flux)
  - Zero-crossing rate
  - Energy, RMS

### 6. **Veri Artırma (Data Augmentation)**
- **Teknikler**:
  - Time stretching
  - Pitch shifting
  - Adding noise
  - Time masking (SpecAugment)
  - Frequency masking
  - Mixup

### 7. **Validation Stratejisi**
- Inter-subject (hastalar arası) ve Intra-subject (hasta içi) test setleri mevcut
- **Öneri**:
  - Patient-wise split yapın
  - Cross-validation için hasta bazlı stratification

### 8. **Model Önerileri**
- **Başlangıç**: CNN-based (VGG, ResNet)
- **Orta Seviye**: CRNN (CNN + RNN)
- **İleri Seviye**: Transformers, Conformers
- **Ensemble**: Birden fazla modeli birleştirme

### 9. **Evaluation Metrics**
- Accuracy yeterli olmayabilir (dengesiz veri)
- **Kullanılması Gerekenler**:
  - F1-Score (özellikle macro/weighted)
  - Sensitivity/Recall (sağlık kritik)
  - Specificity
  - AUC-ROC
  - Confusion Matrix analizi

### 10. **Klinik Önemi**
- **Kritik Sınıflar**: Stridor (acil durum)
- **Yaygın Hastalıklar**: Wheeze (astım), Fine Crackle (pnömoni)
- **Öneri**: Sınıf önemine göre weighted metrics

---

## 🔗 REFERANSLAR

### Ana Yayın
```
Q. Zhang, et al. "SPRSound: Open-Source SJTU Paediatric Respiratory 
Sound Database", IEEE Transactions on Biomedical Circuits and Systems 
(TBioCAS), vol. 16, no. 5, pp. 867-881, Oct, 2022.
```

### Challenge Yayınları
- BioCAS 2022, 2023, 2024 Grand Challenge papers
- IEEE Data Descriptions (DD) 2024

---

## 📞 İLETİŞİM

- **Challenge Website**: http://1.117.17.41/grand-challenge/
- **Organizasyon**: Shanghai Jiao Tong University (SJTU)
- **Kurum**: Shanghai Children's Medical Center (SCMC)

---

## ⚠️ ÖNEMLİ NOTLAR

1. **Pediatrik Popülasyon**: Çocuklara özgü solunum sesleri, yetişkinlerden farklı olabilir
2. **Kayıt Kalitesi**: %9 Poor Quality var, preprocessing önemli
3. **Çoklu Lokasyon**: 4 farklı kayıt konumu var, model lokasyon bilgisini kullanabilir
4. **Temporal Resolution**: Milisaniye hassasiyeti, yüksek çözünürlüklü analiz imkanı
5. **Nadir Sınıflar**: Stridor ve Wheeze+Crackle çok az, özel ilgi gerektirir

---

**Rapor Tarihi**: Ocak 2026  
**Veri Seti Versiyonu**: SPRSound 2022-2025
