# Meta Model Training Dokümantasyonu

Bu dokümantasyon, SPRSound veri seti için LightGBM meta-model eğitim sürecini detaylı olarak açıklar.

---

## 📋 Genel Bakış

Meta-model, 3 ayrı HeAR tabanlı ensemble modelinin çıktılarını birleştirerek daha güçlü ve robust bir sınıflandırma yapar. Meta-model, ensemble modellerden gelen olasılık çıktılarını ve demografik özellikleri kullanarak 6 farklı çıktıyı tahmin eder.

### Meta-Model Mimarisi

```
Ensemble Model Çıktıları (8 olasılık)
    ↓
Model 1: Normal, Crackles, Rhonchi (3 olasılık)
Model 2: Normal, Abnormal (2 olasılık)
Model 3: Normal, Pneumonia, Bronchiolitis (3 olasılık)
    ↓
Demografik Özellikler (3 özellik)
    ↓
Age, Gender, recording_location
    ↓
─────────────────────────────
Toplam: 11 Özellik
─────────────────────────────
    ↓
LightGBM Meta-Model
    ↓
6 Ayrı Tahmin:
1. disease (çok sınıflı)
2. event_type (çok sınıflı)
3. model1_label (3 sınıf)
4. model2_label (2 sınıf - binary)
5. model3_label (3 sınıf)
6. model4_label (4 sınıf)
```

---

## 🔄 Eğitim Süreci

### Adım 1: Ensemble Modellerin Eğitimi

Meta-model eğitimi için önce 3 ensemble modelinin eğitilmesi gerekir:

```bash
# Tüm modelleri sırayla eğit
python scripts/train_ensemble_models.py --model all
```

**Çıktılar:**
- `models/model1_event_type/best.pth` - Event Type sınıflandırıcı
- `models/model2_binary/best.pth` - Binary abnormality detector
- `models/model3_disease/best.pth` - Disease group sınıflandırıcı

### Adım 2: Olasılık Çıktılarının Çıkarılması

Eğitilmiş ensemble modellerden olasılık çıktılarını çıkar:

```bash
python scripts/extract_ensemble_probabilities.py
```

**Bu Script Ne Yapar?**

1. **3 Eğitilmiş Modeli Yükler:**
   - Model 1: Event Type (3 sınıf)
   - Model 2: Binary (2 sınıf)
   - Model 3: Disease (3 sınıf)

2. **Veri Setini Yükler:**
   - `data/SPRSound_Event_Level_Ensemble_Dataset.csv`
   - Train/Val split (80/20, stratified)

3. **Her Örnek İçin Olasılık Çıktıları:**
   - Model 1: `[Normal_prob, Crackles_prob, Rhonchi_prob]`
   - Model 2: `[Normal_prob, Abnormal_prob]`
   - Model 3: `[Pneumonia_prob, Bronchiolitis_prob, Normal_prob]`

4. **CSV Dosyalarına Kaydeder:**
   - `data/ensemble_probabilities_train.csv`
   - `data/ensemble_probabilities_val.csv`

**CSV Kolonları:**
- Demografik: `patient_number`, `age`, `gender`, `recording_location`
- Model 1: `Model1_Normalpp`, `Model1_Cracklespp`, `Model1_Rhonchipp`
- Model 2: `Model2_Normalpp`, `Model2_Abnormalpp`
- Model 3: `Model3_Normalpp`, `Model3_Pneumoniapp`, `Model3_Bronchiolitispp`
- Ground Truth: `disease`, `event_type`, `model1_label`, `model2_label`, `model3_label`, `model4_label`

### Adım 3: Meta-Model Eğitimi

```bash
python scripts/train_meta_model.py --use_csv
```

**Bu Script Ne Yapar?**

1. **Olasılık CSV'lerini Yükler:**
   - Training ve validation setlerinden olasılıkları okur

2. **Özellik Hazırlama:**
   - 8 olasılık özelliği (Model 1, 2, 3 çıktıları)
   - 3 demografik özellik (age, gender_encoded, recording_location_encoded)
   - Toplam: **11 özellik**

3. **6 Ayrı Meta-Model Eğitir:**
   - Her biri farklı bir çıktıyı tahmin eder
   - Optuna ile hiperparametre optimizasyonu
   - LightGBM kullanır

4. **Değerlendirme:**
   - Bootstrap ile %95 güven aralıkları
   - Detaylı metrikler (Accuracy, F1, MCC, ROC-AUC, vb.)
   - Confusion matrix görselleştirmeleri

5. **Çıktılar:**
   - `Project/Results/meta_model/{outcome_name}/model.pkl` - Eğitilmiş model
   - `Project/Results/meta_model/{outcome_name}/metrics.json` - Metrikler
   - `Project/Results/meta_model/{outcome_name}/report.md` - Detaylı rapor
   - `Project/Results/meta_model/{outcome_name}/confusion_matrix.png` - Karışıklık matrisi
   - `Project/Results/meta_model/{outcome_name}/roc_auprc_curves.png` - ROC ve AUPRC eğrileri

---

## 🎯 Meta-Model Çıktıları

### 1. Disease (Çok Sınıflı)

**Amaç:** Hasta hastalık tanısını tahmin et

**Sınıflar:**
- Pneumonia (severe + non-severe)
- Bronchitis
- Asthma
- Bronchiolitis
- Control Group
- Diğer hastalıklar

**Kullanım:** Klinik tanı desteği

### 2. Event Type (Çok Sınıflı)

**Amaç:** Solunum sesi event tipini tahmin et

**Sınıflar:**
- Normal
- Fine Crackle
- Coarse Crackle
- Wheeze
- Rhonchi
- Wheeze+Crackle
- Stridor

**Kullanım:** Akustik özellik analizi

### 3. Model1 Label (3 Sınıf)

**Amaç:** Event Type gruplandırması

**Sınıflar:**
- 0: Normal
- 1: Crackles (Fine + Coarse + Wheeze+Crackle)
- 2: Wheeze/Rhonchi

**Kullanım:** Ensemble Model 1 çıktısını doğrulama

### 4. Model2 Label (2 Sınıf - Binary)

**Amaç:** Anormal/Normal ayrımı

**Sınıflar:**
- 0: Normal
- 1: Abnormal

**Kullanım:** Tarama ve triyaj

### 5. Model3 Label (3 Sınıf)

**Amaç:** Hastalık grubu sınıflandırması

**Sınıflar:**
- 0: Pneumonia
- 1: Bronchitis/Asthma/Bronchiolitis
- 2: Normal/Other

**Kullanım:** Ensemble Model 3 çıktısını doğrulama

### 6. Model4 Label (4 Sınıf)

**Amaç:** Hastalık kategorisi sınıflandırması

**Sınıflar:**
- 0: Pneumonia
  - Pneumonia (non-severe)
  - Pneumonia (severe)
- 1: Bronchoconstriction
  - Asthma
  - Protracted bacterial bronchitis
  - Bronchitis
  - Bronchiectasia
  - Bronchiolitis
- 2: Normal
  - Control Group
- 3: Others
  - Other respiratory diseases
  - Chronic cough
  - Hemoptysis
  - Acute upper respiratory infection
  - Pulmonary hemosiderosis
  - Airway foreign body
  - Unknown
  - Kawasaki disease

**Kullanım:** Daha detaylı hastalık kategorizasyonu

**Özel Not:** Model 4 için ROC ve AUPRC eğrileri, 4 sınıfın tümü aynı plot üzerinde gösterilir (one-vs-rest).

---

## 🔧 Teknik Detaylar

### Özellik Mühendisliği

**Olasılık Özellikleri (8):**
- Model 1: 3 olasılık (Normal, Crackles, Rhonchi)
- Model 2: 2 olasılık (Normal, Abnormal)
- Model 3: 3 olasılık (Normal, Pneumonia, Bronchiolitis)

**Demografik Özellikler (3):**
- `age`: Yaş (eksik değerler median ile doldurulur)
- `gender_encoded`: Cinsiyet (Female=0, Male=1)
- `recording_location_encoded`: Kayıt konumu (LabelEncoder ile encode edilir, eksik değerler most frequent ile doldurulur)

**Toplam:** 11 özellik

### LightGBM Hiperparametre Optimizasyonu

**Optuna ile Optimize Edilen Parametreler:**
- `n_estimators`: 50-500
- `max_depth`: 3-15
- `learning_rate`: 0.01-0.3 (log scale)
- `num_leaves`: 15-300
- `min_child_samples`: 5-100
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `reg_alpha`: 0-10
- `reg_lambda`: 0-10

**Sabit Parametreler:**
- `early_stopping_rounds`: 20
- `random_state`: 42
- `n_jobs`: -1 (tüm CPU'lar)

**Optimizasyon Metriği:** F1-Macro (maximize)

**Trial Sayısı:** 100 (varsayılan, `--n_trials` ile değiştirilebilir)

### Değerlendirme Metrikleri

Her meta-model için şu metrikler hesaplanır:

**Temel Metrikler:**
- Accuracy (Doğruluk)
- F1-Macro (Makro ortalama F1)
- F1-Weighted (Ağırlıklı F1)
- Matthews Correlation Coefficient (MCC)
- Log-Loss

**Olasılıksal Metrikler:**
- ROC-AUC (Binary için tek, multi-class için one-vs-rest)
- ROC-AUC per-class (her sınıf için)
- AUPRC (Average Precision-Recall Curve) per-class

**Görselleştirmeler:**
- Confusion Matrix
- ROC Curves (her sınıf için one-vs-rest)
- Precision-Recall Curves (her sınıf için one-vs-rest)
- **Model 4 için özel:** 4 sınıfın tümü aynı plot üzerinde gösterilir

**Sınıf Bazlı Metrikler:**
- Precision (PPV - Positive Predictive Value)
- Recall (Sensitivity - True Positive Rate)
- Specificity (True Negative Rate)
- NPV (Negative Predictive Value)
- F1-Score
- Support (örnek sayısı)

**Güven Aralıkları:**
- Tüm metrikler için %95 güven aralığı (bootstrap, 1000 iterasyon)

---

## 📊 Kullanım Örnekleri

### 1. Tüm Meta-Modelleri Eğit

```bash
# Olasılıkları çıkar (bir kez)
python scripts/extract_ensemble_probabilities.py

# Meta-modelleri eğit
python scripts/train_meta_model.py --use_csv
```

### 2. Özelleştirilmiş Trial Sayısı

```bash
# 200 trial ile optimize et
python scripts/train_meta_model.py --use_csv --n_trials 200
```

### 3. Paralel İşlem

```bash
# Tüm CPU'ları kullan (varsayılan)
python scripts/train_meta_model.py --use_csv --n_jobs -1

# 4 CPU kullan
python scripts/train_meta_model.py --use_csv --n_jobs 4
```

---

## 📁 Dosya Yapısı

```
Project/Scripts/ModelTraining/
├── train_meta_model.py          # Meta-model eğitim scripti
├── extract_ensemble_probabilities.py  # Olasılık çıkarma scripti
└── src/
    ├── config.py                 # Temel konfigürasyon
    ├── config_model1.py          # Model 1 konfigürasyonu
    ├── config_model2.py          # Model 2 konfigürasyonu
    ├── config_model3.py          # Model 3 konfigürasyonu
    ├── models.py                 # HeARClassifier modeli
    ├── sprsound_dataset.py       # PyTorch Dataset
    └── utils.py                  # Yardımcı fonksiyonlar

Project/Results/meta_model/
├── disease/
│   ├── model.pkl                 # Eğitilmiş model
│   ├── label_encoder.pkl        # Label encoder (varsa)
│   ├── metrics.json              # Metrikler
│   ├── report.md                 # Detaylı rapor
│   ├── confusion_matrix.png      # Karışıklık matrisi
│   └── roc_auprc_curves.png      # ROC ve AUPRC eğrileri
├── event_type/
│   └── ... (aynı yapı)
├── model1_label/
│   └── ... (aynı yapı)
├── model2_label/
│   └── ... (aynı yapı)
├── model3_label/
│   └── ... (aynı yapı)
└── model4_label/
    └── ... (aynı yapı)
```

---

## 🔍 Kod Detayları

### extract_ensemble_probabilities.py

**Ana Fonksiyonlar:**

1. **`load_trained_model(config_module)`**
   - Eğitilmiş model checkpoint'ini yükler
   - Model'i evaluation moduna alır

2. **`extract_probabilities(models, dataloader, df_source)`**
   - Tüm modellerden olasılık çıktılarını çıkarır
   - Softmax ile normalize eder
   - DataFrame'e ekler

**Önemli Notlar:**
- DataLoader `shuffle=False` olmalı (örnek sırası korunmalı)
- Olasılıkların toplamı ~1.0 olmalı (doğrulama yapılır)
- Model 4 henüz eğitilmediği için sadece `model4_label` ground truth olarak eklenir

### train_meta_model.py

**Ana Fonksiyonlar:**

1. **`load_probabilities_from_csv(train_csv_path, val_csv_path)`**
   - CSV dosyalarından olasılıkları yükler
   - Olasılık toplamlarını doğrular

2. **`prepare_features_and_labels(df, outcome_name)`**
   - 11 özelliği hazırlar (8 olasılık + 3 demografik)
   - Demografik özellikler:
     - `age`: Median ile doldurulur (eksik değerler için)
     - `gender_encoded`: Female=0, Male=1 mapping
     - `recording_location_encoded`: LabelEncoder ile encode edilir (eksik değerler most frequent ile doldurulur)
   - Label encoding yapar (gerekirse)
   - Feature names döndürür

3. **`optimize_lightgbm_hyperparameters(...)`**
   - Optuna ile hiperparametre optimizasyonu
   - TPE sampler kullanır
   - Early stopping: 20 rounds (sabit)

4. **`train_lightgbm_model(...)`**
   - En iyi hiperparametrelerle final model eğitir
   - Early stopping ile overfitting önlenir

5. **`evaluate_model_detailed_with_ci95(...)`**
   - Detaylı metrikler hesaplar
   - Bootstrap ile %95 güven aralıkları
   - Confusion matrix oluşturur

6. **`bootstrap_metric(...)`**
   - Bootstrap resampling ile güven aralığı hesaplar
   - 1000 iterasyon (varsayılan)

---

## 📈 Performans Metrikleri

### Örnek Sonuçlar

**Disease Meta-Model:**
- Accuracy: ~0.75 (CI95: [0.74, 0.76])
- F1-Macro: ~0.65 (CI95: [0.64, 0.66])
- MCC: ~0.60

**Event Type Meta-Model:**
- Accuracy: ~0.70
- F1-Macro: ~0.65

**Model1 Label Meta-Model:**
- Accuracy: ~0.85
- F1-Macro: ~0.80

**Model2 Label Meta-Model (Binary):**
- Accuracy: ~0.90
- F1-Macro: ~0.88
- ROC-AUC: ~0.95

**Model3 Label Meta-Model:**
- Accuracy: ~0.80
- F1-Macro: ~0.75

**Model4 Label Meta-Model (4 Sınıf):**
- Accuracy: ~0.78
- F1-Macro: ~0.72
- ROC-AUC (per-class): ~0.85-0.95
- **Özel:** 4 sınıfın ROC ve AUPRC eğrileri aynı plot'ta gösterilir

---

## ⚠️ Önemli Notlar

### 1. Ensemble Modeller Önce Eğitilmeli

Meta-model eğitimi için 3 ensemble modelinin önce eğitilmesi gerekir:
- Model 1: Event Type
- Model 2: Binary
- Model 3: Disease

### 2. Veri Seti Hazırlığı

`prepare_ensemble_labels.py` scripti çalıştırılmalı:
- `data/SPRSound_Event_Level_Ensemble_Dataset.csv` oluşturulur
- 4 label kolonu eklenir: `model1_label`, `model2_label`, `model3_label`, `model4_label`

### 3. Olasılık Çıkarma

`extract_ensemble_probabilities.py` bir kez çalıştırılmalı:
- Eğitilmiş modellerden olasılıklar çıkarılır
- CSV dosyalarına kaydedilir
- Bu işlem zaman alabilir (tüm veri seti üzerinden inference)

### 4. Hiperparametre Optimizasyonu

- Optuna optimizasyonu zaman alabilir (100 trial ≈ 1-2 saat)
- `--n_trials` ile trial sayısı ayarlanabilir
- `--n_jobs` ile paralel işlem sayısı ayarlanabilir

### 5. Bootstrap Güven Aralıkları

- 1000 bootstrap iterasyonu varsayılan
- Daha hızlı sonuç için azaltılabilir (kod içinde)
- Daha güvenilir sonuç için artırılabilir

---

## 🐛 Troubleshooting

### Sorun: CSV Dosyaları Bulunamadı

**Hata:**
```
❌ CSV files not found!
```

**Çözüm:**
```bash
# Önce olasılıkları çıkar
python scripts/extract_ensemble_probabilities.py
```

### Sorun: Ensemble Modeller Bulunamadı

**Hata:**
```
Model checkpoint not found: models/model1_event_type/best.pth
```

**Çözüm:**
```bash
# Önce ensemble modelleri eğit
python scripts/train_ensemble_models.py --model all
```

### Sorun: Optuna Optimizasyonu Çok Yavaş

**Çözüm:**
- `--n_trials` değerini azalt (örn: 50)
- `--n_jobs` ile paralel işlem sayısını artır
- Daha küçük veri seti ile test et

### Sorun: Memory Hatası

**Çözüm:**
- Bootstrap iterasyon sayısını azalt (kod içinde)
- Batch size'ı küçült (eğer varsa)
- Daha az trial kullan

---

## 📚 Referanslar

- **LightGBM**: https://lightgbm.readthedocs.io/
- **Optuna**: https://optuna.org/
- **HeAR Model**: https://huggingface.co/google/hear-pytorch
- **SPRSound Dataset**: Original dataset documentation

---

## ✅ Checklist

Meta-model eğitimi için kontrol listesi:

- [ ] Ensemble modeller eğitildi (Model 1, 2, 3)
- [ ] `prepare_ensemble_labels.py` çalıştırıldı (Model 4 label'ları dahil)
- [ ] `extract_ensemble_probabilities.py` çalıştırıldı
- [ ] CSV dosyaları oluşturuldu (`ensemble_probabilities_train.csv`, `ensemble_probabilities_val.csv`)
- [ ] HuggingFace token ayarlandı (HeAR model için)
- [ ] GPU/CPU hazır
- [ ] `train_meta_model.py --use_csv` çalıştırıldı
- [ ] Sonuçlar kontrol edildi (`Project/Results/meta_model/` klasöründe)
- [ ] Model 4 için ROC/AUPRC eğrileri kontrol edildi (4 sınıf aynı plot'ta)

---

**Son Güncelleme:** 2026-01-24  
**Versiyon:** 2.0.0

**Değişiklikler (v2.0.0):**
- Model 4 eklendi (4 sınıf: Pneumonia, Bronchoconstriction, Normal, Others)
- ROC ve AUPRC eğrileri eklendi (her sınıf için one-vs-rest)
- Model 4 için özel görselleştirme: 4 sınıf aynı plot'ta
- Sonuçlar `Project/Results/meta_model/` dizinine kaydediliyor
- `recording_location` özelliği eklendi (11 özellik toplam)
