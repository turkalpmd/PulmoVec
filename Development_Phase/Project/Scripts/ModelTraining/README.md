# Meta Model Training Scripts

Bu klasör, LightGBM meta-model eğitimi için gerekli tüm scriptleri ve kaynak kodları içerir.

## 📁 Dosya Yapısı

```
ModelTraining/
├── train_meta_model.py              # Meta-model eğitim scripti (ana script)
├── extract_ensemble_probabilities.py # Ensemble modellerden olasılık çıkarma
├── README.md                        # Bu dosya
└── src/                             # Kaynak kod modülleri
    ├── config.py                    # Temel konfigürasyon
    ├── config_model1.py             # Model 1 konfigürasyonu
    ├── config_model2.py             # Model 2 konfigürasyonu
    ├── config_model3.py             # Model 3 konfigürasyonu
    ├── models.py                    # HeARClassifier model tanımı
    ├── sprsound_dataset.py          # PyTorch Dataset sınıfı
    └── utils.py                     # Yardımcı fonksiyonlar
```

## 🚀 Kullanım

### 1. Olasılık Çıkarma

Ensemble modellerden olasılık çıktılarını çıkar:

```bash
cd Project/Scripts/ModelTraining
python extract_ensemble_probabilities.py
```

**Gereksinimler:**
- 3 eğitilmiş ensemble model (Model 1, 2, 3)
- `data/SPRSound_Event_Level_Ensemble_Dataset.csv`

**Çıktılar:**
- `data/ensemble_probabilities_train.csv`
- `data/ensemble_probabilities_val.csv`

### 2. Meta-Model Eğitimi

LightGBM meta-modelleri eğit:

```bash
python train_meta_model.py --use_csv
```

**Seçenekler:**
- `--use_csv`: CSV dosyalarından olasılıkları yükle (önerilen)
- `--n_trials`: Optuna trial sayısı (varsayılan: 100)
- `--n_jobs`: Paralel işlem sayısı (varsayılan: -1, tüm CPU'lar)

**Örnek:**
```bash
# 200 trial ile optimize et
python train_meta_model.py --use_csv --n_trials 200

# 4 CPU kullan
python train_meta_model.py --use_csv --n_jobs 4
```

**Çıktılar:**
- `Project/Results/meta_model/disease/` - Disease tahmin modeli
- `Project/Results/meta_model/event_type/` - Event type tahmin modeli
- `Project/Results/meta_model/model1_label/` - Model1 label tahmin modeli
- `Project/Results/meta_model/model2_label/` - Model2 label tahmin modeli
- `Project/Results/meta_model/model3_label/` - Model3 label tahmin modeli
- `Project/Results/meta_model/model4_label/` - Model4 label tahmin modeli (4 sınıf)

Her klasörde:
- `model.pkl` - Eğitilmiş LightGBM modeli
- `metrics.json` - Detaylı metrikler
- `report.md` - Markdown raporu
- `confusion_matrix.png` - Karışıklık matrisi görselleştirmesi
- `roc_auprc_curves.png` - ROC ve AUPRC eğrileri (her sınıf için one-vs-rest)
  - **Model 4 için özel:** 4 sınıfın tümü aynı plot üzerinde gösterilir

## 📋 Ön Gereksinimler

1. **Ensemble Modeller:**
   - Model 1: `models/model1_event_type/best.pth`
   - Model 2: `models/model2_binary/best.pth`
   - Model 3: `models/model3_disease/best.pth`

2. **Veri Seti:**
   - `data/SPRSound_Event_Level_Ensemble_Dataset.csv`
   - `model4_label` kolonu dahil (4 sınıf: Pneumonia, Bronchoconstriction, Normal, Others)

3. **Python Paketleri:**
   - torch
   - pandas
   - numpy
   - lightgbm
   - optuna
   - scikit-learn
   - matplotlib
   - seaborn

## 🔍 Detaylı Dokümantasyon

Detaylı açıklamalar için `../ModelTraining.md` dosyasına bakın.

## ⚠️ Önemli Notlar

- Meta-model eğitimi için önce ensemble modellerin eğitilmesi gerekir (Model 1, 2, 3)
- Model 4 henüz eğitilmediği için sadece `model4_label` ground truth olarak kullanılır
- Olasılık çıkarma işlemi zaman alabilir (tüm veri seti üzerinden inference)
- Optuna optimizasyonu CPU yoğun bir işlemdir (1-2 saat sürebilir)
- Bootstrap güven aralıkları hesaplama zaman alabilir (1000 iterasyon)
- Model 4 için ROC/AUPRC eğrileri özel olarak 4 sınıfı aynı plot'ta gösterir

## 📊 Model 4 Detayları

Model 4, 4 sınıflı hastalık kategorisi sınıflandırması yapar:

- **Class 0 (Pneumonia):** Pneumonia (non-severe), Pneumonia (severe)
- **Class 1 (Bronchoconstriction):** Asthma, Protracted bacterial bronchitis, Bronchitis, Bronchiectasia, Bronchiolitis
- **Class 2 (Normal):** Control Group
- **Class 3 (Others):** Other respiratory diseases, Chronic cough, Hemoptysis, Acute upper respiratory infection, Pulmonary hemosiderosis, Airway foreign body, Unknown, Kawasaki disease

**Özel Görselleştirme:** Model 4 için ROC ve AUPRC eğrileri, 4 sınıfın tümünü aynı plot üzerinde gösterir (one-vs-rest yaklaşımı).