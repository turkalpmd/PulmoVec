# SPRSound Event-Level Dataset Oluşturma Süreci

Bu dokümantasyon, SPRSound veri setinden event-level (olay bazlı) bir CSV veri setinin nasıl oluşturulduğunu detaylı olarak açıklar.

---

## 📋 Genel Bakış

SPRSound veri seti, pediatrik solunum sesleri için JSON ve WAV dosyaları içerir. Bu scriptler, JSON dosyalarındaki event (olay) bilgilerini çıkararak, her event için ayrı bir satır içeren bir CSV dosyası oluşturur.

**Ana Çıktı:** `SPRSound_Event_Level_Dataset_CLEAN.csv`

---

## 🔄 İşlem Adımları

### 1. Adım: Event-Level CSV Oluşturma
**Script:** `archive/create_event_level_dataset_CLEAN.py`

Bu script, tüm JSON dosyalarını tarayarak event bilgilerini çıkarır ve duplikasyonları önler.

#### 1.1. Hasta Bilgileri Yükleme

**Kaynak Dosyalar:**
- `SPRSound/Patient Summary/SPRSound_patient_summary.csv`
- `SPRSound/Patient Summary/Grand_Challenge'23_patient_summary.csv`
- `SPRSound/Patient Summary/Grand_Challenge'24_patient_summary.csv`

**İşlem:**
- Her CSV dosyasından `patient_num` ve `disease` bilgileri okunur
- Bir dictionary (`patient_diseases`) oluşturulur: `{patient_num: disease}`
- Bu bilgi, her event'e hasta hastalık tanısını eklemek için kullanılır

#### 1.2. JSON Dosyalarının İşlenmesi

**İşlenen Dizinler (Öncelik Sırasına Göre):**

1. **Classification Datasets (SPRSound-main):**
   - `SPRSound-main/Classification/train_classification_json` → `Classification-Train`
   - `SPRSound-main/Classification/valid_classification_json/2022/inter_test_json` → `Classification-Valid-2022-Inter`
   - `SPRSound-main/Classification/valid_classification_json/2022/intra_test_json` → `Classification-Valid-2022-Intra`
   - `SPRSound-main/Classification/valid_classification_json/2023` → `Classification-Valid-2023`

2. **Detection Datasets (SPRSound-main):**
   - `SPRSound-main/Detection/train_detection_json` → `Detection-Train`
   - `SPRSound-main/Detection/valid_detection_json` → `Detection-Valid`
   - `SPRSound-main/Detection/test2024_detection_json` → `Detection-Test-2024`

3. **BioCAS Test Datasets (SPRSound):**
   - `SPRSound/BioCAS2023/test2023_json` → `BioCAS2023-Test`
   - `SPRSound/BioCAS2024/test2024_json` → `BioCAS2024-Test`
   - `SPRSound/BioCAS2025/test2025_json` → `BioCAS2025-Test`

**Önemli:** Duplikasyon önleme mekanizması sayesinde, aynı dosya adı birden fazla kez işlenmez. İlk görülen dosya işlenir, sonrakiler atlanır.

#### 1.3. Dosya Adı Parsing

**Format:** `patient_number_age_gender_location_recording_number.json`

**Örnek:** `40996284_3.0_0_p1_6664.json`

**Çıkarılan Bilgiler:**
- `patient_number`: Hasta numarası (örn: `40996284`)
- `age`: Yaş (örn: `3.0`)
- `gender_code`: Cinsiyet kodu (`0` = Male, `1` = Female)
- `gender`: Cinsiyet açıklaması (`Male` / `Female`)
- `recording_location`: Kayıt konumu kodu (örn: `p1`)
- `recording_location_name`: Konum açıklaması (örn: `Left Posterior`)
- `recording_number`: Kayıt numarası (örn: `6664`)

**Konum Kodları:**
- `p1` → `Left Posterior`
- `p2` → `Left Lateral`
- `p3` → `Right Posterior`
- `p4` → `Right Lateral`
- `p5-p8` → `Additional Location 5-8`

#### 1.4. JSON İçeriği İşleme

Her JSON dosyası şu yapıya sahiptir:

```json
{
  "record_annotation": "Normal" | "Crackles" | "Wheeze-Rhonchi" | ...,
  "event_annotation": [
    {
      "start": 1000,  // milisaniye
      "end": 3000,    // milisaniye
      "type": "Crackles" | "Wheeze" | "Rhonchi" | ...
    },
    ...
  ]
}
```

**İşlem Mantığı:**

1. **Event Yoksa:**
   - Tek bir satır oluşturulur
   - `event_type` = `'No Event'`
   - `event_start_ms`, `event_end_ms`, `event_duration_ms` = boş
   - `event_index` = 0
   - `total_events_in_file` = 0

2. **Event Varsa:**
   - Her event için ayrı bir satır oluşturulur
   - `event_start_ms` = event'in başlangıç zamanı (ms)
   - `event_end_ms` = event'in bitiş zamanı (ms)
   - `event_duration_ms` = `end - start` (hesaplanır)
   - `event_type` = event tipi
   - `event_index` = event'in dosyadaki sırası (1, 2, 3, ...)
   - `total_events_in_file` = dosyadaki toplam event sayısı

#### 1.5. CSV Kolonları

**Oluşturulan CSV'nin Kolonları:**

1. `dataset` - Veri seti adı (örn: `Classification-Train`)
2. `file_path` - JSON dosyasının tam yolu
3. `filename` - JSON dosya adı
4. `patient_number` - Hasta numarası
5. `age` - Yaş
6. `gender` - Cinsiyet (Male/Female)
7. `gender_code` - Cinsiyet kodu (0/1)
8. `recording_location` - Kayıt konumu kodu (p1-p8)
9. `recording_location_name` - Konum açıklaması
10. `recording_number` - Kayıt numarası
11. `record_annotation` - Dosya seviyesi annotation (Normal/Crackles/Wheeze-Rhonchi)
12. `disease` - Hasta hastalık tanısı (Patient Summary'den)
13. `event_start_ms` - Event başlangıç zamanı (milisaniye)
14. `event_end_ms` - Event bitiş zamanı (milisaniye)
15. `event_duration_ms` - Event süresi (milisaniye)
16. `event_type` - Event tipi (Crackles/Wheeze/Rhonchi/No Event)
17. `event_index` - Event'in dosyadaki sırası
18. `total_events_in_file` - Dosyadaki toplam event sayısı

#### 1.6. Duplikasyon Önleme

**Mekanizma:**
- İşlenen her dosya adı bir `set` içinde saklanır (`processed_files`)
- Yeni bir dosya işlenmeden önce, dosya adı bu set'te kontrol edilir
- Eğer dosya daha önce işlendiyse, atlanır ve `skipped_count` artırılır
- Bu sayede aynı dosya birden fazla kez işlenmez

**Neden Gerekli:**
- Bazı dosyalar birden fazla dizinde bulunabilir
- Örneğin, aynı dosya hem `SPRSound-main` hem de `SPRSound` dizinlerinde olabilir
- Duplikasyon önleme, temiz ve tutarlı bir veri seti sağlar

---

### 2. Adım: WAV Dosya Yollarını Ekleme
**Script:** `archive/add_wav_paths_to_csv.py`

Bu script, oluşturulan CSV'ye WAV dosya yollarını ekler.

#### 2.1. WAV Yolu Hesaplama

**Temel Dönüşüm:**
```python
wav_path = json_path.replace('.json', '.wav').replace('_json', '_wav')
```

**Örnek:**
- JSON: `SPRSound-main/Classification/train_classification_json/file.json`
- WAV: `SPRSound-main/Classification/train_classification_wav/file.wav`

#### 2.2. Özel Durum: 2022 Validation

2022 validation dosyaları için özel bir işlem yapılır:

**Sorun:**
- JSON yolu: `.../valid_classification_json/2022/inter_test_json/file.json`
- WAV yolu: `.../valid_classification_wav/2022/file.wav`
- `inter_test_json` veya `intra_test_json` klasörü WAV yolunda yok

**Çözüm:**
1. Yol parçalara ayrılır
2. `inter_test_json` veya `intra_test_json` kaldırılır
3. `_json` → `_wav` dönüşümü yapılır
4. `.json` → `.wav` uzantısı değiştirilir

**Örnek:**
```
JSON: .../2022/inter_test_json/file.json
→ WAV: .../2022/file.wav
```

#### 2.3. Yeni Kolonlar

**Eklenen Kolonlar:**
- `wav_path`: Hesaplanan WAV dosya yolu
- `wav_exists`: `'yes'` veya `'no'` (dosya sisteminde var mı?)

#### 2.4. İstatistikler

Script, şu istatistikleri üretir:
- Toplam event sayısı
- WAV bulunan event sayısı ve yüzdesi
- WAV eksik event sayısı ve yüzdesi
- Dataset bazında WAV durumu

---

## 📊 Veri Seti Özellikleri

### Event Tipleri

SPRSound veri setinde şu event tipleri bulunur:
- `Normal` - Normal solunum sesi
- `Crackles` - Çıtırtı sesleri
- `Wheeze` - Hışıltı
- `Rhonchi` - Hırıltı
- `Wheeze-Rhonchi` - Hışıltı-Hırıltı kombinasyonu
- `No Event` - Event yok (sadece record annotation var)

### Record Annotations

Dosya seviyesi annotation'lar:
- `Normal`
- `Crackles`
- `Wheeze-Rhonchi`
- Diğer kombinasyonlar

### Hastalık Kategorileri

Patient Summary dosyalarından gelen hastalık tanıları:
- `Pneumonia` - Zatürre
- `Bronchitis` - Bronşit
- `Asthma` - Astım
- `Bronchiolitis` - Bronşiyolit
- `Normal` - Normal
- `Unknown` - Bilinmeyen (Patient Summary'de yoksa)

---

## 🔧 Kullanım

### 1. Event-Level CSV Oluşturma

```bash
cd /home/izzet/Desktop/PulmoVec/archive
python create_event_level_dataset_CLEAN.py
```

**Çıktı:** `SPRSound_Event_Level_Dataset_CLEAN.csv`

### 2. WAV Yollarını Ekleme

```bash
cd /home/izzet/Desktop/PulmoVec/archive
python add_wav_paths_to_csv.py
```

**Girdi:** `SPRSound_Event_Level_Dataset_CLEAN.csv`  
**Çıktı:** `SPRSound_Event_Level_Dataset_CLEAN_with_WAV.csv`

---

## 📈 İstatistikler ve Analiz

Script çalıştırıldığında şu istatistikler üretilir:

### Dataset Bazında Event Sayıları
Her veri setinden kaç event çıkarıldığı gösterilir.

### Event Type Dağılımı
Her event tipinin sayısı ve yüzdesi.

### Hastalık Bazında Event Sayıları
Her hastalık kategorisindeki event sayıları (Top 10).

### WAV Dosya Durumu
- Hangi dataset'lerde WAV dosyaları mevcut
- Hangi dataset'lerde WAV dosyaları eksik
- Bulunma yüzdeleri

---

## ⚠️ Önemli Notlar

1. **Duplikasyon Önleme:** CLEAN versiyonu, aynı dosyanın birden fazla kez işlenmesini önler. Bu, veri setinin tutarlılığı için kritiktir.

2. **Dosya Yolu Farklılıkları:** Bazı dataset'lerde (özellikle 2022 validation) JSON ve WAV dosya yolları farklı klasör yapılarına sahiptir. `add_wav_paths_to_csv.py` bu durumu handle eder.

3. **Event Yoksa:** Eğer bir JSON dosyasında event yoksa, yine de bir satır oluşturulur (`event_type = 'No Event'`). Bu, dosya seviyesi annotation'ları korumak için önemlidir.

4. **Hasta Bilgisi:** Eğer bir hasta numarası Patient Summary dosyalarında bulunamazsa, `disease = 'Unknown'` olarak işaretlenir.

5. **Dosya Adı Formatı:** Bazı dosyalar standart format dışında olabilir (örn: `_` ile başlayan dosyalar). Script bu durumları handle eder ve `'Unknown'` değerleri kullanır.

---

## 🔄 Versiyonlar

### `create_event_level_dataset.py` (Eski)
- Duplikasyon kontrolü yok
- Tüm dosyalar işlenir (duplikasyonlar dahil)
- Daha fazla satır üretir

### `create_event_level_dataset_CLEAN.py` (Kullanılan)
- Duplikasyon kontrolü var
- Her dosya sadece bir kez işlenir
- Daha temiz ve tutarlı veri seti

**Önerilen:** CLEAN versiyonu kullanılmalıdır.

---

## 📝 Örnek CSV Satırı

```csv
dataset,file_path,filename,patient_number,age,gender,gender_code,recording_location,recording_location_name,recording_number,record_annotation,disease,event_start_ms,event_end_ms,event_duration_ms,event_type,event_index,total_events_in_file
Classification-Train,SPRSound-main/Classification/train_classification_json/40996284_3.0_0_p1_6664.json,40996284_3.0_0_p1_6664.json,40996284,3.0,Male,0,p1,Left Posterior,6664,Normal,Pneumonia,1000,3000,2000,Crackles,1,2
```

---

## 🎯 Sonuç

Bu süreç, SPRSound veri setindeki tüm JSON dosyalarını tarayarak, her event için ayrı bir satır içeren kapsamlı bir CSV dosyası oluşturur. Bu CSV, makine öğrenmesi modelleri için hazır bir veri seti sağlar ve her event için gerekli tüm metadata'yı içerir.
