# Veri Seti Oluşturma Scripti

Bu klasör, SPRSound veri setinden event-level CSV dosyaları oluşturmak için kullanılan scripti içerir.

## 📋 Script

### `create_dataset.py`

SPRSound JSON dosyalarını tarayarak event-level CSV oluşturur ve WAV yollarını ekler.

**Özellikler:**
- ✅ Duplikasyon önleme (her dosya sadece bir kez işlenir)
- ✅ Tüm dataset'leri tarar (Classification, Detection, BioCAS)
- ✅ Hasta bilgilerini Patient Summary'den yükler
- ✅ Event bilgilerini çıkarır ve her event için ayrı satır oluşturur
- ✅ WAV dosya yollarını otomatik ekler
- ✅ 2022 Validation için özel yol düzeltmesi
- ✅ Detaylı istatistikler gösterir

**Kullanım:**
```bash
cd Project/Scripts
python create_dataset.py
```

**Çıktı:**
- `data/SPRSound_Event_Level_Dataset_CLEAN_with_WAV.csv` - Hazır veri seti

**İstatistikler:**
- Dataset bazında event sayıları
- Event type dağılımı
- Hastalık bazında event sayıları
- Dataset bazında WAV dosya durumu

---

## 📊 Çıktı Dosyası Yapısı

CSV dosyası şu kolonları içerir:

1. `dataset` - Veri seti adı
2. `file_path` - JSON dosya yolu
3. `filename` - Dosya adı
4. `patient_number` - Hasta numarası
5. `age` - Yaş
6. `gender` - Cinsiyet
7. `gender_code` - Cinsiyet kodu (0/1)
8. `recording_location` - Kayıt konumu kodu (p1-p8)
9. `recording_location_name` - Konum açıklaması
10. `recording_number` - Kayıt numarası
11. `record_annotation` - Dosya seviyesi annotation
12. `disease` - Hasta hastalık tanısı
13. `event_start_ms` - Event başlangıç zamanı (ms)
14. `event_end_ms` - Event bitiş zamanı (ms)
15. `event_duration_ms` - Event süresi (ms)
16. `event_type` - Event tipi
17. `event_index` - Event'in dosyadaki sırası
18. `total_events_in_file` - Dosyadaki toplam event sayısı
19. `wav_path` - WAV dosya yolu (add_wav_paths_to_csv.py ile eklenir)
20. `wav_exists` - WAV dosyası var mı? (add_wav_paths_to_csv.py ile eklenir)

---

## ⚙️ Gereksinimler

- Python 3.6+
- Standart kütüphaneler (os, json, csv, collections)

---

## 📝 Notlar

- Scriptler, proje kök dizinini otomatik olarak bulur
- Çıktı dosyaları `data/` klasörüne yazılır
- Duplikasyon önleme sayesinde her dosya sadece bir kez işlenir
- 2022 Validation dosyaları için özel yol düzeltmesi yapılır

---

## 🔍 Detaylı Dokümantasyon

Daha detaylı bilgi için `../Createdata.md` dosyasına bakın.
