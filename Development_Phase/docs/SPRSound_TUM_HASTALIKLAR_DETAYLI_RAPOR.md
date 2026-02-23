# SPRSound VERİ SETİ - TÜM HASTALIKLAR DETAYLI KLİNİK RAPOR

## 📋 İÇİNDEKİLER

1. [Veri Seti Genel Bakış](#genel-bakış)
2. [Hastalıklar Alfabetik İndeks](#alfabetik-indeks)
3. [Detaylı Hastalık Analizleri](#hastalık-analizleri)
4. [Hastalık-Ses İlişkileri Özet Tablosu](#ses-ilişkileri)
5. [Yaş Gruplarına Göre Hastalık Dağılımı](#yaş-dağılımı)
6. [Klinik Öncelik Sıralaması](#klinik-öncelik)

---

## 📊 GENEL BAKIŞ {#genel-bakış}

**Toplam Veri**: 1,181 kayıt
**Benzersiz Hasta**: 725+ hasta
**Hastalık Kategorisi**: 17 farklı tanı
**Yaş Aralığı**: 0.1 yaş (1 ay) - 16.8 yaş
**Veri Kaynakları**: 
- SPRSound Ana Dataset (292 kayıt)
- Grand Challenge 2023 (94 kayıt)
- Grand Challenge 2024 (795 kayıt)

---

## 🔤 ALFABETİK HASTALIK İNDEKSİ {#alfabetik-indeks}

| # | Hastalık | Vaka Sayısı | Sayfa |
|---|----------|-------------|-------|
| 1 | [Acute Upper Respiratory Infection](#auri) | 14 | ↓ |
| 2 | [Airway Foreign Body](#afb) | 2 | ↓ |
| 3 | [Asthma](#asthma) | 114 | ↓ |
| 4 | [Bronchiectasia](#bronchiectasia) | 9 | ↓ |
| 5 | [Bronchiolitis](#bronchiolitis) | 9 | ↓ |
| 6 | [Bronchitis](#bronchitis) | 144 | ↓ |
| 7 | [Chronic Cough](#chronic-cough) | 4 | ↓ |
| 8 | [Control Group (Sağlıklı)](#control) | 120 | ↓ |
| 9 | [Hemoptysis](#hemoptysis) | 4 | ↓ |
| 10 | [Kawasaki Disease](#kawasaki) | 1 | ↓ |
| 11 | [Other Respiratory Diseases](#other) | 49 | ↓ |
| 12 | [Pneumonia (Non-severe)](#pneumonia-ns) | 641 | ↓ |
| 13 | [Pneumonia (Severe)](#pneumonia-s) | 66 | ↓ |
| 14 | [Protracted Bacterial Bronchitis](#pbb) | 1 | ↓ |
| 15 | [Pulmonary Hemosiderosis](#hemosiderosis) | 2 | ↓ |

---

# 🏥 DETAYLI HASTALIK ANALİZLERİ {#hastalık-analizleri}

---

## 1. PNEUMONIA (NON-SEVERE) - Hafif/Orta Pnömoni {#pneumonia-ns}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 641 (%54.3 - EN YAYGIN)
- **Benzersiz Hasta**: 418
- **Yaş Aralığı**: 0.1 - 14.7 yaş
- **Ortalama Yaş**: 5.3 yaş
- **Cinsiyet Dağılımı**: 
  - Erkek: 289 (%45.1)
  - Kadın: 352 (%54.9) ← Kadınlarda daha yaygın

### 🔬 Klinik Tanım
Akciğer dokusunun enfeksiyonu. Alveollerde sıvı ve enflamatuar hücre birikimi. Pediatrik popülasyonda en yaygın alt solunum yolu enfeksiyonu.

### 🦠 Etiyoloji (Nedenler)
- **Bakteriyel**: Streptococcus pneumoniae (%40-50), Mycoplasma pneumoniae, H. influenzae
- **Viral**: RSV, Influenza, Adenovirus, Rhinovirus
- **Atipik**: Mycoplasma, Chlamydia

### 📋 Semptomlar
- Öksürük (produktif veya kuru)
- Ateş
- Hızlı/zorlu solunum (takipne)
- Göğüs ağrısı
- Letarji, iştahsızlık

### 🔊 Solunum Sesi Özellikleri
**Dominant Sesler**:
- **Fine Crackle** (İnce çatırtı) - +++
- **Coarse Crackle** (Kaba çatırtı) - ++
- Bronchial breathing (bronşiyal solunum)
- Egophony (keçi sesi)

**Beklenen Record-Level Etiket**: 
- `DAS` (Discontinuous Adventitious Sounds) - Crackle dominant
- Bazı vakalarda `Normal` (hafif/iyileşme aşaması)

**Beklenen Event-Level Etiketler**:
- `Fine Crackle` - Alveolar involvement
- `Normal` - Etkilenmemiş bölgeler
- Nadiren `Coarse Crackle` - Bronchial secretions

### 🎯 Klinik Önem
- **Yüksek**: Sık görülür, hospitalizasyon gerektirebilir
- **Mortalite**: Düşük (non-severe)
- **Komplikasyonlar**: Plevral efüzyon, ampiyem (nadir)

### 💡 Model İpuçları
- Fine Crackle tespiti kritik
- Crackle sayısı ve yoğunluğu şiddet göstergesi
- Temporal pattern: Inspiratory crackles
- Lokasyon: Unilateral veya bilateral

---

## 2. BRONCHITIS - Bronşit {#bronchitis}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 144 (%12.2 - 2. EN YAYGIN)
- **Benzersiz Hasta**: 108
- **Yaş Aralığı**: 0.2 - 16.8 yaş
- **Ortalama Yaş**: 4.3 yaş (daha genç)
- **Cinsiyet Dağılımı**: 
  - Erkek: 83 (%57.6) ← Erkeklerde daha yaygın
  - Kadın: 61 (%42.4)

### 🔬 Klinik Tanım
Bronş ağacının enflamasyonu. Pediatrik popülasyonda genellikle akut viral bronşit. Bronşiyal duvarın ödemi ve mukus hipersekresyonu.

### 🦠 Etiyoloji
- **Viral** (%90): RSV, Influenza, Parainfluenza, Rhinovirus, Adenovirus
- **Bakteriyel** (nadir): M. catarrhalis, S. pneumoniae, H. influenzae
- **İrritanlar**: Pasif sigara dumanı, hava kirliliği

### 📋 Semptomlar
- Produktif öksürük
- Wheezing (hırıltı)
- Hafif ateş
- Göğüste tıkanıklık hissi
- Mukus (renksiz veya sarımsı)

### 🔊 Solunum Sesi Özellikleri
**Dominant Sesler**:
- **Wheeze** (Hırıltı) - +++
- **Rhonchi** (Ronküs) - +++
- **Coarse Crackle** (Kaba çatırtı) - ++
- Rale sesleri

**Beklenen Record-Level Etiket**: 
- `CAS` (Continuous Adventitious Sounds) - Wheeze/Rhonchi dominant
- Bazı vakalarda `CAS & DAS` - Mixed pattern

**Beklenen Event-Level Etiketler**:
- `Wheeze` - Airway narrowing
- `Rhonchi` - Mucus in airways
- `Coarse Crackle` - Secretions
- `Normal` - Ekspiratory fazda daha belirgin

### 🎯 Klinik Önem
- **Orta**: Self-limiting, genellikle 1-3 hafta
- **Komplikasyonlar**: Nadir (pnömoniye ilerleyebilir)
- **RSV İlişkisi**: ⚠️ Yüksek - Özellikle bebeklerde

### 💡 Model İpuçları
- Wheeze ve Rhonchi kombinasyonu tipik
- Düşük frekanslı, continuous sesler
- Öksürükle değişen sesler
- Bilateral pattern

---

## 3. ASTHMA - Astım {#asthma}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 114 (%9.7 - 3. EN YAYGIN)
- **Benzersiz Hasta**: 71
- **Yaş Aralığı**: 0.9 - 13.9 yaş
- **Ortalama Yaş**: 6.8 yaş (daha yaşlı çocuklar)
- **Cinsiyet Dağılımı**: 
  - Erkek: 71 (%62.3) ← Erkeklerde 2x daha fazla
  - Kadın: 43 (%37.7)

### 🔬 Klinik Tanım
Kronik hava yolu enflamasyonu. Reversible airway obstruction. Bronkospazm, mukus hipersekresyonu ve hava yolu hiperreaktivitesi ile karakterize.

### 🦠 Etiyoloji
- **Genetik**: Atopi, aile öyküsü
- **Tetikleyiciler**: Alerjenler (polen, toz akarı), egzersiz, viral enfeksiyonlar
- **Çevresel**: Sigara dumanı, hava kirliliği
- **Immunolojik**: IgE-mediated

### 📋 Semptomlar
- Wheezing (özellikle ekspiratory)
- Dispne (nefes darlığı)
- Göğüste sıkışma
- Öksürük (özellikle gece/sabah)
- Atak şeklinde (episodik)

### 🔊 Solunum Sesi Özellikleri
**Dominant Sesler**:
- **Wheeze** (Hırıltı) - ++++ (EN DOMINANT)
- Polyphonic wheeze (çoklu frekanslı)
- Ekspiratory uzaması
- Sessiz göğüs (severe attack)

**Beklenen Record-Level Etiket**: 
- `CAS` (Continuous Adventitious Sounds) - Pure wheeze

**Beklenen Event-Level Etiketler**:
- `Wheeze` - ++++ (predominant)
- `Normal` - İyi kontrol edilen astım
- Nadiren `Fine Crackle` - Eşlik eden enfeksiyon

### 🎯 Klinik Önem
- **Yüksek**: Kronik hastalık, yaşam kalitesi etkiler
- **Mortalite**: Düşük ama status asthmaticus tehlikeli
- **Morbidite**: Okul devamsızlığı, acil başvurular

### 💡 Model İpuçları
- Yüksek frekanslı wheeze
- Ekspiratory fazda dominant
- Polyphonic (birden fazla frekans)
- Bronkodilatatöre yanıt (reversible)
- Temporal variation (circadian rhythm)

---

## 4. CONTROL GROUP - Sağlıklı Kontrol Grubu {#control}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 120 (105 + 15) (%10.2)
- **Benzersiz Hasta**: 100
- **Yaş Aralığı**: 0.2 - 15.3 yaş
- **Ortalama Yaş**: 5.3 yaş
- **Cinsiyet Dağılımı**: 
  - Erkek: 70 (%58.3)
  - Kadın: 50 (%41.7)

### 🔬 Klinik Tanım
Sağlıklı çocuklar. Solunum hastalığı yok. Normal fiziksel muayene. Model validasyonu ve normal ses paterni öğrenimi için kritik.

### 📋 Kriterler
- Akut/kronik solunum hastalığı yok
- Normal fiziksel muayene
- Normal vital signs
- İyi genel durum

### 🔊 Solunum Sesi Özellikleri
**Normal Sesler**:
- Vesicular breathing (veziküler solunum)
- Inspiratory/Ekspiratory ratio: 3:1
- Yumuşak, düşük frekanslı
- Adventitious ses YOK

**Beklenen Record-Level Etiket**: 
- `Normal` - %100

**Beklenen Event-Level Etiketler**:
- `Normal` - %100

### 🎯 Klinik Önem
- **Kritik**: Baseline referans
- Model training için negatif sınıf
- Specificity hesaplaması için gerekli

### 💡 Model İpuçları
- Temiz sinyal
- Adventitious ses yokluğu
- Düzenli solunum paterni
- Lokasyonlar arası konsistans

---

## 5. PNEUMONIA (SEVERE) - Ağır Pnömoni {#pneumonia-s}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 66 (%5.6)
- **Benzersiz Hasta**: 43
- **Yaş Aralığı**: 0.2 - 16.2 yaş
- **Ortalama Yaş**: 5.2 yaş
- **Cinsiyet Dağılımı**: 
  - Erkek: 27 (%40.9)
  - Kadın: 39 (%59.1) ← Kadınlarda daha fazla

### 🔬 Klinik Tanım
Yaşamı tehdit eden pnömoni. Yoğun bakım gerektirebilir. Bilateral veya multilobar tutulum. Komplikasyonlar mevcut veya yüksek risk.

### 🦠 Etiyoloji
- **Bakteriyel**: S. pneumoniae (empyema), S. aureus (MRSA)
- **Viral**: Influenza, RSV (bebekler)
- **Komplike**: Necrotizing pneumonia, sepsis

### 📋 Semptomlar
- Ciddi takipne (hızlı solunum)
- Retraksiyon (göğüs çekilmesi)
- Siyanoz (morarmak)
- Letarji, bilinç değişikliği
- Hipoksi (O2 saturasyonu ↓)
- Dehidratasyon

### 🔊 Solunum Sesi Özellikleri
**Dominant Sesler**:
- **Fine Crackle** - ++++ (yaygın)
- **Coarse Crackle** - +++
- Bronchial breathing
- Decreased breath sounds (plevral efüzyon)
- Pleural friction rub (plevral tutulum)

**Beklenen Record-Level Etiket**: 
- `DAS` - Yaygın crackle
- Bazı vakalarda `CAS & DAS` - Mixed pattern

**Beklenen Event-Level Etiketler**:
- `Fine Crackle` - ++++
- `Coarse Crackle` - +++
- Nadiren `Wheeze` - Bronchial involvement

### 🎯 Klinik Önem
- **ÇOK YÜKSEK**: Hastaneye yatış, ICU potansiyeli
- **Mortalite**: Orta-yüksek risk
- **Komplikasyonlar**: Empyema, sepsis, ARDS

### 💡 Model İpuçları
- Yaygın, yoğun crackle
- Bilateral pattern
- Ses azalması (konsolidasyon)
- Non-severe'dan ayırt etme kritik

---

## 6. OTHER RESPIRATORY DISEASES - Diğer Solunum Hastalıkları {#other}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 49 (40 + 9) (%4.1)
- **Benzersiz Hasta**: 45
- **Yaş Aralığı**: 0.4 - 13.3 yaş
- **Ortalama Yaş**: 6.1 yaş
- **Cinsiyet Dağılımı**: 
  - Erkek: 32 (%65.3)
  - Kadın: 17 (%34.7)

### 🔬 Klinik Tanım
Spesifik kategoriye girmeyen solunum hastalıkları. Heterojen grup. Viral enfeksiyonlar, nadir patolojiler, atipik prezentasyonlar.

### 🦠 Olası Tanılar
- Viral üst solunum yolu enfeksiyonları
- RSV enfeksiyonu (bronşiyolit dışı)
- Adenovirus pnömonisi
- İnterstisyel akciğer hastalıkları
- Nadir patolojiler

### 🔊 Solunum Sesi Özellikleri
**Değişken**:
- Tanıya bağlı farklılık
- `Normal`, `CAS`, `DAS` veya `CAS & DAS` olabilir
- Heterojen patern

**Beklenen Event-Level Etiketler**:
- Tüm etiket tipleri görülebilir

### 🎯 Klinik Önem
- **Değişken**: Hafiften ciddi'ye kadar

### 💡 Model İpuçları
- Heterojen veri
- Outlier olarak işlenebilir
- Advanced clustering gerekebilir

---

## 7. ACUTE UPPER RESPIRATORY INFECTION - Akut Üst Solunum Yolu Enfeksiyonu {#auri}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 14 (%1.2)
- **Benzersiz Hasta**: 14
- **Yaş Aralığı**: 2.3 - 14.9 yaş
- **Ortalama Yaş**: 6.5 yaş
- **Cinsiyet Dağılımı**: 
  - Erkek: 7 (%50)
  - Kadın: 7 (%50)

### 🔬 Klinik Tanım
Üst solunum yolları enfeksiyonu (burun, farinks, larinks). "Common cold" (soğuk algınlığı), farenjit, tonsillit. Genellikle viral, self-limiting.

### 🦠 Etiyoloji
- **Viral**: Rhinovirus (%50), Coronavirus, RSV, Adenovirus
- **Bakteriyel** (nadir): Strep throat (GAS)

### 📋 Semptomlar
- Burun akıntısı
- Faringeal irritasyon
- Öksürük (kuru, irritatif)
- Ateş (düşük-orta)
- Boğaz ağrısı

### 🔊 Solunum Sesi Özellikleri
**Genellikle Normal**:
- Alt solunum yolları etkilenmez
- Normal vesicular breathing
- Nadiren üst hava yolu sesleri (stridor - krup durumunda)

**Beklenen Record-Level Etiket**: 
- `Normal` - En yaygın
- Bazı vakalarda `CAS` - Üst hava yolu irritasyonu

**Beklenen Event-Level Etiketler**:
- `Normal` - +++
- Nadiren `Wheeze` veya `Stridor`

### 🎯 Klinik Önem
- **Düşük**: Self-limiting, komplikasyon nadir
- **Komplikasyonlar**: Sinüzit, otitis media, alt solunum yoluna yayılım

### 💡 Model İpuçları
- Genellikle temiz sesler
- Üst hava yolu varyasyonları
- Alt solunum yolu patolojisinden ayırt

---

## 8. BRONCHIOLITIS - Bronşiyolit {#bronchiolitis}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 9 (%0.8) ⚠️ AZ VAKA
- **Benzersiz Hasta**: 9
- **Yaş Aralığı**: 0.3 - 5.8 yaş
- **Ortalama Yaş**: 1.3 yaş ← EN GENÇ GRUP
- **Cinsiyet Dağılımı**: 
  - Erkek: 7 (%77.8) ← Erkeklerde çok daha yaygın
  - Kadın: 2 (%22.2)

### 🔬 Klinik Tanım
Küçük hava yollarının (bronşioller) akut viral enflamasyonu. **Genellikle <2 yaş bebekler**. En yaygın nedeni RSV. Bronşiol mukozasında ödem, nekroz ve mukus tıkacı.

### 🦠 Etiyoloji
- **RSV (Respiratory Syncytial Virus)**: %70-80 ⚠️⚠️⚠️
- **Diğer virusler**: Rhinovirus, Parainfluenza, Adenovirus, Human metapneumovirus
- **Mevsimsel**: Kış ayları epidemik

### 📋 Semptomlar
- Wheezing (ilk kez wheezing)
- Takipne (hızlı solunum)
- Kosta altı/interkostal retraksiyonlar
- Nazal flaring (burun kanadı açılması)
- Öksürük
- Beslenme güçlüğü
- Apne (küçük bebeklerde)

### 🔊 Solunum Sesi Özellikleri
**KARAKTERİSTİK PATERN**:
- **Wheeze** (Bilateral) - +++
- **Fine Crackle** (İnce çatırtı) - +++ 
- Mixed pattern: **Wheeze + Crackle** ← TIPIK!
- Prolonged ekspiratory faz
- Bilateral, yaygın

**Beklenen Record-Level Etiket**: 
- `CAS & DAS` - MIXED PATTERN (en tipik)
- Bazı vakalarda `CAS` veya `DAS`

**Beklenen Event-Level Etiketler**:
- `Wheeze` - +++
- `Fine Crackle` - +++
- `Wheeze+Crackle` - ++ (kombine event)
- `Normal` - Hafif vakalar

### 🎯 Klinik Önem
- **ÇOK YÜKSEK**: Bebekler için ciddi
- **Hospitalizasyon**: %3 (genelde <6 ay)
- **Mortalite**: Düşük ama risk var (prematüreler, kardiyak sorunlar)
- **RSV**: Aşı/profilaksi (palivizumab) geliştirilmekte

### 🎯 RSV İLİŞKİSİ
🚨 **EN YÜKSEK RSV OLASILĞI**: Bu 9 vakanın %70-80'i (6-7 vaka) muhtemelen RSV!

### 💡 Model İpuçları
- **Wheeze + Fine Crackle kombinasyonu** → Bronchiolitis marker
- Yaş <2 yaş → Bronchiolitis olasılığı ↑
- Bilateral, yaygın pattern
- Kış mevsimi kayıtları (varsa timestamp)
- Temporal pattern: Viral course (kötüleşme 3-5. gün)

### 🔬 Araştırma Potansiyeli
- **RSV tespiti** için proxy olarak kullanılabilir
- Ses özellikleri ile virolojik doğrulama korelasyonu
- Erken tanı modelleri (RSV testine gerek kalmadan)

---

## 9. BRONCHIECTASIA - Bronşektazi {#bronchiectasia}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 9 (%0.8)
- **Benzersiz Hasta**: 9
- **Yaş Aralığı**: 4.0 - 13.2 yaş
- **Ortalama Yaş**: 9.0 yaş ← DAHA BÜYÜK ÇOCUKLAR
- **Cinsiyet Dağılımı**: 
  - Erkek: 6 (%66.7)
  - Kadın: 3 (%33.3)

### 🔬 Klinik Tanım
Bronşların kalıcı, irreversible dilatasyonu. Kronik enflamasyon ve rekürren enfeksiyonlar. Mukus klirensi bozulmuş. Pediatrik popülasyonda nadir ama ciddi.

### 🦠 Etiyoloji
- **Post-enfeksiyöz**: Ağır pnömoni, tüberküloz
- **Kistik Fibrozis**
- **İmmun yetmezlik**: Primary ciliary dyskinesia
- **Aspiration**: Rekürren aspirasyon
- **Konjenital**: Bronş anomalileri

### 📋 Semptomlar
- Kronik produktif öksürük
- Bol pürulan balgam
- Hemoptysis (kan tükürme)
- Rekürren akciğer enfeksiyonları
- Wheezing
- Clubbing (tırnak çomaklaşması)

### 🔊 Solunum Sesi Özellikleri
**Karakteristik**:
- **Coarse Crackle** (Kaba çatırtı) - ++++ (DOMINANT)
- Lokalize veya yaygın
- Persistent (değişmez)
- Secretion sounds
- Wheeze (eşlik edebilir)

**Beklenen Record-Level Etiket**: 
- `DAS` - Coarse crackle
- Bazı vakalarda `CAS & DAS`

**Beklenen Event-Level Etiketler**:
- `Coarse Crackle` - ++++
- `Rhonchi` - ++ (sekresyon)
- `Wheeze` - + (obstrüksiyon)

### 🎯 Klinik Önem
- **Yüksek**: Kronik, progresif hastalık
- **Morbidite**: Yaşam kalitesi ↓, akciğer fonksiyonu ↓
- **Tedavi**: Antibiyotik, fizyoterapi, cerrahi (lokal)

### 💡 Model İpuçları
- Persistent coarse crackle
- Lokalizasyon sabit
- Öksürükle değişmeyen
- Temporal stability

---

## 10. CHRONIC COUGH - Kronik Öksürük {#chronic-cough}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 4 (%0.3) - ÇOK NADIR
- **Benzersiz Hasta**: 4
- **Yaş Aralığı**: 4.0 - 10.7 yaş
- **Ortalama Yaş**: 7.4 yaş
- **Cinsiyet Dağılımı**: 
  - Erkek: 3 (%75)
  - Kadın: 1 (%25)

### 🔬 Klinik Tanım
4 haftadan uzun süren öksürük. Altında yatan patoloji araştırılmalı. Pediatrik popülasyonda çok sayıda neden.

### 🦠 Etiyoloji
- **Post-infectious**: Viral enfeksiyon sonrası
- **Astım**: Variant astım
- **Üst hava yolu**: Postnasal drip
- **GERD**: Gastro-özofageal reflü
- **Pertussis** (Boğmaca)
- **Psikojenik**: Habit cough

### 🔊 Solunum Sesi Özellikleri
**Değişken**:
- Nedenine bağlı
- Normal veya spesifik patoloji sesleri

### 🎯 Klinik Önem
- **Orta**: Altta yatan neden önemli
- Yaşam kalitesi etkilenir

---

## 11. HEMOPTYSIS - Hemoptizi (Kan Tükürme) {#hemoptysis}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 4 (%0.3) - ÇOK NADIR
- **Benzersiz Hasta**: 4
- **Yaş Aralığı**: 6.7 - 13.2 yaş
- **Ortalama Yaş**: 11.1 yaş ← DAHA BÜYÜK ÇOCUKLAR
- **Cinsiyet Dağılımı**: 
  - Erkek: 2 (%50)
  - Kadın: 2 (%50)

### 🔬 Klinik Tanım
Solunum yollarından kan gelmesi. Pediatrik popülasyonda nadir ama ciddi. Acil değerlendirme gerektirir.

### 🦠 Etiyoloji
- **Enfeksiyöz**: Tüberküloz, bronşit, pnömoni
- **Bronşektazi**: Kronik hastalık
- **Yabancı cisim**: Aspirasyon
- **Vasküler**: Arteriovenöz malformasyon
- **Tümör** (çok nadir pediatrik)

### 🔊 Solunum Sesi Özellikleri
**Altta yatan hastalığa bağlı**:
- Coarse crackle (bronşektazi)
- Normal veya azalmış sesler

### 🎯 Klinik Önem
- **ÇOK YÜKSEK**: Potansiyel acil durum
- Etiyoloji araştırması kritik

---

## 12. AIRWAY FOREIGN BODY - Hava Yolu Yabancı Cisim {#afb}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 2 (%0.2) - ÇOK NADIR
- **Benzersiz Hasta**: 2
- **Yaş Aralığı**: 3.0 - 6.5 yaş
- **Ortalama Yaş**: 4.8 yaş ← KÜÇÜK ÇOCUKLAR
- **Cinsiyet Dağılımı**: 
  - Erkek: 2 (%100)
  - Kadın: 0

### 🔬 Klinik Tanım
Trakeobronşiyal ağaca yabancı cisim aspirasyonu. Pediatrik acil. En yaygın 1-3 yaş. Fındık, fıstık, oyuncak parçaları.

### 📋 Semptomlar
- Ani başlangıçlı öksürük, wheezing
- Stridor (larengeal obstrüksiyon)
- Unilateral azalmış sesler
- Asfiksi riski (tam tıkanıklık)

### 🔊 Solunum Sesi Özellikleri
**Karakteristik**:
- **Unilateral** azalmış/olmayan sesler
- **Wheeze** (lokal)
- **Stridor** (larengeal/trakeal)
- Ekspiratory obstrüksiyon

**Beklenen Record-Level Etiket**: 
- `CAS` veya `Normal` (tek taraflı)

**Beklenen Event-Level Etiketler**:
- `Wheeze` veya `Stridor`
- Asimetri

### 🎯 Klinik Önem
- **ACİL DURUM**: Bronkoskopi gerekir
- **Mortalite**: Yüksek (tam obstrüksiyon)

### 💡 Model İpuçları
- Unilateral pattern
- Ani başlangıç (anamnez önemli)
- Lokalize wheeze

---

## 13. PULMONARY HEMOSIDEROSIS - Pulmoner Hemosiderozis {#hemosiderosis}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 2 (%0.2) - ÇOK NADIR
- **Benzersiz Hasta**: 2
- **Yaş Aralığı**: 4.7 - 5.5 yaş
- **Ortalama Yaş**: 5.1 yaş
- **Cinsiyet Dağılımı**: 
  - Erkek: 0
  - Kadın: 2 (%100) ← SADECE KADINLAR

### 🔬 Klinik Tanım
Alveollerde rekürren kanama. Demir (hemosiderin) birikimi. Nadir, kronik hastalık. Hipokromik anemi, hemoptizi, akciğer infiltratları triadı.

### 🦠 Etiyoloji
- **İdiopatik**: Heiner sendromu
- **İmmunolojik**: Goodpasture sendromu
- **Sekonder**: Kardiyak hastalık, vaskülit

### 🔊 Solunum Sesi Özellikleri
- Fine crackle (kanama)
- Değişken pattern

### 🎯 Klinik Önem
- **Yüksek**: Nadir ama ciddi
- Kronik tedavi gerektirir

---

## 14. PROTRACTED BACTERIAL BRONCHITIS - Uzamış Bakteriyel Bronşit {#pbb}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 1 (%0.1) - ÇOK NADIR
- **Benzersiz Hasta**: 1
- **Yaş**: 2.6 yaş
- **Cinsiyet**: Erkek

### 🔬 Klinik Tanım
Kronik produktif öksürük (>4 hafta). Bakteriyel enfeksiyon (H. influenzae, S. pneumoniae). Antibiyotik tedavisine yanıt veren.

### 🔊 Solunum Sesi Özellikleri
- Coarse crackle, Rhonchi
- Produktif sesler

---

## 15. KAWASAKI DISEASE - Kawasaki Hastalığı {#kawasaki}

### 📊 Veri Seti İstatistikleri
- **Toplam Vaka**: 1 (%0.1) - ÇOK NADIR
- **Benzersiz Hasta**: 1
- **Yaş**: 1.5 yaş
- **Cinsiyet**: Erkek

### 🔬 Klinik Tanım
Sistemik vaskülit. Orta boy damarları tutar. Kardiyak komplikasyonlar (koroner arter anevrizma). Primer olarak solunum hastalığı DEĞİL.

### 📋 Semptomlar
- Yüksek ateş (>5 gün)
- Konjonktival injeksiyon
- Oral mukoza değişiklikleri
- Rash (döküntü)
- Lenfadenopati
- **Solunum**: Nadir, interstisyel pnömonit

### 🔊 Solunum Sesi Özellikleri
- Genellikle normal
- Nadiren fine crackle (pnömonit)

### 🎯 Klinik Önem
- **Yüksek**: Kardiyak komplikasyon riski
- Erken IVIG tedavisi kritik

---

# 📊 HASTALIK-SES İLİŞKİLERİ ÖZET TABLOSU {#ses-ilişkileri}

| Hastalık | Record-Level | Event-Level (Dominant) | Ses Şiddeti | Karakteristik Özellik |
|----------|-------------|----------------------|------------|---------------------|
| **Pneumonia (non-severe)** | DAS | Fine Crackle +++ | Orta | Inspiratory crackle |
| **Pneumonia (severe)** | DAS | Fine+Coarse Crackle ++++ | Yüksek | Yaygın, bilateral |
| **Bronchitis** | CAS veya CAS&DAS | Wheeze++, Rhonchi++ | Orta | Secretion sounds |
| **Asthma** | CAS | Wheeze ++++ | Yüksek | Polyphonic, ekspiratory |
| **Bronchiolitis** 🦠 | **CAS & DAS** | **Wheeze + Fine Crackle** | Orta-Yüksek | **MIXED (tipik)** |
| **Bronchiectasia** | DAS | Coarse Crackle ++++ | Yüksek | Persistent, lokalize |
| **Control** | Normal | Normal | - | Temiz sesler |
| **AURI** | Normal | Normal (Nadiren Stridor) | Düşük | Üst hava yolu |
| **Foreign Body** | CAS | Wheeze/Stridor (unilateral) | Değişken | **Asimetrik** |
| **Other** | Değişken | Değişken | Değişken | Heterojen |

**Legend**:
- `+` = Hafif
- `++` = Orta
- `+++` = Belirgin
- `++++` = Çok belirgin/dominant

---

# 📈 YAŞ GRUPLARINA GÖRE HASTALIK DAĞILIMI {#yaş-dağılımı}

## 👶 Bebekler (0-2 yaş)

**En Yaygın Hastalıklar**:
1. **Bronchiolitis** (ort. 1.3 yaş) 🦠 RSV+++
2. Pneumonia (0.1+ yaş)
3. Bronchitis (0.2+ yaş)

**Klinik Önem**: RSV mevsimi, hospitalizasyon riski ↑

---

## 🧒 Erken Çocukluk (2-5 yaş)

**En Yaygın Hastalıklar**:
1. Pneumonia
2. Bronchitis
3. Asthma (başlangıç)

**Klinik Önem**: İlk wheezing epizodu, astım ayırıcı tanısı

---

## 👦 Okul Çağı (6-12 yaş)

**En Yaygın Hastalıklar**:
1. Asthma (ort. 6.8 yaş)
2. Pneumonia
3. Bronchiectasia (ort. 9.0 yaş)

**Klinik Önem**: Kronik hastalıklar, okul devamsızlığı

---

## 🧑 Ergenler (12+ yaş)

**En Yaygın Hastalıklar**:
1. Asthma
2. Pneumonia
3. Hemoptysis (ort. 11.1 yaş)

**Klinik Önem**: Compliance sorunları, kronik hastalık yönetimi

---

# 🎯 KLİNİK ÖNCELİK SIRALAMASI {#klinik-öncelik}

## 🚨 ACİL/KRİTİK (Immediate)

1. **Airway Foreign Body** - Asfiksi riski
2. **Pneumonia (Severe)** - Sepsis, ARDS riski
3. **Hemoptysis** - Masif kanama potansiyeli

## ⚠️ YÜKSEK ÖNCELİK (Urgent)

4. **Bronchiolitis** - Özellikle <6 ay bebek
5. **Pneumonia (Non-severe)** - Kötüleşme riski
6. **Asthma** - Atak durumu

## 🟡 ORTA ÖNCELİK (Semi-urgent)

7. **Bronchiectasia** - Kronik, komplikasyon riski
8. **Bronchitis** - Self-limiting ama takip
9. **Pulmonary Hemosiderosis** - Kronik yönetim

## 🟢 DÜŞÜK ÖNCELİK (Routine)

10. **AURI** - Self-limiting
11. **Chronic Cough** - Araştırma
12. **Control** - Sağlıklı

---

# 💡 MODEL GELİŞTİRME İÇİN ÖZEL NOTLAR

## Sınıf Dengesizliği

**Aşırı Temsil Edilen** (Data Augmentation GEREKMEYEBİLİR):
- Pneumonia (non-severe): 641 vaka
- Bronchitis: 144 vaka
- Asthma: 114 vaka

**Az Temsil Edilen** (Data Augmentation ŞART):
- Bronchiolitis: 9 vaka 🦠
- Bronchiectasia: 9 vaka
- Chronic cough: 4 vaka
- Hemoptysis: 4 vaka
- Airway foreign body: 2 vaka
- Pulmonary hemosiderosis: 2 vaka
- PBB: 1 vaka
- Kawasaki: 1 vaka

## Ses-Hastalık Korelasyonu

**Yüksek Korelasyon** (Model güvenilir):
- Asthma → Wheeze (çok güçlü)
- Pneumonia → Fine Crackle (güçlü)
- Bronchiectasia → Coarse Crackle (güçlü)
- Bronchiolitis → Wheeze+Crackle (karakteristik)

**Düşük Korelasyon** (Zorlu sınıflar):
- Bronchitis vs Asthma (her ikisi wheeze)
- Pneumonia (non-severe vs severe) (her ikisi crackle)
- Other diseases (heterojen)

## Multi-label vs Single-label

Bazı hastalar birden fazla kayda sahip → Temporal evolution analizi mümkün

## Transfer Learning Potansiyeli

- Pneumonia modeli → Genel crackle detection
- Asthma modeli → Genel wheeze detection
- Bronchiolitis modeli → RSV prediction (indirect)

---

# 📚 REFERANSLAR VE KAYNAKLAR

1. **SPRSound Dataset**: Zhang et al., IEEE TBioCAS, 2022
2. **Pediatric Pneumonia**: WHO Guidelines, 2014
3. **Bronchiolitis Management**: AAP Clinical Practice Guideline, 2014
4. **RSV Epidemiology**: Hall et al., NEJM, 2009
5. **Pediatric Asthma**: GINA Guidelines, 2023
6. **Bronchiectasis**: ERS Guidelines, 2017

---

**Rapor Hazırlayan**: AI Medical Assistant  
**Tarih**: Ocak 2026  
**Veri Seti**: SPRSound (v2022-2024)  
**Toplam Analiz Edilen Vaka**: 1,181  
**Hastalık Kategorisi**: 17

---

## 🔍 ÖZEL NOTLAR

### RSV (Respiratory Syncytial Virus) Çalışması İçin:

🎯 **Hedef Hastalık**: Bronchiolitis (9 vaka)
- Veri setindeki 9 Bronchiolitis vakasının **%70-80'i (6-7 vaka) muhtemelen RSV**
- Yaş: 0.3-5.8 yaş (ort. 1.3) - RSV peak age ile uyumlu
- Ses paterni: **Wheeze + Fine Crackle** - RSV tipik
- Erkek dominant (%78) - RSV epidemiyolojisi ile uyumlu

🔬 **Yaklaşım**:
1. Bronchiolitis vakalarını ayrı analiz et
2. Yaş <2 filtrele → RSV olasılığı ↑↑
3. Wheeze+Crackle kombinasyonunu ara
4. Data augmentation (az vaka var)
5. External RSV-labeled dataset ile transfer learning

---

**SON GÜNCELLEME**: 2026-01-14
