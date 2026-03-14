# Hastalık - Event Type İlişkisi: Detaylı Analiz

## Özet

Bu rapor, SPRSound veri setindeki **16 farklı hastalık** ile **8 farklı event type** arasındaki ilişkiyi detaylı olarak analiz eder.

**Temel İstatistikler:**
- Toplam Event Sayısı: **24,808**
- Unique Hastalık Sayısı: **16**
- Unique Event Type Sayısı: **8**

---

## 📊 Tam Pivot Tablo: Mutlak Sayılar

| Hastalık | Coarse Crackle | Fine Crackle | No Event | Normal | Rhonchi | Stridor | Wheeze | Wheeze+Crackle | **TOPLAM** |
|----------|----------------|--------------|----------|--------|---------|---------|--------|----------------|------------|
| **Pneumonia (non-severe)** | 50 | 1,341 | 146 | 8,821 | 104 | 2 | 569 | 34 | **11,067** |
| **Unknown** | 65 | 1,838 | 3 | 2,367 | 9 | 8 | 262 | 261 | **4,813** |
| **Bronchitis** | 18 | 141 | 6 | 1,906 | 53 | 15 | 183 | 0 | **2,322** |
| **Control Group** | 10 | 2 | 27 | 1,932 | 34 | 23 | 16 | 0 | **2,044** |
| **Asthma** | 1 | 28 | 15 | 1,379 | 0 | 7 | 162 | 0 | **1,592** |
| **Pneumonia (severe)** | 32 | 117 | 9 | 952 | 8 | 0 | 146 | 7 | **1,271** |
| **Other respiratory diseases** | 0 | 2 | 2 | 365 | 2 | 19 | 20 | 1 | **411** |
| **Bronchiectasia** | 0 | 41 | 3 | 258 | 7 | 0 | 7 | 0 | **316** |
| **Bronchiolitis** | 0 | 17 | 0 | 157 | 0 | 0 | 131 | 0 | **305** |
| **Acute upper respiratory infection** | 0 | 2 | 1 | 233 | 0 | 0 | 0 | 0 | **236** |
| **Hemoptysis** | 0 | 1 | 15 | 218 | 0 | 0 | 0 | 0 | **234** |
| **Pulmonary hemosiderosis** | 0 | 0 | 2 | 70 | 0 | 0 | 0 | 0 | **72** |
| **Chronic cough** | 1 | 0 | 1 | 53 | 0 | 0 | 0 | 0 | **55** |
| **Airway foreign body** | 0 | 0 | 0 | 38 | 0 | 0 | 0 | 0 | **38** |
| **Protracted bacterial bronchitis** | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | **21** |
| **Kawasaki disease** | 0 | 0 | 0 | 2 | 0 | 0 | 9 | 0 | **11** |

---

## 📈 Yüzde Dağılımı (Hastalık İçinde)

Her satır %100'e tamamlanır - yani her hastalığın kendi içindeki event dağılımı.

| Hastalık | Coarse Crackle | Fine Crackle | No Event | Normal | Rhonchi | Stridor | Wheeze | Wheeze+Crackle |
|----------|----------------|--------------|----------|--------|---------|---------|--------|----------------|
| **Pneumonia (non-severe)** | 0.5% | 12.1% | 1.3% | **79.7%** | 0.9% | 0.0% | 5.1% | 0.3% |
| **Unknown** | 1.4% | 38.2% | 0.1% | **49.2%** | 0.2% | 0.2% | 5.4% | 5.4% |
| **Bronchitis** | 0.8% | 6.1% | 0.3% | **82.1%** | 2.3% | 0.6% | 7.9% | 0.0% |
| **Control Group** | 0.5% | 0.1% | 1.3% | **94.5%** | 1.7% | 1.1% | 0.8% | 0.0% |
| **Asthma** | 0.1% | 1.8% | 0.9% | **86.6%** | 0.0% | 0.4% | 10.2% | 0.0% |
| **Pneumonia (severe)** | 2.5% | 9.2% | 0.7% | **74.9%** | 0.6% | 0.0% | 11.5% | 0.6% |
| **Other respiratory diseases** | 0.0% | 0.5% | 0.5% | **88.8%** | 0.5% | 4.6% | 4.9% | 0.2% |
| **Bronchiectasia** | 0.0% | 13.0% | 0.9% | **81.6%** | 2.2% | 0.0% | 2.2% | 0.0% |
| **Bronchiolitis** | 0.0% | 5.6% | 0.0% | **51.5%** | 0.0% | 0.0% | **43.0%** | 0.0% |
| **Acute upper respiratory infection** | 0.0% | 0.8% | 0.4% | **98.7%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Hemoptysis** | 0.0% | 0.4% | 6.4% | **93.2%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Pulmonary hemosiderosis** | 0.0% | 0.0% | 2.8% | **97.2%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Chronic cough** | 1.8% | 0.0% | 1.8% | **96.4%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Airway foreign body** | 0.0% | 0.0% | 0.0% | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Protracted bacterial bronchitis** | 0.0% | 0.0% | 0.0% | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Kawasaki disease** | 0.0% | 0.0% | 0.0% | 18.2% | 0.0% | 0.0% | **81.8%** | 0.0% |

---

## ❓ KRİTİK SORU: "Severe Pneumonia'da %75 Normal Ses Nasıl Olabilir?"

### Klinik Açıklama

**Severe Pneumonia** hastalarında **%74.9 Normal** ses çok şaşırtıcı görünebilir, ancak bunun birkaç önemli klinik ve teknik açıklaması var:

#### 1. **Fokal (Odaksal) Hastalık Karakteri** 🎯

Pneumonia genellikle **fokal** bir hastalıktır:
- Akciğerlerin sadece **belirli lob veya segmentlerini** etkiler
- Tüm akciğer yüzeyinin %20-40'ı etkilenmiş olabilir
- **Etkilenmeyen bölgeler tamamen normal ses üretir**

**SPRSound kayıt sistemi:**
- 4 farklı göğüs lokasyonunda (p1, p2, p3, p4) kayıt alır
- Her kayıt **farklı bir akciğer bölgesini** dinler
- Eğer konsolidasyon sol alt lobdaysa → Sağ üst lob kaydı NORMAL olur!

**Örnek Senaryo:**
```
Severe Pneumonia Hastası:
  └─ Sol alt lob: Konsolidasyon + Crackles
  └─ Sol üst lob: NORMAL
  └─ Sağ alt lob: NORMAL  
  └─ Sağ üst lob: NORMAL

Sonuç: 4 kayıttan 3'ü normal → %75 Normal!
```

#### 2. **Temporal (Zamansal) Değişkenlik** ⏰

Solunum sesleri sürekli değil, **intermittent** (aralıklı):
- Crackle'lar genellikle **inspirasyon sonunda** duyulur
- Wheeze'ler **ekspirasyon sırasında** belirginleşir
- 2 saniyelik bir kayıtta **sadece 0.5 saniye patolojik** olabilir

**Event Annotation Mantığı:**
- Eğer 2 saniyelik kayıtta **sürekli crackle yoksa** → "Normal" olarak etiketlenmiş olabilir
- Sadece **dominant/belirgin** sesler event olarak işaretlenir

#### 3. **Hastalık Evresi ve Şiddeti** 📊

"Severe" şiddeti **klinik durum** ile ilgili:
- Ateş, oksijen satürasyonu, solunum sıkıntısı
- Mutlaka **tüm akciğer alanlarında patolojik ses** anlamına gelmez

**Örnekler:**
- **Erken evre severe pneumonia**: Radyolojik bulgular var ama akustik bulgular henüz gelişmemiş
- **Tedavi yanıtı sonrası**: Klinik ciddi ama akciğer sesleri düzelmeye başlamış
- **Lokalize ciddi pnömoni**: Küçük ama kritik bölgede (örn: sağ orta lob)

#### 4. **Veri Toplama ve Etiketleme Metodolojisi** 📝

SPRSound veri seti özellikleri:
- **Multiple recordings per patient**: Bir hastadan 10-20+ kayıt alınmış olabilir
- **Different time points**: Farklı zamanlarda (yatış, tedavi sırası, taburcu)
- **Different locations**: Farklı göğüs bölgeleri

**Sonuç:**
```
Severe Pneumonia Hastası (toplam 100 event):
  • 20 event: Hastalık bölgesi (crackles, wheeze)
  • 80 event: Sağlam bölgeler + farklı zaman noktaları (normal)
```

#### 5. **Severity vs Extent (Şiddet vs Yayılım)** 🔍

```
Severe ≠ Yaygın (Extensive)

Severe Pneumonia:
  ✓ Klinik olarak ciddi (sepsis, solunum yetmezliği)
  ✗ Mutlaka tüm akciğerleri tutmuyor

Yaygın Pneumonia:
  ✓ Bilateral, multilober tutulum
  ✗ Klinik olarak daha hafif olabilir
```

---

## 🔬 Hastalık Bazında Detaylı Analizler

### 1. **Pneumonia (Severe) - Detayllı Profil**

**Toplam Event**: 1,271

**Akustik Profil:**
- Normal: **74.9%** (952 event) ← En yüksek
- Wheeze: **11.5%** (146 event) ← İkinci en yüksek
- Fine Crackle: **9.2%** (117 event)
- Coarse Crackle: **2.5%** (32 event)

**Neden bu dağılım?**
1. **Fokal tutulum**: Pneumonia genellikle 1-2 lob tutar
2. **4 farklı lokasyon**: p1, p2, p3, p4 → En az 2-3'ü sağlam
3. **Wheeze varlığı (11.5%)**: Ciddi inflamasyon → Bronkospazm
4. **İki tip crackle**: Fine (alveolar) + Coarse (sekresyon)

**Severe vs Non-Severe Karşılaştırması:**

| Özellik | Severe | Non-Severe | Fark |
|---------|--------|------------|------|
| Normal | 74.9% | 79.7% | -4.8% |
| Wheeze | 11.5% | 5.1% | **+6.4%** ⬆️ |
| Fine Crackle | 9.2% | 12.1% | -2.9% |
| Coarse Crackle | 2.5% | 0.5% | **+2.0%** ⬆️ |

**Sonuç**: Severe pneumonia'da daha **az normal**, daha **fazla wheeze** ve **coarse crackle** var - bu mantıklı!

---

### 2. **Bronchiolitis - Dikkat Çekici Bulgular** 🎯

**Toplam Event**: 305

**Akustik Profil:**
- **Wheeze: 43.0%** ← En yüksek wheeze oranı!
- Normal: 51.5%
- Fine Crackle: 5.6%

**Neden %43 Wheeze?**

Bronchiolitis patofizyolojisi:
1. **Küçük hava yolları inflamasyonu** → Dar lümen
2. **Mukus sekresyonu** → Hava akımı kısıtlanması
3. **Bronkospazm** → Yüksek hızlı hava akımı
4. **Ekspiratuvar wheeze** → Karakteristik bulgu

**Normal %51.5 neden var?**
- Hastalık akut fazda **ilerleyici**
- Bazı kayıtlar erken evre veya iyileşme döneminde
- Tüm bronşiyoller aynı anda tutulmaz

---

### 3. **Control Group - Beklenen Sonuç** ✅

**Toplam Event**: 2,044

**Akustik Profil:**
- Normal: **94.5%** ← Mükemmel!
- Rhonchi: 1.7%
- Stridor: 1.1%
- No Event: 1.3%

**Neden %100 değil?**

1. **Normal varyasyonlar**:
   - Fizyolojik bronkial sesler
   - Transmisyon artefaktları
   - Kalp sesleri interferansı

2. **Subklinik bulgular**:
   - Asemptomatik postnasal akıntı
   - Hafif mukus
   - Geçici bronkospazm

3. **Etiketleme hassasiyeti**:
   - Çok hafif sesler "rhonchi" olarak işaretlenmiş olabilir

---

### 4. **Unknown Category - En Heterojen Grup** ❓

**Toplam Event**: 4,813 (2. en büyük grup!)

**Akustik Profil:**
- Normal: 49.2%
- **Fine Crackle: 38.2%** ← Çok yüksek!
- Wheeze+Crackle: 5.4%
- Wheeze: 5.4%

**Neden "Unknown"?**

1. **Tanı konulmamış hastalar**
2. **Multiple tanılar** (kategorize edilememiş)
3. **Nadir hastalıklar**
4. **Incomplete medical records**

**Fine Crackle %38.2 ne anlama gelir?**
- Bu hastalar muhtemelen **parenkimal hastalık** var
- İnterstisyel akciğer hastalığı, pulmoner ödem, atipik enfeksiyonlar
- Tanı için ek testler gerekiyor

---

### 5. **Kawasaki Disease - İlginç Durum** 🔥

**Toplam Event**: Sadece **11** (en nadir!)

**Akustik Profil:**
- **Wheeze: 81.8%** ← Ekstrem!
- Normal: 18.2%

**Neden bu kadar wheeze?**

Kawasaki Disease:
- **Vaskülit** → Damar iltihabı
- **Sistemik inflamasyon** → Hava yolları da etkilenir
- **Bronkial hiperreaktivite** → Wheeze
- Asthma-like semptomlar

**Ancak dikkat:**
- Sadece 11 event → İstatistiksel olarak az güvenilir
- 1 hastadan birden fazla kayıt alınmış olabilir

---

## 📊 Event Type Bazlı Analizler

### Normal Event Analizi

**En Yüksek Normal Oranı:**
1. Airway foreign body: **100%** (38 event)
2. Protracted bacterial bronchitis: **100%** (21 event)
3. Chronic cough: **96.4%** (53 event)

**Neden yabancı cisimde %100 normal?**
- Yabancı cisim **lokalize obstrüksiyon**
- Etkilenen bölge tek bir bronş
- Diğer tüm akciğer alanları sağlam
- Akustik bulgu oluşmamış olabilir

---

### Fine Crackle Analizi

**En Yüksek Fine Crackle Oranı:**
1. **Unknown: 38.2%** (1,838 event)
2. Bronchiectasia: **13.0%** (41 event)
3. Pneumonia (non-severe): **12.1%** (1,341 event)

**Fine Crackle karakteristiği:**
- **Alveolar düzeyde** patoloji
- Pnömoni, interstisyel hastalıklar
- İnspirasyon sonu en belirgin

---

### Wheeze Analizi

**En Yüksek Wheeze Oranı:**
1. **Kawasaki: 81.8%** (9 event)
2. **Bronchiolitis: 43.0%** (131 event)
3. Severe Pneumonia: **11.5%** (146 event)
4. Asthma: **10.2%** (162 event)

**Asthma'da neden sadece %10 wheeze?**
- Asthma **atak dışında** normal olabilir
- Kayıtlar muhtemelen stabil dönemde alınmış
- Kontrollü asthma → Minimal semptom

---

## 💡 Machine Learning İçin Çıkarımlar

### 1. **Class Imbalance Stratejileri**

```
Severe Imbalance:
  • Pneumonia (non-severe): 11,067 event (44.6%)
  • Kawasaki: 11 event (0.04%)
  
  Oran: 1,006:1 !!
```

**Çözümler:**
- SMOTE veya data augmentation
- Class weights kullan
- Focal Loss
- Undersampling majority class

### 2. **Feature Engineering Önerileri**

**Event Distribution Features:**
```python
# Her hasta için:
features = {
    'wheeze_ratio': count_wheeze / total_events,
    'crackle_ratio': count_crackle / total_events,
    'normal_ratio': count_normal / total_events,
    'diversity_score': entropy(event_distribution)
}
```

**Spatial Features:**
```python
# Lokasyon bazlı:
features = {
    'affected_locations': count_pathological_locations,
    'bilateral': is_both_sides_affected,
    'upper_vs_lower': ratio_upper_lower_events
}
```

### 3. **Model Architecture Önerileri**

#### Option A: Multi-Task Learning
```
Input: Audio
   ↓
HeAR Encoder
   ↓
   ├─→ Event Type Classifier (7 classes)
   └─→ Disease Classifier (16 classes)
```

**Avantaj**: Event type auxiliary task olarak hastalık öğrenmeye yardımcı olur

#### Option B: Hierarchical Classification
```
Level 1: Healthy (Control) vs Pathological
   ↓
Level 2: Disease Groups
   ├─→ Pneumonia Group (severe, non-severe)
   ├─→ Asthma/Wheeze Group (asthma, bronchiolitis, kawasaki)
   ├─→ Infection Group (bronchitis, URI)
   └─→ Other
```

**Avantaj**: Class imbalance azalır, daha kolay öğrenme

#### Option C: Ensemble of Specialists
```
Specialist 1: Normal vs Abnormal (binary)
Specialist 2: Wheeze presence predictor
Specialist 3: Crackle presence predictor
Specialist 4: Final disease classifier (uses all outputs)
```

### 4. **Grouped Disease Classification** (Recommended)

16 hastalığı **5-6 ana gruba** indir:

```
Group 1: Pneumonia (severe + non-severe) → 12,338 events
Group 2: Bronchitis/Infections → 2,558 events  
Group 3: Asthma/Wheeze disorders → 1,908 events
Group 4: Control/Healthy → 2,044 events
Group 5: Other/Unknown → 5,224 events
```

**Avantaj**: 
- Daha dengeli sınıflar
- Klinik olarak anlamlı gruplar
- Daha iyi performans

---

## 🎯 Sonuç ve Öneriler

### Ana Bulgular:

1. **"Normal" dominansı doğal ve beklenen**
   - Fokal hastalıklar → Sağlam bölgeler var
   - Multiple lokasyonlar → Çoğunluk sağlam
   - %70-95 normal oran NORMAL!

2. **Bazı hastalıklar karakteristik akustik profil gösteriyor**
   - Bronchiolitis → Wheeze (43%)
   - Unknown → Fine Crackle (38%)
   - Severe Pneumonia → More adventitious sounds

3. **Class imbalance çok ciddi**
   - Top 3 hastalık = %70 of all data
   - Bottom 5 hastalık = %1.5 of all data

### İkinci Model İçin En İyi Yaklaşım:

**ÖNERİ: Hierarchical Multi-Task Learning**

```
Stage 1: Event Type Classification (7 classes)
   ↓ (use embeddings)
Stage 2: Disease Group Classification (5-6 groups)
   ↓ (use both audio + event predictions)
Stage 3: Fine-grained Disease (optional, for large groups)
```

**Neden?**
- Event type öğrenme hastalık için feature sağlar
- Grouped classification imbalance'ı azaltır
- Hierarchical yapı klinik mantığa uygun
- Her stage ayrı optimize edilebilir

---

## 📚 Referanslar

- SPRSound Dataset: Respiratory sound database with disease labels
- Event annotations: Manual annotation by medical experts
- Clinical interpretation: Based on respiratory physiology

---

**Rapor Tarihi**: 14 Ocak 2026  
**Analiz Eden**: HeAR Training Pipeline  
**Veri Seti**: SPRSound Event Level Dataset (24,808 events)

---

---

---

---

---

---

---

---

## Tablo 1. Veri Seti Özellikleri (Train vs Test)

Train ve test gruplarının aynı popülasyondan geldiğini göstermek için karşılaştırma. Stratifiye hasta bazlı bölme (train %80 / test %20). p < 0.05 anlamlı fark gösterir.

| Değişken | Train | Test | p değeri |
|----------|-------|------|----------|
| Toplam hasta sayısı | 1,321 | 331 | — |
| Cinsiyet, n (%) | Erkek: 666 (50.4%), Kız: 655 (49.6%) | Erkek: 184 (55.6%), Kız: 147 (44.4%) | 0.105 |
| Yaş (yıl), medyan (IQR) | 5.0 (3.4–7.4) | 4.5 (3.3–7.0) | 0.066 |
| Hastalık tanı dağılımı (hasta bazında) |  |  | 1.000 |
|   Pnömoni | 723 (54.7%) | 181 (54.7%) | — |
|   Bronkokonstriksiyon | 287 (21.7%) | 72 (21.8%) | — |
|   Normal | 127 (9.6%) | 32 (9.7%) | — |
|   Diğer | 184 (13.9%) | 46 (13.9%) | — |
| Toplam olay sayısı | 20,567 | 4,241 | — |
| Olay tipi dağılımı (alt gruplar: Normal, Crackles, Rhonchi) |  |  | 0.181 |
|   Normal | 15623 (76.0%) | 3149 (74.3%) | — |
|   Fine Crackle | 2909 (14.1%) | 621 (14.6%) | — |
|   Coarse Crackle | 161 (0.8%) | 16 (0.4%) | — |
|   Wheeze | 1254 (6.1%) | 251 (5.9%) | — |
|   Wheeze+Crackle | 229 (1.1%) | 74 (1.7%) | — |
|   Rhonchi | 201 (1.0%) | 16 (0.4%) | — |
|   Stridor | 18 (0.1%) | 56 (1.3%) | — |
|   No Event | 172 (0.8%) | 58 (1.4%) | — |
| Sound pattern dağılımı |  |  | 0.181 |
|   Normal | 15623 (76.0%) | 3149 (74.3%) | — |
|   Crackles | 3299 (16.0%) | 711 (16.8%) | — |
|   Rhonchi | 1473 (7.2%) | 323 (7.6%) | — |
| Kayıt lokasyonu dağılımı (p1–p4) |  |  | 0.307 |
|   p1 | 5162 (25.1%) | 1081 (25.5%) | — |
|   p2 | 5400 (26.3%) | 1155 (27.2%) | — |
|   p3 | 4814 (23.4%) | 951 (22.4%) | — |
|   p4 | 5030 (24.5%) | 1008 (23.8%) | — |

**Notlar:**
- Toplam hasta: Train 1321 + Test 331 = 1652 (stratifiye hasta bazlı bölme, örtüşme yok)
- Toplam olay: Train 20,567 + Test 4,241 = 24,808
- Yaş: Medyan (IQR); Mann-Whitney U veya t-test
- Kategorik değişkenler: Ki-kare testi


## Tablo 2. Hastalık Tanı Dağılımı — Tüm Hastalıklar (Hasta Bazında)

17 hastalık sınıfı, Train vs Test. p = 0.997 (anlamlı değil).

| Hastalık | Train n (%) | Test n (%) |
|----------|-------------|------------|
| Acute upper respiratory infection | 20 (1.5%) | 3 (0.9%) |
| Airway foreign body | 3 (0.2%) | 0 (0.0%) |
| Asthma | 111 (8.4%) | 25 (7.6%) |
| Bronchiectasia | 9 (0.7%) | 1 (0.3%) |
| Bronchiolitis | 8 (0.6%) | 3 (0.9%) |
| Bronchitis | 168 (12.7%) | 44 (13.3%) |
| Chronic cough | 3 (0.2%) | 1 (0.3%) |
| Control Group | 127 (9.6%) | 32 (9.7%) |
| Hemoptysis | 3 (0.2%) | 1 (0.3%) |
| Kawasaki disease | 1 (0.1%) | 0 (0.0%) |
| Other respiratory diseases | 46 (3.5%) | 13 (3.9%) |
| Pneumonia (non-severe) | 657 (49.7%) | 164 (49.5%) |
| Pneumonia (severe) | 66 (5.0%) | 17 (5.1%) |
| Protracted bacterial bronchitis | 2 (0.2%) | 0 (0.0%) |
| Pulmonary hemosiderosis | 2 (0.2%) | 1 (0.3%) |
| Unknown | 95 (7.2%) | 26 (7.9%) |
