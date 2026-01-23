# Model Yeniden Eğitimi Gerekli

## Sorun

Mevcut model eski labeling stratejisi ile eğitilmiş. Eski stratejide:
- Overlap ratio threshold = 0.2 (çok düşük)
- Birçok window yanlış label=1 almış
- Model tüm window'ları event olarak öğrenmiş

Bu yüzden model inference'da tüm window'ları yüksek probability (0.57-0.73) ile tahmin ediyor ve tek bir büyük segment oluşturuyor.

## Çözüm

Yeni labeling stratejisi ile modeli yeniden eğitmek gerekiyor:

### Yeni Labeling Stratejisi

1. **Window center kontrolü**: Window'un merkezi bir event içindeyse → label=1
2. **Overlap ratio**: Overlap yeterince yüksekse (threshold=0.3) → label=1  
3. **Toplam event süresi**: Window içindeki toplam event süresi ≥ 0.1s ise → label=1

Bu strateji daha hassas ve kısa event'leri daha iyi yakalayacak.

### Yeniden Eğitim Adımları

```bash
cd EventDetect

# Embedding cache'i temizlemeye GEREK YOK
# Çünkü sadece labels değişti, embeddings aynı

# Modeli yeni labeling ile eğit
python scripts/train_hear_bilstm.py
```

### Beklenen İyileşme

- Model kısa event'leri daha iyi öğrenecek
- Window'lar arasında daha fazla label=0 olacak
- Inference'da ayrı event segment'leri tahmin edecek
- Post-processing daha iyi çalışacak

### Not

Eğer embedding cache'i temizlemek isterseniz (gerekli değil):
```bash
rm -rf models/hear_embeddings_cache/*
```

Ama bu tüm embeddings'leri yeniden hesaplamak demek, zaman alır.
