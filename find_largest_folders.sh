#!/bin/bash

# En büyük klasörleri bulan script

echo "=== PulmoVec Projesi - En Büyük Klasörler ==="
echo ""
du -h --max-depth=1 . 2>/dev/null | sort -hr | head -15
echo ""

echo "=== Sistem Genelinde En Büyük Klasörler (Home Dizini) ==="
echo ""
du -h --max-depth=1 ~ 2>/dev/null | sort -hr | head -15
echo ""

echo "=== En Büyük 10 Klasör (Tüm Sistem) ==="
echo ""
# Root dizininden başlayarak en büyük klasörleri bul (sadece okunabilir olanlar)
sudo du -h --max-depth=1 / 2>/dev/null | sort -hr | head -10 2>/dev/null || echo "Root erişimi gerekli, sadece home dizini gösteriliyor..."
echo ""

echo "=== Detaylı Analiz: models Klasörü ==="
echo ""
if [ -d "models" ]; then
    echo "models klasörü içeriği:"
    du -h --max-depth=2 models 2>/dev/null | sort -hr | head -10
else
    echo "models klasörü bulunamadı"
fi
