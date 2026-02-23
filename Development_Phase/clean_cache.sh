#!/bin/bash

# Cache temizleme scripti
# Güvenli ve dikkatli temizlenmesi gereken cache'leri ayırır

echo "=== Cache Analizi ==="
echo ""

# Sistem cache'leri
echo "Sistem Cache'leri (~/.cache):"
du -h --max-depth=1 ~/.cache 2>/dev/null | sort -hr | head -10
echo ""

# Python cache'leri
echo "Python __pycache__ Klasörleri:"
cd /home/izzet/Desktop/PulmoVec
find . -type d -name "__pycache__" -exec du -sh {} \; 2>/dev/null | awk '{sum+=$1} END {print "Toplam: " sum}'
find . -type d -name "__pycache__" -exec du -sh {} \; 2>/dev/null | sort -hr
echo ""

# Proje özel cache'leri
echo "Proje Özel Cache'leri:"
if [ -d "EventDetect/models/hear_embeddings_cache" ]; then
    du -sh EventDetect/models/hear_embeddings_cache
fi
echo ""

echo "=== Öneriler ==="
echo ""
echo "GÜVENLE TEMİZLENEBİLİR:"
echo "1. pip cache (~12GB) - pip cache purge"
echo "2. uv cache (~7.4GB) - uv cache clean"
echo "3. Chrome cache (~1.7GB) - Tarayıcıdan temizlenebilir"
echo "4. __pycache__ klasörleri (~600KB) - find . -type d -name '__pycache__' -exec rm -r {} +"
echo ""
echo "DİKKATLİ TEMİZLENMELİ:"
echo "1. HuggingFace cache (~11GB) - Model dosyaları, tekrar indirilmesi gerekebilir"
echo "2. fairseq2 cache (~11GB) - Model dosyaları, tekrar indirilmesi gerekebilir"
echo "3. hear_embeddings_cache (~486MB) - Proje cache'i, temizlenirse tekrar hesaplanır"
echo ""
echo "=== Temizleme Komutları ==="
echo ""
echo "# Python paket cache'lerini temizle:"
echo "pip cache purge"
echo "uv cache clean"
echo ""
echo "# __pycache__ klasörlerini temizle:"
echo "find /home/izzet/Desktop/PulmoVec -type d -name '__pycache__' -exec rm -r {} +"
echo ""
echo "# HuggingFace cache'i temizle (DİKKATLİ!):"
echo "rm -rf ~/.cache/huggingface"
echo ""
echo "# fairseq2 cache'i temizle (DİKKATLİ!):"
echo "rm -rf ~/.cache/fairseq2"
echo ""
echo "# hear_embeddings_cache temizle (Proje cache'i):"
echo "rm -rf /home/izzet/Desktop/PulmoVec/EventDetect/models/hear_embeddings_cache"
echo ""
