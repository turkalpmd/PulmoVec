#!/bin/bash

# Güvenli cache temizleme scripti
# Sadece güvenle temizlenebilecek cache'leri temizler

echo "=== Güvenli Cache Temizleme ==="
echo ""

# Önce mevcut durumu göster
echo "Temizleme öncesi durum:"
du -sh ~/.cache/pip ~/.cache/uv 2>/dev/null
echo ""

# pip cache temizle
if [ -d ~/.cache/pip ]; then
    echo "pip cache temizleniyor..."
    pip cache purge 2>/dev/null || echo "pip cache purge komutu çalışmadı, manuel temizleme gerekebilir"
fi

# uv cache temizle
if [ -d ~/.cache/uv ]; then
    echo "uv cache temizleniyor..."
    uv cache clean 2>/dev/null || rm -rf ~/.cache/uv/* 2>/dev/null
fi

# __pycache__ temizle
echo "Python __pycache__ klasörleri temizleniyor..."
cd /home/izzet/Desktop/PulmoVec
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "  ✓ __pycache__ klasörleri temizlendi"

# Son durumu göster
echo ""
echo "Temizleme sonrası durum:"
du -sh ~/.cache/pip ~/.cache/uv 2>/dev/null
echo ""
echo "=== Temizleme Tamamlandı ==="
echo ""
echo "NOT: HuggingFace ve fairseq2 cache'leri korundu (model dosyaları)"
echo "     Bunları temizlemek isterseniz manuel olarak yapabilirsiniz."
