#!/bin/bash

# Disk kullanımını gösteren script
# Toplam kullanılan alanı GB cinsinden gösterir

echo "=== Disk Kullanım Raporu ==="
echo ""

# Tüm dosya sistemlerini göster
df -h

echo ""
echo "=== Özet ==="
echo ""

# Toplam kullanılan alanı GB cinsinden hesapla
TOTAL_USED=$(df -BG --total | tail -1 | awk '{print $3}' | sed 's/G//')
TOTAL_SIZE=$(df -BG --total | tail -1 | awk '{print $2}' | sed 's/G//')
TOTAL_AVAIL=$(df -BG --total | tail -1 | awk '{print $4}' | sed 's/G//')
USAGE_PERCENT=$(df -h --total | tail -1 | awk '{print $5}')

echo "Toplam Disk Boyutu: ${TOTAL_SIZE} GB"
echo "Kullanılan Alan: ${TOTAL_USED} GB"
echo "Kullanılabilir Alan: ${TOTAL_AVAIL} GB"
echo "Kullanım Yüzdesi: ${USAGE_PERCENT}"

echo ""
echo "=== Her Sürücü Detayı ==="
echo ""

# Her sürücü için detaylı bilgi
df -h | grep -E '^/dev/' | while read line; do
    FILESYSTEM=$(echo $line | awk '{print $1}')
    SIZE=$(echo $line | awk '{print $2}')
    USED=$(echo $line | awk '{print $3}')
    AVAIL=$(echo $line | awk '{print $4}')
    PERCENT=$(echo $line | awk '{print $5}')
    MOUNT=$(echo $line | awk '{print $6}')
    
    echo "Sürücü: $FILESYSTEM"
    echo "  Bağlantı Noktası: $MOUNT"
    echo "  Toplam: $SIZE"
    echo "  Kullanılan: $USED"
    echo "  Boş: $AVAIL"
    echo "  Kullanım: $PERCENT"
    echo ""
done
