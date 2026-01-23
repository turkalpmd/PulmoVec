#!/usr/bin/env python3
"""
CSV'ye doğru WAV yollarını ekle

2022 Validation WAV dosyalarının klasör yapısı farklı olduğu için
doğru yolu hesaplayan bir kolon ekler.
"""

import csv
import os

def get_correct_wav_path(json_path):
    """JSON yolundan doğru WAV yolunu hesapla"""
    
    # Basit dönüşüm
    wav_path = json_path.replace('.json', '.wav').replace('_json', '_wav')
    
    # Eğer dosya varsa, direkt dön
    if os.path.exists(wav_path):
        return wav_path, True
    
    # 2022 Validation özel durumu
    if 'valid_classification_json/2022/' in json_path:
        # inter_test_json veya intra_test_json'ı kaldır
        # Örnek: .../2022/inter_test_json/file.json -> .../2022/file.wav
        parts = json_path.split('/')
        
        # inter_test_json veya intra_test_json'ı bul ve kaldır
        if 'inter_test_json' in parts:
            parts.remove('inter_test_json')
        elif 'intra_test_json' in parts:
            parts.remove('intra_test_json')
        
        # _json -> _wav
        parts = [p.replace('_json', '_wav') for p in parts]
        
        # .json -> .wav
        parts[-1] = parts[-1].replace('.json', '.wav')
        
        corrected_path = '/'.join(parts)
        
        return corrected_path, os.path.exists(corrected_path)
    
    return wav_path, False

def main():
    print("="*80)
    print("CSV'YE DOĞRU WAV YOLLARI EKLENİYOR")
    print("="*80)
    
    input_file = 'SPRSound_Event_Level_Dataset_CLEAN.csv'
    output_file = 'SPRSound_Event_Level_Dataset_CLEAN_with_WAV.csv'
    
    # CSV'yi oku
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"\nOrijinal CSV: {input_file}")
    print(f"Toplam satır: {len(rows):,}")
    
    # Yeni kolon ekle
    new_fieldnames = list(fieldnames) + ['wav_path', 'wav_exists']
    
    # Her satır için WAV yolu hesapla
    found_count = 0
    missing_count = 0
    
    for row in rows:
        json_path = row['file_path']
        wav_path, exists = get_correct_wav_path(json_path)
        
        row['wav_path'] = wav_path
        row['wav_exists'] = 'yes' if exists else 'no'
        
        if exists:
            found_count += 1
        else:
            missing_count += 1
    
    # Yeni CSV'yi yaz
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✅ Yeni CSV oluşturuldu: {output_file}")
    print(f"   Toplam event: {len(rows):,}")
    print(f"   WAV bulundu: {found_count:,} ({found_count/len(rows)*100:.2f}%)")
    print(f"   WAV eksik: {missing_count:,} ({missing_count/len(rows)*100:.2f}%)")
    
    # Dataset bazında istatistik
    from collections import defaultdict
    dataset_stats = defaultdict(lambda: {'total': 0, 'found': 0})
    
    for row in rows:
        dataset = row['dataset']
        dataset_stats[dataset]['total'] += 1
        if row['wav_exists'] == 'yes':
            dataset_stats[dataset]['found'] += 1
    
    print("\n" + "="*80)
    print("DATASET BAZINDA WAV DURUMU")
    print("="*80)
    
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        percentage = (stats['found'] / stats['total'] * 100) if stats['total'] > 0 else 0
        status = "✅" if percentage == 100 else "⚠️" if percentage > 90 else "❌"
        
        print(f"\n{status} {dataset}:")
        print(f"   WAV: {stats['found']}/{stats['total']} ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("YENİ KOLONLAR:")
    print("="*80)
    print("  • wav_path: Doğru WAV dosya yolu")
    print("  • wav_exists: 'yes' veya 'no' (dosya var mı?)")
    
    print("\n" + "="*80)
    print("✅ İŞLEM TAMAMLANDI")
    print("="*80)

if __name__ == '__main__':
    main()
