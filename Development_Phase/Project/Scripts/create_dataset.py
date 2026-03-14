#!/usr/bin/env python3
"""
SPRSound Event-Level Dataset Generator (TAM VERSİYON)

Bu script:
1. JSON dosyalarını tarayarak event-level CSV oluşturur (duplikasyon önleme ile)
2. WAV dosya yollarını ekler
3. İstatistikler gösterir

Kullanım:
    python create_dataset.py

Çıktı:
    data/SPRSound_Event_Level_Dataset_CLEAN_with_WAV.csv
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================================
# DOSYA ADI PARSING
# ============================================================================

def parse_filename(filename):
    """Dosya adından bilgileri çıkar"""
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    
    if len(parts) >= 5:
        return {
            'patient_number': parts[0],
            'age': parts[1],
            'gender': 'Male' if parts[2] == '0' else 'Female' if parts[2] == '1' else parts[2],
            'gender_code': parts[2],
            'recording_location': parts[3],
            'recording_number': parts[4]
        }
    else:
        return {
            'patient_number': 'Unknown',
            'age': parts[0] if len(parts) > 0 else 'Unknown',
            'gender': 'Male' if len(parts) > 1 and parts[1] == '0' else 'Female' if len(parts) > 1 and parts[1] == '1' else 'Unknown',
            'gender_code': parts[1] if len(parts) > 1 else 'Unknown',
            'recording_location': parts[2] if len(parts) > 2 else 'Unknown',
            'recording_number': parts[3] if len(parts) > 3 else 'Unknown'
        }

# ============================================================================
# HASTA BİLGİLERİ YÜKLEME
# ============================================================================

def load_patient_diseases(csv_paths):
    """Patient Summary CSV dosyalarından hastalık bilgilerini yükle"""
    patient_diseases = {}
    
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV bulunamadı: {csv_path}")
            continue
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_num = row.get('patient_num', '').strip()
                disease = row.get('disease', 'Unknown').strip()
                if patient_num:
                    patient_diseases[patient_num] = disease
    
    print(f"✅ {len(patient_diseases)} hasta bilgisi yüklendi")
    return patient_diseases

# ============================================================================
# KONUM BİLGİLERİ
# ============================================================================

def get_location_name(location_code):
    """Kayıt konumu kodunu açıklayıcı isme çevir"""
    location_map = {
        'p1': 'Left Posterior',
        'p2': 'Left Lateral',
        'p3': 'Right Posterior',
        'p4': 'Right Lateral',
        'p5': 'Additional Location 5',
        'p6': 'Additional Location 6',
        'p7': 'Additional Location 7',
        'p8': 'Additional Location 8'
    }
    return location_map.get(location_code, location_code)

# ============================================================================
# JSON DOSYALARINI İŞLEME
# ============================================================================

def process_json_directory(json_dir, patient_diseases, dataset_name, processed_files):
    """
    Bir JSON dizinindeki tüm dosyaları işle
    processed_files: Daha önce işlenmiş dosya isimlerini içeren set
    """
    events_data = []
    processed_count = 0
    skipped_count = 0
    total_events = 0
    
    if not os.path.exists(json_dir):
        print(f"⚠️  Dizin bulunamadı: {json_dir}")
        return events_data, processed_files
    
    print(f"\n🔍 İşleniyor: {dataset_name}")
    print(f"   Dizin: {json_dir}")
    
    for root, dirs, files in os.walk(json_dir):
        json_files = [f for f in files if f.endswith('.json')]
        
        for filename in json_files:
            # DUPLIKASYON KONTROLÜ
            if filename in processed_files:
                skipped_count += 1
                continue
            
            filepath = os.path.join(root, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Dosyayı işlenmiş olarak işaretle
                processed_files.add(filename)
                
                # Dosya adından bilgileri çıkar
                file_info = parse_filename(filename)
                
                # Record level annotation
                record_annotation = data.get('record_annotation', 'Unknown')
                
                # Hastalık bilgisi
                patient_num = file_info['patient_number']
                disease = patient_diseases.get(patient_num, 'Unknown')
                
                # Event annotations
                event_annotations = data.get('event_annotation', [])
                
                if len(event_annotations) == 0:
                    # Event yok ama yine de kaydı ekle
                    events_data.append({
                        'dataset': dataset_name,
                        'file_path': filepath,
                        'filename': filename,
                        'patient_number': patient_num,
                        'age': file_info['age'],
                        'gender': file_info['gender'],
                        'gender_code': file_info['gender_code'],
                        'recording_location': file_info['recording_location'],
                        'recording_location_name': get_location_name(file_info['recording_location']),
                        'recording_number': file_info['recording_number'],
                        'record_annotation': record_annotation,
                        'disease': disease,
                        'event_start_ms': '',
                        'event_end_ms': '',
                        'event_duration_ms': '',
                        'event_type': 'No Event',
                        'event_index': 0,
                        'total_events_in_file': 0
                    })
                else:
                    # Her event için ayrı satır
                    for idx, event in enumerate(event_annotations, 1):
                        start = event.get('start', '0')
                        end = event.get('end', '0')
                        event_type = event.get('type', 'Unknown')
                        
                        # Duration hesapla
                        try:
                            duration = int(end) - int(start)
                        except:
                            duration = 0
                        
                        events_data.append({
                            'dataset': dataset_name,
                            'file_path': filepath,
                            'filename': filename,
                            'patient_number': patient_num,
                            'age': file_info['age'],
                            'gender': file_info['gender'],
                            'gender_code': file_info['gender_code'],
                            'recording_location': file_info['recording_location'],
                            'recording_location_name': get_location_name(file_info['recording_location']),
                            'recording_number': file_info['recording_number'],
                            'record_annotation': record_annotation,
                            'disease': disease,
                            'event_start_ms': start,
                            'event_end_ms': end,
                            'event_duration_ms': duration,
                            'event_type': event_type,
                            'event_index': idx,
                            'total_events_in_file': len(event_annotations)
                        })
                        total_events += 1
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Hata ({filename}): {e}")
    
    print(f"   ✅ {processed_count} dosya işlendi, {skipped_count} duplikasyon atlandı, {total_events} event")
    return events_data, processed_files

# ============================================================================
# WAV YOLU HESAPLAMA
# ============================================================================

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

def add_wav_paths(events_data):
    """Event verilerine WAV yollarını ekle"""
    print("\n" + "="*80)
    print("WAV YOLLARI EKLENİYOR")
    print("="*80)
    
    found_count = 0
    missing_count = 0
    
    print(f"\n⏳ {len(events_data):,} event için WAV yolları hesaplanıyor...")
    
    for i, event in enumerate(events_data):
        if (i + 1) % 10000 == 0:
            print(f"   İşleniyor: {i+1:,}/{len(events_data):,} event...")
        
        json_path = event['file_path']
        wav_path, exists = get_correct_wav_path(json_path)
        
        event['wav_path'] = wav_path
        event['wav_exists'] = 'yes' if exists else 'no'
        
        if exists:
            found_count += 1
        else:
            missing_count += 1
    
    print(f"\n✅ WAV yolları eklendi!")
    print(f"   WAV bulundu: {found_count:,} ({found_count/len(events_data)*100:.2f}%)")
    print(f"   WAV eksik: {missing_count:,} ({missing_count/len(events_data)*100:.2f}%)")
    
    return events_data, found_count, missing_count

# ============================================================================
# İSTATİSTİKLER
# ============================================================================

def print_statistics(events_data, processed_files, found_count, missing_count):
    """Detaylı istatistikleri yazdır"""
    print("\n" + "="*80)
    print("📊 DETAYLI İSTATİSTİKLER")
    print("="*80)
    
    # Dataset bazında
    dataset_counts = Counter(e['dataset'] for e in events_data)
    print("\n📁 Dataset Bazında Event Sayıları:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"   {dataset:40s}: {count:6,} events")
    
    # Event type bazında
    event_type_counts = Counter(e['event_type'] for e in events_data if e['event_type'] != 'No Event')
    print("\n🔊 Event Type Dağılımı:")
    total_events = sum(event_type_counts.values())
    for event_type, count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_events * 100) if total_events > 0 else 0
        print(f"   {event_type:30s}: {count:6,} ({percentage:5.2f}%)")
    
    # Hastalık bazında
    disease_counts = Counter(e['disease'] for e in events_data if e['disease'] != 'Unknown')
    print("\n🏥 Hastalık Bazında Event Sayıları (Top 10):")
    for disease, count in disease_counts.most_common(10):
        print(f"   {disease:40s}: {count:6,} events")
    
    # Dataset bazında WAV durumu
    dataset_stats = defaultdict(lambda: {'total': 0, 'found': 0})
    for event in events_data:
        dataset = event['dataset']
        dataset_stats[dataset]['total'] += 1
        if event.get('wav_exists') == 'yes':
            dataset_stats[dataset]['found'] += 1
    
    print("\n🎵 Dataset Bazında WAV Durumu:")
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        percentage = (stats['found'] / stats['total'] * 100) if stats['total'] > 0 else 0
        status = "✅" if percentage == 100 else "⚠️" if percentage > 90 else "❌"
        print(f"   {status} {dataset:40s}: {stats['found']:6,}/{stats['total']:6,} ({percentage:5.1f}%)")

# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    print("="*80)
    print("SPRSound Event-Level Dataset Generator")
    print("(Duplikasyon Önleme + WAV Yolları)")
    print("="*80)
    
    # Ana dizinleri tanımla
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Patient summary CSV dosyaları
    patient_csv_paths = [
        os.path.join(base_dir, 'SPRSound/Patient Summary/SPRSound_patient_summary.csv'),
        os.path.join(base_dir, 'SPRSound/Patient Summary/Grand_Challenge\'23_patient_summary.csv'),
        os.path.join(base_dir, 'SPRSound/Patient Summary/Grand_Challenge\'24_patient_summary.csv'),
    ]
    
    # Hastalık bilgilerini yükle
    print("\n📋 Hasta bilgileri yükleniyor...")
    patient_diseases = load_patient_diseases(patient_csv_paths)
    
    # İşlenecek JSON dizinleri (ÖNCELİK SIRASINA GÖRE)
    json_directories = [
        # ÖNCE SPRSound-main Classification (en temiz)
        (os.path.join(base_dir, 'SPRSound-main/Classification/train_classification_json'), 'Classification-Train'),
        (os.path.join(base_dir, 'SPRSound-main/Classification/valid_classification_json/2022/inter_test_json'), 'Classification-Valid-2022-Inter'),
        (os.path.join(base_dir, 'SPRSound-main/Classification/valid_classification_json/2022/intra_test_json'), 'Classification-Valid-2022-Intra'),
        (os.path.join(base_dir, 'SPRSound-main/Classification/valid_classification_json/2023'), 'Classification-Valid-2023'),
        
        # SONRA SPRSound-main Detection
        (os.path.join(base_dir, 'SPRSound-main/Detection/train_detection_json'), 'Detection-Train'),
        (os.path.join(base_dir, 'SPRSound-main/Detection/valid_detection_json'), 'Detection-Valid'),
        (os.path.join(base_dir, 'SPRSound-main/Detection/test2024_detection_json'), 'Detection-Test-2024'),
        
        # SONRA SPRSound'dan eksik olanlar (BioCAS testleri)
        (os.path.join(base_dir, 'SPRSound/BioCAS2023/test2023_json'), 'BioCAS2023-Test'),
        (os.path.join(base_dir, 'SPRSound/BioCAS2024/test2024_json'), 'BioCAS2024-Test'),
        (os.path.join(base_dir, 'SPRSound/BioCAS2025/test2025_json'), 'BioCAS2025-Test'),
    ]
    
    # Tüm eventleri topla
    print("\n" + "="*80)
    print("JSON DOSYALARI İŞLENİYOR")
    print("="*80)
    
    all_events = []
    processed_files = set()  # İşlenmiş dosya isimleri
    
    for json_dir, dataset_name in json_directories:
        events, processed_files = process_json_directory(json_dir, patient_diseases, dataset_name, processed_files)
        all_events.extend(events)
    
    if not all_events:
        print("\n❌ Hiç event bulunamadı!")
        return
    
    # WAV yollarını ekle
    all_events, found_count, missing_count = add_wav_paths(all_events)
    
    # CSV'ye yaz
    output_file = os.path.join(base_dir, 'data', 'SPRSound_Event_Level_Dataset_CLEAN_with_WAV.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\n💾 CSV oluşturuluyor: {output_file}")
    
    fieldnames = [
        'dataset',
        'file_path',
        'filename',
        'patient_number',
        'age',
        'gender',
        'gender_code',
        'recording_location',
        'recording_location_name',
        'recording_number',
        'record_annotation',
        'disease',
        'event_start_ms',
        'event_end_ms',
        'event_duration_ms',
        'event_type',
        'event_index',
        'total_events_in_file',
        'wav_path',
        'wav_exists'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_events)
    
    print(f"✅ CSV oluşturuldu!")
    print(f"   Toplam UNIQUE dosya: {len(processed_files):,}")
    print(f"   Toplam event sayısı: {len(all_events):,}")
    print(f"   Dosya boyutu: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # İstatistikler
    print_statistics(all_events, processed_files, found_count, missing_count)
    
    print("\n" + "="*80)
    print("✅ İŞLEM TAMAMLANDI!")
    print("="*80)
    print(f"\n📄 Çıktı dosyası: {output_file}")

if __name__ == '__main__':
    main()
