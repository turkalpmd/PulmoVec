#!/usr/bin/env python3
"""
SPRSound Event-Level Dataset Generator

Bu script tüm JSON dosyalarındaki event'leri ayrı satırlara dönüştürür.
Her event için hasta bilgileri, hastalık tanısı ve event detayları CSV'ye yazılır.

Kullanım:
    python create_event_level_dataset.py

Çıktı:
    SPRSound_Event_Level_Dataset.csv
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict
import re

def parse_filename(filename):
    """
    Dosya adından bilgileri çıkar
    Format: patient_number_age_gender_location_recording_number.json
    Örnek: 40996284_3.0_0_p1_6664.json
    """
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
        # Bazı dosyalar _ ile başlayabilir (patient number eksik)
        return {
            'patient_number': 'Unknown',
            'age': parts[0] if len(parts) > 0 else 'Unknown',
            'gender': 'Male' if len(parts) > 1 and parts[1] == '0' else 'Female' if len(parts) > 1 and parts[1] == '1' else 'Unknown',
            'gender_code': parts[1] if len(parts) > 1 else 'Unknown',
            'recording_location': parts[2] if len(parts) > 2 else 'Unknown',
            'recording_number': parts[3] if len(parts) > 3 else 'Unknown'
        }

def load_patient_diseases(csv_paths):
    """
    Patient Summary CSV dosyalarından hastalık bilgilerini yükle
    """
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

def process_json_directory(json_dir, patient_diseases, dataset_name):
    """
    Bir JSON dizinindeki tüm dosyaları işle
    """
    events_data = []
    processed_files = 0
    total_events = 0
    
    if not os.path.exists(json_dir):
        print(f"⚠️  Dizin bulunamadı: {json_dir}")
        return events_data
    
    print(f"\n🔍 İşleniyor: {dataset_name}")
    print(f"   Dizin: {json_dir}")
    
    for root, dirs, files in os.walk(json_dir):
        json_files = [f for f in files if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(root, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
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
                
                processed_files += 1
                
            except Exception as e:
                print(f"❌ Hata ({filename}): {e}")
    
    print(f"   ✅ {processed_files} dosya, {total_events} event işlendi")
    return events_data

def main():
    print("="*80)
    print("SPRSound Event-Level Dataset Generator")
    print("="*80)
    
    # Ana dizinleri tanımla
    base_dir = '/home/izzet/Desktop/PulmoVec'
    
    # Patient summary CSV dosyaları
    patient_csv_paths = [
        os.path.join(base_dir, 'SPRSound/Patient Summary/SPRSound_patient_summary.csv'),
        os.path.join(base_dir, 'SPRSound/Patient Summary/Grand_Challenge\'23_patient_summary.csv'),
        os.path.join(base_dir, 'SPRSound/Patient Summary/Grand_Challenge\'24_patient_summary.csv'),
    ]
    
    # Hastalık bilgilerini yükle
    print("\n📋 Hasta bilgileri yükleniyor...")
    patient_diseases = load_patient_diseases(patient_csv_paths)
    
    # İşlenecek JSON dizinleri
    json_directories = [
        # Classification datasets
        (os.path.join(base_dir, 'SPRSound-main/Classification/train_classification_json'), 'Classification-Train'),
        (os.path.join(base_dir, 'SPRSound-main/Classification/valid_classification_json/2022/inter_test_json'), 'Classification-Valid-2022-Inter'),
        (os.path.join(base_dir, 'SPRSound-main/Classification/valid_classification_json/2022/intra_test_json'), 'Classification-Valid-2022-Intra'),
        (os.path.join(base_dir, 'SPRSound-main/Classification/valid_classification_json/2023'), 'Classification-Valid-2023'),
        
        # Detection datasets (varsa)
        (os.path.join(base_dir, 'SPRSound/Detection/train_detection_json'), 'Detection-Train'),
        (os.path.join(base_dir, 'SPRSound/Detection/valid_detection_json'), 'Detection-Valid'),
        (os.path.join(base_dir, 'SPRSound/Detection/test2024_detection_json'), 'Detection-Test-2024'),
        
        # BioCAS datasets (varsa)
        (os.path.join(base_dir, 'SPRSound/BioCAS2022/train2022_json'), 'BioCAS2022-Train'),
        (os.path.join(base_dir, 'SPRSound/BioCAS2022/test2022_json/inter_test_json'), 'BioCAS2022-Test-Inter'),
        (os.path.join(base_dir, 'SPRSound/BioCAS2022/test2022_json/intra_test_json'), 'BioCAS2022-Test-Intra'),
        (os.path.join(base_dir, 'SPRSound/BioCAS2023/test2023_json'), 'BioCAS2023-Test'),
        (os.path.join(base_dir, 'SPRSound/BioCAS2024/test2024_json'), 'BioCAS2024-Test'),
        (os.path.join(base_dir, 'SPRSound/BioCAS2025/test2025_json'), 'BioCAS2025-Test'),
    ]
    
    # Tüm eventleri topla
    all_events = []
    
    for json_dir, dataset_name in json_directories:
        events = process_json_directory(json_dir, patient_diseases, dataset_name)
        all_events.extend(events)
    
    # CSV'ye yaz
    if all_events:
        output_file = os.path.join(base_dir, 'SPRSound_Event_Level_Dataset.csv')
        
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
            'total_events_in_file'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_events)
        
        print(f"✅ CSV oluşturuldu!")
        print(f"   Toplam event sayısı: {len(all_events):,}")
        print(f"   Dosya boyutu: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # İstatistikler
        print("\n" + "="*80)
        print("📊 İSTATİSTİKLER")
        print("="*80)
        
        # Dataset bazında
        from collections import Counter
        dataset_counts = Counter(e['dataset'] for e in all_events)
        print("\n📁 Dataset Bazında Event Sayıları:")
        for dataset, count in sorted(dataset_counts.items()):
            print(f"   {dataset:40s}: {count:6,} events")
        
        # Event type bazında
        event_type_counts = Counter(e['event_type'] for e in all_events if e['event_type'] != 'No Event')
        print("\n🔊 Event Type Dağılımı:")
        total_events = sum(event_type_counts.values())
        for event_type, count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_events * 100) if total_events > 0 else 0
            print(f"   {event_type:30s}: {count:6,} ({percentage:5.2f}%)")
        
        # Hastalık bazında
        disease_counts = Counter(e['disease'] for e in all_events if e['disease'] != 'Unknown')
        print("\n🏥 Hastalık Bazında Event Sayıları (Top 10):")
        for disease, count in disease_counts.most_common(10):
            print(f"   {disease:40s}: {count:6,} events")
        
        # Record annotation bazında
        record_counts = Counter(e['record_annotation'] for e in all_events)
        print("\n📋 Record Annotation Dağılımı:")
        for record_type, count in sorted(record_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_events) * 100) if len(all_events) > 0 else 0
            print(f"   {record_type:30s}: {count:6,} ({percentage:5.2f}%)")
        
        print("\n" + "="*80)
        print("✅ İŞLEM TAMAMLANDI!")
        print("="*80)
        
    else:
        print("\n❌ Hiç event bulunamadı!")

if __name__ == '__main__':
    main()
