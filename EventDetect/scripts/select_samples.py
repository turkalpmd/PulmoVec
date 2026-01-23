"""Select sample audio files for event detection evaluation."""

import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "EventDetect"))
from config import CSV_PATH, SELECTED_SAMPLES_JSON, PROJECT_ROOT


def select_samples():
    """Select one sample per event type (Wheezing, Crackles, Normal)."""
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Filter: WAV file must exist
    df = df[df['wav_exists'] == 'yes'].copy()
    
    # Define event type groups
    event_type_mapping = {
        'wheezing': ['Wheeze'],
        'crackles': ['Fine Crackle', 'Coarse Crackle'],
        'normal': ['Normal']
    }
    
    selected_samples = {}
    
    for event_group, event_types in event_type_mapping.items():
        print(f"\nSelecting sample for {event_group} (event types: {event_types})...")
        
        # Filter by event type
        group_df = df[df['event_type'].isin(event_types)].copy()
        
        if len(group_df) == 0:
            print(f"  ⚠️  No samples found for {event_group}")
            continue
        
        # Group by WAV file (same wav_path)
        wav_groups = group_df.groupby('wav_path')
        
        # Find files with multiple events and good duration
        best_file = None
        best_score = 0
        
        for wav_path, group in wav_groups:
            # Count events in this file
            num_events = len(group)
            
            # Get total duration (max event_end_ms)
            max_duration_ms = group['event_end_ms'].max()
            min_duration_ms = group['event_start_ms'].min()
            total_duration_sec = (max_duration_ms - min_duration_ms) / 1000.0
            
            # Score: prefer files with multiple events and longer duration
            score = num_events * 10 + total_duration_sec
            
            if score > best_score:
                best_score = score
                best_file = wav_path
        
        if best_file is None:
            print(f"  ⚠️  No suitable file found for {event_group}")
            continue
        
        # Get all events for this file
        file_events = group_df[group_df['wav_path'] == best_file].iloc[0]
        
        # Get JSON path
        json_path = file_events['file_path']
        
        # Verify files exist
        wav_path = Path(best_file)
        json_path_obj = Path(json_path)
        
        if not wav_path.exists():
            print(f"  ⚠️  WAV file not found: {wav_path}")
            continue
        if not json_path_obj.exists():
            print(f"  ⚠️  JSON file not found: {json_path_obj}")
            continue
        
        selected_samples[event_group] = {
            'wav_path': str(wav_path),
            'json_path': str(json_path_obj),
            'filename': wav_path.name,
            'patient_number': file_events['patient_number'],
            'num_events': len(group_df[group_df['wav_path'] == best_file]),
            'total_duration_sec': (group_df[group_df['wav_path'] == best_file]['event_end_ms'].max() - 
                                   group_df[group_df['wav_path'] == best_file]['event_start_ms'].min()) / 1000.0,
            'event_types': event_types
        }
        
        print(f"  ✓ Selected: {wav_path.name}")
        print(f"    - Patient: {selected_samples[event_group]['patient_number']}")
        print(f"    - Events: {selected_samples[event_group]['num_events']}")
        print(f"    - Duration: {selected_samples[event_group]['total_duration_sec']:.2f}s")
    
    # Save selected samples
    SELECTED_SAMPLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTED_SAMPLES_JSON, 'w') as f:
        json.dump(selected_samples, f, indent=2)
    
    print(f"\n✓ Saved selected samples to: {SELECTED_SAMPLES_JSON}")
    print(f"  Selected {len(selected_samples)} samples")
    
    return selected_samples


if __name__ == "__main__":
    select_samples()
