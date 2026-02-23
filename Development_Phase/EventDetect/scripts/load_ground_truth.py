"""Load ground truth event segments from JSON files and CSV (model1_label)."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

# Model 1 class mapping
MODEL1_CLASS_NAMES = ["Normal", "Crackles", "Rhonchi"]


def load_ground_truth_from_json(json_path: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Load ground truth event segments from JSON file.
    
    Args:
        json_path: Path to JSON annotation file
        
    Returns:
        segments: List of (start_sec, end_sec) tuples
        event_types: List of event type strings corresponding to segments
    """
    json_path_obj = Path(json_path)
    
    if not json_path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path_obj, 'r') as f:
        data = json.load(f)
    
    segments = []
    event_types = []
    
    if 'event_annotation' not in data:
        return segments, event_types
    
    for event in data['event_annotation']:
        start_ms = float(event['start'])
        end_ms = float(event['end'])
        event_type = event.get('type', 'Unknown')
        
        # Convert milliseconds to seconds
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        
        segments.append((start_sec, end_sec))
        event_types.append(event_type)
    
    return segments, event_types


def load_ground_truth_model1_labels(csv_path: str, json_path: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Load ground truth from CSV using model1_label (0=Normal, 1=Crackles, 2=Rhonchi).
    
    Args:
        csv_path: Path to CSV file with model1_label column
        json_path: Path to JSON file to get event timings
        
    Returns:
        segments: List of (start_sec, end_sec) tuples
        event_types: List of event type strings (Normal, Crackles, Rhonchi)
    """
    # Load JSON to get event timings
    json_path_obj = Path(json_path)
    if not json_path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path_obj, 'r') as f:
        json_data = json.load(f)
    
    # Load CSV to get model1_label
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Get filename from JSON path to match in CSV
    json_filename = json_path_obj.name
    
    # Load CSV
    df = pd.read_csv(csv_path_obj)
    
    # Filter rows matching this JSON file
    df_file = df[df['filename'] == json_filename].copy()
    
    if len(df_file) == 0:
        print(f"Warning: No matching rows in CSV for {json_filename}")
        return [], []
    
    segments = []
    event_types = []
    
    # Match JSON events with CSV rows by timing
    if 'event_annotation' in json_data:
        for event in json_data['event_annotation']:
            start_ms = float(event['start'])
            end_ms = float(event['end'])
            
            # Find matching row in CSV
            matching_row = df_file[
                (df_file['event_start_ms'] == start_ms) & 
                (df_file['event_end_ms'] == end_ms)
            ]
            
            if len(matching_row) > 0:
                model1_label = int(matching_row.iloc[0]['model1_label'])
                event_type = MODEL1_CLASS_NAMES[model1_label]
                
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                
                segments.append((start_sec, end_sec))
                event_types.append(event_type)
    
    return segments, event_types


def load_ground_truth(json_path: str, csv_path: str = None) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Load ground truth event segments. If csv_path provided, uses model1_label from CSV.
    Otherwise, uses event types from JSON.
    
    Args:
        json_path: Path to JSON annotation file
        csv_path: Optional path to CSV with model1_label column
        
    Returns:
        segments: List of (start_sec, end_sec) tuples
        event_types: List of event type strings
    """
    if csv_path:
        return load_ground_truth_model1_labels(csv_path, json_path)
    else:
        return load_ground_truth_from_json(json_path)


def get_recording_duration(json_path: str, csv_path: str = None) -> float:
    """
    Get total recording duration from JSON file.
    
    Args:
        json_path: Path to JSON annotation file
        csv_path: Optional CSV path (for compatibility, passed to load_ground_truth)
        
    Returns:
        duration_sec: Total duration in seconds
    """
    segments, _ = load_ground_truth(json_path, csv_path=csv_path)
    
    if len(segments) == 0:
        return 0.0
    
    # Find max end time
    max_end = max(seg[1] for seg in segments)
    return max_end


if __name__ == "__main__":
    # Test loading
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "EventDetect"))
    from config import SELECTED_SAMPLES_JSON
    
    with open(SELECTED_SAMPLES_JSON, 'r') as f:
        samples = json.load(f)
    
    for event_group, sample_info in samples.items():
        print(f"\n{event_group}:")
        segments, event_types = load_ground_truth(sample_info['json_path'])
        print(f"  Found {len(segments)} events:")
        for i, (seg, evt_type) in enumerate(zip(segments, event_types)):
            print(f"    {i+1}. {evt_type}: {seg[0]:.2f}s - {seg[1]:.2f}s")
