"""
Visualize first 10 samples from the dataset.
Shows waveform and spectrogram with event annotations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
from typing import List, Tuple

# Add EventDetect to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

EVENT_DETECT_DIR = PROJECT_ROOT / "EventDetect"
CSV_PATH = PROJECT_ROOT / "data" / "SPRSound_Event_Level_Dataset_CLEAN.csv"
OUTPUT_DIR = EVENT_DETECT_DIR / "EDA" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_audio_and_annotations(wav_path: str, json_path_from_csv: str = None) -> Tuple[np.ndarray, int, List[Tuple[int, int, str]]]:
    """
    Load audio file and corresponding JSON annotations.
    
    Args:
        wav_path: Path to WAV file
        json_path_from_csv: Optional JSON path from CSV file_path column
    
    Returns:
        audio: Audio signal
        sr: Sample rate
        events: List of (start_ms, end_ms, event_type) tuples (in milliseconds)
    """
    # Load audio
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Find JSON file
    wav_path_obj = Path(wav_path)
    json_path = None
    
    # First try path from CSV if provided
    if json_path_from_csv and Path(json_path_from_csv).exists():
        json_path = Path(json_path_from_csv)
    else:
        # Try different patterns
        patterns = [
            wav_path_obj.with_suffix('.json'),
            wav_path_obj.parent.parent / (wav_path_obj.parent.name.replace('_wav', '_json')) / wav_path_obj.name.replace('.wav', '.json'),
            wav_path_obj.parent / wav_path_obj.name.replace('.wav', '.json'),
            # Try replacing train_classification_wav with train_classification_json
            Path(str(wav_path_obj).replace('train_classification_wav', 'train_classification_json').replace('.wav', '.json')),
            Path(str(wav_path_obj).replace('valid_classification_wav', 'valid_classification_json').replace('.wav', '.json')),
            Path(str(wav_path_obj).replace('test_classification_wav', 'test_classification_json').replace('.wav', '.json')),
        ]
        
        for pattern in patterns:
            if pattern.exists():
                json_path = pattern
                break
    
    events = []
    if json_path and json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            event_annotations = data.get('event_annotation', [])
            for event in event_annotations:
                start_ms = int(float(event.get('start', 0)))
                end_ms = int(float(event.get('end', 0)))
                event_type = event.get('type', 'Normal')
                
                # Keep in milliseconds, include all events (including Normal)
                events.append((start_ms, end_ms, event_type))
        except Exception as e:
            print(f"Warning: Could not load JSON {json_path}: {e}")
    
    return audio, sr, events


def add_event_spans(ax, events_ms: List[Tuple[int, int, str]], duration_sec: float, 
                   color: str, alpha: float = 0.25, label: str = None, zorder: int = 0):
    """
    Add event spans to axis.
    
    Args:
        ax: Matplotlib axis
        events_ms: List of (start_ms, end_ms, event_type) tuples
        duration_sec: Audio duration in seconds
        color: Color for spans
        alpha: Transparency
        label: Label for legend
        zorder: Z-order for drawing
    """
    first = True
    for start_ms, end_ms, event_type in events_ms:
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        
        # Clamp to valid range
        start_sec = max(0.0, min(start_sec, duration_sec))
        end_sec = max(0.0, min(end_sec, duration_sec))
        
        if end_sec > start_sec:
            ax.axvspan(start_sec, end_sec, facecolor=color, alpha=alpha, 
                      edgecolor=color, linewidth=0.5, zorder=zorder,
                      label=(label if first and label else None))
            
            # Add text annotation with ms values
            mid_time = (start_sec + end_sec) / 2
            text = f"{event_type}\n{start_ms}-{end_ms}ms"
            ax.text(mid_time, ax.get_ylim()[1] * 0.95, text,
                   ha='center', va='top', fontsize=7, color='black',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            alpha=0.8, edgecolor=color, linewidth=1),
                   zorder=zorder+1)
            first = False


def plot_sample(audio: np.ndarray, sr: int, events_ms: List[Tuple[int, int, str]], 
                wav_path: str, output_path: Path):
    """
    Plot waveform and spectrogram with event annotations (aligned timelines).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        events_ms: List of (start_ms, end_ms, event_type) tuples (in milliseconds)
        wav_path: Path to audio file
        output_path: Path to save plot
    """
    duration_sec = len(audio) / sr
    t = np.arange(len(audio)) / sr
    
    # Create figure with 2 subplots, share x-axis for alignment
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Audio Analysis: {Path(wav_path).name}', fontsize=16, fontweight='bold')
    
    # Convert events to seconds for plotting
    events_sec = [(start_ms/1000.0, end_ms/1000.0, event_type) 
                  for start_ms, end_ms, event_type in events_ms]
    # Filter events within duration
    events_sec = [(max(0.0, s), min(duration_sec, e), et) 
                  for s, e, et in events_sec if min(duration_sec, e) > max(0.0, s)]
    
    # Subplot 1: Waveform
    ax1 = axes[0]
    ax1.plot(t, audio, linewidth=0.6, color='black', alpha=0.8)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'Waveform with Event Annotations (sr={sr} Hz, dur={duration_sec:.2f}s)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration_sec)
    # Don't show xlabel on top plot (will be on bottom plot)
    ax1.set_xlabel('')
    
    # Add event annotations to waveform
    event_colors = {
        'Normal': '#808080',  # Gray for Normal
        'Crackles': '#2ca02c',  # Green
        'Wheeze': '#1f77b4',  # Blue
        'Rhonchi': '#ff7f0e',  # Orange
        'Fine Crackle': '#d62728',  # Red
        'Coarse Crackle': '#9467bd',  # Purple
        'Wheeze+Crackle': '#8c564b',  # Brown
        'Stridor': '#e377c2'  # Pink
    }
    
    # Group events by type for legend
    event_types_seen = set()
    for start_sec, end_sec, event_type in events_sec:
        color = event_colors.get(event_type, '#2ca02c')
        label = event_type if event_type not in event_types_seen else None
        if label:
            event_types_seen.add(event_type)
        
        ax1.axvspan(start_sec, end_sec, facecolor=color, alpha=0.25, 
                   edgecolor=color, linewidth=0.5, zorder=0, label=label)
        
        # Add text annotation with ms values
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        mid_time = (start_sec + end_sec) / 2
        text = f"{event_type}\n{start_ms}-{end_ms}ms"
        ax1.text(mid_time, ax1.get_ylim()[1] * 0.9, text,
                ha='center', va='top', fontsize=7, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         alpha=0.8, edgecolor=color, linewidth=1),
                zorder=1)
    
    if event_types_seen:
        ax1.legend(loc='upper right', frameon=True, fontsize=8)
    
    # Subplot 2: Spectrogram (using specgram for proper time alignment)
    ax2 = axes[1]
    
    # Use specgram for proper time alignment
    Pxx, freqs, bins, im = ax2.specgram(
        audio, 
        NFFT=1024, 
        Fs=sr, 
        noverlap=768, 
        cmap='magma'
    )
    ax2.set_ylim(0, 2000)  # Focus on 0-2000 Hz
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_title('Spectrogram (0-2000 Hz) with Event Annotations', fontsize=14)
    
    # Add colorbar (positioned outside)
    cbar = plt.colorbar(im, ax=ax2, label='Power (dB)', pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    
    # Add event annotations to spectrogram (same as waveform)
    for start_sec, end_sec, event_type in events_sec:
        color = event_colors.get(event_type, '#2ca02c')
        ax2.axvspan(start_sec, end_sec, facecolor=color, alpha=0.22, 
                   edgecolor=color, linewidth=0.5, zorder=3)
        
        # Add text annotation
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        mid_time = (start_sec + end_sec) / 2
        text = f"{event_type}\n{start_ms}-{end_ms}ms"
        ax2.text(mid_time, ax2.get_ylim()[1] * 0.95, text,
                ha='center', va='top', fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, 
                         alpha=0.7, edgecolor='white', linewidth=1),
                zorder=4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def main():
    """Main function to visualize first 10 samples."""
    print("=" * 80)
    print("Data Visualization: First 10 Samples")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df)}")
    
    # Get unique wav files with events (non-Normal)
    # First, find files that have at least one non-Normal event
    files_with_events = df[df['event_type'] != 'Normal']['wav_path'].unique()
    
    if len(files_with_events) > 0:
        # Get first 10 files with events
        selected_files = files_with_events[:10]
        print(f"\nFound {len(files_with_events)} files with events")
        print(f"Visualizing first {len(selected_files)} files with events...")
        
        # Get rows for selected files
        unique_files = df[df['wav_path'].isin(selected_files)].groupby('wav_path').first().reset_index()
    else:
        # Fallback: use first 10 unique files
        print(f"\nNo files with non-Normal events found, using first 10 files...")
        unique_files = df.groupby('wav_path').first().reset_index().head(10)
    
    # Process each file
    for idx, row in unique_files.iterrows():
        wav_path_str = str(row['wav_path'])
        json_path_from_csv = str(row.get('file_path', ''))  # This is the JSON path in CSV
        
        if not Path(wav_path_str).exists():
            print(f"\n[{idx+1}/10] Skipping (file not found): {wav_path_str}")
            continue
        
        print(f"\n[{idx+1}/10] Processing: {Path(wav_path_str).name}")
        if json_path_from_csv and json_path_from_csv != 'nan':
            print(f"  JSON path from CSV: {json_path_from_csv}")
        
        try:
            # Load audio and annotations
            audio, sr, events = load_audio_and_annotations(wav_path_str, json_path_from_csv if json_path_from_csv != 'nan' else None)
            
            print(f"  Duration: {len(audio)/sr:.2f}s")
            print(f"  Events found: {len(events)}")
            for start_ms, end_ms, etype in events:
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                print(f"    - {etype}: {start_ms}-{end_ms}ms ({start_sec:.3f}s - {end_sec:.3f}s, dur: {end_sec-start_sec:.3f}s)")
            
            # Create output filename
            output_filename = f"sample_{idx:02d}_{Path(wav_path_str).stem}.png"
            output_path = OUTPUT_DIR / output_filename
            
            # Plot
            plot_sample(audio, sr, events, wav_path_str, output_path)
            
        except Exception as e:
            print(f"  ✗ Error processing {wav_path_str}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Visualization completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
