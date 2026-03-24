#!/bin/bash

# PulmoVec Data Restoration Script
# This script extracts the local backups to their original locations.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

echo "Checking for backup archives..."

# Restore PNGs
if [ -f "results_png_backup.zip" ]; then
    echo "📦 Restoring PNG results from results_png_backup.zip..."
    unzip -o results_png_backup.zip
    echo "✅ PNG files restored."
else
    echo "❌ Error: results_png_backup.zip not found in root."
fi

echo ""

# Restore WAVs
if [ -f "wav_data_backup.zip" ]; then
    echo "📦 Restoring WAV data from wav_data_backup.zip (this may take a moment)..."
    unzip -o wav_data_backup.zip
    echo "✅ WAV files restored."
else
    echo "❌ Error: wav_data_backup.zip not found in root."
fi

echo ""
echo "✨ Data structure successfully restored to original state."
