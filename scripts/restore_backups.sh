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
    echo "⚠️  Warning: results_png_backup.zip not found."
fi

echo ""

# Restore WAVs
if [ -f "wav_data_backup.zip" ]; then
    echo "📦 Restoring WAV data from wav_data_backup.zip (this may take a moment)..."
    unzip -o wav_data_backup.zip
    echo "✅ WAV files restored."
else
    echo "⚠️  Warning: wav_data_backup.zip not found."
fi

echo ""

# Restore Legacy Development Phase
if [ -f "development_phase_legacy.zip" ]; then
    echo "📦 Restoring legacy Development_Phase from development_phase_legacy.zip..."
    unzip -o development_phase_legacy.zip
    echo "✅ Development_Phase restored."
else
    echo "⚠️  Warning: development_phase_legacy.zip not found."
fi

echo ""
echo "✨ Data structure successfully restored to original state."
