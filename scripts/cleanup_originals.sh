#!/bin/bash

# PulmoVec Disk Cleanup Script
# This script removes the original .wav and .png files that are now stored in the .zip backups.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Safety check: Ensure backups exist before deleting originals
if [ ! -f "results_png_backup.zip" ] || [ ! -f "wav_data_backup.zip" ]; then
    echo "❌ Error: Backup zip files not found. Run the backup process first."
    exit 1
fi

echo "⚠️  WARNING: This will delete all original .wav and .png files from the working directory."
echo "They have been previously backed up to results_png_backup.zip and wav_data_backup.zip."
read -p "Are you sure you want to proceed? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo "🧹 Removing original .wav files..."
find . -type f -name "*.wav" -not -path "./.git/*" -delete

echo "🧹 Removing original .png files..."
find . -type f -name "*.png" -not -path "./.git/*" -delete

echo "✅ Cleanup complete. Disk space has been freed."
echo "To restore the files, run: bash scripts/restore_backups.sh"
