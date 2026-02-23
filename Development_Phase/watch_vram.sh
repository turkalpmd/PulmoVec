#!/bin/bash
# VRAM usage monitoring script

echo "👁️  Monitoring GPU VRAM (Press Ctrl+C to stop)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

while true; do
    clear
    echo "🖥️  GPU Status - $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits | \
    awk -F', ' '{
        used = $3
        total = $4
        free = $5
        util = $6
        temp = $7
        pct = (used/total)*100
        
        printf "GPU %s: %s\n", $1, $2
        printf "├─ VRAM: %.1f / %.1f GB (%.1f%% used)\n", used/1024, total/1024, pct
        printf "├─ Free: %.1f GB\n", free/1024
        printf "├─ Utilization: %d%%\n", util
        printf "└─ Temperature: %d°C\n", temp
        
        if (pct > 90) {
            printf "\n⚠️  WARNING: VRAM usage > 90%% !\n"
        }
        if (pct > 95) {
            printf "🚨 CRITICAL: VRAM usage > 95%% - OOM risk!\n"
        }
    }'
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Training log (last 3 lines):"
    tail -n 3 ensemble_training_log.txt 2>/dev/null || echo "Log not available yet"
    
    sleep 2
done
