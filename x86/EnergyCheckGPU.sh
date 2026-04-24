#!/bin/bash

if [ "$#" -ne 3 ]; then 
    echo "Usage: $0 <num_bodies> <num_steps> <G|GB>" 
    exit 1 
fi

NUM_BODIES=$1
NUM_STEPS=$2
MODE=$3

if [[ "$MODE" != "G" && "$MODE" != "GB" ]]; then 
    echo "Error: Mode must be G (GPU) or GB (GPU Barnes-Hut)" 
    exit 1 
fi

SIMULATION="./simulation"
LOGFILE="gpu_log.csv"
SUMMARYFILE="gpu_summary.txt"
INTERVAL="0.1"

echo "Starting GPU logging to $LOGFILE"
echo "timestamp,power_W,temp_C,gpu_util_percent,mem_used_MiB" > "$LOGFILE"

# Temporary accumulator file (shared between processes)
ACC_FILE=$(mktemp)

(
    prev_time=$(date +%s.%N)
    total_energy=0
    temp_sum=0
    mem_sum=0
    samples=0

    while true; do
        timestamp=$(date +%s.%N)

        read power temp util mem <<< $(nvidia-smi \
            --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used \
            --format=csv,noheader,nounits \
            | sed 's/,/ /g')

        echo "$timestamp,$power,$temp,$util,$mem" >> "$LOGFILE"

        # Time delta
        dt=$(echo "$timestamp - $prev_time" | bc -l)

        # Energy increment (J = W * s)
        energy=$(echo "$power * $dt" | bc -l)
        total_energy=$(echo "$total_energy + $energy" | bc -l)

        temp_sum=$(echo "$temp_sum + $temp" | bc -l)
        mem_sum=$(echo "$mem_sum + $mem" | bc -l)
        samples=$((samples + 1))

        prev_time=$timestamp

        # Write accumulators for parent process
        echo "$total_energy $temp_sum $mem_sum $samples" > "$ACC_FILE"

        sleep "$INTERVAL"
    done
) &
LOGGER_PID=$!

echo "Logger PID = $LOGGER_PID"
echo "Running simulation: $SIMULATION"
echo "---------------------------------------"

$SIMULATION $NUM_BODIES $NUM_STEPS

echo "Simulation finished! Stopping logger..."
kill $LOGGER_PID
wait $LOGGER_PID 2>/dev/null

# Read final accumulated values
read total_energy temp_sum mem_sum samples < "$ACC_FILE"
rm "$ACC_FILE"

mean_temp=$(echo "$temp_sum / $samples" | bc -l)
mean_mem=$(echo "$mem_sum / $samples" | bc -l)

echo "Writing summary to $SUMMARYFILE"
cat << EOF > "$SUMMARYFILE"
GPU Energy & Usage Summary
--------------------------
Total energy consumed (J): $total_energy
Mean temperature (°C):     $mean_temp
Mean memory usage (MiB):   $mean_mem
Samples collected:         $samples
Sampling interval (s):     $INTERVAL
EOF

echo "Done."
echo "Raw log:     $LOGFILE"
echo "Summary log: $SUMMARYFILE"

