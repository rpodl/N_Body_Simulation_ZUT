#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_bodies> <num_steps>"
    exit 1
fi

NUM_BODIES=$1
NUM_STEPS=$2

SIMULATION="./simGPUOnly"
LOGFILE="gpu_log.csv"
INTERVAL="0.5"
echo "Starting GPU logging to $LOGFILE"
echo "timestamp,power_W,temp_C,gpu_util_percent,mem_used_MiB" > "$LOGFILE"

(
    while true; do
        timestamp=$(date +%s.%N)
        read power temp util mem <<< $(nvidia-smi \
            --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used \
            --format=csv,noheader,nounits \
            | sed 's/,/ /g')

        echo "$timestamp,$power,$temp,$util,$mem" >> "$LOGFILE"
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

echo "Done. Log saved to $LOGFILE"

