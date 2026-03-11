#!/bin/bash

# Usage check
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_bodies> <num_iterations>"
    exit 1
fi

NUM_BODIES=$1
NUM_ITERS=$2
EXECUTABLE=./simulation

# Output file
OUT_FILE="cpu_energy_results.csv"

# Create CSV header if file doesn't exist
if [ ! -f "$OUT_FILE" ]; then
    echo "num_bodies,num_iterations,energy_joules,time_seconds" > "$OUT_FILE"
fi

echo "Running N-body with:"
echo "  Bodies: $NUM_BODIES"
echo "  Iterations: $NUM_ITERS"

# Run and measure energy
RESULT=$(perf stat -x, \
    -e power/energy-pkg/ \
    $EXECUTABLE $NUM_BODIES $NUM_ITERS 2>&1)

# Extract energy (Joules)
ENERGY=$(echo "$RESULT" | grep "energy-pkg" | awk -F, '{print $1}')

# Extract elapsed time
TIME=$(echo "$RESULT" | grep "seconds time elapsed" | awk -F, '{print $1}')

# Save results
echo "$NUM_BODIES,$NUM_ITERS,$ENERGY,$TIME" >> "$OUT_FILE"

echo "Done."
echo "CPU Energy (J): $ENERGY"
echo "Time (s): $TIME"

