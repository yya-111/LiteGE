#!/bin/bash

# Define the base command
BASE_CMD="python -u train.py --num_epoch 1 -bs 3072 "

# Define the output file prefix
OUTPUT_PREFIX="train400neuronsrepeat_samplepc_forcenter_rotateTNet_filter"

# Loop 8 times
for i in {1..8}; do
    echo "Running experiment $i..."
    # Execute the command, appending output to a new file for each run
    # 2>&1 redirects stderr to stdout, then >> redirects both to the file
    $BASE_CMD -neu 400 >> "${OUTPUT_PREFIX}_run.out" 2>&1
    echo "Experiment $i finished."
done

echo "All 8 experiments completed."