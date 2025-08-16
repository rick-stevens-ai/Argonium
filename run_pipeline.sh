#!/bin/bash
# Simple wrapper script to run the dataset analysis pipeline

# Change to the script directory
cd "$(dirname "$0")"

# Run the Python pipeline script with all passed arguments
python3 run_pipeline.py "$@"