#!/bin/bash

# TRACE.sh - Comprehensive Reasoning Trace Analysis Script
# Usage: ./TRACE.sh HR-GOOD-1-MC (without .json extension)
# Assumes input files: HR-GOOD-1-MC.json and HR-GOOD-1-MC-Score.json exist

if [ -z "$1" ]; then
    echo "Usage: $0 <base_name>"
    echo "Example: $0 HR-GOOD-1-MC"
    echo "Assumes files exist: <base_name>.json and <base_name>-Score.json"
    exit 1
fi

base_name=$1
input_file="${base_name}.json"
score_file="${base_name}-Score.json"
output_file="${base_name}-TRACE.json"
output_dir="${base_name}_trace_analysis"
incorrect_file="${base_name}-TRACE-incorrect.json"
whole_trace_output="${base_name}-TRACE-whole.json"

# Check if required input files exist
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found!"
    exit 1
fi

if [ ! -f "$score_file" ]; then
    echo "Error: Score file '$score_file' not found!"
    exit 1
fi

# Create output directory
mkdir -p "$output_dir"

echo "Starting comprehensive reasoning trace analysis..."
echo "Input file: $input_file"
echo "Score file: $score_file"
echo "Output file: $output_file"
echo "Output directory: $output_dir"
echo "=================================="

# Run reasoning_traces_parallel_v6.py with all features enabled
python reasoning_traces_parallel_v6.py "$input_file" \
    --output "$output_file" \
    --model gpt-4.1 \
    --config argo_local.yaml \
    --workers 10 \
    --specialty expert \
    --reasoning-mode detailed \
    --save-interval 5 \
    --whole-trace-analysis \
    --whole-trace-model gpt-4.1 \
    --whole-trace-output "$whole_trace_output" \
    --enhanced-discrepancy \
    --dual-prediction \
    --grading gpt-4.1 \
    --method0-model gpt-4.1 \
    --method1-analysis-model gpt-4.1 \
    --method1-synthesis-model gpt-4.1 \
    --method2-model gpt-4.1 \
    --comparison-model gpt-4.1 \
    --text-analysis-model gpt-4.1 \
    --require-grading-model \
    --verbose-grading \
    --capture-incorrect "$incorrect_file" \
    --argonium-results "$score_file" \
    --split-output \
    --output-dir "$output_dir" \
    --create-stream-files

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "=================================="
    echo "✅ Reasoning trace analysis completed successfully!"
    echo ""
    echo "Generated files:"
    echo "  Main output: $output_file"
    echo "  Whole trace: $whole_trace_output"
    echo "  Incorrect answers: $incorrect_file"
    echo "  Analysis directory: $output_dir/"
    echo ""
    echo "Analysis directory contains:"
    echo "  - Individual JSON files per question"
    echo "  - Stream of thought analysis files"
    echo ""
    ls -la "$output_dir" | head -10
    if [ $(ls -1 "$output_dir" | wc -l) -gt 10 ]; then
        echo "  ... and $(( $(ls -1 "$output_dir" | wc -l) - 10 )) more files"
    fi
else
    echo "❌ Error: Reasoning trace analysis failed!"
    exit 1
fi
