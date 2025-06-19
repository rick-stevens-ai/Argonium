#!/bin/bash

# Model Benchmarking Workflow Script
# Comprehensive evaluation of multiple AI models with detailed analysis

set -e  # Exit on any error

# Default values
QUESTIONS_FILE=""
GRADER_MODEL="gpt41"
SAMPLE_SIZE=50
PARALLEL_WORKERS=10
RANDOM_SEED=""
FORMAT="auto"
OUTPUT_DIR="./benchmark_results"
VERBOSE=false
SAVE_INCORRECT=false
INCLUDE_MODELS=""
EXCLUDE_MODELS=""
SKIP_AVAILABILITY=false
TIMEOUT=10

# Function to display usage
usage() {
    echo "Model Benchmarking Workflow"
    echo ""
    echo "Usage: $0 -q QUESTIONS_FILE [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -q, --questions FILE         Questions file for benchmarking (required)"
    echo ""
    echo "Options:"
    echo "  -g, --grader MODEL          Grader model (default: gpt41)"
    echo "  -s, --sample-size NUM       Number of questions to sample (default: 50)"
    echo "  -p, --parallel NUM          Parallel workers (default: 10)"
    echo "  -r, --seed NUM              Random seed for reproducibility"
    echo "  -f, --format FORMAT         Question format: auto|mc|qa (default: auto)"
    echo "  -o, --output-dir DIR        Output directory (default: ./benchmark_results)"
    echo "  --include MODELS            Comma-separated list of models to include"
    echo "  --exclude MODELS            Comma-separated list of models to exclude"
    echo "  --verbose                   Enable verbose output"
    echo "  --save-incorrect            Save incorrect answers for analysis"
    echo "  --skip-availability         Skip model availability checks"
    echo "  --timeout NUM               Availability check timeout (default: 10s)"
    echo "  --quick                     Quick benchmark (sample=20, parallel=5)"
    echo "  --full                      Full benchmark (all questions, max parallel)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -q questions.json --quick --save-incorrect"
    echo "  $0 -q questions.json -s 100 -p 15 --include gpt4,claude3"
    echo "  $0 -q questions.json --full --exclude legacy_model"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--questions)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        -g|--grader)
            GRADER_MODEL="$2"
            shift 2
            ;;
        -s|--sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL_WORKERS="$2"
            shift 2
            ;;
        -r|--seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --include)
            INCLUDE_MODELS="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_MODELS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --save-incorrect)
            SAVE_INCORRECT=true
            shift
            ;;
        --skip-availability)
            SKIP_AVAILABILITY=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --quick)
            SAMPLE_SIZE=20
            PARALLEL_WORKERS=5
            shift
            ;;
        --full)
            SAMPLE_SIZE=0  # Will use all questions
            PARALLEL_WORKERS=20
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$QUESTIONS_FILE" ]]; then
    echo "Error: Questions file is required (-q/--questions)"
    usage
fi

if [[ ! -f "$QUESTIONS_FILE" ]]; then
    echo "Error: Questions file '$QUESTIONS_FILE' not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================="
echo "Model Benchmarking Workflow"
echo "=========================================="
echo "Questions file: $QUESTIONS_FILE"
echo "Grader model: $GRADER_MODEL"
echo "Sample size: $SAMPLE_SIZE"
echo "Parallel workers: $PARALLEL_WORKERS"
echo "Format: $FORMAT"
echo "Output directory: $(pwd)"
if [[ -n "$RANDOM_SEED" ]]; then
    echo "Random seed: $RANDOM_SEED"
fi
echo ""

# Prepare command arguments
CMD_ARGS=("../$QUESTIONS_FILE")
CMD_ARGS+=("--grader" "$GRADER_MODEL")
CMD_ARGS+=("--random" "$SAMPLE_SIZE")
CMD_ARGS+=("--parallel" "$PARALLEL_WORKERS")
CMD_ARGS+=("--format" "$FORMAT")
CMD_ARGS+=("--availability-timeout" "$TIMEOUT")

if [[ -n "$RANDOM_SEED" ]]; then
    CMD_ARGS+=("--seed" "$RANDOM_SEED")
fi

if [[ -n "$INCLUDE_MODELS" ]]; then
    # Convert comma-separated to space-separated for --include
    IFS=',' read -ra MODELS_ARRAY <<< "$INCLUDE_MODELS"
    CMD_ARGS+=("--include" "${MODELS_ARRAY[@]}")
fi

if [[ -n "$EXCLUDE_MODELS" ]]; then
    # Convert comma-separated to space-separated for --exclude  
    IFS=',' read -ra MODELS_ARRAY <<< "$EXCLUDE_MODELS"
    CMD_ARGS+=("--exclude" "${MODELS_ARRAY[@]}")
fi

if [[ "$VERBOSE" == true ]]; then
    CMD_ARGS+=("--verbose")
fi

if [[ "$SAVE_INCORRECT" == true ]]; then
    CMD_ARGS+=("--save-incorrect")
fi

if [[ "$SKIP_AVAILABILITY" == true ]]; then
    CMD_ARGS+=("--skip-availability-check")
fi

# Step 1: Run the main benchmark
echo "Step 1: Running multi-model benchmark..."
echo "----------------------------------------"
echo "Command: python ../run_all_models.py ${CMD_ARGS[*]}"
echo ""

START_TIME=$(date +%s)
python ../run_all_models.py "${CMD_ARGS[@]}"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✓ Benchmark completed in ${DURATION}s"
echo ""

# Step 2: Find and analyze results
echo "Step 2: Analyzing benchmark results..."
echo "----------------------------------------"

# Find the most recent benchmark summary file
SUMMARY_FILE=$(ls -t benchmark_summary_*.json 2>/dev/null | head -n1)

if [[ -n "$SUMMARY_FILE" ]]; then
    echo "Found results file: $SUMMARY_FILE"
    
    # Reprocess results to ensure correct parsing
    echo "Reprocessing results for accuracy..."
    python ../reprocess_results.py "$SUMMARY_FILE"
    
    echo "✓ Results reprocessed"
else
    echo "Warning: No benchmark summary file found"
fi

# Step 3: Analyze incorrect answers (if save-incorrect was enabled)
if [[ "$SAVE_INCORRECT" == true ]]; then
    echo ""
    echo "Step 3: Analyzing incorrect answers..."
    echo "----------------------------------------"
    
    # Find incorrect answer files
    INCORRECT_FILES=$(ls incorrect_*.json 2>/dev/null | head -5)  # Limit to first 5
    
    if [[ -n "$INCORRECT_FILES" ]]; then
        echo "Found incorrect answer files:"
        for file in $INCORRECT_FILES; do
            echo "  - $file"
        done
        
        # Merge incorrect answers for analysis
        echo ""
        echo "Merging incorrect answers for intersection analysis..."
        python ../merge_incorrect_answers.py $INCORRECT_FILES \
            --mode intersection \
            --format argonium \
            --output merged_difficult_questions.json
        
        echo "✓ Incorrect answer analysis completed"
        echo "  - See merged_difficult_questions.json for hardest questions"
    else
        echo "No incorrect answer files found"
    fi
fi

# Step 4: Generate summary report
echo ""
echo "Step 4: Generating summary report..."
echo "----------------------------------------"

# Create a summary markdown report
REPORT_FILE="benchmark_report_$(date +%Y%m%d_%H%M%S).md"

cat > "$REPORT_FILE" << EOF
# Model Benchmark Report

Generated: $(date)

## Configuration
- Questions file: $QUESTIONS_FILE
- Grader model: $GRADER_MODEL
- Sample size: $SAMPLE_SIZE
- Parallel workers: $PARALLEL_WORKERS
- Format: $FORMAT
- Random seed: ${RANDOM_SEED:-"Not set"}
- Duration: ${DURATION}s

## Results Summary
EOF

if [[ -n "$SUMMARY_FILE" ]]; then
    echo "Detailed results available in: $SUMMARY_FILE" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Extract top performers using jq if available
    if command -v jq &> /dev/null; then
        echo "## Top Performing Models" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        jq -r '.summary[] | select(.success == true) | "- \(.model): \(.accuracy_percent)% accuracy, \(.average_confidence) confidence"' "$SUMMARY_FILE" | head -5 >> "$REPORT_FILE"
    fi
else
    echo "Results processing failed - check console output for details" >> "$REPORT_FILE"
fi

if [[ "$SAVE_INCORRECT" == true ]]; then
    echo "" >> "$REPORT_FILE"
    echo "## Error Analysis" >> "$REPORT_FILE"
    echo "- Incorrect answer files generated for detailed analysis" >> "$REPORT_FILE"
    echo "- See merged_difficult_questions.json for questions multiple models got wrong" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "## Files Generated" >> "$REPORT_FILE"
echo "- $SUMMARY_FILE: Detailed benchmark results" >> "$REPORT_FILE"
if [[ "$SAVE_INCORRECT" == true ]]; then
    echo "- incorrect_*.json: Per-model incorrect answers" >> "$REPORT_FILE"
    echo "- merged_difficult_questions.json: Hardest questions across models" >> "$REPORT_FILE"
fi
echo "- $REPORT_FILE: This summary report" >> "$REPORT_FILE"

echo "✓ Summary report generated: $REPORT_FILE"

# Final summary
echo ""
echo "=========================================="
echo "Model Benchmarking Workflow Complete!"
echo "=========================================="
echo "Duration: ${DURATION}s"
echo "Output directory: $(pwd)"
echo ""
echo "Key files:"
if [[ -n "$SUMMARY_FILE" ]]; then
    echo "  - $SUMMARY_FILE (detailed results)"
fi
echo "  - $REPORT_FILE (summary report)"
if [[ "$SAVE_INCORRECT" == true ]]; then
    echo "  - incorrect_*.json (error analysis)"
fi
echo ""
echo "Next steps:"
echo "  - Review summary report for performance insights"
echo "  - Use incorrect answer files for model improvement"
echo "  - Run reasoning analysis on difficult questions"
echo "=========================================="