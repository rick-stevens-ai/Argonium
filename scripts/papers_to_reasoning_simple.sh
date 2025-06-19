#!/bin/bash

# Simple Papers to Reasoning Traces Script
# Takes a directory of papers and outputs reasoning traces

set -e  # Exit on any error

# Default values
PAPERS_DIR=""
OUTPUT_DIR="./reasoning_output"
MODEL="gpt41"
SPECIALTY="expert"
MAX_QUESTIONS=""

# Function to display usage
usage() {
    echo "Simple Papers to Reasoning Traces"
    echo ""
    echo "Usage: $0 -d PAPERS_DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -d, --papers-dir DIR        Directory containing papers (PDF, TXT files)"
    echo ""
    echo "Options:"
    echo "  -o, --output-dir DIR        Output directory (default: ./reasoning_output)"
    echo "  -m, --model MODEL           Model for analysis (default: gpt41)"
    echo "  -s, --specialty SPECIALTY   Expert specialty (default: expert)"
    echo "  -q, --max-questions NUM     Maximum questions to process"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d ./research_papers -s microbiologist -m gpt41"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--papers-dir)
            PAPERS_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -s|--specialty)
            SPECIALTY="$2"
            shift 2
            ;;
        -q|--max-questions)
            MAX_QUESTIONS="$2"
            shift 2
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
if [[ -z "$PAPERS_DIR" ]]; then
    echo "Error: Papers directory is required (-d/--papers-dir)"
    usage
fi

if [[ ! -d "$PAPERS_DIR" ]]; then
    echo "Error: Papers directory '$PAPERS_DIR' not found"
    exit 1
fi

# Check if directory contains papers
PAPER_COUNT=$(find "$PAPERS_DIR" -name "*.pdf" -o -name "*.txt" | wc -l)
if [[ $PAPER_COUNT -eq 0 ]]; then
    echo "Error: No papers found in '$PAPERS_DIR' (looking for .pdf, .txt files)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================="
echo "Papers to Reasoning Traces"
echo "=========================================="
echo "Papers directory: $PAPERS_DIR"
echo "Papers found: $PAPER_COUNT"
echo "Output directory: $(pwd)"
echo "Model: $MODEL"
echo "Specialty: $SPECIALTY"
echo ""

# Step 1: Generate questions from papers using make_v21.py
echo "Step 1: Generating questions from papers..."
echo "----------------------------------------"

QUESTIONS_FILE="questions_from_papers.json"

echo "Running: python ../make_v21.py \"$PAPERS_DIR\" --output \"$QUESTIONS_FILE\" --model \"$MODEL\""
python ../make_v21.py "$PAPERS_DIR" \
    --output "$QUESTIONS_FILE" \
    --model "$MODEL" \
    --type mc \
    --recursive

if [[ ! -f "$QUESTIONS_FILE" ]]; then
    echo "Error: Question generation failed - $QUESTIONS_FILE not created"
    exit 1
fi

# Count generated questions
QUESTION_COUNT=$(python -c "
import json
try:
    with open('$QUESTIONS_FILE', 'r') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
")

echo "✓ Generated $QUESTION_COUNT questions"

if [[ $QUESTION_COUNT -eq 0 ]]; then
    echo "Error: No questions were generated"
    exit 1
fi

# Step 2: Generate reasoning traces using reasoning_traces_v6.py
echo ""
echo "Step 2: Generating reasoning traces..."
echo "----------------------------------------"

REASONING_OUTPUT="reasoning_traces_$(date +%Y%m%d_%H%M%S).json"

# Build command arguments
CMD_ARGS=("$QUESTIONS_FILE")
CMD_ARGS+=("--output" "$REASONING_OUTPUT")
CMD_ARGS+=("--model" "$MODEL")
CMD_ARGS+=("--specialty" "$SPECIALTY")

if [[ -n "$MAX_QUESTIONS" ]]; then
    CMD_ARGS+=("--max-questions" "$MAX_QUESTIONS")
fi

echo "Generating reasoning traces with $SPECIALTY perspective..."
python ../reasoning_traces_v6.py "${CMD_ARGS[@]}"

if [[ ! -f "$REASONING_OUTPUT" ]]; then
    echo "Error: Reasoning trace generation failed - $REASONING_OUTPUT not created"
    exit 1
fi

# Extract basic statistics
REASONING_STATS=$(python -c "
import json
try:
    with open('$REASONING_OUTPUT', 'r') as f:
        data = json.load(f)
    
    total = len(data)
    correct = sum(1 for trace in data if trace.get('prediction_correct', False))
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f'{total},{correct},{accuracy:.1f}')
except:
    print('0,0,0.0')
")

IFS=',' read -r TOTAL_Q CORRECT_Q ACCURACY_PCT <<< "$REASONING_STATS"

echo "✓ Generated reasoning traces for $TOTAL_Q questions"
echo "✓ Accuracy: $ACCURACY_PCT% ($CORRECT_Q/$TOTAL_Q correct)"

# Final summary
echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Input: $PAPERS_DIR ($PAPER_COUNT papers)"
echo "Output: $(pwd)"
echo ""
echo "Generated Files:"
echo "  - $QUESTIONS_FILE ($QUESTION_COUNT questions)"
echo "  - $REASONING_OUTPUT ($TOTAL_Q reasoning traces)"
echo ""
echo "Results:"
echo "  - Questions generated: $QUESTION_COUNT"
echo "  - Reasoning traces: $TOTAL_Q"
echo "  - Accuracy: $ACCURACY_PCT%"
echo "  - Expert perspective: $SPECIALTY"
echo ""
echo "Reasoning traces are ready in: $REASONING_OUTPUT"
echo "=========================================="