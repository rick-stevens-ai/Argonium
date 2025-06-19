#!/bin/bash

# Literature Discovery Workflow Script
# Automates the process of discovering, downloading, and analyzing academic papers

set -e  # Exit on any error

# Default values
BASE_KEYWORD=""
OUTPUT_DIR="./research_output"
MAX_PAPERS=50
MODEL="gpt41"
RELEVANCE_MODEL="scout"
GENERATE_KEYWORDS=false
KEYWORDS_FILE=""
ORGANIZE_FILES=false

# Function to display usage
usage() {
    echo "Literature Discovery Workflow"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -k, --keyword KEYWORD        Base keyword for research (required if not using -f)"
    echo "  -f, --keywords-file FILE     File containing keywords (one per line)"
    echo "  -g, --generate-keywords      Generate related keywords using AI"
    echo "  -o, --output-dir DIR         Output directory (default: ./research_output)"
    echo "  -m, --max-papers NUM         Maximum papers per keyword (default: 50)"
    echo "  --model MODEL                Model for keyword generation (default: gpt41)"
    echo "  --relevance-model MODEL      Model for relevance checking (default: scout)"
    echo "  --organize                   Organize files by classification"
    echo "  --analyze-only               Only analyze existing papers (skip download)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -k \"antibiotic resistance\" -g --organize"
    echo "  $0 -f keywords.txt -m 100 --model gpt41"
    echo "  $0 --analyze-only -o ./existing_papers"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--keyword)
            BASE_KEYWORD="$2"
            shift 2
            ;;
        -f|--keywords-file)
            KEYWORDS_FILE="$2"
            shift 2
            ;;
        -g|--generate-keywords)
            GENERATE_KEYWORDS=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--max-papers)
            MAX_PAPERS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --relevance-model)
            RELEVANCE_MODEL="$2"
            shift 2
            ;;
        --organize)
            ORGANIZE_FILES=true
            shift
            ;;
        --analyze-only)
            ANALYZE_ONLY=true
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

# Validate arguments
if [[ -z "$BASE_KEYWORD" && -z "$KEYWORDS_FILE" && -z "$ANALYZE_ONLY" ]]; then
    echo "Error: Either --keyword or --keywords-file must be specified (unless using --analyze-only)"
    usage
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================="
echo "Literature Discovery Workflow"
echo "=========================================="
echo "Output directory: $(pwd)"
echo "Max papers per keyword: $MAX_PAPERS"
echo "Model: $MODEL"
echo "Relevance model: $RELEVANCE_MODEL"
echo ""

# Step 1: Download papers (unless analyze-only mode)
if [[ -z "$ANALYZE_ONLY" ]]; then
    echo "Step 1: Downloading papers..."
    echo "----------------------------------------"
    
    if [[ -n "$KEYWORDS_FILE" ]]; then
        # Use keywords from file
        echo "Using keywords from file: $KEYWORDS_FILE"
        python ../download_papers_v8.py \
            --keywords-file "../$KEYWORDS_FILE" \
            --relevance-model "$RELEVANCE_MODEL" \
            --max-relevant-papers "$MAX_PAPERS" \
            --skip-relevance-check false
    elif [[ "$GENERATE_KEYWORDS" == true ]]; then
        # Generate keywords and download
        echo "Generating keywords from base: '$BASE_KEYWORD'"
        python ../download_papers_v8.py \
            --generate-keywords "$BASE_KEYWORD" \
            --model "$MODEL" \
            --relevance-model "$RELEVANCE_MODEL" \
            --max-relevant-papers "$MAX_PAPERS"
    else
        # Use single keyword
        echo "Using single keyword: '$BASE_KEYWORD'"
        python ../download_papers_v8.py \
            --keywords "$BASE_KEYWORD" \
            --relevance-model "$RELEVANCE_MODEL" \
            --max-relevant-papers "$MAX_PAPERS"
    fi
    
    echo "✓ Paper download completed"
    echo ""
else
    echo "Skipping download step (analyze-only mode)"
    echo ""
fi

# Step 2: Classify papers (if organize flag is set)
if [[ "$ORGANIZE_FILES" == true ]]; then
    echo "Step 2: Classifying and organizing papers..."
    echo "----------------------------------------"
    
    # Find directories with papers to classify
    for dir in */; do
        if [[ -d "$dir" && "$dir" != "PROCESSED/" ]]; then
            echo "Classifying papers in: $dir"
            python ../classify_papers.py "$dir" \
                --classification-model "$RELEVANCE_MODEL" \
                --organize-files \
                --max-files 1000
        fi
    done
    
    echo "✓ Paper classification completed"
    echo ""
fi

# Step 3: Analyze resources
echo "Step 3: Analyzing resources and generating summaries..."
echo "----------------------------------------"

# Run resource analysis
python ../analyze_resources.py

echo "✓ Resource analysis completed"
echo ""

# Step 4: Generate summary report
echo "Step 4: Generating workflow summary..."
echo "----------------------------------------"

# Count total papers downloaded
TOTAL_PAPERS=0
TOTAL_DIRS=0

for dir in */; do
    if [[ -d "$dir" && "$dir" != "PROCESSED/" ]]; then
        PAPER_COUNT=$(find "$dir" -name "*.pdf" -o -name "*.txt" | grep -v "_syn_abs.txt" | wc -l)
        echo "  - $dir: $PAPER_COUNT papers"
        TOTAL_PAPERS=$((TOTAL_PAPERS + PAPER_COUNT))
        TOTAL_DIRS=$((TOTAL_DIRS + 1))
    fi
done

echo ""
echo "=========================================="
echo "Literature Discovery Workflow Complete!"
echo "=========================================="
echo "Summary:"
echo "  - Directories processed: $TOTAL_DIRS"
echo "  - Total papers: $TOTAL_PAPERS"
echo "  - Output location: $(pwd)"
echo ""
echo "Generated files:"
echo "  - README.md: Overview of collected papers"
echo "  - Individual syn_abs files: Paper summaries"
if [[ "$ORGANIZE_FILES" == true ]]; then
    echo "  - Topic-organized directories"
    echo "  - PROCESSED/: Original files"
fi
echo ""
echo "Next steps:"
echo "  - Review README.md for paper summaries"
echo "  - Use papers for question generation or analysis"
echo "  - Run model evaluation workflows on generated content"
echo "=========================================="