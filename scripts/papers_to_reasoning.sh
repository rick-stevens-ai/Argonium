#!/bin/bash

# Papers to Reasoning Chains Workflow Script
# Takes a directory of papers and outputs reasoning chains for derived questions

set -e  # Exit on any error

# Default values
PAPERS_DIR=""
OUTPUT_DIR="./papers_to_reasoning_output"
CLASSIFICATION_MODEL="scout"
REASONING_MODEL="gpt41"
SPECIALTY="expert"
MAX_QUESTIONS=50
SAMPLE_SIZE=25
ORGANIZE_PAPERS=true
WHOLE_TRACE=false
QUICK_MODE=false

# Function to display usage
usage() {
    echo "Papers to Reasoning Chains Workflow"
    echo ""
    echo "Takes a directory of papers and generates reasoning chains for questions derived from them."
    echo ""
    echo "Usage: $0 -d PAPERS_DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -d, --papers-dir DIR        Directory containing papers (PDF, TXT, MD files)"
    echo ""
    echo "Options:"
    echo "  -o, --output-dir DIR        Output directory (default: ./papers_to_reasoning_output)"
    echo "  --classification-model M    Model for paper classification (default: scout)"
    echo "  --reasoning-model M         Model for reasoning analysis (default: gpt41)"
    echo "  -s, --specialty SPECIALTY   Expert specialty for reasoning (default: expert)"
    echo "  -q, --max-questions NUM     Max questions to generate/analyze (default: 50)"
    echo "  --sample-size NUM           Sample size for reasoning analysis (default: 25)"
    echo "  --no-organize               Skip paper organization step"
    echo "  --whole-trace               Enable whole trace meta-analysis"
    echo "  --quick                     Quick mode (fewer questions, faster processing)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Specialty Options:"
    echo "  - microbiologist: Bacterial resistance and pathogenesis expert"
    echo "  - physicist: Theoretical and computational physics expert"
    echo "  - quantum physicist: Quantum mechanics and computing expert"
    echo "  - historian: Historical analysis expert"
    echo "  - expert: Generic expert (default)"
    echo ""
    echo "Examples:"
    echo "  $0 -d ./research_papers -s microbiologist --whole-trace"
    echo "  $0 -d ./quantum_papers -s physicist --quick"
    echo "  $0 -d ./papers --reasoning-model gpt41 -q 100"
    echo ""
    echo "Workflow Steps:"
    echo "  1. Classify and organize papers by topic"
    echo "  2. Extract and analyze paper content"
    echo "  3. Generate questions from paper content (manual step)"
    echo "  4. Analyze reasoning chains with expert perspective"
    echo "  5. Generate comprehensive reports"
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
        --classification-model)
            CLASSIFICATION_MODEL="$2"
            shift 2
            ;;
        --reasoning-model)
            REASONING_MODEL="$2"
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
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --no-organize)
            ORGANIZE_PAPERS=false
            shift
            ;;
        --whole-trace)
            WHOLE_TRACE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            MAX_QUESTIONS=20
            SAMPLE_SIZE=15
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
if [[ -z "$PAPERS_DIR" ]]; then
    echo "Error: Papers directory is required (-d/--papers-dir)"
    usage
fi

if [[ ! -d "$PAPERS_DIR" ]]; then
    echo "Error: Papers directory '$PAPERS_DIR' not found"
    exit 1
fi

# Check if directory contains papers
PAPER_COUNT=$(find "$PAPERS_DIR" -name "*.pdf" -o -name "*.txt" -o -name "*.md" | wc -l)
if [[ $PAPER_COUNT -eq 0 ]]; then
    echo "Error: No papers found in '$PAPERS_DIR' (looking for .pdf, .txt, .md files)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================="
echo "Papers to Reasoning Chains Workflow"
echo "=========================================="
echo "Papers directory: $PAPERS_DIR"
echo "Papers found: $PAPER_COUNT"
echo "Output directory: $(pwd)"
echo "Classification model: $CLASSIFICATION_MODEL"
echo "Reasoning model: $REASONING_MODEL"
echo "Expert specialty: $SPECIALTY"
echo "Max questions: $MAX_QUESTIONS"
echo "Sample size: $SAMPLE_SIZE"
echo "Organize papers: $ORGANIZE_PAPERS"
echo "Whole trace analysis: $WHOLE_TRACE"
echo ""

WORKFLOW_START=$(date +%s)

# Step 1: Classify and organize papers
echo "=========================================="
echo "STEP 1: Paper Classification & Organization"
echo "=========================================="

mkdir -p classified_papers
cp -r "$PAPERS_DIR"/* classified_papers/ 2>/dev/null || true

cd classified_papers

if [[ "$ORGANIZE_PAPERS" == true ]]; then
    echo "Classifying and organizing papers by topic..."
    
    python ../../classify_papers.py . \
        --classification-model "$CLASSIFICATION_MODEL" \
        --organize-files \
        --max-files 1000
    
    echo "✓ Paper classification completed"
    
    # Count organized papers
    ORGANIZED_DIRS=$(find . -maxdepth 1 -type d ! -name "." ! -name "PROCESSED" | wc -l)
    echo "Papers organized into $ORGANIZED_DIRS topic directories"
else
    echo "Skipping paper organization (--no-organize specified)"
    
    # Still run classification for synopsis generation
    echo "Generating paper synopses..."
    python ../../classify_papers.py . \
        --classification-model "$CLASSIFICATION_MODEL" \
        --skip-classification \
        --max-files 1000
    
    echo "✓ Synopsis generation completed"
fi

cd ..

# Step 2: Analyze paper content
echo ""
echo "=========================================="
echo "STEP 2: Content Analysis & Summary"
echo "=========================================="

cd classified_papers

echo "Analyzing paper content and generating summaries..."
python ../../analyze_resources.py

echo "✓ Content analysis completed"

# Find the generated README
if [[ -f "README.md" ]]; then
    echo "Generated comprehensive literature analysis: README.md"
    TOTAL_ANALYZED=$(grep -o "Total subdirectories analyzed: [0-9]*" README.md | grep -o "[0-9]*" || echo "N/A")
    echo "Subdirectories analyzed: $TOTAL_ANALYZED"
else
    echo "Warning: README.md not generated by analyze_resources.py"
fi

cd ..

# Step 3: Question Generation (Manual Step with Guidance)
echo ""
echo "=========================================="
echo "STEP 3: Question Generation"
echo "=========================================="

QUESTIONS_FILE="questions_from_papers.json"

if [[ -f "$QUESTIONS_FILE" ]]; then
    echo "Found existing questions file: $QUESTIONS_FILE"
    EXISTING_QUESTIONS=$(python -c "import json; print(len(json.load(open('$QUESTIONS_FILE'))))" 2>/dev/null || echo "0")
    echo "Existing questions: $EXISTING_QUESTIONS"
else
    echo "No existing questions file found. Generating sample questions based on paper topics..."
    
    # Extract topics from classified papers to help generate sample questions
    SAMPLE_TOPICS=""
    if [[ "$ORGANIZE_PAPERS" == true ]]; then
        # Get directory names as topics
        SAMPLE_TOPICS=$(find classified_papers -maxdepth 1 -type d ! -name "." ! -name "PROCESSED" | head -3 | sed 's|classified_papers/||' | tr '\n' ', ' | sed 's/,$//')
    fi
    
    if [[ -z "$SAMPLE_TOPICS" ]]; then
        SAMPLE_TOPICS="research topic, scientific methodology, data analysis"
    fi
    
    echo "Creating sample questions based on topics: $SAMPLE_TOPICS"
    
    # Generate sample questions file
    cat > "$QUESTIONS_FILE" << EOF
[
    {
        "question": "Based on the research papers, what is the primary methodology used in the studies?\n\n1. Experimental analysis with control groups\n2. Computational modeling and simulation\n3. Literature review and meta-analysis\n4. All of the above (*)",
        "answer": "All of the above",
        "text": "Research methodologies vary across studies and often combine multiple approaches for comprehensive analysis.",
        "type": "multiple-choice"
    },
    {
        "question": "What are the main limitations identified in the research papers?\n\n1. Sample size constraints\n2. Methodological limitations\n3. Data availability issues\n4. All of the above (*)",
        "answer": "All of the above",
        "text": "Research limitations are commonly reported across multiple dimensions including sample, method, and data constraints.",
        "type": "multiple-choice"
    },
    {
        "question": "Which statistical approach is most commonly used for data analysis?\n\n1. Descriptive statistics only\n2. Inferential statistical testing (*)\n3. Machine learning algorithms\n4. Qualitative analysis",
        "answer": "Inferential statistical testing",
        "text": "Inferential statistics allow researchers to make conclusions beyond the immediate data and test hypotheses.",
        "type": "multiple-choice"
    }
]
EOF
    
    echo "✓ Created sample questions file: $QUESTIONS_FILE"
    echo ""
    echo "IMPORTANT: Manual Step Required"
    echo "=============================="
    echo "The sample questions are generic. For better results:"
    echo "1. Review the paper summaries in classified_papers/README.md"
    echo "2. Edit $QUESTIONS_FILE to create domain-specific questions"
    echo "3. Base questions on actual findings and concepts from your papers"
    echo "4. Ensure questions test understanding of the specific research area"
    echo ""
    echo "Press Enter to continue with sample questions, or Ctrl+C to exit and create custom questions..."
    read -r
fi

# Validate questions file
QUESTION_COUNT=$(python -c "
import json
try:
    with open('$QUESTIONS_FILE', 'r') as f:
        data = json.load(f)
    print(len(data))
except:
    print(0)
")

if [[ $QUESTION_COUNT -eq 0 ]]; then
    echo "Error: No valid questions found in $QUESTIONS_FILE"
    exit 1
fi

echo "✓ Questions available: $QUESTION_COUNT"

# Limit questions if requested
if [[ $QUESTION_COUNT -gt $MAX_QUESTIONS ]]; then
    echo "Sampling $MAX_QUESTIONS questions from $QUESTION_COUNT available..."
    python ../../sample_json.py "$QUESTIONS_FILE" "$MAX_QUESTIONS" > "sampled_questions.json"
    QUESTIONS_FILE="sampled_questions.json"
    QUESTION_COUNT=$MAX_QUESTIONS
fi

# Step 4: Reasoning Analysis
echo ""
echo "=========================================="
echo "STEP 4: Reasoning Chain Analysis"
echo "=========================================="

mkdir -p reasoning_analysis
cd reasoning_analysis

echo "Generating detailed reasoning chains with $SPECIALTY perspective..."

# Prepare reasoning analysis command
CMD_ARGS=("../$QUESTIONS_FILE")
CMD_ARGS+=("--model" "$REASONING_MODEL")
CMD_ARGS+=("--specialty" "$SPECIALTY")
CMD_ARGS+=("--save-interval" "5")

if [[ $SAMPLE_SIZE -lt $QUESTION_COUNT ]]; then
    CMD_ARGS+=("--max-questions" "$SAMPLE_SIZE")
fi

# Generate output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="reasoning_traces_${SPECIALTY}_${TIMESTAMP}.json"
CMD_ARGS+=("--output" "$OUTPUT_FILE")

if [[ "$WHOLE_TRACE" == true ]]; then
    CMD_ARGS+=("--whole-trace-analysis")
    WHOLE_TRACE_OUTPUT="whole_trace_analysis_${SPECIALTY}_${TIMESTAMP}.json"
    CMD_ARGS+=("--whole-trace-output" "$WHOLE_TRACE_OUTPUT")
fi

echo "Command: python ../../reasoning_traces_v6.py ${CMD_ARGS[*]}"
echo ""

START_TIME=$(date +%s)
python ../../reasoning_traces_v6.py "${CMD_ARGS[@]}"
END_TIME=$(date +%s)
REASONING_DURATION=$((END_TIME - START_TIME))

echo ""
echo "✓ Reasoning analysis completed in ${REASONING_DURATION}s"

# Extract performance statistics
if [[ -f "$OUTPUT_FILE" ]]; then
    REASONING_STATS=$(python -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    total = len(data)
    correct = sum(1 for trace in data if trace.get('prediction_correct', False))
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Count confidence levels
    conf_counts = {'high': 0, 'medium': 0, 'low': 0}
    for trace in data:
        conf = trace.get('reasoning', {}).get('prediction', {}).get('confidence_level', 'unknown').lower()
        if conf in conf_counts:
            conf_counts[conf] += 1
    
    print(f'{total},{correct},{accuracy:.1f},{conf_counts[\"high\"]},{conf_counts[\"medium\"]},{conf_counts[\"low\"]}')
except:
    print('0,0,0.0,0,0,0')
")
    
    IFS=',' read -r TOTAL_Q CORRECT_Q ACCURACY_PCT HIGH_CONF MED_CONF LOW_CONF <<< "$REASONING_STATS"
    
    echo ""
    echo "Reasoning Performance:"
    echo "  - Questions analyzed: $TOTAL_Q"
    echo "  - Correct predictions: $CORRECT_Q"
    echo "  - Accuracy: $ACCURACY_PCT%"
    echo "  - High confidence: $HIGH_CONF"
    echo "  - Medium confidence: $MED_CONF"
    echo "  - Low confidence: $LOW_CONF"
fi

cd ..

# Step 5: Generate comprehensive report
echo ""
echo "=========================================="
echo "STEP 5: Final Report Generation"
echo "=========================================="

WORKFLOW_END=$(date +%s)
TOTAL_DURATION=$((WORKFLOW_END - WORKFLOW_START))

FINAL_REPORT="papers_to_reasoning_report_$(date +%Y%m%d_%H%M%S).md"

cat > "$FINAL_REPORT" << EOF
# Papers to Reasoning Chains Report

**Generated**: $(date)  
**Duration**: ${TOTAL_DURATION}s  
**Expert Perspective**: $SPECIALTY  

## Input Summary
- **Papers directory**: $PAPERS_DIR
- **Papers processed**: $PAPER_COUNT
- **Classification model**: $CLASSIFICATION_MODEL
- **Reasoning model**: $REASONING_MODEL

## Workflow Results

### Step 1: Paper Classification
EOF

if [[ "$ORGANIZE_PAPERS" == true ]]; then
    echo "- **Organization**: Papers classified and organized by topic" >> "$FINAL_REPORT"
    echo "- **Topic directories**: $ORGANIZED_DIRS created" >> "$FINAL_REPORT"
else
    echo "- **Organization**: Skipped (synopsis generation only)" >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF
- **Synopses**: Generated for all papers

### Step 2: Content Analysis
- **Literature analysis**: Comprehensive summaries generated
- **Key themes**: Extracted using TF-IDF and TextRank
- **Analysis file**: classified_papers/README.md

### Step 3: Question Generation
- **Questions available**: $QUESTION_COUNT
- **Questions analyzed**: $TOTAL_Q
- **Source**: $(if [[ -f "sampled_questions.json" ]]; then echo "Sampled from $QUESTIONS_FILE"; else echo "$QUESTIONS_FILE"; fi)

### Step 4: Reasoning Analysis
- **Accuracy**: $ACCURACY_PCT% ($CORRECT_Q/$TOTAL_Q correct)
- **High confidence decisions**: $HIGH_CONF
- **Medium confidence decisions**: $MED_CONF
- **Low confidence decisions**: $LOW_CONF
- **Analysis duration**: ${REASONING_DURATION}s

## Key Insights

### Paper Content
The classified papers cover $(echo "$SAMPLE_TOPICS" | tr ',' '\n' | wc -l) main topic areas. Detailed content analysis and keyword extraction provide comprehensive coverage of the research domain.

### Reasoning Quality
The $SPECIALTY perspective achieved $ACCURACY_PCT% accuracy on questions derived from the paper content, with detailed thought processes for each question option.

### Confidence Calibration
- High confidence: $HIGH_CONF decisions
- Medium confidence: $MED_CONF decisions  
- Low confidence: $LOW_CONF decisions

This distribution indicates the model's self-assessment when reasoning about domain-specific content.

## Generated Files

### Paper Analysis
- \`classified_papers/README.md\`: Comprehensive literature analysis
EOF

if [[ "$ORGANIZE_PAPERS" == true ]]; then
    echo "- \`classified_papers/*/\`: Topic-organized paper collections" >> "$FINAL_REPORT"
    echo "- \`classified_papers/PROCESSED/\`: Original papers archive" >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF
- \`classified_papers/*_syn_abs.txt\`: Paper synopses

### Question Analysis
- \`$QUESTIONS_FILE\`: Questions used for reasoning analysis
EOF

if [[ -f "sampled_questions.json" ]]; then
    echo "- \`sampled_questions.json\`: Sampled subset for analysis" >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF

### Reasoning Analysis
- \`reasoning_analysis/$OUTPUT_FILE\`: Detailed reasoning traces
EOF

if [[ "$WHOLE_TRACE" == true && -n "$WHOLE_TRACE_OUTPUT" ]]; then
    echo "- \`reasoning_analysis/$WHOLE_TRACE_OUTPUT\`: Meta-cognitive analysis" >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF
- \`$FINAL_REPORT\`: This comprehensive report

## Usage for Further Analysis

### Extract High-Quality Reasoning Examples
\`\`\`bash
# Sample best reasoning traces
python ../sample_json.py reasoning_analysis/$OUTPUT_FILE 10 > best_reasoning_examples.json

# Focus on high-confidence correct predictions
python -c "
import json
with open('reasoning_analysis/$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
high_conf_correct = [t for t in data 
                    if t.get('prediction_correct', False) and 
                    t.get('reasoning', {}).get('prediction', {}).get('confidence_level', '').lower() == 'high']
with open('high_confidence_correct.json', 'w') as f:
    json.dump(high_conf_correct[:5], f, indent=2)
print(f'Extracted {len(high_conf_correct[:5])} high-confidence correct examples')
"
\`\`\`

### Analyze Error Patterns
\`\`\`bash
# Extract incorrect predictions for analysis
python -c "
import json
with open('reasoning_analysis/$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
incorrect = [t for t in data if not t.get('prediction_correct', False)]
with open('reasoning_errors.json', 'w') as f:
    json.dump(incorrect, f, indent=2)
print(f'Extracted {len(incorrect)} incorrect predictions for error analysis')
"
\`\`\`

### Generate Formatted Reports
\`\`\`bash
# Convert reasoning traces to readable format
python ../render_json.py reasoning_analysis/$OUTPUT_FILE reasoning_report.md
\`\`\`

## Next Steps

1. **Review high-quality reasoning examples** for cognitive patterns
2. **Analyze error cases** to identify improvement opportunities  
3. **Compare different expert perspectives** on the same questions
4. **Use insights** for model training or validation datasets
5. **Expand question set** based on additional paper analysis

## Reproduction
\`\`\`bash
./scripts/papers_to_reasoning.sh \\
    --papers-dir "$PAPERS_DIR" \\
    --specialty "$SPECIALTY" \\
    --reasoning-model "$REASONING_MODEL" \\
    --max-questions "$MAX_QUESTIONS"
\`\`\`
EOF

echo "✓ Final report generated: $FINAL_REPORT"

# Create directory summary
echo ""
echo "=========================================="
echo "PAPERS TO REASONING CHAINS COMPLETE!"
echo "=========================================="
echo "Input: $PAPERS_DIR ($PAPER_COUNT papers)"
echo "Duration: ${TOTAL_DURATION}s"
echo "Output: $(pwd)"
echo ""
echo "Results Summary:"
echo "  📚 Papers: $PAPER_COUNT processed and analyzed"
echo "  ❓ Questions: $QUESTION_COUNT available, $TOTAL_Q analyzed"
echo "  🧠 Reasoning: $ACCURACY_PCT% accuracy ($SPECIALTY perspective)"
echo "  📊 Confidence: $HIGH_CONF high, $MED_CONF medium, $LOW_CONF low"
echo ""
echo "Key Outputs:"
echo "  📋 Literature analysis: classified_papers/README.md"
echo "  🔗 Reasoning traces: reasoning_analysis/$OUTPUT_FILE"
if [[ "$WHOLE_TRACE" == true && -n "$WHOLE_TRACE_OUTPUT" ]]; then
    echo "  🌊 Meta-analysis: reasoning_analysis/$WHOLE_TRACE_OUTPUT"
fi
echo "  📄 Final report: $FINAL_REPORT"
echo ""
echo "Next Steps:"
echo "  1. Review the final report for comprehensive insights"
echo "  2. Examine high-quality reasoning examples"
echo "  3. Analyze error patterns for improvement opportunities"
echo "  4. Consider testing different expert perspectives"
echo "=========================================="