#!/bin/bash

# Complete Research Pipeline Script
# End-to-end workflow from literature discovery to model evaluation and reasoning analysis

set -e  # Exit on any error

# Default values
RESEARCH_TOPIC=""
OUTPUT_DIR="./research_pipeline"
MAX_PAPERS=30
SAMPLE_QUESTIONS=50
MODEL="gpt41"
RELEVANCE_MODEL="scout"
SPECIALTY="expert"
QUICK_MODE=false
SKIP_DOWNLOAD=false
SKIP_BENCHMARK=false
SKIP_REASONING=false

# Function to display usage
usage() {
    echo "Complete Research Pipeline"
    echo ""
    echo "Usage: $0 -t TOPIC [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -t, --topic TOPIC           Research topic (required)"
    echo ""
    echo "Options:"
    echo "  -o, --output-dir DIR        Output directory (default: ./research_pipeline)"
    echo "  -p, --max-papers NUM        Max papers to download (default: 30)"
    echo "  -q, --sample-questions NUM  Questions for benchmarking (default: 50)"
    echo "  -m, --model MODEL           Primary model (default: gpt41)"
    echo "  --relevance-model MODEL     Model for relevance checking (default: scout)"
    echo "  -s, --specialty SPECIALTY   Expert specialty for reasoning (default: expert)"
    echo "  --quick                     Quick mode (fewer papers/questions)"
    echo "  --skip-download             Skip literature download phase"
    echo "  --skip-benchmark            Skip model benchmarking phase"
    echo "  --skip-reasoning            Skip reasoning analysis phase"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -t \"antibiotic resistance\" -s microbiologist --quick"
    echo "  $0 -t \"quantum computing\" -s physicist -p 50 -q 100"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--topic)
            RESEARCH_TOPIC="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--max-papers)
            MAX_PAPERS="$2"
            shift 2
            ;;
        -q|--sample-questions)
            SAMPLE_QUESTIONS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --relevance-model)
            RELEVANCE_MODEL="$2"
            shift 2
            ;;
        -s|--specialty)
            SPECIALTY="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            MAX_PAPERS=15
            SAMPLE_QUESTIONS=25
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --skip-reasoning)
            SKIP_REASONING=true
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
if [[ -z "$RESEARCH_TOPIC" ]]; then
    echo "Error: Research topic is required (-t/--topic)"
    usage
fi

# Create main output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Create subdirectories for different phases
mkdir -p literature_data
mkdir -p benchmark_results  
mkdir -p reasoning_analysis
mkdir -p final_reports

echo "=========================================="
echo "Complete Research Pipeline"
echo "=========================================="
echo "Research topic: $RESEARCH_TOPIC"
echo "Output directory: $(pwd)"
echo "Max papers: $MAX_PAPERS"
echo "Sample questions: $SAMPLE_QUESTIONS"
echo "Primary model: $MODEL"
echo "Relevance model: $RELEVANCE_MODEL"
echo "Specialty: $SPECIALTY"
echo "Quick mode: $QUICK_MODE"
echo ""

PIPELINE_START=$(date +%s)

# Phase 1: Literature Discovery and Analysis
if [[ "$SKIP_DOWNLOAD" != true ]]; then
    echo "=========================================="
    echo "PHASE 1: Literature Discovery & Analysis"
    echo "=========================================="
    
    cd literature_data
    
    echo "Discovering and downloading papers on: '$RESEARCH_TOPIC'"
    ../scripts/literature_discovery.sh \
        --keyword "$RESEARCH_TOPIC" \
        --generate-keywords \
        --max-papers "$MAX_PAPERS" \
        --model "$MODEL" \
        --relevance-model "$RELEVANCE_MODEL" \
        --organize
    
    PAPERS_DOWNLOADED=$(find . -name "*.pdf" -o -name "*.txt" | grep -v "_syn_abs.txt" | wc -l)
    echo "✓ Phase 1 completed: $PAPERS_DOWNLOADED papers downloaded and analyzed"
    
    cd ..
else
    echo "Skipping literature download phase"
    PAPERS_DOWNLOADED="N/A (skipped)"
fi

# Check if we have questions file for benchmarking
QUESTIONS_FILE=""
if [[ -f "literature_data/questions.json" ]]; then
    QUESTIONS_FILE="literature_data/questions.json"
elif [[ -f "questions.json" ]]; then
    QUESTIONS_FILE="questions.json"
else
    echo ""
    echo "Note: No questions file found. Using sample questions for demonstration."
    echo "To use your own questions, place them in questions.json"
    
    # Create a sample questions file for demonstration
    cat > questions.json << EOF
[
    {
        "question": "What is the primary mechanism of antibiotic resistance in bacteria?\n\n1. Efflux pumps that remove antibiotics from the cell\n2. Enzymatic degradation of antibiotics\n3. Alteration of antibiotic target sites\n4. All of the above (*)",
        "answer": "All of the above",
        "text": "Antibiotic resistance mechanisms are diverse and often work in combination.",
        "type": "multiple-choice"
    },
    {
        "question": "Which enzyme is commonly responsible for beta-lactam resistance?\n\n1. DNA gyrase\n2. Beta-lactamase (*)\n3. RNA polymerase\n4. Peptidyl transferase",
        "answer": "Beta-lactamase",
        "text": "Beta-lactamases cleave the beta-lactam ring structure of penicillins and related antibiotics.",
        "type": "multiple-choice"
    }
]
EOF
    QUESTIONS_FILE="questions.json"
    echo "Created sample questions file: $QUESTIONS_FILE"
fi

# Phase 2: Model Benchmarking
if [[ "$SKIP_BENCHMARK" != true ]]; then
    echo ""
    echo "=========================================="
    echo "PHASE 2: Multi-Model Benchmarking"
    echo "=========================================="
    
    cd benchmark_results
    
    echo "Running comprehensive model evaluation..."
    ../scripts/model_benchmark.sh \
        --questions "../$QUESTIONS_FILE" \
        --grader "$MODEL" \
        --sample-size "$SAMPLE_QUESTIONS" \
        --parallel 8 \
        --save-incorrect \
        --verbose
    
    # Find benchmark results
    BENCHMARK_FILE=$(ls -t benchmark_summary_*.json 2>/dev/null | head -n1)
    if [[ -n "$BENCHMARK_FILE" ]]; then
        echo "✓ Phase 2 completed: Benchmark results in $BENCHMARK_FILE"
        MODELS_TESTED=$(jq -r '.metadata.total_models_tested' "$BENCHMARK_FILE" 2>/dev/null || echo "N/A")
        SUCCESSFUL_RUNS=$(jq -r '.metadata.successful_runs' "$BENCHMARK_FILE" 2>/dev/null || echo "N/A")
    else
        echo "Warning: Benchmark results file not found"
        MODELS_TESTED="N/A"
        SUCCESSFUL_RUNS="N/A"
    fi
    
    cd ..
else
    echo "Skipping model benchmarking phase"
    MODELS_TESTED="N/A (skipped)"
    SUCCESSFUL_RUNS="N/A (skipped)"
fi

# Phase 3: Reasoning Analysis
if [[ "$SKIP_REASONING" != true ]]; then
    echo ""
    echo "=========================================="
    echo "PHASE 3: Detailed Reasoning Analysis"
    echo "=========================================="
    
    cd reasoning_analysis
    
    REASONING_QUESTIONS=20
    if [[ "$QUICK_MODE" != true ]]; then
        REASONING_QUESTIONS=50
    fi
    
    echo "Generating detailed reasoning traces with $SPECIALTY perspective..."
    ../scripts/reasoning_analysis.sh \
        --questions "../$QUESTIONS_FILE" \
        --model "$MODEL" \
        --specialty "$SPECIALTY" \
        --max-questions "$REASONING_QUESTIONS" \
        --whole-trace \
        --save-interval 5
    
    # Find reasoning results
    REASONING_FILE=$(ls -t reasoning_traces_*.json 2>/dev/null | head -n1)
    if [[ -n "$REASONING_FILE" ]]; then
        echo "✓ Phase 3 completed: Reasoning analysis in $REASONING_FILE"
        
        # Extract accuracy from reasoning analysis
        REASONING_ACCURACY=$(python3 -c "
import json
try:
    with open('$REASONING_FILE', 'r') as f:
        data = json.load(f)
    total = len(data)
    correct = sum(1 for trace in data if trace.get('prediction_correct', False))
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f'{accuracy:.1f}%')
except:
    print('N/A')
")
    else
        echo "Warning: Reasoning analysis results not found"
        REASONING_ACCURACY="N/A"
    fi
    
    cd ..
else
    echo "Skipping reasoning analysis phase"
    REASONING_ACCURACY="N/A (skipped)"
fi

# Phase 4: Generate Final Report
echo ""
echo "=========================================="
echo "PHASE 4: Final Report Generation"
echo "=========================================="

cd final_reports

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

# Generate comprehensive final report
FINAL_REPORT="research_pipeline_report_$(date +%Y%m%d_%H%M%S).md"

cat > "$FINAL_REPORT" << EOF
# Complete Research Pipeline Report

**Research Topic**: $RESEARCH_TOPIC  
**Generated**: $(date)  
**Duration**: ${PIPELINE_DURATION}s  
**Specialty Perspective**: $SPECIALTY  

## Pipeline Configuration
- Primary model: $MODEL
- Relevance model: $RELEVANCE_MODEL
- Max papers: $MAX_PAPERS
- Sample questions: $SAMPLE_QUESTIONS
- Quick mode: $QUICK_MODE

## Pipeline Results Summary

### Phase 1: Literature Discovery
- **Papers downloaded**: $PAPERS_DOWNLOADED
- **Organization**: Topic-based classification completed
- **Analysis**: Resource summaries generated

### Phase 2: Model Benchmarking  
- **Models tested**: $MODELS_TESTED
- **Successful runs**: $SUCCESSFUL_RUNS
- **Error analysis**: Incorrect answers saved for review

### Phase 3: Reasoning Analysis
- **Perspective**: $SPECIALTY expert reasoning
- **Accuracy**: $REASONING_ACCURACY
- **Analysis depth**: Detailed thought processes with confidence assessment

## Key Insights

### Literature Coverage
The literature discovery phase successfully identified and organized research papers related to "$RESEARCH_TOPIC". Papers were automatically classified by topic and analyzed for key themes and content.

### Model Performance
EOF

if [[ "$SKIP_BENCHMARK" != true && -n "$BENCHMARK_FILE" ]]; then
    echo "Multiple AI models were evaluated on questions derived from the research topic. Detailed performance metrics are available in the benchmark results." >> "$FINAL_REPORT"
else
    echo "Model benchmarking was skipped or incomplete." >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF

### Reasoning Quality
EOF

if [[ "$SKIP_REASONING" != true ]]; then
    echo "The $SPECIALTY perspective provided detailed reasoning analysis with $REASONING_ACCURACY accuracy. This includes comprehensive thought processes, confidence assessments, and meta-cognitive analysis." >> "$FINAL_REPORT"
else
    echo "Reasoning analysis was skipped." >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF

## Directory Structure
\`\`\`
research_pipeline/
├── literature_data/          # Downloaded papers and analysis
├── benchmark_results/        # Model evaluation results  
├── reasoning_analysis/       # Detailed reasoning traces
└── final_reports/           # Summary reports (you are here)
\`\`\`

## Key Files Generated
EOF

if [[ "$SKIP_DOWNLOAD" != true ]]; then
    echo "- \`literature_data/README.md\`: Literature analysis summary" >> "$FINAL_REPORT"
    echo "- \`literature_data/*/\`: Topic-organized paper collections" >> "$FINAL_REPORT"
fi

if [[ "$SKIP_BENCHMARK" != true && -n "$BENCHMARK_FILE" ]]; then
    echo "- \`benchmark_results/$BENCHMARK_FILE\`: Detailed model performance" >> "$FINAL_REPORT"
    echo "- \`benchmark_results/incorrect_*.json\`: Error analysis files" >> "$FINAL_REPORT"
fi

if [[ "$SKIP_REASONING" != true && -n "$REASONING_FILE" ]]; then
    echo "- \`reasoning_analysis/$REASONING_FILE\`: Detailed reasoning traces" >> "$FINAL_REPORT"
    echo "- \`reasoning_analysis/whole_trace_*.json\`: Meta-cognitive analysis" >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF
- \`$FINAL_REPORT\`: This comprehensive report

## Next Steps

### Research Validation
1. Review literature analysis in \`literature_data/README.md\`
2. Validate paper classification and topic coverage
3. Identify gaps in literature coverage

### Model Improvement
EOF

if [[ "$SKIP_BENCHMARK" != true ]]; then
    echo "1. Analyze model performance patterns in benchmark results" >> "$FINAL_REPORT"
    echo "2. Review incorrect answers for systematic weaknesses" >> "$FINAL_REPORT"
    echo "3. Use error analysis for targeted model improvements" >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF

### Reasoning Enhancement
EOF

if [[ "$SKIP_REASONING" != true ]]; then
    echo "1. Review detailed reasoning traces for quality assessment" >> "$FINAL_REPORT"
    echo "2. Analyze confidence calibration patterns" >> "$FINAL_REPORT"
    echo "3. Use meta-cognitive insights for reasoning improvements" >> "$FINAL_REPORT"
fi

cat >> "$FINAL_REPORT" << EOF

### Future Research
1. Expand literature coverage with additional keywords
2. Develop domain-specific evaluation questions
3. Compare reasoning quality across different expert perspectives
4. Integrate findings into systematic research workflows

## Reproduction
To reproduce this analysis:
\`\`\`bash
./scripts/complete_research_pipeline.sh \\
    --topic "$RESEARCH_TOPIC" \\
    --specialty "$SPECIALTY" \\
    --max-papers "$MAX_PAPERS" \\
    --sample-questions "$SAMPLE_QUESTIONS"
\`\`\`
EOF

echo "✓ Final report generated: $FINAL_REPORT"

cd ..

# Create pipeline summary
echo ""
echo "=========================================="
echo "COMPLETE RESEARCH PIPELINE FINISHED!"
echo "=========================================="
echo "Research topic: $RESEARCH_TOPIC"
echo "Total duration: ${PIPELINE_DURATION}s"
echo "Output directory: $(pwd)"
echo ""
echo "Phase Results:"
echo "  📚 Literature: $PAPERS_DOWNLOADED papers"
echo "  🤖 Models: $MODELS_TESTED tested, $SUCCESSFUL_RUNS successful"
echo "  🧠 Reasoning: $REASONING_ACCURACY accuracy ($SPECIALTY perspective)"
echo ""
echo "Key outputs:"
echo "  - Literature analysis and organization"
echo "  - Multi-model performance benchmarks"
echo "  - Detailed cognitive reasoning analysis"
echo "  - Comprehensive final report"
echo ""
echo "Final report: final_reports/$FINAL_REPORT"
echo ""
echo "Next steps:"
echo "  1. Review the final report for comprehensive insights"
echo "  2. Explore individual phase results for detailed analysis"
echo "  3. Use findings to guide further research or model development"
echo "=========================================="