#!/bin/bash

# Reasoning Analysis Workflow Script
# Deep cognitive analysis of AI model reasoning processes

set -e  # Exit on any error

# Default values
QUESTIONS_FILE=""
MODEL="gpt41"
SPECIALTY="expert"
MAX_QUESTIONS=""
OUTPUT_DIR="./reasoning_analysis"
SAVE_INTERVAL=10
CONTINUE_FROM=""
WHOLE_TRACE=false
WHOLE_TRACE_MODEL=""
ENHANCED_DISCREPANCY=false
QUICK_ANALYSIS=false

# Function to display usage
usage() {
    echo "Reasoning Analysis Workflow"
    echo ""
    echo "Usage: $0 -q QUESTIONS_FILE [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -q, --questions FILE         Questions file for analysis (required)"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL           Model for reasoning analysis (default: gpt41)"
    echo "  -s, --specialty SPECIALTY   Expert specialty: microbiologist, physicist, historian, etc. (default: expert)"
    echo "  -n, --max-questions NUM     Maximum questions to analyze"
    echo "  -o, --output-dir DIR        Output directory (default: ./reasoning_analysis)"
    echo "  --save-interval NUM         Save progress every N questions (default: 10)"
    echo "  --continue-from FILE        Continue from previous analysis file"
    echo "  --whole-trace               Enable whole trace meta-analysis"
    echo "  --whole-trace-model MODEL   Model for whole trace analysis (default: same as main model)"
    echo "  --enhanced-discrepancy      Enable enhanced discrepancy analysis"
    echo "  --quick                     Quick analysis (max 20 questions, save every 5)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Specialty Options:"
    echo "  - microbiologist: Antibiotic resistance and bacterial pathogenesis expert"
    echo "  - physicist: Theoretical and computational physics expert"
    echo "  - quantum physicist: Quantum mechanics and computing expert"
    echo "  - historian: Historical analysis and documentation expert"
    echo "  - expert: Generic expert (default)"
    echo ""
    echo "Examples:"
    echo "  $0 -q questions.json -s microbiologist --whole-trace"
    echo "  $0 -q questions.json -m gpt41 -s physicist -n 50"
    echo "  $0 -q questions.json --continue-from previous_analysis.json"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--questions)
            QUESTIONS_FILE="$2"
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
        -n|--max-questions)
            MAX_QUESTIONS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --continue-from)
            CONTINUE_FROM="$2"
            shift 2
            ;;
        --whole-trace)
            WHOLE_TRACE=true
            shift
            ;;
        --whole-trace-model)
            WHOLE_TRACE_MODEL="$2"
            shift 2
            ;;
        --enhanced-discrepancy)
            ENHANCED_DISCREPANCY=true
            shift
            ;;
        --quick)
            QUICK_ANALYSIS=true
            MAX_QUESTIONS=20
            SAVE_INTERVAL=5
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

if [[ ! -f "$QUESTIONS_FILE" && -z "$CONTINUE_FROM" ]]; then
    echo "Error: Questions file '$QUESTIONS_FILE' not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================="
echo "Reasoning Analysis Workflow"
echo "=========================================="
echo "Questions file: $QUESTIONS_FILE"
echo "Model: $MODEL"
echo "Specialty: $SPECIALTY"
if [[ -n "$MAX_QUESTIONS" ]]; then
    echo "Max questions: $MAX_QUESTIONS"
fi
echo "Save interval: $SAVE_INTERVAL"
echo "Output directory: $(pwd)"
if [[ "$WHOLE_TRACE" == true ]]; then
    echo "Whole trace analysis: enabled"
    if [[ -n "$WHOLE_TRACE_MODEL" ]]; then
        echo "Whole trace model: $WHOLE_TRACE_MODEL"
    fi
fi
if [[ -n "$CONTINUE_FROM" ]]; then
    echo "Continuing from: $CONTINUE_FROM"
fi
echo ""

# Prepare command arguments
CMD_ARGS=("../$QUESTIONS_FILE")
CMD_ARGS+=("--model" "$MODEL")
CMD_ARGS+=("--specialty" "$SPECIALTY")
CMD_ARGS+=("--save-interval" "$SAVE_INTERVAL")

if [[ -n "$MAX_QUESTIONS" ]]; then
    CMD_ARGS+=("--max-questions" "$MAX_QUESTIONS")
fi

if [[ -n "$CONTINUE_FROM" ]]; then
    CMD_ARGS+=("--continue-from" "$CONTINUE_FROM")
fi

if [[ "$WHOLE_TRACE" == true ]]; then
    CMD_ARGS+=("--whole-trace-analysis")
    if [[ -n "$WHOLE_TRACE_MODEL" ]]; then
        CMD_ARGS+=("--whole-trace-model" "$WHOLE_TRACE_MODEL")
    fi
fi

if [[ "$ENHANCED_DISCREPANCY" == true ]]; then
    CMD_ARGS+=("--enhanced-discrepancy")
fi

# Generate output filename based on parameters
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="reasoning_traces_${SPECIALTY}_${TIMESTAMP}.json"
CMD_ARGS+=("--output" "$OUTPUT_FILE")

if [[ "$WHOLE_TRACE" == true ]]; then
    WHOLE_TRACE_OUTPUT="whole_trace_analysis_${SPECIALTY}_${TIMESTAMP}.json"
    CMD_ARGS+=("--whole-trace-output" "$WHOLE_TRACE_OUTPUT")
fi

# Step 1: Run reasoning analysis
echo "Step 1: Generating detailed reasoning traces..."
echo "----------------------------------------"
echo "Command: python ../reasoning_traces_v6.py ${CMD_ARGS[*]}"
echo ""

START_TIME=$(date +%s)
python ../reasoning_traces_v6.py "${CMD_ARGS[@]}"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✓ Reasoning analysis completed in ${DURATION}s"
echo ""

# Step 2: Generate summary statistics
echo "Step 2: Analyzing reasoning performance..."
echo "----------------------------------------"

if [[ -f "$OUTPUT_FILE" ]]; then
    # Extract accuracy statistics using Python
    STATS=$(python3 -c "
import json
import sys

try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    total = len(data)
    correct = sum(1 for trace in data if trace.get('prediction_correct', False))
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Count confidence levels
    confidence_counts = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
    for trace in data:
        conf = trace.get('reasoning', {}).get('prediction', {}).get('confidence_level', 'unknown').lower()
        if conf in confidence_counts:
            confidence_counts[conf] += 1
        else:
            confidence_counts['unknown'] += 1
    
    print(f'{total},{correct},{accuracy:.1f},{confidence_counts[\"high\"]},{confidence_counts[\"medium\"]},{confidence_counts[\"low\"]},{confidence_counts[\"unknown\"]}')
    
except Exception as e:
    print('0,0,0.0,0,0,0,0')
")

    IFS=',' read -r TOTAL_Q CORRECT_Q ACCURACY_PCT HIGH_CONF MED_CONF LOW_CONF UNK_CONF <<< "$STATS"
    
    echo "Performance Summary:"
    echo "  - Total questions analyzed: $TOTAL_Q"
    echo "  - Correct predictions: $CORRECT_Q"
    echo "  - Accuracy: $ACCURACY_PCT%"
    echo "  - High confidence: $HIGH_CONF"
    echo "  - Medium confidence: $MED_CONF"
    echo "  - Low confidence: $LOW_CONF"
    echo ""
    
    echo "✓ Performance analysis completed"
else
    echo "Warning: Output file $OUTPUT_FILE not found"
    TOTAL_Q=0
    ACCURACY_PCT="0.0"
fi

# Step 3: Process whole trace analysis (if enabled)
if [[ "$WHOLE_TRACE" == true && -f "$WHOLE_TRACE_OUTPUT" ]]; then
    echo ""
    echo "Step 3: Processing whole trace meta-analysis..."
    echo "----------------------------------------"
    
    # Extract meta-analysis insights
    echo "Whole trace analysis generated: $WHOLE_TRACE_OUTPUT"
    echo "✓ Meta-analysis completed"
fi

# Step 4: Generate reasoning samples for review
echo ""
echo "Step 4: Generating sample reasoning extracts..."
echo "----------------------------------------"

if [[ -f "$OUTPUT_FILE" ]]; then
    # Create a samples file with interesting cases
    SAMPLES_FILE="reasoning_samples_${SPECIALTY}_${TIMESTAMP}.json"
    
    python3 -c "
import json
import random

try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    # Select interesting samples
    samples = []
    
    # Get correct high-confidence predictions
    correct_high_conf = [trace for trace in data 
                        if trace.get('prediction_correct', False) and 
                        trace.get('reasoning', {}).get('prediction', {}).get('confidence_level', '').lower() == 'high']
    if correct_high_conf:
        samples.extend(random.sample(correct_high_conf, min(2, len(correct_high_conf))))
    
    # Get incorrect predictions for analysis
    incorrect = [trace for trace in data if not trace.get('prediction_correct', False)]
    if incorrect:
        samples.extend(random.sample(incorrect, min(2, len(incorrect))))
    
    # Get low-confidence predictions
    low_conf = [trace for trace in data 
               if trace.get('reasoning', {}).get('prediction', {}).get('confidence_level', '').lower() == 'low']
    if low_conf:
        samples.extend(random.sample(low_conf, min(1, len(low_conf))))
    
    # Save samples
    with open('$SAMPLES_FILE', 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f'Generated {len(samples)} reasoning samples')
    
except Exception as e:
    print(f'Error generating samples: {e}')
"
    
    echo "✓ Sample reasoning cases saved to: $SAMPLES_FILE"
fi

# Step 5: Generate markdown report
echo ""
echo "Step 5: Generating analysis report..."
echo "----------------------------------------"

REPORT_FILE="reasoning_analysis_report_${SPECIALTY}_${TIMESTAMP}.md"

cat > "$REPORT_FILE" << EOF
# Reasoning Analysis Report

Generated: $(date)
Specialty: **$SPECIALTY**
Model: **$MODEL**
Duration: **${DURATION}s**

## Configuration
- Questions file: $QUESTIONS_FILE
- Max questions: ${MAX_QUESTIONS:-"All"}
- Save interval: $SAVE_INTERVAL
- Whole trace analysis: ${WHOLE_TRACE}
- Enhanced discrepancy: ${ENHANCED_DISCREPANCY}

## Performance Summary
- **Total questions**: $TOTAL_Q
- **Accuracy**: $ACCURACY_PCT%
- **Correct predictions**: $CORRECT_Q
- **High confidence decisions**: $HIGH_CONF
- **Medium confidence decisions**: $MED_CONF  
- **Low confidence decisions**: $LOW_CONF

## Key Insights

### Reasoning Quality
The $SPECIALTY perspective was applied to analyze ${TOTAL_Q} questions, achieving $ACCURACY_PCT% accuracy.

### Confidence Calibration
- High confidence decisions: $HIGH_CONF
- Medium confidence decisions: $MED_CONF
- Low confidence decisions: $LOW_CONF

This distribution indicates the model's self-assessment capabilities when reasoning from a $SPECIALTY perspective.

## Files Generated
- \`$OUTPUT_FILE\`: Complete reasoning traces with detailed thought processes
EOF

if [[ "$WHOLE_TRACE" == true && -f "$WHOLE_TRACE_OUTPUT" ]]; then
    echo "- \`$WHOLE_TRACE_OUTPUT\`: Meta-cognitive analysis of reasoning patterns" >> "$REPORT_FILE"
fi

if [[ -f "$SAMPLES_FILE" ]]; then
    echo "- \`$SAMPLES_FILE\`: Sample reasoning cases for detailed review" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF
- \`$REPORT_FILE\`: This summary report

## Next Steps
1. **Review sample cases** in $SAMPLES_FILE for reasoning quality
2. **Analyze incorrect predictions** to identify improvement areas
3. **Compare confidence vs accuracy** to assess calibration
EOF

if [[ "$WHOLE_TRACE" == true ]]; then
    echo "4. **Review meta-analysis** in $WHOLE_TRACE_OUTPUT for systematic patterns" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## Usage for Further Analysis
\`\`\`bash
# Convert to different formats
python ../render_json.py $OUTPUT_FILE analysis_report.md

# Sample specific cases  
python ../sample_json.py $OUTPUT_FILE 10 > sample_cases.json

# Merge with other analyses
python ../merge_json.py $OUTPUT_FILE other_analysis.json > combined.json
\`\`\`
EOF

echo "✓ Analysis report generated: $REPORT_FILE"

# Final summary
echo ""
echo "=========================================="
echo "Reasoning Analysis Workflow Complete!"
echo "=========================================="
echo "Duration: ${DURATION}s"
echo "Output directory: $(pwd)"
echo ""
echo "Key files:"
echo "  - $OUTPUT_FILE (detailed reasoning traces)"
if [[ "$WHOLE_TRACE" == true && -f "$WHOLE_TRACE_OUTPUT" ]]; then
    echo "  - $WHOLE_TRACE_OUTPUT (meta-analysis)"
fi
if [[ -f "$SAMPLES_FILE" ]]; then
    echo "  - $SAMPLES_FILE (sample cases)"
fi
echo "  - $REPORT_FILE (summary report)"
echo ""
echo "Performance:"
echo "  - Questions analyzed: $TOTAL_Q"
echo "  - Accuracy: $ACCURACY_PCT%"
echo "  - Specialty perspective: $SPECIALTY"
echo ""
echo "Next steps:"
echo "  - Review detailed reasoning traces for quality assessment"
echo "  - Analyze confidence calibration patterns"
echo "  - Use insights for model improvement or validation"
echo "=========================================="