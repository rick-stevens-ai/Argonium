#!/bin/bash

# TRACE_INCORRECT.sh - Analyze Only Incorrectly Answered Questions
# Usage: ./TRACE_INCORRECT.sh HR-GOOD-1-MC (without .json extension)
# Assumes input files: HR-GOOD-1-MC.json and HR-GOOD-1-MC-Score-incorrect.json exist

if [ -z "$1" ]; then
    echo "Usage: $0 <base_name>"
    echo "Example: $0 HR-GOOD-1-MC"
    echo "Assumes files exist: <base_name>.json and <base_name>-Score-incorrect.json"
    echo ""
    echo "This script will:"
    echo "  1. Extract incorrectly answered questions from the incorrect file"
    echo "  2. Create a temporary MC file with only those questions"
    echo "  3. Run comprehensive reasoning trace analysis on incorrect questions only"
    echo "  4. Generate detailed analysis in a dedicated directory"
    exit 1
fi

base_name=$1
original_file="${base_name}.json"
incorrect_file="${base_name}-Score-incorrect.json"
temp_incorrect_mc="${base_name}-INCORRECT-MC.json"
trace_output_file="${base_name}-INCORRECT-TRACE.json"
trace_output_dir="${base_name}_incorrect_trace_analysis"
whole_trace_output="${base_name}-INCORRECT-TRACE-whole.json"

echo "======================================="
echo "TRACE_INCORRECT.sh - Analyzing Failed Questions"
echo "======================================="

# Check if required files exist
if [ ! -f "$original_file" ]; then
    echo "❌ Error: Original MC file '$original_file' not found!"
    echo "   This file is needed to extract original question structure."
    exit 1
fi

if [ ! -f "$incorrect_file" ]; then
    echo "❌ Error: Incorrect answers file '$incorrect_file' not found!"
    echo "   Please run SCORE.sh first to generate the incorrect answers file."
    echo "   Make sure to use --save-incorrect flag in argonium_score_parallel_v9.py"
    exit 1
fi

# Check if the incorrect file has any content
incorrect_count=$(python3 -c "
import json
try:
    with open('$incorrect_file', 'r') as f:
        data = json.load(f)
    if 'incorrect_answers' in data:
        print(len(data['incorrect_answers']))
    else:
        print(0)
except Exception as e:
    print(0)
")

if [ "$incorrect_count" -eq 0 ]; then
    echo "🎉 No incorrect answers found in '$incorrect_file'!"
    echo "   All questions were answered correctly. No trace analysis needed."
    exit 0
fi

echo "📊 Found $incorrect_count incorrectly answered questions"
echo "📁 Original file: $original_file"
echo "📁 Incorrect file: $incorrect_file"
echo "📁 Temp MC file: $temp_incorrect_mc"
echo "📁 Output file: $trace_output_file"
echo "📁 Output directory: $trace_output_dir"

# Python script to extract incorrectly answered questions and create MC format
echo "🔄 Extracting incorrectly answered questions..."

python3 - "$original_file" "$incorrect_file" "$temp_incorrect_mc" << 'EOF'
import json
import sys

# Read the files
try:
    with open(sys.argv[1], 'r') as f:
        original_data = json.load(f)
    
    with open(sys.argv[2], 'r') as f:
        incorrect_data = json.load(f)
except Exception as e:
    print(f"❌ Error reading files: {e}")
    sys.exit(1)

# Extract question content that was answered incorrectly
incorrect_questions = {}
for item in incorrect_data.get('incorrect_answers', []):
    incorrect_questions[item['question']] = item['question_id']

print(f"📋 Extracting {len(incorrect_questions)} incorrect questions...")

# Filter original questions to include only the incorrect ones based on question content
filtered_questions = []
for i, question in enumerate(original_data):
    if question['question'] in incorrect_questions:
        # Create a proper question structure with id for the reasoning trace system
        filtered_question = question.copy()
        filtered_question['id'] = incorrect_questions[question['question']]
        filtered_questions.append(filtered_question)

# Save the filtered questions in MC format
try:
    with open(sys.argv[3], 'w') as f:
        json.dump(filtered_questions, f, indent=2)
    print(f"✅ Created temporary MC file with {len(filtered_questions)} questions")
except Exception as e:
    print(f"❌ Error writing filtered questions: {e}")
    sys.exit(1)

print(f"📝 Sample incorrect question IDs: {sorted(list(incorrect_questions.values()))[:10]}...")

EOF

# Check if Python script succeeded
if [ $? -ne 0 ]; then
    echo "❌ Failed to extract incorrect questions"
    exit 1
fi

# Create output directory
mkdir -p "$trace_output_dir"

echo ""
echo "🚀 Starting comprehensive reasoning trace analysis on incorrect questions..."
echo "⚙️  Configuration:"
echo "   - Model: gpt-4.1"
echo "   - Workers: 10"
echo "   - Features: All advanced features enabled"
echo "   - Specialty: expert"
echo "   - Mode: detailed reasoning"
echo ""

# Run reasoning_traces_parallel_v6.py with all features enabled on incorrect questions only
python reasoning_traces_parallel_v6.py "$temp_incorrect_mc" \
    --output "$trace_output_file" \
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
    --split-output \
    --output-dir "$trace_output_dir" \
    --create-stream-files

# Check if the reasoning analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================="
    echo "✅ REASONING TRACE ANALYSIS COMPLETED!"
    echo "======================================="
    echo ""
    echo "📈 Analysis Summary:"
    echo "   • Questions analyzed: $incorrect_count incorrect questions"
    echo "   • Original dataset: $base_name"
    echo "   • Focus: Questions that failed initial scoring"
    echo ""
    echo "📁 Generated Files:"
    echo "   • Main trace output: $trace_output_file"
    echo "   • Whole trace analysis: $whole_trace_output"
    echo "   • Detailed analysis directory: $trace_output_dir/"
    echo "   • Temporary MC file: $temp_incorrect_mc (can be deleted)"
    echo ""
    echo "📂 Analysis Directory Contents:"
    ls -la "$trace_output_dir" | head -10
    
    file_count=$(ls -1 "$trace_output_dir" | wc -l)
    if [ "$file_count" -gt 10 ]; then
        echo "   ... and $(( file_count - 10 )) more files"
    fi
    
    # Display metadata about the analysis
    echo ""
    echo "📊 Quick Stats:"
    echo "   • Original incorrect rate: $(python3 -c "
import json
with open('$incorrect_file', 'r') as f:
    data = json.load(f)
metadata = data.get('metadata', {})
rate = metadata.get('incorrect_rate', 0) * 100
total = metadata.get('total_processed', 0)
print(f'{rate:.1f}% ({incorrect_count}/{total})')
")"
    
    # Run LLM Formal Logic Analysis on the generated trace directory
    echo ""
    echo "======================================="
    echo "🧠 STARTING LLM FORMAL LOGIC ANALYSIS..."
    echo "======================================="
    echo ""
    echo "🔍 Analyzing logical argument structures in reasoning traces..."
    echo "⚙️  Configuration:"
    echo "   - Model: gpt-4.1"
    echo "   - Target: Stream analysis files in $trace_output_dir/"
    echo "   - Output: ${base_name}-INCORRECT-LOGIC-ANALYSIS.txt"
    echo ""
    
    logic_analysis_output="${base_name}-INCORRECT-LOGIC-ANALYSIS.txt"
    
    # Check if llm_formal_logic_analyzer.py exists
    if [ -f "llm_formal_logic_analyzer.py" ]; then
        # Run the LLM formal logic analysis
        python llm_formal_logic_analyzer.py "$trace_output_dir" \
            --output "$logic_analysis_output" \
            --format text \
            --model gpt-4.1 \
            --config argo_local.yaml
        
        # Check if logic analysis was successful
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ LLM FORMAL LOGIC ANALYSIS COMPLETED!"
            echo ""
            echo "📋 Logic Analysis Results:"
            echo "   • Analysis file: $logic_analysis_output"
            echo "   • Format: Human-readable text report"
            echo "   • Model used: gpt-4.1"
            echo ""
            
            # Display a quick preview of the logic analysis results
            if [ -f "$logic_analysis_output" ]; then
                echo "📊 Logic Analysis Preview:"
                echo "   $(head -20 "$logic_analysis_output" | grep -E '(Total|Arguments|Quality|Validity)' | head -5)"
            fi
        else
            echo "⚠️  LLM formal logic analysis encountered an error, but continuing..."
        fi
    else
        echo "⚠️  llm_formal_logic_analyzer.py not found - skipping logic analysis"
        echo "   To enable logic analysis, ensure the script is in the current directory"
    fi
    
    echo ""
    echo "======================================="
    echo "🎉 COMPLETE ANALYSIS FINISHED!"
    echo "======================================="
    echo ""
    echo "🎯 Next Steps:"
    echo "   • Review $whole_trace_output for overall reasoning patterns"
    echo "   • Examine $logic_analysis_output for logical argument analysis"
    echo "   • Explore individual question files in $trace_output_dir/"
    echo "   • Compare reasoning patterns with correct answers"
    echo "   • Optional: Remove temporary file with 'rm $temp_incorrect_mc'"
    echo ""
    echo "📁 Complete File Summary:"
    echo "   1. Reasoning Traces: $trace_output_file"
    echo "   2. Whole Trace Analysis: $whole_trace_output"
    echo "   3. Logic Analysis: $logic_analysis_output"
    echo "   4. Detailed Directory: $trace_output_dir/"
    echo "   5. Temporary MC File: $temp_incorrect_mc"
    
else
    echo ""
    echo "❌ ERROR: Reasoning trace analysis failed!"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   • Check that reasoning_traces_parallel_v6.py is available"
    echo "   • Verify argo_local.yaml configuration exists"
    echo "   • Ensure gpt-4.1 model is accessible"
    echo "   • Check temporary file: $temp_incorrect_mc"
    echo ""
    echo "🗑️  Cleaning up temporary file..."
    rm -f "$temp_incorrect_mc"
    exit 1
fi
