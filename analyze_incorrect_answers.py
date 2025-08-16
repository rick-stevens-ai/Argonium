#!/usr/bin/env python3
"""
Analyze Incorrect Answers with Formal Logic

This script examines the output directory from reasoning_traces_parallel_v6.py,
identifies questions that were answered incorrectly by the models, and applies
llm_formal_logic_analyzer.py to analyze the logical structure of incorrect reasoning.

Uses the same --model and --config structure as make_v22.py.
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from openai import OpenAI
from tqdm import tqdm

def configure_apis(model_name: str, config_file: str = "model_servers.yaml") -> tuple:
    """
    Configure the necessary APIs based on model selection.
    
    Args:
        model_name: The model shortname to use
        config_file: Path to the model configuration file
    
    Returns:
        Tuple of (actual_model_name, openai_client)
    """
    # Load the servers configuration
    try:
        with open(config_file, "r") as f:
            servers_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {config_file}: {e}")
        sys.exit(1)
    
    # Find the selected model's configuration
    selected_server = None
    for server in servers_config["servers"]:
        if server["shortname"] == model_name:
            selected_server = server
            break
    
    if not selected_server:
        print(f"Error: Model '{model_name}' not found in {config_file}")
        print(f"Available models: {', '.join(s['shortname'] for s in servers_config['servers'])}")
        sys.exit(1)
    
    # Configure OpenAI API with server details
    api_key = selected_server.get("openai_api_key", "dummy_key_not_used")
    # Handle environment variables in the API key
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            print(f"Error: Environment variable {env_var} not set")
            sys.exit(1)
    
    # Initialize the OpenAI client
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=selected_server.get("openai_api_base")
        )
    except ImportError:
        print("Error: OpenAI library not found")
        sys.exit(1)
    
    actual_model_name = selected_server.get("openai_model", model_name)
    return actual_model_name, client

def parse_fault_analysis(fault_file: str) -> Dict[str, any]:
    """Parse a fault analysis file to extract answer information"""
    try:
        with open(fault_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        result = {
            'has_disagreement': False,
            'correct_answer': None,
            'method1_prediction': None,
            'method2_prediction': None,
            'disagreement_type': 'none'
        }
        
        # Parse line by line
        for line in lines:
            line = line.strip()
            
            if line.startswith('Correct Answer:'):
                result['correct_answer'] = line.replace('Correct Answer:', '').strip()
            elif line.startswith('Method 1 Prediction:'):
                result['method1_prediction'] = line.replace('Method 1 Prediction:', '').strip()
            elif line.startswith('Method 2 Prediction:'):
                result['method2_prediction'] = line.replace('Method 2 Prediction:', '').strip()
            elif line.startswith('Has Disagreement:'):
                disagreement_value = line.replace('Has Disagreement:', '').strip().lower()
                result['has_disagreement'] = disagreement_value == 'true'
            elif line.startswith('Disagreement Type:'):
                result['disagreement_type'] = line.replace('Disagreement Type:', '').strip()
        
        return result
        
    except Exception as e:
        print(f"Error parsing fault analysis file {fault_file}: {e}")
        return None

def extract_option_number(answer_text: str) -> str:
    """Extract option number from answer text like 'Some text (Option 2)' -> '2'"""
    import re
    if not answer_text:
        return ""
    
    # Look for pattern like "(Option X)" at the end
    match = re.search(r'\(Option\s+(\d+)\)', answer_text)
    if match:
        return match.group(1)
    
    # If no option pattern, return the text as-is (might already be just a number)
    return answer_text.strip()

def is_answer_incorrect(fault_data: Dict[str, any]) -> Tuple[bool, str]:
    """
    Determine if either method gave an incorrect answer
    
    Returns:
        (is_incorrect, reason)
    """
    if not fault_data:
        return False, "No fault data"
    
    correct = fault_data.get('correct_answer')
    method1 = fault_data.get('method1_prediction')
    method2 = fault_data.get('method2_prediction')
    
    if not correct:
        return False, "No correct answer found"
    
    # Extract option number from correct answer if it's in text format
    correct_option = extract_option_number(correct)
    
    incorrect_methods = []
    
    # Check method 1
    if method1 and str(method1).strip() != str(correct_option).strip():
        incorrect_methods.append("method1")
    
    # Check method 2  
    if method2 and str(method2).strip() != str(correct_option).strip():
        incorrect_methods.append("method2")
    
    if incorrect_methods:
        return True, f"Incorrect: {', '.join(incorrect_methods)}"
    
    return False, "Both methods correct"

def find_incorrect_questions(output_dir: str) -> List[Dict[str, str]]:
    """Find all questions that were answered incorrectly"""
    
    incorrect_questions = []
    
    # Find all fault analysis files
    fault_files = list(Path(output_dir).glob("*_fault_analysis.txt"))
    
    print(f"Found {len(fault_files)} fault analysis files")
    sys.stdout.flush()  # Ensure output is displayed immediately
    
    for fault_file in tqdm(fault_files, desc="Processing fault files", unit="file", 
                          leave=True, dynamic_ncols=True):
        # Parse the fault analysis
        fault_data = parse_fault_analysis(str(fault_file))
        if not fault_data:
            continue
            
        # Check if incorrect
        is_incorrect, reason = is_answer_incorrect(fault_data)
        
        if is_incorrect:
            # Extract base filename
            base_name = fault_file.stem.replace('_fault_analysis', '')
            
            # Find corresponding STREAM_ANALYSIS file
            stream_file = fault_file.parent / f"{base_name}_STREAM_ANALYSIS.txt"
            
            if stream_file.exists():
                incorrect_questions.append({
                    'base_name': base_name,
                    'fault_file': str(fault_file),
                    'stream_file': str(stream_file),
                    'reason': reason,
                    'correct_answer': fault_data.get('correct_answer'),
                    'method1_prediction': fault_data.get('method1_prediction'),
                    'method2_prediction': fault_data.get('method2_prediction')
                })
            else:
                tqdm.write(f"⚠ Missing STREAM file for {base_name}")
    
    # Print summary of incorrect questions found  
    print(f"\nFound {len(incorrect_questions)} incorrect questions:")
    for q in incorrect_questions[:10]:  # Show first 10
        print(f"✗ {q['base_name']}: {q['reason']}")
    if len(incorrect_questions) > 10:
        print(f"... and {len(incorrect_questions) - 10} more")
    
    return incorrect_questions

def run_formal_logic_analysis(stream_files: List[str], model_name: str, config_file: str, 
                            output_file: str) -> bool:
    """Run the formal logic analyzer on the stream files"""
    
    if not stream_files:
        print("No stream files to analyze")
        return False
    
    # Create temporary directory with symlinks to stream files
    temp_dir = Path("temp_incorrect_analysis")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create symlinks to the stream files
        print("Setting up analysis files...")
        for stream_file in tqdm(stream_files, desc="Creating symlinks", unit="file", leave=False):
            src = Path(stream_file)
            dst = temp_dir / src.name
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.absolute())
        
        # Run the formal logic analyzer
        cmd = [
            "python", "llm_formal_logic_analyzer.py",
            str(temp_dir),
            "--output", output_file,
            "--model", model_name,
            "--config", config_file,
            "--format", "text"
        ]
        
        print(f"🚀 Starting formal logic analysis...")
        print(f"   Command: python llm_formal_logic_analyzer.py {temp_dir} --model {model_name} --config {config_file}")
        print(f"   Processing {len(stream_files)} files individually...")
        print(f"   Note: Progress will be shown by the formal logic analyzer itself")
        sys.stdout.flush()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("✅ Formal logic analysis completed successfully")
            return True
        else:
            print(f"❌ Formal logic analysis failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Formal logic analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Error running formal logic analysis: {e}")
        return False
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                if file.is_symlink():
                    file.unlink()
            temp_dir.rmdir()

def generate_summary_report(incorrect_questions: List[Dict[str, str]], 
                          logic_analysis_file: str,
                          summary_file: str,
                          model_name: str, client) -> None:
    """Generate a summary report using LLM analysis"""
    
    # Read the logic analysis results
    logic_content = ""
    if os.path.exists(logic_analysis_file):
        with open(logic_analysis_file, 'r', encoding='utf-8') as f:
            logic_content = f.read()
    
    # Prepare summary data
    summary_data = {
        'total_incorrect': len(incorrect_questions),
        'method1_errors': len([q for q in incorrect_questions if 'method1' in q['reason']]),
        'method2_errors': len([q for q in incorrect_questions if 'method2' in q['reason']]),
        'questions': incorrect_questions[:10]  # First 10 for analysis
    }
    
    # Create prompt for LLM summary
    prompt = f"""Analyze the following data about incorrectly answered questions and their logical reasoning patterns.

SUMMARY STATISTICS:
- Total questions answered incorrectly: {summary_data['total_incorrect']}
- Method 1 errors: {summary_data['method1_errors']}
- Method 2 errors: {summary_data['method2_errors']}

FORMAL LOGIC ANALYSIS RESULTS:
{logic_content[:3000] if logic_content else "No logic analysis available"}

SAMPLE INCORRECT QUESTIONS:
{json.dumps(summary_data['questions'], indent=2)[:2000]}

Please provide a comprehensive analysis covering:
1. Common patterns in logical reasoning errors
2. Types of formal logic structures found in incorrect answers
3. Comparison between Method 1 and Method 2 error patterns
4. Recommendations for improving reasoning quality

Keep the analysis concise but insightful."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        
        llm_summary = response.choices[0].message.content
        
        # Write the summary report
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("INCORRECT ANSWERS ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Questions Analyzed: {len(incorrect_questions)}\n")
            f.write(f"Method 1 Errors: {summary_data['method1_errors']}\n")
            f.write(f"Method 2 Errors: {summary_data['method2_errors']}\n")
            f.write(f"Both Methods Errors: {len([q for q in incorrect_questions if 'method1' in q['reason'] and 'method2' in q['reason']])}\n\n")
            
            f.write("LLM ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(llm_summary)
            f.write("\n\n")
            
            f.write("DETAILED QUESTION LIST:\n")
            f.write("-" * 40 + "\n")
            for i, q in enumerate(incorrect_questions, 1):
                f.write(f"{i:3d}. {q['base_name']}\n")
                f.write(f"     Reason: {q['reason']}\n")
                f.write(f"     Correct: {q['correct_answer']}\n")
                f.write(f"     Method1: {q['method1_prediction']}\n") 
                f.write(f"     Method2: {q['method2_prediction']}\n\n")
        
        print(f"✅ Summary report written to {summary_file}")
        
    except Exception as e:
        print(f"❌ Error generating LLM summary: {e}")
        # Write basic summary without LLM analysis
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("INCORRECT ANSWERS ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Questions Analyzed: {len(incorrect_questions)}\n")
            f.write(f"Method 1 Errors: {summary_data['method1_errors']}\n")
            f.write(f"Method 2 Errors: {summary_data['method2_errors']}\n\n")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze incorrect answers using formal logic analysis"
    )
    parser.add_argument(
        "output_dir", 
        help="Directory containing reasoning traces output files"
    )
    parser.add_argument(
        "--model", "-m", 
        default="gpt-4.1",
        help="Model shortname from configuration file (default: gpt-4.1)"
    )
    parser.add_argument(
        "--config", "-c",
        default="argo_local.yaml", 
        help="Model configuration file (default: argo_local.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        default="incorrect_answers_logic_analysis.txt",
        help="Output file for formal logic analysis (default: incorrect_answers_logic_analysis.txt)"
    )
    parser.add_argument(
        "--summary", "-s",
        default="incorrect_answers_summary.txt", 
        help="Output file for summary report (default: incorrect_answers_summary.txt)"
    )
    parser.add_argument(
        "--max-questions", "-n",
        type=int,
        default=5,
        help="Maximum number of incorrect questions to analyze (default: 5, use 0 for all)"
    )
    
    args = parser.parse_args()
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    # Configure APIs
    try:
        actual_model_name, client = configure_apis(args.model, args.config)
        print(f"✅ Using model: {actual_model_name} (shortname: {args.model})")
    except Exception as e:
        print(f"❌ Failed to configure model: {e}")
        sys.exit(1)
    
    print(f"📁 Analyzing output directory: {args.output_dir}")
    print("🔍 Finding incorrectly answered questions...")
    sys.stdout.flush()
    
    # Find incorrect questions
    incorrect_questions = find_incorrect_questions(args.output_dir)
    
    if not incorrect_questions:
        print("✅ No incorrectly answered questions found!")
        return
    
    print(f"\n📊 Found {len(incorrect_questions)} incorrectly answered questions")
    
    # Limit if requested (default is 5, use 0 for all)
    if args.max_questions > 0 and len(incorrect_questions) > args.max_questions:
        print(f"🔄 Limiting analysis to first {args.max_questions} questions (use --max-questions 0 for all)")
        incorrect_questions = incorrect_questions[:args.max_questions]
    elif args.max_questions == 0:
        print(f"🔄 Analyzing all {len(incorrect_questions)} questions (this may take a very long time)")
    else:
        print(f"🔄 Analyzing {len(incorrect_questions)} questions")
    
    # Extract stream files for analysis
    stream_files = [q['stream_file'] for q in incorrect_questions]
    
    estimated_time = len(stream_files) * 30  # Rough estimate: 30 seconds per file
    print(f"\n⚙️  Running formal logic analysis on {len(stream_files)} stream files...")
    print(f"⏱️  Estimated time: ~{estimated_time//60} minutes ({estimated_time} seconds)")
    
    # Run formal logic analysis
    success = run_formal_logic_analysis(
        stream_files, 
        args.model, 
        args.config,
        args.output
    )
    
    if success:
        print(f"✅ Formal logic analysis saved to: {args.output}")
    else:
        print(f"⚠️  Formal logic analysis failed, but continuing with summary...")
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    generate_summary_report(
        incorrect_questions,
        args.output if success else "",
        args.summary,
        actual_model_name,
        client
    )
    
    print(f"\n🎉 Analysis complete!")
    print(f"📄 Logic analysis: {args.output}")
    print(f"📄 Summary report: {args.summary}")
    print(f"📈 Total incorrect questions: {len(incorrect_questions)}")

if __name__ == "__main__":
    main()