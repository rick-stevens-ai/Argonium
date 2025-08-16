#!/usr/bin/env python3
"""
Fast Analyze Incorrect Answers with Formal Logic (Non-hanging version)

This script examines the output directory from reasoning_traces_parallel_v6.py,
identifies questions that were answered incorrectly by the models, and optionally
applies llm_formal_logic_analyzer.py to analyze a limited subset.

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
    """Configure the necessary APIs based on model selection."""
    try:
        with open(config_file, "r") as f:
            servers_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {config_file}: {e}")
        sys.exit(1)
    
    selected_server = None
    for server in servers_config["servers"]:
        if server["shortname"] == model_name:
            selected_server = server
            break
    
    if not selected_server:
        print(f"Error: Model '{model_name}' not found in {config_file}")
        sys.exit(1)
    
    api_key = selected_server.get("openai_api_key", "dummy_key_not_used")
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            print(f"Error: Environment variable {env_var} not set")
            sys.exit(1)
    
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

def is_answer_incorrect(fault_data: Dict[str, any]) -> Tuple[bool, str]:
    """Determine if either method gave an incorrect answer"""
    if not fault_data:
        return False, "No fault data"
    
    correct = fault_data.get('correct_answer')
    method1 = fault_data.get('method1_prediction')
    method2 = fault_data.get('method2_prediction')
    
    if not correct:
        return False, "No correct answer found"
    
    incorrect_methods = []
    
    if method1 and str(method1) != str(correct):
        incorrect_methods.append("method1")
    
    if method2 and str(method2) != str(correct):
        incorrect_methods.append("method2")
    
    if incorrect_methods:
        return True, f"Incorrect: {', '.join(incorrect_methods)}"
    
    return False, "Both methods correct"

def find_incorrect_questions(output_dir: str) -> List[Dict[str, str]]:
    """Find all questions that were answered incorrectly"""
    
    incorrect_questions = []
    fault_files = list(Path(output_dir).glob("*_fault_analysis.txt"))
    
    print(f"Found {len(fault_files)} fault analysis files")
    
    for fault_file in tqdm(fault_files, desc="Analyzing files", unit="file"):
        fault_data = parse_fault_analysis(str(fault_file))
        if not fault_data:
            continue
            
        is_incorrect, reason = is_answer_incorrect(fault_data)
        
        if is_incorrect:
            base_name = fault_file.stem.replace('_fault_analysis', '')
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
                print(f"⚠ Missing STREAM file for {base_name}")
    
    return incorrect_questions

def generate_basic_report(incorrect_questions: List[Dict[str, str]], output_file: str) -> None:
    """Generate a basic analysis report without LLM processing"""
    
    method1_errors = len([q for q in incorrect_questions if 'method1' in q['reason']])
    method2_errors = len([q for q in incorrect_questions if 'method2' in q['reason']])
    both_errors = len([q for q in incorrect_questions if 'method1' in q['reason'] and 'method2' in q['reason']])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("INCORRECT ANSWERS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Questions with Errors: {len(incorrect_questions)}\n")
        f.write(f"Method 1 Errors: {method1_errors}\n")
        f.write(f"Method 2 Errors: {method2_errors}\n")
        f.write(f"Both Methods Error: {both_errors}\n")
        f.write(f"Method 1 Only Errors: {method1_errors - both_errors}\n")
        f.write(f"Method 2 Only Errors: {method2_errors - both_errors}\n\n")
        
        f.write("ERROR BREAKDOWN BY QUESTION:\n")
        f.write("-" * 40 + "\n")
        for i, q in enumerate(incorrect_questions, 1):
            f.write(f"{i:3d}. {q['base_name']}\n")
            f.write(f"     Error Type: {q['reason']}\n")
            f.write(f"     Correct Answer: {q['correct_answer']}\n")
            f.write(f"     Method 1 Predicted: {q['method1_prediction']}\n")
            f.write(f"     Method 2 Predicted: {q['method2_prediction']}\n\n")
        
        # Group by error patterns
        f.write("ERROR PATTERNS:\n")
        f.write("-" * 40 + "\n")
        
        # Count different types of errors
        method1_only = [q for q in incorrect_questions if 'method1' in q['reason'] and 'method2' not in q['reason']]
        method2_only = [q for q in incorrect_questions if 'method2' in q['reason'] and 'method1' not in q['reason']]
        both_methods = [q for q in incorrect_questions if 'method1' in q['reason'] and 'method2' in q['reason']]
        
        f.write(f"Method 1 Only Errors ({len(method1_only)}):\n")
        for q in method1_only[:10]:  # Show first 10
            f.write(f"  - {q['base_name']}\n")
        if len(method1_only) > 10:
            f.write(f"  ... and {len(method1_only) - 10} more\n")
        f.write("\n")
        
        f.write(f"Method 2 Only Errors ({len(method2_only)}):\n")
        for q in method2_only[:10]:
            f.write(f"  - {q['base_name']}\n")
        if len(method2_only) > 10:
            f.write(f"  ... and {len(method2_only) - 10} more\n")
        f.write("\n")
        
        f.write(f"Both Methods Error ({len(both_methods)}):\n")
        for q in both_methods[:10]:
            f.write(f"  - {q['base_name']}\n")
        if len(both_methods) > 10:
            f.write(f"  ... and {len(both_methods) - 10} more\n")
        f.write("\n")
        
        # Analysis insights
        f.write("BASIC INSIGHTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Method 1 Error Rate: {method1_errors/len(incorrect_questions)*100:.1f}%\n")
        f.write(f"• Method 2 Error Rate: {method2_errors/len(incorrect_questions)*100:.1f}%\n")
        f.write(f"• Agreement in Errors: {both_errors/len(incorrect_questions)*100:.1f}%\n")
        
        if method1_errors > method2_errors:
            f.write("• Method 1 appears more error-prone than Method 2\n")
        elif method2_errors > method1_errors:
            f.write("• Method 2 appears more error-prone than Method 1\n")
        else:
            f.write("• Both methods have similar error rates\n")
        
        high_agreement = both_errors / min(method1_errors, method2_errors) if min(method1_errors, method2_errors) > 0 else 0
        f.write(f"• Error Agreement Rate: {high_agreement*100:.1f}% (when errors occur)\n")

def run_limited_formal_logic_analysis(stream_files: List[str], model_name: str, config_file: str, 
                                    output_file: str, max_files: int = 5) -> bool:
    """Run formal logic analysis on a limited subset of files"""
    
    if not stream_files:
        print("No stream files to analyze")
        return False
    
    # Limit the number of files
    limited_files = stream_files[:max_files]
    print(f"Limiting formal logic analysis to {len(limited_files)} files (from {len(stream_files)} total)")
    
    # Create temporary directory with symlinks
    temp_dir = Path("temp_limited_analysis")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        for stream_file in limited_files:
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
        
        print(f"Running formal logic analysis on {len(limited_files)} files...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Limited formal logic analysis completed successfully")
            return True
        else:
            print(f"❌ Formal logic analysis failed:")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Formal logic analysis timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running formal logic analysis: {e}")
        return False
    finally:
        # Clean up
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                if file.is_symlink():
                    file.unlink()
            temp_dir.rmdir()

def main():
    parser = argparse.ArgumentParser(
        description="Fast analysis of incorrect answers (non-hanging version)"
    )
    parser.add_argument(
        "output_dir", 
        help="Directory containing reasoning traces output files"
    )
    parser.add_argument(
        "--model", "-m", 
        default="gpt-4.1",
        help="Model shortname (default: gpt-4.1)"
    )
    parser.add_argument(
        "--config", "-c",
        default="argo_local.yaml", 
        help="Model configuration file (default: argo_local.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        default="fast_incorrect_analysis.txt",
        help="Output file for basic analysis (default: fast_incorrect_analysis.txt)"
    )
    parser.add_argument(
        "--logic-analysis", "-l",
        help="Enable formal logic analysis on limited subset (specify output file)"
    )
    parser.add_argument(
        "--logic-limit", "-n",
        type=int,
        default=3,
        help="Max files for formal logic analysis (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    print(f"📁 Analyzing output directory: {args.output_dir}")
    
    # Find incorrect questions
    print("\n🔍 Finding incorrectly answered questions...")
    incorrect_questions = find_incorrect_questions(args.output_dir)
    
    if not incorrect_questions:
        print("✅ No incorrectly answered questions found!")
        return
    
    print(f"\n📊 Found {len(incorrect_questions)} incorrectly answered questions")
    
    # Show breakdown
    method1_errors = len([q for q in incorrect_questions if 'method1' in q['reason']])
    method2_errors = len([q for q in incorrect_questions if 'method2' in q['reason']])
    both_errors = len([q for q in incorrect_questions if 'method1' in q['reason'] and 'method2' in q['reason']])
    
    print(f"   • Method 1 errors: {method1_errors}")
    print(f"   • Method 2 errors: {method2_errors}") 
    print(f"   • Both methods error: {both_errors}")
    
    # Generate basic report
    print(f"\n📝 Generating basic analysis report...")
    generate_basic_report(incorrect_questions, args.output)
    print(f"✅ Basic report saved to: {args.output}")
    
    # Optional formal logic analysis on limited subset
    if args.logic_analysis:
        print(f"\n⚙️  Running limited formal logic analysis...")
        
        # Configure APIs
        try:
            actual_model_name, client = configure_apis(args.model, args.config)
            print(f"✅ Using model: {actual_model_name}")
        except Exception as e:
            print(f"❌ Failed to configure model: {e}")
            return
        
        stream_files = [q['stream_file'] for q in incorrect_questions]
        success = run_limited_formal_logic_analysis(
            stream_files, 
            args.model, 
            args.config,
            args.logic_analysis,
            args.logic_limit
        )
        
        if success:
            print(f"✅ Limited formal logic analysis saved to: {args.logic_analysis}")
        else:
            print(f"⚠️  Formal logic analysis failed")
    
    print(f"\n🎉 Analysis complete!")
    print(f"📄 Basic analysis: {args.output}")
    if args.logic_analysis:
        print(f"📄 Logic analysis: {args.logic_analysis}")
    print(f"📈 Total incorrect questions: {len(incorrect_questions)}")

if __name__ == "__main__":
    main()