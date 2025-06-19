#!/usr/bin/env python3

"""
Multi-Model Benchmark Runner

Runs argonium_score_parallel_v9.py for each model in model_servers.yaml
and organizes the results for easy viewing.

Usage:
    python run_all_models.py <questions_file.json> [--grader MODEL] [--random N] [--parallel N] [--seed N] [--format auto|mc|qa] [--verbose] [--save-incorrect] [--skip-availability-check] [--availability-timeout N]

Example:
    python run_all_models.py FRG-200-MC7.json --grader gpt41 --random 50 --parallel 10 --seed 42 --format auto --verbose
"""

import os
import sys
import yaml
import subprocess
import json
import argparse
from datetime import datetime
import time
import openai

def load_models():
    """Load available models from model_servers.yaml"""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_servers.yaml")
    
    try:
        with open(yaml_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        
        models = [server['shortname'] for server in config['servers']]
        return models
    except Exception as e:
        print(f"Error loading model_servers.yaml: {e}")
        sys.exit(1)

def load_model_config(model_shortname):
    """
    Load model configuration from the model_servers.yaml file.
    Returns a dictionary with api_key, api_base, and model_name.
    """
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_servers.yaml")
    
    try:
        with open(yaml_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            
        # Look for the model by shortname
        for server in config['servers']:
            if server['shortname'] == model_shortname:
                api_key = server['openai_api_key']
                # Handle environment variable in api key if present
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    api_key = os.environ.get(env_var, "")
                    if not api_key:
                        print(f"Error: Environment variable {env_var} not set for model {model_shortname}")
                        return None
                
                return {
                    'api_key': api_key,
                    'api_base': server['openai_api_base'],
                    'model_name': server['openai_model']
                }
                
        # If not found
        print(f"Error: Model '{model_shortname}' not found in model_servers.yaml")
        return None
        
    except Exception as e:
        print(f"Error loading model configuration for {model_shortname}: {e}")
        return None

def check_model_availability(model_shortname, timeout=10):
    """
    Quick check to see if a model is available by making a simple API call.
    Returns True if available, False otherwise.
    """
    print(f"Checking availability of model: {model_shortname}...", end=" ")
    
    config = load_model_config(model_shortname)
    if not config:
        print("✗ (config error)")
        return False
    
    try:
        # Create a client for the model
        client = openai.OpenAI(
            api_key=config['api_key'],
            base_url=config['api_base'],
            timeout=timeout
        )
        
        # Check if this is a reasoning model (o3, o4mini) that has parameter restrictions
        is_reasoning_model = any(name in config['model_name'].lower() for name in ["o3", "o4-mini", "o4mini"])
        
        # Prepare minimal test message - use only the most basic parameters
        if is_reasoning_model:
            # For reasoning models, use only required parameters
            params = {
                "model": config['model_name'],
                "messages": [{"role": "user", "content": "Hi"}]
            }
        else:
            # For regular models, include max_tokens and temperature for better control
            params = {
                "model": config['model_name'],
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "temperature": 0.0
            }
        
        # Make a minimal API call
        response = client.chat.completions.create(**params)
        
        # If we get here without exception, the model is available
        print("✓")
        return True
        
    except openai.AuthenticationError:
        print("✗ (auth error)")
        return False
    except openai.NotFoundError:
        print("✗ (model not found)")
        return False
    except openai.RateLimitError:
        print("✓ (rate limited but available)")
        return True  # Rate limit means the model exists but is busy
    except openai.BadRequestError as e:
        # Check if it's an unsupported parameter error but model exists
        error_msg = str(e).lower()
        if "unsupported" in error_msg or "not supported" in error_msg:
            print("✓ (parameter restrictions but available)")
            return True
        else:
            print(f"✗ (bad request: {str(e)[:30]}...)")
            return False
    except Exception as e:
        print(f"✗ ({str(e)[:50]}...)")
        return False

def run_model_test(model, questions_file, grader, random_count, parallel_count, seed=None, format_type=None, verbose=False, save_incorrect=False):
    """Run the benchmark for a single model"""
    print(f"\n{'='*60}")
    print(f"Running benchmark for model: {model}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "argonium_score_parallel_v9.py",
        questions_file,
        "--model", model,
        "--grader", grader,
        "--random", str(random_count),
        "--parallel", str(parallel_count)
    ]
    
    # Add seed if provided
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    # Add format if provided
    if format_type is not None:
        cmd.extend(["--format", format_type])
    
    # Add verbose if requested
    if verbose:
        cmd.append("--verbose")
    
    # Add save-incorrect if requested
    if save_incorrect:
        cmd.append("--save-incorrect")
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ {model} completed successfully in {end_time - start_time:.1f}s")
            return {
                'model': model,
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'runtime_seconds': end_time - start_time,
                'returncode': result.returncode
            }
        else:
            print(f"✗ {model} failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {
                'model': model,
                'success': False,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'runtime_seconds': end_time - start_time,
                'returncode': result.returncode
            }
    except subprocess.TimeoutExpired:
        print(f"✗ {model} timed out after 1 hour")
        return {
            'model': model,
            'success': False,
            'stdout': '',
            'stderr': 'Process timed out after 1 hour',
            'runtime_seconds': 3600,
            'returncode': -1
        }
    except Exception as e:
        print(f"✗ {model} failed with exception: {e}")
        return {
            'model': model,
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'runtime_seconds': 0,
            'returncode': -1
        }

def extract_accuracy_from_output(stdout):
    """Extract accuracy percentage from the output"""
    lines = stdout.split('\n')
    for line in lines:
        # Look for "Overall accuracy: 88.00% (44.0/50)"
        if 'Overall accuracy:' in line:
            try:
                accuracy_part = line.split('Overall accuracy:')[1].strip()
                percentage = accuracy_part.split('%')[0].strip()
                return float(percentage)
            except:
                return None
        # Also look for "Multiple-choice questions: 88.00% accuracy (44/50)"
        elif 'Multiple-choice questions:' in line and 'accuracy' in line:
            try:
                # Parse "Multiple-choice questions: 88.00% accuracy (44/50)"
                parts = line.split('%')[0]
                percentage = parts.split()[-1]  # Get the last part before %
                return float(percentage)
            except:
                return None
    return None

def extract_confidence_from_output(stdout):
    """Extract average confidence from the output"""
    lines = stdout.split('\n')
    for line in lines:
        # Look for "Average MC confidence: 1.00"
        if 'Average MC confidence:' in line:
            try:
                confidence_part = line.split('Average MC confidence:')[1].strip()
                return float(confidence_part)
            except:
                return None
        # Also look for original format
        elif 'Average Confidence:' in line:
            try:
                confidence_part = line.split('Average Confidence:')[1].strip()
                return float(confidence_part)
            except:
                return None
    return None

def create_summary_report(results, questions_file, grader, random_count, parallel_count, seed=None, format_type=None, verbose=False, save_incorrect=False, unavailable_models=None):
    """Create a summary report of all results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"benchmark_summary_{timestamp}.json"
    
    # Extract key metrics
    summary_data = []
    for result in results:
        if result['success']:
            accuracy = extract_accuracy_from_output(result['stdout'])
            confidence = extract_confidence_from_output(result['stdout'])
        else:
            accuracy = None
            confidence = None
        
        summary_data.append({
            'model': result['model'],
            'success': result['success'],
            'accuracy_percent': accuracy,
            'average_confidence': confidence,
            'runtime_seconds': result['runtime_seconds'],
            'error': result['stderr'] if not result['success'] else None
        })
    
    # Sort by accuracy (successful runs first, then by accuracy descending)
    summary_data.sort(key=lambda x: (x['success'], x['accuracy_percent'] or -1), reverse=True)
    
    # Create full report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'questions_file': questions_file,
            'grader_model': grader,
            'random_sample_size': random_count,
            'parallel_workers': parallel_count,
            'random_seed': seed,
            'format_type': format_type,
            'verbose': verbose,
            'save_incorrect': save_incorrect,
            'total_models_tested': len(results),
            'successful_runs': sum(1 for r in results if r['success']),
            'failed_runs': sum(1 for r in results if not r['success']),
            'unavailable_models': unavailable_models or []
        },
        'summary': summary_data,
        'detailed_results': results
    }
    
    # Save to file
    with open(summary_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return summary_file, summary_data

def print_summary_table(summary_data):
    """Print a nice summary table to console"""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<12} {'Status':<8} {'Accuracy':<10} {'Confidence':<11} {'Runtime':<8}")
    print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*11} {'-'*8}")
    
    for data in summary_data:
        model = data['model']
        status = "✓ PASS" if data['success'] else "✗ FAIL"
        accuracy = f"{data['accuracy_percent']:.1f}%" if data['accuracy_percent'] is not None else "N/A"
        confidence = f"{data['average_confidence']:.2f}" if data['average_confidence'] is not None else "N/A"
        runtime = f"{data['runtime_seconds']:.1f}s"
        
        print(f"{model:<12} {status:<8} {accuracy:<10} {confidence:<11} {runtime:<8}")
    
    print(f"{'='*80}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run benchmarks for all models')
    parser.add_argument('questions_file', 
                       help='Questions file to use (e.g., FRG-200-MC7.json)')
    parser.add_argument('--grader', default='gpt41',
                       help='Grader model to use (default: gpt41)')
    parser.add_argument('--random', type=int, default=50,
                       help='Number of random questions to sample (default: 50)')
    parser.add_argument('--parallel', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    parser.add_argument('--exclude', nargs='*', default=[],
                       help='Models to exclude from testing')
    parser.add_argument('--include', nargs='*', default=[],
                       help='Only test these specific models')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible question selection')
    parser.add_argument('--format', choices=['auto', 'mc', 'qa'], default=None,
                       help='Format of questions: auto=auto-detect, mc=multiple-choice, qa=free-form QA (default: auto)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output for each model')
    parser.add_argument('--save-incorrect', action='store_true',
                       help='Save incorrectly answered questions to separate JSON files')
    parser.add_argument('--skip-availability-check', action='store_true',
                       help='Skip checking model availability before running tests')
    parser.add_argument('--availability-timeout', type=int, default=10,
                       help='Timeout in seconds for availability checks (default: 10)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Load available models
    all_models = load_models()
    
    # Filter models based on include/exclude
    if args.include:
        models_to_test = [m for m in args.include if m in all_models]
        if not models_to_test:
            print(f"Error: None of the specified models {args.include} were found")
            print(f"Available models: {all_models}")
            sys.exit(1)
    else:
        models_to_test = [m for m in all_models if m not in args.exclude]
    
    print(f"Models to test: {models_to_test}")
    print(f"Questions file: {args.questions_file}")
    print(f"Grader: {args.grader}")
    print(f"Random sample: {args.random}")
    print(f"Parallel workers: {args.parallel}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    if args.format is not None:
        print(f"Format: {args.format}")
    if args.verbose:
        print(f"Verbose output: enabled")
    if args.save_incorrect:
        print(f"Save incorrect answers: enabled")
    
    # Check if questions file exists
    if not os.path.exists(args.questions_file):
        print(f"Error: Questions file '{args.questions_file}' not found")
        sys.exit(1)
    
    # Check model availability before running tests
    available_models = []
    unavailable_models = []
    
    if not args.skip_availability_check:
        print(f"\n{'='*60}")
        print("CHECKING MODEL AVAILABILITY")
        print(f"{'='*60}")
        
        # Check grader model first
        if not check_model_availability(args.grader, args.availability_timeout):
            print(f"Error: Grader model '{args.grader}' is not available")
            sys.exit(1)
        
        # Check each test model
        for model in models_to_test:
            if check_model_availability(model, args.availability_timeout):
                available_models.append(model)
            else:
                unavailable_models.append(model)
        
        if unavailable_models:
            print(f"\nWarning: {len(unavailable_models)} model(s) are unavailable and will be skipped:")
            for model in unavailable_models:
                print(f"  - {model}")
        
        if not available_models:
            print("Error: No models are available for testing")
            sys.exit(1)
        
        models_to_test = available_models
        print(f"\nProceeding with {len(models_to_test)} available model(s): {models_to_test}")
    else:
        print("Skipping availability check as requested")
    
    # Run benchmarks
    results = []
    total_start_time = time.time()
    
    for i, model in enumerate(models_to_test, 1):
        print(f"\nProgress: {i}/{len(models_to_test)} models")
        result = run_model_test(model, args.questions_file, args.grader, args.random, args.parallel, 
                               args.seed, args.format, args.verbose, args.save_incorrect)
        results.append(result)
    
    total_end_time = time.time()
    
    # Create summary
    summary_file, summary_data = create_summary_report(results, args.questions_file, args.grader, args.random, args.parallel, 
                                                       args.seed, args.format, args.verbose, args.save_incorrect, unavailable_models)
    
    # Print results
    print_summary_table(summary_data)
    
    print(f"\nTotal benchmark time: {total_end_time - total_start_time:.1f}s")
    print(f"Detailed results saved to: {summary_file}")

if __name__ == "__main__":
    main()