#!/usr/bin/env python3
"""
clean_mcqa_v1.py - Clean existing MCQA JSON files using content relevance criteria

This script reads existing MCQA JSON files and applies the same content relevance 
and quality criteria used in Make_v21.py to filter out questions that don't meet 
the standards for core scientific/technical content.

Author: Generated for Argonium project
"""

import json
import re
import time
import argparse
import yaml
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

# Import OpenAI for LLM calls
try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed. Please install with: pip install openai")
    sys.exit(1)

# Global variables for progress tracking
_processed_questions = 0
_total_questions = 0
_filtered_questions = 0
_counter_lock = threading.Lock()
_start_time = None

@dataclass
class CleaningConfig:
    """Configuration for cleaning MCQA files"""
    input_file: str
    output_file: str
    model_name: str
    min_score: int
    relevance_threshold: int
    batch_size: int
    max_workers: int
    verbose: bool
    dry_run: bool
    filtered_output_file: Optional[str] = None

def log_message(message: str, log_level: str = "INFO", error_type: str = None):
    """Log a message with timestamp and level"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if error_type:
        print(f"[{timestamp}] {log_level} ({error_type}): {message}")
    else:
        print(f"[{timestamp}] {log_level}: {message}")

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log_message(f"Configuration file {config_file} not found", log_level="ERROR")
        return {}
    except yaml.YAMLError as e:
        log_message(f"Error parsing configuration file: {e}", log_level="ERROR")
        return {}

def configure_apis(model_name: str, config: dict) -> tuple:
    """Configure OpenAI API client based on model name and config"""
    if model_name.startswith("argo:"):
        # Argo AI configuration
        base_url = config.get('argo', {}).get('base_url', 'https://api.argo.ai/v1')
        api_key = config.get('argo', {}).get('api_key', os.getenv('ARGO_API_KEY'))
        actual_model = model_name.replace("argo:", "")
        
        if not api_key:
            raise ValueError("Argo AI API key not found in config or environment")
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        return client, actual_model
    
    elif model_name.startswith("openai:"):
        # OpenAI configuration
        api_key = config.get('openai', {}).get('api_key', os.getenv('OPENAI_API_KEY'))
        actual_model = model_name.replace("openai:", "")
        
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        client = OpenAI(api_key=api_key)
        return client, actual_model
    
    else:
        # Default to OpenAI
        api_key = config.get('openai', {}).get('api_key', os.getenv('OPENAI_API_KEY'))
        
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        client = OpenAI(api_key=api_key)
        return client, model_name

def batched_openai_completion(model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000, client: OpenAI = None) -> Dict:
    """Make a batched OpenAI completion request"""
    if client is None:
        raise ValueError("OpenAI client not provided")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.model_dump()
    except Exception as e:
        log_message(f"Error in OpenAI completion: {e}", log_level="ERROR", error_type="api_call")
        raise

def check_content_relevance(chunk_text: str, model_name: str, client: OpenAI) -> Dict:
    """
    Check if the chunk content is relevant to the paper's core content.
    Returns relevance score and reasoning.
    """
    system_message = (
        "You are an expert content evaluator who determines if text content is relevant "
        "to the core scientific/technical content of a paper versus non-relevant material "
        "like copyright notices, licensing information, references, acknowledgments, or metadata."
    )
    
    user_message = (
        f"Evaluate the following text chunk and determine if it contains core scientific/technical content "
        f"that would be appropriate for generating educational questions.\n\n"
        f"TEXT CHUNK:\n{chunk_text}\n\n"
        f"EVALUATION CRITERIA:\n"
        f"- CORE CONTENT (High relevance): Scientific concepts, research findings, technical explanations, "
        f"methodology, data analysis, theories, experimental results, clinical information, etc.\n"
        f"- NON-CORE CONTENT (Low relevance): Copyright notices, licensing text, reference lists, "
        f"acknowledgments, author information, publication metadata, figure/table captions only, "
        f"page headers/footers, disclaimers, etc.\n\n"
        f"SCORING:\n"
        f"- Score 8-10: Rich core content ideal for question generation\n"
        f"- Score 5-7: Some core content but mixed with non-relevant material\n"
        f"- Score 1-4: Primarily non-relevant content (references, metadata, etc.)\n\n"
        f"Provide your response in this format:\n"
        f"RELEVANCE_SCORE: <numeric score between 1-10>\n"
        f"REASONING: <brief explanation of why this content is or isn't relevant for question generation>\n"
        f"CONTENT_TYPE: <primary type of content: 'core_scientific', 'mixed', 'references', 'metadata', 'copyright', etc.>\n"
    )
    
    try:
        response = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            client=client
        )
        
        output = response['choices'][0]['message']['content'].strip()
        
        # Extract relevance score
        score_match = re.search(r"RELEVANCE_SCORE:\s*(\d+(?:\.\d+)?)", output)
        relevance_score = int(float(score_match.group(1))) if score_match else 5
        
        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?:\n|$)", output, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        # Extract content type
        content_type_match = re.search(r"CONTENT_TYPE:\s*(.*?)(?:\n|$)", output)
        content_type = content_type_match.group(1).strip() if content_type_match else "unknown"
        
        return {
            'relevance_score': relevance_score,
            'reasoning': reasoning,
            'content_type': content_type,
            'is_relevant': relevance_score >= 6,  # Threshold for relevance
            'raw_output': output
        }
        
    except Exception as e:
        log_message(f"Error checking content relevance: {e}", log_level="ERROR", error_type="relevance_check")
        return {
            'relevance_score': 5,  # Default to medium relevance on error
            'reasoning': f"Error during relevance check: {str(e)}",
            'content_type': 'unknown',
            'is_relevant': True,  # Default to relevant on error to avoid losing content
            'raw_output': ""
        }

def evaluate_question_quality(question_data: Dict, relevance_check: Dict, model_name: str, client: OpenAI) -> Dict:
    """
    Evaluate the quality of an existing MCQA question using the same criteria as Make_v21.py
    """
    question_text = question_data.get('question', '')
    chunk_text = question_data.get('text', '')
    
    system_message = (
        "You are an expert teacher evaluating the quality of a multiple choice question. "
        "Your role is to ensure questions are clear, fair, and educationally valuable."
    )
    
    user_message = (
        f"Evaluate the following multiple-choice question on a scale from 1-10, "
        f"where 10 is a perfect question.\n\n"
        f"ORIGINAL CONTENT:\n{chunk_text}\n\n"
        f"QUESTION:\n{question_text}\n\n"
        f"CONTENT RELEVANCE INFO:\n"
        f"- Relevance Score: {relevance_check['relevance_score']}/10\n"
        f"- Content Type: {relevance_check['content_type']}\n"
        f"- Relevance Reasoning: {relevance_check['reasoning']}\n\n"
        f"Rate the question based on these criteria:\n"
        f"- Clarity: Is the question clear and unambiguous?\n"
        f"- Accuracy: Is the content factually correct and aligned with the source material?\n"
        f"- Difficulty: Is the difficulty appropriate (challenging but fair)?\n"
        f"- Distractors: Are the incorrect options plausible but clearly wrong?\n"
        f"- Educational value: Does answering this question demonstrate understanding?\n"
        f"- Self-contained: CRITICAL - Does the question stand alone without ANY references to external materials?\n"
        f"- Content relevance: IMPORTANT - Questions based on low-relevance content (references, metadata, etc.) should receive lower scores\n\n"
        f"AUTOMATIC DISQUALIFIERS (score must be 1-3 if ANY are present):\n"
        f"- References to 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"- References to 'the author', 'according to', 'as mentioned', 'as described'\n"
        f"- References to 'Appendix', 'Figure', 'Table', 'Section', 'Chapter'\n"
        f"- References to 'above', 'below', 'previously mentioned', 'following'\n"
        f"- Any other references that assume the reader has access to external materials\n"
        f"- Content based primarily on references, copyright notices, or metadata (should score 1-4)\n\n"
        f"SCORING ADJUSTMENT FOR CONTENT RELEVANCE:\n"
        f"- If content relevance score is 1-4: Maximum question score should be 4\n"
        f"- If content relevance score is 5-7: Maximum question score should be 7\n"
        f"- If content relevance score is 8-10: Normal scoring applies\n\n"
        f"A truly self-contained question should read like a general knowledge question on the topic.\n\n"
        f"Provide your response in this format:\n"
        f"SCORE: <numeric score between 1-10>\n"
        f"CRITIQUE: <brief explanation of score>\n"
    )
    
    try:
        response = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            client=client
        )
        
        output = response['choices'][0]['message']['content'].strip()
        
        # Extract score
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", output)
        score = int(float(score_match.group(1))) if score_match else 0
        
        # Extract critique
        critique_match = re.search(r"CRITIQUE:(.*?)(?:\n\n|$)", output, re.DOTALL)
        critique = critique_match.group(1).strip() if critique_match else "No critique provided"
        
        return {
            'score': score,
            'critique': critique,
            'raw_output': output
        }
        
    except Exception as e:
        log_message(f"Error evaluating question quality: {e}", log_level="ERROR", error_type="question_eval")
        return {
            'score': 0,
            'critique': f"Error during evaluation: {str(e)}",
            'raw_output': ""
        }

def process_question(question_data: Dict, question_idx: int, config: CleaningConfig, client: OpenAI) -> Dict:
    """
    Process a single question for cleaning
    Returns a dictionary with 'result' (cleaned question or None) and 'filtered' (filtered question or None)
    """
    global _processed_questions, _filtered_questions
    
    try:
        # Get the text content for relevance checking
        chunk_text = question_data.get('text', '')
        
        if not chunk_text:
            log_message(f"Question {question_idx}: No text content found, skipping", log_level="WARNING")
            with _counter_lock:
                _processed_questions += 1
            return {'result': None, 'filtered': None}
        
        # Check content relevance
        relevance_check = check_content_relevance(chunk_text, config.model_name, client)
        
        # Filter based on relevance threshold
        if relevance_check['relevance_score'] < config.relevance_threshold:
            log_message(f"Question {question_idx}: Filtered out - relevance score {relevance_check['relevance_score']} < {config.relevance_threshold}", 
                       log_level="INFO")
            
            # Create filtered question record
            filtered_question = question_data.copy()
            filtered_question['filter_reason'] = 'low_relevance'
            filtered_question['filter_details'] = {
                'relevance_check': relevance_check,
                'threshold': config.relevance_threshold,
                'filtered_at': time.time(),
                'filter_version': '1.0'
            }
            
            with _counter_lock:
                _processed_questions += 1
                _filtered_questions += 1
            return {'result': None, 'filtered': filtered_question}
        
        # Evaluate question quality
        quality_check = evaluate_question_quality(question_data, relevance_check, config.model_name, client)
        
        # Filter based on minimum score
        if quality_check['score'] < config.min_score:
            log_message(f"Question {question_idx}: Filtered out - quality score {quality_check['score']} < {config.min_score}", 
                       log_level="INFO")
            
            # Create filtered question record
            filtered_question = question_data.copy()
            filtered_question['filter_reason'] = 'low_quality'
            filtered_question['filter_details'] = {
                'relevance_check': relevance_check,
                'quality_check': quality_check,
                'min_score_threshold': config.min_score,
                'filtered_at': time.time(),
                'filter_version': '1.0'
            }
            
            with _counter_lock:
                _processed_questions += 1
                _filtered_questions += 1
            return {'result': None, 'filtered': filtered_question}
        
        # Question passes all filters - add metadata
        cleaned_question = question_data.copy()
        cleaned_question['cleaning_metadata'] = {
            'relevance_check': relevance_check,
            'quality_check': quality_check,
            'cleaned_at': time.time(),
            'cleaning_version': '1.0'
        }
        
        if config.verbose:
            log_message(f"Question {question_idx}: Passed - relevance {relevance_check['relevance_score']}, quality {quality_check['score']}", 
                       log_level="INFO")
        
        with _counter_lock:
            _processed_questions += 1
        
        return {'result': cleaned_question, 'filtered': None}
        
    except Exception as e:
        log_message(f"Error processing question {question_idx}: {e}", log_level="ERROR", error_type="question_processing")
        with _counter_lock:
            _processed_questions += 1
        return {'result': None, 'filtered': None}

def update_progress():
    """Update progress display"""
    if _start_time is None:
        return
    
    elapsed = time.time() - _start_time
    rate = _processed_questions / elapsed if elapsed > 0 else 0
    eta = (_total_questions - _processed_questions) / rate if rate > 0 else 0
    
    progress_pct = (_processed_questions / _total_questions) * 100 if _total_questions > 0 else 0
    
    print(f"\rProgress: {_processed_questions}/{_total_questions} ({progress_pct:.1f}%) | "
          f"Filtered: {_filtered_questions} | Rate: {rate:.1f}/s | ETA: {eta:.0f}s", end="", flush=True)

def clean_mcqa_file(config: CleaningConfig) -> bool:
    """
    Clean an MCQA JSON file by filtering questions based on content relevance and quality
    """
    global _processed_questions, _total_questions, _filtered_questions, _start_time
    
    # Initialize counters
    _processed_questions = 0
    _filtered_questions = 0
    _start_time = time.time()
    
    log_message(f"Starting MCQA cleaning process")
    log_message(f"Input file: {config.input_file}")
    log_message(f"Output file: {config.output_file}")
    log_message(f"Model: {config.model_name}")
    log_message(f"Min quality score: {config.min_score}")
    log_message(f"Relevance threshold: {config.relevance_threshold}")
    log_message(f"Max workers: {config.max_workers}")
    
    # Load input file
    try:
        with open(config.input_file, 'r', encoding='utf-8') as f:
            mcqa_data = json.load(f)
    except Exception as e:
        log_message(f"Error loading input file: {e}", log_level="ERROR", error_type="file_loading")
        return False
    
    # Validate input format
    if not isinstance(mcqa_data, list):
        log_message("Input file must contain a list of questions", log_level="ERROR")
        return False
    
    _total_questions = len(mcqa_data)
    log_message(f"Loaded {_total_questions} questions from input file")
    
    if config.dry_run:
        log_message("DRY RUN MODE - No output file will be written")
    
    # Configure API client
    try:
        yaml_config = load_config(config.config_file) if hasattr(config, 'config_file') else {}
        client, actual_model = configure_apis(config.model_name, yaml_config)
        config.model_name = actual_model
    except Exception as e:
        log_message(f"Error configuring API client: {e}", log_level="ERROR", error_type="api_config")
        return False
    
    # Process questions in parallel
    cleaned_questions = []
    filtered_questions = []
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all questions for processing
        future_to_idx = {
            executor.submit(process_question, question_data, idx, config, client): idx
            for idx, question_data in enumerate(mcqa_data)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            
            try:
                result = future.result()
                if result['result'] is not None:
                    cleaned_questions.append(result['result'])
                if result['filtered'] is not None:
                    filtered_questions.append(result['filtered'])
            except Exception as e:
                log_message(f"Error processing question {idx}: {e}", log_level="ERROR", error_type="question_processing")
            
            # Update progress
            if not config.verbose:
                update_progress()
    
    print()  # New line after progress
    
    # Sort cleaned questions by original order if needed
    # (Since we're using threading, order might be mixed)
    
    # Final statistics
    original_count = len(mcqa_data)
    cleaned_count = len(cleaned_questions)
    filtered_count = len(filtered_questions)
    
    # Count filtered reasons
    relevance_filtered = sum(1 for q in filtered_questions if q.get('filter_reason') == 'low_relevance')
    quality_filtered = sum(1 for q in filtered_questions if q.get('filter_reason') == 'low_quality')
    
    log_message(f"Cleaning completed:")
    log_message(f"  Original questions: {original_count}")
    log_message(f"  Questions kept: {cleaned_count}")
    log_message(f"  Questions filtered: {filtered_count}")
    log_message(f"    - Filtered for low relevance: {relevance_filtered}")
    log_message(f"    - Filtered for low quality: {quality_filtered}")
    log_message(f"  Retention rate: {(cleaned_count/original_count)*100:.1f}%")
    
    # Save cleaned data
    if not config.dry_run:
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(config.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save cleaned questions
            with open(config.output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_questions, f, indent=2, ensure_ascii=False)
            
            log_message(f"Cleaned data saved to {config.output_file}")
            
            # Save filtered questions if requested and there are any
            if config.filtered_output_file and filtered_questions:
                filtered_dir = os.path.dirname(config.filtered_output_file)
                if filtered_dir and not os.path.exists(filtered_dir):
                    os.makedirs(filtered_dir)
                
                with open(config.filtered_output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_questions, f, indent=2, ensure_ascii=False)
                
                log_message(f"Filtered questions saved to {config.filtered_output_file}")
            elif config.filtered_output_file and not filtered_questions:
                log_message("No questions were filtered, skipping filtered output file")
            
        except Exception as e:
            log_message(f"Error saving output files: {e}", log_level="ERROR", error_type="file_saving")
            return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Clean existing MCQA JSON files using content relevance and quality criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean HR-1k-1k.json with default settings
  python clean_mcqa_v1.py -i HR-1k-1k.json -o HR-1k-1k-cleaned.json
  
  # Clean with custom model and thresholds, save filtered questions for inspection
  python clean_mcqa_v1.py -i input.json -o output.json --filtered-output filtered.json --model argo:gpt-4.1 --min-score 6 --relevance-threshold 5
  
  # Dry run to see what would be filtered
  python clean_mcqa_v1.py -i input.json -o output.json --dry-run --verbose
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, help='Input MCQA JSON file to clean')
    parser.add_argument('-o', '--output', required=True, help='Output file for cleaned MCQA data')
    
    # Model and API configuration
    parser.add_argument('--model', default='gpt-4', help='LLM model to use for evaluation (default: gpt-4)')
    parser.add_argument('--config', help='YAML configuration file for API settings')
    
    # Filtering parameters
    parser.add_argument('--min-score', type=int, default=7, help='Minimum quality score (1-10) to keep question (default: 7)')
    parser.add_argument('--relevance-threshold', type=int, default=6, help='Minimum relevance score (1-10) to keep question (default: 6)')
    
    # Performance settings
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing (default: 10)')
    
    # Output options
    parser.add_argument('--filtered-output', help='Output file for filtered questions (for inspection)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without saving output')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        log_message(f"Input file not found: {args.input}", log_level="ERROR")
        sys.exit(1)
    
    # Create config object
    config = CleaningConfig(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        min_score=args.min_score,
        relevance_threshold=args.relevance_threshold,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        verbose=args.verbose,
        dry_run=args.dry_run,
        filtered_output_file=args.filtered_output
    )
    
    # Add config file if provided
    if args.config:
        config.config_file = args.config
    
    # Run the cleaning process
    try:
        success = clean_mcqa_file(config)
        if success:
            log_message("MCQA cleaning completed successfully!")
            sys.exit(0)
        else:
            log_message("MCQA cleaning failed!", log_level="ERROR")
            sys.exit(1)
    except KeyboardInterrupt:
        log_message("Cleaning interrupted by user", log_level="WARNING")
        sys.exit(130)
    except Exception as e:
        log_message(f"Unexpected error: {e}", log_level="ERROR", error_type="unexpected")
        sys.exit(1)

if __name__ == "__main__":
    main()