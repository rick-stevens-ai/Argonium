#!/usr/bin/env python3

"""
Merge Incorrect Answers Script

This script merges multiple JSON files containing incorrectly answered questions 
and produces a deduplicated list. It can output in the format expected by 
argonium_score_parallel (array of objects with "question" and "answer" fields).

Usage:
    python merge_incorrect_answers.py file1.json file2.json ... [--output output.json] [--format argonium|original] [--mode union|intersection]

Where:
    - file1.json, file2.json: Input JSON files with incorrect answers
    - --output: Output file (default: merged_incorrect_<timestamp>.json)
    - --format: Output format (argonium for argonium_score_parallel, original for preserving original structure)
    - --mode: Merge mode (union=all incorrect answers, intersection=only questions incorrect in multiple models)

Examples:
    # Merge all incorrect answers (union)
    python merge_incorrect_answers.py incorrect_qwen_*.json --format argonium --output merged_for_testing.json
    
    # Find questions that multiple models got wrong (intersection)
    python merge_incorrect_answers.py incorrect_qwen_*.json incorrect_scout_*.json --mode intersection --format argonium --output hardest_questions.json
"""

import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
import hashlib

def load_incorrect_answers_file(filepath):
    """Load and parse an incorrect answers JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'incorrect_answers' not in data:
            print(f"Warning: {filepath} doesn't contain 'incorrect_answers' field, skipping")
            return []
            
        return data['incorrect_answers']
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def create_question_hash(question_text):
    """Create a hash for deduplication based on question text."""
    # Normalize whitespace and create hash
    normalized = ' '.join(question_text.split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def deduplicate_questions(all_incorrect):
    """Remove duplicate questions based on question text."""
    seen_hashes = set()
    deduplicated = []
    
    for item in all_incorrect:
        question_text = item.get('question', '')
        question_hash = create_question_hash(question_text)
        
        if question_hash not in seen_hashes:
            seen_hashes.add(question_hash)
            deduplicated.append(item)
    
    return deduplicated

def convert_to_argonium_format(incorrect_answers):
    """Convert incorrect answers to the format expected by argonium_score_parallel."""
    argonium_format = []
    
    for item in incorrect_answers:
        # Extract question and reference answer
        question = item.get('question', '')
        reference_answer = item.get('reference_answer', '')
        
        if question and reference_answer:
            argonium_format.append({
                "question": question,
                "answer": reference_answer
            })
    
    return argonium_format

def find_intersection(input_files, min_occurrences=2):
    """Find questions that appear in multiple files (intersection)."""
    question_occurrences = {}  # hash -> list of (filepath, question_data)
    file_stats = {}
    
    print(f"Processing {len(input_files)} files for intersection analysis...")
    
    for filepath in input_files:
        print(f"Loading {filepath}...")
        incorrect_answers = load_incorrect_answers_file(filepath)
        file_stats[filepath] = len(incorrect_answers)
        print(f"  Found {len(incorrect_answers)} incorrect answers")
        
        for item in incorrect_answers:
            question_text = item.get('question', '')
            question_hash = create_question_hash(question_text)
            
            if question_hash not in question_occurrences:
                question_occurrences[question_hash] = []
            
            question_occurrences[question_hash].append((filepath, item))
    
    # Find questions that appear in multiple files
    intersection_questions = []
    intersection_stats = {}
    
    for question_hash, occurrences in question_occurrences.items():
        if len(occurrences) >= min_occurrences:
            # Use the first occurrence as the canonical version
            _, question_data = occurrences[0]
            intersection_questions.append(question_data)
            
            # Track which files had this question
            files_with_question = [filepath for filepath, _ in occurrences]
            intersection_stats[question_hash] = {
                "question_preview": question_data.get('question', '')[:100] + "...",
                "files": files_with_question,
                "count": len(occurrences)
            }
    
    print(f"\nIntersection analysis:")
    print(f"Questions appearing in >= {min_occurrences} files: {len(intersection_questions)}")
    
    # Show distribution of intersection counts
    count_distribution = {}
    for stats in intersection_stats.values():
        count = stats["count"]
        count_distribution[count] = count_distribution.get(count, 0) + 1
    
    for count in sorted(count_distribution.keys(), reverse=True):
        print(f"  Questions in {count} files: {count_distribution[count]}")
    
    return intersection_questions, file_stats, intersection_stats

def merge_files(input_files, output_format='original', mode='union', min_models=2):
    """Merge multiple incorrect answers files."""
    
    if mode == 'intersection':
        if len(input_files) < min_models:
            print(f"Error: Intersection mode with min-models={min_models} requires at least {min_models} input files")
            sys.exit(1)
        
        deduplicated, file_stats, intersection_stats = find_intersection(input_files, min_models)
        total_before_dedup = sum(file_stats.values())
        
        # Convert format if requested
        if output_format == 'argonium':
            result = convert_to_argonium_format(deduplicated)
            print(f"Converted to argonium format: {len(result)} question-answer pairs")
        else:
            # Preserve original structure with metadata
            result = {
                "metadata": {
                    "mode": "intersection",
                    "merged_from": list(file_stats.keys()),
                    "file_stats": file_stats,
                    "total_across_files": total_before_dedup,
                    "intersection_questions": len(deduplicated),
                    "intersection_stats": intersection_stats,
                    "timestamp": datetime.now().isoformat(),
                    "format": "original"
                },
                "incorrect_answers": deduplicated
            }
        
        return result
    
    else:  # Union mode (original behavior)
        all_incorrect = []
        file_stats = {}
        
        print(f"Processing {len(input_files)} files...")
        
        for filepath in input_files:
            print(f"Loading {filepath}...")
            incorrect_answers = load_incorrect_answers_file(filepath)
            file_stats[filepath] = len(incorrect_answers)
            all_incorrect.extend(incorrect_answers)
            print(f"  Found {len(incorrect_answers)} incorrect answers")
        
        print(f"\nTotal incorrect answers before deduplication: {len(all_incorrect)}")
        
        # Deduplicate based on question text
        deduplicated = deduplicate_questions(all_incorrect)
        print(f"Total incorrect answers after deduplication: {len(deduplicated)}")
        print(f"Removed {len(all_incorrect) - len(deduplicated)} duplicates")
        
        # Convert format if requested
        if output_format == 'argonium':
            result = convert_to_argonium_format(deduplicated)
            print(f"Converted to argonium format: {len(result)} question-answer pairs")
        else:
            # Preserve original structure with metadata
            result = {
                "metadata": {
                    "mode": "union",
                    "merged_from": list(file_stats.keys()),
                    "file_stats": file_stats,
                    "total_before_dedup": len(all_incorrect),
                    "total_after_dedup": len(deduplicated),
                    "duplicates_removed": len(all_incorrect) - len(deduplicated),
                    "timestamp": datetime.now().isoformat(),
                    "format": "original"
                },
                "incorrect_answers": deduplicated
            }
        
        return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Merge incorrect answers JSON files')
    parser.add_argument('files', nargs='+', help='Input JSON files to merge')
    parser.add_argument('--output', help='Output file (default: merged_incorrect_<timestamp>.json)')
    parser.add_argument('--format', choices=['original', 'argonium'], default='original',
                        help='Output format: original (preserve structure) or argonium (for argonium_score_parallel)')
    parser.add_argument('--mode', choices=['union', 'intersection'], default='union',
                        help='Merge mode: union (all incorrect answers) or intersection (questions incorrect in multiple models)')
    parser.add_argument('--min-models', type=int, default=2,
                        help='Minimum number of models that must have answered incorrectly (intersection mode only, default: 2)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate input files
    input_files = []
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File {filepath} does not exist")
            sys.exit(1)
        input_files.append(str(path))
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = f"_{args.mode}" if args.mode != 'union' else ""
        if args.format == 'argonium':
            output_file = f"merged_incorrect{mode_suffix}_argonium_{timestamp}.json"
        else:
            output_file = f"merged_incorrect{mode_suffix}_{timestamp}.json"
    
    print(f"Merging {len(input_files)} files...")
    print(f"Merge mode: {args.mode}")
    if args.mode == 'intersection':
        print(f"Minimum models: {args.min_models}")
    print(f"Output format: {args.format}")
    print(f"Output file: {output_file}")
    print()
    
    # Merge files
    result = merge_files(input_files, args.format, args.mode, args.min_models)
    
    # Save result
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved merged results to {output_file}")
        
        # Print summary
        if args.format == 'argonium':
            print(f"Ready to use with argonium_score_parallel: {len(result)} questions")
        else:
            print(f"Total questions in output: {len(result['incorrect_answers'])}")
            if args.mode == 'intersection':
                print(f"These are questions that were answered incorrectly by multiple models")
            
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()