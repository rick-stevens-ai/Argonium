#!/usr/bin/env python3
"""
Comprehensive Fault Pattern Analysis

This script analyzes the fault analysis files to identify root causes of failures:
1. MCQA Creation Issues vs Reasoning Issues
2. Method 1 vs Method 2 failure patterns  
3. Common failure modes within each category
4. Knowledge requirement patterns

Uses the same --model and --config structure as make_v22.py.
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
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

def parse_comprehensive_fault_analysis(fault_file: str) -> Dict[str, any]:
    """Parse a fault analysis file to extract comprehensive fault information"""
    try:
        with open(fault_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            # Basic answer info
            'has_disagreement': False,
            'correct_answer': None,
            'method1_prediction': None,
            'method2_prediction': None,
            'disagreement_type': 'none',
            
            # Fault analysis
            'primary_fault': None,
            'fault_confidence': None,
            'fault_description': None,
            'secondary_factors': [],
            
            # Knowledge requirements
            'requires_external_knowledge': False,
            'knowledge_type': None,
            'knowledge_specificity': None,
            
            # Reasoning quality
            'method1_reasoning_validity': None,
            'method2_reasoning_validity': None,
            'better_reasoning_approach': None,
            
            # Root cause
            'primary_root_cause': None,
            'problem_severity': None,
            'problem_scope': None,
            'contributing_factors': [],
            
            # Recommendations
            'immediate_fix': None,
            'mcqa_improvement': None,
            'prevention_strategy': None
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Basic answer comparison
            if line.startswith('Correct Answer:'):
                result['correct_answer'] = line.replace('Correct Answer:', '').strip()
            elif line.startswith('Method 1 Prediction:'):
                result['method1_prediction'] = line.replace('Method 1 Prediction:', '').strip()
            elif line.startswith('Method 2 Prediction:'):
                result['method2_prediction'] = line.replace('Method 2 Prediction:', '').strip()
            elif line.startswith('Has Disagreement:'):
                result['has_disagreement'] = line.replace('Has Disagreement:', '').strip().lower() == 'true'
            elif line.startswith('Disagreement Type:'):
                result['disagreement_type'] = line.replace('Disagreement Type:', '').strip()
            
            # Fault analysis
            elif 'PRIMARY FAULT:' in line:
                result['primary_fault'] = line.split('PRIMARY FAULT:')[1].strip()
            elif line.startswith('Fault Confidence:'):
                result['fault_confidence'] = line.replace('Fault Confidence:', '').strip()
            elif line.startswith('Fault Description:'):
                result['fault_description'] = line.replace('Fault Description:', '').strip()
            
            # Knowledge requirements
            elif line.startswith('Requires External Knowledge:'):
                result['requires_external_knowledge'] = line.replace('Requires External Knowledge:', '').strip().lower() == 'true'
            elif line.startswith('Knowledge Type:'):
                result['knowledge_type'] = line.replace('Knowledge Type:', '').strip()
            elif line.startswith('Knowledge Specificity:'):
                result['knowledge_specificity'] = line.replace('Knowledge Specificity:', '').strip()
            
            # Reasoning quality
            elif line.startswith('Method 1 Reasoning Validity:'):
                result['method1_reasoning_validity'] = line.replace('Method 1 Reasoning Validity:', '').strip()
            elif line.startswith('Method 2 Reasoning Validity:'):
                result['method2_reasoning_validity'] = line.replace('Method 2 Reasoning Validity:', '').strip()
            elif line.startswith('Better Reasoning Approach:'):
                result['better_reasoning_approach'] = line.replace('Better Reasoning Approach:', '').strip()
            
            # Root cause
            elif 'Primary Root Cause:' in line:
                result['primary_root_cause'] = line.split('Primary Root Cause:')[1].strip()
            elif line.startswith('Problem Severity:'):
                result['problem_severity'] = line.replace('Problem Severity:', '').strip()
            elif line.startswith('Problem Scope:'):
                result['problem_scope'] = line.replace('Problem Scope:', '').strip()
            
            # Recommendations
            elif 'Immediate Fix:' in line:
                result['immediate_fix'] = line.split('Immediate Fix:')[1].strip()
            elif line.startswith('MCQA Improvement:'):
                result['mcqa_improvement'] = line.replace('MCQA Improvement:', '').strip()
            elif line.startswith('Prevention Strategy:'):
                result['prevention_strategy'] = line.replace('Prevention Strategy:', '').strip()
        
        return result
        
    except Exception as e:
        print(f"Error parsing fault analysis file {fault_file}: {e}")
        return None

def analyze_fault_patterns(output_dir: str) -> Dict[str, any]:
    """Analyze fault patterns across all fault analysis files"""
    
    fault_files = list(Path(output_dir).glob("*_fault_analysis.txt"))
    print(f"Found {len(fault_files)} fault analysis files")
    
    # Parse all fault analyses with progress bar
    all_faults = []
    parse_errors = 0
    
    print("Parsing fault analysis files...")
    for fault_file in tqdm(fault_files, desc="Processing files", unit="file"):
        fault_data = parse_comprehensive_fault_analysis(str(fault_file))
        if fault_data:
            base_name = fault_file.stem.replace('_fault_analysis', '')
            fault_data['base_name'] = base_name
            all_faults.append(fault_data)
        else:
            parse_errors += 1
    
    print(f"Successfully parsed {len(all_faults)} files ({parse_errors} parse errors)")
    
    # Analyze patterns with progress bar
    print("Analyzing fault patterns...")
    analysis = {
        'total_questions': len(all_faults),
        'parse_errors': parse_errors,
        
        # Primary fault categories
        'fault_categories': Counter(),
        'fault_severity': Counter(),
        'fault_scope': Counter(),
        
        # Method performance
        'method1_validity': Counter(),
        'method2_validity': Counter(),
        'better_approach': Counter(),
        
        # Knowledge requirements
        'external_knowledge_required': 0,
        'knowledge_types': Counter(),
        'knowledge_specificity': Counter(),
        
        # Specific fault patterns
        'mcqa_creation_issues': [],
        'reasoning_issues': [],
        'knowledge_issues': [],
        'other_issues': [],
        
        # Method-specific patterns
        'method1_failures': [],
        'method2_failures': [],
        'both_method_failures': []
    }
    
    for fault in tqdm(all_faults, desc="Analyzing patterns", unit="question"):
        # Fault categories
        if fault['primary_fault']:
            analysis['fault_categories'][fault['primary_fault']] += 1
        
        if fault['problem_severity']:
            analysis['fault_severity'][fault['problem_severity']] += 1
            
        if fault['problem_scope']:
            analysis['fault_scope'][fault['problem_scope']] += 1
        
        # Method validity
        if fault['method1_reasoning_validity']:
            analysis['method1_validity'][fault['method1_reasoning_validity']] += 1
        if fault['method2_reasoning_validity']:
            analysis['method2_validity'][fault['method2_reasoning_validity']] += 1
        if fault['better_reasoning_approach']:
            analysis['better_approach'][fault['better_reasoning_approach']] += 1
        
        # Knowledge requirements
        if fault['requires_external_knowledge']:
            analysis['external_knowledge_required'] += 1
        if fault['knowledge_type']:
            analysis['knowledge_types'][fault['knowledge_type']] += 1
        if fault['knowledge_specificity']:
            analysis['knowledge_specificity'][fault['knowledge_specificity']] += 1
        
        # Categorize specific issues
        primary_fault = fault['primary_fault'] or 'UNKNOWN'
        
        if 'MCQA' in primary_fault.upper():
            analysis['mcqa_creation_issues'].append(fault)
        elif any(word in primary_fault.upper() for word in ['REASONING', 'METHOD', 'LOGIC']):
            analysis['reasoning_issues'].append(fault)
        elif 'KNOWLEDGE' in primary_fault.upper():
            analysis['knowledge_issues'].append(fault)
        else:
            analysis['other_issues'].append(fault)
        
        # Method-specific failures
        method1_wrong = fault['method1_prediction'] != fault['correct_answer']
        method2_wrong = fault['method2_prediction'] != fault['correct_answer']
        
        if method1_wrong and method2_wrong:
            analysis['both_method_failures'].append(fault)
        elif method1_wrong:
            analysis['method1_failures'].append(fault)
        elif method2_wrong:
            analysis['method2_failures'].append(fault)
    
    return analysis

def generate_comprehensive_report(analysis: Dict[str, any], output_file: str) -> None:
    """Generate comprehensive fault pattern analysis report"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE FAULT PATTERN ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        total = analysis['total_questions']
        f.write(f"Total Questions Analyzed: {total}\n")
        f.write(f"Parse Errors: {analysis['parse_errors']}\n\n")
        
        # Primary fault breakdown
        f.write("PRIMARY FAULT BREAKDOWN:\n")
        f.write("-" * 40 + "\n")
        mcqa_issues = len(analysis['mcqa_creation_issues'])
        reasoning_issues = len(analysis['reasoning_issues'])
        knowledge_issues = len(analysis['knowledge_issues'])
        other_issues = len(analysis['other_issues'])
        
        f.write(f"MCQA Creation Issues: {mcqa_issues} ({mcqa_issues/total*100:.1f}%)\n")
        f.write(f"Reasoning Issues: {reasoning_issues} ({reasoning_issues/total*100:.1f}%)\n")
        f.write(f"Knowledge Issues: {knowledge_issues} ({knowledge_issues/total*100:.1f}%)\n")
        f.write(f"Other Issues: {other_issues} ({other_issues/total*100:.1f}%)\n\n")
        
        # Method performance comparison
        f.write("METHOD PERFORMANCE COMPARISON:\n")
        f.write("-" * 40 + "\n")
        method1_only = len(analysis['method1_failures'])
        method2_only = len(analysis['method2_failures'])
        both_methods = len(analysis['both_method_failures'])
        
        f.write(f"Method 1 Only Failures: {method1_only}\n")
        f.write(f"Method 2 Only Failures: {method2_only}\n")
        f.write(f"Both Methods Fail: {both_methods}\n")
        f.write(f"Method 1 Total Failures: {method1_only + both_methods}\n")
        f.write(f"Method 2 Total Failures: {method2_only + both_methods}\n\n")
        
        # Reasoning validity
        f.write("REASONING VALIDITY PATTERNS:\n")
        f.write("-" * 40 + "\n")
        f.write("Method 1 Reasoning Validity:\n")
        for validity, count in analysis['method1_validity'].most_common():
            f.write(f"  {validity}: {count} ({count/total*100:.1f}%)\n")
        f.write("\nMethod 2 Reasoning Validity:\n")
        for validity, count in analysis['method2_validity'].most_common():
            f.write(f"  {validity}: {count} ({count/total*100:.1f}%)\n")
        f.write("\nBetter Reasoning Approach:\n")
        for approach, count in analysis['better_approach'].most_common():
            f.write(f"  {approach}: {count} ({count/total*100:.1f}%)\n")
        f.write("\n")
        
        # Knowledge requirements
        f.write("KNOWLEDGE REQUIREMENT ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        ext_knowledge = analysis['external_knowledge_required']
        f.write(f"Requires External Knowledge: {ext_knowledge} ({ext_knowledge/total*100:.1f}%)\n\n")
        
        f.write("Knowledge Types:\n")
        for ktype, count in analysis['knowledge_types'].most_common():
            f.write(f"  {ktype}: {count}\n")
        f.write("\nKnowledge Specificity:\n")
        for spec, count in analysis['knowledge_specificity'].most_common():
            f.write(f"  {spec}: {count}\n")
        f.write("\n")
        
        # Detailed fault categories
        f.write("DETAILED FAULT CATEGORIES:\n")
        f.write("-" * 40 + "\n")
        for fault_type, count in analysis['fault_categories'].most_common():
            f.write(f"{fault_type}: {count} ({count/total*100:.1f}%)\n")
        f.write("\n")
        
        # Problem severity and scope
        f.write("PROBLEM SEVERITY:\n")
        f.write("-" * 40 + "\n")
        for severity, count in analysis['fault_severity'].most_common():
            f.write(f"{severity}: {count}\n")
        f.write("\nPROBLEM SCOPE:\n")
        f.write("-" * 40 + "\n")
        for scope, count in analysis['fault_scope'].most_common():
            f.write(f"{scope}: {count}\n")
        f.write("\n")
        
        # Specific issue examples
        f.write("MCQA CREATION ISSUES (Examples):\n")
        f.write("-" * 40 + "\n")
        for i, issue in enumerate(analysis['mcqa_creation_issues'][:5], 1):
            f.write(f"{i}. {issue['base_name']}\n")
            f.write(f"   Fault: {issue['primary_fault']}\n")
            f.write(f"   Description: {issue['fault_description']}\n\n")
        
        f.write("REASONING ISSUES (Examples):\n")
        f.write("-" * 40 + "\n")
        for i, issue in enumerate(analysis['reasoning_issues'][:5], 1):
            f.write(f"{i}. {issue['base_name']}\n")
            f.write(f"   Fault: {issue['primary_fault']}\n")
            f.write(f"   Description: {issue['fault_description']}\n\n")
        
        # Summary insights
        f.write("KEY INSIGHTS:\n")
        f.write("-" * 40 + "\n")
        
        # Calculate key metrics
        most_common_fault = analysis['fault_categories'].most_common(1)[0] if analysis['fault_categories'] else ('Unknown', 0)
        method1_sound = analysis['method1_validity'].get('sound', 0)
        method2_sound = analysis['method2_validity'].get('sound', 0)
        
        f.write(f"• Most Common Fault Type: {most_common_fault[0]} ({most_common_fault[1]} cases)\n")
        f.write(f"• Method 1 Sound Reasoning: {method1_sound/total*100:.1f}%\n")
        f.write(f"• Method 2 Sound Reasoning: {method2_sound/total*100:.1f}%\n")
        f.write(f"• External Knowledge Required: {ext_knowledge/total*100:.1f}% of questions\n")
        
        if mcqa_issues > reasoning_issues:
            f.write("• MCQA Creation is the primary source of failures\n")
        else:
            f.write("• Reasoning Issues are the primary source of failures\n")
        
        if method1_only + both_methods > method2_only + both_methods:
            f.write("• Method 1 (Two-Stage Reasoning) shows more failures than Method 2\n")
        else:
            f.write("• Method 2 (Argonium-Style) shows more failures than Method 1\n")

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive fault pattern analysis"
    )
    parser.add_argument(
        "output_dir", 
        help="Directory containing reasoning traces output files"
    )
    parser.add_argument(
        "--output", "-o",
        default="comprehensive_fault_analysis.txt",
        help="Output file for comprehensive analysis (default: comprehensive_fault_analysis.txt)"
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
    
    args = parser.parse_args()
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    print(f"📁 Analyzing fault patterns in: {args.output_dir}")
    
    # Analyze fault patterns
    print("\n🔍 Analyzing fault patterns...")
    analysis = analyze_fault_patterns(args.output_dir)
    
    # Generate comprehensive report
    print(f"\n📝 Generating comprehensive report...")
    generate_comprehensive_report(analysis, args.output)
    print(f"✅ Comprehensive report saved to: {args.output}")
    
    # Print key insights
    total = analysis['total_questions']
    mcqa_issues = len(analysis['mcqa_creation_issues'])
    reasoning_issues = len(analysis['reasoning_issues'])
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   📊 Total Questions: {total}")
    print(f"   🏗️  MCQA Creation Issues: {mcqa_issues} ({mcqa_issues/total*100:.1f}%)")
    print(f"   🧠 Reasoning Issues: {reasoning_issues} ({reasoning_issues/total*100:.1f}%)")
    print(f"   📚 External Knowledge Required: {analysis['external_knowledge_required']} ({analysis['external_knowledge_required']/total*100:.1f}%)")
    
    method1_failures = len(analysis['method1_failures']) + len(analysis['both_method_failures'])
    method2_failures = len(analysis['method2_failures']) + len(analysis['both_method_failures'])
    print(f"   🔧 Method 1 Failures: {method1_failures}")
    print(f"   ⚡ Method 2 Failures: {method2_failures}")

if __name__ == "__main__":
    main()