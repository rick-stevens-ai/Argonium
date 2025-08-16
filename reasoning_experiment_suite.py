#!/usr/bin/env python3
"""
Reasoning Structure Experiment Suite

This is the main orchestration script for analyzing logical structure 
in Stream of Thought outputs. It provides a complete workflow for:

1. Generating stream analysis files from reasoning traces
2. Analyzing logical structure patterns
3. Visualizing reasoning patterns
4. Comparing reasoning across different conditions

Usage Examples:
    # Full analysis pipeline
    python reasoning_experiment_suite.py --analyze-directory stream_files/
    
    # Compare two groups
    python reasoning_experiment_suite.py --compare group1:analysis1.json group2:analysis2.json
    
    # Generate comprehensive report
    python reasoning_experiment_suite.py --full-report --input stream_files/
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

class ReasoningExperimentSuite:
    """Main orchestrator for reasoning structure experiments"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.analyzer_script = self.script_dir / "analyze_reasoning_structure.py"
        self.visualizer_script = self.script_dir / "visualize_reasoning_patterns.py"
        self.comparator_script = self.script_dir / "compare_reasoning_structures.py"
    
    def check_dependencies(self):
        """Check if all required scripts exist"""
        scripts = [self.analyzer_script, self.visualizer_script, self.comparator_script]
        missing = [s for s in scripts if not s.exists()]
        
        if missing:
            print("Error: Missing required scripts:")
            for script in missing:
                print(f"  - {script}")
            return False
        return True
    
    def run_analysis(self, directory: str, output_file: str = None) -> str:
        """Run logical structure analysis on a directory of stream files"""
        if not output_file:
            output_file = f"{directory}_logical_analysis.json"
        
        print(f"🔍 Analyzing logical structure in {directory}...")
        
        cmd = [
            "python", str(self.analyzer_script),
            directory,
            "--output", output_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Analysis complete: {output_file}")
                return output_file
            else:
                print(f"❌ Analysis failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Error running analysis: {e}")
            return None
    
    def run_visualization(self, analysis_file: str, save_plots: bool = True) -> bool:
        """Run visualization on analysis results"""
        print(f"📊 Creating visualizations for {analysis_file}...")
        
        cmd = [
            "python", str(self.visualizer_script),
            analysis_file
        ]
        
        if save_plots:
            cmd.extend(["--save-plots"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Visualizations complete")
                # Print the sequence analysis part
                if "REASONING SEQUENCE PATTERN ANALYSIS" in result.stdout:
                    lines = result.stdout.split('\n')
                    start_idx = None
                    for i, line in enumerate(lines):
                        if "REASONING SEQUENCE PATTERN ANALYSIS" in line:
                            start_idx = i
                            break
                    
                    if start_idx:
                        print("\n".join(lines[start_idx:start_idx+20]))  # Print analysis section
                return True
            else:
                print(f"❌ Visualization failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error running visualization: {e}")
            return False
    
    def run_comparison(self, groups: list) -> bool:
        """Run comparison between groups"""
        print(f"⚖️  Comparing reasoning structures across {len(groups)} groups...")
        
        cmd = [
            "python", str(self.comparator_script)
        ]
        
        # Add groups
        for group_name, analysis_file in groups:
            cmd.extend(["--group", group_name, analysis_file])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Comparison complete")
                return True
            else:
                print(f"❌ Comparison failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error running comparison: {e}")
            return False
    
    def generate_experiment_summary(self, analysis_files: list, output_file: str = "experiment_summary.md"):
        """Generate a markdown summary of the entire experiment"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Reasoning Structure Analysis Experiment\n\n")
            f.write("## Overview\n\n")
            f.write("This experiment analyzed the logical structure of expert reasoning ")
            f.write("in Stream of Thought outputs, independent of domain-specific biological content.\n\n")
            
            f.write("## Methodology\n\n")
            f.write("1. **Stream Generation**: Created individual coherent stream of thought analysis files\n")
            f.write("2. **Pattern Recognition**: Identified logical reasoning patterns using regex-based detection\n")
            f.write("3. **Structural Analysis**: Quantified uncertainty markers, logical connectors, evidence evaluation\n")
            f.write("4. **Flow Analysis**: Mapped argumentation flow sequences and decision patterns\n")
            f.write("5. **Visualization**: Created visual representations of reasoning patterns\n\n")
            
            f.write("## Key Metrics Analyzed\n\n")
            f.write("- **Uncertainty Markers**: 'hmm', 'wait', 'actually', 'I think'\n")
            f.write("- **Logical Connectors**: 'because', 'therefore', 'if-then', 'compared to'\n")
            f.write("- **Evidence Evaluation**: 'the evidence shows', 'based on', 'research shows'\n")
            f.write("- **Contradictions**: 'but that contradicts', 'that's wrong', 'opposite'\n")
            f.write("- **Conclusions**: 'I conclude', 'ultimately', 'my answer is'\n")
            f.write("- **Decision Points**: 'I need to decide', 'the key is'\n\n")
            
            f.write("## Argumentation Flow Categories\n\n")
            f.write("- **INITIAL_APPROACH**: Setting up the problem\n")
            f.write("- **OPTION_INTRODUCTION**: Presenting possibilities\n")
            f.write("- **COUNTER_CONSIDERATION**: Challenging ideas\n")
            f.write("- **RECONSIDERATION**: Rethinking positions\n")
            f.write("- **LOGICAL_INFERENCE**: Drawing conclusions\n")
            f.write("- **CAUSAL_REASONING**: Explaining cause-effect\n")
            f.write("- **COMPARISON**: Weighing alternatives\n")
            f.write("- **ELABORATION**: Developing ideas\n")
            f.write("- **CONCLUSION**: Final decision\n\n")
            
            f.write("## Files Analyzed\n\n")
            for i, analysis_file in enumerate(analysis_files, 1):
                f.write(f"{i}. `{analysis_file}`\n")
            
            f.write("\n## Research Questions Addressed\n\n")
            f.write("1. **Universal Reasoning Patterns**: What logical structures appear consistently ")
            f.write("across different scientific domains?\n")
            f.write("2. **Uncertainty Dynamics**: How does uncertainty evolve throughout the reasoning process?\n")
            f.write("3. **Decision Architecture**: What are the common pathways experts use to reach conclusions?\n")
            f.write("4. **Argumentation Flow**: Are there predictable sequences in expert reasoning?\n")
            f.write("5. **Domain Independence**: Which reasoning patterns are universal vs domain-specific?\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("- Compare reasoning patterns across different scientific domains\n")
            f.write("- Analyze correlation between reasoning complexity and accuracy\n")
            f.write("- Identify predictive patterns for correct vs incorrect conclusions\n")
            f.write("- Develop metrics for reasoning quality assessment\n")
        
        print(f"📋 Experiment summary generated: {output_file}")

def main():
    """Main function for the reasoning experiment suite"""
    parser = argparse.ArgumentParser(description="Reasoning Structure Experiment Suite")
    
    # Main modes
    parser.add_argument("--analyze-directory", help="Directory with stream analysis files to analyze")
    parser.add_argument("--compare", nargs='+', help="Compare groups: group1:file1.json group2:file2.json")
    parser.add_argument("--full-report", action="store_true", help="Generate complete analysis report")
    
    # Options
    parser.add_argument("--input", help="Input directory for full report mode")
    parser.add_argument("--output", help="Output file for analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    suite = ReasoningExperimentSuite()
    
    if not suite.check_dependencies():
        return 1
    
    if args.analyze_directory:
        # Single directory analysis
        analysis_file = suite.run_analysis(args.analyze_directory, args.output)
        if analysis_file and not args.no_plots:
            suite.run_visualization(analysis_file)
        
        if analysis_file:
            suite.generate_experiment_summary([analysis_file])
    
    elif args.compare:
        # Parse comparison groups
        groups = []
        for group_spec in args.compare:
            try:
                group_name, analysis_file = group_spec.split(':', 1)
                groups.append((group_name, analysis_file))
            except ValueError:
                print(f"Error: Invalid group specification '{group_spec}'. Use format 'name:file.json'")
                return 1
        
        if len(groups) < 2:
            print("Error: Need at least 2 groups to compare")
            return 1
        
        suite.run_comparison(groups)
    
    elif args.full_report:
        # Full analysis workflow
        if not args.input:
            print("Error: --input directory required for full report mode")
            return 1
        
        print("🚀 Running full reasoning structure analysis experiment...")
        
        # Step 1: Analyze
        analysis_file = suite.run_analysis(args.input)
        if not analysis_file:
            return 1
        
        # Step 2: Visualize
        if not args.no_plots:
            suite.run_visualization(analysis_file)
        
        # Step 3: Generate summary
        suite.generate_experiment_summary([analysis_file])
        
        print("🎉 Full experiment complete!")
        print("\nGenerated files:")
        print(f"  - {analysis_file} (detailed analysis)")
        print("  - reasoning_patterns_*.png (visualizations)")
        print("  - reasoning_patterns_report.txt (pattern report)")  
        print("  - experiment_summary.md (experiment overview)")
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())