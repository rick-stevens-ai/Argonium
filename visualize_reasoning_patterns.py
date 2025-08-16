#!/usr/bin/env python3
"""
Reasoning Pattern Visualizer

This script creates visualizations of logical reasoning patterns from the
analyzed stream of thought data, helping to understand the structure of
expert reasoning across different questions and domains.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any

class ReasoningPatternVisualizer:
    """Creates visualizations of reasoning patterns"""
    
    def __init__(self, analysis_file: str):
        """Initialize with analysis data from JSON file"""
        with open(analysis_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Remove metadata entries
        self.file_data = {k: v for k, v in self.data.items() 
                         if not k.startswith(('SUMMARY_', 'ANALYSIS_'))}
        
        print(f"Loaded analysis for {len(self.file_data)} files")
    
    def plot_uncertainty_distribution(self, save_file: str = None):
        """Plot the distribution of uncertainty markers across reasoning process"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect all uncertainty positions
        all_positions = []
        uncertainty_counts = []
        
        for filename, data in self.file_data.items():
            positions = [marker['position'] for marker in data['uncertainty_markers']]
            all_positions.extend(positions)
            uncertainty_counts.append(len(positions))
        
        # Plot 1: Histogram of uncertainty positions
        ax1.hist(all_positions, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_xlabel('Position in Reasoning Process (0=start, 1=end)')
        ax1.set_ylabel('Number of Uncertainty Markers')
        ax1.set_title('Distribution of Uncertainty Throughout Reasoning')
        ax1.grid(True, alpha=0.3)
        
        # Add vertical lines for thirds
        ax1.axvline(0.33, color='red', linestyle='--', alpha=0.5, label='Early/Middle')
        ax1.axvline(0.67, color='red', linestyle='--', alpha=0.5, label='Middle/Late')
        ax1.legend()
        
        # Plot 2: Distribution of total uncertainty counts per file
        ax2.hist(uncertainty_counts, bins=max(1, len(set(uncertainty_counts))), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Number of Uncertainty Markers per File')
        ax2.set_ylabel('Number of Files')
        ax2.set_title('Uncertainty Marker Frequency Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Uncertainty distribution plot saved to {save_file}")
        
        plt.show()
    
    def plot_argumentation_flow_patterns(self, save_file: str = None):
        """Plot common argumentation flow patterns"""
        # Collect all flow patterns
        all_flows = []
        flow_lengths = []
        
        for filename, data in self.file_data.items():
            flows = data['argumentation_flow']
            all_flows.extend(flows)
            flow_lengths.append(len(flows))
        
        # Count most common flow types
        flow_counter = Counter(all_flows)
        top_flows = flow_counter.most_common(12)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Most common reasoning step types
        flow_types = [item[0] for item in top_flows]
        flow_counts = [item[1] for item in top_flows]
        
        bars = ax1.barh(range(len(flow_types)), flow_counts, color='lightcoral')
        ax1.set_yticks(range(len(flow_types)))
        ax1.set_yticklabels([ft.replace('_', ' ').title() for ft in flow_types])
        ax1.set_xlabel('Frequency Across All Files')
        ax1.set_title('Most Common Reasoning Step Types')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width}', ha='left', va='center')
        
        # Plot 2: Distribution of reasoning process lengths
        ax2.hist(flow_lengths, bins=max(1, len(set(flow_lengths))), 
                alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Number of Reasoning Steps per File')
        ax2.set_ylabel('Number of Files')
        ax2.set_title('Distribution of Reasoning Process Lengths')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Argumentation flow plot saved to {save_file}")
        
        plt.show()
    
    def plot_pattern_complexity_matrix(self, save_file: str = None):
        """Plot a matrix showing the complexity of different reasoning patterns per file"""
        # Prepare data matrix
        filenames = list(self.file_data.keys())
        pattern_types = ['uncertainty_markers', 'logical_connectors', 'evidence_evaluations', 
                        'contradictions', 'conclusions', 'decision_points']
        
        # Create matrix
        matrix = np.zeros((len(filenames), len(pattern_types)))
        
        for i, filename in enumerate(filenames):
            for j, pattern_type in enumerate(pattern_types):
                matrix[i, j] = len(self.data[filename][pattern_type])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(filenames) * 0.3)))
        
        # Truncate long filenames for display
        display_names = [name[:40] + "..." if len(name) > 40 else name 
                        for name in filenames]
        display_patterns = [pt.replace('_', ' ').title() for pt in pattern_types]
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(pattern_types)))
        ax.set_xticklabels(display_patterns, rotation=45, ha='right')
        ax.set_yticks(range(len(filenames)))
        ax.set_yticklabels(display_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Pattern Instances')
        
        # Add text annotations
        for i in range(len(filenames)):
            for j in range(len(pattern_types)):
                value = int(matrix[i, j])
                ax.text(j, i, str(value), ha='center', va='center',
                       color='white' if value > matrix.max()/2 else 'black')
        
        ax.set_title('Reasoning Pattern Complexity Matrix')
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Pattern complexity matrix saved to {save_file}")
        
        plt.show()
    
    def analyze_reasoning_sequence_patterns(self):
        """Analyze common sequences in reasoning flows"""
        print("\n" + "=" * 60)
        print("REASONING SEQUENCE PATTERN ANALYSIS")
        print("=" * 60)
        
        # Collect all sequences of length 3
        trigrams = []
        for filename, data in self.file_data.items():
            flows = data['argumentation_flow']
            for i in range(len(flows) - 2):
                trigram = (flows[i], flows[i+1], flows[i+2])
                trigrams.append(trigram)
        
        # Find most common trigrams
        trigram_counter = Counter(trigrams)
        common_trigrams = trigram_counter.most_common(10)
        
        print(f"Found {len(trigrams)} total 3-step sequences")
        print(f"Most common reasoning sequences:")
        
        for i, (trigram, count) in enumerate(common_trigrams, 1):
            sequence = " → ".join([step.replace('_', ' ').title() for step in trigram])
            print(f"{i:2d}. {sequence} (appears {count} times)")
        
        # Analyze starting patterns
        starting_patterns = []
        for filename, data in self.file_data.items():
            flows = data['argumentation_flow']
            if flows:
                starting_patterns.append(flows[0])
        
        start_counter = Counter(starting_patterns)
        print(f"\nMost common ways to start reasoning:")
        for i, (start, count) in enumerate(start_counter.most_common(5), 1):
            print(f"{i}. {start.replace('_', ' ').title()} ({count} files)")
        
        # Analyze ending patterns
        ending_patterns = []
        for filename, data in self.file_data.items():
            flows = data['argumentation_flow']
            if flows:
                ending_patterns.append(flows[-1])
        
        end_counter = Counter(ending_patterns)
        print(f"\nMost common ways to end reasoning:")
        for i, (end, count) in enumerate(end_counter.most_common(5), 1):
            print(f"{i}. {end.replace('_', ' ').title()} ({count} files)")
    
    def create_comprehensive_report(self, output_file: str = "reasoning_patterns_report.txt"):
        """Create a comprehensive text report of all patterns"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("LOGICAL STRUCTURE ANALYSIS - COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {self.data.get('ANALYSIS_METADATA', {}).get('generated_at', 'Unknown')}\n")
            f.write(f"Files analyzed: {len(self.file_data)}\n\n")
            
            # Summary statistics
            if 'SUMMARY_STATISTICS' in self.data:
                summary = self.data['SUMMARY_STATISTICS']
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 40 + "\n")
                
                f.write("Average patterns per file:\n")
                for pattern, avg in summary['average_patterns'].items():
                    f.write(f"  {pattern.replace('_', ' ').title()}: {avg}\n")
                
                f.write(f"\nUncertainty distribution:\n")
                unc = summary['uncertainty_distribution']
                total = sum(unc.values())
                if total > 0:
                    f.write(f"  Early phase (0-33%): {unc['early_uncertainty']} ({unc['early_uncertainty']/total*100:.1f}%)\n")
                    f.write(f"  Middle phase (33-67%): {unc['middle_uncertainty']} ({unc['middle_uncertainty']/total*100:.1f}%)\n")
                    f.write(f"  Late phase (67-100%): {unc['late_uncertainty']} ({unc['late_uncertainty']/total*100:.1f}%)\n")
                
                f.write(f"\nDecision complexity:\n")
                comp = summary['decision_complexity_range']
                f.write(f"  Range: {comp['min']} - {comp['max']} decision points\n")
                f.write(f"  Average: {comp['avg']} decision points per file\n")
            
            f.write("\n\nDETAILED FILE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            for filename, data in self.file_data.items():
                f.write(f"\nFile: {filename}\n")
                f.write("  Pattern counts:\n")
                for pattern_type in ['uncertainty_markers', 'logical_connectors', 
                                   'evidence_evaluations', 'contradictions', 
                                   'conclusions', 'decision_points']:
                    count = len(data[pattern_type])
                    f.write(f"    {pattern_type.replace('_', ' ').title()}: {count}\n")
                
                f.write("  Argumentation flow:\n")
                flow = data['argumentation_flow']
                if len(flow) > 10:
                    f.write(f"    {' → '.join(flow[:5])} → ... → {' → '.join(flow[-3:])}\n")
                else:
                    f.write(f"    {' → '.join(flow)}\n")
        
        print(f"Comprehensive report saved to {output_file}")

def main():
    """Main function to run pattern visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize reasoning patterns from analysis data")
    parser.add_argument("analysis_file", help="JSON file from analyze_reasoning_structure.py")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--output-prefix", default="reasoning_patterns", 
                       help="Prefix for output files")
    
    args = parser.parse_args()
    
    try:
        visualizer = ReasoningPatternVisualizer(args.analysis_file)
        
        # Create visualizations
        print("Creating uncertainty distribution plot...")
        save_file = f"{args.output_prefix}_uncertainty.png" if args.save_plots else None
        visualizer.plot_uncertainty_distribution(save_file)
        
        print("Creating argumentation flow patterns plot...")
        save_file = f"{args.output_prefix}_flow.png" if args.save_plots else None
        visualizer.plot_argumentation_flow_patterns(save_file)
        
        print("Creating pattern complexity matrix...")
        save_file = f"{args.output_prefix}_complexity.png" if args.save_plots else None
        visualizer.plot_pattern_complexity_matrix(save_file)
        
        # Analyze sequence patterns
        visualizer.analyze_reasoning_sequence_patterns()
        
        # Create comprehensive report
        visualizer.create_comprehensive_report(f"{args.output_prefix}_report.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()