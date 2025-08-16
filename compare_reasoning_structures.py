#!/usr/bin/env python3
"""
Reasoning Structure Comparator

This script compares logical reasoning structures across different conditions,
such as different scientific domains, question types, or experimental conditions.
It helps identify universal vs domain-specific reasoning patterns.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from scipy import stats
import matplotlib.pyplot as plt

class ReasoningStructureComparator:
    """Compares reasoning structures across different groups/conditions"""
    
    def __init__(self):
        self.groups = {}  # Group name -> analysis data
        
    def add_group(self, group_name: str, analysis_file: str):
        """Add a group of reasoning analyses to compare"""
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Remove metadata entries
        file_data = {k: v for k, v in data.items() 
                    if not k.startswith(('SUMMARY_', 'ANALYSIS_'))}
        
        self.groups[group_name] = file_data
        print(f"Added group '{group_name}' with {len(file_data)} files")
    
    def extract_features(self, group_data: Dict) -> Dict[str, List[float]]:
        """Extract numerical features from a group's reasoning data"""
        features = defaultdict(list)
        
        for filename, data in group_data.items():
            # Basic pattern counts
            features['uncertainty_count'].append(len(data['uncertainty_markers']))
            features['logical_connector_count'].append(len(data['logical_connectors']))
            features['evidence_count'].append(len(data['evidence_evaluations']))
            features['contradiction_count'].append(len(data['contradictions']))
            features['conclusion_count'].append(len(data['conclusions']))
            features['decision_count'].append(len(data['decision_points']))
            
            # Flow characteristics
            flow = data['argumentation_flow']
            features['flow_length'].append(len(flow))
            
            # Calculate flow diversity (unique step types / total steps)
            if len(flow) > 0:
                unique_steps = len(set(flow))
                features['flow_diversity'].append(unique_steps / len(flow))
            else:
                features['flow_diversity'].append(0)
            
            # Uncertainty characteristics
            uncertainty_positions = [marker['position'] for marker in data['uncertainty_markers']]
            if uncertainty_positions:
                features['uncertainty_mean_position'].append(np.mean(uncertainty_positions))
                features['uncertainty_std_position'].append(np.std(uncertainty_positions))
                features['early_uncertainty_ratio'].append(
                    len([pos for pos in uncertainty_positions if pos < 0.33]) / len(uncertainty_positions)
                )
            else:
                features['uncertainty_mean_position'].append(0.5)  # Default to middle
                features['uncertainty_std_position'].append(0)
                features['early_uncertainty_ratio'].append(0)
            
            # Flow pattern analysis
            flow_counter = Counter(flow)
            if flow:
                # Most common pattern frequency
                most_common_freq = flow_counter.most_common(1)[0][1] / len(flow)
                features['flow_repetition_ratio'].append(most_common_freq)
            else:
                features['flow_repetition_ratio'].append(0)
            
            # Calculate conclusion efficiency (conclusions per total patterns)
            total_patterns = sum([
                len(data['uncertainty_markers']),
                len(data['logical_connectors']),
                len(data['evidence_evaluations']),
                len(data['contradictions']),
                len(data['decision_points'])
            ])
            if total_patterns > 0:
                features['conclusion_efficiency'].append(len(data['conclusions']) / total_patterns)
            else:
                features['conclusion_efficiency'].append(0)
        
        return dict(features)
    
    def compare_groups(self) -> Dict[str, Any]:
        """Compare all groups and identify significant differences"""
        if len(self.groups) < 2:
            raise ValueError("Need at least 2 groups to compare")
        
        results = {
            'group_statistics': {},
            'pairwise_comparisons': {},
            'significant_differences': []
        }
        
        # Extract features for each group
        group_features = {}
        for group_name, group_data in self.groups.items():
            group_features[group_name] = self.extract_features(group_data)
        
        # Calculate statistics for each group
        for group_name, features in group_features.items():
            stats_dict = {}
            for feature_name, values in features.items():
                stats_dict[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n': len(values)
                }
            results['group_statistics'][group_name] = stats_dict
        
        # Pairwise comparisons
        group_names = list(self.groups.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1, group2 = group_names[i], group_names[j]
                comparison_key = f"{group1}_vs_{group2}"
                
                comparison_results = {}
                features1 = group_features[group1]
                features2 = group_features[group2]
                
                # Compare each feature
                for feature_name in features1.keys():
                    values1 = features1[feature_name]
                    values2 = features2[feature_name]
                    
                    # Perform statistical tests
                    # Mann-Whitney U test (non-parametric)
                    try:
                        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                        
                        # Effect size (Cohen's d approximation)
                        pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
                        if pooled_std > 0:
                            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
                        else:
                            cohens_d = 0
                        
                        comparison_results[feature_name] = {
                            'mann_whitney_u': statistic,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'mean_diff': np.mean(values1) - np.mean(values2),
                            'significant': p_value < 0.05
                        }
                        
                        # Track significant differences
                        if p_value < 0.05:
                            results['significant_differences'].append({
                                'groups': comparison_key,
                                'feature': feature_name,
                                'p_value': p_value,
                                'effect_size': abs(cohens_d),
                                'direction': 'higher' if np.mean(values1) > np.mean(values2) else 'lower',
                                'group1_mean': np.mean(values1),
                                'group2_mean': np.mean(values2)
                            })
                    
                    except Exception as e:
                        comparison_results[feature_name] = {
                            'error': str(e),
                            'significant': False
                        }
                
                results['pairwise_comparisons'][comparison_key] = comparison_results
        
        # Sort significant differences by effect size
        results['significant_differences'].sort(key=lambda x: x.get('effect_size', 0), reverse=True)
        
        return results
    
    def analyze_flow_pattern_differences(self) -> Dict[str, Any]:
        """Analyze differences in argumentation flow patterns between groups"""
        flow_analysis = {}
        
        for group_name, group_data in self.groups.items():
            # Collect all flow patterns
            all_flows = []
            for filename, data in group_data.items():
                all_flows.extend(data['argumentation_flow'])
            
            # Count pattern frequencies
            flow_counter = Counter(all_flows)
            total_steps = len(all_flows)
            
            # Calculate pattern probabilities
            pattern_probs = {pattern: count/total_steps for pattern, count in flow_counter.items()}
            
            flow_analysis[group_name] = {
                'total_steps': total_steps,
                'unique_patterns': len(flow_counter),
                'pattern_frequencies': dict(flow_counter),
                'pattern_probabilities': pattern_probs,
                'most_common': flow_counter.most_common(10)
            }
        
        # Find patterns that differ significantly between groups
        if len(self.groups) >= 2:
            group_names = list(self.groups.keys())
            pattern_differences = []
            
            # Get all unique patterns across groups
            all_patterns = set()
            for group_analysis in flow_analysis.values():
                all_patterns.update(group_analysis['pattern_probabilities'].keys())
            
            # Compare pattern frequencies
            for pattern in all_patterns:
                group_probs = []
                for group_name in group_names:
                    prob = flow_analysis[group_name]['pattern_probabilities'].get(pattern, 0)
                    group_probs.append(prob)
                
                # Calculate variance in pattern usage
                if max(group_probs) > 0:  # Only consider patterns that appear
                    variance = np.var(group_probs)
                    max_diff = max(group_probs) - min(group_probs)
                    
                    pattern_differences.append({
                        'pattern': pattern,
                        'variance': variance,
                        'max_difference': max_diff,
                        'group_probabilities': dict(zip(group_names, group_probs))
                    })
            
            # Sort by variance (patterns that differ most between groups)
            pattern_differences.sort(key=lambda x: x['variance'], reverse=True)
            flow_analysis['pattern_differences'] = pattern_differences[:20]  # Top 20
        
        return flow_analysis
    
    def create_comparison_report(self, output_file: str = "reasoning_comparison_report.txt"):
        """Create a comprehensive comparison report"""
        comparison_results = self.compare_groups()
        flow_analysis = self.analyze_flow_pattern_differences()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("REASONING STRUCTURE COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Group overview
            f.write("GROUPS ANALYZED:\n")
            f.write("-" * 40 + "\n")
            for group_name, group_data in self.groups.items():
                f.write(f"{group_name}: {len(group_data)} files\n")
            
            # Significant differences
            f.write(f"\nSIGNIFICANT DIFFERENCES (p < 0.05):\n")
            f.write("-" * 40 + "\n")
            
            sig_diffs = comparison_results['significant_differences']
            if sig_diffs:
                for i, diff in enumerate(sig_diffs[:15], 1):  # Top 15
                    feature = diff['feature'].replace('_', ' ').title()
                    f.write(f"{i:2d}. {feature}\n")
                    f.write(f"    Groups: {diff['groups']}\n")
                    f.write(f"    Effect size: {diff['effect_size']:.3f}\n")
                    f.write(f"    P-value: {diff['p_value']:.6f}\n")
                    f.write(f"    {diff['groups'].split('_vs_')[0]} mean: {diff['group1_mean']:.3f}\n")
                    f.write(f"    {diff['groups'].split('_vs_')[1]} mean: {diff['group2_mean']:.3f}\n")
                    f.write(f"    Direction: {diff['groups'].split('_vs_')[0]} is {diff['direction']}\n\n")
            else:
                f.write("No statistically significant differences found.\n")
            
            # Group statistics summary
            f.write("\nGROUP STATISTICS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            stats = comparison_results['group_statistics']
            if stats:
                # Get feature names from first group
                first_group = list(stats.keys())[0]
                feature_names = list(stats[first_group].keys())
                
                for feature in feature_names:
                    f.write(f"\n{feature.replace('_', ' ').title()}:\n")
                    for group_name in stats.keys():
                        group_stats = stats[group_name][feature]
                        f.write(f"  {group_name}: {group_stats['mean']:.3f} ± {group_stats['std']:.3f} "
                               f"(range: {group_stats['min']:.3f}-{group_stats['max']:.3f})\n")
            
            # Flow pattern differences
            f.write(f"\nARGUMENTATION FLOW PATTERN ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            for group_name, analysis in flow_analysis.items():
                if group_name != 'pattern_differences':
                    f.write(f"\n{group_name}:\n")
                    f.write(f"  Total reasoning steps: {analysis['total_steps']}\n")
                    f.write(f"  Unique patterns: {analysis['unique_patterns']}\n")
                    f.write(f"  Most common patterns:\n")
                    for pattern, count in analysis['most_common'][:5]:
                        pct = count / analysis['total_steps'] * 100
                        f.write(f"    {pattern.replace('_', ' ').title()}: {count} ({pct:.1f}%)\n")
            
            if 'pattern_differences' in flow_analysis:
                f.write(f"\nPATTERNS WITH LARGEST GROUP DIFFERENCES:\n")
                f.write("-" * 40 + "\n")
                
                for i, diff in enumerate(flow_analysis['pattern_differences'][:10], 1):
                    f.write(f"{i:2d}. {diff['pattern'].replace('_', ' ').title()}\n")
                    f.write(f"    Variance across groups: {diff['variance']:.4f}\n")
                    f.write(f"    Max difference: {diff['max_difference']:.3f}\n")
                    for group, prob in diff['group_probabilities'].items():
                        f.write(f"    {group}: {prob:.3f}\n")
                    f.write("\n")
        
        print(f"Comparison report saved to {output_file}")
    
    def plot_feature_comparison(self, feature_name: str, save_file: str = None):
        """Plot comparison of a specific feature across groups"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        group_names = []
        feature_values = []
        
        for group_name, group_data in self.groups.items():
            features = self.extract_features(group_data)
            if feature_name in features:
                group_names.append(group_name)
                feature_values.append(features[feature_name])
        
        if not feature_values:
            print(f"Feature '{feature_name}' not found in any group")
            return
        
        # Create box plot
        bp = ax.boxplot(feature_values, labels=group_names, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'wheat', 'plum']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax.set_ylabel(feature_name.replace('_', ' ').title())
        ax.set_title(f'Comparison of {feature_name.replace("_", " ").title()} Across Groups')
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotations if there are exactly 2 groups
        if len(feature_values) == 2:
            try:
                statistic, p_value = stats.mannwhitneyu(feature_values[0], feature_values[1])
                ax.text(0.5, 0.95, f'Mann-Whitney U p-value: {p_value:.4f}', 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Feature comparison plot saved to {save_file}")
        
        plt.show()

def main():
    """Main function for reasoning structure comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare reasoning structures between groups")
    parser.add_argument("--group", action="append", nargs=2, metavar=("NAME", "FILE"),
                       help="Add a group: --group group_name analysis_file.json")
    parser.add_argument("--output", "-o", default="reasoning_comparison_report.txt",
                       help="Output file for comparison report")
    parser.add_argument("--plot-feature", help="Create plot for specific feature")
    parser.add_argument("--save-plot", help="Save plot to file")
    
    args = parser.parse_args()
    
    if not args.group or len(args.group) < 2:
        print("Error: Need at least 2 groups to compare")
        print("Usage: --group group1 file1.json --group group2 file2.json")
        return
    
    # Initialize comparator
    comparator = ReasoningStructureComparator()
    
    # Add groups
    for group_name, analysis_file in args.group:
        try:
            comparator.add_group(group_name, analysis_file)
        except Exception as e:
            print(f"Error loading group '{group_name}': {e}")
            return
    
    # Create comparison report
    try:
        comparator.create_comparison_report(args.output)
        print(f"\nComparison complete! Report saved to {args.output}")
        
        # Create feature plot if requested
        if args.plot_feature:
            comparator.plot_feature_comparison(args.plot_feature, args.save_plot)
            
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()