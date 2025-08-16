#!/usr/bin/env python3
"""
Logical Structure Analyzer for Stream of Thought Outputs

This script analyzes the logical structure and reasoning patterns in
STREAM OF THOUGHT files, independent of biological/domain-specific content.

It looks for:
1. Logical reasoning patterns (if-then, cause-effect, contradiction, comparison)
2. Uncertainty markers and confidence levels
3. Evidence evaluation patterns
4. Decision-making processes
5. Argumentation structure
"""

import os
import re
import json
import glob
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ReasoningPattern:
    """Represents a detected reasoning pattern"""
    pattern_type: str
    text_span: str
    confidence: float
    position: int  # Position in text (0-1)
    context: str   # Surrounding context

@dataclass
class LogicalStructure:
    """Represents the overall logical structure of a reasoning trace"""
    uncertainty_markers: List[ReasoningPattern]
    logical_connectors: List[ReasoningPattern]
    evidence_evaluations: List[ReasoningPattern]
    contradictions: List[ReasoningPattern]
    comparisons: List[ReasoningPattern]
    conclusions: List[ReasoningPattern]
    decision_points: List[ReasoningPattern]
    argumentation_flow: List[str]  # High-level argument flow

class ReasoningStructureAnalyzer:
    """Analyzes logical structure in reasoning traces"""
    
    def __init__(self):
        self.uncertainty_patterns = [
            # Uncertainty markers
            r"(?:hmm|well|actually|wait|but then again|I'm not sure|might be|could be|seems like|appears to|I think|maybe|perhaps|possibly|likely|probably)",
            r"(?:I'm torn|I'm conflicted|I'm hesitant|I'm second-guessing|I keep going back and forth)",
            r"(?:on one hand|on the other hand|however|although|but|yet|still)",
            r"(?:let me reconsider|let me think again|actually, wait|hold on|that makes me wonder)"
        ]
        
        self.logical_connectors = [
            # Causal reasoning
            r"(?:because|since|therefore|thus|hence|as a result|consequently|due to|leads to|causes)",
            r"(?:if.*then|when.*then|given that|assuming|provided that)",
            # Comparative reasoning  
            r"(?:compared to|in contrast|unlike|similar to|just like|analogous to|whereas|while)",
            r"(?:better than|worse than|more.*than|less.*than|superior to|inferior to)",
            # Evidence-based reasoning
            r"(?:the evidence shows|this suggests|this indicates|this implies|this demonstrates)",
            r"(?:supported by|contradicted by|consistent with|inconsistent with)"
        ]
        
        self.evidence_patterns = [
            r"(?:the.*shows|the.*indicates|the.*suggests|the.*demonstrates|the.*proves)",
            r"(?:based on|according to|research shows|studies show|evidence suggests)",
            r"(?:we know that|it's established that|it's clear that|obviously|clearly)",
            r"(?:the fact that|given that|considering that|taking into account)"
        ]
        
        self.contradiction_patterns = [
            r"(?:but.*contradicts|this contradicts|that's wrong|that doesn't make sense|that can't be right)",
            r"(?:actually, no|wait, that's not right|I was wrong|I made an error|that's backwards)",
            r"(?:opposite|contrary to|against|conflicts with|inconsistent)"
        ]
        
        self.conclusion_patterns = [
            r"(?:so I conclude|therefore I think|I'm confident|I believe|my answer is)",
            r"(?:ultimately|finally|in conclusion|to sum up|the answer is|I settle on)",
            r"(?:I'm going with|I choose|I select|my final answer|I decide)"
        ]
        
        self.decision_patterns = [
            r"(?:I need to decide|I have to choose|I must pick|the question is|the key is)",
            r"(?:the critical factor|the main issue|the deciding factor|what matters most)",
            r"(?:I'm leaning toward|I favor|I prefer|I'm inclined to think)"
        ]
    
    def extract_stream_content(self, file_path: str) -> str:
        """Extract just the stream of thought content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the stream of thought section
            stream_start = content.find("STREAM OF THOUGHT:")
            if stream_start == -1:
                return ""
            
            # Find the end (before the footer)
            stream_end = content.find("=" * 80, stream_start + 1)
            if stream_end == -1:
                stream_content = content[stream_start:]
            else:
                stream_content = content[stream_start:stream_end]
            
            # Remove the "STREAM OF THOUGHT:" header and dashes
            lines = stream_content.split('\n')
            clean_lines = []
            skip_header = True
            
            for line in lines:
                if skip_header and ("STREAM OF THOUGHT:" in line or line.strip() == "-" * 40):
                    continue
                skip_header = False
                clean_lines.append(line)
            
            return '\n'.join(clean_lines).strip()
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def find_patterns(self, text: str, patterns: List[str], pattern_type: str) -> List[ReasoningPattern]:
        """Find all instances of reasoning patterns in text"""
        found_patterns = []
        text_length = len(text)
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                # Get surrounding context (50 chars before and after)
                context_start = max(0, start - 50)
                context_end = min(text_length, end + 50)
                context = text[context_start:context_end].replace('\n', ' ')
                
                # Calculate confidence based on pattern specificity
                confidence = min(1.0, len(match.group()) / 20.0)
                
                pattern_obj = ReasoningPattern(
                    pattern_type=pattern_type,
                    text_span=match.group(),
                    confidence=confidence,
                    position=start / text_length,
                    context=context
                )
                found_patterns.append(pattern_obj)
        
        return found_patterns
    
    def analyze_argumentation_flow(self, text: str) -> List[str]:
        """Analyze the high-level flow of argumentation"""
        sentences = re.split(r'[.!?]+', text)
        flow = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
                
            # Classify sentence type based on content
            if re.search(r'(?:let me|I need to|I should)', sentence, re.IGNORECASE):
                flow.append("INITIAL_APPROACH")
            elif re.search(r'(?:first|initially|to start)', sentence, re.IGNORECASE):
                flow.append("OPTION_INTRODUCTION")
            elif re.search(r'(?:but|however|on the other hand)', sentence, re.IGNORECASE):
                flow.append("COUNTER_CONSIDERATION")
            elif re.search(r'(?:actually|wait|hmm)', sentence, re.IGNORECASE):
                flow.append("RECONSIDERATION")
            elif re.search(r'(?:so|therefore|thus|hence)', sentence, re.IGNORECASE):
                flow.append("LOGICAL_INFERENCE")
            elif re.search(r'(?:I think|I believe|I conclude)', sentence, re.IGNORECASE):
                flow.append("CONCLUSION")
            elif re.search(r'(?:because|since|due to)', sentence, re.IGNORECASE):
                flow.append("CAUSAL_REASONING")
            elif re.search(r'(?:compared to|unlike|similar)', sentence, re.IGNORECASE):
                flow.append("COMPARISON")
            else:
                flow.append("ELABORATION")
        
        return flow
    
    def analyze_file(self, file_path: str) -> LogicalStructure:
        """Analyze a single stream of thought file"""
        stream_content = self.extract_stream_content(file_path)
        if not stream_content:
            return LogicalStructure([], [], [], [], [], [], [], [])
        
        # Find all pattern types
        uncertainty_markers = self.find_patterns(stream_content, self.uncertainty_patterns, "UNCERTAINTY")
        logical_connectors = self.find_patterns(stream_content, self.logical_connectors, "LOGICAL_CONNECTOR")
        evidence_evaluations = self.find_patterns(stream_content, self.evidence_patterns, "EVIDENCE")
        contradictions = self.find_patterns(stream_content, self.contradiction_patterns, "CONTRADICTION")
        conclusions = self.find_patterns(stream_content, self.conclusion_patterns, "CONCLUSION")
        decision_points = self.find_patterns(stream_content, self.decision_patterns, "DECISION")
        
        # Analyze argumentation flow
        argumentation_flow = self.analyze_argumentation_flow(stream_content)
        
        return LogicalStructure(
            uncertainty_markers=uncertainty_markers,
            logical_connectors=logical_connectors,
            evidence_evaluations=evidence_evaluations,
            contradictions=contradictions,
            comparisons=[],  # Will implement separately if needed
            conclusions=conclusions,
            decision_points=decision_points,
            argumentation_flow=argumentation_flow
        )
    
    def analyze_directory(self, directory_path: str) -> Dict[str, LogicalStructure]:
        """Analyze all stream analysis files in a directory"""
        results = {}
        
        # Find all stream analysis files
        pattern = os.path.join(directory_path, "*_STREAM_ANALYSIS.txt")
        files = glob.glob(pattern)
        
        print(f"Found {len(files)} stream analysis files in {directory_path}")
        
        for file_path in files:
            filename = os.path.basename(file_path)
            print(f"Analyzing {filename}...")
            
            structure = self.analyze_file(file_path)
            results[filename] = structure
        
        return results
    
    def generate_summary_report(self, analyses: Dict[str, LogicalStructure]) -> Dict[str, Any]:
        """Generate a summary report of logical patterns across all files"""
        if not analyses:
            return {}
        
        # Aggregate statistics
        total_files = len(analyses)
        uncertainty_counts = Counter()
        connector_counts = Counter()
        evidence_counts = Counter()
        conclusion_counts = Counter()
        flow_patterns = Counter()
        
        uncertainty_by_position = []
        decision_complexity = []  # Number of decision points per file
        
        for filename, structure in analyses.items():
            # Count pattern types
            uncertainty_counts[len(structure.uncertainty_markers)] += 1
            connector_counts[len(structure.logical_connectors)] += 1
            evidence_counts[len(structure.evidence_evaluations)] += 1
            conclusion_counts[len(structure.conclusions)] += 1
            
            # Track decision complexity
            decision_complexity.append(len(structure.decision_points))
            
            # Track uncertainty distribution by position
            for marker in structure.uncertainty_markers:
                uncertainty_by_position.append(marker.position)
            
            # Track argumentation flow patterns
            flow_sequence = "->".join(structure.argumentation_flow[:10])  # First 10 steps
            flow_patterns[flow_sequence] += 1
        
        # Calculate averages
        avg_uncertainty = sum(i * count for i, count in uncertainty_counts.items()) / total_files
        avg_connectors = sum(i * count for i, count in connector_counts.items()) / total_files
        avg_evidence = sum(i * count for i, count in evidence_counts.items()) / total_files
        avg_conclusions = sum(i * count for i, count in conclusion_counts.items()) / total_files
        avg_decisions = sum(decision_complexity) / total_files if decision_complexity else 0
        
        # Find most common flow patterns
        common_flows = flow_patterns.most_common(5)
        
        # Analyze uncertainty distribution
        uncertainty_stats = {
            "early_uncertainty": len([pos for pos in uncertainty_by_position if pos < 0.33]),
            "middle_uncertainty": len([pos for pos in uncertainty_by_position if 0.33 <= pos < 0.67]),
            "late_uncertainty": len([pos for pos in uncertainty_by_position if pos >= 0.67])
        }
        
        return {
            "total_files_analyzed": total_files,
            "average_patterns": {
                "uncertainty_markers": round(avg_uncertainty, 2),
                "logical_connectors": round(avg_connectors, 2),  
                "evidence_evaluations": round(avg_evidence, 2),
                "conclusions": round(avg_conclusions, 2),
                "decision_points": round(avg_decisions, 2)
            },
            "uncertainty_distribution": uncertainty_stats,
            "common_argumentation_flows": common_flows,
            "decision_complexity_range": {
                "min": min(decision_complexity) if decision_complexity else 0,
                "max": max(decision_complexity) if decision_complexity else 0,
                "avg": round(avg_decisions, 2)
            }
        }
    
    def save_analysis_report(self, analyses: Dict[str, LogicalStructure], output_file: str):
        """Save detailed analysis report to JSON file"""
        # Convert dataclasses to dictionaries for JSON serialization
        json_data = {}
        
        for filename, structure in analyses.items():
            json_data[filename] = {
                "uncertainty_markers": [
                    {
                        "type": p.pattern_type,
                        "text": p.text_span,
                        "confidence": p.confidence,
                        "position": p.position,
                        "context": p.context
                    } for p in structure.uncertainty_markers
                ],
                "logical_connectors": [
                    {
                        "type": p.pattern_type,
                        "text": p.text_span,
                        "confidence": p.confidence,
                        "position": p.position,
                        "context": p.context
                    } for p in structure.logical_connectors
                ],
                "evidence_evaluations": [
                    {
                        "type": p.pattern_type,
                        "text": p.text_span,
                        "confidence": p.confidence,
                        "position": p.position,
                        "context": p.context
                    } for p in structure.evidence_evaluations
                ],
                "contradictions": [
                    {
                        "type": p.pattern_type,
                        "text": p.text_span,
                        "confidence": p.confidence,
                        "position": p.position,
                        "context": p.context
                    } for p in structure.contradictions
                ],
                "conclusions": [
                    {
                        "type": p.pattern_type,
                        "text": p.text_span,
                        "confidence": p.confidence,
                        "position": p.position,
                        "context": p.context
                    } for p in structure.conclusions
                ],
                "decision_points": [
                    {
                        "type": p.pattern_type,
                        "text": p.text_span,
                        "confidence": p.confidence,
                        "position": p.position,
                        "context": p.context
                    } for p in structure.decision_points
                ],
                "argumentation_flow": structure.argumentation_flow
            }
        
        # Add summary statistics
        summary = self.generate_summary_report(analyses)
        json_data["SUMMARY_STATISTICS"] = summary
        json_data["ANALYSIS_METADATA"] = {
            "generated_at": datetime.now().isoformat(),
            "analyzer_version": "1.0",
            "files_processed": len(analyses)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed analysis saved to {output_file}")

def main():
    """Main function to run the reasoning structure analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze logical structure in Stream of Thought files")
    parser.add_argument("directory", help="Directory containing *_STREAM_ANALYSIS.txt files")
    parser.add_argument("--output", "-o", default="reasoning_structure_analysis.json", 
                       help="Output file for detailed analysis (default: reasoning_structure_analysis.json)")
    parser.add_argument("--summary-only", action="store_true", 
                       help="Only show summary statistics, don't save detailed analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist")
        return
    
    # Run analysis
    analyzer = ReasoningStructureAnalyzer()
    analyses = analyzer.analyze_directory(args.directory)
    
    if not analyses:
        print("No stream analysis files found!")
        return
    
    # Generate and display summary
    summary = analyzer.generate_summary_report(analyses)
    
    print("\n" + "=" * 80)
    print("LOGICAL STRUCTURE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Files analyzed: {summary['total_files_analyzed']}")
    print(f"\nAverage patterns per file:")
    for pattern_type, avg_count in summary['average_patterns'].items():
        print(f"  {pattern_type.replace('_', ' ').title()}: {avg_count}")
    
    print(f"\nUncertainty distribution:")
    uncertainty = summary['uncertainty_distribution']
    total_uncertainty = sum(uncertainty.values())
    if total_uncertainty > 0:
        print(f"  Early (0-33%): {uncertainty['early_uncertainty']} ({uncertainty['early_uncertainty']/total_uncertainty*100:.1f}%)")
        print(f"  Middle (33-67%): {uncertainty['middle_uncertainty']} ({uncertainty['middle_uncertainty']/total_uncertainty*100:.1f}%)")
        print(f"  Late (67-100%): {uncertainty['late_uncertainty']} ({uncertainty['late_uncertainty']/total_uncertainty*100:.1f}%)")
    
    print(f"\nDecision complexity:")
    complexity = summary['decision_complexity_range']
    print(f"  Range: {complexity['min']} - {complexity['max']} decision points")
    print(f"  Average: {complexity['avg']} decision points per file")
    
    print(f"\nMost common argumentation flow patterns:")
    for i, (flow, count) in enumerate(summary['common_argumentation_flows'], 1):
        print(f"  {i}. {flow} (appears {count} times)")
    
    # Save detailed analysis unless summary-only
    if not args.summary_only:
        analyzer.save_analysis_report(analyses, args.output)
        print(f"\nDetailed analysis available in: {args.output}")

if __name__ == "__main__":
    main()