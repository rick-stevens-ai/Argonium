#!/usr/bin/env python3
"""
LLM-Based Formal Logic Analyzer

This analyzer uses Large Language Models to identify formal logical argument structures
without relying on regular expressions. It leverages the sophisticated natural language
understanding capabilities of LLMs to detect:

1. Classical logical argument forms (modus ponens, modus tollens, etc.)
2. Propositional logic structures
3. Quantified statements and their relationships
4. Inference patterns and logical validity
5. Argument strength and confidence measures
"""

import json
import os
import sys
import yaml
import openai
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum

class ArgumentType(Enum):
    """Types of logical arguments"""
    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    CONTRAPOSITION = "contraposition"
    REDUCTIO_AD_ABSURDUM = "reductio_ad_absurdum"
    UNIVERSAL_INSTANTIATION = "universal_instantiation"
    EXISTENTIAL_GENERALIZATION = "existential_generalization"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    INDUCTIVE_GENERALIZATION = "inductive_generalization"
    ELIMINATION = "elimination"
    CONFIRMATION = "confirmation"
    COUNTERFACTUAL = "counterfactual"

@dataclass
class LogicalClause:
    """Represents a logical clause or proposition"""
    id: str
    text: str
    normalized_form: str
    predicate: str
    subjects: List[str]
    objects: List[str]
    modifiers: List[str]
    polarity: bool  # True for positive, False for negative
    certainty: float  # 0.0 to 1.0
    temporal_marker: Optional[str]
    conditional_type: Optional[str]
    quantifier: Optional[str]
    sentence_position: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class LogicalArgument:
    """Represents a detected logical argument"""
    id: str
    argument_type: ArgumentType
    premises: List[LogicalClause]
    conclusion: LogicalClause
    intermediate_steps: List[LogicalClause]
    logical_form: str
    natural_language_form: str
    validity: bool
    soundness_estimate: float
    confidence: float
    text_span: Tuple[int, int]
    inference_rules_used: List[str]
    
    def to_dict(self):
        return {
            'id': self.id,
            'argument_type': self.argument_type.value,
            'premises': [clause.to_dict() for clause in self.premises],
            'conclusion': self.conclusion.to_dict(),
            'intermediate_steps': [clause.to_dict() for clause in self.intermediate_steps],
            'logical_form': self.logical_form,
            'natural_language_form': self.natural_language_form,
            'validity': self.validity,
            'soundness_estimate': self.soundness_estimate,
            'confidence': self.confidence,
            'text_span': self.text_span,
            'inference_rules_used': self.inference_rules_used
        }

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
        # Fallback for older OpenAI library
        openai.api_key = api_key
        openai.api_base = selected_server.get("openai_api_base")
        client = None
    
    actual_model_name = selected_server.get("openai_model", model_name)
    return actual_model_name, client

class LLMLogicalAnalyzer:
    """Uses LLMs to analyze logical structures in text"""
    
    def __init__(self, model_name: str = "gpt-4", openai_client=None):
        self.model_name = model_name
        self.client = openai_client
        
        # Fallback to old method if no client provided
        if not self.client:
            if not openai.api_key:
                openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    def extract_logical_clauses(self, text: str) -> List[LogicalClause]:
        """Extract logical clauses using LLM analysis"""
        
        prompt = f"""Analyze the following reasoning text and extract logical clauses/propositions.
For each sentence or logical unit, identify:
1. The main predicate (verb or relationship)
2. The subjects (who/what the statement is about)
3. The objects (what the action is done to)
4. Modifiers (adjectives, adverbs, prepositional phrases)
5. Polarity (positive or negative statement)
6. Certainty level (0.0-1.0 based on language used)
7. Temporal markers (when, before, after, etc.)
8. Conditional type (if-then, when, unless, etc.)
9. Quantifiers (all, some, most, etc.)

Return a JSON list where each item represents a logical clause with these fields:
- text: the original text
- predicate: main verb/relationship
- subjects: list of subject entities
- objects: list of object entities  
- modifiers: list of modifying phrases
- polarity: true for positive, false for negative
- certainty: float 0.0-1.0
- temporal_marker: string or null
- conditional_type: string or null
- quantifier: string or null

Text to analyze:
{text}

Respond with valid JSON only."""

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
            
            # Clean markdown code block wrappers if present
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ``` 
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
                
            # Handle cases where there's JSON followed by explanation
            # Look for the end of the JSON array/object
            content = content.strip()
            if content.startswith('['):
                # Find the matching closing bracket
                bracket_count = 0
                json_end = -1
                for i, char in enumerate(content):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    content = content[:json_end]
            elif content.startswith('{'):
                # Find the matching closing brace
                brace_count = 0
                json_end = -1
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    content = content[:json_end]
            
            content = content.strip()
            
            result = json.loads(content)
            
            clauses = []
            for i, clause_data in enumerate(result):
                clause = LogicalClause(
                    id=f"C{i}",
                    text=clause_data.get("text", ""),
                    normalized_form=clause_data.get("text", "").lower().strip(),
                    predicate=clause_data.get("predicate", ""),
                    subjects=clause_data.get("subjects", []),
                    objects=clause_data.get("objects", []),
                    modifiers=clause_data.get("modifiers", []),
                    polarity=clause_data.get("polarity", True),
                    certainty=clause_data.get("certainty", 0.7),
                    temporal_marker=clause_data.get("temporal_marker"),
                    conditional_type=clause_data.get("conditional_type"),
                    quantifier=clause_data.get("quantifier"),
                    sentence_position=i
                )
                clauses.append(clause)
            
            return clauses
            
        except Exception as e:
            print(f"Error in clause extraction: {e}")
            return []
    
    def detect_logical_arguments(self, clauses: List[LogicalClause], original_text: str) -> List[LogicalArgument]:
        """Detect logical arguments using LLM analysis"""
        
        # Prepare clause information for the LLM
        clause_info = []
        for clause in clauses:
            clause_info.append({
                "id": clause.id,
                "text": clause.text,
                "predicate": clause.predicate,
                "subjects": clause.subjects,
                "objects": clause.objects,
                "polarity": clause.polarity,
                "certainty": clause.certainty,
                "conditional_type": clause.conditional_type,
                "position": clause.sentence_position
            })
        
        prompt = f"""You are a formal logic expert. Analyze the following reasoning text and extracted clauses to identify logical argument structures.

ORIGINAL TEXT:
{original_text}

EXTRACTED CLAUSES:
{json.dumps(clause_info, indent=2)}

Identify logical arguments of these types:
1. MODUS_PONENS: If P then Q, P, therefore Q
2. MODUS_TOLLENS: If P then Q, not Q, therefore not P  
3. DISJUNCTIVE_SYLLOGISM: P or Q, not P, therefore Q
4. HYPOTHETICAL_SYLLOGISM: If P then Q, if Q then R, therefore if P then R
5. ELIMINATION: Multiple options considered, some eliminated, remaining option chosen
6. ABDUCTIVE: Best explanation reasoning (inference to most likely cause)
7. CAUSAL: X causes Y, structure-property relationships
8. REDUCTIO_AD_ABSURDUM: Assume opposite, derive contradiction, conclude original
9. ANALOGICAL: A is like B, A has property P, so B probably has P
10. CONFIRMATION: Evidence supports hypothesis

For each argument found, provide:
- argument_type: one of the types above
- premise_clause_ids: list of clause IDs that serve as premises
- conclusion_clause_id: single clause ID that is the conclusion
- logical_form: formal symbolic representation (use →, ∧, ∨, ¬, ∀, ∃)
- natural_language_form: clear English description of the argument
- validity: true if deductively valid, false otherwise
- confidence: 0.0-1.0 confidence in the argument detection
- inference_rules: list of logical rules used

Focus on the actual logical structure and reasoning patterns, not just keyword matching.
Pay special attention to:
- Option elimination reasoning patterns
- Conditional statements and their consequences
- Causal relationships between structure and properties
- Requirements leading to conclusions about suitability

Return a JSON list of detected arguments."""

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
            
            # Clean markdown code block wrappers if present
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ``` 
            if content.endswith('```'):
                content = content[:-3]  # Remove trailing ```
                
            # Handle cases where there's JSON followed by explanation
            # Look for the end of the JSON array/object
            content = content.strip()
            if content.startswith('['):
                # Find the matching closing bracket
                bracket_count = 0
                json_end = -1
                for i, char in enumerate(content):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    content = content[:json_end]
            elif content.startswith('{'):
                # Find the matching closing brace
                brace_count = 0
                json_end = -1
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    content = content[:json_end]
            
            content = content.strip()
            
            result = json.loads(content)
            
            arguments = []
            for i, arg_data in enumerate(result):
                # Get premise and conclusion clauses
                premise_ids = arg_data.get("premise_clause_ids", [])
                conclusion_id = arg_data.get("conclusion_clause_id", "")
                
                premises = [c for c in clauses if c.id in premise_ids]
                conclusion = next((c for c in clauses if c.id == conclusion_id), None)
                
                if not premises or not conclusion:
                    continue
                
                argument_type = ArgumentType(arg_data.get("argument_type", "abductive").lower())
                
                argument = LogicalArgument(
                    id=f"ARG_{i}",
                    argument_type=argument_type,
                    premises=premises,
                    conclusion=conclusion,
                    intermediate_steps=[],
                    logical_form=arg_data.get("logical_form", ""),
                    natural_language_form=arg_data.get("natural_language_form", ""),
                    validity=arg_data.get("validity", False),
                    soundness_estimate=min([p.certainty for p in premises] + [conclusion.certainty]),
                    confidence=arg_data.get("confidence", 0.5),
                    text_span=(min(p.sentence_position for p in premises), conclusion.sentence_position),
                    inference_rules_used=arg_data.get("inference_rules", [])
                )
                arguments.append(argument)
            
            return arguments
            
        except Exception as e:
            print(f"Error in argument detection: {e}")
            return []
    
    def assess_argument_quality(self, arguments: List[LogicalArgument], original_text: str) -> Dict[str, Any]:
        """Assess overall argument quality using LLM analysis"""
        
        if not arguments:
            return {
                'overall_validity': 0.0,
                'logical_consistency': 0.0,
                'argument_strength': 0.0,
                'reasoning_quality': 0.0
            }
        
        # Prepare argument summaries
        arg_summaries = []
        for arg in arguments:
            arg_summaries.append({
                "type": arg.argument_type.value,
                "form": arg.logical_form,
                "natural_form": arg.natural_language_form,
                "validity": arg.validity,
                "confidence": arg.confidence
            })
        
        prompt = f"""Assess the overall logical quality of this reasoning based on the detected arguments.

ORIGINAL REASONING:
{original_text}

DETECTED ARGUMENTS:
{json.dumps(arg_summaries, indent=2)}

Provide an assessment with these metrics (0.0-1.0):
1. overall_validity: How logically valid are the arguments overall?
2. logical_consistency: Are the arguments consistent with each other?
3. argument_strength: How strong/convincing are the arguments?
4. reasoning_quality: Overall quality of the reasoning process

Also provide:
5. reasoning_style: Description of the reasoning approach used
6. strengths: List of logical strengths
7. weaknesses: List of logical weaknesses or gaps

Return valid JSON with these fields."""

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)
            else:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error in quality assessment: {e}")
            return {
                'overall_validity': 0.5,
                'logical_consistency': 0.5,
                'argument_strength': 0.5,
                'reasoning_quality': 0.5,
                'reasoning_style': 'Unknown',
                'strengths': [],
                'weaknesses': []
            }

class LLMFormalLogicAnalyzer:
    """Main analyzer class using LLMs"""
    
    def __init__(self, model_name: str = "gpt-4", openai_client=None):
        self.llm_analyzer = LLMLogicalAnalyzer(model_name, openai_client)
        self.model_name = model_name
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single stream file for formal logic structures"""
        stream_content = self._extract_stream_content(file_path)
        if not stream_content:
            return self._empty_analysis(file_path)
        
        print(f"Analyzing stream content ({len(stream_content)} chars)...")
        
        # Extract logical clauses using LLM
        clauses = self.llm_analyzer.extract_logical_clauses(stream_content)
        print(f"Extracted {len(clauses)} logical clauses")
        
        # Detect logical arguments using LLM
        arguments = self.llm_analyzer.detect_logical_arguments(clauses, stream_content)
        print(f"Detected {len(arguments)} logical arguments")
        
        # Assess argument quality using LLM
        quality_assessment = self.llm_analyzer.assess_argument_quality(arguments, stream_content)
        
        # Perform analysis
        analysis = {
            'file_path': file_path,
            'model_used': self.model_name,
            'text_length': len(stream_content),
            'total_arguments': len(arguments),
            'argument_types': self._categorize_arguments(arguments),
            'deductive_arguments': [arg.to_dict() for arg in arguments if arg.validity],
            'non_deductive_arguments': [arg.to_dict() for arg in arguments if not arg.validity],
            'logical_complexity': self._calculate_complexity_metrics(arguments),
            'argument_quality': self._assess_argument_quality(arguments),
            'llm_quality_assessment': quality_assessment,
            'inference_patterns': self._analyze_inference_patterns(arguments),
            'formal_representations': [arg.logical_form for arg in arguments],
            'natural_language_forms': [arg.natural_language_form for arg in arguments],
            'argument_chains': self._detect_argument_chains(arguments),
            'validity_assessment': self._assess_overall_validity(arguments),
            'extracted_clauses': [clause.to_dict() for clause in clauses]
        }
        
        return analysis
    
    def _extract_stream_content(self, file_path: str) -> str:
        """Extract stream content from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find stream section
            stream_start = content.find("STREAM OF THOUGHT:")
            if stream_start == -1:
                return ""
            
            stream_end = content.find("=" * 80, stream_start + 1)
            if stream_end == -1:
                stream_content = content[stream_start:]
            else:
                stream_content = content[stream_start:stream_end]
            
            # Clean up
            lines = stream_content.split('\n')
            clean_lines = []
            skip_header = True
            
            for line in lines:
                if skip_header and ("STREAM OF THOUGHT:" in line or line.strip() == "-" * 40):
                    continue
                skip_header = False
                clean_lines.append(line)
            
            return '\n'.join(clean_lines).strip()
        except Exception:
            return ""
    
    def _empty_analysis(self, file_path: str) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'file_path': file_path,
            'model_used': self.model_name,
            'text_length': 0,
            'total_arguments': 0,
            'argument_types': {},
            'deductive_arguments': [],
            'non_deductive_arguments': [],
            'logical_complexity': {
                'argument_density': 0.0,
                'avg_argument_strength': 0.0,
                'deductive_ratio': 0.0,
                'inference_diversity': 0.0
            },
            'argument_quality': {
                'avg_soundness': 0.0,
                'avg_confidence': 0.0,
                'validity_rate': 0.0
            },
            'llm_quality_assessment': {
                'overall_validity': 0.0,
                'logical_consistency': 0.0,
                'argument_strength': 0.0,
                'reasoning_quality': 0.0
            },
            'inference_patterns': {},
            'formal_representations': [],
            'natural_language_forms': [],
            'argument_chains': [],
            'validity_assessment': {
                'overall_validity': 0.0,
                'logical_consistency': 0.0,
                'argument_strength': 0.0
            },
            'extracted_clauses': []
        }
    
    def _categorize_arguments(self, arguments: List[LogicalArgument]) -> Dict[str, int]:
        """Categorize arguments by type"""
        categories = defaultdict(int)
        for arg in arguments:
            categories[arg.argument_type.value] += 1
        return dict(categories)
    
    def _calculate_complexity_metrics(self, arguments: List[LogicalArgument]) -> Dict[str, float]:
        """Calculate complexity metrics"""
        if not arguments:
            return {
                'argument_density': 0.0,
                'avg_argument_strength': 0.0,
                'deductive_ratio': 0.0,
                'inference_diversity': 0.0
            }
        
        deductive_count = sum(1 for arg in arguments if arg.validity)
        total_args = len(arguments)
        
        return {
            'argument_density': total_args,
            'avg_argument_strength': sum(arg.soundness_estimate for arg in arguments) / total_args,
            'deductive_ratio': deductive_count / total_args,
            'inference_diversity': len(set(arg.argument_type for arg in arguments)) / len(ArgumentType)
        }
    
    def _assess_argument_quality(self, arguments: List[LogicalArgument]) -> Dict[str, float]:
        """Assess overall argument quality"""
        if not arguments:
            return {
                'avg_soundness': 0.0,
                'avg_confidence': 0.0,
                'validity_rate': 0.0
            }
        
        return {
            'avg_soundness': sum(arg.soundness_estimate for arg in arguments) / len(arguments),
            'avg_confidence': sum(arg.confidence for arg in arguments) / len(arguments),
            'validity_rate': sum(1 for arg in arguments if arg.validity) / len(arguments)
        }
    
    def _analyze_inference_patterns(self, arguments: List[LogicalArgument]) -> Dict[str, int]:
        """Analyze patterns in inference rules used"""
        patterns = defaultdict(int)
        for arg in arguments:
            for rule in arg.inference_rules_used:
                patterns[rule] += 1
        return dict(patterns)
    
    def _detect_argument_chains(self, arguments: List[LogicalArgument]) -> List[Dict[str, Any]]:
        """Detect chains of connected arguments"""
        chains = []
        
        for i, arg1 in enumerate(arguments):
            for j, arg2 in enumerate(arguments[i+1:], i+1):
                # Check if conclusion of arg1 relates to premises of arg2
                conclusion_text = arg1.conclusion.normalized_form
                
                for premise in arg2.premises:
                    premise_text = premise.normalized_form
                    
                    # Simple overlap check (could be enhanced with LLM similarity)
                    words1 = set(conclusion_text.split())
                    words2 = set(premise_text.split())
                    overlap = len(words1.intersection(words2))
                    
                    if overlap >= 2:  # At least 2 words in common
                        chains.append({
                            'from_argument': arg1.id,
                            'to_argument': arg2.id,
                            'connection_strength': overlap / max(len(words1), len(words2)),
                            'chain_types': [arg1.argument_type.value, arg2.argument_type.value]
                        })
                        break
        
        return chains
    
    def _assess_overall_validity(self, arguments: List[LogicalArgument]) -> Dict[str, float]:
        """Assess overall logical validity"""
        if not arguments:
            return {
                'overall_validity': 0.0,
                'logical_consistency': 0.0,
                'argument_strength': 0.0
            }
        
        valid_args = [arg for arg in arguments if arg.validity]
        validity_rate = len(valid_args) / len(arguments)
        
        avg_confidence = sum(arg.confidence for arg in arguments) / len(arguments)
        avg_soundness = sum(arg.soundness_estimate for arg in arguments) / len(arguments)
        
        return {
            'overall_validity': validity_rate,
            'logical_consistency': avg_confidence,
            'argument_strength': avg_soundness
        }
    
    def format_text_output(self, results: Dict[str, Any]) -> str:
        """Format analysis results as plain text"""
        output = []
        
        for filename, analysis in results.items():
            output.append("=" * 80)
            output.append(f"FORMAL LOGIC ANALYSIS: {filename}")
            output.append("=" * 80)
            output.append(f"Model Used: {analysis['model_used']}")
            output.append(f"Text Length: {analysis['text_length']} characters")
            output.append(f"Total Arguments Detected: {analysis['total_arguments']}")
            output.append("")
            
            if analysis['total_arguments'] > 0:
                # Argument types breakdown
                output.append("ARGUMENT TYPES:")
                output.append("-" * 40)
                for arg_type, count in analysis['argument_types'].items():
                    formatted_type = arg_type.replace('_', ' ').title()
                    output.append(f"  {formatted_type}: {count}")
                output.append("")
                
                # Quality metrics
                output.append("QUALITY METRICS:")
                output.append("-" * 40)
                quality = analysis['argument_quality']
                llm_quality = analysis['llm_quality_assessment']
                output.append(f"  Average Soundness: {quality['avg_soundness']:.2f}")
                output.append(f"  Average Confidence: {quality['avg_confidence']:.2f}")
                output.append(f"  Validity Rate: {quality['validity_rate']:.2f}")
                output.append(f"  LLM Reasoning Quality: {llm_quality.get('reasoning_quality', 0.0):.2f}")
                output.append(f"  LLM Overall Validity: {llm_quality.get('overall_validity', 0.0):.2f}")
                output.append("")
                
                # Complexity metrics
                complexity = analysis['logical_complexity']
                output.append("COMPLEXITY METRICS:")
                output.append("-" * 40)
                output.append(f"  Argument Density: {complexity['argument_density']:.1f}")
                output.append(f"  Average Argument Strength: {complexity['avg_argument_strength']:.2f}")
                output.append(f"  Deductive Ratio: {complexity['deductive_ratio']:.2f}")
                output.append(f"  Inference Diversity: {complexity['inference_diversity']:.2f}")
                output.append("")
                
                # Deductive arguments
                if analysis['deductive_arguments']:
                    output.append("DEDUCTIVE ARGUMENTS:")
                    output.append("-" * 40)
                    for i, arg in enumerate(analysis['deductive_arguments'], 1):
                        output.append(f"  {i}. {arg['argument_type'].replace('_', ' ').title()}")
                        output.append(f"     Form: {arg['logical_form']}")
                        output.append(f"     Description: {arg['natural_language_form']}")
                        output.append(f"     Confidence: {arg['confidence']:.2f}")
                        output.append("")
                
                # Non-deductive arguments
                if analysis['non_deductive_arguments']:
                    output.append("NON-DEDUCTIVE ARGUMENTS:")
                    output.append("-" * 40)
                    for i, arg in enumerate(analysis['non_deductive_arguments'], 1):
                        output.append(f"  {i}. {arg['argument_type'].replace('_', ' ').title()}")
                        output.append(f"     Form: {arg['logical_form']}")
                        output.append(f"     Description: {arg['natural_language_form']}")
                        output.append(f"     Confidence: {arg['confidence']:.2f}")
                        output.append("")
                
                # Inference patterns
                if analysis['inference_patterns']:
                    output.append("INFERENCE PATTERNS:")
                    output.append("-" * 40)
                    for pattern, count in analysis['inference_patterns'].items():
                        output.append(f"  {pattern}: {count}")
                    output.append("")
                
                # Argument chains
                if analysis['argument_chains']:
                    output.append("ARGUMENT CHAINS:")
                    output.append("-" * 40)
                    for i, chain in enumerate(analysis['argument_chains'], 1):
                        output.append(f"  {i}. {chain['from_argument']} → {chain['to_argument']}")
                        output.append(f"     Connection Strength: {chain['connection_strength']:.2f}")
                        output.append(f"     Chain Types: {' → '.join(chain['chain_types'])}")
                        output.append("")
                
                # LLM Assessment insights
                if 'reasoning_style' in llm_quality:
                    output.append("LLM ASSESSMENT:")
                    output.append("-" * 40)
                    output.append(f"  Reasoning Style: {llm_quality.get('reasoning_style', 'Unknown')}")
                    
                    strengths = llm_quality.get('strengths', [])
                    if strengths:
                        output.append("  Strengths:")
                        for strength in strengths:
                            output.append(f"    - {strength}")
                    
                    weaknesses = llm_quality.get('weaknesses', [])
                    if weaknesses:
                        output.append("  Weaknesses:")
                        for weakness in weaknesses:
                            output.append(f"    - {weakness}")
                    output.append("")
                
            else:
                output.append("No logical arguments detected in this stream.")
                output.append("")
            
            output.append("")
        
        return "\n".join(output)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-based formal logic analysis of reasoning structures")
    parser.add_argument("directory", help="Directory containing stream analysis files")
    parser.add_argument("--output", "-o", default="llm_formal_logic_analysis.txt",
                       help="Output file for analysis results (default: text format)")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text",
                       help="Output format: text (default) or json")
    parser.add_argument("--model", "-m", default="gpt41",
                       help="Model shortname from model configuration file to use (default: gpt41)")
    parser.add_argument("--config", default="model_servers.yaml",
                       help="Path to model configuration file (default: model_servers.yaml)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist")
        return
    
    # Configure APIs based on model selection
    actual_model_name, openai_client = configure_apis(args.model, args.config)
    print(f"Using model: {actual_model_name} (shortname: {args.model})")
    
    analyzer = LLMFormalLogicAnalyzer(actual_model_name, openai_client)
    results = {}
    
    # Find stream analysis files
    import glob
    pattern = os.path.join(args.directory, "*_STREAM_ANALYSIS.txt")
    files = glob.glob(pattern)
    
    print(f"Analyzing {len(files)} stream analysis files using {actual_model_name}...")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing {filename}...")
        
        analysis = analyzer.analyze_file(file_path)
        results[filename] = analysis
    
    # Save results in the specified format
    with open(args.output, 'w', encoding='utf-8') as f:
        if args.format == "json":
            json.dump(results, f, indent=2, default=str)
        else:  # text format (default)
            text_output = analyzer.format_text_output(results)
            f.write(text_output)
    
    print(f"\nAnalysis complete. Results saved to {args.output} ({args.format} format)")
    
    # Print summary
    if results:
        total_args = sum(result['total_arguments'] for result in results.values())
        total_deductive = sum(len(result['deductive_arguments']) for result in results.values())
        
        print(f"\nSUMMARY:")
        print(f"Total logical arguments detected: {total_args}")
        print(f"Deductive arguments: {total_deductive}")
        print(f"Non-deductive arguments: {total_args - total_deductive}")
        
        # Aggregate argument types
        all_types = defaultdict(int)
        for result in results.values():
            for arg_type, count in result['argument_types'].items():
                all_types[arg_type] += count
        
        if all_types:
            print(f"\nArgument types detected:")
            for arg_type, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {arg_type.replace('_', ' ').title()}: {count}")
        
        # Average metrics
        if results:
            avg_validity = sum(r['validity_assessment']['overall_validity'] for r in results.values()) / len(results)
            avg_strength = sum(r['validity_assessment']['argument_strength'] for r in results.values()) / len(results)
            avg_llm_quality = sum(r['llm_quality_assessment']['reasoning_quality'] for r in results.values()) / len(results)
            
            print(f"\nAverage validity rate: {avg_validity:.2f}")
            print(f"Average argument strength: {avg_strength:.2f}")
            print(f"Average LLM reasoning quality: {avg_llm_quality:.2f}")

if __name__ == "__main__":
    main()