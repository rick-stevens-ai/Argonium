#!/usr/bin/env python3
"""
paper_syn_org.py - Scientific Paper Synthesis and Organization Tool

This script reads scientific papers as PDFs and produces standardized JSON documents
with enhanced structure including expanded hypothesis analysis, theory/computational
sections, and deep experimental protocol analysis.

Usage:
    python paper_syn_org.py --file paper.pdf --model gpt4
    python paper_syn_org.py --directory papers_dir --model gpt4
"""

import os
import sys
import json
import argparse
import yaml
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
import PyPDF2
import re

# Optional import for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

MODEL_CONFIG_FILE = 'model_servers.yaml'

@dataclass
class PaperStructure:
    """Data class to hold structured paper information"""
    paper_id: str
    original_filename: str
    title: str
    abstract: str
    keywords: List[str]
    
    # Enhanced sections
    introduction: Dict[str, Any]
    hypothesis: Dict[str, Any]  # Expanded with formal notation and hallmarks
    theory_computational: Dict[str, Any]  # New section
    methods: Dict[str, Any]  # Enhanced with deep protocol analysis
    results: Dict[str, Any]  # Enhanced with hypothesis validation
    discussion: Dict[str, Any]
    conclusion: str
    acknowledgments: str
    references: List[str]
    
    # Metadata
    extraction_timestamp: float
    processing_notes: List[str]

class PaperAnalyzer:
    """
    Analyzes scientific papers and extracts structured information.
    Reuses patterns from existing scripts for OpenAI client setup.
    """
    
    def __init__(self, model_shortname: str, force_synthesis: bool = False):
        self.model_shortname = model_shortname
        self.force_synthesis = force_synthesis
        self.model_config = self._load_model_config()
        self._setup_openai_client()
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        try:
            with open(MODEL_CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Model configuration file '{MODEL_CONFIG_FILE}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{MODEL_CONFIG_FILE}': {e}")
        
        model_config = None
        for server in config.get('servers', []):
            if server.get('shortname') == self.model_shortname:
                model_config = server
                break
        
        if not model_config:
            raise ValueError(f"Error: Model shortname '{self.model_shortname}' not found in '{MODEL_CONFIG_FILE}'.")
        
        return model_config
    
    def _setup_openai_client(self) -> None:
        """Setup OpenAI client"""
        model_config = self._load_model_config()
        
        # Determine OpenAI API Key
        openai_api_key_config = model_config.get('openai_api_key')
        openai_api_key = None
        
        if openai_api_key_config == "${OPENAI_API_KEY}":
            openai_api_key = os.environ.get('OPENAI-API-KEY') or os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("Error: OpenAI API key is configured to use environment variable "
                               "'OPENAI-API-KEY' or 'OPENAI_API_KEY', but neither is set.")
        elif openai_api_key_config == "${VLLM_API_KEY}":
            openai_api_key = os.environ.get('VLLM-API-KEY') or os.environ.get('VLLM_API_KEY')
            if not openai_api_key:
                raise ValueError("Error: VLLM API key is configured to use environment variable "
                               "'VLLM-API-KEY' or 'VLLM_API_KEY', but neither is set.")
        elif openai_api_key_config:
            openai_api_key = openai_api_key_config
        else:
            raise ValueError(f"Error: 'openai_api_key' not specified for model '{self.model_shortname}'.")
        
        # Get API base and model
        openai_api_base = model_config.get('openai_api_base')
        openai_model = model_config.get('openai_model')
        
        if not openai_api_base or not openai_model:
            raise ValueError(f"Error: 'openai_api_base' or 'openai_model' missing for model '{self.model_shortname}'.")
        
        try:
            # Set up OpenAI client for new version
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base
            )
            self.openai_model = openai_model
        except Exception as e:
            raise ValueError(f"Error initializing OpenAI client: {e}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean response text to extract JSON from various formats"""
        if not response_text or not response_text.strip():
            print("Warning: Empty response received from model")
            return "{}"
        
        cleaned_text = response_text.strip()
        original_text = cleaned_text  # Keep original for debugging
        
        # Remove markdown code blocks
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Here's the JSON response:",
            "Here is the JSON:",
            "JSON response:",
            "Response:",
            "Result:",
            "Analysis:",
            "Here's the analysis in JSON format:",
            "The JSON output is:",
            "Sure, here's the JSON:",
            "Based on the analysis, here's the JSON:",
            "The analysis in JSON format:",
            "JSON Analysis:",
            "Here's my analysis in JSON format:",
            "Based on the text, here's the JSON:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # Try to find JSON within the text if it's not at the start
        import re
        # Look for JSON objects that span multiple lines
        json_match = re.search(r'\{.*?\}', cleaned_text, re.DOTALL)
        if json_match:
            cleaned_text = json_match.group(0)
        else:
            # Look for JSON arrays
            json_match = re.search(r'\[.*?\]', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)
        
        # Final cleanup
        cleaned_text = cleaned_text.strip()
        
        # Try to validate JSON before returning
        try:
            json.loads(cleaned_text)
            return cleaned_text
        except json.JSONDecodeError as e:
            print(f"Warning: Model returned non-JSON response, attempting text parsing...")
            print(f"JSON parsing error: {e}")
            print(f"First 200 characters of response: {original_text[:200]}")
            
            # Last resort: try to extract any JSON-like structure
            brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', original_text, re.DOTALL)
            if brace_match:
                try:
                    json.loads(brace_match.group(0))
                    return brace_match.group(0)
                except:
                    pass
            
            # If all else fails, return empty object
            print("Warning: Could not extract valid JSON, returning empty object")
            return "{}"
    
    def _synthesize_comprehensive_content(self, section_type: str, full_paper_text: str, title: str, abstract: str) -> Dict[str, Any]:
        """Synthesize comprehensive section content with both narrative and structured data"""
        if not self.force_synthesis:
            return {}
        
        print(f"    🤖 Synthesizing comprehensive {section_type} content based on full paper...")
        
        # Create a comprehensive context from the full paper
        paper_summary = f"""
Title: {title}
Abstract: {abstract}

Full Paper Content (first 8000 chars):
{full_paper_text[:8000]}
"""
        
        synthesis_prompts = {
            "introduction": {
                "system": "You are an expert at analyzing scientific papers. Generate both structured data and narrative content for the introduction section.",
                "user": f"""Based on this complete scientific paper, analyze the introduction and provide comprehensive content in JSON format:

{paper_summary}

Respond with JSON containing:
1. A comprehensive narrative paragraph for markdown output
2. Structured fields for JSON output compatibility

{{
    "narrative": "Write a comprehensive introduction narrative (3-4 paragraphs) covering: research context and field background, literature review and prior work, problem statement and knowledge gaps, research motivation and significance, novelty claims and contributions, and paper organization. Write as flowing narrative paragraphs, not bullet points.",
    "research_context": {{
        "field": "Primary research field",
        "subfield": "Specific research area", 
        "interdisciplinary_connections": ["field1", "field2"]
    }},
    "background_knowledge": {{
        "key_concepts": ["concept1", "concept2"],
        "foundational_theories": ["theory1", "theory2"],
        "established_methods": ["method1", "method2"]
    }},
    "problem_statement": {{
        "main_problem": "Clear description of the research problem",
        "knowledge_gaps": ["gap1", "gap2"],
        "research_questions": ["question1", "question2"]
    }},
    "research_motivation": {{
        "scientific_importance": "Why this research matters scientifically",
        "practical_applications": ["application1", "application2"],
        "potential_impact": "Expected impact description"
    }},
    "novelty_claims": {{
        "technical_novelty": "What is technically new",
        "methodological_novelty": "New methods or approaches", 
        "conceptual_novelty": "New ideas or frameworks"
    }},
    "synthesis_note": "Comprehensive introduction synthesized from full paper analysis"
}}"""
            },
            
            "hypothesis": {
                "system": "You are an expert at analyzing scientific hypotheses. Generate both structured data and narrative content for the hypothesis section.",
                "user": f"""Based on this complete scientific paper, analyze the hypothesis and provide comprehensive content in JSON format:

{paper_summary}

{{
    "narrative": "Write a comprehensive hypothesis narrative (2-3 paragraphs) explaining: the main hypothesis development and scientific reasoning, theoretical foundations and underlying principles, formal expressions and mathematical notation if applicable, sub-hypotheses and related predictions, alternative hypotheses considered, and testing approaches used.",
    "main_hypothesis": {{
        "statement": "Clear, testable main hypothesis statement",
        "formal_notation": "Mathematical or logical notation if applicable",
        "null_hypothesis": "Corresponding null hypothesis",
        "variables": {{
            "independent": ["list of independent variables"],
            "dependent": ["list of dependent variables"],
            "confounding": ["potential confounding variables"]
        }}
    }},
    "sub_hypotheses": [
        {{
            "statement": "Sub-hypothesis statement",
            "formal_notation": "Formal expression",
            "relationship_to_main": "How this relates to main hypothesis"
        }}
    ],
    "hypothesis_hallmarks_analysis": {{
        "testability": {{"score": 5, "rationale": "Why this score for testability"}},
        "falsifiability": {{"score": 5, "rationale": "Why this score for falsifiability"}},
        "specificity": {{"score": 5, "rationale": "Why this score for specificity"}},
        "parsimony": {{"score": 5, "rationale": "Why this score for parsimony"}},
        "generalizability": {{"score": 5, "rationale": "Why this score for generalizability"}},
        "predictive_power": {{"score": 5, "rationale": "Why this score for predictive power"}}
    }},
    "theoretical_foundation": "Description of theoretical basis for hypothesis",
    "specific_predictions": ["Specific prediction 1", "Specific prediction 2"],
    "alternative_hypotheses": ["Alternative explanation 1", "Alternative explanation 2"],
    "synthesis_note": "Comprehensive hypothesis synthesized from full paper analysis"
}}"""
            },
            
            "methods": {
                "system": "You are an expert in experimental methodology. Generate both structured data and narrative content for the methods section.",
                "user": f"""Based on this complete scientific paper, analyze the methods and provide comprehensive content in JSON format:

{paper_summary}

{{
    "narrative": "Write a comprehensive methods narrative (3-4 paragraphs) covering: overall study design and research strategy, participants/subjects and sample characteristics, materials and equipment used, detailed experimental procedures and protocols, data collection techniques and instruments, statistical analysis methods and software, and quality control measures. Write detailed content that would allow replication.",
    "experimental_design": {{
        "study_type": "Type of study design",
        "sample_size": "Sample size details",
        "randomization": "Randomization approach if applicable",
        "control_groups": "Control group details if applicable"
    }},
    "participants_subjects": {{
        "recruitment_criteria": ["inclusion criteria", "exclusion criteria"],
        "sample_characteristics": "Description of sample demographics",
        "ethical_considerations": "IRB approval and ethical protocols"
    }},
    "materials_equipment": {{
        "instruments": ["instrument 1", "instrument 2"],
        "software": ["software 1", "software 2"],
        "reagents": ["reagent 1", "reagent 2"] 
    }},
    "detailed_protocols": [
        {{
            "protocol_name": "Protocol name",
            "duration": "Time required",
            "steps": ["Step 1", "Step 2", "Step 3"],
            "parameters": {{"param1": "value1", "param2": "value2"}}
        }}
    ],
    "data_collection": {{
        "measurement_techniques": ["technique 1", "technique 2"],
        "data_types": ["type 1", "type 2"],
        "quality_control": ["QC measure 1", "QC measure 2"]
    }},
    "statistical_analysis": {{
        "primary_analyses": ["analysis 1", "analysis 2"],
        "software_used": ["software 1", "software 2"],
        "significance_level": "Alpha level used"
    }},
    "synthesis_note": "Comprehensive methods synthesized from full paper analysis"
}}"""
            },
            
            "results": {
                "system": "You are an expert at analyzing scientific results. Generate both structured data and narrative content for the results section.",
                "user": f"""Based on this complete scientific paper, analyze the results and provide comprehensive content in JSON format:

{paper_summary}

{{
    "narrative": "Write a comprehensive results narrative (3-4 paragraphs) covering: overview and summary of key findings, primary experimental outcomes with statistical details, secondary findings and additional discoveries, detailed statistical analyses and significance testing, description of data visualizations and figures, hypothesis testing outcomes and validation, and effect sizes with practical significance. Include specific numerical results and statistical measures.",
    "hypothesis_validation": {{
        "main_hypothesis_supported": "yes/no/partially",
        "support_strength": "strong/moderate/weak",
        "evidence_summary": "Summary of supporting evidence",
        "statistical_significance": "p-values and confidence intervals"
    }},
    "key_findings": [
        {{
            "finding": "Key finding description",
            "statistical_details": "Statistical measures (p-value, CI, effect size)",
            "significance": "biological/clinical/practical significance",
            "unexpected": "yes/no - was this finding expected"
        }}
    ],
    "primary_outcomes": {{
        "outcome_measures": ["measure 1", "measure 2"],
        "baseline_characteristics": "Baseline data summary",
        "treatment_effects": "Effects observed"
    }},
    "statistical_results": {{
        "descriptive_statistics": "Mean, SD, ranges, etc.",
        "inferential_statistics": "Test results, p-values, effect sizes",
        "confidence_intervals": "95% CI ranges"
    }},
    "data_visualizations": [
        {{
            "figure_type": "Type of visualization",
            "description": "What the figure shows",
            "key_insights": "Main insights from the visualization"
        }}
    ],
    "synthesis_note": "Comprehensive results synthesized from full paper analysis"
}}"""
            },
            
            "discussion": {
                "system": "You are an expert at scientific interpretation and discussion. Generate both structured data and narrative content for the discussion section.",
                "user": f"""Based on this complete scientific paper, analyze the discussion and provide comprehensive content in JSON format:

{paper_summary}

{{
    "narrative": "Write a comprehensive discussion narrative (4-5 paragraphs) covering: interpretation of key findings and their scientific meaning, mechanistic explanations for observed results, comparison with existing literature and prior work, theoretical implications and impact on scientific understanding, practical applications and real-world relevance, study limitations and methodological constraints, future research directions and next steps, and broader impact on the field. Demonstrate deep scientific insight and analytical thinking.",
    "results_interpretation": {{
        "main_findings_interpretation": "What the primary findings mean scientifically",
        "mechanistic_explanations": ["explanation 1", "explanation 2"],
        "unexpected_findings": "Discussion of any surprising results"
    }},
    "literature_comparison": {{
        "consistent_findings": ["findings that align with prior work"],
        "contradictory_findings": ["findings that contradict prior work"],
        "novel_contributions": ["what this work adds to the field"]
    }},
    "implications": {{
        "theoretical_implications": ["theoretical impact 1", "theoretical impact 2"],
        "practical_applications": ["application 1", "application 2"],
        "clinical_relevance": "Clinical significance if applicable"
    }},
    "limitations": {{
        "methodological_limitations": ["limitation 1", "limitation 2"],
        "generalizability_concerns": ["concern 1", "concern 2"],
        "technical_constraints": ["constraint 1", "constraint 2"]
    }},
    "future_directions": {{
        "immediate_next_steps": ["next step 1", "next step 2"],
        "long_term_research_goals": ["goal 1", "goal 2"],
        "methodological_improvements": ["improvement 1", "improvement 2"]
    }},
    "broader_impact": {{
        "field_significance": "Impact on the research field",
        "societal_implications": "Broader societal relevance",
        "policy_considerations": "Policy implications if applicable"
    }},
    "synthesis_note": "Comprehensive discussion synthesized from full paper analysis"
}}"""
            },
            
            "theory_computational": {
                "system": "You are an expert in theoretical frameworks and computational methods. Generate both structured data and narrative content for theory/computational sections.",
                "user": f"""Based on this complete scientific paper, analyze the theoretical and computational aspects and provide comprehensive content in JSON format:

{paper_summary}

{{
    "narrative": "Write a comprehensive theory/computational narrative (3-4 paragraphs) covering: underlying theoretical frameworks and scientific principles, mathematical models and equations used, computational approaches and algorithms, implementation details and technical specifications, model validation and verification methods, and computational resources and requirements. Explain both theoretical foundations and computational strategies.",
    "theoretical_framework": {{
        "primary_theories": ["Theory 1", "Theory 2"],
        "conceptual_models": ["Model 1", "Model 2"],
        "theoretical_basis_description": "Detailed description of theoretical foundation",
        "key_assumptions": ["Assumption 1", "Assumption 2"]
    }},
    "mathematical_models": {{
        "equations": [
            {{
                "equation": "Mathematical equation or formula",
                "description": "What this equation represents",
                "variables": "Definition of variables used"
            }}
        ],
        "model_type": "Type of mathematical model",
        "model_validation": "How the model was validated"
    }},
    "computational_methods": {{
        "algorithms": ["Algorithm 1", "Algorithm 2"],
        "software_tools": ["Software 1", "Software 2"],
        "programming_languages": ["Language 1", "Language 2"],
        "computational_complexity": "Analysis of computational requirements"
    }},
    "implementation_details": {{
        "technical_specifications": "Hardware and software specifications",
        "parameters": {{"param1": "value1", "param2": "value2"}},
        "optimization_strategies": ["strategy 1", "strategy 2"]
    }},
    "validation_verification": {{
        "validation_methods": ["method 1", "method 2"],
        "benchmarking": "Comparison with established methods",
        "accuracy_metrics": "Measures of model accuracy"
    }},
    "synthesis_note": "Comprehensive theory/computational content synthesized from full paper analysis"
}}"""
            }
        }
        
        if section_type not in synthesis_prompts:
            return {"synthesis_note": f"No synthesis template available for {section_type}"}
        
        prompt_config = synthesis_prompts[section_type]
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": prompt_config["system"]},
                    {"role": "user", "content": prompt_config["user"]}
                ],
                temperature=0.3,
                max_tokens=4000  # Increased for comprehensive narratives
            )
            
            response_text = response.choices[0].message.content
            cleaned_text = self._clean_json_response(response_text)
            
            if cleaned_text and cleaned_text != "{}":
                try:
                    # Parse the JSON response which should contain both narrative and structured data
                    result = json.loads(cleaned_text)
                    # Ensure synthesis note is present
                    if "synthesis_note" not in result:
                        result["synthesis_note"] = f"Comprehensive {section_type} synthesized from full paper analysis"
                    return result
                except json.JSONDecodeError as e:
                    print(f"    ⚠️  JSON parsing error in synthesis: {e}")
                    # Fallback: treat as narrative content
                    return {
                        "narrative": response_text.strip(),
                        "synthesis_note": f"Comprehensive {section_type} synthesized from full paper analysis (fallback)"
                    }
            else:
                print(f"    ⚠️  Synthesis failed for {section_type}, using fallback")
                return {"synthesis_note": f"Synthesis attempted but failed for {section_type}"}
                
        except Exception as e:
            print(f"    ⚠️  Synthesis error for {section_type}: {e}")
            return {"synthesis_note": f"Synthesis failed due to error: {str(e)}"}
    
    def _post_process_sections_for_synthesis(self, paper_structure: 'PaperStructure', full_text: str) -> 'PaperStructure':
        """Post-process all sections to replace fallback content with synthesis when force_synthesis is enabled"""
        if not self.force_synthesis:
            return paper_structure
        
        print("  🤖 Post-processing sections for synthesis...")
        
        fallback_indicators = [
            "could not extract", "not available", "not provided", "not specified", 
            "not found", "unable to", "error during", "could not determine"
        ]
        
        def needs_synthesis(content):
            """Check if content contains fallback indicators"""
            if isinstance(content, str):
                return any(indicator in content.lower() for indicator in fallback_indicators)
            elif isinstance(content, dict):
                return any(needs_synthesis(v) for v in content.values())
            elif isinstance(content, list):
                return any(needs_synthesis(item) for item in content)
            return False
        
        # Check and synthesize hypothesis section
        if needs_synthesis(paper_structure.hypothesis):
            synthesized = self._synthesize_comprehensive_content("hypothesis", full_text, paper_structure.title, paper_structure.abstract)
            if synthesized and synthesized != {}:
                # Replace the entire section with synthesized content
                paper_structure.hypothesis = synthesized
        
        # Check and synthesize theory/computational section
        if needs_synthesis(paper_structure.theory_computational):
            synthesized = self._synthesize_comprehensive_content("theory_computational", full_text, paper_structure.title, paper_structure.abstract)
            if synthesized and synthesized != {}:
                paper_structure.theory_computational = synthesized
        
        # Check and synthesize methods section
        if needs_synthesis(paper_structure.methods):
            synthesized = self._synthesize_comprehensive_content("methods", full_text, paper_structure.title, paper_structure.abstract)
            if synthesized and synthesized != {}:
                paper_structure.methods = synthesized
        
        # Check and synthesize results section
        if needs_synthesis(paper_structure.results):
            synthesized = self._synthesize_comprehensive_content("results", full_text, paper_structure.title, paper_structure.abstract)
            if synthesized and synthesized != {}:
                paper_structure.results = synthesized
        
        # Check and synthesize introduction section
        if needs_synthesis(paper_structure.introduction):
            synthesized = self._synthesize_comprehensive_content("introduction", full_text, paper_structure.title, paper_structure.abstract)
            if synthesized and synthesized != {}:
                paper_structure.introduction = synthesized
        
        # Check and synthesize discussion section
        if needs_synthesis(paper_structure.discussion):
            synthesized = self._synthesize_comprehensive_content("discussion", full_text, paper_structure.title, paper_structure.abstract)
            if synthesized and synthesized != {}:
                paper_structure.discussion = synthesized
        
        # Synthesize abstract if missing or poor quality
        if not paper_structure.abstract or len(paper_structure.abstract.strip()) < 50:
            print("    🤖 Synthesizing abstract...")
            try:
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at writing scientific abstracts. Create a concise, informative abstract based on the paper content."},
                        {"role": "user", "content": f"Create a comprehensive abstract (200-300 words) for this scientific paper:\n\n{full_text[:3000]}"}
                    ],
                    temperature=0.2,
                    max_tokens=400
                )
                synthesized_abstract = response.choices[0].message.content.strip()
                if synthesized_abstract and len(synthesized_abstract) > 50:
                    paper_structure.abstract = synthesized_abstract
            except Exception as e:
                print(f"    ⚠️  Abstract synthesis failed: {e}")
        
        # Synthesize conclusion if missing
        if not paper_structure.conclusion or "could not" in paper_structure.conclusion.lower():
            print("    🤖 Synthesizing conclusion...")
            try:
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at writing scientific conclusions. Create a concise conclusion based on the paper content."},
                        {"role": "user", "content": f"Create a comprehensive conclusion for this scientific paper based on the content:\n\n{full_text[:3000]}"}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                synthesized_conclusion = response.choices[0].message.content.strip()
                if synthesized_conclusion:
                    paper_structure.conclusion = synthesized_conclusion
            except Exception as e:
                print(f"    ⚠️  Conclusion synthesis failed: {e}")
        
        return paper_structure
    
    def _validate_and_parse_json(self, response_text: str, section_name: str, fallback_structure: Dict[str, Any], 
                                paper_context: str = "", section_type: str = "") -> Dict[str, Any]:
        """Validate and parse JSON response with comprehensive error handling and synthesis option"""
        try:
            if not response_text or not response_text.strip():
                print(f"Warning: Empty response received from model for {section_name}")
                if self.force_synthesis and paper_context and section_type:
                    return self._synthesize_missing_content(section_name, paper_context, section_type)
                return fallback_structure
            
            cleaned_text = self._clean_json_response(response_text)
            
            if not cleaned_text or cleaned_text == "{}":
                print(f"Warning: Empty or invalid JSON received from model for {section_name}")
                if self.force_synthesis and paper_context and section_type:
                    return self._synthesize_missing_content(section_name, paper_context, section_type)
                return fallback_structure
            
            parsed_json = json.loads(cleaned_text)
            return parsed_json
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in {section_name}: {e}")
            print(f"First 200 characters of response: {response_text[:200]}")
            print(f"Attempted to parse: {cleaned_text[:200] if 'cleaned_text' in locals() else 'No cleaned text available'}")
            if self.force_synthesis and paper_context and section_type:
                return self._synthesize_missing_content(section_name, paper_context, section_type)
            return {**fallback_structure, "analysis_notes": [f"JSON parsing error: {str(e)}"]}
        except Exception as e:
            print(f"Error processing {section_name}: {e}")
            if self.force_synthesis and paper_context and section_type:
                return self._synthesize_missing_content(section_name, paper_context, section_type)
            return {**fallback_structure, "analysis_notes": [f"Error during {section_name}: {str(e)}"]}
    
    def _chunk_text(self, text: str, max_chars: int = 15000) -> List[str]:
        """Split text into manageable chunks for AI processing"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at a reasonable point
            break_point = text.rfind('\n\n', start, end)
            if break_point == -1:
                break_point = text.rfind('\n', start, end)
            if break_point == -1:
                break_point = text.rfind('. ', start, end)
            if break_point == -1:
                break_point = end
            
            chunks.append(text[start:break_point])
            start = break_point
        
        return chunks
    
    def _parse_text_response_for_basic_info(self, response_text: str) -> Dict[str, Any]:
        """Parse non-JSON response text to extract basic paper information"""
        # Simple text parsing fallback
        lines = response_text.split('\n')
        
        title = "Unknown Title"
        abstract = ""
        keywords = []
        authors = []
        
        # Try to find title, abstract, etc. in the text response
        for i, line in enumerate(lines):
            line = line.strip()
            if 'title' in line.lower() and ':' in line:
                title = line.split(':', 1)[1].strip().strip('"')
            elif 'abstract' in line.lower() and ':' in line:
                abstract = line.split(':', 1)[1].strip().strip('"')
            elif 'keyword' in line.lower() and ':' in line:
                keywords_text = line.split(':', 1)[1].strip()
                keywords = [kw.strip().strip('"') for kw in keywords_text.split(',')]
        
        return {
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "authors": authors,
            "affiliations": [],
            "extraction_notes": [f"Parsed from non-JSON response: {response_text[:100]}..."]
        }
    
    def _generate_keywords(self, title: str, abstract: str, text: str) -> List[str]:
        """Generate keywords if they are missing from the paper"""
        
        system_prompt = """You are an expert at generating scientific keywords for research papers. Generate relevant, specific keywords that capture the main concepts, methods, and subject areas of the paper."""
        
        user_prompt = f"""
Generate 5-8 relevant scientific keywords for this research paper based on its title, abstract, and content. 

The keywords should:
1. Capture the main research area/field
2. Include key methodological approaches
3. Represent important concepts or phenomena studied
4. Be specific enough to be useful for literature searches
5. Follow standard scientific keyword conventions

Title: {title}

Abstract: {abstract}

Additional context from paper:
{text[:2000]}

Respond in JSON format:
{{
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "keyword_rationale": "Brief explanation of why these keywords were chosen",
    "generation_notes": ["Any observations about keyword selection"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Slightly higher for more creative keyword generation
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "keywords": []
            }
            
            result = self._validate_and_parse_json(response_text, "keyword generation", fallback_structure)
            return result.get('keywords', [])
            
        except Exception as e:
            print(f"Error generating keywords: {e}")
            # Fallback: simple keyword extraction from title and abstract
            import re
            text_for_keywords = f"{title} {abstract}".lower()
            # Simple extraction of potential keywords (words 4+ chars, not common words)
            words = re.findall(r'\b[a-z]{4,}\b', text_for_keywords)
            common_words = {'this', 'that', 'with', 'from', 'they', 'were', 'been', 'have', 'their', 'said', 'each', 'which', 'what', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'many', 'some', 'time', 'very', 'when', 'much', 'new', 'two', 'may', 'way', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'has', 'her', 'his', 'how', 'man', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
            keywords = [word for word in set(words) if word not in common_words][:6]
            return keywords if keywords else ["research", "analysis", "study"]

    def _analyze_basic_structure(self, text: str) -> Dict[str, Any]:
        """Extract basic paper structure (title, abstract, keywords)"""
        
        system_prompt = """You are an expert at analyzing scientific papers. Extract the basic structure and metadata from the given text."""
        
        user_prompt = f"""
Analyze this scientific paper text and extract the following basic information:

1. **Title**: The main title of the paper
2. **Abstract**: The complete abstract text
3. **Keywords**: List of keywords (if present)
4. **Author Information**: Authors and affiliations (if clear)

Paper Text:
{text[:8000]}  # First portion for basic info

Respond in JSON format:
{{
    "title": "Complete paper title",
    "abstract": "Complete abstract text",
    "keywords": ["keyword1", "keyword2", ...],
    "authors": ["Author 1", "Author 2", ...],
    "affiliations": ["Institution 1", "Institution 2", ...],
    "extraction_notes": ["Any issues or observations about the extraction"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Try to parse as JSON first
            try:
                cleaned_text = self._clean_json_response(response_text)
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                # If not JSON, use fallback text parsing
                print(f"Warning: Model returned non-JSON response, attempting text parsing...")
                return self._parse_text_response_for_basic_info(response_text)
            
        except Exception as e:
            print(f"Error analyzing basic structure: {e}")
            return {
                "title": "Unknown Title", 
                "abstract": "",
                "keywords": [],
                "authors": [],
                "affiliations": [],
                "extraction_notes": [f"Error during extraction: {str(e)}"]
            }
    
    def _analyze_hypothesis_section(self, text: str) -> Dict[str, Any]:
        """Analyze and extract enhanced hypothesis information"""
        
        system_prompt = """You are an expert scientific analyst specializing in hypothesis formulation and analysis. Your task is to identify, formalize, and analyze hypotheses in scientific papers with deep understanding of scientific methodology."""
        
        user_prompt = f"""
Analyze this scientific paper for hypothesis-related content. Provide a comprehensive analysis including:

1. **Hypothesis Identification**: Extract the main hypothesis(es) and sub-hypotheses
2. **Formal Notation**: Express hypotheses in formal scientific notation when applicable (H₀, H₁, mathematical expressions)
3. **Hypothesis Hallmarks**: Evaluate each hypothesis against the hallmarks of good hypotheses:
   - Testability (Can it be empirically tested?)
   - Falsifiability (Can it be proven wrong?)
   - Specificity (Is it precise and clear?)
   - Parsimony (Is it the simplest explanation?)
   - Generalizability (Does it apply broadly?)
   - Predictive power (Does it make specific predictions?)

4. **Theoretical Basis**: What theoretical framework supports the hypothesis?
5. **Predictions**: What specific predictions does the hypothesis make?

Paper Text:
{text}

Respond in detailed JSON format:
{{
    "main_hypothesis": {{
        "statement": "Clear statement of the primary hypothesis",
        "formal_notation": "H₁: [formal expression if applicable]",
        "null_hypothesis": "H₀: [corresponding null hypothesis]",
        "type": "descriptive|comparative|causal|predictive",
        "variables": {{
            "independent": ["list of independent variables"],
            "dependent": ["list of dependent variables"],
            "confounding": ["potential confounding variables"]
        }}
    }},
    "sub_hypotheses": [
        {{
            "statement": "Sub-hypothesis statement",
            "formal_notation": "Formal expression",
            "relationship_to_main": "How this relates to main hypothesis"
        }}
    ],
    "hypothesis_hallmarks_analysis": {{
        "testability": {{
            "score": 1-5,
            "rationale": "Why this score for testability"
        }},
        "falsifiability": {{
            "score": 1-5,
            "rationale": "Why this score for falsifiability"
        }},
        "specificity": {{
            "score": 1-5,
            "rationale": "Why this score for specificity"
        }},
        "parsimony": {{
            "score": 1-5,
            "rationale": "Why this score for parsimony"
        }},
        "generalizability": {{
            "score": 1-5,
            "rationale": "Why this score for generalizability"
        }},
        "predictive_power": {{
            "score": 1-5,
            "rationale": "Why this score for predictive power"
        }}
    }},
    "theoretical_foundation": "Description of theoretical basis for hypothesis",
    "specific_predictions": [
        "Specific prediction 1",
        "Specific prediction 2"
    ],
    "alternative_hypotheses": [
        "Alternative explanation 1",
        "Alternative explanation 2"
    ],
    "hypothesis_origin": "How was this hypothesis developed? (from literature, observation, theory, etc.)",
    "analysis_notes": ["Any important observations about the hypothesis analysis"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "main_hypothesis": {"statement": "Could not extract hypothesis", "formal_notation": "", "null_hypothesis": ""},
                "analysis_notes": ["Error during hypothesis analysis"]
            }
            
            return self._validate_and_parse_json(response_text, "hypothesis analysis", fallback_structure, text, "hypothesis")
            
        except Exception as e:
            print(f"Error analyzing hypothesis section: {e}")
            return {
                "main_hypothesis": {"statement": "Could not extract hypothesis", "formal_notation": "", "null_hypothesis": ""},
                "analysis_notes": [f"Error during hypothesis analysis: {str(e)}"]
            }
    
    def _analyze_theory_computational_section(self, text: str) -> Dict[str, Any]:
        """Analyze theoretical basis and computational methods"""
        
        system_prompt = """You are an expert in theoretical frameworks and computational methods in scientific research. Analyze the theoretical foundations and computational approaches used in this paper."""
        
        user_prompt = f"""
Analyze this scientific paper for theoretical foundations and computational methods. Provide comprehensive analysis of:

1. **Theoretical Framework**: What theories, models, or conceptual frameworks underpin this research?
2. **Mathematical Models**: Any mathematical formulations, equations, or models used
3. **Computational Methods**: Software, algorithms, simulations, or computational approaches
4. **Simulation Details**: If simulations were used, how were they designed and implemented?
5. **Computational Tools**: Specific software, programming languages, or platforms used
6. **Model Validation**: How were computational models validated?

Paper Text:
{text}

Respond in detailed JSON format:
{{
    "theoretical_framework": {{
        "primary_theories": ["Theory 1", "Theory 2"],
        "conceptual_models": ["Model 1", "Model 2"],
        "theoretical_basis_description": "Detailed description of theoretical foundation",
        "key_assumptions": ["Assumption 1", "Assumption 2"],
        "theoretical_gaps_addressed": "What theoretical gaps does this work address?"
    }},
    "mathematical_models": {{
        "equations": [
            {{
                "equation": "Mathematical expression",
                "description": "What this equation represents",
                "variables_defined": {{"var1": "definition", "var2": "definition"}}
            }}
        ],
        "model_types": ["statistical", "mechanistic", "phenomenological", "other"],
        "model_complexity": "simple|moderate|complex",
        "model_novelty": "Are these new or established models?"
    }},
    "computational_methods": {{
        "algorithms_used": ["Algorithm 1", "Algorithm 2"],
        "software_tools": ["Software 1", "Software 2"],
        "programming_languages": ["Python", "R", "MATLAB", "etc."],
        "computational_platforms": ["HPC", "cloud", "local", "etc."],
        "data_processing_methods": ["Method 1", "Method 2"]
    }},
    "simulation_details": {{
        "simulation_used": true/false,
        "simulation_type": "Monte Carlo|molecular dynamics|finite element|agent-based|other",
        "simulation_purpose": "Why simulations were used",
        "simulation_parameters": {{
            "parameter1": "value",
            "parameter2": "value"
        }},
        "simulation_validation": "How simulations were validated",
        "simulation_limitations": "Limitations of the simulation approach"
    }},
    "model_validation": {{
        "validation_methods": ["cross-validation", "experimental validation", "other"],
        "validation_metrics": ["metric1", "metric2"],
        "validation_results": "Summary of validation outcomes"
    }},
    "computational_innovation": "What computational innovations does this work introduce?",
    "computational_reproducibility": "Information about code/data availability for reproduction",
    "analysis_notes": ["Any important observations about computational analysis"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "theoretical_framework": {"theoretical_basis_description": "Could not extract theoretical information"},
                "analysis_notes": ["Error during theory/computational analysis"]
            }
            
            return self._validate_and_parse_json(response_text, "theory/computational analysis", fallback_structure, text, "theory_computational")
            
        except Exception as e:
            print(f"Error analyzing theory/computational section: {e}")
            return {
                "theoretical_framework": {"theoretical_basis_description": "Could not extract theoretical information"},
                "analysis_notes": [f"Error during theory/computational analysis: {str(e)}"]
            }
    
    def _analyze_methods_section(self, text: str) -> Dict[str, Any]:
        """Deep analysis of experimental protocols and methods"""
        
        system_prompt = """You are an expert in experimental design and methodology. Provide a comprehensive analysis of the experimental protocols, methods, and procedures used in this scientific paper."""
        
        user_prompt = f"""
Provide a deep analysis of the experimental methods and protocols in this paper. Focus on:

1. **Experimental Design**: Overall study design, controls, randomization
2. **Detailed Protocols**: Step-by-step experimental procedures
3. **Materials and Equipment**: Specific reagents, instruments, software
4. **Sample Preparation**: How samples were prepared and handled
5. **Data Collection**: Methods for data acquisition and measurement
6. **Quality Control**: Measures to ensure data quality and reliability
7. **Statistical Analysis**: Statistical methods and their appropriateness
8. **Reproducibility**: Information needed for others to reproduce the work

Paper Text:
{text}

Respond in detailed JSON format:
{{
    "experimental_design": {{
        "study_type": "observational|experimental|quasi-experimental|meta-analysis",
        "design_classification": "randomized controlled|case-control|cohort|cross-sectional|other",
        "sample_size": "Number of subjects/samples",
        "sample_size_justification": "How was sample size determined?",
        "randomization": "Description of randomization procedures",
        "blinding": "Single|double|triple blind or none",
        "control_groups": ["Description of control groups"],
        "inclusion_criteria": ["Criteria for inclusion"],
        "exclusion_criteria": ["Criteria for exclusion"]
    }},
    "detailed_protocols": [
        {{
            "protocol_name": "Name of experimental procedure",
            "step_by_step": ["Step 1", "Step 2", "Step 3"],
            "duration": "Time required for protocol",
            "critical_parameters": {{"parameter": "value", "tolerance": "acceptable range"}},
            "troubleshooting_notes": "Common issues and solutions"
        }}
    ],
    "materials_equipment": {{
        "reagents": [
            {{
                "name": "Reagent name",
                "supplier": "Company",
                "catalog_number": "Cat #",
                "concentration": "Working concentration",
                "storage_conditions": "Storage requirements"
            }}
        ],
        "instruments": [
            {{
                "instrument": "Instrument name",
                "manufacturer": "Company",
                "model": "Model number",
                "specifications": "Key technical specifications",
                "calibration": "Calibration procedures"
            }}
        ],
        "software": [
            {{
                "software": "Software name",
                "version": "Version number",
                "purpose": "What it was used for",
                "settings": "Key settings or parameters"
            }}
        ]
    }},
    "sample_preparation": {{
        "sample_types": ["Type 1", "Type 2"],
        "preparation_procedures": ["Procedure 1", "Procedure 2"],
        "preservation_methods": "How samples were preserved",
        "processing_timeline": "Timeline from collection to analysis",
        "quality_checks": "Quality control measures for samples"
    }},
    "data_collection": {{
        "measurement_methods": ["Method 1", "Method 2"],
        "data_acquisition_parameters": {{"parameter": "value"}},
        "measurement_frequency": "How often measurements were taken",
        "data_recording_procedures": "How data was recorded and stored",
        "automated_vs_manual": "Which procedures were automated vs manual"
    }},
    "quality_control": {{
        "internal_standards": ["Standard 1", "Standard 2"],
        "calibration_procedures": "How instruments were calibrated",
        "validation_experiments": "Experiments to validate methods",
        "error_detection": "Methods for detecting errors",
        "data_verification": "Steps to verify data accuracy"
    }},
    "statistical_analysis": {{
        "statistical_software": "Software used for analysis",
        "statistical_tests": [
            {{
                "test_name": "Name of statistical test",
                "purpose": "What it was used to test",
                "assumptions": ["Assumption 1", "Assumption 2"],
                "significance_level": "Alpha level (e.g., 0.05)"
            }}
        ],
        "multiple_testing_correction": "How multiple comparisons were handled",
        "missing_data_handling": "How missing data was addressed",
        "effect_size_measures": ["Measure 1", "Measure 2"]
    }},
    "reproducibility_information": {{
        "protocol_availability": "Where detailed protocols can be found",
        "code_availability": "Information about code sharing",
        "data_availability": "Information about data sharing",
        "materials_availability": "How to obtain materials",
        "standardization": "Use of standard protocols or guidelines"
    }},
    "methodological_innovations": "New methods or modifications introduced",
    "method_limitations": ["Limitation 1", "Limitation 2"],
    "analysis_notes": ["Important observations about methods analysis"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "experimental_design": {"study_type": "Could not determine"},
                "analysis_notes": ["Error during methods analysis"]
            }
            
            return self._validate_and_parse_json(response_text, "methods analysis", fallback_structure, text, "methods")
            
        except Exception as e:
            print(f"Error analyzing methods section: {e}")
            return {
                "experimental_design": {"study_type": "Could not determine"},
                "analysis_notes": [f"Error during methods analysis: {str(e)}"]
            }
    
    def _analyze_introduction_section(self, text: str) -> Dict[str, Any]:
        """Analyze introduction section for context and background"""
        
        system_prompt = """You are an expert at analyzing scientific paper introductions. Extract and analyze the introduction section to understand the research context, background, and motivation."""
        
        user_prompt = f"""
Analyze the introduction section of this scientific paper. Provide comprehensive analysis of:

1. **Research Context**: What field/domain and specific area of research
2. **Background Knowledge**: Key concepts, theories, and prior work mentioned
3. **Problem Statement**: What problem or gap is being addressed
4. **Literature Review**: Key references and how they relate to current work
5. **Research Motivation**: Why this research is important/needed
6. **Novelty Claims**: What makes this work unique or different
7. **Paper Structure**: How the paper is organized (if mentioned)

Paper Text:
{text}

Respond in detailed JSON format:
{{
    "research_context": {{
        "field": "Primary research field",
        "subfield": "Specific area of research",
        "interdisciplinary_connections": ["field1", "field2"]
    }},
    "background_knowledge": {{
        "key_concepts": ["concept1", "concept2"],
        "foundational_theories": ["theory1", "theory2"],
        "established_methods": ["method1", "method2"]
    }},
    "problem_statement": {{
        "main_problem": "Clear statement of the problem being addressed",
        "knowledge_gaps": ["gap1", "gap2"],
        "limitations_of_existing_work": ["limitation1", "limitation2"],
        "research_questions": ["question1", "question2"]
    }},
    "literature_review": {{
        "key_studies": [
            {{
                "study": "Author et al. (Year)",
                "contribution": "What this study contributed",
                "relevance": "How it relates to current work"
            }}
        ],
        "research_trends": ["trend1", "trend2"],
        "methodological_approaches": ["approach1", "approach2"]
    }},
    "research_motivation": {{
        "scientific_importance": "Why this research matters scientifically",
        "practical_applications": ["application1", "application2"],
        "potential_impact": "Expected impact of this research"
    }},
    "novelty_claims": {{
        "technical_novelty": "What is technically new",
        "methodological_novelty": "New methods or approaches",
        "conceptual_novelty": "New ideas or frameworks",
        "empirical_novelty": "New data or observations"
    }},
    "paper_organization": "Description of how the paper is structured",
    "analysis_notes": ["Important observations about the introduction"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "research_context": {"field": "Could not extract"},
                "problem_statement": {"main_problem": "Could not extract"},
                "analysis_notes": ["Error during introduction analysis"]
            }
            
            return self._validate_and_parse_json(response_text, "introduction analysis", fallback_structure, text, "introduction")
            
        except Exception as e:
            print(f"Error analyzing introduction section: {e}")
            return {
                "research_context": {"field": "Could not extract"},
                "problem_statement": {"main_problem": "Could not extract"},
                "analysis_notes": [f"Error during introduction analysis: {str(e)}"]
            }
    
    def _analyze_discussion_section(self, text: str, hypothesis_info: Dict[str, Any], results_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze discussion section for interpretation and implications"""
        
        system_prompt = """You are an expert at analyzing scientific paper discussions. Analyze how authors interpret their results, discuss implications, and relate findings to the broader field."""
        
        user_prompt = f"""
Analyze the discussion section of this scientific paper. Consider the hypothesis and results context provided. Focus on:

1. **Results Interpretation**: How authors interpret their findings
2. **Hypothesis Discussion**: How results relate to original hypotheses
3. **Comparison to Literature**: How findings compare to previous work
4. **Mechanisms**: Proposed explanations for observed phenomena
5. **Implications**: Scientific and practical implications
6. **Limitations**: Study limitations acknowledged by authors
7. **Future Directions**: Suggested future research
8. **Broader Impact**: Significance for the field

Original Hypothesis:
{json.dumps(hypothesis_info.get('main_hypothesis', {}), indent=2)}

Key Results:
{json.dumps(results_info.get('key_findings', [])[:3], indent=2)}

Paper Text:
{text}

Respond in detailed JSON format:
{{
    "results_interpretation": {{
        "main_findings_interpretation": "How authors interpret their key findings",
        "unexpected_results_discussion": "Discussion of unexpected or surprising results",
        "statistical_significance_discussion": "How statistical results are interpreted",
        "effect_size_discussion": "Discussion of practical significance"
    }},
    "hypothesis_discussion": {{
        "hypothesis_support": "How results support or contradict hypotheses",
        "alternative_explanations": ["explanation1", "explanation2"],
        "hypothesis_refinement": "How hypotheses might be refined based on results"
    }},
    "literature_comparison": {{
        "consistent_findings": ["Finding consistent with Study A", "Finding consistent with Study B"],
        "contradictory_findings": ["Finding that contradicts Study C"],
        "novel_contributions": ["What this study adds to the literature"],
        "field_advancement": "How this work advances the field"
    }},
    "mechanisms": {{
        "proposed_mechanisms": ["mechanism1", "mechanism2"],
        "mechanistic_evidence": ["evidence1", "evidence2"],
        "causal_relationships": ["relationship1", "relationship2"]
    }},
    "implications": {{
        "theoretical_implications": ["implication1", "implication2"],
        "methodological_implications": ["implication1", "implication2"],
        "practical_applications": ["application1", "application2"],
        "policy_implications": ["implication1", "implication2"]
    }},
    "limitations": {{
        "methodological_limitations": ["limitation1", "limitation2"],
        "sample_limitations": ["limitation1", "limitation2"],
        "analytical_limitations": ["limitation1", "limitation2"],
        "generalizability_limitations": ["limitation1", "limitation2"]
    }},
    "future_directions": {{
        "immediate_next_steps": ["step1", "step2"],
        "long_term_research_agenda": ["agenda1", "agenda2"],
        "methodological_improvements": ["improvement1", "improvement2"],
        "new_research_questions": ["question1", "question2"]
    }},
    "broader_impact": {{
        "field_significance": "Significance for the research field",
        "interdisciplinary_impact": "Impact on other fields",
        "societal_relevance": "Relevance to society or applications"
    }},
    "analysis_notes": ["Important observations about the discussion"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "results_interpretation": {"main_findings_interpretation": "Could not extract"},
                "implications": {"theoretical_implications": []},
                "analysis_notes": ["Error during discussion analysis"]
            }
            
            return self._validate_and_parse_json(response_text, "discussion analysis", fallback_structure, text, "discussion")
            
        except Exception as e:
            print(f"Error analyzing discussion section: {e}")
            return {
                "results_interpretation": {"main_findings_interpretation": "Could not extract"},
                "implications": {"theoretical_implications": []},
                "analysis_notes": [f"Error during discussion analysis: {str(e)}"]
            }
    
    def _analyze_conclusion_section(self, text: str) -> str:
        """Extract and analyze conclusion section"""
        
        system_prompt = """You are an expert at analyzing scientific paper conclusions. Extract the conclusion and summarize the key takeaways."""
        
        user_prompt = f"""
Extract and analyze the conclusion section of this scientific paper. Provide a clear, comprehensive summary that captures:

1. Main findings and their significance
2. How objectives were met
3. Key contributions to the field
4. Final recommendations or implications
5. Closing statements about future work

Paper Text:
{text}

Respond with a well-structured conclusion summary that preserves the authors' key messages while being clear and comprehensive. Do not use JSON format - provide a natural text summary.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            return response_text.strip()
            
        except Exception as e:
            print(f"Error analyzing conclusion section: {e}")
            return f"Could not extract conclusion. Error: {str(e)}"
    
    def _extract_acknowledgments_section(self, text: str) -> str:
        """Extract acknowledgments section"""
        
        system_prompt = """You are an expert at extracting acknowledgments from scientific papers. Find and extract the acknowledgments section."""
        
        user_prompt = f"""
Extract the acknowledgments section from this scientific paper. Look for sections typically titled:
- Acknowledgments
- Acknowledgements  
- Funding
- Acknowledgment of Support
- Author Contributions
- Conflict of Interest statements

Paper Text:
{text}

If found, provide the complete acknowledgments text. If not found, respond with "No acknowledgments section found." Do not use JSON format - provide the raw acknowledgments text.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            return response_text.strip()
            
        except Exception as e:
            print(f"Error extracting acknowledgments: {e}")
            return f"Could not extract acknowledgments. Error: {str(e)}"
    
    def _extract_references_section(self, text: str) -> List[str]:
        """Extract references/bibliography section"""
        
        system_prompt = """You are an expert at extracting references from scientific papers. Find and parse the references section."""
        
        user_prompt = f"""
Extract the references/bibliography section from this scientific paper. Look for sections typically titled:
- References
- Bibliography
- Literature Cited
- Works Cited

Parse the references and return them as a list. Each reference should be a complete citation.

Paper Text:
{text}

Respond in JSON format:
{{
    "references": [
        "Complete citation 1",
        "Complete citation 2"
    ],
    "reference_count": "number of references found",
    "extraction_notes": ["Any issues or observations"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "references": [f"Could not extract references"]
            }
            
            result = self._validate_and_parse_json(response_text, "references extraction", fallback_structure)
            return result.get('references', [])
            
        except Exception as e:
            print(f"Error extracting references: {e}")
            return [f"Could not extract references. Error: {str(e)}"]

    def _analyze_results_section(self, text: str, hypothesis_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results with focus on hypothesis validation"""
        
        system_prompt = """You are an expert at analyzing scientific results and their relationship to hypotheses. Determine whether results support or contradict hypotheses and identify new hypotheses that emerge."""
        
        user_prompt = f"""
Analyze the results section of this paper with special focus on hypothesis validation. Consider the hypothesis information provided and determine:

1. **Hypothesis Validation**: Do the results support or contradict the stated hypotheses?
2. **Statistical Significance**: What statistical evidence is provided?
3. **Effect Sizes**: What is the magnitude of observed effects?
4. **Confidence Intervals**: What uncertainty is associated with results?
5. **New Hypotheses**: What new hypotheses emerge from these results?

Original Hypothesis Information:
{json.dumps(hypothesis_info, indent=2)}

Results Text:
{text}

Respond in detailed JSON format:
{{
    "hypothesis_validation": {{
        "main_hypothesis_supported": true/false,
        "support_strength": "strong|moderate|weak|none",
        "evidence_summary": "Summary of evidence for/against hypothesis",
        "statistical_significance": {{
            "p_values": ["p < 0.05", "p = 0.001", "etc."],
            "confidence_intervals": ["95% CI: [x, y]"],
            "effect_sizes": ["Cohen's d = 0.8", "r² = 0.6", "etc."]
        }},
        "sub_hypotheses_validation": [
            {{
                "hypothesis": "Sub-hypothesis statement",
                "supported": true/false,
                "evidence": "Evidence for this sub-hypothesis"
            }}
        ]
    }},
    "key_findings": [
        {{
            "finding": "Description of key finding",
            "significance": "Statistical and practical significance",
            "unexpected": true/false,
            "implication": "What this finding means"
        }}
    ],
    "quantitative_results": {{
        "primary_outcomes": [
            {{
                "measure": "Primary outcome measure",
                "value": "Numerical result",
                "uncertainty": "Error bars, CI, etc.",
                "comparison": "Comparison to control/baseline"
            }}
        ],
        "secondary_outcomes": [
            {{
                "measure": "Secondary outcome measure",
                "value": "Numerical result",
                "significance": "Statistical significance"
            }}
        ]
    }},
    "qualitative_observations": [
        "Qualitative observation 1",
        "Qualitative observation 2"
    ],
    "unexpected_results": [
        {{
            "result": "Description of unexpected result",
            "significance": "Why this was unexpected",
            "potential_explanation": "Possible explanation"
        }}
    ],
    "new_hypotheses_generated": [
        {{
            "hypothesis": "New hypothesis statement",
            "basis": "What results led to this hypothesis",
            "testability": "How this could be tested",
            "priority": "high|medium|low priority for future research"
        }}
    ],
    "contradictory_results": [
        {{
            "result": "Result that contradicts expectations",
            "contradiction_details": "What it contradicts and why",
            "potential_resolution": "How this might be resolved"
        }}
    ],
    "data_quality_assessment": {{
        "completeness": "Assessment of data completeness",
        "reliability": "Assessment of data reliability",
        "validity": "Assessment of data validity",
        "limitations": ["Data limitation 1", "Data limitation 2"]
    }},
    "future_research_directions": [
        "Research direction 1 based on results",
        "Research direction 2 based on results"
    ],
    "analysis_notes": ["Important observations about results analysis"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            fallback_structure = {
                "hypothesis_validation": {"main_hypothesis_supported": "unknown"},
                "analysis_notes": ["Error during results analysis"]
            }
            
            return self._validate_and_parse_json(response_text, "results analysis", fallback_structure, text, "results")
            
        except Exception as e:
            print(f"Error analyzing results section: {e}")
            return {
                "hypothesis_validation": {"main_hypothesis_supported": "unknown"},
                "analysis_notes": [f"Error during results analysis: {str(e)}"]
            }
    
    def analyze_paper(self, pdf_path: str) -> Optional[PaperStructure]:
        """Analyze a complete scientific paper and extract structured information"""
        
        print(f"Analyzing paper: {pdf_path}")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return None
        
        # Generate paper ID from file path
        paper_id = hashlib.md5(str(pdf_path).encode()).hexdigest()
        
        processing_notes = []
        
        try:
            # Analyze basic structure
            print("  Extracting basic structure...")
            basic_info = self._analyze_basic_structure(text)
            
            # Generate keywords if missing
            keywords = basic_info.get('keywords', [])
            if not keywords or len(keywords) == 0:
                print("  No keywords found, generating keywords...")
                title = basic_info.get('title', '')
                abstract = basic_info.get('abstract', '')
                generated_keywords = self._generate_keywords(title, abstract, text)
                basic_info['keywords'] = generated_keywords
                basic_info['extraction_notes'] = basic_info.get('extraction_notes', []) + [f"Generated {len(generated_keywords)} keywords automatically"]
                print(f"  Generated keywords: {', '.join(generated_keywords)}")
            else:
                print(f"  Found {len(keywords)} existing keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
            
            # Analyze hypothesis section
            print("  Analyzing hypothesis...")
            hypothesis_info = self._analyze_hypothesis_section(text)
            
            # Analyze theory and computational sections
            print("  Analyzing theory and computational methods...")
            theory_comp_info = self._analyze_theory_computational_section(text)
            
            # Analyze methods section
            print("  Analyzing experimental methods...")
            methods_info = self._analyze_methods_section(text)
            
            # Analyze results section with hypothesis validation
            print("  Analyzing results and hypothesis validation...")
            results_info = self._analyze_results_section(text, hypothesis_info)
            
            # Analyze introduction section
            print("  Analyzing introduction...")
            introduction_info = self._analyze_introduction_section(text)
            
            # Analyze discussion section
            print("  Analyzing discussion...")
            discussion_info = self._analyze_discussion_section(text, hypothesis_info, results_info)
            
            # Extract conclusion section
            print("  Extracting conclusion...")
            conclusion_info = self._analyze_conclusion_section(text)
            
            # Extract acknowledgments
            print("  Extracting acknowledgments...")
            acknowledgments_info = self._extract_acknowledgments_section(text)
            
            # Extract references
            print("  Extracting references...")
            references_info = self._extract_references_section(text)
            
            # Create initial paper structure
            paper_structure = PaperStructure(
                paper_id=paper_id,
                original_filename=os.path.basename(pdf_path),
                title=basic_info.get('title', 'Unknown Title'),
                abstract=basic_info.get('abstract', ''),
                keywords=basic_info.get('keywords', []),
                
                # Enhanced sections - all now implemented
                introduction=introduction_info,
                hypothesis=hypothesis_info,
                theory_computational=theory_comp_info,
                methods=methods_info,
                results=results_info,
                discussion=discussion_info,
                conclusion=conclusion_info,
                acknowledgments=acknowledgments_info,
                references=references_info,
                
                # Metadata
                extraction_timestamp=time.time(),
                processing_notes=processing_notes + basic_info.get('extraction_notes', [])
            )
            
            # Post-process for synthesis if force_synthesis is enabled
            paper_structure = self._post_process_sections_for_synthesis(paper_structure, text)
            
            return paper_structure
            
        except Exception as e:
            print(f"Error analyzing paper {pdf_path}: {e}")
            processing_notes.append(f"Analysis error: {str(e)}")
            return None

def save_json_output(paper_structure: PaperStructure, output_dir: str):
    """Save paper structure as JSON file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_filename = f"{paper_structure.paper_id}.json"
    json_path = output_path / json_filename
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(paper_structure), f, indent=2, ensure_ascii=False)
    
    print(f"JSON output saved to: {json_path}")
    return json_path

def _get_section_content(section_data: Dict[str, Any], section_type: str) -> str:
    """Get section content, preferring narrative over structured data"""
    if isinstance(section_data, dict) and 'narrative' in section_data:
        # Use synthesized narrative content
        content = section_data['narrative']
        if section_data.get('synthesis_note'):
            content += f"\n\n*{section_data['synthesis_note']}*"
        return content
    
    # Fallback to structured content for specific sections
    fallback_generators = {
        'introduction': lambda data: _generate_structured_introduction(data),
        'hypothesis': lambda data: _generate_structured_hypothesis(data),
        'methods': lambda data: _generate_structured_methods(data),
        'results': lambda data: _generate_structured_results(data),
        'discussion': lambda data: _generate_structured_discussion(data),
        'theory_computational': lambda data: _generate_structured_theory(data)
    }
    
    if section_type in fallback_generators:
        return fallback_generators[section_type](section_data)
    
    return "Content not available"

def _generate_structured_introduction(data: Dict[str, Any]) -> str:
    """Generate structured introduction content"""
    return f"""### Research Context
**Field:** {data.get('research_context', {}).get('field', 'Not specified')}  
**Subfield:** {data.get('research_context', {}).get('subfield', 'Not specified')}

### Problem Statement
{data.get('problem_statement', {}).get('main_problem', 'Not provided')}

### Research Motivation
{data.get('research_motivation', {}).get('scientific_importance', 'Not provided')}

### Novelty Claims
- **Technical:** {data.get('novelty_claims', {}).get('technical_novelty', 'Not specified')}
- **Methodological:** {data.get('novelty_claims', {}).get('methodological_novelty', 'Not specified')}
- **Conceptual:** {data.get('novelty_claims', {}).get('conceptual_novelty', 'Not specified')}"""

def _generate_structured_hypothesis(data: Dict[str, Any]) -> str:
    """Generate structured hypothesis content"""
    content = f"""### Main Hypothesis
**Statement:** {data.get('main_hypothesis', {}).get('statement', 'Not identified')}

**Formal Notation:** {data.get('main_hypothesis', {}).get('formal_notation', 'Not provided')}

**Null Hypothesis:** {data.get('main_hypothesis', {}).get('null_hypothesis', 'Not provided')}

### Hypothesis Hallmarks Analysis
"""
    
    # Add hypothesis hallmarks
    hallmarks = data.get('hypothesis_hallmarks_analysis', {})
    for criterion, analysis in hallmarks.items():
        if isinstance(analysis, dict):
            score = analysis.get('score', 'N/A')
            rationale = analysis.get('rationale', 'Not provided')
            content += f"- **{criterion.title()}** ({score}/5): {rationale}\n"
    
    content += f"\n### Theoretical Foundation\n{data.get('theoretical_foundation', 'Not provided')}"
    return content

def _generate_structured_methods(data: Dict[str, Any]) -> str:
    """Generate structured methods content"""
    content = f"""### Study Design
**Type:** {data.get('experimental_design', {}).get('study_type', 'Not specified')}  
**Sample Size:** {data.get('experimental_design', {}).get('sample_size', 'Not specified')}

### Protocols
"""
    
    # Add detailed protocols
    protocols = data.get('detailed_protocols', [])
    for i, protocol in enumerate(protocols, 1):
        if isinstance(protocol, dict):
            content += f"""
#### Protocol {i}: {protocol.get('protocol_name', 'Unnamed Protocol')}
**Duration:** {protocol.get('duration', 'Not specified')}

**Steps:**
"""
            steps = protocol.get('steps', [])
            for step in steps:
                content += f"- {step}\n"
    
    return content

def _generate_structured_results(data: Dict[str, Any]) -> str:
    """Generate structured results content"""
    content = f"""### Hypothesis Validation
**Main Hypothesis Supported:** {data.get('hypothesis_validation', {}).get('main_hypothesis_supported', 'Unknown')}  
**Support Strength:** {data.get('hypothesis_validation', {}).get('support_strength', 'Not assessed')}  
**Evidence Summary:** {data.get('hypothesis_validation', {}).get('evidence_summary', 'Not provided')}

### Key Findings
"""
    
    # Add key findings
    findings = data.get('key_findings', [])
    for finding in findings:
        if isinstance(finding, dict):
            content += f"""- **{finding.get('finding', 'Unknown finding')}**
  - Significance: {finding.get('significance', 'Not specified')}
  - Unexpected: {finding.get('unexpected', 'Not specified')}
"""
    
    return content

def _generate_structured_discussion(data: Dict[str, Any]) -> str:
    """Generate structured discussion content"""
    return f"""### Results Interpretation
{data.get('results_interpretation', {}).get('main_findings_interpretation', 'Not provided')}

### Theoretical Implications
{', '.join(data.get('implications', {}).get('theoretical_implications', []))}

### Practical Applications
{', '.join(data.get('implications', {}).get('practical_applications', []))}

### Study Limitations
{', '.join(data.get('limitations', []))}

### Future Directions
{', '.join(data.get('future_research_directions', []))}"""

def _generate_structured_theory(data: Dict[str, Any]) -> str:
    """Generate structured theory/computational content"""
    content = f"""### Theoretical Framework
{data.get('theoretical_framework', {}).get('theoretical_basis_description', 'Not provided')}

### Mathematical Models
"""
    
    # Add mathematical models
    equations = data.get('mathematical_models', {}).get('equations', [])
    for eq in equations:
        if isinstance(eq, dict):
            content += f"- **{eq.get('equation', 'N/A')}**: {eq.get('description', 'No description')}\n"
    
    content += f"""
### Computational Methods
**Software Tools:** {', '.join(data.get('computational_methods', {}).get('software_tools', []))}  
**Programming Languages:** {', '.join(data.get('computational_methods', {}).get('programming_languages', []))}"""
    
    return content

def generate_markdown_output(paper_structure: PaperStructure, output_dir: str):
    """Generate human-readable markdown version"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    md_filename = f"{paper_structure.paper_id}.md"
    md_path = output_path / md_filename
    
    # Check if keywords were generated
    keywords_generated = any("Generated" in note and "keywords" in note for note in paper_structure.processing_notes)
    keywords_text = ', '.join(paper_structure.keywords) if paper_structure.keywords else 'None specified'
    if keywords_generated and paper_structure.keywords:
        keywords_text += " *(AI-generated)*"
    
    markdown_content = f"""# {paper_structure.title}

**Paper ID:** `{paper_structure.paper_id}`  
**Original File:** {paper_structure.original_filename}  
**Extraction Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(paper_structure.extraction_timestamp))}

## Abstract

{paper_structure.abstract}

## Keywords

{keywords_text}

## Introduction

{_get_section_content(paper_structure.introduction, 'introduction')}

## Hypothesis Analysis

{_get_section_content(paper_structure.hypothesis, 'hypothesis')}

## Theory and Computational Methods

{_get_section_content(paper_structure.theory_computational, 'theory_computational')}

## Experimental Methods

{_get_section_content(paper_structure.methods, 'methods')}

## Results and Hypothesis Validation

{_get_section_content(paper_structure.results, 'results')}

## Discussion

{_get_section_content(paper_structure.discussion, 'discussion')}

## Conclusion

{paper_structure.conclusion}

## Acknowledgments

{paper_structure.acknowledgments}

## References

"""
    
    # Add references
    for i, ref in enumerate(paper_structure.references[:10], 1):  # Limit to first 10 references for readability
        markdown_content += f"{i}. {ref}\n"
    
    if len(paper_structure.references) > 10:
        markdown_content += f"... and {len(paper_structure.references) - 10} more references\n"
    
    markdown_content += f"""
## Processing Notes
"""
    
    for note in paper_structure.processing_notes:
        markdown_content += f"- {note}\n"
    
    # Save markdown file
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Markdown output saved to: {md_path}")
    return md_path

def main():
    parser = argparse.ArgumentParser(description="Scientific Paper Synthesis and Organization Tool")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help='Single PDF file to analyze')
    group.add_argument('--directory', help='Directory containing PDF files')
    
    parser.add_argument('--model', required=True, help='Shortname of the OpenAI model (from model_servers.yaml)')
    parser.add_argument('--output-dir', default='./structured_papers', help='Output directory for results (default: ./structured_papers)')
    parser.add_argument('--no-markdown', action='store_true', help='Skip markdown output generation')
    parser.add_argument('--force-synthesis', action='store_true', help='Force synthesis of missing sections using AI model instead of leaving gaps')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        print(f"Initializing analyzer with model: {args.model}")
        if args.force_synthesis:
            print("Force synthesis mode enabled - all missing sections will be synthesized")
        analyzer = PaperAnalyzer(args.model, force_synthesis=args.force_synthesis)
        
        # Collect PDF files
        if args.file:
            pdf_files = [args.file]
        else:
            print(f"Scanning directory: {args.directory}")
            pdf_files = []
            for file_path in Path(args.directory).rglob('*.pdf'):
                pdf_files.append(str(file_path))
            print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("No PDF files found.")
            return
        
        # Process papers
        print(f"Processing {len(pdf_files)} papers...")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\nProcessing {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            
            try:
                # Analyze paper
                paper_structure = analyzer.analyze_paper(pdf_path)
                
                if paper_structure:
                    # Save JSON output
                    save_json_output(paper_structure, args.output_dir)
                    
                    # Generate markdown output
                    if not args.no_markdown:
                        generate_markdown_output(paper_structure, args.output_dir)
                    
                    print(f"  ✓ Successfully processed")
                else:
                    print(f"  ✗ Failed to process")
                
            except Exception as e:
                print(f"  ✗ Error processing {pdf_path}: {e}")
                continue
        
        print(f"\nProcessing completed. Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()