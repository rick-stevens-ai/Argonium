#!/usr/bin/env python3
"""
select_interesting_papers.py - Paper Interestingness Selection Script

This script reads papers from a directory and identifies the most interesting subset
for human review based on 10 criteria. It reuses code patterns from make_v21.py and
download_papers_v8.py to fit into the existing workflow.

Usage:
    python select_interesting_papers.py --directory papers_dir --model gpt4 --top-n 20
"""

import os
import sys
import json
import argparse
import yaml
import time
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI, OpenAIError
import PyPDF2
import re

MODEL_CONFIG_FILE = 'model_servers.yaml'

@dataclass
class PaperScore:
    """Data class to hold paper scoring information"""
    paper_id: str
    title: str
    total_score: float
    criterion_scores: Dict[str, float]
    reasoning: str
    file_path: str

class PaperInterestinessScorer:
    """
    Evaluates papers based on 10 criteria for interestingness.
    Reuses patterns from make_v21.py for OpenAI client setup.
    """
    
    # 10 Criteria for Paper Interestingness
    CRITERIA = {
        "novelty": {
            "description": "How novel or original are the ideas presented?",
            "weight": 1.2
        },
        "methodological_rigor": {
            "description": "How rigorous and well-designed is the methodology?",
            "weight": 1.1
        },
        "practical_impact": {
            "description": "How significant is the potential real-world impact?",
            "weight": 1.3
        },
        "theoretical_contribution": {
            "description": "How significant is the theoretical contribution to the field?",
            "weight": 1.1
        },
        "experimental_validation": {
            "description": "How comprehensive and convincing is the experimental validation?",
            "weight": 1.0
        },
        "interdisciplinary_relevance": {
            "description": "How relevant is this work across multiple disciplines?",
            "weight": 0.9
        },
        "clarity_and_presentation": {
            "description": "How clear and well-presented is the research?",
            "weight": 0.8
        },
        "reproducibility": {
            "description": "How reproducible are the results and methods?",
            "weight": 1.0
        },
        "building_on_prior_work": {
            "description": "How well does it build on and advance prior work?",
            "weight": 0.9
        },
        "future_research_potential": {
            "description": "How much potential does this work have for spawning future research?",
            "weight": 1.1
        }
    }
    
    def __init__(self, model_shortname: str):
        self.model_shortname = model_shortname
        self.client = self._setup_openai_client()
        self.model_config = self._load_model_config()
        
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file. Reused from download_papers_v8.py"""
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
    
    def _setup_openai_client(self) -> OpenAI:
        """Setup OpenAI client. Reused pattern from download_papers_v8.py"""
        model_config = self._load_model_config()
        
        # Determine OpenAI API Key
        openai_api_key_config = model_config.get('openai_api_key')
        openai_api_key = None
        
        if openai_api_key_config == "${OPENAI_API_KEY}":
            openai_api_key = os.environ.get('OPENAI-API-KEY') or os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("Error: OpenAI API key is configured to use environment variable "
                               "'OPENAI-API-KEY' or 'OPENAI_API_KEY', but neither is set.")
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
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
                timeout=120.0  # Longer timeout for complex scoring
            )
            return client
        except Exception as e:
            raise ValueError(f"Error initializing OpenAI client: {e}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF. Reused pattern from make_v21.py"""
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
    
    def _read_paper_content(self, file_path: str) -> Tuple[str, str]:
        """Read paper content from file (PDF or text). Returns (title, content)"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            content = self._extract_text_from_pdf(str(file_path))
            # Try to extract title from first few lines
            lines = content.split('\n')[:10]
            title = next((line.strip() for line in lines if len(line.strip()) > 10), file_path.stem)
        elif file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Try to extract title from content (assuming format from download script)
                if content.startswith("Title:"):
                    title_line = content.split('\n')[0]
                    title = title_line.replace("Title:", "").strip()
                else:
                    title = file_path.stem
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")
                return file_path.stem, ""
        else:
            print(f"Unsupported file format: {file_path}")
            return file_path.stem, ""
        
        return title, content
    
    def _truncate_content(self, content: str, max_chars: int = 10000) -> str:
        """Truncate content to fit within token limits"""
        if len(content) <= max_chars:
            return content
        
        # Try to truncate at a reasonable breakpoint
        truncated = content[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # If we can find a period in the last 20%
            truncated = truncated[:last_period + 1]
        
        return truncated + "\n\n[Content truncated for analysis]"
    
    def score_paper(self, file_path: str) -> Optional[PaperScore]:
        """Score a single paper based on the 10 criteria"""
        title, content = self._read_paper_content(file_path)
        
        if not content.strip():
            print(f"Warning: No content found for {file_path}")
            return None
        
        # Truncate content to manageable size
        truncated_content = self._truncate_content(content)
        
        # Create scoring prompt
        criteria_descriptions = []
        for criterion, info in self.CRITERIA.items():
            criteria_descriptions.append(f"- {criterion}: {info['description']}")
        
        system_prompt = """You are an expert research evaluator. Your task is to score academic papers based on specific criteria for interestingness and research value. You will provide numerical scores and reasoning for each criterion."""
        
        user_prompt = f"""
Evaluate the following research paper based on these 10 criteria. For each criterion, provide a score from 1-10 (where 10 is excellent) and brief reasoning.

CRITERIA:
{chr(10).join(criteria_descriptions)}

PAPER TITLE: {title}

PAPER CONTENT:
{truncated_content}

Please respond in the following JSON format:
{{
    "scores": {{
        "novelty": {{"score": X, "reasoning": "..."}},
        "methodological_rigor": {{"score": X, "reasoning": "..."}},
        "practical_impact": {{"score": X, "reasoning": "..."}},
        "theoretical_contribution": {{"score": X, "reasoning": "..."}},
        "experimental_validation": {{"score": X, "reasoning": "..."}},
        "interdisciplinary_relevance": {{"score": X, "reasoning": "..."}},
        "clarity_and_presentation": {{"score": X, "reasoning": "..."}},
        "reproducibility": {{"score": X, "reasoning": "..."}},
        "building_on_prior_work": {{"score": X, "reasoning": "..."}},
        "future_research_potential": {{"score": X, "reasoning": "..."}}
    }},
    "overall_assessment": "Brief overall assessment of the paper's interestingness and value"
}}
"""
        
        try:
            print(f"Scoring paper: {title}")
            response = self.client.chat.completions.create(
                model=self.model_config['openai_model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Low temperature for consistent scoring
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
                scores = result.get('scores', {})
                overall_assessment = result.get('overall_assessment', '')
                
                # Calculate weighted total score
                total_score = 0.0
                criterion_scores = {}
                
                for criterion, info in self.CRITERIA.items():
                    if criterion in scores:
                        raw_score = scores[criterion].get('score', 0)
                        weighted_score = raw_score * info['weight']
                        criterion_scores[criterion] = raw_score
                        total_score += weighted_score
                    else:
                        criterion_scores[criterion] = 0
                
                # Generate paper ID from file path
                paper_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                
                return PaperScore(
                    paper_id=paper_id,
                    title=title,
                    total_score=total_score,
                    criterion_scores=criterion_scores,
                    reasoning=overall_assessment,
                    file_path=str(file_path)
                )
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response for {file_path}: {e}")
                print(f"Response was: {response_text}")
                return None
            
        except OpenAIError as e:
            print(f"OpenAI API error while scoring {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error while scoring {file_path}: {e}")
            return None

def find_paper_files(directory: str) -> List[str]:
    """Find all paper files (PDF and TXT) in directory"""
    paper_files = []
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt']:
            # Skip processed_papers.txt files
            if file_path.name == 'processed_papers.txt':
                continue
            paper_files.append(str(file_path))
    
    return paper_files

def save_results(scores: List[PaperScore], output_file: str):
    """Save scoring results to JSON file"""
    results = {
        "timestamp": time.time(),
        "total_papers": len(scores),
        "papers": [asdict(score) for score in scores]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Select most interesting papers from a directory")
    parser.add_argument('--directory', required=True, help='Directory containing papers (PDF/TXT files)')
    parser.add_argument('--model', required=True, help='Shortname of the OpenAI model (from model_servers.yaml)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top papers to select (default: 10)')
    parser.add_argument('--output', default='interesting_papers.json', help='Output file for results (default: interesting_papers.json)')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum score threshold for inclusion')
    parser.add_argument('--random-sample', type=int, default=None, help='Randomly sample N papers from directory before scoring (useful for large directories)')
    
    args = parser.parse_args()
    
    try:
        # Initialize scorer
        print(f"Initializing scorer with model: {args.model}")
        scorer = PaperInterestinessScorer(args.model)
        
        # Find paper files
        print(f"Scanning directory: {args.directory}")
        paper_files = find_paper_files(args.directory)
        print(f"Found {len(paper_files)} paper files")
        
        if not paper_files:
            print("No paper files found. Exiting.")
            return
        
        # Random sampling if requested
        if args.random_sample and args.random_sample < len(paper_files):
            print(f"Randomly sampling {args.random_sample} papers from {len(paper_files)} total papers")
            random.seed(42)  # Set seed for reproducibility
            paper_files = random.sample(paper_files, args.random_sample)
            print(f"Selected {len(paper_files)} papers for scoring")
        
        # Score papers
        print("Scoring papers...")
        scored_papers = []
        
        for i, file_path in enumerate(paper_files, 1):
            print(f"\nProgress: {i}/{len(paper_files)} - {os.path.basename(file_path)}")
            
            try:
                score = scorer.score_paper(file_path)
                if score and score.total_score >= args.min_score:
                    scored_papers.append(score)
                    print(f"  Score: {score.total_score:.2f}")
                else:
                    print(f"  Skipped (score too low or failed to score)")
                
                # Add delay to avoid rate limiting
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving partial results...")
                break
            except Exception as e:
                print(f"  Error scoring paper: {e}")
                continue
        
        # Sort by score and select top N
        scored_papers.sort(key=lambda x: x.total_score, reverse=True)
        top_papers = scored_papers[:args.top_n]
        
        # Save results
        save_results(scored_papers, args.output)
        
        # Display top papers
        print(f"\n" + "="*80)
        print(f"TOP {len(top_papers)} MOST INTERESTING PAPERS")
        print("="*80)
        
        for i, paper in enumerate(top_papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Score: {paper.total_score:.2f}")
            print(f"   File: {paper.file_path}")
            print(f"   Assessment: {paper.reasoning}")
            
            # Show top 3 criterion scores
            sorted_criteria = sorted(paper.criterion_scores.items(), key=lambda x: x[1], reverse=True)
            top_criteria = sorted_criteria[:3]
            print(f"   Top strengths: {', '.join([f'{k}({v})' for k, v in top_criteria])}")
        
        print(f"\nTotal papers processed: {len(paper_files)}")
        print(f"Papers successfully scored: {len(scored_papers)}")
        print(f"Top papers selected: {len(top_papers)}")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()