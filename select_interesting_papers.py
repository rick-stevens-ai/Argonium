#!/usr/bin/env python3
"""
select_interesting_papers.py - Paper Interestingness Selection Script

This script reads papers from a directory and identifies the most interesting subset
for human review based on 10 criteria. It reuses code patterns from make_v21.py and
download_papers_v8.py to fit into the existing workflow.

Usage:
    python select_interesting_papers.py --directory papers_dir --model gpt4 --top-n 20
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2
import yaml
from openai import OpenAI, OpenAIError

# Optional import for progress bar
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

MODEL_CONFIG_FILE = "model_servers.yaml"


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

    # 10 Criteria for Paper Interestingness - Redesigned for Scientific Significance
    CRITERIA = {
        "scientific_significance": {
            "description": "How fundamentally significant is this work to advancing scientific understanding? Does it challenge existing paradigms or reveal new fundamental principles?",
            "weight": 2.5,  # Primary metric
            "guidance": "Score 9-10: Paradigm-shifting discoveries. 7-8: Major advances. 5-6: Solid contributions. 3-4: Incremental work. 1-2: Minimal significance.",
        },
        "novel_mechanisms": {
            "description": "Does this work reveal novel biological, physical, or computational mechanisms? Are the underlying processes unexpected or previously unknown?",
            "weight": 2.0,  # High priority for mechanistic novelty
            "guidance": "Score 9-10: Completely novel mechanisms. 7-8: New variants of known mechanisms. 5-6: Some mechanistic insights. 3-4: Limited novelty. 1-2: No new mechanisms.",
        },
        "unexpected_outcomes": {
            "description": "How surprising or unexpected are the results? Do they contradict established beliefs or reveal counter-intuitive findings?",
            "weight": 1.8,  # High weight for surprises
            "guidance": "Score 9-10: Shocking, counter-intuitive results. 7-8: Surprising findings. 5-6: Some unexpected aspects. 3-4: Mostly expected. 1-2: Entirely predictable.",
        },
        "theoretical_breakthrough": {
            "description": "Does this work represent a theoretical breakthrough that fundamentally changes how we think about the problem domain?",
            "weight": 1.6,
            "guidance": "Score 9-10: Revolutionary theoretical insights. 7-8: Major theoretical advances. 5-6: Good theoretical work. 3-4: Minor theory. 1-2: No theoretical contribution.",
        },
        "methodological_innovation": {
            "description": "How innovative and rigorous are the experimental or computational methods? Do they enable previously impossible investigations?",
            "weight": 1.4,
            "guidance": "Score 9-10: Groundbreaking new methods. 7-8: Significant methodological advances. 5-6: Good methodology. 3-4: Standard methods. 1-2: Poor methodology.",
        },
        "cross_disciplinary_impact": {
            "description": "How likely is this work to impact multiple scientific disciplines and create new research directions?",
            "weight": 1.3,
            "guidance": "Score 9-10: Massive cross-field impact. 7-8: Strong interdisciplinary relevance. 5-6: Some cross-field potential. 3-4: Limited scope. 1-2: Single-field only.",
        },
        "paradigm_shift_potential": {
            "description": "How likely is this work to fundamentally change how the scientific community approaches this area?",
            "weight": 1.5,
            "guidance": "Score 9-10: Will rewrite textbooks. 7-8: Will change field practices. 5-6: Will influence approaches. 3-4: Minor influence. 1-2: No paradigm impact.",
        },
        "experimental_elegance": {
            "description": "How elegant, clever, and convincing are the experimental designs and validation approaches?",
            "weight": 1.1,
            "guidance": "Score 9-10: Brilliantly designed experiments. 7-8: Very clever designs. 5-6: Good experiments. 3-4: Adequate validation. 1-2: Poor experimental design.",
        },
        "intellectual_depth": {
            "description": "How intellectually deep and sophisticated is the thinking behind this work? Does it reveal deep insights?",
            "weight": 1.2,
            "guidance": "Score 9-10: Profound intellectual insights. 7-8: Deep thinking evident. 5-6: Good intellectual level. 3-4: Surface level. 1-2: Shallow thinking.",
        },
        "transformative_potential": {
            "description": "How likely is this work to be transformative for future research, technology, or understanding?",
            "weight": 1.4,
            "guidance": "Score 9-10: Highly transformative. 7-8: Strong transformation potential. 5-6: Some transformative aspects. 3-4: Limited impact. 1-2: No transformation.",
        },
    }

    def __init__(self, model_shortname: str):
        self.model_shortname = model_shortname
        self.client = self._setup_openai_client()
        self.model_config = self._load_model_config()
        self._rate_limit_lock = threading.Lock()
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum seconds between requests

    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file. Reused from download_papers_v8.py"""
        try:
            with open(MODEL_CONFIG_FILE, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: Model configuration file '{MODEL_CONFIG_FILE}' not found."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file '{MODEL_CONFIG_FILE}': {e}")

        model_config = None
        for server in config.get("servers", []):
            if server.get("shortname") == self.model_shortname:
                model_config = server
                break

        if not model_config:
            raise ValueError(
                f"Error: Model shortname '{self.model_shortname}' not found in '{MODEL_CONFIG_FILE}'."
            )

        return model_config

    def _setup_openai_client(self) -> OpenAI:
        """Setup OpenAI client. Reused pattern from download_papers_v8.py"""
        model_config = self._load_model_config()

        # Determine OpenAI API Key
        openai_api_key_config = model_config.get("openai_api_key")
        openai_api_key = None

        if openai_api_key_config == "${OPENAI_API_KEY}":
            openai_api_key = os.environ.get("OPENAI-API-KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
            if not openai_api_key:
                raise ValueError(
                    "Error: OpenAI API key is configured to use environment variable "
                    "'OPENAI-API-KEY' or 'OPENAI_API_KEY', but neither is set."
                )
        elif openai_api_key_config:
            openai_api_key = openai_api_key_config
        else:
            raise ValueError(
                f"Error: 'openai_api_key' not specified for model '{self.model_shortname}'."
            )

        # Get API base and model
        openai_api_base = model_config.get("openai_api_base")
        openai_model = model_config.get("openai_model")

        if not openai_api_base or not openai_model:
            raise ValueError(
                f"Error: 'openai_api_base' or 'openai_model' missing for model '{self.model_shortname}'."
            )

        try:
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
                timeout=120.0,  # Longer timeout for complex scoring
            )
            return client
        except Exception as e:
            raise ValueError(f"Error initializing OpenAI client: {e}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF. Reused pattern from make_v21.py"""
        try:
            with open(pdf_path, "rb") as file:
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

        if file_path.suffix.lower() == ".pdf":
            content = self._extract_text_from_pdf(str(file_path))
            # Try to extract title from first few lines
            lines = content.split("\n")[:10]
            title = next(
                (line.strip() for line in lines if len(line.strip()) > 10),
                file_path.stem,
            )
        elif file_path.suffix.lower() == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Try to extract title from content (assuming format from download script)
                if content.startswith("Title:"):
                    title_line = content.split("\n")[0]
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
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:  # If we can find a period in the last 20%
            truncated = truncated[: last_period + 1]

        return truncated + "\n\n[Content truncated for analysis]"

    def _rate_limit_wait(self):
        """Implement rate limiting to avoid overwhelming the API"""
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self._min_request_interval:
                sleep_time = self._min_request_interval - time_since_last
                time.sleep(sleep_time)

            self._last_request_time = time.time()

    def score_paper(self, file_path: str) -> Optional[PaperScore]:
        """Score a single paper based on the 10 criteria"""
        title, content = self._read_paper_content(file_path)

        if not content.strip():
            print(f"Warning: No content found for {file_path}")
            return None

        # Truncate content to manageable size
        truncated_content = self._truncate_content(content)

        # Create scoring prompt with detailed guidance
        criteria_descriptions = []
        for criterion, info in self.CRITERIA.items():
            criteria_descriptions.append(f"- **{criterion}**: {info['description']}")
            criteria_descriptions.append(f"  Scoring guidance: {info['guidance']}")

        system_prompt = """You are an expert scientific evaluator with deep expertise across multiple disciplines. Your task is to identify the most scientifically significant and intellectually exciting research papers. 

Focus on:
- Scientific breakthroughs that challenge existing paradigms
- Novel mechanisms and unexpected findings that surprise experts
- Work that fundamentally changes how we think about problems
- Research with transformative potential for future discovery

Use the FULL scoring range 1-10. Be generous with high scores (8-10) for truly exceptional work, and don't hesitate to use low scores (1-4) for incremental or routine research. Most papers should NOT cluster around 5-7."""

        user_prompt = f"""
Evaluate this research paper focusing on SCIENTIFIC SIGNIFICANCE and NOVELTY. Look for:

🔬 **Paradigm-shifting discoveries** that challenge established thinking
🧬 **Novel mechanisms** or unexpected biological/physical processes  
💡 **Counter-intuitive findings** that surprise experts in the field
🚀 **Transformative potential** for future research directions

Score each criterion from 1-10 using the FULL RANGE:
- **1-2**: Poor/Minimal (routine, incremental work)
- **3-4**: Below Average (limited novelty or significance)
- **5-6**: Average (solid but expected contributions)  
- **7-8**: Excellent (surprising findings, strong novelty)
- **9-10**: Outstanding (paradigm-shifting, transformative)

CRITERIA & GUIDANCE:
{chr(10).join(criteria_descriptions)}

**PAPER TITLE:** {title}

**PAPER CONTENT:**
{truncated_content}

Respond in JSON format with scores and detailed reasoning:
{{
    "scores": {{
        "scientific_significance": {{"score": X, "reasoning": "Explain why this score - what makes it significant or not?"}},
        "novel_mechanisms": {{"score": X, "reasoning": "What mechanisms are revealed? How novel are they?"}},
        "unexpected_outcomes": {{"score": X, "reasoning": "How surprising are the results? What was unexpected?"}},
        "theoretical_breakthrough": {{"score": X, "reasoning": "What theoretical advances does this make?"}},
        "methodological_innovation": {{"score": X, "reasoning": "How innovative are the methods and approaches?"}},
        "cross_disciplinary_impact": {{"score": X, "reasoning": "What fields will this impact beyond the primary domain?"}},
        "paradigm_shift_potential": {{"score": X, "reasoning": "How will this change the field's approach?"}},
        "experimental_elegance": {{"score": X, "reasoning": "How clever and convincing are the experimental designs?"}},
        "intellectual_depth": {{"score": X, "reasoning": "How deep and sophisticated is the intellectual contribution?"}},
        "transformative_potential": {{"score": X, "reasoning": "How transformative will this be for future research?"}}
    }},
    "overall_assessment": "Comprehensive assessment of why this paper is or isn't scientifically exciting and significant. Focus on breakthrough potential.",
    "surprise_factor": "What was most surprising or unexpected about this work?",
    "significance_summary": "In 1-2 sentences, why should the scientific community pay attention to this work?"
}}
"""

        try:
            # Apply rate limiting
            self._rate_limit_wait()

            print(f"Scoring paper: {title}")
            response = self.client.chat.completions.create(
                model=self.model_config["openai_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Low temperature for consistent scoring
                max_tokens=2000,
            )

            response_text = response.choices[0].message.content

            # Parse JSON response
            try:
                result = json.loads(response_text)
                scores = result.get("scores", {})
                overall_assessment = result.get("overall_assessment", "")
                surprise_factor = result.get("surprise_factor", "")
                significance_summary = result.get("significance_summary", "")

                # Calculate weighted total score
                total_score = 0.0
                criterion_scores = {}

                for criterion, info in self.CRITERIA.items():
                    if criterion in scores:
                        raw_score = scores[criterion].get("score", 0)
                        weighted_score = raw_score * info["weight"]
                        criterion_scores[criterion] = raw_score
                        total_score += weighted_score
                    else:
                        criterion_scores[criterion] = 0

                # Generate paper ID from file path
                paper_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]

                # Enhanced reasoning with surprise factor and significance
                enhanced_reasoning = overall_assessment
                if surprise_factor:
                    enhanced_reasoning += f"\n\nSurprise Factor: {surprise_factor}"
                if significance_summary:
                    enhanced_reasoning += f"\n\nSignificance: {significance_summary}"

                return PaperScore(
                    paper_id=paper_id,
                    title=title,
                    total_score=total_score,
                    criterion_scores=criterion_scores,
                    reasoning=enhanced_reasoning,
                    file_path=str(file_path),
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

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".txt"]:
            # Skip processed_papers.txt files
            if file_path.name == "processed_papers.txt":
                continue
            paper_files.append(str(file_path))

    return paper_files


def score_paper_wrapper(args):
    """Wrapper function for parallel processing"""
    scorer, file_path, paper_index, total_papers = args
    try:
        return scorer.score_paper(file_path)
    except Exception as e:
        print(f"Error scoring paper {file_path}: {e}")
        return None


def score_papers_parallel(
    scorer: PaperInterestinessScorer,
    paper_files: List[str],
    max_workers: int = 3,
    show_progress: bool = True,
) -> List[PaperScore]:
    """Score multiple papers in parallel with rate limiting and progress tracking"""
    scored_papers = []

    # Prepare arguments for parallel processing
    args_list = [
        (scorer, file_path, i, len(paper_files))
        for i, file_path in enumerate(paper_files)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(score_paper_wrapper, args): args for args in args_list
        }

        # Setup progress bar
        if show_progress and HAS_TQDM:
            progress_bar = tqdm(
                total=len(paper_files), desc="Scoring papers", unit="paper"
            )
        else:
            progress_bar = None

        # Collect results as they complete
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            file_path = args[1]

            try:
                result = future.result()
                if result:
                    scored_papers.append(result)

                    # Update progress bar with current stats
                    if progress_bar:
                        valid_scores = len(scored_papers)
                        progress_bar.set_postfix(
                            {
                                "Scored": valid_scores,
                                "Current": os.path.basename(file_path)[:20],
                            }
                        )

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

            # Update progress bar
            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

    return scored_papers


def save_results(scores: List[PaperScore], output_file: str):
    """Save scoring results to JSON file"""
    results = {
        "timestamp": time.time(),
        "total_papers": len(scores),
        "papers": [asdict(score) for score in scores],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Select most interesting papers from a directory"
    )
    parser.add_argument(
        "--directory", required=True, help="Directory containing papers (PDF/TXT files)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Shortname of the OpenAI model (from model_servers.yaml)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top papers to select (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="interesting_papers.json",
        help="Output file for results (default: interesting_papers.json)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold for inclusion",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Randomly sample N papers from directory before scoring (useful for large directories)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers for scoring (default: 3)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Minimum seconds between API requests (default: 1.0)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process papers sequentially)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()

    try:
        # Initialize scorer
        print(f"Initializing scorer with model: {args.model}")
        scorer = PaperInterestinessScorer(args.model)
        scorer._min_request_interval = args.rate_limit

        # Find paper files
        print(f"Scanning directory: {args.directory}")
        paper_files = find_paper_files(args.directory)
        print(f"Found {len(paper_files)} paper files")

        if not paper_files:
            print("No paper files found. Exiting.")
            return

        # Random sampling if requested
        if args.random_sample and args.random_sample < len(paper_files):
            print(
                f"Randomly sampling {args.random_sample} papers from {len(paper_files)} total papers"
            )
            print(f"Using random seed: {args.random_seed}")
            random.seed(args.random_seed)  # Set seed for reproducibility
            paper_files = random.sample(paper_files, args.random_sample)
            print(f"Selected {len(paper_files)} papers for scoring")

        # Score papers
        print("Scoring papers...")

        # Choose processing mode
        show_progress = not args.no_progress

        if args.no_parallel or len(paper_files) == 1:
            # Sequential processing
            print("Using sequential processing...")
            scored_papers = []

            # Setup progress bar for sequential mode
            if show_progress and HAS_TQDM:
                progress_bar = tqdm(paper_files, desc="Scoring papers", unit="paper")
            else:
                progress_bar = paper_files

            for i, file_path in enumerate(progress_bar, 1):
                if not (show_progress and HAS_TQDM):
                    print(
                        f"\nProgress: {i}/{len(paper_files)} - {os.path.basename(file_path)}"
                    )

                try:
                    score = scorer.score_paper(file_path)
                    if score and score.total_score >= args.min_score:
                        scored_papers.append(score)
                        if not (show_progress and HAS_TQDM):
                            print(f"  Score: {score.total_score:.2f}")
                    else:
                        if not (show_progress and HAS_TQDM):
                            print(f"  Skipped (score too low or failed to score)")

                    # Update progress bar
                    if (
                        show_progress
                        and HAS_TQDM
                        and hasattr(progress_bar, "set_postfix")
                    ):
                        progress_bar.set_postfix(
                            {
                                "Scored": len(scored_papers),
                                "Current": os.path.basename(file_path)[:20],
                            }
                        )

                except KeyboardInterrupt:
                    print("\nInterrupted by user. Saving partial results...")
                    break
                except Exception as e:
                    if not (show_progress and HAS_TQDM):
                        print(f"  Error scoring paper: {e}")
                    continue

            if show_progress and HAS_TQDM and hasattr(progress_bar, "close"):
                progress_bar.close()

        else:
            # Parallel processing
            print(f"Using parallel processing with {args.max_workers} workers...")
            print(f"Rate limit: {args.rate_limit} seconds between requests")

            try:
                all_scored_papers = score_papers_parallel(
                    scorer, paper_files, args.max_workers, show_progress
                )

                # Filter by minimum score
                scored_papers = [
                    paper
                    for paper in all_scored_papers
                    if paper.total_score >= args.min_score
                ]

            except KeyboardInterrupt:
                print("\nInterrupted by user. Saving partial results...")
                scored_papers = []

        # Sort by score and select top N
        scored_papers.sort(key=lambda x: x.total_score, reverse=True)
        top_papers = scored_papers[: args.top_n]

        # Save results
        save_results(scored_papers, args.output)

        # Display top papers
        print(f"\n" + "=" * 80)
        print(f"TOP {len(top_papers)} MOST INTERESTING PAPERS")
        print("=" * 80)

        for i, paper in enumerate(top_papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   🎯 Total Score: {paper.total_score:.2f}")
            print(f"   📄 File: {paper.file_path}")

            # Show top 3 criterion scores with weights
            sorted_criteria = sorted(
                paper.criterion_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_criteria = sorted_criteria[:3]
            print(
                f"   🏆 Top strengths: {', '.join([f'{k}({v}/10)' for k, v in top_criteria])}"
            )

            # Show key scientific metrics
            sci_sig = paper.criterion_scores.get("scientific_significance", 0)
            novel_mech = paper.criterion_scores.get("novel_mechanisms", 0)
            unexpected = paper.criterion_scores.get("unexpected_outcomes", 0)
            print(
                f"   🔬 Key metrics: Scientific Significance({sci_sig}/10), Novel Mechanisms({novel_mech}/10), Unexpected Outcomes({unexpected}/10)"
            )

            print(
                f"   📝 Assessment: {paper.reasoning[:200]}{'...' if len(paper.reasoning) > 200 else ''}"
            )

        print(f"\nTotal papers processed: {len(paper_files)}")
        print(f"Papers successfully scored: {len(scored_papers)}")
        print(f"Top papers selected: {len(top_papers)}")
        print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
