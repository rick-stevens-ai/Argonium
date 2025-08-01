#!/usr/bin/env python3
"""
reasoning_traces_v5_robust.py - An improved version with better error handling and JSON extraction
"""

import json
import os
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import yaml
# OpenAI imports
from openai import OpenAI
from tqdm import tqdm

# Global variables
_start_time = time.time()
_total_questions = 0
_processed_questions = 0
_current_model_name = None  # Store current model name for discrepancy analysis


def log_message(message, log_level="INFO"):
    """Log a message with timestamp and log level."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{log_level}] {message}")


def parse_arguments():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate detailed reasoning chains as an expert's internal dialogue with blind prediction of the correct answer."
    )
    parser.add_argument(
        "input_file",
        help="JSON file containing multiple choice questions (output of make_v21.py)",
    )
    parser.add_argument(
        "--output",
        default="reasoning_traces.json",
        help="Output JSON file (default: reasoning_traces.json)",
    )
    parser.add_argument(
        "--model",
        default="gpt41",
        help="Model shortname from model_servers.yaml to use",
    )
    parser.add_argument(
        "--config",
        default="model_servers.yaml",
        help="Path to model configuration file (default: model_servers.yaml)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (default: all)",
    )
    parser.add_argument(
        "--specialty",
        default="expert",
        help='Specialty persona to adopt (e.g., "microbiologist", "quantum physicist", "historian") (default: expert)',
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["detailed", "focused", "efficient"],
        default="detailed",
        help="Reasoning approach: detailed (thorough analysis - current method), focused (structured reasoning with early elimination), efficient (streamlined for accuracy testing) (default: detailed)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save partial results after processing this many questions (default: 10)",
    )
    parser.add_argument(
        "--continue-from",
        default=None,
        help="Continue from a previously saved output file",
    )

    # Advanced analysis options
    advanced_group = parser.add_argument_group("Advanced Analysis Options")
    advanced_group.add_argument(
        "--whole-trace-analysis",
        action="store_true",
        help="Enable whole trace analysis to create a coherent narrative from the reasoning",
    )
    advanced_group.add_argument(
        "--whole-trace-model",
        help="Model to use for whole trace analysis (defaults to same as --model if not specified)",
    )
    advanced_group.add_argument(
        "--whole-trace-output",
        default="whole_trace_output.json",
        help="Output file for whole trace analysis (default: whole_trace_output.json)",
    )
    advanced_group.add_argument(
        "--enhanced-discrepancy",
        action="store_true",
        help="Enable enhanced discrepancy analysis with comprehensive debate about correct answers",
    )
    advanced_group.add_argument(
        "--dual-prediction",
        action="store_true",
        help="Add Argonium-style prediction after detailed reasoning and compare the two approaches",
    )
    advanced_group.add_argument(
        "--grading",
        help="Model to use for grading/verifying answers (defaults to same as --model if not specified)",
    )
    
    # Method-specific model options
    method_group = parser.add_argument_group("Method-Specific Model Options")
    method_group.add_argument(
        "--method0-model",
        help="Model to use for Method 0 (one-shot prediction) - defaults to --model if not specified",
    )
    method_group.add_argument(
        "--method1-analysis-model",
        help="Model to use for Method 1 individual option analysis - defaults to --model if not specified",
    )
    method_group.add_argument(
        "--method1-synthesis-model",
        help="Model to use for Method 1 synthesis and final reasoning - defaults to --model if not specified",
    )
    method_group.add_argument(
        "--method2-model", 
        help="Model to use for Method 2 (Argonium-style prediction) - defaults to --model if not specified",
    )
    advanced_group.add_argument(
        "--require-grading-model",
        action="store_true",
        help="Require a grading model to be specified and fail if grading fails (removes regex fallback)",
    )
    advanced_group.add_argument(
        "--verbose-grading",
        action="store_true",
        help="Show detailed grading LLM input and output for each question",
    )
    advanced_group.add_argument(
        "--capture-incorrect",
        help="Output file to capture incorrect answers with both predicted and correct answers",
    )
    advanced_group.add_argument(
        "--argonium-results",
        help="JSON file with existing argonium results (from argonium_score_parallel) to compare against",
    )
    advanced_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing questions (default: 1)",
    )
    advanced_group.add_argument(
        "--comparison-model",
        help="Model to use for comparison analysis between different reasoning methods",
    )
    advanced_group.add_argument(
        "--text-analysis-model", 
        help="Model to use for text analysis and coherent stream generation",
    )
    advanced_group.add_argument(
        "--split-output",
        action="store_true",
        help="Split output into individual JSON files per question",
    )
    advanced_group.add_argument(
        "--split-only",
        action="store_true", 
        help="Only split existing output file without regenerating reasoning traces",
    )
    advanced_group.add_argument(
        "--output-dir",
        help="Directory to output split files when using --split-output or --split-only",
    )
    advanced_group.add_argument(
        "--create-stream-files",
        action="store_true",
        help="Create individual '🌊 COHERENT STREAM OF THOUGHT ANALYSIS' files for each question in the output directory (requires --model to be specified)",
    )

    return parser.parse_args()


def configure_apis(
    model_name: str, config_file: str = "model_servers.yaml"
) -> tuple[OpenAI, str]:
    """
    Configure the necessary APIs based on model selection.

    Args:
        model_name: The model shortname to use
        config_file: Path to the model configuration file

    Returns:
        Tuple of (OpenAI client, actual model name to use with the API)
    """
    # Load the servers configuration
    try:
        with open(config_file, "r") as f:
            servers_config = yaml.safe_load(f)
    except Exception as e:
        log_message(f"Error loading {config_file}: {e}", log_level="ERROR")
        sys.exit(1)

    # Find the selected model's configuration
    selected_server = None
    for server in servers_config["servers"]:
        if server["shortname"] == model_name:
            selected_server = server
            break

    if not selected_server:
        log_message(
            f"Error: Model '{model_name}' not found in {config_file}", log_level="ERROR"
        )
        log_message(
            f"Available models: {', '.join(s['shortname'] for s in servers_config['servers'])}",
            log_level="INFO",
        )
        sys.exit(1)

    # Configure OpenAI API with server details
    log_message(
        f"Using model '{selected_server['openai_model']}' from {selected_server['server']}"
    )

    # Set OpenAI API parameters
    openai_api_key = selected_server["openai_api_key"]
    # Handle environment variables in the API key
    if openai_api_key.startswith("${") and openai_api_key.endswith("}"):
        env_var = openai_api_key[2:-1]
        openai_api_key = os.environ.get(env_var, "")
        if not openai_api_key:
            log_message(
                f"Error: Environment variable {env_var} is not set or empty",
                log_level="ERROR",
            )
            sys.exit(1)

    # Create OpenAI client with the new API
    client = OpenAI(api_key=openai_api_key, base_url=selected_server["openai_api_base"])

    # Return the client and actual model name to use with the API
    return client, selected_server["openai_model"]


def get_expert_persona(specialty: str) -> str:
    """
    Generate a detailed persona description for the selected specialty.

    Args:
        specialty: The expert specialty (e.g., microbiologist, quantum physicist, historian)

    Returns:
        A detailed persona description tailored to the specialty
    """
    # Pre-defined personas for common specialties
    predefined_personas = {
        "microbiologist": """I am a microbiologist with over 20 years of experience studying antimicrobial resistance and bacterial pathogenesis. 
I've spent countless hours in the lab isolating bacterial strains, conducting susceptibility tests, and analyzing genomic data. 
When I approach a scientific question, I consider the molecular mechanisms at play, evolutionary pressures, and ecological contexts. 
I'm particularly meticulous about methodology and constantly thinking about experimental design, controls, and statistical significance. 
I tend to connect new information to established principles in bacterial physiology, genetics, and ecology. 
I'm familiar with current literature on antimicrobial agents, resistance mechanisms, biofilms, and emerging therapeutic approaches.""",
        "physicist": """I am a physicist with over 20 years of experience in theoretical and computational physics. 
I've worked extensively on quantum mechanics, statistical mechanics, and particle physics. 
When I approach a physics problem, I consider the underlying physical principles, mathematical formulations, and experimental evidence.
I'm particularly attentive to mathematical rigor, dimensional analysis, and the implications of symmetries.
I tend to connect new information to established theories and look for consistency with fundamental laws.
I'm familiar with current research on quantum field theory, cosmology, condensed matter physics, and computational methods.""",
        "quantum physicist": """I am a quantum physicist with extensive experience in quantum mechanics, quantum field theory, and quantum computing. 
My research focuses on understanding the fundamental principles of quantum systems and their applications in technology. 
When approaching problems, I instinctively think about wave functions, quantum states, superposition, entanglement, and quantum measurement theory. 
I consider both the mathematical formalism and the conceptual interpretations of quantum phenomena. 
My approach is rigorous, often using advanced mathematical tools to analyze quantum systems and their behavior. 
I'm familiar with current research in quantum technologies, quantum information processing, and quantum foundations.""",
        "historian": """I am a historian with decades of experience in analyzing historical documents, events, and trends. 
My expertise involves critically examining primary and secondary sources, contextualizing events within their broader historical context. 
When analyzing historical questions, I consider multiple perspectives, sociopolitical factors, economic conditions, and cultural influences. 
I'm particularly attentive to the biases in historical accounts and the importance of evaluating the reliability of sources. 
My approach involves connecting specific events to larger historical patterns and understanding how past developments influence present conditions. 
I'm well-versed in historiography and the evolution of historical interpretations over time.""",
    }

    # Check if the specialty is in our predefined list
    if specialty.lower() in predefined_personas:
        return predefined_personas[specialty.lower()]

    # For unknown specialties, generate a generic expert persona based on the specialty name
    specialty_words = specialty.split()
    specialty_base = specialty_words[-1] if len(specialty_words) > 0 else specialty

    # Is it a scientific field?
    scientific_fields = [
        "biologist",
        "physicist",
        "chemist",
        "geologist",
        "astronomer",
        "mathematician",
        "engineer",
        "scientist",
        "researcher",
    ]

    is_scientific = any(field in specialty_base.lower() for field in scientific_fields)

    if is_scientific:
        return f"""I am a {specialty} with extensive expertise in my field. 
My work involves analyzing complex scientific problems using rigorous methodologies and detailed knowledge of {specialty} principles.
When approaching questions in my field, I think systematically about the underlying mechanisms, relevant theories, and empirical evidence.
I pay particular attention to scientific accuracy, methodological considerations, and the current state of research in {specialty}.
My approach combines theoretical understanding with practical knowledge of experimental techniques and data analysis.
I'm well-versed in the latest research and ongoing debates in the field of {specialty}."""
    else:
        # Generic expert persona for non-scientific fields
        return f"""I am a {specialty} with extensive expertise and experience in my field.
My work involves analyzing complex problems through the specialized lens of a {specialty}.
When approaching questions in my field, I consider multiple factors, theoretical frameworks, and practical implications.
I'm particularly attentive to the nuances, contexts, and specialized knowledge that inform {specialty} analysis.
My approach combines theoretical understanding with practical insights gained through years of experience.
I'm well-versed in the foundational principles, current developments, and ongoing debates in my field."""


def generate_reasoning_prompt(
    specialty: str, persona: str, question_text: str, options: List[str], reasoning_mode: str
) -> str:
    """
    Generate reasoning prompt based on the selected reasoning mode.
    
    Args:
        specialty: Expert specialty
        persona: Expert persona description
        question_text: The question text
        options: List of answer options
        reasoning_mode: "detailed", "focused", or "efficient"
    
    Returns:
        Complete prompt string
    """
    # Check if scientific field
    is_scientific = any(
        field in specialty.lower()
        for field in [
            "scientist", "biologist", "physicist", "chemist", "geologist",
            "astronomer", "mathematician", "engineer",
        ]
    )
    
    # Split the question text to get just the question part (without options)
    question_parts = question_text.split("\n\n", 1)
    question_only = question_parts[0] if len(question_parts) > 0 else question_text
    
    # Base prompt structure
    base_prompt = f"""You are a {specialty} reasoning through a multiple-choice question. Your persona: {persona}

QUESTION:
{question_only}

ANSWER OPTIONS:
"""
    
    # Add options
    for i, option in enumerate(options):
        base_prompt += f"{i+1}. {option}\n"
    
    # Add reasoning instructions based on mode
    if reasoning_mode == "detailed":
        return base_prompt + generate_overthink_instructions(specialty, is_scientific)
    elif reasoning_mode == "focused":
        return base_prompt + generate_balanced_instructions(specialty, is_scientific)
    elif reasoning_mode == "efficient":
        return base_prompt + generate_minimal_instructions(specialty, is_scientific)
    else:
        # Default to detailed
        return base_prompt + generate_overthink_instructions(specialty, is_scientific)


def generate_overthink_instructions(specialty: str, is_scientific: bool) -> str:
    """Generate detailed overthinking instructions (current method)."""
    if is_scientific:
        return f"""
TASK:
Please provide an extremely detailed internal monologue as if you are a {specialty} thinking through this problem. For each answer option:
1. Treat each option as a hypothesis that you're carefully considering
2. Use specialized terminology and concepts from your field in your reasoning
3. Consider relevant mechanisms, processes, theoretical frameworks, and evidence
4. Reason through the implications and logical consequences of each option
5. Reference relevant principles, theories, or frameworks from your field
6. Consider edge cases, exceptions, and nuances for each option
7. Express uncertainty and weigh evidence where appropriate

Structure your response as an expert's stream of consciousness:
- Analyze each option thoroughly in sequential order (Option 1, then Option 2, etc.)
- For each option, begin with "Hmm, let me consider option X..."
- After analyzing ALL options, explicitly predict which answer you think is correct using its NUMBER (e.g., "I predict option 3 is correct")
- Then explain your reasoning for your prediction - what principles and evidence led you to this conclusion?
- Finally, indicate your confidence level in your prediction (high, medium, or low) and explain why

Output your reasoning in JSON format with the following structure:
{{
  "thought_process": {{
    "option_1": "Detailed reasoning about option 1 as a hypothesis",
    "option_2": "Detailed reasoning about option 2 as a hypothesis",
    ... (all options in numerical order)
  }},
  "prediction": {{
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Reasoning for why you predict this answer is correct",
    "confidence_level": "Your confidence level (high, medium, or low)",
    "confidence_explanation": "Why you have this level of confidence in your prediction"
  }},
  "scientific_conclusion": "Final synthesized assessment"
}}

IMPORTANT: Your response must be a valid, parseable JSON object. For each option, include detailed reasoning of at least 150-200 words. You MUST make a prediction and specify ONLY the option number (e.g., '3', NOT 'Option 3').
"""
    else:
        return f"""
TASK:
Please provide an extremely detailed internal monologue as if you are a {specialty} thinking through this problem. For each answer option:
1. Treat each option as a possibility that you're carefully considering
2. Use specialized terminology and concepts from your field in your reasoning
3. Consider relevant frameworks, methodologies, contexts, and evidence
4. Reason through the implications and logical consequences of each option
5. Reference relevant principles, theories, or frameworks from your field
6. Consider edge cases, exceptions, and nuances for each option
7. Express uncertainty and weigh evidence where appropriate

Structure your response as an expert's stream of consciousness:
- Analyze each option thoroughly in sequential order (Option 1, then Option 2, etc.)
- For each option, begin with "Hmm, let me consider option X..."
- After analyzing ALL options, explicitly predict which answer you think is correct using its NUMBER (e.g., "I predict option 3 is correct")
- Then explain your reasoning for your prediction - what frameworks and evidence led you to this conclusion?
- Finally, indicate your confidence level in your prediction (high, medium, or low) and explain why

Output your reasoning in JSON format with the following structure:
{{
  "thought_process": {{
    "option_1": "Detailed reasoning about option 1 as a possibility",
    "option_2": "Detailed reasoning about option 2 as a possibility",
    ... (all options in numerical order)
  }},
  "prediction": {{
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Reasoning for why you predict this answer is correct",
    "confidence_level": "Your confidence level (high, medium, or low)",
    "confidence_explanation": "Why you have this level of confidence in your prediction"
  }},
  "conclusion": "Final synthesized assessment"
}}

IMPORTANT: Your response must be a valid, parseable JSON object. For each option, include detailed reasoning of at least 150-200 words. You MUST make a prediction and specify ONLY the option number (e.g., '3', NOT 'Option 3').
"""


def generate_balanced_instructions(specialty: str, is_scientific: bool) -> str:
    """Generate focused reasoning instructions (focused but thorough)."""
    if is_scientific:
        return f"""
TASK:
As a {specialty}, analyze this question efficiently but thoroughly. Focus on the key scientific principles that differentiate the options:

1. Identify the core scientific concept being tested
2. Quickly eliminate obviously incorrect options with brief reasoning
3. Focus detailed analysis on the most plausible 2-3 options
4. Use your scientific knowledge to identify the decisive factors
5. Make a confident prediction based on the strongest evidence

Structure your response efficiently:
- Briefly explain the key scientific principle at stake
- Quickly dismiss clearly wrong options (1-2 sentences each)
- Provide focused analysis of viable options (3-4 sentences each)
- Make your prediction with clear scientific reasoning
- State your confidence level

Output in JSON format:
{{
  "key_principle": "The main scientific concept being tested",
  "quick_elimination": {{
    "dismissed_options": ["List of obviously wrong option numbers"],
    "reasoning": "Brief explanation why these are clearly incorrect"
  }},
  "focused_analysis": {{
    "viable_options": ["List of plausible option numbers"],
    "detailed_reasoning": "Focused analysis of the key differentiating factors"
  }},
  "prediction": {{
    "predicted_answer": "The option number (e.g., 3)",
    "prediction_reasoning": "Clear scientific reasoning for your choice",
    "confidence_level": "high/medium/low",
    "confidence_explanation": "Why you have this confidence level"
  }},
  "scientific_conclusion": "Final assessment"
}}

IMPORTANT: Be efficient - don't overthink. Focus on the key distinguishing factors. Predict the option number only (e.g., '3').
"""
    else:
        return f"""
TASK:
As a {specialty}, analyze this question efficiently but thoroughly. Focus on the key factors that differentiate the options:

1. Identify the core concept or principle being tested
2. Quickly eliminate obviously incorrect options with brief reasoning
3. Focus detailed analysis on the most plausible 2-3 options
4. Use your expertise to identify the decisive factors
5. Make a confident prediction based on the strongest evidence

Structure your response efficiently:
- Briefly explain the key principle at stake
- Quickly dismiss clearly wrong options (1-2 sentences each)
- Provide focused analysis of viable options (3-4 sentences each)
- Make your prediction with clear reasoning
- State your confidence level

Output in JSON format:
{{
  "key_principle": "The main concept being tested",
  "quick_elimination": {{
    "dismissed_options": ["List of obviously wrong option numbers"],
    "reasoning": "Brief explanation why these are clearly incorrect"
  }},
  "focused_analysis": {{
    "viable_options": ["List of plausible option numbers"],
    "detailed_reasoning": "Focused analysis of the key differentiating factors"
  }},
  "prediction": {{
    "predicted_answer": "The option number (e.g., 3)",
    "prediction_reasoning": "Clear reasoning for your choice",
    "confidence_level": "high/medium/low",
    "confidence_explanation": "Why you have this confidence level"
  }},
  "conclusion": "Final assessment"
}}

IMPORTANT: Be efficient - don't overthink. Focus on the key distinguishing factors. Predict the option number only (e.g., '3').
"""


def generate_minimal_instructions(specialty: str, is_scientific: bool) -> str:
    """Generate efficient reasoning instructions (streamlined approach)."""
    return f"""
TASK:
As a {specialty}, answer this question directly and efficiently:

1. Identify what the question is really asking
2. Apply your core knowledge to eliminate wrong answers
3. Select the best answer with concise reasoning
4. State your confidence

Be direct and focused - trust your expertise.

Output in JSON format:
{{
  "quick_analysis": "Brief explanation of what the question tests",
  "elimination": "Quick reasoning for dismissing wrong options",
  "prediction": {{
    "predicted_answer": "The option number (e.g., 3)",
    "prediction_reasoning": "Concise reasoning for your choice (2-3 sentences)",
    "confidence_level": "high/medium/low"
  }}
}}

IMPORTANT: Be concise and direct. Trust your instincts. Predict the option number only (e.g., '3').
"""


def extract_mc_options(question_text: str) -> List[str]:
    """
    Extract multiple choice options from the question text.

    Args:
        question_text: The full multiple choice question text

    Returns:
        List of extracted options without their numbers/letters
    """
    # Split the question into the actual question and the options
    parts = question_text.split("\n\n", 1)
    if len(parts) < 2:
        # Handle case where there's no clear separation
        return []

    options_text = parts[1]

    # Match different option formats like "1.", "1)", "A.", "A)", etc.
    options = re.findall(
        r"(?:^|\n)(?:\d+|\w)[.)] (.*?)(?=(?:\n(?:\d+|\w)[.)])|$)",
        options_text,
        re.DOTALL,
    )

    # Clean up the options (remove asterisks marking correct answers, etc.)
    cleaned_options = [
        re.sub(r"\s*\(\*\)\s*$", "", option.strip()) for option in options
    ]

    return cleaned_options


def extract_thought_process_from_text(text: str, option_count: int) -> Dict[str, str]:
    """
    Extract thought process for each option from raw text when JSON parsing fails.

    Args:
        text: The raw text from the model
        option_count: The number of options in the question

    Returns:
        Dictionary with thought process for each option
    """
    thought_process = {}

    # Look for patterns like "Option 1:" or "Let me consider option 1"
    option_patterns = [
        r"(?:Option|OPTION)\s+(\d+)[\s:]+(.*?)(?=(?:Option|OPTION)\s+\d+[\s:]|I\s+predict|My\s+prediction|$)",
        r"(?:Let me consider|Considering|Examining|Analyzing)\s+(?:option|Option)\s+(\d+)[\s.:]+(.*?)(?=(?:Let me consider|Considering|Examining|Analyzing)\s+(?:option|Option)|I\s+predict|My\s+prediction|$)",
        r"(?:^|\n)(\d+)[.]:?\s+(.*?)(?=(?:^|\n)\d+[.:]|I\s+predict|My\s+prediction|$)",
    ]

    for pattern in option_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for opt_num, content in matches:
            if not content.strip():  # Skip empty content
                continue
            try:
                opt_idx = int(opt_num)
                if 1 <= opt_idx <= option_count:  # Ensure it's a valid option number
                    thought_process[f"option_{opt_idx}"] = content.strip()
            except ValueError:
                continue

    # If we still don't have all options, try to split by option numbers directly
    if len(thought_process) < option_count:
        # Create a pattern to split by option numbers
        options_pattern = r"(?:Option|OPTION)\s+(\d+)|(?:Let me consider|Considering|Examining|Analyzing)\s+(?:option|Option)\s+(\d+)"

        splits = re.split(options_pattern, text)
        if len(splits) > 2:  # We have at least one match
            current_option = None
            for i, part in enumerate(splits):
                if (
                    i > 0 and i % 2 == 1 and part and part.strip().isdigit()
                ):  # This is an option number
                    current_option = int(part.strip())
                elif (
                    i > 1 and i % 2 == 0 and part and current_option is not None
                ):  # This is content
                    if f"option_{current_option}" not in thought_process:
                        thought_process[f"option_{current_option}"] = part.strip()

    return thought_process


def extract_prediction_from_text(text: str) -> Dict[str, str]:
    """
    Extract prediction information from raw text when JSON parsing fails.

    Args:
        text: The raw text from the model

    Returns:
        Dictionary with prediction information
    """
    prediction = {
        "predicted_answer": "Could not determine",
        "prediction_reasoning": "",
        "confidence_level": "unknown",
        "confidence_explanation": "",
    }

    # Try to find the predicted option number
    predict_patterns = [
        r"I\s+predict\s+(?:that\s+)?(?:option|answer)\s*(?:number|#)?\s*(\d+)",
        r"(?:My prediction|My answer|I believe|I think)\s+(?:is|would be)\s+(?:option|answer)?\s*(?:number|#)?\s*(\d+)",
        r"(?:option|answer)\s+(\d+)\s+(?:is|seems|appears to be)\s+(?:the\s+)?correct",
        r"(?:based on|after)\s+(?:my|this)\s+analysis,\s+(?:option|answer)\s+(\d+)",
        r"(?:therefore|thus|hence),\s+(?:option|answer)\s+(\d+)",
        r"(?:I would|I am going to|I will)\s+(?:choose|select|pick|go with)\s+(?:option|answer)\s+(\d+)",
        r"(?:option|answer)\s+(\d+)[\s.:,]",
        r"(?:the\s+)?correct\s+(?:option|answer)\s+(?:is|would be)\s+(\d+)",
        r"I\s+(?:choose|select|pick)\s+(?:option|answer)\s+(\d+)",
        r"(?:option|answer)\s+(\d+)\s+is\s+(?:the\s+)?(?:most\s+)?(?:correct|accurate|appropriate)",
    ]

    for pattern in predict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prediction["predicted_answer"] = f"Option {match.group(1)}"
            break

    # Word-to-number mapping as a fallback
    if prediction["predicted_answer"] == "Could not determine":
        word_to_num = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
        }

        for word, num in word_to_num.items():
            pattern = (
                r"(?:the\s+)?"
                + word
                + r"(?:\s+option|\s+answer)?\s+(?:is|seems|appears\s+to\s+be)\s+correct"
            )
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                prediction["predicted_answer"] = f"Option {num}"
                break

    # Extract reasoning
    reason_patterns = [
        r"(?:My reasoning|Reasoning|Here\'s my reasoning|Reason for prediction)(?:for this prediction)?(?:is|:)(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)",
        r"(?:I predict|I believe|I think).*?because(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)",
        r"(?:This option|Option \d+) is correct because(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)",
    ]

    for pattern in reason_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            prediction["prediction_reasoning"] = match.group(1).strip()
            break

    # Extract confidence level
    confidence_patterns = [
        r"(?:My confidence|Confidence level|I am)(?:is|:)?\s+(high|medium|low)",
        r"I have\s+(high|medium|low)(?:\s+level of)?\s+confidence",
        r"(?:high|medium|low) confidence in (?:this|my) (?:prediction|answer|conclusion)",
    ]

    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prediction["confidence_level"] = match.group(1).lower()
            break

    # Extract confidence explanation
    explanation_patterns = [
        r"(?:Confidence explanation|Reason for confidence|Why I\'m confident)(?:is|:)(.*?)(?:In conclusion|To summarize|In summary|$)",
        r"(?:I\'m|I am) (?:highly|moderately|somewhat) confident because(.*?)(?:In conclusion|To summarize|In summary|$)",
        r"My confidence is (high|medium|low) because(.*?)(?:In conclusion|To summarize|In summary|$)",
    ]

    for pattern in explanation_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and len(match.groups()) > 0:
            # Get the last group if there are multiple
            last_group = match.group(len(match.groups()))
            if last_group and last_group.strip():
                prediction["confidence_explanation"] = last_group.strip()
                break

    return prediction


def extract_conclusion_from_text(text: str) -> str:
    """
    Extract scientific conclusion from raw text when JSON parsing fails.

    Args:
        text: The raw text from the model

    Returns:
        Extracted conclusion or empty string
    """
    conclusion_patterns = [
        r"(?:Scientific conclusion|Conclusion|Final conclusion|In conclusion|To summarize)(?:is|:)(.*?)(?:$)",
        r"(?:Based on my analysis|After analyzing all options|Having considered the evidence)(.*?)(?:$)",
    ]

    for pattern in conclusion_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()

    # If no conclusion found, try to use the text after prediction as a conclusion
    prediction_match = re.search(
        r"(?:I predict|My prediction|Therefore,|Thus,).*?(\d+).*?(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if prediction_match:
        prediction_pos = prediction_match.end()
        if (
            prediction_pos < len(text) - 100
        ):  # Ensure there's enough text after prediction
            conclusion = text[prediction_pos:].strip()
            if conclusion:
                return conclusion[:500] + "..." if len(conclusion) > 500 else conclusion

    return ""


def detect_choice_identifier_type(question_text):
    """
    Detect whether a question uses letter (A, B, C) or number (1, 2, 3) identifiers.

    Args:
        question_text (str): The question text to analyze

    Returns:
        str: 'letter' if using A,B,C format, 'number' if using 1,2,3 format, 'letter' as default
    """
    # Look for standard multiple choice patterns like "A)" or "1."
    has_letter_option = bool(re.search(r"(?:^|\n)\s*([A-E])[.):,]\s", question_text))
    has_number_option = bool(re.search(r"(?:^|\n)\s*([1-5])[.):,]\s", question_text))

    # Default to the input format: if numbers are found, use numbers; if letters are found, use letters
    if has_number_option:
        return "number"
    elif has_letter_option:
        return "letter"
    else:
        # If neither is clearly detected, default to 'letter'
        return "letter"


def extract_choice_identifier(answer_text):
    """
    Extract the choice identifier (A, B, C, D, E, etc. or 1, 2, 3, 4, 5, etc.) from an answer text.
    Uses multiple regex patterns to handle different formats.
    Copied from argonium_score_parallel_v9.py for exact compatibility.

    Returns a tuple of (identifier_type, identifier) where identifier_type is either 'letter' or 'number'
    and identifier is the letter (A-Z) or number (1-9) that was identified.
    """
    # First, check if this is a multiple-choice format text
    if not answer_text:
        return (None, None)

    # More strict patterns for multiple choice options (look for standard patterns)
    # This looks for options clearly marked, like "A)" or "A." at beginning of lines or text
    # Extended to handle more options (up to 9 for numbers, up to J for letters)
    letter_mc_pattern = r"(?:^|\n)\s*([A-J])[.):]\s"
    number_mc_pattern = r"(?:^|\n)\s*([1-9])[.):]\s"

    # Also look for explicit mentions of options
    letter_explicit = r"(?:option|answer|choice)\s+([A-J])\b"
    number_explicit = r"(?:option|answer|choice)\s+([1-9])\b"

    # Look for standard statements about correct answers
    letter_statement = r"(?:the\s+(?:correct\s+)?answer\s+is\s+([A-J]))|(?:\b([A-J])\s+is\s+(?:the\s+)?correct)"
    number_statement = r"(?:the\s+(?:correct\s+)?answer\s+is\s+([1-9]))|(?:\b([1-9])\s+is\s+(?:the\s+)?correct)"

    # Try the strict patterns first
    for pattern in [letter_mc_pattern, letter_explicit, letter_statement]:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group:
                    return ("letter", group.upper())

    for pattern in [number_mc_pattern, number_explicit, number_statement]:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group:
                    return ("number", group)

    # If strict patterns don't match, look at the beginning of the response
    # This pattern searches for an answer that starts with a letter/number indicator
    first_line = answer_text.split("\n")[0].strip()

    # Look for responses that begin with "A:", "A.", "A)", or just "A"
    letter_start = re.match(r"^([A-J])(?:[.):,]|\s|$)", first_line, re.IGNORECASE)
    if letter_start:
        return ("letter", letter_start.group(1).upper())

    # Look for responses that begin with "1:", "1.", "1)", or just "1"
    number_start = re.match(r"^([1-9])(?:[.):,]|\s|$)", first_line, re.IGNORECASE)
    if number_start:
        return ("number", number_start.group(1))

    # For short answers, scan for a valid option letter (A-J) or number (1-9)
    if len(answer_text) < 100:  # Only for short answers to avoid false matches
        # Look for any standalone A, B, C, D, E, F, G, H, I, J in the text
        standalone_letter = re.search(r"\b([A-J])\b", answer_text, re.IGNORECASE)
        if standalone_letter:
            return ("letter", standalone_letter.group(1).upper())

        # Look for any standalone 1, 2, 3, 4, 5, 6, 7, 8, 9 in the text
        standalone_number = re.search(r"\b([1-9])\b", answer_text)
        if standalone_number:
            return ("number", standalone_number.group(1))

    # If no pattern matches, return None for both
    return (None, None)


def normalize_choice_identifier(identifier, identifier_type, target_type):
    """
    Normalize a choice identifier to the target type.
    Copied from argonium_score_parallel_v9.py for exact compatibility.

    Args:
        identifier (str): The identifier (A, B, C... or 1, 2, 3...)
        identifier_type (str): 'letter' or 'number'
        target_type (str): 'letter' or 'number'

    Returns:
        str: The normalized identifier, or None if conversion fails
    """
    if not identifier or not identifier_type or not target_type:
        return None

    if identifier_type == target_type:
        return identifier

    try:
        if identifier_type == "letter" and target_type == "number":
            # Convert A->1, B->2, etc.
            return str(ord(identifier.upper()) - 64)
        elif identifier_type == "number" and target_type == "letter":
            # Convert 1->A, 2->B, etc.
            num = int(identifier)
            if 1 <= num <= 26:
                return chr(64 + num)
    except (ValueError, TypeError):
        pass

    return None


def extract_option_content(question, choice_identifier, identifier_type):
    """
    Extract the content of a specific option from a multiple-choice question.
    Copied from argonium_score_parallel_v9.py for exact compatibility.

    Args:
        question (str): The full question text
        choice_identifier (str): The choice identifier (A, B, 1, 2, etc.)
        identifier_type (str): 'letter' or 'number'

    Returns:
        str: The content of the option, or None if not found
    """
    if not question or not choice_identifier:
        return None

    # Escape the identifier for regex
    escaped_id = re.escape(choice_identifier)

    # Pattern to match the option and capture its content
    # Looks for the identifier followed by punctuation, then captures until next option or end
    pattern = r"(?:^|\n)\s*" + escaped_id + r"[.):,]\s+(.+?)(?=\n\s*[A-E1-5][.):,]|$)"

    match = re.search(pattern, question, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # Clean up the content (remove extra whitespace, newlines)
        content = re.sub(r"\s+", " ", content)
        return content

    return None


def check_content_consistency(model_answer, correct_option_content):
    """
    Check if the model's answer content is consistent with the correct option content.
    Copied from argonium_score_parallel_v9.py for exact compatibility.

    Args:
        model_answer (str): The model's full answer
        correct_option_content (str): The content of the correct option

    Returns:
        bool: True if the content seems consistent, False otherwise
    """
    if not model_answer or not correct_option_content:
        return False

    # Clean and normalize both texts for comparison
    model_clean = re.sub(r"\s+", " ", model_answer.lower().strip())
    option_clean = re.sub(r"\s+", " ", correct_option_content.lower().strip())

    # If the correct option content is substantial (>15 chars), look for significant overlap
    if len(option_clean) > 15:
        # Look for at least 60% of the option content words in the model answer
        option_words = set(re.findall(r"\w+", option_clean))
        model_words = set(re.findall(r"\w+", model_clean))

        if len(option_words) > 0:
            overlap = len(option_words.intersection(model_words))
            overlap_ratio = overlap / len(option_words)
            return overlap_ratio >= 0.6

    # For shorter content, look for exact substring match
    return option_clean in model_clean


def generate_argonium_style_prediction(
    full_question_text: str, options: List[str], client: OpenAI, model_name: str, specialty: str
) -> Dict[str, Any]:
    """
    Generate a prediction using Argonium's simplified, direct approach.
    Uses the exact same input format as argonium_score_parallel.

    Args:
        full_question_text: The complete question text (may include embedded options)
        options: List of answer options (used for fallback reconstruction only)
        client: OpenAI client
        model_name: Model name to use
        specialty: Expert specialty

    Returns:
        Dictionary with argonium prediction results
    """
    # Use the question text exactly as provided (like argonium_score_parallel does)
    # This ensures identical input format to argonium_score_parallel
    full_question = full_question_text
    
    # Simple check for embedded options without regex - look for numbered patterns
    has_embedded_options = any(
        line.strip().startswith(f"{i}.") or line.strip().startswith(f"{i})")
        for line in full_question_text.split('\n')
        for i in range(1, 10)
    )
    
    if not has_embedded_options and options:
        # Only reconstruct if options are separate (like make_v21 format)
        full_question += "\n\n" + "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        log_message("Reconstructed question with separate options", log_level="DEBUG")
    else:
        # Question already has embedded options (like HR dataset) - use as-is
        log_message("Using question with embedded options as-is", log_level="DEBUG")
    
    # Detect choice identifier type (letter/number) using unified logic - same as argonium_score_parallel
    id_type_in_question = detect_choice_identifier_type(full_question)

    # Determine label format for prompts - exact match to argonium_score_parallel
    if id_type_in_question == "number":
        label_format = "number (1, 2, 3, etc.)"
    else:
        label_format = "letter (A, B, C, etc.)"

    # System prompt designed for multiple-choice questions - exact match to argonium_score_parallel
    system_message = (
        "You are an expert at multiple-choice questions. "
        "Think through the question step by step, but provide only a concise final answer. "
        "Your response should contain ONLY:\n"
        f"1. The correct {label_format}\n"
        "2. A brief explanation (2-3 sentences) of why this choice is correct.\n\n"
        "Do not include the question restatement or list of alternatives in your response."
    )

    # User message - exact match to argonium_score_parallel 
    user_message = (
        f"Please answer this multiple-choice question. Think through it carefully:\n"
        f"- First, restate the question to yourself\n"
        f"- Then, consider all the alternative answers provided\n"
        f"- Finally, provide your response with ONLY the correct {label_format} "
        f"followed by 2-3 sentences explaining why this choice is correct.\n\n"
        f"Question:\n{full_question}"
    )

    try:
        # Check if we need to skip temperature (for reasoning models like o3 and o4mini) - same as argonium
        skip_temperature = any(
            name in model_name.lower() for name in ["o3", "o4-mini", "o4mini"]
        )

        # Prepare parameters like argonium does
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        }

        # Add temperature only for models that support it
        if not skip_temperature:
            params["temperature"] = 0.0

        response = client.chat.completions.create(**params)

        response_text = response.choices[0].message.content.strip()

        # Extract the predicted answer for display purposes
        predicted_answer = "Unknown"
        try:
            # Try to extract the answer using the same logic as argonium scoring
            identifier_type, identifier = extract_choice_identifier(response_text)
            if identifier:
                predicted_answer = identifier
        except Exception:
            # Fallback: look for simple patterns like "1", "2", etc. or "A", "B", etc.
            import re
            # Look for standalone numbers or letters at the beginning of lines or after colons
            number_match = re.search(r'(?:^|\n|:\s*)([1-9])\b', response_text)
            letter_match = re.search(r'(?:^|\n|:\s*)([A-J])\b', response_text, re.IGNORECASE)
            
            if number_match:
                predicted_answer = number_match.group(1)
            elif letter_match:
                predicted_answer = letter_match.group(1).upper()

        # Return response with extracted answer
        return {
            "raw_response": response_text,
            "model_answer": response_text,  # For compatibility with argonium grader
            "predicted_answer": predicted_answer,  # For display purposes
        }

    except Exception as e:
        log_message(
            f"Error generating Argonium-style prediction: {e}", log_level="ERROR"
        )
        return {
            "raw_response": f"Error: {str(e)}",
            "model_answer": f"Error: {str(e)}",
        }


def generate_prediction_comparison(
    detailed_reasoning: Dict[str, Any],
    argonium_prediction: Dict[str, Any],
    question: str,
    options: List[str],
    client: OpenAI,
    model_name: str,
    specialty: str,
) -> str:
    """
    Generate an LLM-based comparison analysis focusing on answer comparison between the two prediction methods.

    Args:
        detailed_reasoning: The detailed reasoning trace results
        argonium_prediction: The Argonium-style prediction results
        question: The question text
        options: List of answer options
        client: OpenAI client
        model_name: Model name to use
        specialty: Expert specialty

    Returns:
        String containing the LLM-based answer comparison analysis
    """
    # Extract predictions from both approaches
    detailed_pred = detailed_reasoning.get("prediction", {}).get(
        "predicted_answer", "Unknown"
    )
    detailed_reasoning_text = detailed_reasoning.get("prediction", {}).get(
        "prediction_reasoning", "No reasoning provided"
    )

    argonium_pred = argonium_prediction.get("predicted_answer", "Unknown")
    argonium_response = argonium_prediction.get("raw_response", "No response")

    # Check if predictions match
    predictions_match = False
    if "predicted_num" in locals() or argonium_prediction.get("predicted_num"):
        # Try to extract numbers from both predictions for comparison
        detailed_num_match = re.search(r"(\d+)", str(detailed_pred))
        argonium_num = argonium_prediction.get("predicted_num")

        if detailed_num_match and argonium_num:
            predictions_match = int(detailed_num_match.group(1)) == argonium_num

    # Create LLM-based answer comparison prompt
    comparison_prompt = f"""You are an expert {specialty} comparing two different answers to the same multiple-choice question.

QUESTION: {question}

OPTIONS:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

METHOD 1 - DETAILED REASONING:
Answer: {detailed_pred}
Reasoning: {detailed_reasoning_text}

METHOD 2 - DIRECT ANSWER (Argonium-style):
Answer: {argonium_pred}
Response: {argonium_response}

TASK: 
If both methods give the SAME answer, simply state: "Both methods agree on the same answer: [answer]. The reasoning approaches differ but lead to the same conclusion."

If the methods give DIFFERENT answers, write a clear paragraph that:
1. States which answer you believe is correct as a {specialty} expert
2. Explains specifically HOW the different reasoning approaches led to different answers
3. Argues convincingly for why one answer is better than the other based on the scientific reasoning

Do NOT use numbered points or bullet lists. Write as a single, coherent paragraph that makes a clear argument for the superior answer when they differ.

Keep your response to 200-300 words maximum."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {specialty} who provides clear, direct comparisons of scientific answers. When answers differ, you argue convincingly for the correct one based on scientific reasoning. You write in coherent paragraphs, not lists.",
                },
                {"role": "user", "content": comparison_prompt},
            ],
            temperature=0.3,  # Moderate temperature for natural paragraph writing
            max_tokens=600,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        log_message(f"Error generating prediction comparison: {e}", log_level="ERROR")
        return f"Error generating comparison analysis: {str(e)}"


def grade_answer(
    predicted_answer: str,
    correct_answer: str,
    question_text: str,
    options: List[str],
    grading_client: OpenAI,
    grading_model_name: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Use a grading model to determine if the predicted answer is correct.

    Args:
        predicted_answer: The answer predicted by the reasoning model
        correct_answer: The correct answer from the input file
        question_text: The original question text
        options: List of answer options
        grading_client: OpenAI client for grading
        grading_model_name: Model name to use for grading

    Returns:
        Dictionary with grading results including is_correct, confidence, and reasoning
    """
    try:
        # Create a comprehensive grading prompt
        grading_prompt = f"""You are an expert grader evaluating whether a predicted answer matches the correct answer for a multiple-choice question.

QUESTION:
{question_text}

OPTIONS:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

PREDICTED ANSWER: {predicted_answer}
CORRECT ANSWER: {correct_answer}

TASK:
Determine if the predicted answer is correct. The predicted answer might be in various formats (e.g., "Option 3", "3", "third option", etc.) and may contain additional explanation text.

Your evaluation should consider:
1. Whether the predicted answer refers to the same option as the correct answer
2. Whether the core meaning matches, regardless of formatting differences
3. Whether the predicted answer is substantially correct even if not perfectly formatted

Respond with a JSON object containing:
{{
  "is_correct": true/false,
  "confidence": "high/medium/low",
  "reasoning": "Brief explanation of your grading decision",
  "extracted_option_number": "The option number you identified from the predicted answer (1-based index)",
  "correct_option_number": "The option number from the correct answer (1-based index)"
}}

IMPORTANT: Be generous in your interpretation - if the predicted answer clearly indicates the same option as the correct answer, mark it as correct even if the formatting is different."""

        # Make the grading request
        response = grading_client.chat.completions.create(
            model=grading_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert grader who evaluates whether predicted answers match correct answers for multiple-choice questions. You are thorough but fair in your evaluation.",
                },
                {"role": "user", "content": grading_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent grading
            max_tokens=500,
        )

        grading_response = response.choices[0].message.content.strip()

        # Show verbose output if requested
        if verbose:
            print("\n" + "=" * 80)
            print("GRADING LLM INPUT:")
            print("=" * 80)
            print(grading_prompt)
            print("\n" + "=" * 80)
            print("GRADING LLM OUTPUT:")
            print("=" * 80)
            print(grading_response)
            print("=" * 80 + "\n")

        # Try to parse the JSON response
        try:
            grading_result = json.loads(grading_response)

            # Ensure required fields exist
            if "is_correct" not in grading_result:
                grading_result["is_correct"] = False
            if "confidence" not in grading_result:
                grading_result["confidence"] = "low"
            if "reasoning" not in grading_result:
                grading_result["reasoning"] = "Unable to determine reasoning"

            # Add metadata
            grading_result["grading_model"] = grading_model_name
            grading_result["grading_successful"] = True
            grading_result["grading_input"] = grading_prompt
            grading_result["grading_output"] = grading_response

            return grading_result

        except json.JSONDecodeError:
            # Fallback: try to extract basic correctness from the response
            response_lower = grading_response.lower()
            is_correct = (
                "true" in response_lower and "is_correct" in response_lower
            ) or "correct" in response_lower

            return {
                "is_correct": is_correct,
                "confidence": "low",
                "reasoning": "Failed to parse JSON response, used fallback extraction",
                "extracted_option_number": "unknown",
                "correct_option_number": "unknown",
                "grading_model": grading_model_name,
                "grading_successful": False,
                "raw_response": grading_response,
                "grading_input": grading_prompt,
                "grading_output": grading_response,
            }

    except Exception as e:
        log_message(f"Error in grading model: {e}", log_level="ERROR")
        return {
            "is_correct": False,
            "confidence": "low",
            "reasoning": f"Error during grading: {str(e)}",
            "extracted_option_number": "error",
            "correct_option_number": "error",
            "grading_model": grading_model_name,
            "grading_successful": False,
            "error": str(e),
            "grading_input": "",
            "grading_output": "",
        }


def load_argonium_results(argonium_file: str) -> Dict[str, Any]:
    """
    Load and parse argonium results file from argonium_score_parallel.
    
    Args:
        argonium_file: Path to the argonium results JSON file
        
    Returns:
        Dictionary with parsed argonium results
    """
    try:
        if not os.path.exists(argonium_file):
            log_message(f"ERROR: Argonium file not found: {argonium_file}", log_level="ERROR")
            return None
            
        with open(argonium_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        log_message(f"Successfully loaded argonium results from {argonium_file}")
        
        # Handle both list format and dict format
        if isinstance(data, list):
            # Direct list of results
            log_message(f"Argonium file contains {len(data)} questions (list format)")
            return {"results": data, "metadata": {}}
        elif isinstance(data, dict):
            # Dictionary with results and metadata
            results = data.get('results', [])
            log_message(f"Argonium file contains {len(results)} questions (dict format)")
            if results:
                sample_result = results[0] if results else {}
                log_message(f"Sample result keys: {list(sample_result.keys())}")
            if 'metadata' in data and 'overall_accuracy' in data['metadata']:
                accuracy = data['metadata']['overall_accuracy']
                if isinstance(accuracy, (int, float)):
                    log_message(f"Argonium overall accuracy: {accuracy:.1%}")
            return data
        else:
            log_message(f"Unknown argonium file format: {type(data)}", log_level="ERROR")
            return None
        
    except Exception as e:
        log_message(f"Error loading argonium results file {argonium_file}: {e}", log_level="ERROR")
        return None


def verify_question_match(reasoning_question: str, argonium_question: str, 
                         grading_client: OpenAI, grading_model_name: str) -> Dict[str, Any]:
    """
    Use grading model to verify that two questions are the same.
    
    Args:
        reasoning_question: Question from reasoning traces
        argonium_question: Question from argonium results file
        grading_client: OpenAI client for grading
        grading_model_name: Model name for grading
        
    Returns:
        Dictionary with verification results
    """
    try:
        verification_prompt = f"""You are comparing two questions to determine if they are identical or equivalent.

QUESTION A:
{reasoning_question}

QUESTION B:
{argonium_question}

TASK:
Determine if these questions are the same. They should be considered a match if:
1. The text is identical or nearly identical (ignoring minor formatting differences)
2. The core question being asked is the same
3. All answer options are present and in the same order

Respond with a JSON object:
{{
  "is_match": true/false,
  "confidence": "high/medium/low",
  "reasoning": "Brief explanation of your determination"
}}"""

        response = grading_client.chat.completions.create(
            model=grading_model_name,
            messages=[
                {"role": "system", "content": "You are an expert at comparing questions for identity. Be thorough but precise."},
                {"role": "user", "content": verification_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        output = response.choices[0].message.content.strip()
        
        try:
            result = json.loads(output)
            result["verification_successful"] = True
            return result
        except json.JSONDecodeError:
            return {
                "is_match": False,
                "confidence": "low", 
                "reasoning": "Failed to parse verification response",
                "verification_successful": False,
                "raw_output": output
            }
            
    except Exception as e:
        return {
            "is_match": False,
            "confidence": "low",
            "reasoning": f"Error during verification: {str(e)}",
            "verification_successful": False,
            "error": str(e)
        }


def analyze_individual_options(
    question_data: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    specialty: str = "expert"
) -> Dict[str, Any]:
    """
    First stage of Method 1: Generate individual analysis for each answer option.
    
    Args:
        question_data: The question data containing question and options
        client: OpenAI client for the analysis model
        model_name: Name of the model being used
        specialty: The expert specialty persona
        
    Returns:
        Dictionary containing individual analyses for each option
    """
    try:
        question = question_data["question"]
        options = question_data["options"]
        
        # Build the prompt for individual option analysis
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        
        system_message = f"""You are an expert {specialty}. Your task is to analyze each possible answer option individually and thoroughly for the given question.

For each option, you should:
1. Evaluate how well it addresses the question
2. Consider any scientific/technical accuracy 
3. Identify strengths and potential weaknesses
4. Note any key considerations or caveats

Be thorough but concise in your analysis. Do not make a final decision - just analyze each option objectively."""

        user_message = f"""Question: {question}

Options:
{options_text}

Please provide a detailed analysis of each option. Structure your response as JSON with the following format:
{{
    "option_1": {{
        "analysis": "Your detailed analysis of option 1",
        "strengths": ["List of strengths"],
        "weaknesses": ["List of weaknesses"],
        "technical_accuracy": "Assessment of technical accuracy",
        "relevance_to_question": "How well it addresses the question"
    }},
    "option_2": {{
        "analysis": "Your detailed analysis of option 2",
        "strengths": ["List of strengths"], 
        "weaknesses": ["List of weaknesses"],
        "technical_accuracy": "Assessment of technical accuracy",
        "relevance_to_question": "How well it addresses the question"
    }},
    ... (continue for all options)
}}"""

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse the JSON response
        try:
            analyses = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a fallback structure
            analyses = {
                f"option_{i+1}": {
                    "analysis": f"Analysis for option {i+1} - parsing failed",
                    "strengths": [],
                    "weaknesses": [],
                    "technical_accuracy": "Could not parse",
                    "relevance_to_question": "Could not parse",
                    "raw_text": response_text
                }
                for i in range(len(options))
            }
        
        return {
            "individual_analyses": analyses,
            "model_used": model_name,
            "analysis_successful": True,
            "raw_response": response_text
        }
        
    except Exception as e:
        return {
            "individual_analyses": {},
            "model_used": model_name,
            "analysis_successful": False,
            "error": str(e),
            "raw_response": ""
        }


def synthesize_final_reasoning(
    question_data: Dict[str, Any],
    individual_analyses: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    specialty: str = "expert"
) -> Dict[str, Any]:
    """
    Second stage of Method 1: Synthesize individual analyses into final reasoning and decision.
    
    Args:
        question_data: The question data containing question and options
        individual_analyses: Results from the first stage analysis
        client: OpenAI client for the synthesis model
        model_name: Name of the model being used
        specialty: The expert specialty persona
        
    Returns:
        Dictionary containing synthesized reasoning and final prediction
    """
    try:
        question = question_data["question"]
        options = question_data["options"]
        analyses = individual_analyses.get("individual_analyses", {})
        
        # Build the prompt for synthesis
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        
        # Format the individual analyses for the synthesis prompt
        analyses_text = ""
        for option_key, analysis_data in analyses.items():
            if isinstance(analysis_data, dict):
                analyses_text += f"\n{option_key.upper()}:\n"
                analyses_text += f"Analysis: {analysis_data.get('analysis', 'Not available')}\n"
                analyses_text += f"Strengths: {', '.join(analysis_data.get('strengths', []))}\n"
                analyses_text += f"Weaknesses: {', '.join(analysis_data.get('weaknesses', []))}\n"
                analyses_text += f"Technical Accuracy: {analysis_data.get('technical_accuracy', 'Not assessed')}\n"
                analyses_text += f"Relevance: {analysis_data.get('relevance_to_question', 'Not assessed')}\n"
        
        system_message = f"""You are an expert {specialty}. You have been provided with detailed individual analyses of each answer option for a question. 

Your task is to:
1. Review all the individual analyses
2. Synthesize the information to determine the best answer
3. Provide detailed reasoning for your final decision
4. Make a definitive prediction

Consider all aspects: technical accuracy, relevance to the question, completeness, and any other relevant factors."""

        user_message = f"""Question: {question}

Options:
{options_text}

Individual Analyses from First Stage:
{analyses_text}

Based on these individual analyses, please provide your synthesis and final reasoning. Structure your response as JSON:
{{
    "question_restatement": "Restate the question in your own words",
    "synthesis_reasoning": "Your detailed reasoning that synthesizes the individual analyses",
    "option_comparison": "Compare the options based on the analyses",
    "decision_rationale": "Explain why you chose your final answer",
    "predicted_answer": "The option number (e.g., '3')",
    "prediction_reasoning": "Brief explanation of your prediction",
    "confidence_level": "high/medium/low",
    "confidence_explanation": "Explanation of your confidence level",
    "scientific_conclusion": "Your final scientific/expert conclusion"
}}"""

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse the JSON response
        try:
            synthesis = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a fallback structure
            synthesis = {
                "question_restatement": question,
                "synthesis_reasoning": "Synthesis parsing failed",
                "option_comparison": "Could not parse comparison",
                "decision_rationale": "Could not parse rationale", 
                "predicted_answer": "1",
                "prediction_reasoning": "Parsing failed - defaulting to option 1",
                "confidence_level": "low",
                "confidence_explanation": "Confidence low due to parsing failure",
                "scientific_conclusion": "Could not generate conclusion due to parsing error",
                "raw_text": response_text
            }
        
        return {
            "synthesis": synthesis,
            "model_used": model_name,
            "synthesis_successful": True,
            "raw_response": response_text
        }
        
    except Exception as e:
        return {
            "synthesis": {},
            "model_used": model_name,
            "synthesis_successful": False,
            "error": str(e),
            "raw_response": ""
        }


def comprehensive_disagreement_analysis(
    question_data: Dict[str, Any],
    method1_result: Dict[str, Any],
    method2_result: Dict[str, Any] = None,
    client: OpenAI = None,
    model_name: str = None,
    specialty: str = "expert",
) -> Dict[str, Any]:
    """
    Enhanced fault-oriented disagreement analysis with root cause diagnosis.
    
    Structure:
    1. Answer comparison (correct vs predicted)
    2. Diagnostic analysis with fault assignment
    3. External knowledge requirements assessment
    4. Actionable recommendations for fixes
    
    Args:
        question_data: The question data including context
        method1_result: Method 1 results with reasoning
        method2_result: Method 2 results (optional)
        client: OpenAI client for analysis
        model_name: Model name for analysis
        specialty: Expert specialty
        
    Returns:
        Dictionary with enhanced fault-oriented analysis
    """
    if not client or not model_name:
        return {"error": "No analysis client provided"}
    
    # Perform enhanced fault-oriented analysis
    fault_analysis = perform_fault_oriented_analysis(
        question_data, method1_result, method2_result, client, model_name, specialty
    )
    
    return fault_analysis


def perform_fault_oriented_analysis(
    question_data: Dict[str, Any],
    method1_result: Dict[str, Any],
    method2_result: Dict[str, Any] = None,
    client: OpenAI = None,
    model_name: str = None,
    specialty: str = "expert"
) -> Dict[str, Any]:
    """Perform systematic fault-oriented disagreement analysis."""
    
    question = question_data.get("question", "")
    context = question_data.get("context", "")
    options = method1_result.get("options", [])
    correct_answer = method1_result.get("correct_answer", "")
    correct_index = method1_result.get("correct_answer_index", -1)
    
    # Extract predictions
    reasoning = method1_result.get("reasoning", {})
    method1_prediction = reasoning.get("prediction", {}).get("predicted_answer", "Unknown")
    method1_reasoning_text = reasoning.get("prediction", {}).get("prediction_reasoning", "")
    
    method2_prediction = method2_result.get("predicted_answer", "Unknown") if method2_result else None
    method2_response = method2_result.get("raw_response", "") if method2_result else ""
    
    system_message = f"""You are an expert {specialty} performing systematic fault analysis of reasoning disagreements.

Your task is to:
1. Compare answers and identify discrepancies
2. Perform diagnostic analysis to determine root cause
3. Assign fault to specific component (MCQA creation, scoring, reasoning logic, assumptions, etc.)
4. Identify external knowledge requirements not in source text
5. Provide actionable recommendations for fixes

Be systematic, precise, and actionable in your analysis."""

    user_message = f"""SOURCE TEXT:
{context}

QUESTION: {question}

ANSWER OPTIONS:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

CORRECT ANSWER: {correct_answer} (Option {correct_index + 1 if correct_index >= 0 else 'Unknown'})

METHOD 1 PREDICTION: {method1_prediction}
METHOD 1 REASONING: {method1_reasoning_text}

{f"METHOD 2 PREDICTION: {method2_prediction}" if method2_prediction else ""}
{f"METHOD 2 RESPONSE: {method2_response}" if method2_response else ""}

Please perform systematic fault analysis and provide your assessment as JSON:

{{
  "answer_comparison": {{
    "correct_answer": "The designated correct answer",
    "method1_prediction": "What Method 1 predicted",
    "method2_prediction": "What Method 2 predicted (if available)",
    "has_disagreement": true/false,
    "disagreement_type": "method1_vs_correct/method2_vs_correct/method1_vs_method2/multiple"
  }},
  
  "diagnostic_analysis": {{
    "primary_fault_category": "mcqa_creation/scoring_error/reasoning_logic/invalid_assumptions/knowledge_gap/text_ambiguity/option_design",
    "fault_confidence": "high/medium/low",
    "specific_fault_description": "Detailed description of what went wrong",
    "fault_evidence": ["Evidence supporting this fault assignment"],
    "secondary_contributing_factors": ["Other factors that may have contributed"]
  }},
  
  "external_knowledge_analysis": {{
    "requires_external_knowledge": true/false,
    "knowledge_type": "domain_expertise/factual_knowledge/procedural_knowledge/definitional_knowledge/none",
    "missing_knowledge_summary": "What knowledge was needed but not in text",
    "likely_knowledge_source": "parametric_pretraining/domain_training/common_knowledge/expert_training/unclear",
    "knowledge_specificity": "highly_specialized/moderately_specialized/general_knowledge"
  }},
  
  "reasoning_quality_assessment": {{
    "method1_reasoning_validity": "sound/flawed/partially_sound",
    "method1_logic_issues": ["Specific logical problems identified"],
    "method2_reasoning_validity": "sound/flawed/partially_sound/not_applicable",
    "method2_logic_issues": ["Specific logical problems identified"],
    "better_reasoning_approach": "method1/method2/neither/unclear"
  }},
  
  "root_cause_analysis": {{
    "primary_root_cause": "Single most important cause of the problem",
    "contributing_factors": ["Other factors that contributed to the issue"],
    "problem_severity": "critical/major/moderate/minor",
    "problem_scope": "systematic/question_specific/context_specific"
  }},
  
  "actionable_recommendations": {{
    "immediate_fix": "Most important immediate action needed",
    "prevention_strategy": "How to prevent similar issues",
    "mcqa_improvement": "Specific improvements to question/answer design",
    "text_requirements": "What the source text should contain for this question",
    "scoring_adjustment": "Any needed changes to answer key or scoring"
  }},
  
  "confidence_assessment": {{
    "overall_analysis_confidence": "high/medium/low",
    "most_certain_finding": "The finding you're most confident about",
    "least_certain_finding": "The finding you're least confident about",
    "additional_analysis_needed": ["What additional analysis would be helpful"]
  }}
}}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            analysis = json.loads(response_text)
            return {
                "analysis_successful": True,
                "fault_analysis": analysis,
                "raw_response": response_text
            }
        except json.JSONDecodeError:
            return {
                "analysis_successful": False,
                "error": "Failed to parse fault analysis JSON",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {
            "analysis_successful": False,
            "error": str(e),
            "raw_response": ""
        }


def analyze_method1_internal_consistency(
    question_data: Dict[str, Any],
    method1_result: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    specialty: str
) -> Dict[str, Any]:
    """Analyze internal consistency within Method 1 reasoning."""
    
    question = question_data.get("question", "")
    options = method1_result.get("options", [])
    reasoning = method1_result.get("reasoning", {})
    
    # Get thought process and final prediction
    thought_process = reasoning.get("thought_process", {})
    prediction = reasoning.get("prediction", {})
    final_answer = prediction.get("predicted_answer", "Unknown")
    final_reasoning = prediction.get("prediction_reasoning", "")
    scientific_conclusion = reasoning.get("scientific_conclusion", "")
    
    # Format thought process for analysis
    thought_analysis = ""
    for i, option in enumerate(options):
        opt_key = f"option_{i+1}"
        if opt_key in thought_process:
            thought_analysis += f"Option {i+1}: {option}\nAnalysis: {thought_process[opt_key]}\n\n"
    
    system_message = f"""You are an expert {specialty} analyzing the internal consistency of reasoning.
    
Your task is to identify inconsistencies, contradictions, or logical gaps within a single reasoning process."""

    user_message = f"""Question: {question}

Options:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

DETAILED THOUGHT PROCESS FOR EACH OPTION:
{thought_analysis}

FINAL PREDICTION: {final_answer}
FINAL REASONING: {final_reasoning}
SCIENTIFIC CONCLUSION: {scientific_conclusion}

Please analyze this reasoning for internal consistency. Look for:
1. Contradictions between individual option analyses and final decision
2. Logical inconsistencies in the reasoning chain
3. Evidence that supports a different conclusion than what was chosen
4. Quality of evidence integration across options
5. Whether the final reasoning adequately addresses the individual option analyses

Provide your analysis as JSON:
{{
  "has_internal_inconsistency": true/false,
  "inconsistency_type": "contradiction/logical_gap/evidence_mismatch/integration_failure/none",
  "specific_issues": ["List of specific problems found"],
  "evidence_assessment": "How well the evidence supports the final conclusion",
  "reasoning_quality": "high/medium/low",
  "recommended_answer": "Which answer the evidence actually supports (if different)",
  "confidence_in_analysis": "high/medium/low",
  "detailed_explanation": "Comprehensive explanation of the analysis"
}}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            analysis = json.loads(response_text)
            return {
                "analysis_successful": True,
                "analysis": analysis,
                "raw_response": response_text
            }
        except json.JSONDecodeError:
            return {
                "analysis_successful": False,
                "error": "Failed to parse analysis JSON",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {
            "analysis_successful": False,
            "error": str(e),
            "raw_response": ""
        }


def analyze_method_disagreement(
    question_data: Dict[str, Any],
    method1_result: Dict[str, Any],
    method2_result: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    specialty: str
) -> Dict[str, Any]:
    """Analyze disagreement between Method 1 and Method 2."""
    
    question = question_data.get("question", "")
    options = method1_result.get("options", [])
    
    # Method 1 data
    reasoning = method1_result.get("reasoning", {})
    method1_answer = reasoning.get("prediction", {}).get("predicted_answer", "Unknown")
    method1_reasoning = reasoning.get("prediction", {}).get("prediction_reasoning", "")
    method1_conclusion = reasoning.get("scientific_conclusion", "")
    
    # Method 2 data
    method2_answer = method2_result.get("predicted_answer", "Unknown")
    method2_response = method2_result.get("raw_response", "")
    
    system_message = f"""You are an expert {specialty} analyzing disagreements between two different reasoning approaches.

Method 1 uses detailed two-stage analysis of each option.
Method 2 uses direct, intuitive reasoning.

Your task is to determine which method's reasoning is more sound and why."""

    user_message = f"""Question: {question}

Options:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

METHOD 1 (Detailed Analysis):
Answer: {method1_answer}
Reasoning: {method1_reasoning}
Conclusion: {method1_conclusion}

METHOD 2 (Direct Reasoning):  
Answer: {method2_answer}
Response: {method2_response}

These methods disagree on the answer. Please analyze:
1. Which reasoning approach is more sound and why
2. What specific errors or strengths each method demonstrates
3. Whether the disagreement reveals a flaw in one approach
4. Which answer is more likely to be correct based on the evidence
5. What this disagreement tells us about the question or options

Provide your analysis as JSON:
{{
  "disagreement_severity": "high/medium/low",
  "more_reliable_method": "method1/method2/unclear", 
  "method1_strengths": ["List strengths of detailed analysis"],
  "method1_weaknesses": ["List weaknesses of detailed analysis"],
  "method2_strengths": ["List strengths of direct reasoning"],
  "method2_weaknesses": ["List weaknesses of direct reasoning"],
  "likely_correct_answer": "Which answer is probably correct",
  "reasoning_quality_comparison": "Detailed comparison of reasoning quality",
  "disagreement_insights": "What this disagreement reveals about the question",
  "confidence_in_assessment": "high/medium/low",
  "detailed_explanation": "Comprehensive analysis of the disagreement"
}}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=2500
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            analysis = json.loads(response_text)
            return {
                "analysis_successful": True,
                "analysis": analysis,
                "raw_response": response_text
            }
        except json.JSONDecodeError:
            return {
                "analysis_successful": False,
                "error": "Failed to parse analysis JSON",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {
            "analysis_successful": False,
            "error": str(e),
            "raw_response": ""
        }


def analyze_question_answer_grounding(
    question_data: Dict[str, Any],
    method1_result: Dict[str, Any],
    method2_result: Dict[str, Any] = None,
    client: OpenAI = None,
    model_name: str = None,
    specialty: str = "expert"
) -> Dict[str, Any]:
    """Analyze how well the question and answers are grounded in the source text."""
    
    question = question_data.get("question", "")
    context = question_data.get("context", "")
    options = method1_result.get("options", [])
    correct_answer = method1_result.get("correct_answer", "")
    
    system_message = f"""You are an expert {specialty} analyzing whether a question and its answer options are well-grounded in the provided source text.

Your task is to verify the validity and grounding of the question-answer relationship."""

    user_message = f"""SOURCE TEXT:
{context}

QUESTION: {question}

ANSWER OPTIONS:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

CORRECT ANSWER: {correct_answer}

Please analyze the grounding of this question-answer pair:
1. Is the question clearly answerable from the source text?
2. Are all answer options reasonable given the context?
3. Is the correct answer well-supported by the source text?
4. Are there any ambiguities or unclear aspects?
5. Could multiple answers be considered correct?
6. Is any critical information missing from the context?

Provide your analysis as JSON:
{{
  "question_well_grounded": true/false,
  "answer_options_appropriate": true/false,
  "correct_answer_supported": true/false,
  "ambiguity_level": "high/medium/low/none",
  "potential_issues": ["List any problems with question/answer grounding"],
  "missing_information": ["What information might be missing"],
  "alternative_valid_answers": ["Other answers that could be considered correct"],
  "grounding_quality": "high/medium/low",
  "text_evidence_strength": "How well the text supports the correct answer",
  "confidence_in_assessment": "high/medium/low",
  "detailed_explanation": "Comprehensive analysis of question-answer grounding"
}}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            analysis = json.loads(response_text)
            return {
                "analysis_successful": True,
                "analysis": analysis,
                "raw_response": response_text
            }
        except json.JSONDecodeError:
            return {
                "analysis_successful": False,
                "error": "Failed to parse analysis JSON",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {
            "analysis_successful": False,
            "error": str(e),
            "raw_response": ""
        }


def generate_overall_disagreement_assessment(
    analysis_results: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    specialty: str
) -> Dict[str, Any]:
    """Generate an overall assessment of all disagreement analyses."""
    
    type_a = analysis_results.get("type_a_disagreement", {})
    type_b = analysis_results.get("type_b_disagreement", {})
    grounding = analysis_results.get("grounding_analysis", {})
    
    system_message = f"""You are an expert {specialty} providing a comprehensive assessment of reasoning disagreements and question quality.

Synthesize multiple analyses to provide actionable insights."""

    user_message = f"""I have conducted three analyses:

TYPE A ANALYSIS (Method 1 Internal Consistency):
{json.dumps(type_a, indent=2) if type_a else "No Type A analysis available"}

TYPE B ANALYSIS (Method 1 vs Method 2 Disagreement):
{json.dumps(type_b, indent=2) if type_b else "No Type B analysis available"}

GROUNDING ANALYSIS (Question-Answer Validity):
{json.dumps(grounding, indent=2) if grounding else "No grounding analysis available"}

Please provide an overall assessment that synthesizes these analyses:

{{
  "primary_issues": ["Most significant problems identified"],
  "reasoning_quality_overall": "high/medium/low",
  "most_reliable_approach": "Which reasoning approach to trust",
  "question_validity": "high/medium/low", 
  "actionable_recommendations": ["Specific recommendations for improvement"],
  "confidence_in_overall_assessment": "high/medium/low",
  "summary": "Executive summary of all findings"
}}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            analysis = json.loads(response_text)
            return {
                "analysis_successful": True,
                "analysis": analysis,
                "raw_response": response_text
            }
        except json.JSONDecodeError:
            return {
                "analysis_successful": False,
                "error": "Failed to parse analysis JSON",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {
            "analysis_successful": False,
            "error": str(e),
            "raw_response": ""
        }


def detect_and_diagnose_disagreement(
    question_data: Dict[str, Any],
    method1_result: Dict[str, Any],
    method0_result: Dict[str, Any] = None,
    method2_result: Dict[str, Any] = None,
    client: OpenAI = None,
    model_name: str = "gpt-4",
    specialty: str = "expert"
) -> Dict[str, Any]:
    """
    Detect disagreement between methods and diagnose the problem.
    
    Args:
        question_data: The question data
        method1_result: Result from Method 1 (two-stage analysis)
        method0_result: Result from Method 0 (one-shot), if available
        method2_result: Result from Method 2 (Argonium-style), if available
        client: OpenAI client for diagnosis
        model_name: Name of the model being used for diagnosis
        specialty: The expert specialty persona
        
    Returns:
        Dictionary containing disagreement analysis and diagnosis
    """
    try:
        # Extract predictions from each method
        method1_answer = method1_result.get("synthesis", {}).get("predicted_answer", "unknown")
        method0_answer = method0_result.get("predicted_answer", "unknown") if method0_result else "not_available"
        method2_answer = method2_result.get("predicted_answer", "unknown") if method2_result else "not_available"
        
        # Check for disagreements
        methods_available = []
        predictions = {}
        
        if method0_result:
            methods_available.append("Method 0 (One-shot)")
            predictions["Method 0"] = method0_answer
            
        methods_available.append("Method 1 (Two-stage)")
        predictions["Method 1"] = method1_answer
        
        if method2_result:
            methods_available.append("Method 2 (Argonium)")
            predictions["Method 2"] = method2_answer
        
        # Check if there are disagreements
        unique_answers = set(predictions.values())
        unique_answers.discard("unknown")
        unique_answers.discard("not_available")
        
        has_disagreement = len(unique_answers) > 1
        
        if not has_disagreement:
            return {
                "has_disagreement": False,
                "diagnosis": "All methods agree or insufficient methods available",
                "recommendations": [],
                "analysis_model": model_name
            }
        
        # If there's disagreement, generate diagnosis
        question = question_data["question"]
        options = question_data["options"]
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        
        # Format predictions for diagnosis
        predictions_text = ""
        for method, answer in predictions.items():
            if answer not in ["unknown", "not_available"]:
                predictions_text += f"{method}: Option {answer}\n"
        
        # Format method 1 detailed reasoning
        method1_reasoning = ""
        if method1_result.get("synthesis"):
            synthesis = method1_result["synthesis"]
            method1_reasoning = f"""
Method 1 Two-Stage Analysis:
- Question Restatement: {synthesis.get('question_restatement', 'Not available')}
- Synthesis Reasoning: {synthesis.get('synthesis_reasoning', 'Not available')}
- Decision Rationale: {synthesis.get('decision_rationale', 'Not available')}
- Confidence: {synthesis.get('confidence_level', 'Not available')}
"""
        
        # Format other methods' reasoning
        other_reasoning = ""
        if method0_result:
            other_reasoning += f"\nMethod 0 Reasoning: {method0_result.get('raw_response', 'Not available')}\n"
        if method2_result:
            other_reasoning += f"\nMethod 2 Reasoning: {method2_result.get('raw_response', 'Not available')}\n"
        
        system_message = f"""You are an expert {specialty} tasked with diagnosing why different reasoning methods disagree on the answer to a question.

Your task is to:
1. Analyze the disagreement between the methods
2. Identify potential causes of the disagreement
3. Determine which method(s) may have reasoning flaws
4. Provide specific recommendations for improvement

Be thorough and objective in your analysis."""

        user_message = f"""Question: {question}

Options:
{options_text}

Method Predictions:
{predictions_text}

Detailed Reasoning from Methods:
{method1_reasoning}
{other_reasoning}

Please analyze this disagreement and provide a diagnosis. Structure your response as JSON:
{{
    "disagreement_summary": "Brief summary of the disagreement",
    "potential_causes": ["List of potential causes for the disagreement"],
    "method_analysis": {{
        "method_0_issues": "Analysis of potential issues with Method 0 (if applicable)",
        "method_1_issues": "Analysis of potential issues with Method 1",
        "method_2_issues": "Analysis of potential issues with Method 2 (if applicable)"
    }},
    "likely_correct_method": "Which method is likely most reliable and why",
    "confidence_in_diagnosis": "high/medium/low",
    "recommendations": ["Specific recommendations for resolving the disagreement"],
    "regeneration_needed": true/false,
    "regeneration_focus": "What aspect needs regeneration (if any)"
}}"""

        # Make the API call for diagnosis
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse the JSON response
        try:
            diagnosis = json.loads(response_text)
        except json.JSONDecodeError:
            diagnosis = {
                "disagreement_summary": "Methods disagree but diagnosis parsing failed",
                "potential_causes": ["JSON parsing error"],
                "method_analysis": {
                    "method_0_issues": "Could not analyze",
                    "method_1_issues": "Could not analyze", 
                    "method_2_issues": "Could not analyze"
                },
                "likely_correct_method": "Unknown due to parsing error",
                "confidence_in_diagnosis": "low",
                "recommendations": ["Fix diagnosis parsing"],
                "regeneration_needed": False,
                "regeneration_focus": "None",
                "raw_diagnosis": response_text
            }
        
        return {
            "has_disagreement": True,
            "methods_compared": methods_available,
            "predictions": predictions,
            "diagnosis": diagnosis,
            "analysis_model": model_name,
            "diagnosis_successful": True
        }
        
    except Exception as e:
        return {
            "has_disagreement": True,
            "diagnosis": f"Error during disagreement analysis: {str(e)}",
            "analysis_model": model_name,
            "diagnosis_successful": False,
            "error": str(e)
        }


def regenerate_method1_reasoning(
    question_data: Dict[str, Any],
    disagreement_analysis: Dict[str, Any],
    individual_analyses: Dict[str, Any],
    original_result: Dict[str, Any] = None,
    method0_result: Dict[str, Any] = None,
    method2_result: Dict[str, Any] = None,
    analysis_client: OpenAI = None,
    synthesis_client: OpenAI = None,
    analysis_model: str = "gpt-4",
    synthesis_model: str = "gpt-4",
    specialty: str = "expert"
) -> Dict[str, Any]:
    """
    Regenerate Method 1 reasoning when disagreement is detected.
    
    Args:
        question_data: The question data
        disagreement_analysis: Results from disagreement diagnosis
        individual_analyses: Original individual analyses from stage 1
        method0_result: Method 0 result for context
        method2_result: Method 2 result for context  
        analysis_client: Client for analysis model
        synthesis_client: Client for synthesis model
        analysis_model: Name of analysis model
        synthesis_model: Name of synthesis model
        specialty: Expert specialty persona
        
    Returns:
        Dictionary containing regenerated reasoning
    """
    try:
        diagnosis = disagreement_analysis.get("diagnosis", {})
        regeneration_needed = diagnosis.get("regeneration_needed", False)
        regeneration_focus = diagnosis.get("regeneration_focus", "overall")
        
        if not regeneration_needed:
            return {
                "regeneration_performed": False,
                "reason": "Diagnosis indicated no regeneration needed",
                "original_result_maintained": True
            }
        
        # For now, implement a simple regeneration approach
        # In full implementation, this would have more sophisticated logic
        regenerated_synthesis = synthesize_final_reasoning(
            question_data, individual_analyses, synthesis_client, synthesis_model, specialty
        )
        
        return {
            "regeneration_performed": True,
            "regeneration_focus": regeneration_focus,
            "regenerated_synthesis": regenerated_synthesis,
            "diagnosis_context": diagnosis,
            "analysis_model": analysis_model,
            "synthesis_model": synthesis_model
        }
        
    except Exception as e:
        return {
            "regeneration_performed": False,
            "error": str(e),
            "reason": f"Error during regeneration: {str(e)}"
        }


def generate_reasoning_trace(
    question_data: Dict[str, Any],
    client: OpenAI,
    model_name: str,
    specialty: str = "microbiologist",
    enable_dual_prediction: bool = False,
    grading_client: OpenAI = None,
    grading_model_name: str = None,
    verbose_grading: bool = False,
    reasoning_mode: str = "detailed",
    method2_model: str = None,
    config_file: str = None,
    method1_analysis_model: str = None,
    method1_synthesis_model: str = None,
) -> Dict[str, Any]:
    """
    Generate a reasoning trace for a multiple choice question.

    Args:
        question_data: Dictionary containing the question data
        model_name: The model name to use for generating the reasoning
        specialty: The scientific specialty persona to adopt

    Returns:
        Dictionary with the original question and the reasoning trace
    """
    # Make model_name accessible to print_readable_output function for discrepancy analysis
    global _current_model_name
    _current_model_name = model_name
    global _processed_questions, _total_questions

    # Extract question components with graceful defaults (match argonium_score_parallel approach)
    question_text = question_data.get("question", "")
    correct_answer = question_data.get("answer", "")
    context_text = question_data.get("text", "")

    # Extract options from the question text
    options = extract_mc_options(question_text)

    # Identify the correct option (look for (*) marking or other indicators)
    correct_option_index = -1
    for i, option in enumerate(options):
        if "(*)" in question_text.split("\n\n", 1)[1].split("\n")[i]:
            correct_option_index = i
            break

    if correct_option_index == -1:
        # If we couldn't find the correct answer marker, try to determine from the answer text
        answer_text = correct_answer.lower()
        for i, option in enumerate(options):
            opt_marker = f"option {i+1}"
            ans_marker = f"answer {i+1}"
            letter_marker = chr(ord("a") + i)  # a, b, c, ...

            if (
                opt_marker in answer_text
                or ans_marker in answer_text
                or answer_text.startswith(letter_marker + ".")
                or answer_text.startswith(letter_marker + ")")
            ):
                correct_option_index = i
                break

            # Also check for exact match of the option text
            if option.lower() in answer_text:
                correct_option_index = i
                break

    # If we still don't have a correct answer index, try with "all of the above"
    if correct_option_index == -1 and len(options) > 0:
        last_option = options[-1].lower()
        if "all of the above" in last_option:
            correct_option_index = len(options) - 1

    # If we still don't have a correct answer index, log a warning
    if correct_option_index == -1:
        log_message(
            f"Warning: Could not determine correct answer index for question: {question_text[:100]}...",
            log_level="WARNING",
        )
        # Use the first option as a fallback
        correct_option_index = 0

    # Get the expert persona
    persona = get_expert_persona(specialty)

    # NEW METHOD 1: Two-stage approach with separate models
    
    # Configure Method 1 clients (use defaults if not specified)
    method1_analysis_model_name = method1_analysis_model if method1_analysis_model else model_name
    method1_synthesis_model_name = method1_synthesis_model if method1_synthesis_model else model_name
    
    method1_analysis_client = client
    method1_synthesis_client = client
    
    # If different models specified, create separate clients
    if method1_analysis_model_name != model_name:
        method1_analysis_client, method1_analysis_model_name = configure_apis(method1_analysis_model_name, config_file)
    if method1_synthesis_model_name != model_name:
        method1_synthesis_client, method1_synthesis_model_name = configure_apis(method1_synthesis_model_name, config_file)
    
    try:
        # Stage 1: Generate individual option analyses
        log_message(f"Method 1 Stage 1: Analyzing individual options with {method1_analysis_model_name}", log_level="INFO")
        individual_analyses = analyze_individual_options(
            {"question": question_text, "options": options},
            method1_analysis_client,
            method1_analysis_model_name,
            specialty
        )
        
        # Stage 2: Synthesize final reasoning
        log_message(f"Method 1 Stage 2: Synthesizing final reasoning with {method1_synthesis_model_name}", log_level="INFO")
        synthesis_result = synthesize_final_reasoning(
            {"question": question_text, "options": options},
            individual_analyses,
            method1_synthesis_client,
            method1_synthesis_model_name,
            specialty
        )
        
        # Convert synthesis result to format expected by rest of system
        synthesis_data = synthesis_result.get("synthesis", {})
        
        json_content = {
            "thought_process": {
                f"option_{i+1}": individual_analyses.get("individual_analyses", {}).get(f"option_{i+1}", {}).get("analysis", f"Analysis for option {i+1}")
                for i in range(len(options))
            },
            "prediction": {
                "predicted_answer": synthesis_data.get("predicted_answer", "1"),
                "prediction_reasoning": synthesis_data.get("prediction_reasoning", "No reasoning provided"),
                "confidence_level": synthesis_data.get("confidence_level", "medium"),
                "confidence_explanation": synthesis_data.get("confidence_explanation", "No explanation provided")
            },
            "scientific_conclusion": synthesis_data.get("scientific_conclusion", "No conclusion provided"),
            "two_stage_method": True,
            "question_restatement": synthesis_data.get("question_restatement", ""),
            "synthesis_reasoning": synthesis_data.get("synthesis_reasoning", ""),
            "option_comparison": synthesis_data.get("option_comparison", ""),
            "decision_rationale": synthesis_data.get("decision_rationale", "")
        }
        
        log_message("Successfully generated Method 1 two-stage reasoning", log_level="INFO")

        # Create the result structure
        result = {
            "question": question_text,
            "context": context_text,
            "correct_answer_index": correct_option_index,
            "correct_answer": (
                options[correct_option_index]
                if correct_option_index < len(options)
                else ""
            ),
            "options": options,
            "reasoning": json_content,
            "model_used": f"{method1_analysis_model} + {method1_synthesis_model}",  # Show both models
            "method1_details": {
                "individual_analyses": individual_analyses,
                "synthesis_result": synthesis_result,
                "analysis_model": method1_analysis_model,
                "synthesis_model": method1_synthesis_model,
                "two_stage_approach": True
            },
            "raw_response": synthesis_result.get("raw_response", ""),
        }

        # Generate dual prediction if enabled
        dual_prediction_data = None
        if enable_dual_prediction:
            log_message("Generating Method 2 (Argonium-style) prediction", log_level="INFO")
            
            # Get Method 2 model 
            method2_model_name = method2_model if method2_model else model_name
            method2_client = client
            if method2_model_name != model_name:
                method2_client, method2_model_name = configure_apis(method2_model_name, config_file)
            
            argonium_prediction = generate_argonium_style_prediction(
                question_text, options, method2_client, method2_model_name, specialty
            )
            
            # Generate comparison analysis between Method 1 and Method 2
            comparison_analysis = generate_prediction_comparison(
                json_content,  # Method 1 reasoning data
                argonium_prediction,  # Method 2 argonium prediction
                question_text,
                options,
                client,
                model_name,
                specialty,
            )
            
            # Create dual prediction data structure
            dual_prediction_data = {
                "argonium_prediction": argonium_prediction,
                "comparison_analysis": comparison_analysis
            }
        
        # Add dual prediction data to result if available
        if dual_prediction_data:
            result["dual_prediction"] = dual_prediction_data

        # Detect disagreements between methods and regenerate if needed
        if dual_prediction_data:
            log_message("Analyzing agreements/disagreements between methods", log_level="INFO")
            
            # Perform comprehensive disagreement analysis (Type A + Type B + Grounding)
            comprehensive_analysis = comprehensive_disagreement_analysis(
                question_data=question_data,
                method1_result=result,
                method2_result=dual_prediction_data["argonium_prediction"],
                client=client,
                model_name=model_name,
                specialty=specialty
            )
            
            # Store comprehensive analysis in result
            result["comprehensive_disagreement_analysis"] = comprehensive_analysis
            
            # Also store simplified version for backward compatibility
            fault_analysis = comprehensive_analysis.get("fault_analysis", {})
            result["disagreement_analysis"] = {
                "has_disagreement": fault_analysis.get("answer_comparison", {}).get("has_disagreement", False),
                "analysis_type": "fault_oriented", 
                "primary_fault": fault_analysis.get("diagnostic_analysis", {}).get("primary_fault_category", "unknown"),
                "summary": fault_analysis.get("root_cause_analysis", {}).get("primary_root_cause", "")
            }
            
            # If disagreement is found and regeneration is recommended, regenerate Method 1
            fault_analysis = comprehensive_analysis.get("fault_analysis", {})
            should_regenerate = (
                fault_analysis.get("answer_comparison", {}).get("has_disagreement", False) and
                fault_analysis.get("diagnostic_analysis", {}).get("primary_fault_category") in ["reasoning_logic", "invalid_assumptions"]
            )
            if should_regenerate:
                
                log_message("Disagreement detected - regenerating Method 1 reasoning", log_level="INFO")
                
                regeneration_result = regenerate_method1_reasoning(
                    question_data=question_data,
                    original_result=result,
                    disagreement_analysis=comprehensive_analysis,
                    individual_analyses=individual_analyses,
                    method0_result=None,
                    method2_result=dual_prediction_data["argonium_prediction"],
                    analysis_client=method1_analysis_client,
                    synthesis_client=method1_synthesis_client,
                    analysis_model=method1_analysis_model,
                    synthesis_model=method1_synthesis_model,
                    specialty=specialty
                )
                
                # Store regeneration results
                result["regeneration_result"] = regeneration_result
                
                # If regeneration was successful, log the outcome
                if regeneration_result.get("regeneration_performed", False):
                    log_message("Method 1 regeneration completed", log_level="INFO")
                else:
                    log_message(f"Method 1 regeneration skipped: {regeneration_result.get('reason', 'Unknown')}", log_level="INFO")
        
        # Grade the Method 1 prediction if grading model is available
        if grading_client is not None and grading_model_name is not None:
            predicted_answer = json_content.get("prediction", {}).get("predicted_answer", "")
            correct_answer = result["correct_answer"]

            if predicted_answer and correct_answer:
                grading_result = grade_answer(
                    predicted_answer=predicted_answer,
                    correct_answer=correct_answer,
                    question_text=question_text,
                    options=options,
                    grading_client=grading_client,
                    grading_model_name=grading_model_name,
                    verbose=verbose_grading,
                )

                # Store grading results
                result["grading_result"] = grading_result
                result["prediction_correct"] = grading_result.get("is_correct", False)

                log_message(
                    f"Method 1 grading result: {grading_result.get('is_correct', False)} (confidence: {grading_result.get('confidence', 'unknown')})",
                    log_level="INFO",
                )

        # Return the result
        return result

    except Exception as e:
        error_msg = str(e)
        log_message(f"Error generating reasoning trace: {error_msg}", log_level="ERROR")

        # Create a safe options accessor
        correct_answer = "Unknown"
        try:
            if correct_option_index >= 0 and correct_option_index < len(options):
                correct_answer = options[correct_option_index]
        except Exception:
            pass

        # Create a more informative error structure to help debug
        error_structure = {
            "error_type": type(e).__name__,
            "error_message": error_msg,
            "model_used": f"{method1_analysis_model} + {method1_synthesis_model}",
            "query_successful": False,
        }

        return {
            "question": question_text,
            "context": (
                context_text[:500] + "..." if len(context_text) > 500 else context_text
            ),  # Include truncated context
            "correct_answer_index": correct_option_index,
            "correct_answer": correct_answer,
            "options": options,
            "reasoning": {
                "thought_process": {},
                "prediction": {
                    "predicted_answer": "Error occurred",
                    "prediction_reasoning": "An error occurred while generating reasoning.",
                    "confidence_level": "unknown",
                    "confidence_explanation": "Processing error",
                },
                "scientific_conclusion": f"Failed to generate reasoning due to an error: {error_msg}",
            },
            "error_details": error_structure,
        }


def generate_coherent_stream_analysis(
    reasoning_trace: Dict[str, Any],
    specialty: str = "expert",
    client: OpenAI = None,
    model_name: str = None,
) -> str:
    """
    Generate a coherent stream of thought analysis showing internal debate between options.

    Args:
        reasoning_trace: The reasoning trace to analyze
        specialty: The expert specialty persona
        model_name: The model to use for generating the synthesis

    Returns:
        A natural internal dialogue showing the debate between different options
    """
    # Extract all the raw content from the reasoning trace
    question = reasoning_trace.get("question", "Unknown question")
    options = reasoning_trace.get("options", [])
    correct_answer_idx = reasoning_trace.get("correct_answer_index", -1)
    correct_answer = (
        options[correct_answer_idx]
        if 0 <= correct_answer_idx < len(options)
        else "Unknown"
    )
    
    reasoning = reasoning_trace.get("reasoning", {})
    thought_process = reasoning.get("thought_process", {})
    prediction = reasoning.get("prediction", {})
    scientific_conclusion = reasoning.get(
        "scientific_conclusion", reasoning.get("conclusion", "")
    )
    
    return f"Stream analysis for {question[:50]}... [Implementation placeholder]"



def generate_coherent_stream_analysis(
    reasoning_trace: Dict[str, Any],
    specialty: str = "expert",
    client: OpenAI = None,
    model_name: str = None,
) -> str:
    """
    Generate a coherent stream of thought analysis showing internal debate between options.

    Args:
        reasoning_trace: The reasoning trace to analyze
        specialty: The expert specialty persona
        model_name: The model to use for generating the synthesis

    Returns:
        A natural internal dialogue showing the debate between different options
    """
    # Extract all the raw content from the reasoning trace
    question = reasoning_trace.get("question", "Unknown question")
    options = reasoning_trace.get("options", [])
    correct_answer_idx = reasoning_trace.get("correct_answer_index", -1)
    correct_answer = (
        options[correct_answer_idx]
        if 0 <= correct_answer_idx < len(options)
        else "Unknown"
    )
    reasoning = reasoning_trace.get("reasoning", {})
    thought_process = reasoning.get("thought_process", {})
    prediction = reasoning.get("prediction", {})
    scientific_conclusion = reasoning.get(
        "scientific_conclusion", reasoning.get("conclusion", "")
    )
    was_correct = reasoning_trace.get("prediction_correct", False)
    predicted_num = reasoning_trace.get("predicted_num", None)

    # Extract dual prediction data if available
    dual_prediction = reasoning_trace.get("dual_prediction", {})
    argonium_prediction = dual_prediction.get("argonium_prediction", {})
    has_dual_prediction = bool(dual_prediction)

    # Build comprehensive context for synthesis - focusing on the internal debate
    synthesis_context = f"""QUESTION I'M WORKING ON:
{question}

POSSIBLE THOUGHTS THAT CAME TO MIND:
"""

    # Present options as natural thoughts that occurred, not explicit choices
    for i, option in enumerate(options):
        synthesis_context += f"• {option}\n"

    synthesis_context += f"\nMY DETAILED THINKING ABOUT EACH POSSIBILITY:\n"

    # Add all the detailed reasoning for each option, but frame as thoughts
    option_keys = sorted(
        [k for k in thought_process.keys() if k.startswith("option_")],
        key=lambda x: (
            int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else float("inf")
        ),
    )

    for opt_key in option_keys:
        try:
            opt_idx = int(opt_key.split("_")[-1])
            if opt_idx <= len(options):
                synthesis_context += f"\nAbout the idea that {options[opt_idx-1] if opt_idx-1 < len(options) else 'Unknown'}:\n"
                synthesis_context += f"{thought_process[opt_key]}\n"
        except (ValueError, IndexError):
            continue

    # Add what I eventually decided
    synthesis_context += f"\nWHAT I ULTIMATELY SETTLED ON:\n"
    synthesis_context += f"My Final Thought (Detailed Approach): {prediction.get('predicted_answer', 'Unknown')}\n"
    synthesis_context += f"My Reasoning: {prediction.get('prediction_reasoning', 'No reasoning provided')}\n"
    synthesis_context += (
        f"My Confidence: {prediction.get('confidence_level', 'unknown')}\n"
    )

    # Add dual prediction information if available
    if has_dual_prediction:
        synthesis_context += f"\nMY QUICK/DIRECT APPROACH ANSWER:\n"
        synthesis_context += (
            f"Direct Answer: {argonium_prediction.get('predicted_answer', 'Unknown')}\n"
        )
        synthesis_context += f"Direct Response: {argonium_prediction.get('raw_response', 'No response')}\n"

    # The actual outcome (but don't reveal this in the internal dialogue initially)
    correct_thought = (
        options[correct_answer_idx] if correct_answer_idx < len(options) else "Unknown"
    )
    actual_result = f"[ACTUAL RESULT: The idea that '{correct_thought}' was correct"
    if predicted_num is not None:
        if was_correct:
            actual_result += f" - I got it right!]"
        else:
            predicted_thought = (
                options[predicted_num - 1]
                if predicted_num - 1 < len(options)
                else f"option {predicted_num}"
            )
            actual_result += f" - I was wrong, I went with '{predicted_thought}']"
    else:
        actual_result += f" - my final choice was unclear]"

    # Create the synthesis prompt for internal debate
    dual_instruction = ""
    if has_dual_prediction:
        dual_instruction = f"""
7. You also tried a quick/direct approach and got a different perspective
8. You compare your thorough analysis with your quick gut response
"""

    synthesis_prompt = f"""You are a {specialty} working through this question. I want you to create a natural internal dialogue that shows you actively debating between different thoughts and possibilities as they occur to you.

CRITICAL INSTRUCTIONS:
- Don't refer to these as "Option 1, Option 2" etc. Instead, treat them as natural thoughts that just came to mind, and show yourself debating between these ideas organically.
- REVEAL ALL YOUR INTERNAL DETAILS AND REASONING - don't hold anything back or try to be brief
- Include ALL the technical details, scientific knowledge, and specific reasoning that goes through your mind
- Show your complete thought process with full scientific depth and complexity
- Don't summarize or abbreviate - show the full internal scientific debate
{f"- Include your reflection on trying both detailed and quick approaches to the same problem" if has_dual_prediction else ""}

This should be written as your ACTUAL internal thoughts while you're working on the problem - the real-time mental conversation you're having with yourself as you:

1. Read the question and start thinking
2. Different possibilities come to mind naturally
3. You weigh each idea, arguing for and against them in your mind with FULL scientific detail
4. You go back and forth between different thoughts, including ALL technical reasoning
5. You feel yourself leaning toward certain ideas, then second-guessing with complete scientific rationale
6. You finally settle on your answer with full explanation{dual_instruction}

Write this as a genuine stream of consciousness internal debate. Use phrases like:
- "Hmm, let me think about this..."
- "Well, it could be that... but then again..."
- "Actually, what if it's more about... no wait..."
- "I'm second-guessing myself here..."
- "Going back to that thought about... the thing that bothers me is..."
- "I keep coming back to the idea that... and here's why..."
- "Okay, I'm torn between thinking it's X versus Y..."
- "My gut is telling me... but my analytical side says..."
- "Wait, that makes me think it might actually be..."

Show the ACTUAL mental back-and-forth debate between different ideas WITH FULL SCIENTIFIC DETAIL. Include ALL your reasoning - molecular mechanisms, biochemical pathways, experimental evidence, literature knowledge, etc. Include moments where you:
- Favor one idea, then change your mind (with full scientific reasoning)
- See strengths and weaknesses in different possibilities with technical details
- Feel uncertainty and work through it with complete scientific analysis
- Draw on your expertise as a {specialty} with specific technical knowledge
- Experience that "aha!" moment when something clicks scientifically
- Have competing thoughts pulling you in different directions with full explanations

Make it feel like I'm listening to your actual thought process in real-time, with all the uncertainty, reconsideration, and mental debate that goes into making a decision. Include ALL the scientific details and technical reasoning - don't hold back or summarize anything. The ideas should feel like they're naturally occurring to you, not like you're systematically going through a list.

Write about 500-700 words. Focus on the LIVE internal debate between naturally occurring thoughts with COMPLETE scientific detail, not a post-hoc analysis. REVEAL ALL INTERNAL REASONING AND TECHNICAL DETAILS.

{actual_result}"""

    # Generate the synthesis using the AI model
    if client is None or model_name is None:
        # Fallback to simple template if no model available
        return f"Unable to generate coherent stream analysis - no model specified for synthesis."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {specialty} engaging in honest self-reflection about your own reasoning process. You write in a natural, conversational internal monologue style that captures authentic thought patterns, including moments of uncertainty, connections to knowledge, and genuine self-assessment.",
                },
                {
                    "role": "user",
                    "content": f"{synthesis_context}\n\n{synthesis_prompt}",
                },
            ],
            temperature=0.7,  # Higher temperature for more natural, varied expression
            max_tokens=2000,  # Increased for full detailed scientific reasoning
        )

        analysis_text = response.choices[0].message.content.strip()
        return analysis_text

    except Exception as e:
        log_message(
            f"Error generating coherent stream analysis: {e}", log_level="ERROR"
        )
        return f"Error generating stream analysis: {str(e)}"


def print_readable_output(
    question_data: Dict[str, Any],
    reasoning_trace: Dict[str, Any],
    specialty: str = "expert",
    show_stream_analysis: bool = False,
    client: OpenAI = None,
):
    """Print a human-readable version of the reasoning trace to the console.

    Args:
        question_data: The original question data
        reasoning_trace: The generated reasoning trace
        specialty: The specialty persona used in the analysis
        show_stream_analysis: Whether to show coherent stream analysis after the reasoning
    """
    # Access the global model name
    global _current_model_name
    try:
        print("\n" + "=" * 80)
        # Get the full question text
        question_only = question_data["question"]
        print(f"QUESTION: {question_only}")
        print("-" * 80)

        # Print options
        options = reasoning_trace["options"]
        correct_index = reasoning_trace["correct_answer_index"]

        print("OPTIONS:")
        for i, option in enumerate(options):
            marker = "✓" if i == correct_index else " "
            print(f"{marker} {i+1}. {option}")

        print("-" * 100)
        print(f"{specialty.upper()}'S THOUGHT PROCESS:")

        # Check for errors
        has_error = (
            reasoning_trace["reasoning"].get("extraction_failed")
            or reasoning_trace.get("error_details") is not None
        )

        if has_error:
            print("Error in response - showing available information:")

            # Show error details if present
            if reasoning_trace.get("error_details"):
                error_details = reasoning_trace["error_details"]
                print(f"Error type: {error_details.get('error_type')}")
                print(f"Error message: {error_details.get('error_message')}")

            # Show the raw response or summary if available
            raw_response = reasoning_trace["reasoning"].get("raw_text", "")
            if not raw_response:
                raw_response = reasoning_trace["reasoning"].get(
                    "scientific_conclusion", ""
                )

            if len(raw_response) > 20:  # Only show if it has meaningful content
                print("\nRaw response snippet:")
                print(
                    raw_response[:500] + "..."
                    if len(raw_response) > 500
                    else raw_response
                )
        else:
            # Print reasoning based on the mode used
            reasoning_data = reasoning_trace["reasoning"]
            
            # Check if this is focused mode
            if "key_principle" in reasoning_data:
                # Focused mode display
                print(f"\n🎯 KEY PRINCIPLE: {reasoning_data.get('key_principle', 'Not specified')}")
                
                quick_elim = reasoning_data.get('quick_elimination', {})
                if quick_elim.get('dismissed_options'):
                    print(f"\n❌ QUICKLY ELIMINATED: Options {', '.join(quick_elim.get('dismissed_options', []))}")
                    print(f"   Reasoning: {quick_elim.get('reasoning', 'Not specified')}")
                
                focused_analysis = reasoning_data.get('focused_analysis', {})
                if focused_analysis.get('viable_options'):
                    print(f"\n🔍 FOCUSED ANALYSIS: Options {', '.join(focused_analysis.get('viable_options', []))}")
                    print(f"   {focused_analysis.get('detailed_reasoning', 'Not specified')}")
                    
            # Check if this is efficient mode
            elif "quick_analysis" in reasoning_data:
                # Efficient mode display
                print(f"\n⚡ QUICK ANALYSIS: {reasoning_data.get('quick_analysis', 'Not specified')}")
                print(f"\n🚫 ELIMINATION: {reasoning_data.get('elimination', 'Not specified')}")
                
            else:
                # Standard detailed mode - Print thought process for each option
                thought_process = reasoning_data.get("thought_process", {})

                # Sort option keys numerically
                option_keys = sorted(
                    [k for k in thought_process.keys() if k.startswith("option_")],
                    key=lambda x: (
                        int(x.split("_")[-1])
                        if x.split("_")[-1].isdigit()
                        else float("inf")
                    ),
                )

                # Print options in order
                for opt_key in option_keys:
                    try:
                        opt_idx = int(opt_key.split("_")[-1]) - 1
                        if opt_idx >= 0 and opt_idx < len(options):
                            print(f"\n💭 OPTION {opt_idx+1}: {options[opt_idx]}")

                            # Get the thoughts and clean them
                            thoughts = thought_process[opt_key]

                            # Format the thought process text with indentation
                            if isinstance(thoughts, str):
                                # Clean the text to handle potential formatting issues
                                clean_text = re.sub(
                                    r'"\s*$', "", thoughts
                                )  # Remove trailing quotes
                                clean_text = re.sub(
                                    r'^\s*"', "", clean_text
                                )  # Remove leading quotes

                                # Handle remaining escape sequences
                                clean_text = clean_text.replace("\\", "")

                                # Format with indentation
                                formatted_thoughts = "\n".join(
                                    "   " + line for line in clean_text.split("\n")
                                )
                                print(formatted_thoughts)
                            elif isinstance(thoughts, dict):
                                # Convert dict to indented text
                                for k, v in thoughts.items():
                                    print(f"   {k}:")
                                    if isinstance(v, str):
                                        print(
                                            "\n".join(
                                                "      " + line for line in v.split("\n")
                                            )
                                        )
                                    else:
                                        print(f"      {v}")
                            else:
                                print(f"   {thoughts}")

                            print("-" * 80)  # Separator between options
                    except (ValueError, IndexError) as e:
                        pass

            # Print the prediction
            print("\n" + "=" * 80)
            print(f"🔮 {specialty.upper()}'S PREDICTION:")
            prediction = reasoning_trace["reasoning"].get("prediction", {})

            # Check if prediction is a string or a dict - handle both cases
            if isinstance(prediction, str):
                # Try to parse it as JSON if it's a string (might be a direct JSON output)
                try:
                    prediction_dict = json.loads(prediction)
                    if (
                        isinstance(prediction_dict, dict)
                        and "predicted_answer" in prediction_dict
                    ):
                        prediction = prediction_dict
                    else:
                        # Just use it as the predicted answer
                        prediction = {"predicted_answer": prediction}
                except:
                    # If it's not valid JSON, just use it as the predicted answer
                    prediction = {"predicted_answer": prediction}

            predicted_answer = prediction.get(
                "predicted_answer", "No prediction provided"
            )

            # Use grading model results - no regex fallback
            if "grading_result" in reasoning_trace:
                grading_result = reasoning_trace["grading_result"]
                predicted_correct = grading_result.get("is_correct", False)
                predicted_num = reasoning_trace.get("predicted_num", None)

                grading_confidence = grading_result.get("confidence", "unknown")
                grading_reasoning = grading_result.get(
                    "reasoning", "No reasoning provided"
                )

                # Display the grading result
                if predicted_correct:
                    status_text = "✅ CORRECT (graded)"
                else:
                    status_text = f"❌ INCORRECT (graded) - actual correct answer is Option {correct_index+1}"

                if predicted_num:
                    print(f"   Predicted Answer: Option {predicted_num} {status_text}")
                else:
                    print(f"   Predicted Answer: {predicted_answer} {status_text}")

                print(f"   Grading Confidence: {grading_confidence}")
                print(f"   Grading Reasoning: {grading_reasoning}")

                # Store the grading result for overall accuracy calculation
                reasoning_trace["prediction_correct"] = predicted_correct
                if predicted_num:
                    reasoning_trace["predicted_num"] = predicted_num

            else:
                # No grading model available - cannot determine correctness
                print(
                    f"   Predicted Answer: {predicted_answer} (No grading model available - correctness unknown)"
                )
                print(
                    f"   ⚠️  Warning: No grading model was available to verify this answer"
                )
                print(f"   To enable answer verification, use --grading argument")

                # Mark as unknown since we cannot verify without grading model
                reasoning_trace["prediction_correct"] = None
                reasoning_trace["predicted_num"] = None
                reasoning_trace["grading_unavailable"] = True

            print("\n   Prediction Reasoning:")
            prediction_reasoning = prediction.get(
                "prediction_reasoning", "No reasoning provided"
            )

            # Clean up the prediction reasoning
            if isinstance(prediction_reasoning, str):
                # Remove excess quotes and JSON escapes
                prediction_reasoning = re.sub(r'^\s*"', "", prediction_reasoning)
                prediction_reasoning = re.sub(r'"\s*$', "", prediction_reasoning)
                prediction_reasoning = prediction_reasoning.replace('\\"', '"').replace(
                    "\\n", "\n"
                )
                prediction_reasoning = re.sub(
                    r'",\s*"confidence_level.*$', "", prediction_reasoning
                )

            # Format the reasoning with indentation
            formatted_reasoning = "\n".join(
                "      " + line for line in str(prediction_reasoning).split("\n")
            )
            print(formatted_reasoning)

            # Extract and clean confidence level
            confidence_level = prediction.get("confidence_level", "Not specified")
            if isinstance(confidence_level, str):
                confidence_level = confidence_level.strip().lower()
                confidence_level = re.sub(r'^\s*"', "", confidence_level)
                confidence_level = re.sub(r'"\s*$', "", confidence_level)

            print(f"\n   Confidence Level: {confidence_level}")
            print("\n   Confidence Explanation:")

            # Clean up confidence explanation
            confidence_explanation = prediction.get(
                "confidence_explanation", "No explanation provided"
            )
            if isinstance(confidence_explanation, str):
                # Remove excess quotes and JSON escapes
                confidence_explanation = re.sub(r'^\s*"', "", confidence_explanation)
                confidence_explanation = re.sub(r'"\s*$', "", confidence_explanation)
                confidence_explanation = confidence_explanation.replace(
                    '\\"', '"'
                ).replace("\\n", "\n")
                confidence_explanation = re.sub(
                    r'",\s*"scientific_conclusion.*$', "", confidence_explanation
                )

            # Format the explanation with indentation
            formatted_explanation = "\n".join(
                "      " + line for line in str(confidence_explanation).split("\n")
            )
            print(formatted_explanation)

        print("\n" + "=" * 80)
        print("🔬 SCIENTIFIC CONCLUSION:")

        scientific_conclusion = reasoning_trace["reasoning"].get(
            "scientific_conclusion", ""
        )

        if (
            not scientific_conclusion
            or scientific_conclusion == "No conclusion provided."
        ):
            scientific_conclusion = reasoning_trace["reasoning"].get(
                "conclusion", "No conclusion provided"
            )

        if reasoning_trace["reasoning"].get("extraction_failed", False):
            print("Error in response parsing - conclusion not available")
        else:
            # Clean up the scientific conclusion
            if isinstance(scientific_conclusion, str):
                # Remove excess quotes and JSON escapes
                scientific_conclusion = re.sub(r'^\s*"', "", scientific_conclusion)
                scientific_conclusion = re.sub(r'"\s*$', "", scientific_conclusion)
                scientific_conclusion = scientific_conclusion.replace(
                    '\\"', '"'
                ).replace("\\n", "\n")

            # Format the conclusion with indentation and block quotes
            formatted_conclusion = "\n".join(
                "> " + line for line in str(scientific_conclusion).split("\n")
            )
            print(formatted_conclusion)
        print("=" * 80)

        # Add dual prediction analysis if available
        if "dual_prediction" in reasoning_trace:
            print("\n" + "=" * 80)
            print("🔄 DUAL PREDICTION COMPARISON")
            print("=" * 80)

            dual_data = reasoning_trace["dual_prediction"]
            argonium_pred = dual_data.get("argonium_prediction", {})
            comparison = dual_data.get("comparison_analysis", "No comparison available")

            print("\n🎯 ARGONIUM-STYLE PREDICTION:")
            print(f"   Answer: {argonium_pred.get('predicted_answer', 'Unknown')}")
            print("   Response:")
            argonium_response = argonium_pred.get("raw_response", "No response")
            formatted_response = "\n".join(
                "      " + line for line in argonium_response.split("\n")
            )
            print(formatted_response)

            # Add file comparison if available
            if reasoning_trace.get("argonium_file_comparison"):
                file_comparison = reasoning_trace["argonium_file_comparison"]
                argonium_file_result = file_comparison.get("argonium_result", {})
                question_match = file_comparison.get("question_match", {})
                
                print(f"\n📋 COMPARISON WITH ARGONIUM FILE:")
                if question_match.get("is_match", False):
                    file_answer = argonium_file_result.get("model_answer", "")
                    file_score = argonium_file_result.get("score", 0)
                    file_correct = file_score >= 1
                    
                    new_answer = argonium_pred.get("predicted_answer", "Unknown")
                    new_extraction_successful = argonium_pred.get("extraction_successful", False)
                    new_correct = argonium_pred.get("prediction_correct", False)
                    
                    print(f"   File prediction: {file_answer[:100]}{'...' if len(file_answer) > 100 else ''}")
                    print(f"   File result: {file_score} ({'✓ Correct' if file_correct else '✗ Incorrect'})")
                    print(f"   New prediction: {new_answer}")
                    print(f"   New result: {'✓ Correct' if new_correct else '✗ Incorrect'} ({'✓ Extracted' if new_extraction_successful else '✗ Failed to extract'})")
                    
                    # Clear comparison based on actual outcomes
                    if file_correct and new_correct:
                        print("   🎯 Both methods got the correct answer")
                    elif file_correct and not new_correct:
                        if new_extraction_successful:
                            print("   📉 File method correct, new method chose wrong answer")
                        else:
                            print("   ⚠️  File method correct, new method failed to extract answer")
                    elif not file_correct and new_correct:
                        print("   📈 File method wrong, new method got correct answer")
                    else:
                        if new_extraction_successful:
                            print("   ❌ Both methods chose wrong answers")
                        else:
                            print("   💥 File method wrong, new method failed to extract")
                        
                    # Answer content comparison
                    file_choice = argonium_file_result.get("evaluation", {}).get("model_choice", "")
                    new_choice = argonium_pred.get("extracted_choice", "")
                    
                    if file_choice and new_choice:
                        if str(file_choice).upper() == str(new_choice).upper():
                            print("   🎯 Answer match: Same choice selected")
                        else:
                            print(f"   🔄 Answer difference: File='{file_choice}', New='{new_choice}'")
                    
                else:
                    print("   ⚠️  Question verification failed - comparison skipped")
                    print(f"   Reason: {question_match.get('reasoning', 'Unknown')}")

            print("\n" + "-" * 80)
            print("🤔 COMPARISON ANALYSIS:")
            print("-" * 80)

            # Format the comparison analysis with proper indentation
            formatted_comparison = "\n".join(line for line in comparison.split("\n"))
            print(formatted_comparison)

            print("=" * 80)

        # Add coherent stream analysis if requested
        if show_stream_analysis:
            print("\n" + "=" * 80)
            print("🌊 COHERENT STREAM OF THOUGHT ANALYSIS")
            print("=" * 80)

            try:
                # Use the current model for synthesis
                global _current_model_name
                stream_analysis = generate_coherent_stream_analysis(
                    reasoning_trace, specialty, client, _current_model_name
                )

                # Print the stream analysis without truncation (preserve natural line breaks)
                print(stream_analysis)

            except Exception as stream_error:
                print(f"Error generating stream analysis: {str(stream_error)}")

            print("=" * 80)

    except Exception as e:
        print("\nError displaying reasoning output:")
        print(f"Error: {str(e)}")
        print(f"Question: {question_data.get('question', 'Unknown')[:100]}...")
        print("=" * 80 + "\n")


def process_trace_batch(trace_batch: List[Tuple[int, Dict[str, Any]]]) -> Tuple[List[Dict], Dict[str, int], Dict[str, List]]:
    """
    Process a batch of traces in parallel to extract summaries and statistics.
    
    Args:
        trace_batch: List of (index, trace) tuples
        
    Returns:
        Tuple of (batch_summary, batch_stats, batch_confidence)
    """
    batch_summary = []
    batch_stats = {"correct": 0, "incorrect": 0, "total": 0, "ungraded": 0}
    batch_confidence = {"high": [], "medium": [], "low": []}
    
    for i, trace in trace_batch:
        if "reasoning" not in trace:
            continue

        # Extract key information
        question = trace.get("question", "Unknown question")[:100] + "..."
        predicted_answer = (
            trace.get("reasoning", {})
            .get("prediction", {})
            .get("predicted_answer", "Unknown")
        )
        correct_answer_idx = trace.get("correct_answer_index", -1)
        is_correct = trace.get("prediction_correct", None)
        confidence = (
            trace.get("reasoning", {})
            .get("prediction", {})
            .get("confidence_level", "unknown")
        )

        # Update statistics (only count graded predictions)
        if is_correct is not None:
            batch_stats["total"] += 1
            if is_correct:
                batch_stats["correct"] += 1
            else:
                batch_stats["incorrect"] += 1
        else:
            # Track ungraded predictions
            batch_stats["ungraded"] += 1

        # Track confidence levels (only for graded predictions)
        if is_correct is not None and confidence.lower() in batch_confidence:
            batch_confidence[confidence.lower()].append(
                {
                    "question_num": i + 1,
                    "correct": is_correct,
                    "predicted": predicted_answer,
                }
            )

        # Create a summary entry
        batch_summary.append(
            {
                "question_number": i + 1,
                "question_snippet": question,
                "predicted_answer": predicted_answer,
                "correct_answer_index": (
                    correct_answer_idx + 1 if correct_answer_idx >= 0 else "Unknown"
                ),
                "was_correct": is_correct,
                "confidence_level": confidence,
                "key_reasoning_points": trace.get("reasoning", {})
                .get("prediction", {})
                .get("prediction_reasoning", "")[:200]
                + "...",
            }
        )
    
    return batch_summary, batch_stats, batch_confidence


def generate_whole_trace_analysis(
    reasoning_traces: List[Dict[str, Any]],
    client: OpenAI,
    model_name: str,
    specialty: str = "expert",
) -> Dict[str, Any]:
    """
    Generate a coherent narrative analysis from the collected reasoning traces.

    Args:
        reasoning_traces: List of reasoning trace dictionaries
        model_name: The model name to use for generating the analysis
        specialty: The expert specialty persona to adopt

    Returns:
        Dictionary containing the whole trace analysis
    """
    log_message("Generating whole trace analysis...", log_level="INFO")

    # Prepare a summary of all the reasoning traces using parallel processing
    log_message("Processing traces in parallel for analysis...", log_level="INFO")
    
    # Determine optimal batch size and number of workers
    num_traces = len(reasoning_traces)
    max_workers = min(4, max(1, num_traces // 20))  # 4 workers max, at least 20 traces per worker
    batch_size = max(1, num_traces // max_workers)
    
    # Split traces into batches with their original indices
    batches = []
    for i in range(0, num_traces, batch_size):
        batch = list(enumerate(reasoning_traces[i:i + batch_size], start=i))
        if batch:  # Only add non-empty batches
            batches.append(batch)
    
    log_message(f"Processing {num_traces} traces in {len(batches)} batches using {max_workers} workers", log_level="INFO")
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_trace_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        batch_results = {}
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_results[batch_id] = future.result()
            except Exception as e:
                log_message(f"Error processing batch {batch_id}: {e}", log_level="ERROR")
                # Create empty result for failed batch
                batch_results[batch_id] = ([], {"correct": 0, "incorrect": 0, "total": 0, "ungraded": 0}, {"high": [], "medium": [], "low": []})
    
    # Merge results from all batches in order
    trace_summary = []
    accuracy_stats = {"correct": 0, "incorrect": 0, "total": 0, "ungraded": 0}
    confidence_breakdown = {"high": [], "medium": [], "low": []}
    
    for batch_id in sorted(batch_results.keys()):
        batch_summary, batch_stats, batch_confidence = batch_results[batch_id]
        
        # Merge summaries
        trace_summary.extend(batch_summary)
        
        # Merge statistics
        for key in accuracy_stats:
            accuracy_stats[key] += batch_stats.get(key, 0)
        
        # Merge confidence data
        for level in confidence_breakdown:
            confidence_breakdown[level].extend(batch_confidence.get(level, []))
    
    log_message(f"Processed {len(trace_summary)} trace summaries from {num_traces} total traces", log_level="INFO")

    # Calculate accuracy percentage
    accuracy_percentage = (
        (accuracy_stats["correct"] / accuracy_stats["total"] * 100)
        if accuracy_stats["total"] > 0
        else 0
    )

    # Create the prompt for whole trace analysis
    persona = get_expert_persona(specialty)

    # Adapt the analysis approach based on specialty type
    is_scientific = any(
        field in specialty.lower()
        for field in [
            "scientist",
            "biologist",
            "physicist",
            "chemist",
            "geologist",
            "astronomer",
            "mathematician",
            "engineer",
        ]
    )

    prompt = f"""You are a {specialty} conducting a comprehensive meta-analysis of your reasoning performance across multiple questions.

Your persona: {persona}

PERFORMANCE SUMMARY:
- Total questions analyzed: {accuracy_stats["total"]}
- Correct predictions: {accuracy_stats["correct"]} ({accuracy_percentage:.1f}%)
- Incorrect predictions: {accuracy_stats["incorrect"]}
- High confidence decisions: {len(confidence_breakdown["high"])}
- Medium confidence decisions: {len(confidence_breakdown["medium"])}
- Low confidence decisions: {len(confidence_breakdown["low"])}

DETAILED TRACE SUMMARY:
"""

    # Add trace summaries to the prompt
    for trace in trace_summary[:10]:  # Limit to first 10 for brevity
        status = "✓" if trace["was_correct"] else "✗"
        prompt += f"""
Question {trace["question_number"]}: {trace["question_snippet"]}
Prediction: {trace["predicted_answer"]} {status}
Confidence: {trace["confidence_level"]}
Key reasoning: {trace["key_reasoning_points"]}
"""

    if len(trace_summary) > 10:
        prompt += f"\n... and {len(trace_summary) - 10} more questions\n"

    # Customize analysis instructions based on specialty
    if is_scientific:
        prompt += f"""
TASK: As a {specialty}, provide a comprehensive meta-analysis of your reasoning performance. Analyze:

1. METHODOLOGICAL PATTERNS: What reasoning approaches did you consistently use? Were there common analytical frameworks or scientific principles you relied on?

2. ACCURACY PATTERNS: Where were you most/least accurate? What types of questions or concepts challenged your expertise?

3. CONFIDENCE CALIBRATION: How well-calibrated was your confidence? Were you overconfident or underconfident in specific areas?

4. DOMAIN-SPECIFIC INSIGHTS: What does this performance reveal about the intersection of your {specialty} expertise with these questions?

5. SYSTEMATIC BIASES: Did you exhibit any consistent biases or blind spots in your reasoning?

6. LEARNING OPPORTUNITIES: What areas would benefit from deeper investigation or different analytical approaches?

7. METHODOLOGICAL RECOMMENDATIONS: How might your analytical approach be refined for future similar analyses?

Format your response as a detailed scientific analysis with clear sections and evidence-based conclusions.
"""
    else:
        prompt += f"""
TASK: As a {specialty}, provide a comprehensive meta-analysis of your reasoning performance. Analyze:

1. ANALYTICAL PATTERNS: What reasoning approaches did you consistently use? Were there common frameworks or principles you relied on?

2. ACCURACY PATTERNS: Where were you most/least accurate? What types of questions or concepts challenged your expertise?

3. CONFIDENCE CALIBRATION: How well-calibrated was your confidence? Were you overconfident or underconfident in specific areas?

4. DOMAIN-SPECIFIC INSIGHTS: What does this performance reveal about applying your {specialty} perspective to these questions?

5. SYSTEMATIC TENDENCIES: Did you exhibit any consistent patterns or preferences in your reasoning?

6. IMPROVEMENT OPPORTUNITIES: What areas would benefit from deeper investigation or different analytical approaches?

7. STRATEGIC RECOMMENDATIONS: How might your analytical approach be refined for future similar analyses?

Format your response as a detailed professional analysis with clear sections and evidence-based conclusions.
"""

    try:
        # Generate the whole trace analysis
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert {specialty} conducting a reflective meta-analysis of your own reasoning performance. You approach this analysis with the same rigor and expertise you bring to questions in your field. You are honest about both strengths and weaknesses in your reasoning patterns.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Slightly higher temperature for more creative analysis
            max_tokens=2000,
        )

        analysis_text = response.choices[0].message.content.strip()

        # Create structured output
        whole_trace_result = {
            "specialty": specialty,
            "total_questions_analyzed": accuracy_stats["total"],
            "overall_accuracy": accuracy_percentage,
            "performance_breakdown": {
                "correct_predictions": accuracy_stats["correct"],
                "incorrect_predictions": accuracy_stats["incorrect"],
                "confidence_distribution": {
                    "high_confidence": len(confidence_breakdown["high"]),
                    "medium_confidence": len(confidence_breakdown["medium"]),
                    "low_confidence": len(confidence_breakdown["low"]),
                },
            },
            "meta_analysis": analysis_text,
            "detailed_trace_summary": trace_summary,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": model_name,
        }

        log_message("Successfully generated whole trace analysis", log_level="INFO")
        return whole_trace_result

    except Exception as e:
        log_message(f"Error generating whole trace analysis: {e}", log_level="ERROR")
        return {
            "specialty": specialty,
            "total_questions_analyzed": accuracy_stats["total"],
            "overall_accuracy": accuracy_percentage,
            "performance_breakdown": {
                "correct_predictions": accuracy_stats["correct"],
                "incorrect_predictions": accuracy_stats["incorrect"],
            },
            "meta_analysis": f"Error generating analysis: {str(e)}",
            "error": True,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def process_question_wrapper(args_tuple):
    """
    Wrapper function for parallel processing of questions.

    Args:
        args_tuple: Tuple containing (question, client, model_name, specialty,
                   enable_dual_prediction, grading_client, grading_model_name, 
                   verbose_grading, reasoning_mode, method2_model, config_file,
                   method1_analysis_model, method1_synthesis_model)

    Returns:
        Dictionary containing the processed trace and metadata
    """
    try:
        (
            question,
            client,
            model_name,
            specialty,
            enable_dual_prediction,
            grading_client,
            grading_model_name,
            verbose_grading,
            reasoning_mode,
            method2_model,
            config_file,
            method1_analysis_model,
            method1_synthesis_model,
        ) = args_tuple

        # Generate the reasoning trace
        trace = generate_reasoning_trace(
            question,
            client,
            model_name,
            specialty,
            enable_dual_prediction=enable_dual_prediction,
            grading_client=grading_client,
            grading_model_name=grading_model_name,
            verbose_grading=verbose_grading,
            reasoning_mode=reasoning_mode,
            method2_model=method2_model,
            config_file=config_file,
            method1_analysis_model=method1_analysis_model,
            method1_synthesis_model=method1_synthesis_model,
        )

        return {
            "success": True,
            "trace": trace,
            "error": None,
        }

    except Exception as e:
        log_message(f"Error processing question: {e}", log_level="ERROR")
        return {
            "success": False,
            "trace": None,
            "error": str(e),
        }


def process_questions_parallel(
    mc_questions: List[Dict[str, Any]],
    client: OpenAI,
    model_name: str,
    specialty: str,
    enable_dual_prediction: bool,
    grading_client: OpenAI,
    grading_model_name: str,
    verbose_grading: bool,
    reasoning_mode: str = "detailed",
    num_workers: int = 1,
    method2_model: str = None,
    config_file: str = None,
    method1_analysis_model: str = None,
    method1_synthesis_model: str = None,
) -> List[Dict[str, Any]]:
    """
    Process questions in parallel using ThreadPoolExecutor.

    Args:
        mc_questions: List of question dictionaries
        client: OpenAI client for main processing
        model_name: Model name for main processing
        specialty: Specialty persona
        enable_dual_prediction: Whether to enable dual prediction
        grading_client: OpenAI client for grading
        grading_model_name: Model name for grading
        verbose_grading: Whether to show verbose grading output
        num_workers: Number of parallel workers

    Returns:
        List of processed traces
    """
    if num_workers == 1:
        # Sequential processing (original behavior)
        results = []
        for question in tqdm(
            mc_questions, desc=f"Generating {specialty}'s reasoning traces"
        ):
            trace = generate_reasoning_trace(
                question,
                client,
                model_name,
                specialty,
                enable_dual_prediction=enable_dual_prediction,
                grading_client=grading_client,
                grading_model_name=grading_model_name,
                verbose_grading=verbose_grading,
                reasoning_mode=reasoning_mode,
                method2_model=method2_model,
                config_file=config_file,
                method1_analysis_model=method1_analysis_model,
                method1_synthesis_model=method1_synthesis_model,
            )
            results.append(trace)
        return results

    # Parallel processing
    log_message(f"Using {num_workers} parallel workers for question processing")

    # Prepare arguments for parallel processing
    args_list = [
        (
            question,
            client,
            model_name,
            specialty,
            enable_dual_prediction,
            grading_client,
            grading_model_name,
            verbose_grading,
            reasoning_mode,
            method2_model,
            config_file,
            method1_analysis_model,
            method1_synthesis_model,
        )
        for question in mc_questions
    ]

    results = []
    failed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_question_wrapper, args): i
            for i, args in enumerate(args_list)
        }

        # Collect results with progress tracking
        with tqdm(
            total=len(mc_questions),
            desc=f"Generating {specialty}'s reasoning traces (parallel)",
        ) as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result["success"]:
                        results.append((index, result["trace"]))
                    else:
                        failed_count += 1
                        log_message(
                            f"Failed to process question {index}: {result['error']}",
                            log_level="ERROR",
                        )
                        # Add empty result to maintain order
                        results.append((index, None))
                except Exception as e:
                    failed_count += 1
                    log_message(
                        f"Exception processing question {index}: {e}", log_level="ERROR"
                    )
                    results.append((index, None))

                pbar.update(1)
                pbar.set_postfix({"Failed": failed_count})

    # Sort results by original order and filter out None values
    results.sort(key=lambda x: x[0])
    final_results = [trace for _, trace in results if trace is not None]

    if failed_count > 0:
        log_message(
            f"Failed to process {failed_count} out of {len(mc_questions)} questions",
            log_level="WARNING",
        )

    log_message(
        f"Successfully processed {len(final_results)} out of {len(mc_questions)} questions using {num_workers} workers"
    )
    return final_results


def print_whole_trace_analysis(analysis: Dict[str, Any]):
    """
    Print a human-readable version of the whole trace analysis.

    Args:
        analysis: The whole trace analysis dictionary
    """
    print("\n" + "=" * 100)
    print("🧠 WHOLE TRACE META-ANALYSIS")
    print("=" * 100)

    specialty = (analysis.get("specialty", "expert") or "expert").upper()
    print(f"Perspective: {specialty}")
    print(f"Generated at: {analysis.get('generated_at', 'Unknown')}")
    print(f"Questions analyzed: {analysis.get('total_questions_analyzed', 0)}")
    print(f"Overall accuracy: {analysis.get('overall_accuracy', 0):.1f}%")

    # Performance breakdown
    breakdown = analysis.get("performance_breakdown", {})
    if breakdown:
        print(f"\nPerformance Summary:")
        print(f"  • Correct predictions: {breakdown.get('correct_predictions', 0)}")
        print(f"  • Incorrect predictions: {breakdown.get('incorrect_predictions', 0)}")

        conf_dist = breakdown.get("confidence_distribution", {})
        if conf_dist:
            print(
                f"  • High confidence decisions: {conf_dist.get('high_confidence', 0)}"
            )
            print(
                f"  • Medium confidence decisions: {conf_dist.get('medium_confidence', 0)}"
            )
            print(f"  • Low confidence decisions: {conf_dist.get('low_confidence', 0)}")

    print("-" * 100)
    print("📊 DETAILED META-ANALYSIS:")
    print("-" * 100)

    meta_analysis = analysis.get("meta_analysis", "No analysis available")

    if analysis.get("error"):
        print("❌ Error occurred during analysis generation:")
        print(meta_analysis)
    else:
        # Format the meta-analysis with proper indentation
        formatted_analysis = "\n".join(line for line in meta_analysis.split("\n"))
        print(formatted_analysis)

    print("=" * 100)


def create_stream_analysis_files(
    results: List[Dict[str, Any]], 
    mc_questions: List[Dict[str, Any]], 
    output_dir: str, 
    specialty: str = "expert",
    client: OpenAI = None,
    model_name: str = None
) -> None:
    """Create individual stream analysis files for each question.
    
    Args:
        results: List of reasoning trace results
        mc_questions: List of original question data
        output_dir: Directory to output individual files
        specialty: Expert specialty persona
        client: OpenAI client for generating analysis
        model_name: Model name for generating analysis
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    log_message(f"Creating individual stream analysis files for {len(results)} questions in {output_dir}")
    
    for i, (result, question_data) in enumerate(zip(results, mc_questions)):
        try:
            # Generate filename based on question
            question_text = question_data.get('question', f'question_{i+1}')
            safe_filename = "".join(c if c.isalnum() else '_' for c in question_text[:50]).strip('_')
            safe_filename = '_'.join(filter(None, safe_filename.split('_')))
            if not safe_filename:
                safe_filename = f"question_{i+1}"
            
            stream_filename = f"{i+1:03d}_{safe_filename}_STREAM_ANALYSIS.txt"
            stream_file_path = os.path.join(output_dir, stream_filename)
            
            # Generate the coherent stream analysis
            if client and model_name:
                log_message(f"Generating stream analysis for question {i+1}: {question_text[:50]}...")
                stream_analysis = generate_coherent_stream_analysis(
                    result, specialty, client, model_name
                )
            else:
                stream_analysis = "Stream analysis not available - no model specified for synthesis."
            
            # Write the stream analysis to file
            with open(stream_file_path, 'w', encoding='utf-8') as f:
                f.write(f"🌊 COHERENT STREAM OF THOUGHT ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"QUESTION {i+1}:\n")
                f.write(f"{question_data.get('question', 'Unknown question')}\n\n")
                f.write("STREAM OF THOUGHT:\n")
                f.write("-" * 40 + "\n")
                f.write(stream_analysis)
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated by: {model_name or 'Unknown model'}\n")
                f.write(f"Specialty: {specialty}\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if i < 3:  # Log first few for confirmation
                log_message(f"Created stream analysis file: {stream_filename}")
                
        except Exception as e:
            log_message(f"Error creating stream analysis file for question {i+1}: {e}", log_level="ERROR")
    
    log_message(f"Successfully created stream analysis files in {output_dir}")


def split_output_file(input_file: str, output_dir: str = None) -> None:
    """Split a JSON output file into human-readable text files for each question section.
    
    Args:
        input_file: Path to the JSON file containing reasoning traces
        output_dir: Directory to output individual files (optional)
    """
    import os
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"{input_file}_split"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data is a list of results
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and 'results' in data:
            results = data['results']
        else:
            log_message(f"Unknown JSON structure in {input_file}", log_level="ERROR")
            return
            
        log_message(f"Splitting {len(results)} questions into text files in {output_dir}")
        
        # Write each question to separate text files
        for i, result in enumerate(results):
            # Generate base filename based on question or index
            question_text = result.get('question', f'question_{i+1}')
            # Clean filename - replace spaces with underscores, keep only alphanumeric and underscores
            safe_filename = "".join(c if c.isalnum() else '_' for c in question_text[:50]).strip('_')
            # Remove multiple consecutive underscores
            safe_filename = '_'.join(filter(None, safe_filename.split('_')))
            if not safe_filename:
                safe_filename = f"question_{i+1}"
            
            base_filename = f"{i+1:03d}_{safe_filename}"
            
            # Create the main question file with all sections
            main_file = os.path.join(output_dir, f"{base_filename}.txt")
            with open(main_file, 'w', encoding='utf-8') as f:
                # Write the formatted question output using existing function
                write_question_sections(f, result, i+1)
            
            # Also create separate files for major sections
            create_section_files(output_dir, base_filename, result)
                
        log_message(f"Successfully split into {len(results)} text files in {output_dir}")
        
    except Exception as e:
        log_message(f"Error splitting file: {e}", log_level="ERROR")
        sys.exit(1)


def write_question_sections(file_handle, reasoning_trace: dict, question_num: int):
    """Write all sections of a question to a file handle."""
    
    # Header
    file_handle.write("=" * 80 + "\n")
    file_handle.write(f"QUESTION {question_num}\n")
    file_handle.write("=" * 80 + "\n\n")
    
    # Question and options
    question_text = reasoning_trace.get("question", "No question available")
    file_handle.write(f"QUESTION: {question_text}\n")
    file_handle.write("-" * 80 + "\n\n")
    
    options = reasoning_trace.get("options", [])
    correct_index = reasoning_trace.get("correct_answer_index", -1)
    for i, option in enumerate(options):
        marker = "✓ " if i == correct_index else "  "
        file_handle.write(f"{marker}{i+1}. {option}\n")
    
    if correct_index >= 0 and correct_index < len(options):
        file_handle.write(f"\nCORRECT ANSWER: {options[correct_index]}\n")
    
    file_handle.write("\n" + "=" * 80 + "\n")
    file_handle.write("METHOD 1: TWO-STAGE REASONING\n")
    file_handle.write("=" * 80 + "\n\n")
    
    # Method 1 reasoning
    reasoning = reasoning_trace.get("reasoning", {})
    
    # Method 1 prediction
    prediction = reasoning.get("prediction", {})
    file_handle.write(f"PREDICTED ANSWER: {prediction.get('predicted_answer', 'Unknown')}\n")
    file_handle.write(f"CONFIDENCE: {prediction.get('confidence_level', 'Unknown')}\n")
    file_handle.write(f"REASONING: {prediction.get('prediction_reasoning', 'No reasoning available')}\n\n")
    
    # Detailed thought process for each option
    thought_process = reasoning.get("thought_process", {})
    if thought_process:
        file_handle.write("DETAILED THOUGHT PROCESS FOR EACH OPTION:\n")
        file_handle.write("-" * 40 + "\n")
        
        # Sort option keys numerically
        option_keys = sorted(
            [k for k in thought_process.keys() if k.startswith("option_")],
            key=lambda x: (
                int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else float("inf")
            ),
        )
        
        for opt_key in option_keys:
            try:
                opt_idx = int(opt_key.split("_")[-1])
                if opt_idx <= len(options):
                    option_text = options[opt_idx-1] if opt_idx-1 < len(options) else f"Option {opt_idx}"
                    file_handle.write(f"\nOption {opt_idx}: {option_text}\n")
                    file_handle.write("~" * 40 + "\n")
                    
                    thoughts = thought_process[opt_key]
                    if isinstance(thoughts, str):
                        # Format the thought process text with indentation
                        for line in thoughts.split('\n'):
                            file_handle.write(f"  {line}\n")
                    else:
                        file_handle.write(f"  {str(thoughts)}\n")
                    file_handle.write("\n")
            except (ValueError, IndexError):
                continue
        file_handle.write("\n")
    
    # Two-stage details if available
    if reasoning.get("two_stage_approach"):
        file_handle.write("STAGE 1 - INDIVIDUAL OPTION ANALYSIS:\n")
        file_handle.write("-" * 40 + "\n")
        question_restatement = reasoning.get("question_restatement", "Not available")
        file_handle.write(f"Question Restatement: {question_restatement}\n\n")
        
        file_handle.write("STAGE 2 - SYNTHESIS:\n")
        file_handle.write("-" * 40 + "\n")
        synthesis_reasoning = reasoning.get("synthesis_reasoning", "Not available")
        decision_rationale = reasoning.get("decision_rationale", "Not available")
        file_handle.write(f"Synthesis Reasoning: {synthesis_reasoning}\n")
        file_handle.write(f"Decision Rationale: {decision_rationale}\n\n")
    
    # Scientific conclusion
    conclusion = reasoning.get("scientific_conclusion", reasoning.get("conclusion", "No conclusion available"))
    file_handle.write("SCIENTIFIC CONCLUSION:\n")
    file_handle.write("-" * 40 + "\n")
    file_handle.write(f"{conclusion}\n\n")
    
    # Dual prediction if available
    dual_prediction = reasoning_trace.get("dual_prediction")
    if dual_prediction:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("METHOD 2: ARGONIUM-STYLE PREDICTION\n")
        file_handle.write("=" * 80 + "\n\n")
        
        argonium_pred = dual_prediction.get("argonium_prediction", {})
        file_handle.write(f"PREDICTED ANSWER: {argonium_pred.get('predicted_answer', 'Unknown')}\n")
        file_handle.write("RESPONSE:\n")
        argonium_response = argonium_pred.get("raw_response", "No response available")
        for line in argonium_response.split("\n"):
            file_handle.write(f"   {line}\n")
        
        file_handle.write("\n" + "-" * 80 + "\n")
        file_handle.write("COMPARISON ANALYSIS:\n")
        file_handle.write("-" * 80 + "\n")
        comparison = dual_prediction.get("comparison_analysis", "No comparison available")
        file_handle.write(f"{comparison}\n\n")
    
    # Disagreement analysis if available
    disagreement = reasoning_trace.get("disagreement_analysis")
    if disagreement and disagreement.get("has_disagreement"):
        file_handle.write("=" * 80 + "\n")
        file_handle.write("DISAGREEMENT ANALYSIS\n")  
        file_handle.write("=" * 80 + "\n\n")
        
        diagnosis = disagreement.get("diagnosis", {})
        file_handle.write(f"DISAGREEMENT DETECTED: {disagreement.get('has_disagreement', False)}\n")
        file_handle.write(f"DIAGNOSIS: {diagnosis.get('disagreement_explanation', 'No explanation')}\n")
        file_handle.write(f"LIKELY CORRECT METHOD: {diagnosis.get('likely_correct_method', 'Unknown')}\n")
        
        regeneration = reasoning_trace.get("regeneration_result")
        if regeneration and regeneration.get("regeneration_performed"):
            file_handle.write("\nREGENERATION PERFORMED: Yes\n")
        file_handle.write("\n")
    
    # Grading results
    if reasoning_trace.get("prediction_correct") is not None:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("GRADING RESULTS\n")
        file_handle.write("=" * 80 + "\n\n")
        
        is_correct = reasoning_trace.get("prediction_correct", False)
        file_handle.write(f"PREDICTION CORRECT: {'✓ YES' if is_correct else '✗ NO'}\n")
        
        grading_result = reasoning_trace.get("grading_result", {})
        if grading_result:
            confidence = grading_result.get("confidence", "Unknown")
            file_handle.write(f"GRADING CONFIDENCE: {confidence}\n")
            reasoning_grade = grading_result.get("reasoning", "No reasoning provided")
            file_handle.write(f"GRADING REASONING: {reasoning_grade}\n")
        file_handle.write("\n")


def create_section_files(output_dir: str, base_filename: str, result: dict):
    """Create separate files for major sections."""
    
    reasoning = result.get("reasoning", {})
    
    # Method 1 reasoning section
    method1_file = os.path.join(output_dir, f"{base_filename}_method1.txt") 
    with open(method1_file, 'w', encoding='utf-8') as f:
        f.write("METHOD 1: TWO-STAGE REASONING\n")
        f.write("=" * 40 + "\n\n")
        
        prediction = reasoning.get("prediction", {})
        f.write(f"Predicted Answer: {prediction.get('predicted_answer', 'Unknown')}\n")
        f.write(f"Confidence: {prediction.get('confidence_level', 'Unknown')}\n\n")
        f.write(f"Reasoning: {prediction.get('prediction_reasoning', 'No reasoning available')}\n\n")
        
        # Include detailed thought process for each option
        thought_process = reasoning.get("thought_process", {})
        if thought_process:
            f.write("DETAILED THOUGHT PROCESS FOR EACH OPTION:\n")
            f.write("-" * 40 + "\n")
            
            # Sort option keys numerically
            option_keys = sorted(
                [k for k in thought_process.keys() if k.startswith("option_")],
                key=lambda x: (
                    int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else float("inf")
                ),
            )
            
            options = result.get("options", [])
            for opt_key in option_keys:
                try:
                    opt_idx = int(opt_key.split("_")[-1])
                    if opt_idx <= len(options):
                        option_text = options[opt_idx-1] if opt_idx-1 < len(options) else f"Option {opt_idx}"
                        f.write(f"\nOption {opt_idx}: {option_text}\n")
                        f.write("~" * 40 + "\n")
                        
                        thoughts = thought_process[opt_key]
                        if isinstance(thoughts, str):
                            for line in thoughts.split('\n'):
                                f.write(f"  {line}\n")
                        else:
                            f.write(f"  {str(thoughts)}\n")
                        f.write("\n")
                except (ValueError, IndexError):
                    continue
            f.write("\n")
        
        conclusion = reasoning.get("scientific_conclusion", reasoning.get("conclusion", "No conclusion available"))
        f.write(f"Scientific Conclusion:\n{conclusion}\n")
    
    # Method 2 (Argonium) section if available
    dual_prediction = result.get("dual_prediction")
    if dual_prediction:
        method2_file = os.path.join(output_dir, f"{base_filename}_method2.txt")
        with open(method2_file, 'w', encoding='utf-8') as f:
            f.write("METHOD 2: ARGONIUM-STYLE PREDICTION\n")
            f.write("=" * 40 + "\n\n")
            
            argonium_pred = dual_prediction.get("argonium_prediction", {})
            f.write(f"Predicted Answer: {argonium_pred.get('predicted_answer', 'Unknown')}\n\n")
            f.write("Response:\n")
            argonium_response = argonium_pred.get("raw_response", "No response available")
            f.write(f"{argonium_response}\n")
        
        # Comparison analysis section
        comparison_file = os.path.join(output_dir, f"{base_filename}_comparison.txt")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("COMPARISON ANALYSIS\n")
            f.write("=" * 40 + "\n\n")
            comparison = dual_prediction.get("comparison_analysis", "No comparison available")
            f.write(f"{comparison}\n")
    
    # Enhanced fault-oriented disagreement analysis if available
    comprehensive_analysis = result.get("comprehensive_disagreement_analysis")
    if comprehensive_analysis and comprehensive_analysis.get("analysis_successful"):
        disagreement_file = os.path.join(output_dir, f"{base_filename}_fault_analysis.txt")
        with open(disagreement_file, 'w', encoding='utf-8') as f:
            f.write("FAULT-ORIENTED DISAGREEMENT ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            fault_analysis = comprehensive_analysis.get("fault_analysis", {})
            
            # 1. Answer Comparison Section
            answer_comp = fault_analysis.get("answer_comparison", {})
            f.write("1. ANSWER COMPARISON\n")
            f.write("-" * 30 + "\n")
            f.write(f"Correct Answer: {answer_comp.get('correct_answer', 'Unknown')}\n")
            f.write(f"Method 1 Prediction: {answer_comp.get('method1_prediction', 'Unknown')}\n")
            if answer_comp.get('method2_prediction'):
                f.write(f"Method 2 Prediction: {answer_comp.get('method2_prediction', 'Unknown')}\n")
            f.write(f"Has Disagreement: {answer_comp.get('has_disagreement', 'Unknown')}\n")
            f.write(f"Disagreement Type: {answer_comp.get('disagreement_type', 'Unknown')}\n\n")
            
            # 2. Diagnostic Analysis Section  
            diagnostic = fault_analysis.get("diagnostic_analysis", {})
            f.write("2. DIAGNOSTIC ANALYSIS & FAULT ASSIGNMENT\n")
            f.write("-" * 45 + "\n")
            primary_fault = diagnostic.get('primary_fault_category', 'Unknown') or 'Unknown'
            f.write(f"🎯 PRIMARY FAULT: {primary_fault.upper()}\n")
            fault_confidence = diagnostic.get('fault_confidence', 'Unknown') or 'Unknown'
            fault_description = diagnostic.get('specific_fault_description', 'N/A') or 'N/A'
            f.write(f"Fault Confidence: {fault_confidence}\n")
            f.write(f"Fault Description: {fault_description}\n")
            
            evidence = diagnostic.get('fault_evidence', [])
            if evidence:
                f.write("\nFault Evidence:\n")
                for item in evidence:
                    f.write(f"  • {item}\n")
            
            contributing = diagnostic.get('secondary_contributing_factors', [])
            if contributing:
                f.write("\nSecondary Contributing Factors:\n")
                for factor in contributing:
                    f.write(f"  • {factor}\n")
            f.write("\n")
            
            # 3. External Knowledge Analysis Section
            knowledge = fault_analysis.get("external_knowledge_analysis", {})
            f.write("3. EXTERNAL KNOWLEDGE REQUIREMENTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Requires External Knowledge: {knowledge.get('requires_external_knowledge', 'Unknown')}\n")
            if knowledge.get('requires_external_knowledge'):
                f.write(f"Knowledge Type: {knowledge.get('knowledge_type', 'Unknown')}\n")
                f.write(f"Knowledge Specificity: {knowledge.get('knowledge_specificity', 'Unknown')}\n")
                f.write(f"Likely Source: {knowledge.get('likely_knowledge_source', 'Unknown')}\n")
                f.write(f"\n📚 Missing Knowledge Summary:\n{knowledge.get('missing_knowledge_summary', 'N/A')}\n")
            f.write("\n")
            
            # 4. Reasoning Quality Assessment
            reasoning_qual = fault_analysis.get("reasoning_quality_assessment", {})
            f.write("4. REASONING QUALITY ASSESSMENT\n")
            f.write("-" * 35 + "\n")
            f.write(f"Method 1 Reasoning Validity: {reasoning_qual.get('method1_reasoning_validity', 'Unknown')}\n")
            
            method1_issues = reasoning_qual.get('method1_logic_issues', [])
            if method1_issues:
                f.write("Method 1 Logic Issues:\n")
                for issue in method1_issues:
                    f.write(f"  • {issue}\n")
            
            if reasoning_qual.get('method2_reasoning_validity', 'not_applicable') != 'not_applicable':
                f.write(f"Method 2 Reasoning Validity: {reasoning_qual.get('method2_reasoning_validity', 'Unknown')}\n")
                method2_issues = reasoning_qual.get('method2_logic_issues', [])
                if method2_issues:
                    f.write("Method 2 Logic Issues:\n")
                    for issue in method2_issues:
                        f.write(f"  • {issue}\n")
            
            f.write(f"Better Reasoning Approach: {reasoning_qual.get('better_reasoning_approach', 'Unknown')}\n\n")
            
            # 5. Root Cause Analysis
            root_cause = fault_analysis.get("root_cause_analysis", {})
            f.write("5. ROOT CAUSE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"🔍 Primary Root Cause: {root_cause.get('primary_root_cause', 'Unknown')}\n")
            f.write(f"Problem Severity: {root_cause.get('problem_severity', 'Unknown')}\n")
            f.write(f"Problem Scope: {root_cause.get('problem_scope', 'Unknown')}\n")
            
            contributing_factors = root_cause.get('contributing_factors', [])
            if contributing_factors:
                f.write("\nContributing Factors:\n")
                for factor in contributing_factors:
                    f.write(f"  • {factor}\n")
            f.write("\n")
            
            # 6. Actionable Recommendations
            recommendations = fault_analysis.get("actionable_recommendations", {})
            f.write("6. ACTIONABLE RECOMMENDATIONS\n")
            f.write("-" * 35 + "\n")
            f.write(f"🚀 Immediate Fix: {recommendations.get('immediate_fix', 'N/A')}\n")
            f.write(f"Prevention Strategy: {recommendations.get('prevention_strategy', 'N/A')}\n")
            f.write(f"MCQA Improvement: {recommendations.get('mcqa_improvement', 'N/A')}\n")
            f.write(f"Text Requirements: {recommendations.get('text_requirements', 'N/A')}\n")
            f.write(f"Scoring Adjustment: {recommendations.get('scoring_adjustment', 'N/A')}\n\n")
            
            # 7. Confidence Assessment
            confidence = fault_analysis.get("confidence_assessment", {})
            f.write("7. CONFIDENCE ASSESSMENT\n")
            f.write("-" * 30 + "\n")
            f.write(f"Overall Analysis Confidence: {confidence.get('overall_analysis_confidence', 'Unknown')}\n")
            f.write(f"Most Certain Finding: {confidence.get('most_certain_finding', 'N/A')}\n")
            f.write(f"Least Certain Finding: {confidence.get('least_certain_finding', 'N/A')}\n")
            
            additional_analysis = confidence.get('additional_analysis_needed', [])
            if additional_analysis:
                f.write("\nAdditional Analysis Needed:\n")
                for item in additional_analysis:
                    f.write(f"  • {item}\n")
            
            # Regeneration details if available
            regeneration = result.get("regeneration_result")
            if regeneration:
                f.write("\n" + "=" * 60 + "\n")
                f.write("REGENERATION DETAILS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Regeneration Performed: {regeneration.get('regeneration_performed', False)}\n")
                f.write(f"Reason: {regeneration.get('reason', 'Not specified')}\n")


def main():
    """Main entry point function."""
    global _total_questions, _processed_questions

    # Parse command-line arguments
    args = parse_arguments()

    # Handle split-only mode
    if args.split_only:
        split_output_file(args.input_file, args.output_dir)
        
        # Also create stream analysis files if requested
        if args.create_stream_files:
            # We need to configure the API and load the data for stream analysis
            client, model_name = configure_apis(args.model, args.config)
            
            # Load the results file to extract questions and results
            try:
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract results
                if isinstance(data, list):
                    results = data
                elif isinstance(data, dict) and 'results' in data:
                    results = data['results']
                else:
                    log_message("Cannot create stream analysis files - unknown data format", log_level="ERROR")
                    return
                
                # Extract questions from results (they should contain the original question data)
                mc_questions = []
                for result in results:
                    question_data = {
                        'question': result.get('question', 'Unknown question'),
                        'options': result.get('options', []),
                        'correct_answer': result.get('correct_answer', 'Unknown')
                    }
                    mc_questions.append(question_data)
                
                # Create stream analysis files
                output_dir = args.output_dir or f"{args.input_file}_stream_analysis"
                create_stream_analysis_files(
                    results, 
                    mc_questions, 
                    output_dir, 
                    args.specialty,
                    client, 
                    model_name
                )
                
            except Exception as e:
                log_message(f"Error creating stream analysis files in split-only mode: {e}", log_level="ERROR")
        
        return

    # Configure the OpenAI API for the selected model
    client, model_name = configure_apis(args.model, args.config)

    # Configure the whole trace analysis model (if different)
    whole_trace_client = client
    whole_trace_model_name = model_name
    if args.whole_trace_model:
        whole_trace_client, whole_trace_model_name = configure_apis(
            args.whole_trace_model, args.config
        )
        log_message(
            f"Using different model for whole trace analysis: {args.whole_trace_model}"
        )

    # Configure the grading model (if different)
    grading_client = client
    grading_model_name = model_name
    if args.grading:
        grading_client, grading_model_name = configure_apis(args.grading, args.config)
        log_message(f"Using different model for grading: {args.grading}")

    # Configure the comparison model (if different)
    comparison_client = client
    comparison_model_name = model_name
    if args.comparison_model:
        comparison_client, comparison_model_name = configure_apis(args.comparison_model, args.config)
        log_message(f"Using different model for comparison analysis: {args.comparison_model}")

    # Configure the text analysis model (if different) 
    text_analysis_client = client
    text_analysis_model_name = model_name
    if args.text_analysis_model:
        text_analysis_client, text_analysis_model_name = configure_apis(args.text_analysis_model, args.config)
        log_message(f"Using different model for text analysis: {args.text_analysis_model}")

    # Check grading model requirements
    if args.require_grading_model and not args.grading:
        log_message(
            "Error: --require-grading-model specified but no --grading provided",
            log_level="ERROR",
        )
        log_message(
            "Please specify a grading model with --grading when using --require-grading-model",
            log_level="ERROR",
        )
        sys.exit(1)

    # Inform user about scoring method
    if args.grading:
        log_message(
            f"Answer verification will be performed using grading model: {args.grading}"
        )
    else:
        log_message(
            "Warning: No grading model specified. Answer correctness cannot be determined."
        )
        log_message("Use --grading argument to enable answer verification.")
        if args.require_grading_model:
            log_message(
                "Error: Grading model is required but not specified", log_level="ERROR"
            )
            sys.exit(1)

    # Load argonium results for comparison if provided
    argonium_data = None
    argonium_results_map = {}
    argonium_text_map = {}
    if args.argonium_results:
        log_message(f"Attempting to load Argonium results from: {args.argonium_results}")
        argonium_data = load_argonium_results(args.argonium_results)
        log_message(f"Argonium data loaded: {argonium_data is not None}")
        if argonium_data:
            # Create a map of question_id to results for easier lookup
            # Also create a text-based lookup for more robust matching
            for result in argonium_data.get('results', []):
                question_id = result.get('question_id')
                if question_id:
                    argonium_results_map[question_id] = result
                
                # Also index by question text (first 100 chars for matching)
                question_text = result.get('question', '').strip()
                if question_text:
                    text_key = question_text[:100].strip()
                    argonium_text_map[text_key] = result
            log_message(f"Mapped {len(argonium_results_map)} argonium results for comparison")
            # Debug: Show the first few question IDs in the map
            question_ids = list(argonium_results_map.keys())[:5]
            log_message(f"First few argonium question IDs: {question_ids}")
        else:
            log_message("Failed to load argonium results file", log_level="ERROR")

    # Initialize results list
    results = []

    # Check if we're continuing from a previous run
    starting_index = 0
    if args.continue_from and os.path.exists(args.continue_from):
        try:
            with open(args.continue_from, "r", encoding="utf-8") as f:
                results = json.load(f)
                starting_index = len(results)
                log_message(
                    f"Continuing from previous run - {starting_index} questions already processed",
                    log_level="INFO",
                )
        except Exception as e:
            log_message(f"Error reading continue-from file: {e}", log_level="ERROR")
            log_message("Starting from scratch", log_level="INFO")

    # Read the input file
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
    except Exception as e:
        log_message(f"Error reading input file: {e}", log_level="ERROR")
        sys.exit(1)

    # Filter for multiple choice questions only
    # Process all questions like argonium_score_parallel (no filtering by type)
    mc_questions = questions_data

    if not mc_questions:
        log_message(
            "No multiple choice questions found in the input file.", log_level="ERROR"
        )
        sys.exit(1)

    # Apply maximum questions limit if specified
    if args.max_questions is not None and args.max_questions > 0:
        mc_questions = mc_questions[: args.max_questions]

    # Skip questions we've already processed if continuing
    if starting_index > 0:
        mc_questions = mc_questions[starting_index:]

    _total_questions = len(mc_questions) + starting_index
    _processed_questions = starting_index

    log_message(
        f"Found {len(mc_questions)} multiple choice questions to process. Starting processing..."
    )
    log_message(f"Using {args.specialty} persona for scientific reasoning")

    # Generate reasoning traces for each question (parallel or sequential)
    results = process_questions_parallel(
        mc_questions,
        client,
        model_name,
        args.specialty,
        args.dual_prediction,
        grading_client,
        grading_model_name,
        args.verbose_grading,
        args.reasoning_mode,
        args.workers,
        args.method2_model,
        args.config,
        args.method1_analysis_model,
        args.method1_synthesis_model,
    )

    # Add argonium file comparison if available
    if argonium_results_map:
        log_message(f"Starting argonium comparison with {len(results)} results, starting_index={starting_index}")
        available_keys = list(argonium_results_map.keys())[:10]  # First 10 keys
        log_message(f"Available argonium keys: {available_keys}")
        
        matches_found = 0
        for i, trace in enumerate(results):
            # Use the actual question index from the trace, or fall back to position-based indexing
            question_index = trace.get('question_index', starting_index + i + 1)
            if i < 5:  # Debug first few
                log_message(f"Looking for question index {question_index} (i={i}, trace question_index={trace.get('question_index')}) in argonium map")
            # Try to match by question index first
            argonium_result = None
            match_method = None
            
            if question_index in argonium_results_map:
                argonium_result = argonium_results_map[question_index]
                match_method = "index"
            else:
                # Try to match by question text
                if i < len(mc_questions):
                    current_question_text = mc_questions[i].get('question', '').strip()
                    text_key = current_question_text[:100].strip()
                    if text_key in argonium_text_map:
                        argonium_result = argonium_text_map[text_key]
                        match_method = "text"
            
            if argonium_result:
                matches_found += 1
                if i < 5:  # Debug first few
                    log_message(f"Found match for question {i+1} using {match_method} method")
                
                # Verify question match using grading model
                if grading_client and grading_model_name:
                    question_match = verify_question_match(
                        mc_questions[i].get("question", ""),
                        argonium_result.get("question", ""),
                        grading_client,
                        grading_model_name
                    )
                    trace["argonium_file_comparison"] = {
                        "question_match": question_match,
                        "argonium_result": argonium_result,
                        "argonium_file_path": args.argonium_results
                    }
                    
                    if not question_match.get("is_match", False):
                        log_message(f"Warning: Question {question_index} doesn't match argonium file question", log_level="WARNING")
                    else:
                        log_message(f"Question {question_index} verified to match argonium file", log_level="DEBUG")
                else:
                    # Store without verification if no grading model
                    trace["argonium_file_comparison"] = {
                        "question_match": {"is_match": None, "reasoning": "No grading model available for verification"},
                        "argonium_result": argonium_result,
                        "argonium_file_path": args.argonium_results
                    }
        
        log_message(f"Argonium comparison complete: {matches_found} matches found out of {len(results)} questions")

    # Print readable output to console for each result
    for i, (question, trace) in enumerate(zip(mc_questions, results)):
        print_readable_output(
            question,
            trace,
            args.specialty,
            show_stream_analysis=args.whole_trace_analysis,
            client=client,
        )

        # Save intermediate results at specified intervals
        current_index = starting_index + i + 1
        if (current_index % args.save_interval == 0) or (i == len(results) - 1):
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                log_message(
                    f"Saved intermediate results to {args.output} after processing {current_index} questions",
                    log_level="INFO",
                )
            except Exception as e:
                log_message(
                    f"Error saving intermediate results: {e}", log_level="ERROR"
                )

    # Save final results
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log_message(f"Successfully wrote reasoning traces to {args.output}")
        
        # Split output if requested
        if args.split_output:
            split_output_file(args.output, args.output_dir)
        
        # Create individual stream analysis files if requested
        if args.create_stream_files:
            output_dir = args.output_dir or f"{args.output}_stream_analysis"
            # Use the main client and model for stream analysis
            create_stream_analysis_files(
                results, 
                mc_questions, 
                output_dir, 
                args.specialty,
                client, 
                model_name
            )
            
    except Exception as e:
        log_message(f"Error writing output file: {e}", log_level="ERROR")

    # Capture incorrect answers if requested
    if args.capture_incorrect:
        try:
            incorrect_answers = []
            for trace in results:
                # Only include answers that were explicitly marked as incorrect
                if trace.get("prediction_correct") is False:
                    predicted_answer = (
                        trace.get("reasoning", {})
                        .get("prediction", {})
                        .get("predicted_answer", "")
                    )
                    correct_answer = trace.get("correct_answer", "")

                    incorrect_entry = {
                        "question": trace.get("question", ""),
                        "options": trace.get("options", []),
                        "predicted_answer": predicted_answer,
                        "correct_answer": correct_answer,
                        "correct_answer_index": trace.get("correct_answer_index", -1),
                        "grading_result": trace.get("grading_result", {}),
                        "reasoning_summary": trace.get("reasoning", {})
                        .get("prediction", {})
                        .get("reasoning", ""),
                        "confidence_level": trace.get("reasoning", {})
                        .get("prediction", {})
                        .get("confidence_level", ""),
                    }
                    incorrect_answers.append(incorrect_entry)

            # Save incorrect answers to file
            incorrect_data = {
                "timestamp": time.time(),
                "total_incorrect": len(incorrect_answers),
                "grading_model_used": (
                    grading_model_name if grading_model_name else "None"
                ),
                "incorrect_answers": incorrect_answers,
            }

            with open(args.capture_incorrect, "w", encoding="utf-8") as f:
                json.dump(incorrect_data, f, indent=2)

            log_message(
                f"Captured {len(incorrect_answers)} incorrect answers to {args.capture_incorrect}"
            )

        except Exception as e:
            log_message(f"Error capturing incorrect answers: {e}", log_level="ERROR")

    # Calculate accuracy statistics
    # Only count predictions that were actually graded (exclude None values)
    graded_predictions = [
        trace for trace in results if trace.get("prediction_correct") is not None
    ]
    correct_predictions = sum(
        1 for trace in graded_predictions if trace.get("prediction_correct", False)
    )
    ungraded_predictions = sum(
        1 for trace in results if trace.get("prediction_correct") is None
    )

    total_predictions = len(results)
    total_graded = len(graded_predictions)

    if total_graded > 0:
        accuracy_percentage = (correct_predictions / total_graded) * 100
    else:
        accuracy_percentage = 0

    # Count by confidence level if available
    high_confidence_correct = 0
    high_confidence_total = 0
    medium_confidence_correct = 0
    medium_confidence_total = 0
    low_confidence_correct = 0
    low_confidence_total = 0

    for trace in results:
        if trace.get("prediction_correct") is None:
            continue

        confidence = (
            trace.get("reasoning", {})
            .get("prediction", {})
            .get("confidence_level", "")
            .lower()
        )
        is_correct = trace.get("prediction_correct", False)

        if confidence == "high":
            high_confidence_total += 1
            if is_correct:
                high_confidence_correct += 1
        elif confidence == "medium":
            medium_confidence_total += 1
            if is_correct:
                medium_confidence_correct += 1
        elif confidence == "low":
            low_confidence_total += 1
            if is_correct:
                low_confidence_correct += 1

    # Print final summary
    elapsed_time = time.time() - _start_time
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.1f} seconds"
    elif elapsed_time < 3600:
        time_str = f"{elapsed_time/60:.1f} minutes"
    else:
        time_str = f"{elapsed_time/3600:.1f} hours"

    log_message(
        f"Processing complete! Generated {args.specialty}'s reasoning traces for {_processed_questions} questions in {time_str}."
    )
    log_message(f"Results saved to {args.output}")

    # Print accuracy summary
    print("\n" + "=" * 80)
    print("📈 PREDICTION ACCURACY SUMMARY")
    print("=" * 80)

    # Check if grading model was used
    grading_model_used = any(trace.get("grading_result") for trace in results)
    
    # Check if dual prediction (Argonium-style) was used
    dual_prediction_used = any(trace.get("dual_prediction") for trace in results)

    print(f"Total questions processed: {total_predictions}")
    print(f"Questions with grading: {total_graded}")
    if ungraded_predictions > 0:
        print(f"Questions without grading: {ungraded_predictions} (accuracy unknown)")

    if total_graded > 0:
        print(f"\n🔍 REASONING TRACES METHOD ACCURACY:")
        print(
            f"Overall accuracy: {correct_predictions}/{total_graded} correct predictions ({accuracy_percentage:.1f}%)"
        )
        print(f"Verification method: grading model")
        print(f"Grading model: {grading_model_name}")
        
        # Analyze dual prediction accuracy if available
        if dual_prediction_used:
            print(f"\n🎯 ARGONIUM-STYLE METHOD ACCURACY:")
            
            # Calculate Argonium method statistics
            argonium_correct = 0
            argonium_total = 0
            both_correct = 0
            both_incorrect = 0
            reasoning_correct_argonium_wrong = 0
            reasoning_wrong_argonium_correct = 0
            agreement_but_both_wrong = 0
            agreement_and_both_right = 0
            
            for trace in results:
                if trace.get("prediction_correct") is not None and trace.get("dual_prediction"):
                    dual_data = trace["dual_prediction"]
                    argonium_pred = dual_data.get("argonium_prediction", {})
                    
                    # Check if argonium prediction is correct (compare with grading result)
                    reasoning_correct = trace.get("prediction_correct", False)
                    
                    # For argonium accuracy, we need to check if the argonium prediction matches the correct answer
                    # We'll use the same grading logic but for the argonium prediction
                    argonium_answer = argonium_pred.get("predicted_answer", "")
                    correct_answer = trace.get("correct_answer", "")
                    reasoning_answer = trace.get("reasoning", {}).get("prediction", {}).get("predicted_answer", "")
                    
                    # Use proper grading model for argonium accuracy calculation
                    argonium_correct_bool = False
                    argonium_num = None
                    reasoning_num = None
                    
                    # Always try to extract numeric values for agreement analysis
                    import re
                    if argonium_answer:
                        argonium_match = re.search(r'\b(\d+)\b', str(argonium_answer))
                        if argonium_match:
                            argonium_num = int(argonium_match.group(1))
                    
                    if reasoning_answer:
                        reasoning_match = re.search(r'\b(\d+)\b', str(reasoning_answer))
                        if reasoning_match:
                            reasoning_num = int(reasoning_match.group(1))
                    
                    if argonium_answer and correct_answer and grading_client is not None and grading_model_name is not None:
                        # Get question data for proper grading
                        question_text = trace.get("question", "")
                        options = trace.get("options", [])
                        
                        # Use the same grading function as for reasoning traces
                        argonium_grading_result = grade_answer(
                            predicted_answer=argonium_answer,
                            correct_answer=correct_answer,
                            question_text=question_text,
                            options=options,
                            grading_client=grading_client,
                            grading_model_name=grading_model_name,
                            verbose=False  # Keep this quiet to avoid too much output
                        )
                        
                        argonium_correct_bool = argonium_grading_result.get("is_correct", False)
                    elif argonium_answer and correct_answer:
                        # Fallback to simple regex matching if no grading model available
                        correct_match = re.search(r'\b(\d+)\b', str(correct_answer))
                        
                        if argonium_num is not None and correct_match:
                            correct_num = int(correct_match.group(1))
                            argonium_correct_bool = (argonium_num == correct_num)
                    
                    argonium_total += 1
                    if argonium_correct_bool:
                        argonium_correct += 1
                    
                    # Agreement analysis
                    if reasoning_correct and argonium_correct_bool:
                        both_correct += 1
                        agreement_and_both_right += 1
                    elif not reasoning_correct and not argonium_correct_bool:
                        both_incorrect += 1
                        # Check if they agreed on the same wrong answer
                        if argonium_num is not None and reasoning_num is not None and argonium_num == reasoning_num:
                            agreement_but_both_wrong += 1
                    elif reasoning_correct and not argonium_correct_bool:
                        reasoning_correct_argonium_wrong += 1
                    elif not reasoning_correct and argonium_correct_bool:
                        reasoning_wrong_argonium_correct += 1
            
            if argonium_total > 0:
                argonium_accuracy = (argonium_correct / argonium_total) * 100
                print(f"Overall accuracy: {argonium_correct}/{argonium_total} correct predictions ({argonium_accuracy:.1f}%)")
                
                print(f"\n📊 METHOD COMPARISON ANALYSIS:")
                print(f"Questions with both Reasoning Traces + Argonium-Style predictions: {argonium_total}")
                print(f"Both Reasoning Traces and Argonium-Style correct: {both_correct} ({(both_correct/argonium_total)*100:.1f}%)")
                print(f"Both Reasoning Traces and Argonium-Style incorrect: {both_incorrect} ({(both_incorrect/argonium_total)*100:.1f}%)")
                print(f"Reasoning Traces correct, Argonium-Style wrong: {reasoning_correct_argonium_wrong}")
                print(f"Reasoning Traces wrong, Argonium-Style correct: {reasoning_wrong_argonium_correct}")
                
                print(f"\n⚠️  AGREEMENT BUT BOTH WRONG:")
                print(f"Cases where Reasoning Traces and Argonium-Style agree but both incorrect: {agreement_but_both_wrong}")
                if agreement_but_both_wrong > 0:
                    print(f"This represents {(agreement_but_both_wrong/argonium_total)*100:.1f}% of all dual predictions")
                
                total_agreement = agreement_and_both_right + agreement_but_both_wrong
                if argonium_total > 0:
                    agreement_rate = (total_agreement / argonium_total) * 100
                    print(f"\n🤝 OVERALL AGREEMENT RATE (Reasoning Traces ↔ Argonium-Style): {total_agreement}/{argonium_total} ({agreement_rate:.1f}%)")
            else:
                print("No dual predictions available for comparison")
        else:
            print("\n💡 Use --dual-prediction to enable Argonium-style method comparison")
    else:
        print("Overall accuracy: No predictions could be graded (0 graded questions)")
        print("Verification method: None (no grading model available)")
        print("⚠️  Use --grading argument to enable answer verification")

    # Print confidence-based accuracy if there's enough data
    if high_confidence_total + medium_confidence_total + low_confidence_total > 0:
        print("\nAccuracy by confidence level:")

        if high_confidence_total > 0:
            high_acc = (high_confidence_correct / high_confidence_total) * 100
            print(
                f"• High confidence: {high_confidence_correct}/{high_confidence_total} correct ({high_acc:.1f}%)"
            )

        if medium_confidence_total > 0:
            med_acc = (medium_confidence_correct / medium_confidence_total) * 100
            print(
                f"• Medium confidence: {medium_confidence_correct}/{medium_confidence_total} correct ({med_acc:.1f}%)"
            )

        if low_confidence_total > 0:
            low_acc = (low_confidence_correct / low_confidence_total) * 100
            print(
                f"• Low confidence: {low_confidence_correct}/{low_confidence_total} correct ({low_acc:.1f}%)"
            )

    # Argonium file comparison analysis if available
    if argonium_data:
        print("\n📋 ARGONIUM FILE COMPARISON ANALYSIS:")
        print("=" * 80)
        
        # Debug information about argonium_data
        log_message(f"DEBUG: argonium_data type: {type(argonium_data)}")
        log_message(f"DEBUG: argonium_data keys: {list(argonium_data.keys()) if isinstance(argonium_data, dict) else 'Not a dict'}")
        if isinstance(argonium_data, dict):
            metadata = argonium_data.get('metadata', {})
            log_message(f"DEBUG: metadata type: {type(metadata)}")
            log_message(f"DEBUG: metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'Not a dict'}")
            log_message(f"DEBUG: overall_accuracy: {metadata.get('overall_accuracy') if isinstance(metadata, dict) else 'No metadata'}")
        
        # Count questions with file comparisons
        file_comparison_count = 0
        matched_questions = 0
        file_vs_new_matches = 0
        file_vs_new_discrepancies = 0
        argonium_file_accuracy = argonium_data.get('metadata', {}).get('overall_accuracy', 0) * 100
        
        for trace in results:
            if trace.get("argonium_file_comparison"):
                file_comparison_count += 1
                
                comparison = trace["argonium_file_comparison"]
                question_match = comparison.get("question_match", {})
                argonium_file_result = comparison.get("argonium_result", {})
                
                if question_match.get("is_match", False):
                    matched_questions += 1
                    
                    # Compare file result with new argonium-style prediction if available
                    if trace.get("dual_prediction"):
                        file_score = argonium_file_result.get("score", 0)
                        
                        # Get new argonium prediction correctness
                        dual_data = trace["dual_prediction"]
                        argonium_pred = dual_data.get("argonium_prediction", {})
                        argonium_answer = argonium_pred.get("predicted_answer", "")
                        correct_answer = trace.get("correct_answer", "")
                        
                        # Use grading model to check new prediction if available
                        new_argonium_correct = False
                        if argonium_answer and correct_answer and grading_client and grading_model_name:
                            question_text = trace.get("question", "")
                            options = trace.get("options", [])
                            
                            grading_result = grade_answer(
                                predicted_answer=argonium_answer,
                                correct_answer=correct_answer,
                                question_text=question_text,
                                options=options,
                                grading_client=grading_client,
                                grading_model_name=grading_model_name,
                                verbose=False
                            )
                            new_argonium_correct = grading_result.get("is_correct", False)
                        
                        # Compare results
                        file_correct = (file_score >= 1)
                        if file_correct == new_argonium_correct:
                            file_vs_new_matches += 1
                        else:
                            file_vs_new_discrepancies += 1
        
        print(f"Argonium file loaded: {args.argonium_results}")
        print(f"Argonium file accuracy: {argonium_file_accuracy:.1f}%")
        print(f"Questions with file comparison: {file_comparison_count}")
        print(f"Questions verified to match: {matched_questions}")
        
        if matched_questions > 0 and dual_prediction_used:
            print(f"\n🔄 FILE vs NEW ARGONIUM-STYLE PREDICTIONS:")
            print(f"Agreements: {file_vs_new_matches}")
            print(f"Discrepancies: {file_vs_new_discrepancies}")
            if file_vs_new_matches + file_vs_new_discrepancies > 0:
                agreement_pct = (file_vs_new_matches / (file_vs_new_matches + file_vs_new_discrepancies)) * 100
                print(f"Agreement rate: {agreement_pct:.1f}%")
                
            # Detailed discrepancy analysis
            if file_vs_new_discrepancies > 0:
                print(f"\n🔍 DETAILED DISCREPANCY ANALYSIS:")
                file_correct_new_wrong = 0
                file_wrong_new_correct = 0
                both_wrong_different_answers = 0
                extraction_failures = 0
                
                for trace in results:
                    if trace.get("argonium_file_comparison") and trace.get("dual_prediction"):
                        comparison = trace["argonium_file_comparison"]
                        question_match = comparison.get("question_match", {})
                        
                        if question_match.get("is_match", False):
                            argonium_file_result = comparison.get("argonium_result", {})
                            file_score = argonium_file_result.get("score", 0)
                            file_correct = (file_score >= 1)
                            
                            dual_data = trace["dual_prediction"]
                            argonium_pred = dual_data.get("argonium_prediction", {})
                            new_extraction_successful = argonium_pred.get("extraction_successful", False)
                            
                            # Check for correctness using grading if available
                            new_argonium_correct = False
                            if new_extraction_successful and grading_client and grading_model_name:
                                argonium_answer = argonium_pred.get("predicted_answer", "")
                                correct_answer = trace.get("correct_answer", "")
                                question_text = trace.get("question", "")
                                options = trace.get("options", [])
                                
                                if argonium_answer and correct_answer:
                                    grading_result = grade_answer(
                                        predicted_answer=argonium_answer,
                                        correct_answer=correct_answer,
                                        question_text=question_text,
                                        options=options,
                                        grading_client=grading_client,
                                        grading_model_name=grading_model_name,
                                        verbose=False
                                    )
                                    new_argonium_correct = grading_result.get("is_correct", False)
                            
                            # Categorize discrepancies
                            if file_correct != new_argonium_correct:
                                if file_correct and not new_argonium_correct:
                                    file_correct_new_wrong += 1
                                elif not file_correct and new_argonium_correct:
                                    file_wrong_new_correct += 1
                                elif not file_correct and not new_argonium_correct:
                                    both_wrong_different_answers += 1
                            
                            if not new_extraction_successful:
                                extraction_failures += 1
                
                print(f"  File correct, new wrong: {file_correct_new_wrong}")
                print(f"  File wrong, new correct: {file_wrong_new_correct}")
                print(f"  Both wrong, different answers: {both_wrong_different_answers}")
                print(f"  New method extraction failures: {extraction_failures}")
                
                if file_correct_new_wrong > 0:
                    print(f"  ⚠️  Regression: {file_correct_new_wrong} cases where file method was better")
                if file_wrong_new_correct > 0:
                    print(f"  📈 Improvement: {file_wrong_new_correct} cases where new method is better")
        
        print("=" * 60)

    print("=" * 80)

    # Generate whole trace analysis if requested
    if args.whole_trace_analysis and results:
        log_message("Generating whole trace analysis...", log_level="INFO")

        # Generate the analysis
        whole_trace_analysis = generate_whole_trace_analysis(
            results, whole_trace_client, whole_trace_model_name, args.specialty
        )

        # Save the analysis to file (but don't print to console)
        try:
            with open(args.whole_trace_output, "w", encoding="utf-8") as f:
                json.dump(whole_trace_analysis, f, indent=2)
            log_message(f"Whole trace analysis saved to {args.whole_trace_output}")
        except Exception as e:
            log_message(f"Error saving whole trace analysis: {e}", log_level="ERROR")


if __name__ == "__main__":
    main()
