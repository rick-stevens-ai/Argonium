#!/usr/bin/env python3
'''
reasoning_traces_v5_robust.py - An improved version with better error handling and JSON extraction
'''

import json
import os
import sys
import time
import yaml
import re
import textwrap
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# OpenAI imports
import openai

# Global variables
_start_time = time.time()
_total_questions = 0
_processed_questions = 0
_current_model_name = None  # Store current model name for discrepancy analysis

def log_message(message, log_level="INFO"):
    '''Log a message with timestamp and log level.'''
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{log_level}] {message}")

def parse_arguments():
    '''Parse command-line arguments.'''
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate detailed reasoning chains as an expert\'s internal dialogue with blind prediction of the correct answer.')
    parser.add_argument('input_file', help='JSON file containing multiple choice questions (output of make_v21.py)')
    parser.add_argument('--output', default='reasoning_traces.json', 
                        help='Output JSON file (default: reasoning_traces.json)')
    parser.add_argument('--model', default='gpt41', 
                        help='Model shortname from model_servers.yaml to use')
    parser.add_argument('--config', default='model_servers.yaml',
                        help='Path to model configuration file (default: model_servers.yaml)')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Maximum number of questions to process (default: all)')
    parser.add_argument('--specialty', default='expert',
                        help='Specialty persona to adopt (e.g., "microbiologist", "quantum physicist", "historian") (default: expert)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save partial results after processing this many questions (default: 10)')
    parser.add_argument('--continue-from', default=None,
                        help='Continue from a previously saved output file')
    
    # Advanced analysis options
    advanced_group = parser.add_argument_group('Advanced Analysis Options')
    advanced_group.add_argument('--whole-trace-analysis', action='store_true',
                        help='Enable whole trace analysis to create a coherent narrative from the reasoning')
    advanced_group.add_argument('--whole-trace-model', 
                        help='Model to use for whole trace analysis (defaults to same as --model if not specified)')
    advanced_group.add_argument('--whole-trace-output', default='whole_trace_output.json',
                        help='Output file for whole trace analysis (default: whole_trace_output.json)')
    advanced_group.add_argument('--enhanced-discrepancy', action='store_true',
                        help='Enable enhanced discrepancy analysis with comprehensive debate about correct answers')
    
    return parser.parse_args()

def configure_apis(model_name: str, config_file: str = "model_servers.yaml") -> str:
    '''
    Configure the necessary APIs based on model selection.
    
    Args:
        model_name: The model shortname to use
        config_file: Path to the model configuration file
    
    Returns:
        The actual model name to use with the API
    '''
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
        log_message(f"Error: Model '{model_name}' not found in {config_file}", log_level="ERROR")
        log_message(f"Available models: {', '.join(s['shortname'] for s in servers_config['servers'])}", log_level="INFO")
        sys.exit(1)
    
    # Configure OpenAI API with server details
    log_message(f"Using model '{selected_server['openai_model']}' from {selected_server['server']}")
    
    # Set OpenAI API parameters
    openai_api_key = selected_server["openai_api_key"]
    # Handle environment variables in the API key
    if openai_api_key.startswith("${") and openai_api_key.endswith("}"):
        env_var = openai_api_key[2:-1]
        openai_api_key = os.environ.get(env_var, "")
        if not openai_api_key:
            log_message(f"Error: Environment variable {env_var} is not set or empty", log_level="ERROR")
            sys.exit(1)
    
    # Configure OpenAI API
    openai.api_key = openai_api_key
    openai.api_base = selected_server["openai_api_base"]
    openai.default_model = selected_server["openai_model"]
    
    # Return the actual model name to use with the API
    return selected_server["openai_model"]

def get_expert_persona(specialty: str) -> str:
    '''
    Generate a detailed persona description for the selected specialty.
    
    Args:
        specialty: The expert specialty (e.g., microbiologist, quantum physicist, historian)
        
    Returns:
        A detailed persona description tailored to the specialty
    '''
    # Pre-defined personas for common specialties
    predefined_personas = {
        "microbiologist": '''I am a microbiologist with over 20 years of experience studying antimicrobial resistance and bacterial pathogenesis. 
I've spent countless hours in the lab isolating bacterial strains, conducting susceptibility tests, and analyzing genomic data. 
When I approach a scientific question, I consider the molecular mechanisms at play, evolutionary pressures, and ecological contexts. 
I'm particularly meticulous about methodology and constantly thinking about experimental design, controls, and statistical significance. 
I tend to connect new information to established principles in bacterial physiology, genetics, and ecology. 
I'm familiar with current literature on antimicrobial agents, resistance mechanisms, biofilms, and emerging therapeutic approaches.''',
        
        "physicist": '''I am a physicist with over 20 years of experience in theoretical and computational physics. 
I've worked extensively on quantum mechanics, statistical mechanics, and particle physics. 
When I approach a physics problem, I consider the underlying physical principles, mathematical formulations, and experimental evidence.
I'm particularly attentive to mathematical rigor, dimensional analysis, and the implications of symmetries.
I tend to connect new information to established theories and look for consistency with fundamental laws.
I'm familiar with current research on quantum field theory, cosmology, condensed matter physics, and computational methods.''',
        
        "quantum physicist": '''I am a quantum physicist with extensive experience in quantum mechanics, quantum field theory, and quantum computing. 
My research focuses on understanding the fundamental principles of quantum systems and their applications in technology. 
When approaching problems, I instinctively think about wave functions, quantum states, superposition, entanglement, and quantum measurement theory. 
I consider both the mathematical formalism and the conceptual interpretations of quantum phenomena. 
My approach is rigorous, often using advanced mathematical tools to analyze quantum systems and their behavior. 
I'm familiar with current research in quantum technologies, quantum information processing, and quantum foundations.''',

        "historian": '''I am a historian with decades of experience in analyzing historical documents, events, and trends. 
My expertise involves critically examining primary and secondary sources, contextualizing events within their broader historical context. 
When analyzing historical questions, I consider multiple perspectives, sociopolitical factors, economic conditions, and cultural influences. 
I'm particularly attentive to the biases in historical accounts and the importance of evaluating the reliability of sources. 
My approach involves connecting specific events to larger historical patterns and understanding how past developments influence present conditions. 
I'm well-versed in historiography and the evolution of historical interpretations over time.'''
    }
    
    # Check if the specialty is in our predefined list
    if specialty.lower() in predefined_personas:
        return predefined_personas[specialty.lower()]
    
    # For unknown specialties, generate a generic expert persona based on the specialty name
    specialty_words = specialty.split()
    specialty_base = specialty_words[-1] if len(specialty_words) > 0 else specialty
    
    # Is it a scientific field?
    scientific_fields = ["biologist", "physicist", "chemist", "geologist", "astronomer", 
                         "mathematician", "engineer", "scientist", "researcher"]
    
    is_scientific = any(field in specialty_base.lower() for field in scientific_fields)
    
    if is_scientific:
        return f'''I am a {specialty} with extensive expertise in my field. 
My work involves analyzing complex scientific problems using rigorous methodologies and detailed knowledge of {specialty} principles.
When approaching questions in my field, I think systematically about the underlying mechanisms, relevant theories, and empirical evidence.
I pay particular attention to scientific accuracy, methodological considerations, and the current state of research in {specialty}.
My approach combines theoretical understanding with practical knowledge of experimental techniques and data analysis.
I'm well-versed in the latest research and ongoing debates in the field of {specialty}.'''
    else:
        # Generic expert persona for non-scientific fields
        return f'''I am a {specialty} with extensive expertise and experience in my field.
My work involves analyzing complex problems through the specialized lens of a {specialty}.
When approaching questions in my field, I consider multiple factors, theoretical frameworks, and practical implications.
I'm particularly attentive to the nuances, contexts, and specialized knowledge that inform {specialty} analysis.
My approach combines theoretical understanding with practical insights gained through years of experience.
I'm well-versed in the foundational principles, current developments, and ongoing debates in my field.'''

def extract_mc_options(question_text: str) -> List[str]:
    '''
    Extract multiple choice options from the question text.
    
    Args:
        question_text: The full multiple choice question text
        
    Returns:
        List of extracted options without their numbers/letters
    '''
    # Split the question into the actual question and the options
    parts = question_text.split('\n\n', 1)
    if len(parts) < 2:
        # Handle case where there's no clear separation
        return []
    
    options_text = parts[1]
    
    # Match different option formats like "1.", "1)", "A.", "A)", etc.
    options = re.findall(r'(?:^|\n)(?:\d+|\w)[.)] (.*?)(?=(?:\n(?:\d+|\w)[.)])|$)', options_text, re.DOTALL)
    
    # Clean up the options (remove asterisks marking correct answers, etc.)
    cleaned_options = [re.sub(r'\s*\(\*\)\s*$', '', option.strip()) for option in options]
    
    return cleaned_options

def extract_thought_process_from_text(text: str, option_count: int) -> Dict[str, str]:
    '''
    Extract thought process for each option from raw text when JSON parsing fails.
    
    Args:
        text: The raw text from the model
        option_count: The number of options in the question
        
    Returns:
        Dictionary with thought process for each option
    '''
    thought_process = {}
    
    # Look for patterns like "Option 1:" or "Let me consider option 1"
    option_patterns = [
        r'(?:Option|OPTION)\s+(\d+)[\s:]+(.*?)(?=(?:Option|OPTION)\s+\d+[\s:]|I\s+predict|My\s+prediction|$)',
        r'(?:Let me consider|Considering|Examining|Analyzing)\s+(?:option|Option)\s+(\d+)[\s.:]+(.*?)(?=(?:Let me consider|Considering|Examining|Analyzing)\s+(?:option|Option)|I\s+predict|My\s+prediction|$)',
        r'(?:^|\n)(\d+)[.]:?\s+(.*?)(?=(?:^|\n)\d+[.:]|I\s+predict|My\s+prediction|$)'
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
        options_pattern = r'(?:Option|OPTION)\s+(\d+)|(?:Let me consider|Considering|Examining|Analyzing)\s+(?:option|Option)\s+(\d+)'
        
        splits = re.split(options_pattern, text)
        if len(splits) > 2:  # We have at least one match
            current_option = None
            for i, part in enumerate(splits):
                if i > 0 and i % 2 == 1 and part and part.strip().isdigit():  # This is an option number
                    current_option = int(part.strip())
                elif i > 1 and i % 2 == 0 and part and current_option is not None:  # This is content
                    if f"option_{current_option}" not in thought_process:
                        thought_process[f"option_{current_option}"] = part.strip()
    
    return thought_process

def extract_prediction_from_text(text: str) -> Dict[str, str]:
    '''
    Extract prediction information from raw text when JSON parsing fails.
    
    Args:
        text: The raw text from the model
        
    Returns:
        Dictionary with prediction information
    '''
    prediction = {
        "predicted_answer": "Could not determine",
        "prediction_reasoning": "",
        "confidence_level": "unknown",
        "confidence_explanation": ""
    }
    
    # Try to find the predicted option number
    predict_patterns = [
        r'I\s+predict\s+(?:that\s+)?(?:option|answer)\s*(?:number|#)?\s*(\d+)',
        r'(?:My prediction|My answer|I believe|I think)\s+(?:is|would be)\s+(?:option|answer)?\s*(?:number|#)?\s*(\d+)',
        r'(?:option|answer)\s+(\d+)\s+(?:is|seems|appears to be)\s+(?:the\s+)?correct',
        r'(?:based on|after)\s+(?:my|this)\s+analysis,\s+(?:option|answer)\s+(\d+)',
        r'(?:therefore|thus|hence),\s+(?:option|answer)\s+(\d+)',
        r'(?:I would|I am going to|I will)\s+(?:choose|select|pick|go with)\s+(?:option|answer)\s+(\d+)',
        r'(?:option|answer)\s+(\d+)[\s.:,]',
        r'(?:the\s+)?correct\s+(?:option|answer)\s+(?:is|would be)\s+(\d+)',
        r'I\s+(?:choose|select|pick)\s+(?:option|answer)\s+(\d+)',
        r'(?:option|answer)\s+(\d+)\s+is\s+(?:the\s+)?(?:most\s+)?(?:correct|accurate|appropriate)'
    ]
    
    for pattern in predict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prediction["predicted_answer"] = f"Option {match.group(1)}"
            break
    
    # Word-to-number mapping as a fallback
    if prediction["predicted_answer"] == "Could not determine":
        word_to_num = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, 
                      "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
                      "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}
        
        for word, num in word_to_num.items():
            pattern = r'(?:the\s+)?' + word + r'(?:\s+option|\s+answer)?\s+(?:is|seems|appears\s+to\s+be)\s+correct'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                prediction["predicted_answer"] = f"Option {num}"
                break
    
    # Extract reasoning
    reason_patterns = [
        r'(?:My reasoning|Reasoning|Here\'s my reasoning|Reason for prediction)(?:for this prediction)?(?:is|:)(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)',
        r'(?:I predict|I believe|I think).*?because(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)',
        r'(?:This option|Option \d+) is correct because(.*?)(?:Confidence|In conclusion|To summarize|In summary|$)'
    ]
    
    for pattern in reason_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            prediction["prediction_reasoning"] = match.group(1).strip()
            break
    
    # Extract confidence level
    confidence_patterns = [
        r'(?:My confidence|Confidence level|I am)(?:is|:)?\s+(high|medium|low)',
        r'I have\s+(high|medium|low)(?:\s+level of)?\s+confidence',
        r'(?:high|medium|low) confidence in (?:this|my) (?:prediction|answer|conclusion)'
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            prediction["confidence_level"] = match.group(1).lower()
            break
    
    # Extract confidence explanation
    explanation_patterns = [
        r'(?:Confidence explanation|Reason for confidence|Why I\'m confident)(?:is|:)(.*?)(?:In conclusion|To summarize|In summary|$)',
        r'(?:I\'m|I am) (?:highly|moderately|somewhat) confident because(.*?)(?:In conclusion|To summarize|In summary|$)',
        r'My confidence is (high|medium|low) because(.*?)(?:In conclusion|To summarize|In summary|$)'
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
    '''
    Extract scientific conclusion from raw text when JSON parsing fails.
    
    Args:
        text: The raw text from the model
        
    Returns:
        Extracted conclusion or empty string
    '''
    conclusion_patterns = [
        r'(?:Scientific conclusion|Conclusion|Final conclusion|In conclusion|To summarize)(?:is|:)(.*?)(?:$)',
        r'(?:Based on my analysis|After analyzing all options|Having considered the evidence)(.*?)(?:$)'
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
    
    # If no conclusion found, try to use the text after prediction as a conclusion
    prediction_match = re.search(r'(?:I predict|My prediction|Therefore,|Thus,).*?(\d+).*?(?:\.|$)', text, re.IGNORECASE)
    if prediction_match:
        prediction_pos = prediction_match.end()
        if prediction_pos < len(text) - 100:  # Ensure there's enough text after prediction
            conclusion = text[prediction_pos:].strip()
            if conclusion:
                return conclusion[:500] + "..." if len(conclusion) > 500 else conclusion
    
    return ""

def generate_reasoning_trace(question_data: Dict[str, Any], model_name: str, specialty: str = "microbiologist") -> Dict[str, Any]:
    '''
    Generate a reasoning trace for a multiple choice question.
    
    Args:
        question_data: Dictionary containing the question data
        model_name: The model name to use for generating the reasoning
        specialty: The scientific specialty persona to adopt
        
    Returns:
        Dictionary with the original question and the reasoning trace
    '''
    # Make model_name accessible to print_readable_output function for discrepancy analysis
    global _current_model_name
    _current_model_name = model_name
    global _processed_questions, _total_questions
    
    # Extract question components
    question_text = question_data["question"]
    correct_answer = question_data["answer"]
    context_text = question_data["text"]
    
    # Extract options from the question text
    options = extract_mc_options(question_text)
    
    # Identify the correct option (look for (*) marking or other indicators)
    correct_option_index = -1
    for i, option in enumerate(options):
        if "(*)" in question_text.split('\n\n', 1)[1].split('\n')[i]:
            correct_option_index = i
            break
    
    if correct_option_index == -1:
        # If we couldn't find the correct answer marker, try to determine from the answer text
        answer_text = correct_answer.lower()
        for i, option in enumerate(options):
            opt_marker = f"option {i+1}"
            ans_marker = f"answer {i+1}"
            letter_marker = chr(ord('a') + i)  # a, b, c, ...
            
            if (opt_marker in answer_text or 
                ans_marker in answer_text or 
                answer_text.startswith(letter_marker + ".") or
                answer_text.startswith(letter_marker + ")")):
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
        log_message(f"Warning: Could not determine correct answer index for question: {question_text[:100]}...", log_level="WARNING")
        # Use the first option as a fallback
        correct_option_index = 0
    
    # Get the expert persona
    persona = get_expert_persona(specialty)
    
    # Construct the prompt for generating the reasoning trace - without relying on context
    # Adapt wording based on whether this is a scientific field
    is_scientific = any(field in specialty.lower() for field in ["scientist", "biologist", "physicist", "chemist", "geologist", "astronomer", "mathematician", "engineer"])
    
    if is_scientific:
        prompt = f'''You are a {specialty} reasoning through a multiple-choice question. You will think through each option thoroughly as if considering a hypothesis, using detailed knowledge and reasoning from your field.

Your persona: {persona}

'''
    else:
        prompt = f'''You are a {specialty} reasoning through a multiple-choice question. You will think through each option thoroughly, using detailed knowledge and reasoning from your field of expertise.

Your persona: {persona}

'''
    
    # Split the question text to get just the question part (without options)
    question_parts = question_text.split('\n\n', 1)
    question_only = question_parts[0] if len(question_parts) > 0 else question_text
    prompt += f"QUESTION:\n{question_only}\n\n"
    prompt += "ANSWER OPTIONS:\n"

    # Add each option to the prompt without revealing the correct answer
    for i, option in enumerate(options):
        prompt += f"{i+1}. {option}\n"
    
    # Don't reveal the correct answer in the initial prompt
    # The model must reason through all options and predict which one is correct
    
    # Customize the task instructions based on specialty
    if is_scientific:
        prompt += f'''TASK:
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

'''
    else:
        prompt += f'''TASK:
Please provide an extremely detailed internal monologue as if you are a {specialty} thinking through this problem. For each answer option:
1. Treat each option as a possibility that you're carefully considering
2. Use specialized terminology and concepts from your field in your reasoning
3. Consider relevant frameworks, methodologies, contexts, and evidence
4. Reason through the implications and logical consequences of each option
5. Reference relevant principles, theories, or frameworks from your domain of expertise
6. Consider alternative interpretations, exceptions, and nuances for each option
7. Express uncertainty and weigh evidence where appropriate

Structure your response as an expert's stream of consciousness:
- Analyze each option thoroughly in sequential order (Option 1, then Option 2, etc.)
- For each option, begin with "Hmm, let me consider option X..."
- After analyzing ALL options, explicitly predict which answer you think is correct using its NUMBER (e.g., "I predict option 3 is correct")
- Then explain your reasoning for your prediction - what principles and evidence led you to this conclusion?
- Finally, indicate your confidence level in your prediction (high, medium, or low) and explain why

'''

    # Add JSON output format instructions - adapt based on specialty
    if is_scientific:
        prompt += '''Output your reasoning in JSON format with the following structure:
{
  "thought_process": {
    "option_1": "Detailed reasoning about option 1 as a hypothesis",
    "option_2": "Detailed reasoning about option 2 as a hypothesis",
    ... (all options in numerical order)
  },
  "prediction": {
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Reasoning for why you predict this answer is correct",
    "confidence_level": "Your confidence level (high, medium, or low)",
    "confidence_explanation": "Why you have this level of confidence in your prediction"
  },
  "scientific_conclusion": "Final synthesized assessment"
}
'''
    else:
        prompt += '''Output your reasoning in JSON format with the following structure:
{
  "thought_process": {
    "option_1": "Detailed reasoning about option 1",
    "option_2": "Detailed reasoning about option 2",
    ... (all options in numerical order)
  },
  "prediction": {
    "predicted_answer": "The option number you predict is correct (e.g., 3)",
    "prediction_reasoning": "Reasoning for why you predict this answer is correct",
    "confidence_level": "Your confidence level (high, medium, or low)",
    "confidence_explanation": "Why you have this level of confidence in your prediction"
  },
  "conclusion": "Final synthesized assessment"
}
'''
    # Add important reminders for all specialties
    prompt += '''IMPORTANT: Your response must be a valid, parseable JSON object.
- Do not include backticks, markdown formatting, or any text outside the JSON object
- Ensure all keys are properly quoted
- Escape all quotes within strings using backslashes
- Do not use trailing commas
- For each option, include detailed reasoning of at least 150-200 words
- After analyzing all options, carefully determine which one you believe is correct
- In 'predicted_answer', specify ONLY the option number you believe is correct (e.g., '3', NOT 'Option 3')
- Provide detailed reasoning (250+ words) for your prediction
- REMEMBER: You are not told which answer is correct - you must make a genuine prediction
- CRITICAL: You MUST make a prediction. In the "predicted_answer" field, put only a number (1, 2, 3, etc.)'''

    try:
        # Create the completion request
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You are an expert {specialty} with deep knowledge in your field. You meticulously analyze questions using detailed reasoning and technical terminology appropriate to your domain. You express your thought process as a rich internal monologue, considering multiple angles, frameworks, and implications. VERY IMPORTANT: Do not assume you know which answer is correct - you must reason through each option carefully and make your own prediction based on your expertise. After your analysis, you must PREDICT which answer you think is correct and explain your reasoning.\n\nIMPORTANT FORMATTING INSTRUCTIONS:\n1. When you make your prediction, you MUST specify a clear NUMERIC answer (e.g., 'Option 3' or just '3').\n2. DO NOT use words like 'first option', 'second option', etc. - use the actual number.\n3. The 'predicted_answer' field in your JSON output must be a simple format like '3' or 'Option 3'.\n4. Your JSON must be properly formatted with no trailing commas and properly escaped characters.\n5. If you have high confidence in one option, state it clearly with 'I predict that Option X is correct'"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=4000
        )
        
        # Extract the response
        response_text = response.choices[0].message.content.strip()
        
        # First try to parse as JSON directly
        try:
            json_content = json.loads(response_text)
            log_message("Successfully parsed response as valid JSON", log_level="INFO")
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if code_block_match:
                try:
                    json_content = json.loads(code_block_match.group(1))
                    log_message("Successfully parsed JSON from code block", log_level="INFO")
                except json.JSONDecodeError:
                    # Fallback to structured extraction from raw text
                    log_message("JSON parsing failed, extracting structured data from raw text", log_level="INFO")
                    
                    # Extract thought process for each option
                    thought_process = extract_thought_process_from_text(response_text, len(options))
                    
                    # Extract prediction details
                    prediction = extract_prediction_from_text(response_text)
                    
                    # Extract scientific conclusion
                    conclusion = extract_conclusion_from_text(response_text)
                    
                    # Create structured JSON
                    json_content = {
                        "thought_process": thought_process,
                        "prediction": prediction,
                        "scientific_conclusion": conclusion,
                        "extracted_from_text": True
                    }
            else:
                # No code block found, extract structured data directly from raw text
                log_message("No JSON code block found, extracting directly from text", log_level="INFO")
                
                # Clean response text to remove any corruptions
                cleaned_text = response_text
                # Remove code block artifacts
                cleaned_text = re.sub(r'```json', '', cleaned_text)
                cleaned_text = re.sub(r'```', '', cleaned_text)
                # Remove any stray JSON fragments like "option_X": or "scientific_conclusion":
                cleaned_text = re.sub(r'"[a-z_]+":\s*', '', cleaned_text)
                # Remove quotes around paragraphs
                cleaned_text = re.sub(r'"\s*(.*?)\s*"', r'\1', cleaned_text, flags=re.DOTALL)
                
                # Look for actual JSON within the text
                json_match = re.search(r'\{.*?\}', cleaned_text, re.DOTALL)
                if json_match:
                    try:
                        potential_json = json_match.group(0)
                        log_message("Found potential JSON embedded in text", log_level="INFO")
                        json_content = json.loads(potential_json)
                        
                        # If we found JSON but it's missing thought_process
                        if "thought_process" not in json_content:
                            thought_process = extract_thought_process_from_text(cleaned_text, len(options))
                            json_content["thought_process"] = thought_process
                            
                        # If we found JSON but it's missing prediction
                        if "prediction" not in json_content:
                            prediction = extract_prediction_from_text(cleaned_text)
                            json_content["prediction"] = prediction
                            
                        # If we found JSON but it's missing conclusion
                        if "scientific_conclusion" not in json_content and "conclusion" not in json_content:
                            conclusion = extract_conclusion_from_text(cleaned_text)
                            json_content["scientific_conclusion"] = conclusion
                            
                        json_content["partially_extracted"] = True
                    except json.JSONDecodeError:
                        # Fallback to full extraction
                        log_message("Embedded JSON invalid, falling back to full extraction", log_level="INFO")
                        # Extract thought process for each option
                        thought_process = extract_thought_process_from_text(cleaned_text, len(options))
                        
                        # Extract prediction details
                        prediction = extract_prediction_from_text(cleaned_text)
                        
                        # Extract scientific conclusion
                        conclusion = extract_conclusion_from_text(cleaned_text)
                        
                        # Create structured JSON
                        json_content = {
                            "thought_process": thought_process,
                            "prediction": prediction,
                            "scientific_conclusion": conclusion,
                            "extracted_from_text": True
                        }
                else:
                    # No embedded JSON, do full extraction
                    # Extract thought process for each option
                    thought_process = extract_thought_process_from_text(cleaned_text, len(options))
                    
                    # Extract prediction details
                    prediction = extract_prediction_from_text(cleaned_text)
                    
                    # Extract scientific conclusion
                    conclusion = extract_conclusion_from_text(cleaned_text)
                    
                    # Create structured JSON
                    json_content = {
                        "thought_process": thought_process,
                        "prediction": prediction,
                        "scientific_conclusion": conclusion,
                        "extracted_from_text": True
                    }
                
                # If we couldn't extract meaningful structured data, save the raw text
                if not thought_process and not prediction["predicted_answer"]:
                    log_message("Structured extraction failed, using raw text", log_level="WARNING")
                    json_content = {
                        "thought_process": {},
                        "prediction": {
                            "predicted_answer": "Could not extract from response",
                            "prediction_reasoning": "Error parsing response",
                            "confidence_level": "unknown",
                            "confidence_explanation": "Could not determine"
                        },
                        "scientific_conclusion": response_text,
                        "raw_text": response_text,
                        "extraction_failed": True
                    }
        
        # If we haven't set a key for the raw text and extraction wasn't explicitly marked as failed,
        # store the original response for debugging
        if "raw_text" not in json_content and not json_content.get("extraction_failed", False):
            json_content["raw_text"] = response_text
        
        # Process the result
        reasoning_data = json_content
        
        # Update processed question count and log progress
        _processed_questions += 1
        completion_percentage = (_processed_questions / _total_questions) * 100
        elapsed_time = time.time() - _start_time
        avg_time_per_question = elapsed_time / _processed_questions if _processed_questions > 0 else 0
        
        estimated_remaining = avg_time_per_question * (_total_questions - _processed_questions)
        if estimated_remaining < 60:
            eta = f"{estimated_remaining:.0f} seconds"
        elif estimated_remaining < 3600:
            eta = f"{estimated_remaining/60:.1f} minutes"
        else:
            eta = f"{estimated_remaining/3600:.1f} hours"
        
        log_message(f"Processed {_processed_questions}/{_total_questions} questions ({completion_percentage:.1f}%) - ETA: {eta}")
        
        # Return the result
        return {
            "question": question_text,
            "context": context_text,
            "correct_answer_index": correct_option_index,
            "correct_answer": options[correct_option_index] if correct_option_index < len(options) else "",
            "options": options,
            "reasoning": reasoning_data
        }
        
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
            "model_used": model_name,
            "query_successful": False
        }
        
        return {
            "question": question_text,
            "context": context_text[:500] + "..." if len(context_text) > 500 else context_text,  # Include truncated context
            "correct_answer_index": correct_option_index,
            "correct_answer": correct_answer,
            "options": options,
            "reasoning": {
                "thought_process": {},
                "prediction": {
                    "predicted_answer": "Error occurred",
                    "prediction_reasoning": "An error occurred while generating reasoning.",
                    "confidence_level": "unknown",
                    "confidence_explanation": "Processing error"
                },
                "scientific_conclusion": f"Failed to generate reasoning due to an error: {error_msg}"
            },
            "error_details": error_structure
        }

def generate_coherent_stream_analysis(reasoning_trace: Dict[str, Any], specialty: str = "expert", model_name: str = None) -> str:
    '''
    Generate a coherent stream of thought analysis showing internal debate between options.
    
    Args:
        reasoning_trace: The reasoning trace to analyze
        specialty: The expert specialty persona
        model_name: The model to use for generating the synthesis
        
    Returns:
        A natural internal dialogue showing the debate between different options
    '''
    # Extract all the raw content from the reasoning trace
    question = reasoning_trace.get('question', 'Unknown question')
    options = reasoning_trace.get('options', [])
    correct_answer_idx = reasoning_trace.get('correct_answer_index', -1)
    correct_answer = options[correct_answer_idx] if 0 <= correct_answer_idx < len(options) else "Unknown"
    reasoning = reasoning_trace.get('reasoning', {})
    thought_process = reasoning.get('thought_process', {})
    prediction = reasoning.get('prediction', {})
    scientific_conclusion = reasoning.get('scientific_conclusion', reasoning.get('conclusion', ''))
    was_correct = reasoning_trace.get('prediction_correct', False)
    predicted_num = reasoning_trace.get('predicted_num', None)
    
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
    option_keys = sorted([k for k in thought_process.keys() if k.startswith('option_')], 
                        key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf'))
    
    for opt_key in option_keys:
        try:
            opt_idx = int(opt_key.split('_')[-1])
            if opt_idx <= len(options):
                synthesis_context += f"\nAbout the idea that {options[opt_idx-1] if opt_idx-1 < len(options) else 'Unknown'}:\n"
                synthesis_context += f"{thought_process[opt_key]}\n"
        except (ValueError, IndexError):
            continue
    
    # Add what I eventually decided
    synthesis_context += f"\nWHAT I ULTIMATELY SETTLED ON:\n"
    synthesis_context += f"My Final Thought: {prediction.get('predicted_answer', 'Unknown')}\n"
    synthesis_context += f"My Reasoning: {prediction.get('prediction_reasoning', 'No reasoning provided')}\n"
    synthesis_context += f"My Confidence: {prediction.get('confidence_level', 'unknown')}\n"
    
    # The actual outcome (but don't reveal this in the internal dialogue initially)
    correct_thought = options[correct_answer_idx] if correct_answer_idx < len(options) else "Unknown"
    actual_result = f"[ACTUAL RESULT: The idea that '{correct_thought}' was correct"
    if predicted_num is not None:
        if was_correct:
            actual_result += f" - I got it right!]"
        else:
            predicted_thought = options[predicted_num-1] if predicted_num-1 < len(options) else f"option {predicted_num}"
            actual_result += f" - I was wrong, I went with '{predicted_thought}']"
    else:
        actual_result += f" - my final choice was unclear]"
    
    # Create the synthesis prompt for internal debate
    synthesis_prompt = f"""You are a {specialty} working through this question. I want you to create a natural internal dialogue that shows you actively debating between different thoughts and possibilities as they occur to you.

CRITICAL INSTRUCTIONS:
- Don't refer to these as "Option 1, Option 2" etc. Instead, treat them as natural thoughts that just came to mind, and show yourself debating between these ideas organically.
- REVEAL ALL YOUR INTERNAL DETAILS AND REASONING - don't hold anything back or try to be brief
- Include ALL the technical details, scientific knowledge, and specific reasoning that goes through your mind
- Show your complete thought process with full scientific depth and complexity
- Don't summarize or abbreviate - show the full internal scientific debate

This should be written as your ACTUAL internal thoughts while you're working on the problem - the real-time mental conversation you're having with yourself as you:

1. Read the question and start thinking
2. Different possibilities come to mind naturally
3. You weigh each idea, arguing for and against them in your mind with FULL scientific detail
4. You go back and forth between different thoughts, including ALL technical reasoning
5. You feel yourself leaning toward certain ideas, then second-guessing with complete scientific rationale
6. You finally settle on your answer with full explanation

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
    if model_name is None:
        # Fallback to simple template if no model available
        return f"Unable to generate coherent stream analysis - no model specified for synthesis."
    
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You are an expert {specialty} engaging in honest self-reflection about your own reasoning process. You write in a natural, conversational internal monologue style that captures authentic thought patterns, including moments of uncertainty, connections to knowledge, and genuine self-assessment."},
                {"role": "user", "content": f"{synthesis_context}\n\n{synthesis_prompt}"}
            ],
            temperature=0.7,  # Higher temperature for more natural, varied expression
            max_tokens=2000  # Increased for full detailed scientific reasoning
        )
        
        analysis_text = response.choices[0].message.content.strip()
        return analysis_text
        
    except Exception as e:
        log_message(f"Error generating coherent stream analysis: {e}", log_level="ERROR")
        return f"Error generating stream analysis: {str(e)}"

def print_readable_output(question_data: Dict[str, Any], reasoning_trace: Dict[str, Any], specialty: str = "expert", show_stream_analysis: bool = False):
    '''Print a human-readable version of the reasoning trace to the console.
    
    Args:
        question_data: The original question data
        reasoning_trace: The generated reasoning trace
        specialty: The specialty persona used in the analysis
        show_stream_analysis: Whether to show coherent stream analysis after the reasoning
    '''
    # Access the global model name
    global _current_model_name
    try:
        print("\n" + "="*80)
        # Split the question to get just the question part (without options)
        question_parts = question_data['question'].split('\n\n', 1)
        question_only = question_parts[0] if len(question_parts) > 0 else question_data['question']
        print(f"QUESTION: {question_only}")
        print("-"*80)
    
        # Print options
        options = reasoning_trace['options']
        correct_index = reasoning_trace['correct_answer_index']
        
        print("OPTIONS:")
        for i, option in enumerate(options):
            marker = "✓" if i == correct_index else " "
            print(f"{marker} {i+1}. {option}")
    
        print("-"*100)
        print(f"{specialty.upper()}'S THOUGHT PROCESS:")
        
        # Check for errors
        has_error = (
            reasoning_trace['reasoning'].get('extraction_failed') or
            reasoning_trace.get('error_details') is not None
        )
        
        if has_error:
            print("Error in response - showing available information:")
            
            # Show error details if present
            if reasoning_trace.get('error_details'):
                error_details = reasoning_trace['error_details']
                print(f"Error type: {error_details.get('error_type')}")
                print(f"Error message: {error_details.get('error_message')}")
            
            # Show the raw response or summary if available
            raw_response = reasoning_trace['reasoning'].get('raw_text', '')
            if not raw_response:
                raw_response = reasoning_trace['reasoning'].get('scientific_conclusion', '')
            
            if len(raw_response) > 20:  # Only show if it has meaningful content
                print("\nRaw response snippet:")
                print(raw_response[:500] + "..." if len(raw_response) > 500 else raw_response)
        else:
            # Print thought process for each option
            thought_process = reasoning_trace['reasoning'].get('thought_process', {})
            
            # Sort option keys numerically
            option_keys = sorted([k for k in thought_process.keys() if k.startswith('option_')], 
                                key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf'))
            
            # Print options in order
            for opt_key in option_keys:
                try:
                    opt_idx = int(opt_key.split('_')[-1]) - 1
                    if opt_idx >= 0 and opt_idx < len(options):
                        print(f"\n💭 OPTION {opt_idx+1}: {options[opt_idx]}")
                        
                        # Get the thoughts and clean them
                        thoughts = thought_process[opt_key]
                        
                        # Format the thought process text with indentation
                        if isinstance(thoughts, str):
                            # Clean the text to handle potential formatting issues
                            clean_text = re.sub(r'"\s*$', '', thoughts)  # Remove trailing quotes
                            clean_text = re.sub(r'^\s*"', '', clean_text)  # Remove leading quotes
                            
                            # Handle remaining escape sequences
                            clean_text = clean_text.replace('\\', '')
                            
                            # Format with indentation
                            formatted_thoughts = "\n".join("   " + line for line in clean_text.split("\n"))
                            print(formatted_thoughts)
                        elif isinstance(thoughts, dict):
                            # Convert dict to indented text
                            for k, v in thoughts.items():
                                print(f"   {k}:")
                                if isinstance(v, str):
                                    print("\n".join("      " + line for line in v.split("\n")))
                                else:
                                    print(f"      {v}")
                        else:
                            print(f"   {thoughts}")
                        
                        print("-"*80)  # Separator between options
                except (ValueError, IndexError) as e:
                    pass
            
            # Print the prediction
            print("\n" + "="*80)
            print(f"🔮 {specialty.upper()}'S PREDICTION:")
            prediction = reasoning_trace['reasoning'].get('prediction', {})
            
            # Check if prediction is a string or a dict - handle both cases
            if isinstance(prediction, str):
                # Try to parse it as JSON if it's a string (might be a direct JSON output)
                try:
                    prediction_dict = json.loads(prediction)
                    if isinstance(prediction_dict, dict) and 'predicted_answer' in prediction_dict:
                        prediction = prediction_dict
                    else:
                        # Just use it as the predicted answer
                        prediction = {'predicted_answer': prediction}
                except:
                    # If it's not valid JSON, just use it as the predicted answer
                    prediction = {'predicted_answer': prediction}
            
            predicted_answer = prediction.get('predicted_answer', 'No prediction provided')
            try:
                # More robust number extraction - look for various patterns
                # Check for "option X", "answer X", "X", or just a number
                option_match = re.search(r'(?:option|answer|opt\.?|ans\.?)\s*(?:number|#)?\s*[:#]?\s*(\d+)', predicted_answer, re.IGNORECASE)
                # If no match found with the above pattern, try finding just a number
                if not option_match:
                    option_match = re.search(r'(?<!\w)(\d+)(?!\w)', predicted_answer)
                
                if option_match:
                    predicted_num = int(option_match.group(1))
                    predicted_correct = (predicted_num - 1) == correct_index  # 0-indexed vs 1-indexed
                    
                    if predicted_correct:
                        print(f"   Predicted Answer: Option {predicted_num} ✅ CORRECT")
                    else:
                        print(f"   Predicted Answer: Option {predicted_num} ❌ INCORRECT - actual correct answer is Option {correct_index+1}")
                else:
                    # Try to check if the answer contains words like "first", "second", etc.
                    word_to_num = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, 
                                  "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
                    for word, num in word_to_num.items():
                        if re.search(r'\b' + word + r'\b', predicted_answer, re.IGNORECASE):
                            predicted_num = num
                            predicted_correct = (predicted_num - 1) == correct_index
                            
                            if predicted_correct:
                                print(f"   Predicted Answer: Option {predicted_num} ✅ CORRECT")
                            else:
                                print(f"   Predicted Answer: Option {predicted_num} ❌ INCORRECT - actual correct answer is Option {correct_index+1}")
                            break
                    else:  # No word match found
                        print(f"   Predicted Answer: {predicted_answer} (Could not parse option number)")
                        
                # Store the prediction result for overall accuracy calculation
                if 'predicted_num' in locals():
                    reasoning_trace['prediction_correct'] = predicted_correct
                    reasoning_trace['predicted_num'] = predicted_num
                else:
                    reasoning_trace['prediction_correct'] = False
                    reasoning_trace['predicted_num'] = None
                    
            except Exception as e:
                print(f"   Predicted Answer: {predicted_answer}")
                log_message(f"Error parsing prediction: {str(e)}", log_level="DEBUG")
                reasoning_trace['prediction_correct'] = False
                
            print("\n   Prediction Reasoning:")
            prediction_reasoning = prediction.get('prediction_reasoning', 'No reasoning provided')
            
            # Clean up the prediction reasoning
            if isinstance(prediction_reasoning, str):
                # Remove excess quotes and JSON escapes
                prediction_reasoning = re.sub(r'^\s*"', '', prediction_reasoning)
                prediction_reasoning = re.sub(r'"\s*$', '', prediction_reasoning)
                prediction_reasoning = prediction_reasoning.replace('\\"', '"').replace('\\n', '\n')
                prediction_reasoning = re.sub(r'",\s*"confidence_level.*$', '', prediction_reasoning)
            
            # Format the reasoning with indentation
            formatted_reasoning = "\n".join("      " + line for line in str(prediction_reasoning).split("\n"))
            print(formatted_reasoning)
            
            # Extract and clean confidence level
            confidence_level = prediction.get('confidence_level', 'Not specified')
            if isinstance(confidence_level, str):
                confidence_level = confidence_level.strip().lower()
                confidence_level = re.sub(r'^\s*"', '', confidence_level)
                confidence_level = re.sub(r'"\s*$', '', confidence_level)
                
            print(f"\n   Confidence Level: {confidence_level}")
            print("\n   Confidence Explanation:")
            
            # Clean up confidence explanation
            confidence_explanation = prediction.get('confidence_explanation', 'No explanation provided')
            if isinstance(confidence_explanation, str):
                # Remove excess quotes and JSON escapes
                confidence_explanation = re.sub(r'^\s*"', '', confidence_explanation)
                confidence_explanation = re.sub(r'"\s*$', '', confidence_explanation)
                confidence_explanation = confidence_explanation.replace('\\"', '"').replace('\\n', '\n')
                confidence_explanation = re.sub(r'",\s*"scientific_conclusion.*$', '', confidence_explanation)
            
            # Format the explanation with indentation
            formatted_explanation = "\n".join("      " + line for line in str(confidence_explanation).split("\n"))
            print(formatted_explanation)
    
        print("\n" + "="*80)
        print("🔬 SCIENTIFIC CONCLUSION:")
        
        scientific_conclusion = reasoning_trace['reasoning'].get('scientific_conclusion', '')
        
        if not scientific_conclusion or scientific_conclusion == "No conclusion provided.":
            scientific_conclusion = reasoning_trace['reasoning'].get('conclusion', 'No conclusion provided')
        
        if reasoning_trace['reasoning'].get('extraction_failed', False):
            print("Error in response parsing - conclusion not available")
        else:
            # Clean up the scientific conclusion
            if isinstance(scientific_conclusion, str):
                # Remove excess quotes and JSON escapes
                scientific_conclusion = re.sub(r'^\s*"', '', scientific_conclusion)
                scientific_conclusion = re.sub(r'"\s*$', '', scientific_conclusion)
                scientific_conclusion = scientific_conclusion.replace('\\"', '"').replace('\\n', '\n')
            
            # Format the conclusion with indentation and block quotes
            formatted_conclusion = "\n".join("> " + line for line in str(scientific_conclusion).split("\n"))
            print(formatted_conclusion)
        print("="*80)
        
        # Add coherent stream analysis if requested
        if show_stream_analysis:
            print("\n" + "="*80)
            print("🌊 COHERENT STREAM OF THOUGHT ANALYSIS")
            print("="*80)
            
            try:
                # Use the current model for synthesis
                global _current_model_name
                stream_analysis = generate_coherent_stream_analysis(reasoning_trace, specialty, _current_model_name)
                
                # Print the stream analysis without truncation (preserve natural line breaks)
                print(stream_analysis)
                
            except Exception as stream_error:
                print(f"Error generating stream analysis: {str(stream_error)}")
                
            print("="*80)
            
    except Exception as e:
        print("\nError displaying reasoning output:")
        print(f"Error: {str(e)}")
        print(f"Question: {question_data.get('question', 'Unknown')[:100]}...")
        print("="*80 + "\n")

def generate_whole_trace_analysis(reasoning_traces: List[Dict[str, Any]], model_name: str, specialty: str = "expert") -> Dict[str, Any]:
    '''
    Generate a coherent narrative analysis from the collected reasoning traces.
    
    Args:
        reasoning_traces: List of reasoning trace dictionaries
        model_name: The model name to use for generating the analysis
        specialty: The expert specialty persona to adopt
        
    Returns:
        Dictionary containing the whole trace analysis
    '''
    log_message("Generating whole trace analysis...", log_level="INFO")
    
    # Prepare a summary of all the reasoning traces
    trace_summary = []
    accuracy_stats = {"correct": 0, "incorrect": 0, "total": 0}
    confidence_breakdown = {"high": [], "medium": [], "low": []}
    
    for i, trace in enumerate(reasoning_traces):
        if 'reasoning' not in trace:
            continue
            
        # Extract key information
        question = trace.get('question', 'Unknown question')[:100] + "..."
        predicted_answer = trace.get('reasoning', {}).get('prediction', {}).get('predicted_answer', 'Unknown')
        correct_answer_idx = trace.get('correct_answer_index', -1)
        is_correct = trace.get('prediction_correct', False)
        confidence = trace.get('reasoning', {}).get('prediction', {}).get('confidence_level', 'unknown')
        
        # Update statistics
        accuracy_stats["total"] += 1
        if is_correct:
            accuracy_stats["correct"] += 1
        else:
            accuracy_stats["incorrect"] += 1
            
        # Track confidence levels
        if confidence.lower() in confidence_breakdown:
            confidence_breakdown[confidence.lower()].append({
                "question_num": i + 1,
                "correct": is_correct,
                "predicted": predicted_answer
            })
        
        # Create a summary entry
        trace_summary.append({
            "question_number": i + 1,
            "question_snippet": question,
            "predicted_answer": predicted_answer,
            "correct_answer_index": correct_answer_idx + 1 if correct_answer_idx >= 0 else "Unknown",
            "was_correct": is_correct,
            "confidence_level": confidence,
            "key_reasoning_points": trace.get('reasoning', {}).get('prediction', {}).get('prediction_reasoning', '')[:200] + "..."
        })
    
    # Calculate accuracy percentage
    accuracy_percentage = (accuracy_stats["correct"] / accuracy_stats["total"] * 100) if accuracy_stats["total"] > 0 else 0
    
    # Create the prompt for whole trace analysis
    persona = get_expert_persona(specialty)
    
    # Adapt the analysis approach based on specialty type
    is_scientific = any(field in specialty.lower() for field in ["scientist", "biologist", "physicist", "chemist", "geologist", "astronomer", "mathematician", "engineer"])
    
    prompt = f'''You are a {specialty} conducting a comprehensive meta-analysis of your reasoning performance across multiple questions.

Your persona: {persona}

PERFORMANCE SUMMARY:
- Total questions analyzed: {accuracy_stats["total"]}
- Correct predictions: {accuracy_stats["correct"]} ({accuracy_percentage:.1f}%)
- Incorrect predictions: {accuracy_stats["incorrect"]}
- High confidence decisions: {len(confidence_breakdown["high"])}
- Medium confidence decisions: {len(confidence_breakdown["medium"])}
- Low confidence decisions: {len(confidence_breakdown["low"])}

DETAILED TRACE SUMMARY:
'''

    # Add trace summaries to the prompt
    for trace in trace_summary[:10]:  # Limit to first 10 for brevity
        status = "✓" if trace["was_correct"] else "✗"
        prompt += f'''
Question {trace["question_number"]}: {trace["question_snippet"]}
Prediction: {trace["predicted_answer"]} {status}
Confidence: {trace["confidence_level"]}
Key reasoning: {trace["key_reasoning_points"]}
'''
    
    if len(trace_summary) > 10:
        prompt += f"\n... and {len(trace_summary) - 10} more questions\n"
    
    # Customize analysis instructions based on specialty
    if is_scientific:
        prompt += f'''
TASK: As a {specialty}, provide a comprehensive meta-analysis of your reasoning performance. Analyze:

1. METHODOLOGICAL PATTERNS: What reasoning approaches did you consistently use? Were there common analytical frameworks or scientific principles you relied on?

2. ACCURACY PATTERNS: Where were you most/least accurate? What types of questions or concepts challenged your expertise?

3. CONFIDENCE CALIBRATION: How well-calibrated was your confidence? Were you overconfident or underconfident in specific areas?

4. DOMAIN-SPECIFIC INSIGHTS: What does this performance reveal about the intersection of your {specialty} expertise with these questions?

5. SYSTEMATIC BIASES: Did you exhibit any consistent biases or blind spots in your reasoning?

6. LEARNING OPPORTUNITIES: What areas would benefit from deeper investigation or different analytical approaches?

7. METHODOLOGICAL RECOMMENDATIONS: How might your analytical approach be refined for future similar analyses?

Format your response as a detailed scientific analysis with clear sections and evidence-based conclusions.
'''
    else:
        prompt += f'''
TASK: As a {specialty}, provide a comprehensive meta-analysis of your reasoning performance. Analyze:

1. ANALYTICAL PATTERNS: What reasoning approaches did you consistently use? Were there common frameworks or principles you relied on?

2. ACCURACY PATTERNS: Where were you most/least accurate? What types of questions or concepts challenged your expertise?

3. CONFIDENCE CALIBRATION: How well-calibrated was your confidence? Were you overconfident or underconfident in specific areas?

4. DOMAIN-SPECIFIC INSIGHTS: What does this performance reveal about applying your {specialty} perspective to these questions?

5. SYSTEMATIC TENDENCIES: Did you exhibit any consistent patterns or preferences in your reasoning?

6. IMPROVEMENT OPPORTUNITIES: What areas would benefit from deeper investigation or different analytical approaches?

7. STRATEGIC RECOMMENDATIONS: How might your analytical approach be refined for future similar analyses?

Format your response as a detailed professional analysis with clear sections and evidence-based conclusions.
'''
    
    try:
        # Generate the whole trace analysis
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"You are an expert {specialty} conducting a reflective meta-analysis of your own reasoning performance. You approach this analysis with the same rigor and expertise you bring to questions in your field. You are honest about both strengths and weaknesses in your reasoning patterns."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Slightly higher temperature for more creative analysis
            max_tokens=2000
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
                    "low_confidence": len(confidence_breakdown["low"])
                }
            },
            "meta_analysis": analysis_text,
            "detailed_trace_summary": trace_summary,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": model_name
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
                "incorrect_predictions": accuracy_stats["incorrect"]
            },
            "meta_analysis": f"Error generating analysis: {str(e)}",
            "error": True,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def print_whole_trace_analysis(analysis: Dict[str, Any]):
    '''
    Print a human-readable version of the whole trace analysis.
    
    Args:
        analysis: The whole trace analysis dictionary
    '''
    print("\n" + "="*100)
    print("🧠 WHOLE TRACE META-ANALYSIS")
    print("="*100)
    
    specialty = analysis.get('specialty', 'expert').upper()
    print(f"Perspective: {specialty}")
    print(f"Generated at: {analysis.get('generated_at', 'Unknown')}")
    print(f"Questions analyzed: {analysis.get('total_questions_analyzed', 0)}")
    print(f"Overall accuracy: {analysis.get('overall_accuracy', 0):.1f}%")
    
    # Performance breakdown
    breakdown = analysis.get('performance_breakdown', {})
    if breakdown:
        print(f"\nPerformance Summary:")
        print(f"  • Correct predictions: {breakdown.get('correct_predictions', 0)}")
        print(f"  • Incorrect predictions: {breakdown.get('incorrect_predictions', 0)}")
        
        conf_dist = breakdown.get('confidence_distribution', {})
        if conf_dist:
            print(f"  • High confidence decisions: {conf_dist.get('high_confidence', 0)}")
            print(f"  • Medium confidence decisions: {conf_dist.get('medium_confidence', 0)}")
            print(f"  • Low confidence decisions: {conf_dist.get('low_confidence', 0)}")
    
    print("-"*100)
    print("📊 DETAILED META-ANALYSIS:")
    print("-"*100)
    
    meta_analysis = analysis.get('meta_analysis', 'No analysis available')
    
    if analysis.get('error'):
        print("❌ Error occurred during analysis generation:")
        print(meta_analysis)
    else:
        # Format the meta-analysis with proper indentation
        formatted_analysis = "\n".join(line for line in meta_analysis.split("\n"))
        print(formatted_analysis)
    
    print("="*100)

def main():
    '''Main entry point function.'''
    global _total_questions, _processed_questions
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure the OpenAI API for the selected model
    model_name = configure_apis(args.model, args.config)
    
    # Configure the whole trace analysis model (if different)
    whole_trace_model_name = model_name
    if args.whole_trace_model:
        whole_trace_model_name = configure_apis(args.whole_trace_model, args.config)
        log_message(f"Using different model for whole trace analysis: {args.whole_trace_model}")
    
    # Initialize results list
    results = []
    
    # Check if we're continuing from a previous run
    starting_index = 0
    if args.continue_from and os.path.exists(args.continue_from):
        try:
            with open(args.continue_from, 'r', encoding='utf-8') as f:
                results = json.load(f)
                starting_index = len(results)
                log_message(f"Continuing from previous run - {starting_index} questions already processed", log_level="INFO")
        except Exception as e:
            log_message(f"Error reading continue-from file: {e}", log_level="ERROR")
            log_message("Starting from scratch", log_level="INFO")
    
    # Read the input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    except Exception as e:
        log_message(f"Error reading input file: {e}", log_level="ERROR")
        sys.exit(1)
    
    # Filter for multiple choice questions only
    mc_questions = [q for q in questions_data if q.get("type") == "multiple-choice"]
    
    if not mc_questions:
        log_message("No multiple choice questions found in the input file.", log_level="ERROR")
        sys.exit(1)
    
    # Apply maximum questions limit if specified
    if args.max_questions is not None and args.max_questions > 0:
        mc_questions = mc_questions[:args.max_questions]
    
    # Skip questions we've already processed if continuing
    if starting_index > 0:
        mc_questions = mc_questions[starting_index:]
    
    _total_questions = len(mc_questions) + starting_index
    _processed_questions = starting_index
    
    log_message(f"Found {len(mc_questions)} multiple choice questions to process. Starting processing...")
    log_message(f"Using {args.specialty} persona for scientific reasoning")
    
    # Generate reasoning traces for each question
    for i, question in enumerate(tqdm(mc_questions, desc=f"Generating {args.specialty}'s reasoning traces")):
        # Generate the basic reasoning trace
        trace = generate_reasoning_trace(question, model_name, args.specialty)
        results.append(trace)
        
        # Print readable output to console (with stream analysis if whole-trace-analysis is enabled)
        print_readable_output(question, trace, args.specialty, show_stream_analysis=args.whole_trace_analysis)
        
        # Save intermediate results at specified intervals
        current_index = starting_index + i + 1
        if (current_index % args.save_interval == 0) or (i == len(mc_questions) - 1):
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                log_message(f"Saved intermediate results to {args.output} after processing {current_index} questions", log_level="INFO")
            except Exception as e:
                log_message(f"Error saving intermediate results: {e}", log_level="ERROR")
    
    # Save final results
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        log_message(f"Successfully wrote reasoning traces to {args.output}")
    except Exception as e:
        log_message(f"Error writing output file: {e}", log_level="ERROR")
    
    # Calculate accuracy statistics
    correct_predictions = sum(1 for trace in results if trace.get('prediction_correct', False))
    total_with_predictions = sum(1 for trace in results if 'prediction_correct' in trace)
    
    if total_with_predictions > 0:
        accuracy_percentage = (correct_predictions / total_with_predictions) * 100
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
        if trace.get('prediction_correct') is None:
            continue
            
        confidence = trace.get('reasoning', {}).get('prediction', {}).get('confidence_level', '').lower()
        is_correct = trace.get('prediction_correct', False)
        
        if confidence == 'high':
            high_confidence_total += 1
            if is_correct:
                high_confidence_correct += 1
        elif confidence == 'medium':
            medium_confidence_total += 1
            if is_correct:
                medium_confidence_correct += 1
        elif confidence == 'low':
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
    
    log_message(f"Processing complete! Generated {args.specialty}'s reasoning traces for {_processed_questions} questions in {time_str}.")
    log_message(f"Results saved to {args.output}")
    
    # Print accuracy summary
    print("\n" + "="*80)
    print("📈 PREDICTION ACCURACY SUMMARY")
    print("="*80)
    print(f"Overall accuracy: {correct_predictions}/{total_with_predictions} correct predictions ({accuracy_percentage:.1f}%)")
    
    # Print confidence-based accuracy if there's enough data
    if high_confidence_total + medium_confidence_total + low_confidence_total > 0:
        print("\nAccuracy by confidence level:")
        
        if high_confidence_total > 0:
            high_acc = (high_confidence_correct / high_confidence_total) * 100
            print(f"• High confidence: {high_confidence_correct}/{high_confidence_total} correct ({high_acc:.1f}%)")
            
        if medium_confidence_total > 0:
            med_acc = (medium_confidence_correct / medium_confidence_total) * 100
            print(f"• Medium confidence: {medium_confidence_correct}/{medium_confidence_total} correct ({med_acc:.1f}%)")
            
        if low_confidence_total > 0:
            low_acc = (low_confidence_correct / low_confidence_total) * 100
            print(f"• Low confidence: {low_confidence_correct}/{low_confidence_total} correct ({low_acc:.1f}%)")
    
    print("="*80)
    
    # Generate whole trace analysis if requested
    if args.whole_trace_analysis and results:
        log_message("Generating whole trace analysis...", log_level="INFO")
        
        # Generate the analysis
        whole_trace_analysis = generate_whole_trace_analysis(
            results, 
            whole_trace_model_name, 
            args.specialty
        )
        
        # Save the analysis to file (but don't print to console)
        try:
            with open(args.whole_trace_output, 'w', encoding='utf-8') as f:
                json.dump(whole_trace_analysis, f, indent=2)
            log_message(f"Whole trace analysis saved to {args.whole_trace_output}")
        except Exception as e:
            log_message(f"Error saving whole trace analysis: {e}", log_level="ERROR")

if __name__ == "__main__":
    main()