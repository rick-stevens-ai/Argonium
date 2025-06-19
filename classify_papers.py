# External dependencies:
# pip install pyyaml openai PyPDF2
import os
import sys
import argparse
import yaml
from openai import OpenAI, OpenAIError
import glob
import hashlib
import random
import shutil

# PDF text extraction
try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. PDF processing will be skipped.")
    PyPDF2 = None

DEFAULT_MODEL_CONFIG_FILE = 'model_servers.yaml'

# Predefined topics for classification
PREDEFINED_TOPICS = [
    "Molecular Biology",
    "Microbiology",
    "Antibiotic Resistance",
    "Mycobacterium tuberculosis",
    "Antibacterial resistance",
    "Antibiotic development",
    "Origins of Life",
    "RNA world hypothesis",
    "Radiation Biology",
    "Origins of Cancer",
    "Molecular Evolution",
    "Infectious Disease",
    "Quantum Computing",
    "Quantum Algorithms",
    "Quantum Chemistry",
    "Quantum Materials",
    "Renormalization Group Technique",
    "Functional Renormalization Group",
    "Theoretical Physics",
    "Radiation Exposure",
    "Radiation Studies",
    "Infectious Disease",
    "Therapy Development",
    "Fundamental Physics",
    "Partial Differential Equations",
    "Numerical Methods",
    "Computational Science",
    "Computer Science",
    "Other"
]

def sanitize_filename(name):
    """Create a file-system friendly name."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '.', '_')).rstrip()

def create_keyword_directories(base_directory, topics):
    """
    Create directories for each keyword/topic.
    
    Args:
        base_directory (str): Base directory where keyword dirs will be created.
        topics (list): List of topics to create directories for.
        
    Returns:
        dict: Mapping of topic names to their directory paths.
    """
    topic_dirs = {}
    
    for topic in topics:
        # Sanitize topic name for directory
        dir_name = sanitize_filename(topic.replace(' ', '_'))
        dir_path = os.path.join(base_directory, dir_name)
        
        try:
            os.makedirs(dir_path, exist_ok=True)
            topic_dirs[topic] = dir_path
            print(f"Created/verified directory: {dir_path}")
        except Exception as e:
            print(f"Error creating directory for topic '{topic}': {e}")
            
    return topic_dirs

def create_processed_directory(base_directory):
    """
    Create a PROCESSED directory for storing processed files.
    
    Args:
        base_directory (str): Base directory where PROCESSED dir will be created.
        
    Returns:
        str: Path to the PROCESSED directory.
    """
    processed_dir = os.path.join(base_directory, "PROCESSED")
    
    try:
        os.makedirs(processed_dir, exist_ok=True)
        print(f"Created/verified PROCESSED directory: {processed_dir}")
        return processed_dir
    except Exception as e:
        print(f"Error creating PROCESSED directory: {e}")
        return None

def copy_file_to_keyword_directory(source_file, topic, topic_dirs):
    """
    Copy a file to the appropriate keyword directory.
    
    Args:
        source_file (str): Path to the source file.
        topic (str): The topic/keyword the file was classified as.
        topic_dirs (dict): Mapping of topics to directory paths.
        
    Returns:
        str: Path to the copied file, or None if failed.
    """
    if topic not in topic_dirs:
        print(f"Warning: No directory found for topic '{topic}'")
        return None
        
    target_dir = topic_dirs[topic]
    filename = os.path.basename(source_file)
    target_path = os.path.join(target_dir, filename)
    
    try:
        shutil.copy2(source_file, target_path)
        print(f"Copied {filename} to {topic} directory")
        return target_path
    except Exception as e:
        print(f"Error copying {filename} to {topic} directory: {e}")
        return None

def move_file_to_processed(source_file, processed_dir):
    """
    Move a file to the PROCESSED directory.
    
    Args:
        source_file (str): Path to the source file.
        processed_dir (str): Path to the PROCESSED directory.
        
    Returns:
        str: Path to the moved file, or None if failed.
    """
    if not processed_dir:
        print("Warning: PROCESSED directory not available")
        return None
        
    filename = os.path.basename(source_file)
    target_path = os.path.join(processed_dir, filename)
    
    try:
        shutil.move(source_file, target_path)
        print(f"Moved {filename} to PROCESSED directory")
        return target_path
    except Exception as e:
        print(f"Error moving {filename} to PROCESSED directory: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file using PyPDF2.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text from the PDF.
    """
    if not PyPDF2:
        print(f"Skipping PDF {pdf_path}: PyPDF2 not available")
        return ""
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from a page in {pdf_path}: {e}")
                    continue
                    
            return text.strip()
            
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def read_text_file(file_path):
    """
    Read text from a .txt or .md file.
    
    Args:
        file_path (str): Path to the text file.
        
    Returns:
        str: Content of the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

def get_first_n_words(text, n=500):
    """
    Get the first n words from text.
    
    Args:
        text (str): Input text.
        n (int): Number of words to extract.
        
    Returns:
        str: First n words of the text.
    """
    words = text.split()
    if len(words) <= n:
        return text
    return ' '.join(words[:n])

def generate_file_id(file_path):
    """
    Generate a file ID based on the filename body (without extension).
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        str: File ID based on filename.
    """
    # Use just the filename without extension as the ID
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return filename

def process_directory(directory_path, max_files=None, random_sample=False):
    """
    Process a directory containing .txt, .md, and .pdf files.
    Extract text and create individual syn_abs.txt files for each processed file.
    
    Args:
        directory_path (str): Path to the directory to process.
        max_files (int, optional): Maximum number of files to process.
        random_sample (bool): Whether to randomly sample files.
        
    Returns:
        list: List of tuples (file_id, file_path, extracted_text)
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return []
    
    # Find all relevant files
    file_patterns = ['*.txt', '*.md', '*.pdf']
    files_to_process = []
    
    for pattern in file_patterns:
        files_to_process.extend(glob.glob(os.path.join(directory_path, pattern)))
        # Also search in subdirectories
        files_to_process.extend(glob.glob(os.path.join(directory_path, '**', pattern), recursive=True))
    
    if not files_to_process:
        print(f"No .txt, .md, or .pdf files found in {directory_path}")
        return []
    
    print(f"Found {len(files_to_process)} files in directory")
    
    # Apply random sampling if requested
    if random_sample:
        random.shuffle(files_to_process)
        print("Files randomly shuffled for sampling")
    
    # Limit number of files if specified
    if max_files and max_files < len(files_to_process):
        files_to_process = files_to_process[:max_files]
        print(f"Processing {len(files_to_process)} files (limited by max_files argument)")
    else:
        print(f"Processing all {len(files_to_process)} files")
    
    processed_files = []
    
    for file_path in files_to_process:
        print(f"Processing: {file_path}")
        
        # Generate file ID based on filename
        file_id = generate_file_id(file_path)
        
        # Extract text based on file type
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            full_text = extract_text_from_pdf(file_path)
        elif file_ext in ['.txt', '.md']:
            full_text = read_text_file(file_path)
        else:
            print(f"Unsupported file type: {file_ext}")
            continue
        
        if not full_text.strip():
            print(f"Warning: No text extracted from {file_path} - will be classified as 'Other'")
            # Create a minimal syn_abs file for empty content
            output_filename = f"{file_id}_syn_abs.txt"
            output_path = os.path.join(directory_path, output_filename)
            
            try:
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(f"# File ID: {file_id}\n")
                    out_f.write(f"# Source: {os.path.relpath(file_path, directory_path)}\n")
                    out_f.write(f"# Status: Empty file - no text extracted\n\n")
                    out_f.write("[No content available]")
                
                print(f"Created: {output_filename} (empty file)")
                # Add to processed files with empty text and mark for "Other" classification
                processed_files.append((file_id, file_path, ""))
                
            except Exception as e:
                print(f"Error writing to output file {output_path}: {e}")
            continue
        
        # Get first 500 words for the syn_abs file
        excerpt = get_first_n_words(full_text, 500)
        
        # Create individual syn_abs.txt file for this document
        output_filename = f"{file_id}_syn_abs.txt"
        output_path = os.path.join(directory_path, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(f"# File ID: {file_id}\n")
                out_f.write(f"# Source: {os.path.relpath(file_path, directory_path)}\n")
                out_f.write(f"# First 500 words extracted\n\n")
                out_f.write(excerpt)
            
            print(f"Created: {output_filename}")
            processed_files.append((file_id, file_path, full_text))
            
        except Exception as e:
            print(f"Error writing to output file {output_path}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_files)} files")
    print(f"Individual syn_abs files created in: {directory_path}")
    
    return processed_files

def classify_document_best_topic(text: str, model_shortname: str, config_file: str, topics: list = None) -> str:
    """
    Classifies document text into the single best matching topic using an OpenAI-compatible model.

    Args:
        text (str): The text content to classify.
        model_shortname (str): The shortname of the model config in model_servers.yaml.
        config_file (str): Path to the model configuration file.
        topics (list, optional): List of topics to classify against. Uses PREDEFINED_TOPICS if None.

    Returns:
        str: The best matching topic, or "Other" on error.
    """
    if not text:
        print("Warning: Cannot classify empty text.")
        return "Other"
    
    if topics is None:
        topics = PREDEFINED_TOPICS

    print(f"Classifying text into best topic using model '{model_shortname}'...")

    # --- 1. Load Model Configuration ---
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Model configuration file '{config_file}' not found during relevance check.")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{config_file}' during relevance check: {e}")
        return False

    model_config = None
    for server in config.get('servers', []):
        if server.get('shortname') == model_shortname:
            model_config = server
            break

    if not model_config:
        print(f"Error: Model shortname '{model_shortname}' not found in '{config_file}' for topic classification.")
        return "Other"

    # --- 2. Determine OpenAI API Key ---
    openai_api_key_config = model_config.get('openai_api_key')
    openai_api_key = None

    if openai_api_key_config == "${OPENAI_API_KEY}":
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            print("Error: OpenAI API key ('OPENAI_API_KEY' env var) not set for topic classification.")
            return "Other"
    elif openai_api_key_config:
        openai_api_key = openai_api_key_config
    else:
        print(f"Error: 'openai_api_key' not specified for model '{model_shortname}' for topic classification.")
        return "Other"

    # --- 3. Instantiate OpenAI Client ---
    openai_api_base = model_config.get('openai_api_base')
    openai_model = model_config.get('openai_model')

    if not openai_api_base or not openai_model:
        print(f"Error: 'openai_api_base' or 'openai_model' missing for model '{model_shortname}'.")
        return "Other"

    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=60.0,
        )
    except Exception as e:
        print(f"Error initializing OpenAI client for topic classification: {e}")
        return "Other"

    # --- 4. Define Prompts ---
    system_prompt = "You are an AI assistant helping classify scientific documents into relevant topics."
    # Truncate text if very long to avoid exceeding token limits
    max_text_chars = 4000  # Approx 1000 tokens, adjust as needed
    truncated_text = text[:max_text_chars] + ('...' if len(text) > max_text_chars else '')
    
    topics_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics)])
    user_prompt = (f"Analyze the following scientific document and classify it into the SINGLE BEST matching topic from the numbered list below. "
                   f"Choose the ONE topic that is most relevant to the document's primary focus. "
                   f"Respond with ONLY the exact topic name as it appears in the list. "
                   f"If none are clearly relevant, respond with 'Other'.\n\n"
                   f"Available Topics:\n{topics_list}\n\n"
                   f"Document Content:\n{truncated_text}")

    # --- 5. Make API Call ---
    print(f"Calling model '{openai_model}' for topic classification...")
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for deterministic response
            max_tokens=50,    # Response should be short - just one topic
        )

        result_text = response.choices[0].message.content
        if not result_text:
            print("Warning: Topic classification API returned an empty response.")
            return "Other"

        # --- 6. Parse Response ---
        result_text = result_text.strip()
        print(f"Topic classification raw response: '{result_text}'")
        
        # Validate the response is in our predefined list
        if result_text in topics:
            best_topic = result_text
        elif result_text.lower() == "other" or not result_text:
            best_topic = "Other"
        else:
            # Try to find a partial match
            best_topic = "Other"
            for topic in topics:
                if topic.lower() in result_text.lower() or result_text.lower() in topic.lower():
                    best_topic = topic
                    break
            if best_topic == "Other":
                print(f"Warning: Unrecognized topic '{result_text}' returned by model, defaulting to 'Other'")
        
        print(f"Best classification: {best_topic}")
        return best_topic

    except OpenAIError as e:
        print(f"Error calling OpenAI API for topic classification: {e}")
        return "Other"
    except Exception as e:
        print(f"An unexpected error occurred during topic classification: {e}")
        return "Other"

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Classify papers from local files.")
    parser.add_argument('directory', type=str, help='Directory containing .txt, .md, or .pdf files to process.')
    parser.add_argument('--config', type=str, default=DEFAULT_MODEL_CONFIG_FILE, 
                       help=f'Path to model servers configuration file (default: {DEFAULT_MODEL_CONFIG_FILE}).')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (default: process all files).')
    parser.add_argument('--random-sample', action='store_true', default=False,
                       help='Randomly sample files to process (useful with --max-files).')
    parser.add_argument('--classification-model', type=str, default='scout', 
                       help='Shortname of the model to use for topic classification (default: scout). Set to None to skip classification.')
    parser.add_argument('--skip-classification', action='store_true', default=False, 
                       help='Skip the topic classification even if model is provided.')
    parser.add_argument('--organize-files', action='store_true', default=False,
                       help='Create keyword directories and organize files by classification. Also creates PROCESSED directory.')

    args = parser.parse_args()

    # --- Validate Arguments ---

    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist or is not a directory.")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)

    # --- Process Directory ---
    print(f"Processing directory: {args.directory}")
    processed_files = process_directory(args.directory, args.max_files, args.random_sample)
    
    if not processed_files:
        print("No files were successfully processed.")
        sys.exit(1)

    # --- Topic Classification (Optional) ---
    file_classifications = {}
    if not args.skip_classification and args.classification_model and args.classification_model.lower() != 'none':
        print(f"\nPerforming topic classification using predefined topics")
        
        for file_id, file_path, full_text in processed_files:
            print(f"\nClassifying: {file_path}")
            try:
                # Check if file has empty content
                if not full_text.strip():
                    best_topic = "Other"
                    print(f"✓ {file_path} classified as: {best_topic} (empty content)")
                else:
                    best_topic = classify_document_best_topic(full_text, args.classification_model, args.config)
                    print(f"✓ {file_path} classified as: {best_topic}")
                
                file_classifications[file_id] = best_topic
            except Exception as e:
                print(f"Error during classification for {file_path}: {e}")
                file_classifications[file_id] = "Other"
        
        print(f"\nClassification complete:")
        print(f"Total files processed: {len(processed_files)}")
        print(f"Files classified: {len(file_classifications)}")
        
        # Show classification summary
        topic_counts = {}
        for topic in file_classifications.values():
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if topic_counts:
            print("\nTopic distribution:")
            for topic, count in sorted(topic_counts.items()):
                print(f"  {topic}: {count} files")
    else:
        print("\nTopic classification skipped.")
    
    # Update syn_abs files with classification results if available
    if file_classifications:
        print("\nUpdating syn_abs files with classification results...")
        for file_id, file_path, full_text in processed_files:
            if file_id in file_classifications:
                best_topic = file_classifications[file_id]
                output_filename = f"{file_id}_syn_abs.txt"
                output_path = os.path.join(args.directory, output_filename)
                
                try:
                    # Read current content
                    with open(output_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()
                    
                    # Add classification to the beginning
                    classification_header = f"# Classification: {best_topic}\n"
                    
                    # Insert after the existing headers
                    lines = current_content.split('\n')
                    header_end = 0
                    for i, line in enumerate(lines):
                        if line.startswith('#'):
                            header_end = i + 1
                        else:
                            break
                    
                    lines.insert(header_end, classification_header.rstrip())
                    updated_content = '\n'.join(lines)
                    
                    # Write back to file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    
                except Exception as e:
                    print(f"Error updating {output_filename} with classification: {e}")
    
    # File organization if requested
    if args.organize_files and file_classifications:
        print("\nOrganizing files by classification...")
        
        # Get unique topics that were actually used
        used_topics = set(file_classifications.values())
        
        # Create keyword directories
        topic_dirs = create_keyword_directories(args.directory, used_topics)
        
        # Create PROCESSED directory
        processed_dir = create_processed_directory(args.directory)
        
        # Copy files to keyword directories and move to PROCESSED
        for file_id, file_path, full_text in processed_files:
            if file_id in file_classifications:
                # Skip syn_abs files - they should remain in the original directory
                if file_path.endswith('_syn_abs.txt'):
                    print(f"Skipping syn_abs file: {os.path.basename(file_path)}")
                    continue
                    
                best_topic = file_classifications[file_id]
                
                # Copy to keyword directory
                copy_file_to_keyword_directory(file_path, best_topic, topic_dirs)
                
                # Move original to PROCESSED
                if processed_dir:
                    move_file_to_processed(file_path, processed_dir)
        
        print(f"\nFile organization complete:")
        print(f"Files copied to {len(used_topics)} keyword directories")
        if processed_dir:
            print(f"Original files moved to PROCESSED directory")

    print(f"\nProcessing complete. Individual syn_abs files created in: {args.directory}")

if __name__ == "__main__":
    main()