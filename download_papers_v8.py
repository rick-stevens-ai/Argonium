# External dependencies:
# pip install requests pyyaml openai
import requests
import os
import time
import sys
import argparse
import yaml
from openai import OpenAI, OpenAIError

# Base URL for Semantic Scholar API
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = 'title,authors,year,externalIds,url,venue,openAccessPdf,abstract,paperId'
MODEL_CONFIG_FILE = 'model_servers.yaml'

def sanitize_filename(name):
    """Create a file-system friendly name."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '.', '_')).rstrip()

def search_papers(keyword, api_key, limit=100, offset=0):
    """
    Search papers for a given 'keyword' using the Semantic Scholar API.
    Returns the JSON response if successful, otherwise None.
    Includes diagnostic prints for debugging.

    Args:
        keyword (str): The search term.
        api_key (str | None): The Semantic Scholar API key.
        limit (int): Max results per request.
        offset (int): Starting offset for results.

    Returns:
        dict | None: The JSON response from the API or None on failure.
    """
    url = BASE_URL
    params = {
        'query': keyword,
        'limit': limit,
        'offset': offset,
        'fields': FIELDS
    }
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key

    # Print debug info about the request
    print("-" * 50)
    print(f"Debug Info: Attempting to retrieve papers with:")
    print(f"  Keyword: {keyword}")
    print(f"  URL: {url}")
    print(f"  Parameters: {params}")
    if api_key:
        print(f"  Using SS API key: {api_key[:4]}... (truncated)")
    else:
        print("  No SS API key provided.")
    print("-" * 50)

    max_retries = 5
    backoff_time = 5  # Start with 5 seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30) # Added timeout
            print(f"Request URL (final): {response.url}")
            print(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                print(f"Successfully fetched data on attempt {attempt}")
                return response.json()
            elif response.status_code == 429:
                print(f"Rate limit exceeded (attempt {attempt}). "
                      f"Retrying after {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
            elif response.status_code == 403:
                print(f"Error: Forbidden (403). Check your API key permissions or usage limits. Attempt {attempt}.")
                # Stop retrying on 403
                return None
            else:
                print(f"Error: Unexpected status code {response.status_code} on attempt {attempt}.")
                print(f"Response text: {response.text[:200]} ...")
                # Consider stopping retry for certain errors, maybe based on status code ranges
                # For now, returning None after first failure for non-429 errors
                return None

        except requests.exceptions.Timeout:
            print(f"Request timed out on attempt {attempt}. Retrying after {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2
        except requests.exceptions.RequestException as e:
            print(f"Network or request error on attempt {attempt}: {e}")
            time.sleep(backoff_time)
            backoff_time *= 2

    print(f"Failed to fetch papers for keyword '{keyword}' after {max_retries} retries.")
    return None

def download_pdf(pdf_url, save_path):
    """
    Download a PDF from 'pdf_url' and save it to 'save_path'.
    Includes error handling and diagnostics.
    """
    print(f"Attempting to download PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url, stream=True, timeout=60) # Added timeout
        print(f"Download response status code: {response.status_code}")
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded PDF to: {save_path}")
        else:
            print(f"Failed to download PDF from {pdf_url} - "
                  f"status code: {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"Timeout while downloading PDF from {pdf_url}")
    except Exception as e:
        print(f"Exception while downloading PDF from {pdf_url}: {e}")

def generate_keywords_with_openai(base_keyword, model_shortname):
    """
    Generates related research keywords using an OpenAI-compatible model.

    Args:
        base_keyword (str): The keyword to generate related terms from.
        model_shortname (str): The shortname of the model config in model_servers.yaml.

    Returns:
        list[str]: A list of generated keywords.

    Raises:
        FileNotFoundError: If model_servers.yaml is not found.
        ValueError: If the model shortname is not found or API key is missing.
        OpenAIError: If the API call fails.
    """
    print(f"Generating keywords based on '{base_keyword}' using model '{model_shortname}'...")

    # --- 1. Load Model Configuration ---
    try:
        with open(MODEL_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Model configuration file '{MODEL_CONFIG_FILE}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{MODEL_CONFIG_FILE}': {e}")

    model_config = None
    for server in config.get('servers', []):
        if server.get('shortname') == model_shortname:
            model_config = server
            break

    if not model_config:
        raise ValueError(f"Error: Model shortname '{model_shortname}' not found in '{MODEL_CONFIG_FILE}'.")

    # --- 2. Determine OpenAI API Key ---
    openai_api_key_config = model_config.get('openai_api_key')
    openai_api_key = None

    if openai_api_key_config == "${OPENAI_API_KEY}":
        openai_api_key = os.environ.get('OPENAI-API-KEY') or os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("Error: OpenAI API key is configured to use environment variable "
                             "'OPENAI-API-KEY' or 'OPENAI_API_KEY', but neither is set.")
        if os.environ.get('OPENAI-API-KEY'):
            print("Using OpenAI API key from environment variable 'OPENAI-API-KEY'.")
        else:
            print("Using OpenAI API key from environment variable 'OPENAI_API_KEY'.")
    elif openai_api_key_config:
        openai_api_key = openai_api_key_config
        print(f"Using OpenAI API key configured for model '{model_shortname}'.")
    else:
        raise ValueError(f"Error: 'openai_api_key' not specified for model '{model_shortname}' "
                         f"in '{MODEL_CONFIG_FILE}'.")

    # --- 3. Instantiate OpenAI Client ---
    openai_api_base = model_config.get('openai_api_base')
    openai_model = model_config.get('openai_model')

    if not openai_api_base or not openai_model:
        raise ValueError(f"Error: 'openai_api_base' or 'openai_model' missing for model "
                         f"'{model_shortname}' in '{MODEL_CONFIG_FILE}'.")

    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    except Exception as e:
        raise ValueError(f"Error initializing OpenAI client: {e}")

    # --- 4. Define Prompts ---
    system_prompt = ("You are an AI assistant helping with research. Generate related keywords "
                     "based on the user's input.")
    user_prompt = (f"Given the base keyword '{base_keyword}', generate 5 related research "
                   f"keywords suitable for searching academic papers. Output *only* the keywords, "
                   f"each on a new line, with no other text, numbering, or formatting.")

    # --- 5. Make API Call ---
    print(f"Calling OpenAI model '{openai_model}' at '{openai_api_base}'...")
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5, # Adjust creativity
            max_tokens=100,  # Limit response length
        )

        generated_text = response.choices[0].message.content
        if not generated_text:
            print("Warning: OpenAI API returned an empty response.")
            return []

        # --- 6. Parse Response ---
        generated_keywords = [kw.strip() for kw in generated_text.strip().split('\n') if kw.strip()]
        print(f"Successfully generated {len(generated_keywords)} keywords.")
        return generated_keywords

    except OpenAIError as e:
        print(f"Error calling OpenAI API: {e}")
        raise  # Re-raise the exception to be caught in main
    except Exception as e:
        print(f"An unexpected error occurred during keyword generation: {e}")
        raise # Re-raise for handling in main

def check_relevance_with_openai(abstract: str, keyword: str, model_shortname: str) -> bool:
    """
    Checks if a paper abstract is relevant to a given keyword using an OpenAI-compatible model.

    Args:
        abstract (str): The paper's abstract text.
        keyword (str): The search keyword the abstract should be relevant to.
        model_shortname (str): The shortname of the model config in model_servers.yaml.

    Returns:
        bool: True if the abstract is deemed relevant, False otherwise (including errors).
    """
    if not abstract:
        print("Warning: Cannot check relevance for an empty abstract.")
        return False # Cannot be relevant if no abstract

    print(f"Checking relevance of abstract against keyword '{keyword}' using model '{model_shortname}'...")

    # --- 1. Load Model Configuration ---
    try:
        with open(MODEL_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Model configuration file '{MODEL_CONFIG_FILE}' not found during relevance check.")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{MODEL_CONFIG_FILE}' during relevance check: {e}")
        return False

    model_config = None
    for server in config.get('servers', []):
        if server.get('shortname') == model_shortname:
            model_config = server
            break

    if not model_config:
        print(f"Error: Model shortname '{model_shortname}' not found in '{MODEL_CONFIG_FILE}' for relevance check.")
        return False

    # --- 2. Determine OpenAI API Key ---
    openai_api_key_config = model_config.get('openai_api_key')
    openai_api_key = None

    if openai_api_key_config == "${OPENAI_API_KEY}":
        openai_api_key = os.environ.get('OPENAI-API-KEY') or os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            print("Error: OpenAI API key ('OPENAI-API-KEY' or 'OPENAI_API_KEY' env var) not set for relevance check.")
            return False
    elif openai_api_key_config:
        openai_api_key = openai_api_key_config
    else:
        print(f"Error: 'openai_api_key' not specified for model '{model_shortname}' for relevance check.")
        return False

    # --- 3. Instantiate OpenAI Client ---
    openai_api_base = model_config.get('openai_api_base')
    openai_model = model_config.get('openai_model')

    if not openai_api_base or not openai_model:
        print(f"Error: 'openai_api_base' or 'openai_model' missing for model '{model_shortname}'.")
        return False

    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=60.0, # Increased timeout for potentially longer analysis
        )
    except Exception as e:
        print(f"Error initializing OpenAI client for relevance check: {e}")
        return False

    # --- 4. Define Prompts ---
    system_prompt = "You are an AI assistant helping evaluate research paper relevance."
    # Truncate abstract if very long to avoid exceeding token limits
    max_abstract_chars = 4000 # Approx 1000 tokens, adjust as needed
    truncated_abstract = abstract[:max_abstract_chars] + ('...' if len(abstract) > max_abstract_chars else '')

    user_prompt = (f"Analyze the following abstract based on the keyword '{keyword}'. "
                   f"Is the abstract highly relevant to this specific keyword? "
                   f"Respond ONLY with the single word 'Relevant' or 'Not Relevant'.\n\n"
                   f"Keyword: {keyword}\n"
                   f"Abstract:\n{truncated_abstract}")

    # --- 5. Make API Call ---
    print(f"Calling model '{openai_model}' for relevance assessment...")
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, # Low temperature for deterministic response
            max_tokens=10,  # Response should be very short
        )

        result_text = response.choices[0].message.content
        if not result_text:
            print("Warning: Relevance check API returned an empty response.")
            return False # Treat empty response as not relevant

        # --- 6. Parse Response ---
        decision = result_text.strip().lower()
        print(f"Relevance check raw response: '{result_text.strip()}'")
        if decision == "relevant":
            print("Result: Relevant")
            return True
        else:
            print("Result: Not Relevant")
            return False

    except OpenAIError as e:
        print(f"Error calling OpenAI API for relevance check: {e}")
        return False # Treat API errors as not relevant
    except Exception as e:
        print(f"An unexpected error occurred during relevance check: {e}")
        return False # Treat other errors as not relevant

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Download academic papers from Semantic Scholar.")
    parser.add_argument('--ss-api-key', type=str, help='Semantic Scholar API key.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--keywords-file', type=str, help='Path to a file containing keywords (one per line).')
    group.add_argument('--generate-keywords', type=str, help='A base keyword to generate related keywords using OpenAI.')
    group.add_argument('--keywords', type=str, help='Comma-separated keywords to search for (e.g., "machine learning,neural networks,deep learning").')

    parser.add_argument('--model', type=str, help='Shortname of the OpenAI model (from model_servers.yaml) to use for keyword generation. Required if --generate-keywords is used.')
    parser.add_argument('--relevance-model', type=str, default=None, help='Shortname of the OpenAI model (from model_servers.yaml) to use for relevance checking. If not provided, relevance check is skipped.')
    parser.add_argument('--skip-relevance-check', action='store_true', default=False, help='Skip the abstract relevance check before downloading PDFs.')
    parser.add_argument('--max-relevant-papers', type=int, default=None, help='Maximum number of relevant papers to download per keyword. If not specified, downloads all relevant papers found.')

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.generate_keywords and not args.model:
        parser.error("--model is required when using --generate-keywords")
    # Note: No need to validate relevance_model here, as it's optional.
    #       The check function will handle cases where it's needed but invalid.
    # --- Determine Semantic Scholar API Key ---
    ss_api_key = os.environ.get('SS-API-KEY') or os.environ.get('SEMANTIC_SCHOLAR_API_KEY') or args.ss_api_key
    if ss_api_key:
        if os.environ.get('SS-API-KEY'):
            print("Using Semantic Scholar API key from environment variable 'SS-API-KEY'.")
        elif os.environ.get('SEMANTIC_SCHOLAR_API_KEY'):
            print("Using Semantic Scholar API key from environment variable 'SEMANTIC_SCHOLAR_API_KEY'.")
        else:
            print("Using Semantic Scholar API key from command line argument.")
    else:
        print("Warning: No Semantic Scholar API key provided (via env vars SS-API-KEY/SEMANTIC_SCHOLAR_API_KEY or --ss-api-key). Rate limits may be stricter.")

    # --- Determine Keywords ---
    keywords = []
    if args.keywords_file:
        try:
            with open(args.keywords_file, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
            if not keywords:
                print(f"Warning: No keywords found in '{args.keywords_file}'.")
                sys.exit(0)
            print(f"Read {len(keywords)} keywords from '{args.keywords_file}'.")
        except FileNotFoundError:
            print(f"Error: Keywords file '{args.keywords_file}' not found.")
            sys.exit(1)
    elif args.keywords:
        keywords = [keyword.strip() for keyword in args.keywords.split(',') if keyword.strip()]
        if not keywords:
            print("Warning: No valid keywords found in the provided string.")
            sys.exit(0)
        print(f"Using {len(keywords)} keywords from command line argument.")
    elif args.generate_keywords:
        try:
            keywords = generate_keywords_with_openai(args.generate_keywords, args.model)
            if not keywords:
                print("No keywords were generated. Exiting.")
                sys.exit(0)
            # Include the base keyword in the list to process
            if args.generate_keywords not in keywords:
                 keywords.insert(0, args.generate_keywords)
            print(f"Generated {len(keywords)} keywords using model '{args.model}'.")
        except (FileNotFoundError, ValueError, OpenAIError, Exception) as e:
            print(f"Error during keyword generation: {e}")
            sys.exit(1)

    print("Keywords to process:")
    for k in keywords:
        print(f"  - {k}")

    # --- Main Processing Loop ---
    for keyword in keywords:
        print("=" * 50)
        print(f"Processing keyword: '{keyword}'")
        print("=" * 50)

        directory_name = sanitize_filename(keyword)
        os.makedirs(directory_name, exist_ok=True)

        processed_papers_file = os.path.join(directory_name, 'processed_papers.txt')
        processed_papers = set()
        if os.path.exists(processed_papers_file):
            try:
                with open(processed_papers_file, 'r', encoding='utf-8') as f:
                    processed_papers = set(line.strip() for line in f)
                print(f"Loaded {len(processed_papers)} processed paper IDs from {processed_papers_file}")
            except Exception as e:
                 print(f"Warning: Could not read {processed_papers_file}. Starting fresh for this keyword. Error: {e}")
                 processed_papers = set()


        all_papers = []
        total_papers_to_fetch = 1000 # Max papers to attempt fetching per keyword
        fetch_limit = 100 # Papers per API call

        print(f"Attempting to fetch up to {total_papers_to_fetch} papers for '{keyword}'...")
        for offset in range(0, total_papers_to_fetch, fetch_limit):
            print(f"Fetching papers (offset={offset})...")
            data = search_papers(keyword, api_key=ss_api_key, limit=fetch_limit, offset=offset)

            if data and 'data' in data:
                papers = data['data']
                all_papers.extend(papers)
                print(f"Retrieved {len(papers)} papers for '{keyword}' at offset {offset}.")
                # Check if 'next' field indicates more results, or if fewer than limit returned
                if len(papers) < fetch_limit or not data.get('next'):
                    print("Reached end of results or fetched less than limit, stopping fetch loop for this keyword.")
                    break
            else:
                print(f"No valid data returned or error occurred for '{keyword}' at offset {offset}. Stopping fetch loop for this keyword.")
                break # Stop fetching for this keyword if an error occurs

            # Sleep to avoid hitting rate limits too quickly
            print("Sleeping for 5 seconds before next fetch...")
            time.sleep(5)

        print(f"Total papers collected for '{keyword}': {len(all_papers)}")
        if not all_papers:
             print(f"No papers found for keyword '{keyword}'. Moving to next keyword.")
             continue

        # --- Process Collected Papers ---
        papers_processed_this_run = 0
        papers_skipped = 0
        pdfs_downloaded = 0
        papers_skipped_relevance = 0
        relevant_papers_downloaded = 0
        for paper in all_papers:
            # Check if we've reached the maximum number of relevant papers
            if args.max_relevant_papers and relevant_papers_downloaded >= args.max_relevant_papers:
                print(f"Reached maximum of {args.max_relevant_papers} relevant papers for keyword '{keyword}'. Stopping processing.")
                break
            paper_id = paper.get('paperId')
            title = paper.get('title', 'No Title Provided')

            if not paper_id:
                print(f"Skipping paper with no paperId: '{title}'")
                continue

            # Skip if already processed
            if paper_id in processed_papers:
                # print(f"Skipping already-processed paper ID: {paper_id} ('{title}')")
                papers_skipped += 1
                continue

            print(f"\nProcessing Paper ID: {paper_id} - '{title}'")
            abstract = paper.get('abstract', '') or ''
            year = paper.get('year', 'N/A')

            # Save abstract
            abstract_filename = os.path.join(directory_name, f"{paper_id}.txt")
            print(f"Saving abstract (Year: {year}) to: {abstract_filename}")
            try:
                with open(abstract_filename, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"Year: {year}\n")
                    f.write(f"Paper ID: {paper_id}\n")
                    f.write(f"Abstract:\n{abstract}\n")
            except Exception as e:
                print(f"Error writing abstract file {abstract_filename}: {e}")
                # Optionally continue to next paper or handle error differently
                continue # Skip this paper if abstract cannot be saved

            # --- Relevance Check (Optional) ---
            is_relevant = True # Assume relevant by default or if check is skipped
            if not args.skip_relevance_check and args.relevance_model and abstract:
                try:
                    is_relevant = check_relevance_with_openai(abstract, keyword, args.relevance_model)
                    # Add a small delay after relevance check API call
                    time.sleep(2)
                except Exception as e:
                    # Catch any unexpected error from the check function itself
                    print(f"Error during relevance check call: {e}")
                    is_relevant = False # Treat errors as not relevant

            if not is_relevant:
                print(f"Skipping paper ID {paper_id} ('{title}') due to relevance check.")
                papers_skipped_relevance += 1
                # Record as processed even if skipped for relevance, to avoid re-checking later
                processed_papers.add(paper_id)
                try:
                    with open(processed_papers_file, 'a', encoding='utf-8') as f:
                        f.write(paper_id + "\n")
                except Exception as e:
                    print(f"Error writing skipped-relevance paper ID to processed file {processed_papers_file}: {e}")
                continue # Skip to the next paper

            # --- Download PDF if available and relevant ---
            pdf_downloaded_this_paper = False
            # Now we know is_relevant is True if we reach here
            if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
                pdf_url = paper['openAccessPdf']['url']
                pdf_filename = paper_id + ".pdf"
                save_path = os.path.join(directory_name, pdf_filename)

                if not os.path.exists(save_path):
                    print(f"Found Open Access PDF URL. Downloading...")
                    download_pdf(pdf_url, save_path)
                    if os.path.exists(save_path): # Verify download succeeded
                         pdfs_downloaded += 1
                         pdf_downloaded_this_paper = True
                    # Sleep after download attempt
                    print("Sleeping for 5 seconds after PDF download attempt...")
                    time.sleep(5)
                else:
                    print(f"PDF already exists: {save_path}")
            else:
                print(f"No Open Access PDF URL found for paper '{title}'")

            # Record as processed only after handling abstract and PDF
            processed_papers.add(paper_id)
            try:
                with open(processed_papers_file, 'a', encoding='utf-8') as f:
                    f.write(paper_id + "\n")
            except Exception as e:
                print(f"Error writing to processed papers file {processed_papers_file}: {e}")
                # Critical error? Maybe exit or log prominently
            papers_processed_this_run += 1
            relevant_papers_downloaded += 1  # Increment counter for relevant papers

        print(f"\nFinished processing for keyword '{keyword}'.")
        print(f"  Papers processed in this run: {papers_processed_this_run}")
        print(f"  Papers skipped (already processed): {papers_skipped}")
        print(f"  Papers skipped (relevance check): {papers_skipped_relevance}")
        print(f"  New PDFs downloaded: {pdfs_downloaded}")
        print(f"  Relevant papers downloaded: {relevant_papers_downloaded}")
        if args.max_relevant_papers:
            print(f"  Maximum relevant papers limit: {args.max_relevant_papers}")
    print("\nAll keywords processed.")

if __name__ == "__main__":
    main()
