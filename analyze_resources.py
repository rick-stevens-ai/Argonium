#!/usr/bin/env python3

import os
import glob
import random
import re
import nltk
import logging
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional
import warnings
import math

# PDF text extraction
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.pdfparser import PDFSyntaxError

# NLTK for text processing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

# For TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# For TextRank
import networkx as nx

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resource_analysis.log"),
        logging.StreamHandler()
    ]
)

# Constants
MAX_SAMPLE_SIZE = 100
MAX_FILE_SIZE_MB = 10  # Max file size to process in MB
OUTPUT_README = os.path.join(os.getcwd(), "README.md")
RESOURCES_DIR = os.getcwd()
SUPPORTED_EXTENSIONS = {'.txt', '.pdf'}

class TextProcessor:
    """Class for text processing and keyword extraction"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add domain-specific stopwords
        self.stop_words.update([
            'figure', 'table', 'et', 'al', 'doi', 'isbn', 'vol', 'pp', 'abstract',
            'introduction', 'conclusion', 'references', 'et al', 'et al.',
            'fig', 'eq', 'eqs', 'equation', 'equations', 'ref', 'refs', 'page',
            'pages', 'chapter', 'section', 'result', 'results', 'method', 'methods',
            'discussion', 'acknowledgements', 'author', 'authors', 'publisher',
            'journal', 'volume', 'issue', 'year', 'press', 'university'
        ])
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters, 
        numbers and convert to lowercase
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Remove citations like [1], [2,3]
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        # Remove special characters but keep letters, numbers and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove excessive whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize text, remove stopwords, and lemmatize
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def extract_keywords_tfidf(self, documents: List[str], top_n: int = 20) -> List[str]:
        """
        Extract keywords using TF-IDF
        """
        if not documents:
            return []
            
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            preprocessor=self.preprocess_text,
            tokenizer=self.tokenize_and_clean,
            max_features=100,
            ngram_range=(1, 2)
        )
        
        try:
            # Fit and transform documents
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF score for each feature across all documents
            mean_tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Get indices of top keywords
            top_indices = mean_tfidf_scores.argsort()[-top_n:][::-1]
            
            # Get top keywords
            top_keywords = [feature_names[i] for i in top_indices]
            
            return top_keywords
        except Exception as e:
            logging.error(f"Error in TF-IDF extraction: {e}")
            return []
    
    def extract_keywords_textrank(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract keywords using TextRank algorithm
        """
        if not text:
            return []
            
        try:
            # Preprocess and tokenize text
            tokens = self.tokenize_and_clean(text)
            
            # Create a graph
            graph = nx.Graph()
            
            # Build the graph using co-occurrence window of size 3
            window_size = 3
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i:i+window_size]
                for j in range(len(window)):
                    for k in range(j+1, len(window)):
                        if window[j] != window[k]:  # Avoid self-loops
                            if graph.has_edge(window[j], window[k]):
                                graph[window[j]][window[k]]['weight'] += 1.0
                            else:
                                graph.add_edge(window[j], window[k], weight=1.0)
            
            # Apply PageRank to find important nodes (keywords)
            scores = nx.pagerank(graph)
            
            # Get top keywords
            sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [keyword for keyword, score in sorted_keywords[:top_n]]
            
            return top_keywords
        except Exception as e:
            logging.error(f"Error in TextRank extraction: {e}")
            return []
    
    def extract_collocations(self, text: str, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Extract bigram and trigram collocations
        """
        if not text:
            return {"bigrams": [], "trigrams": []}
            
        try:
            # Preprocess and tokenize text
            tokens = self.tokenize_and_clean(text)
            
            # Extract bigrams
            bigram_finder = BigramCollocationFinder.from_words(tokens)
            bigram_finder.apply_freq_filter(3)  # Filter out bigrams that appear less than 3 times
            bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
            
            # Extract trigrams
            trigram_finder = TrigramCollocationFinder.from_words(tokens)
            trigram_finder.apply_freq_filter(2)  # Filter out trigrams that appear less than 2 times
            trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, top_n)
            
            # Convert to human-readable format
            bigrams_readable = [' '.join(b) for b in bigrams]
            trigrams_readable = [' '.join(t) for t in trigrams]
            
            return {
                "bigrams": bigrams_readable,
                "trigrams": trigrams_readable
            }
        except Exception as e:
            logging.error(f"Error in collocation extraction: {e}")
            return {"bigrams": [], "trigrams": []}
    
    def combine_keywords(self, tfidf_keywords: List[str], textrank_keywords: List[str], 
                        collocations: Dict[str, List[str]], top_n: int = 15) -> List[str]:
        """
        Combine keywords from different methods and return a curated list
        """
        # Initialize weights for each method
        keyword_scores = defaultdict(float)
        
        # Assign scores based on rank in TF-IDF results
        for i, keyword in enumerate(tfidf_keywords):
            keyword_scores[keyword] += (len(tfidf_keywords) - i) / len(tfidf_keywords)
        
        # Assign scores based on rank in TextRank results
        for i, keyword in enumerate(textrank_keywords):
            keyword_scores[keyword] += (len(textrank_keywords) - i) / len(textrank_keywords)
        
        # Add bigrams and trigrams with a boost
        for bigram in collocations["bigrams"]:
            keyword_scores[bigram] += 0.8
            
        for trigram in collocations["trigrams"]:
            keyword_scores[trigram] += 1.0
        
        # Sort keywords by score
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [k for k, _ in sorted_keywords[:top_n]]


class DirectoryAnalyzer:
    """Class for analyzing directory structure and file content"""
    
    def __init__(self, base_dir: str, output_file: str, max_sample: int = 100):
        self.base_dir = base_dir
        self.output_file = output_file
        self.max_sample = max_sample
        self.text_processor = TextProcessor()
        self.stats = {}
        
    def is_valid_file(self, file_path: str) -> bool:
        """Check if a file is valid for processing"""
        # Check if file exists
        if not os.path.isfile(file_path):
            return False
            
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            return False
            
        # Check file size
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                logging.warning(f"Skipping large file: {file_path} ({size_mb:.2f} MB)")
                return False
        except OSError:
            logging.warning(f"Could not get size of file: {file_path}")
            return False
            
        return True
    
    def extract_file_text(self, file_path: str) -> Optional[str]:
        """Extract text from a file (TXT or PDF)"""
        try:
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
            elif ext.lower() == '.pdf':
                return pdf_extract_text(file_path)
                
            return None
        except Exception as e:
            logging.warning(f"Error reading file {file_path}: {e}")
            return None
    
    def analyze_directory(self, dir_path: str) -> Dict:
        """Analyze a directory and return stats and keywords"""
        logging.info(f"Analyzing directory: {dir_path}")
        
        # Initialize stats
        dir_name = os.path.basename(dir_path)
        stats = {
            "name": dir_name,
            "path": dir_path,
            "file_counts": defaultdict(int),
            "total_files": 0,
            "sampled_files": 0,
            "keywords": [],
            "description": ""
        }
        
        # Get all files in directory (including subdirectories)
        all_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                
                # Count files by extension
                if ext:
                    stats["file_counts"][ext.lower()] += 1
                    stats["total_files"] += 1
                
                # Add valid files to the list for potential sampling
                if self.is_valid_file(file_path):
                    all_files.append(file_path)
        
        # Sample files
        if len(all_files) > self.max_sample:
            sampled_files = random.sample(all_files, self.max_sample)
        else:
            sampled_files = all_files
            
        stats["sampled_files"] = len(sampled_files)
        
        if not sampled_files:
            logging.warning(f"No valid files found in {dir_path}")
            return stats
            
        # Extract text from sampled files
        documents = []
        combined_text = ""
        
        for file_path in tqdm(sampled_files, desc=f"Processing files in {dir_name}", leave=False):
            text = self.extract_file_text(file_path)
            if text and len(text.strip()) > 100:  # Ensure text has some content
                documents.append(text)
                combined_text += text + " "
        
        if not documents:
            logging.warning(f"No text could be extracted from files in {dir_path}")
            return stats
            
        # Extract keywords using TF-IDF
        tfidf_keywords = self.text_processor.extract_keywords_tfidf(documents)
        
        # Extract keywords using TextRank
        textrank_keywords = self.text_processor.extract_keywords_textrank(combined_text)
        
        # Extract collocations
        collocations = self.text_processor.extract_collocations(combined_text)
        
        # Combine keywords
        stats["keywords"] = self.text_processor.combine_keywords(
            tfidf_keywords, textrank_keywords, collocations
        )
        
        # Generate a description based on keywords
        stats["description"] = self.generate_description(stats["keywords"], dir_name)
        
        return stats
    
    def generate_description(self, keywords: List[str], dir_name: str) -> str:
        """Generate a description based on keywords"""
        if not keywords:
            return f"No meaningful keywords could be extracted from the {dir_name} directory."
            
        # Create a human-readable description using the top keywords
        top_keywords = keywords[:5]
        other_keywords = keywords[5:10]
        
        description = f"The {dir_name} directory contains documents primarily focused on "
        description += f"{', '.join(top_keywords[:-1])} and {top_keywords[-1]}. "
        
        if other_keywords:
            description += f"Other relevant topics include {', '.join(other_keywords[:-1])}"
            if len(other_keywords) > 1:
                description += f" and {other_keywords[-1]}."
            else:
                description += "."
                
        return description
    
    def analyze_all_directories(self) -> Dict[str, Dict]:
        """Analyze all subdirectories in the base directory"""
        # Get all immediate subdirectories
        subdirs = [d for d in glob.glob(os.path.join(self.base_dir, "*")) if os.path.isdir(d)]
        
        if not subdirs:
            logging.warning(f"No subdirectories found in {self.base_dir}")
            return {}
        
        # Analyze each subdirectory
        results = {}
        for subdir in tqdm(subdirs, desc="Analyzing directories"):
            dir_stats = self.analyze_directory(subdir)
            results[os.path.basename(subdir)] = dir_stats
            
        return results
    
    def generate_readme(self, results: Dict[str, Dict]) -> None:
        """Generate a README.md file with analysis results"""
        if not results:
            logging.warning("No results to generate README")
            return
            
        # Create README content
        content = [
            "# Resource Directory Analysis",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Overview",
            f"This README provides an analysis of the contents of the RESOURCES directory, "
            f"including file statistics and thematic summaries for each subdirectory.",
            "",
            f"Total subdirectories analyzed: {len(results)}",
            "",
            "## Directory Summaries",
            ""
        ]
        
        # Sort directories alphabetically
        for dir_name, stats in sorted(results.items()):
            content.extend([
                f"### {dir_name}",
                "",
                f"{stats['description']}",
                "",
                "**File Statistics:**",
                ""
            ])
            
            # Add file statistics
            content.append("| File Type | Count |")
            content.append("|-----------|-------|")
            
            for ext, count in sorted(stats["file_counts"].items()):
                content.append(f"| {ext} | {count} |")
                
            content.extend([
                "",
                f"Total files: {stats['total_files']}",
                "",
                "**Key Topics and Themes:**",
                ""
            ])
            
            # Add keywords as a bulleted list
            for keyword in stats["keywords"]:
                content.append(f"- {keyword}")
                
            content.append("")
            content.append("---")
            content.append("")
            
        # Write to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
            
        logging.info(f"README generated at {self.output_file}")


def main():
    """Main function to run the analysis"""
    logging.info("Starting resource directory analysis")
    
    # Create analyzer
    analyzer = DirectoryAnalyzer(
        base_dir=RESOURCES_DIR, 
        output_file=OUTPUT_README,
        max_sample=MAX_SAMPLE_SIZE
    )
    
    # Analyze directories
    results = analyzer.analyze_all_directories()
    
    # Generate README
    analyzer.generate_readme(results)
    
    logging.info("Analysis complete")


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    main()

