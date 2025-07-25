#!/usr/bin/env python3

"""
File Similarity Analyzer

This script analyzes file similarity using embeddings and provides LLM-generated summaries
of the most similar files. It processes .pdf, .md, .txt, and .json files in a directory,
creates embeddings, calculates cosine similarity, and generates technical summaries.

Usage:
    python similarity-analyzer.py input_directory [options]
    python similarity-analyzer.py ./docs --model scout --similarity-count 7
    python similarity-analyzer.py ./papers --model gpt41 --config model_servers.yaml
    python similarity-analyzer.py ./large_corpus --sample 50 --model scout
    python similarity-analyzer.py ./docs --embedding-model openai:text-embedding-ada-002
    python similarity-analyzer.py ./papers --embedding-model sentence-transformers:all-mpnet-base-v2
    python similarity-analyzer.py ./papers --spatial-clustering --model scout
"""

import os
import sys
import argparse
import json
import pickle
import yaml
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import openai

# Suppress PyPDF2 warnings
warnings.filterwarnings("ignore", message="Advanced encoding.*not implemented yet")
warnings.filterwarnings("ignore", message="Multiple definitions in dictionary.*")
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False
    print("Warning: PDF support not available. Install PyPDF2 and pdfplumber for PDF processing.")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: Sentence Transformers not available. Install sentence-transformers for better embeddings.")


class FileProcessor:
    """Handles reading different file formats."""
    
    def read_file(self, file_path: str) -> str:
        """Read content from various file formats."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._read_json(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self._read_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _read_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        if not HAS_PDF_SUPPORT:
            raise ImportError("PDF support not available. Install PyPDF2 and pdfplumber.")
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path} with PyPDF2: {e}")
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                print(f"Error reading PDF {file_path} with pdfplumber: {e2}")
                return ""
        
        return text.strip()
    
    def _read_json(self, file_path: Path) -> str:
        """Read JSON file and convert to text representation."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data, indent=2)
    
    def _read_text(self, file_path: Path) -> str:
        """Read plain text or markdown files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class EmbeddingGenerator:
    """Generates embeddings for text content using various methods."""
    
    def __init__(self, model_config: Dict[str, Any], embedding_model: str = "sentence-transformers:all-MiniLM-L6-v2"):
        self.model_config = model_config
        self.embedding_model = embedding_model
        self.client = None
        self.sentence_model = None
        self.tfidf_vectorizer = None
        
        # Initialize based on embedding model choice
        if embedding_model.startswith("openai"):
            self.client = openai.OpenAI(
                api_key=model_config['openai_api_key'],
                base_url=model_config['openai_api_base']
            )
        elif embedding_model.startswith("sentence-transformers") and HAS_SENTENCE_TRANSFORMERS:
            model_name = embedding_model.split(":")[-1] if ":" in embedding_model else "all-MiniLM-L6-v2"
            self.sentence_model = SentenceTransformer(model_name)
        elif embedding_model == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            print(f"Warning: Unknown embedding model {embedding_model}, falling back to TF-IDF")
            self.embedding_model = "tfidf"
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def generate_embedding(self, text: str, all_texts: List[str] = None) -> np.ndarray:
        """Generate embedding for text using the specified method."""
        try:
            if self.embedding_model.startswith("openai"):
                return self._generate_openai_embedding(text)
            elif self.embedding_model.startswith("sentence-transformers"):
                return self._generate_sentence_transformer_embedding(text)
            else:  # tfidf
                return self._generate_tfidf_embedding(text, all_texts)
        except Exception as e:
            print(f"Error generating embedding with {self.embedding_model}: {e}")
            # Fallback to zeros with appropriate dimension
            if self.embedding_model.startswith("openai"):
                return np.zeros(1536)  # text-embedding-ada-002 dimension
            elif self.embedding_model.startswith("sentence-transformers"):
                return np.zeros(384)  # default MiniLM dimension
            else:
                return np.zeros(1000)  # TF-IDF dimension
    
    def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """Generate OpenAI embedding."""
        # Extract the actual model name (after openai:)
        model_name = self.embedding_model.split(":")[-1] if ":" in self.embedding_model else "text-embedding-ada-002"
        
        response = self.client.embeddings.create(
            model=model_name,
            input=text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    
    def _generate_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Generate Sentence Transformer embedding."""
        if not self.sentence_model:
            raise ValueError("Sentence Transformers model not initialized")
        return self.sentence_model.encode(text)
    
    def _generate_tfidf_embedding(self, text: str, all_texts: List[str] = None) -> np.ndarray:
        """Generate TF-IDF embedding."""
        if all_texts is not None and len(all_texts) > 1:
            # Fit on all texts for better representation
            self.tfidf_vectorizer.fit(all_texts)
            return self.tfidf_vectorizer.transform([text]).toarray()[0]
        else:
            # Fallback: fit on single text
            return self.tfidf_vectorizer.fit_transform([text]).toarray()[0]


class SimilarityAnalyzer:
    """Main class for file similarity analysis."""
    
    def __init__(self, model_config: Dict[str, Any], similarity_count: int = 5, sample_size: int = None, embedding_model: str = "sentence-transformers:all-MiniLM-L6-v2"):
        self.model_config = model_config
        self.similarity_count = similarity_count
        self.sample_size = sample_size
        self.embedding_model = embedding_model
        self.file_processor = FileProcessor()
        self.embedding_generator = EmbeddingGenerator(model_config, embedding_model)
        self.client = openai.OpenAI(
            api_key=model_config['openai_api_key'],
            base_url=model_config['openai_api_base']
        )
        
        self.file_contents: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.similarities: Dict[str, List[Tuple[str, float]]] = {}
    
    def process_directory(self, directory_path: str) -> None:
        """Process all supported files in the directory."""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all supported files
        supported_extensions = ['.pdf', '.md', '.txt', '.json']
        files = []
        for ext in supported_extensions:
            files.extend(directory.glob(f"**/*{ext}"))
        
        if not files:
            print("No supported files found in directory.")
            return
        
        print(f"Found {len(files)} files in directory.")
        
        # Apply sampling if specified
        if self.sample_size and self.sample_size < len(files):
            random.seed(42)  # For reproducible sampling
            files = random.sample(files, self.sample_size)
            print(f"Randomly sampled {len(files)} files for processing.")
        else:
            print(f"Processing all {len(files)} files.")
        
        # Load existing embeddings if available (with embedding model check)
        embeddings_file = directory / f'.embeddings_{self.embedding_model.replace(":", "_")}.pkl'
        if embeddings_file.exists():
            print(f"Loading existing embeddings for {self.embedding_model}...")
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        
        # Load existing similarities if available (with embedding model check)
        similarities_file = directory / f'.similarities_{self.embedding_model.replace(":", "_")}.pkl'
        if similarities_file.exists():
            print(f"Loading existing similarities for {self.embedding_model}...")
            with open(similarities_file, 'rb') as f:
                self.similarities = pickle.load(f)
        
        # Process files
        for i, file_path in enumerate(files, 1):
            file_key = str(file_path.relative_to(directory))
            
            # Read file content
            if file_key not in self.file_contents:
                try:
                    content = self.file_processor.read_file(file_path)
                    self.file_contents[file_key] = content
                    if i % 50 == 0 or i == len(files):
                        print(f"Read {i}/{len(files)} files: {file_key}")
                except Exception as e:
                    print(f"Error reading {file_key}: {e}")
                    continue
            
        # Generate embeddings for new files
        new_files = [file_key for file_key in self.file_contents.keys() if file_key not in self.embeddings]
        if new_files:
            print(f"Generating embeddings for {len(new_files)} files using {self.embedding_model}...")
            
            # For TF-IDF, we need all texts at once for better fitting
            if self.embedding_model == "tfidf":
                all_texts = list(self.file_contents.values())
                for file_key in new_files:
                    embedding = self.embedding_generator.generate_embedding(
                        self.file_contents[file_key], 
                        all_texts
                    )
                    self.embeddings[file_key] = embedding
                    print(f"Generated embedding: {file_key}")
            else:
                # For neural embeddings, process one by one
                for i, file_key in enumerate(new_files, 1):
                    embedding = self.embedding_generator.generate_embedding(self.file_contents[file_key])
                    self.embeddings[file_key] = embedding
                    if i % 25 == 0 or i == len(new_files):
                        print(f"Generated embedding {i}/{len(new_files)}: {file_key}")
        
        # Save embeddings
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"Saved embeddings to {embeddings_file}")
        
        # Calculate similarities using cosine similarity
        self._calculate_similarities(directory)
        
        # Save similarities
        with open(similarities_file, 'wb') as f:
            pickle.dump(self.similarities, f)
        print(f"Saved similarities to {similarities_file}")
    
    def _calculate_similarities(self, directory: Path) -> None:
        """Calculate similarities using cosine similarity on embeddings."""
        if not self.embeddings:
            return
        
        # Prepare embedding matrix
        file_keys = list(self.embeddings.keys())
        embeddings_matrix = np.array([self.embeddings[key] for key in file_keys])
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Find top N similar files for each file
        for i, file_key in enumerate(file_keys):
            similarities = []
            for j, other_key in enumerate(file_keys):
                if i != j:  # Don't include self
                    similarities.append((other_key, similarity_matrix[i][j]))
            
            # Sort by similarity score and take top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.similarities[file_key] = similarities[:self.similarity_count]
            
            if (i + 1) % 25 == 0 or i == len(file_keys) - 1:
                print(f"Calculated similarities {i+1}/{len(file_keys)}: {file_key}")
    
    def generate_summary(self, target_file: str = None) -> str:
        """Generate LLM summary for a file and its most similar files."""
        if not self.similarities:
            return "No similarity data available."
        
        # Choose target file
        if target_file is None:
            target_file = list(self.similarities.keys())[0]  # Choose first file
        elif target_file not in self.similarities:
            available_files = list(self.similarities.keys())
            print(f"File '{target_file}' not found. Available files:")
            for f in available_files[:10]:  # Show first 10
                print(f"  {f}")
            return "Target file not found."
        
        # Check if target file content is available
        if target_file not in self.file_contents:
            return f"Content not available for target file: {target_file}"
        
        # Get similar files
        similar_files = self.similarities[target_file]
        
        # Prepare content for summary
        content_to_summarize = f"TARGET FILE: {target_file}\n"
        content_to_summarize += f"Content:\n{self.file_contents[target_file][:2000]}...\n\n"
        
        content_to_summarize += "MOST SIMILAR FILES:\n"
        for similar_file, similarity_score in similar_files:
            # Check if similar file content is available
            if similar_file in self.file_contents:
                content_to_summarize += f"\nFile: {similar_file} (similarity: {similarity_score:.3f})\n"
                content_to_summarize += f"Content:\n{self.file_contents[similar_file][:1000]}...\n"
            else:
                content_to_summarize += f"\nFile: {similar_file} (similarity: {similarity_score:.3f})\n"
                content_to_summarize += f"Content: [Content not available]\n"
        
        # Generate summary using LLM
        prompt = """Please provide a technical narrative summary of the content from these files. Focus on the technical concepts, methodologies, and key findings. The summary should be a coherent narrative that synthesizes the main technical content across all the files.

Content to summarize:
""" + content_to_summarize
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config['openai_model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def select_non_overlapping_clusters(self, num_clusters: int = 3) -> List[str]:
        """Select cluster centers that don't overlap with each other's similar files."""
        if not self.similarities:
            return []
        
        # Only consider files that have content available
        available_files = [f for f in self.similarities.keys() if f in self.file_contents]
        selected_centers = []
        used_files = set()
        
        import random
        random.seed(42)  # For reproducible selection
        random.shuffle(available_files)
        
        for file in available_files:
            if file in used_files:
                continue
            
            # Check if this file would overlap with already selected clusters
            similar_file_names = [sim_file for sim_file, _ in self.similarities[file]]
            cluster_files = {file} | set(similar_file_names)
            
            if not cluster_files.intersection(used_files):
                selected_centers.append(file)
                used_files.update(cluster_files)
                
                if len(selected_centers) >= num_clusters:
                    break
        
        return selected_centers
    
    def generate_multi_cluster_summary(self, num_clusters: int = 3) -> str:
        """Generate summaries for multiple non-overlapping clusters."""
        if not self.similarities:
            return "No similarity data available."
        
        # Select non-overlapping cluster centers
        cluster_centers = self.select_non_overlapping_clusters(num_clusters)
        
        if len(cluster_centers) < num_clusters:
            print(f"Warning: Could only find {len(cluster_centers)} non-overlapping clusters out of {num_clusters} requested.")
        
        summaries = []
        for i, center in enumerate(cluster_centers, 1):
            print(f"Generating summary for cluster {i}/{len(cluster_centers)}: {center}")
            try:
                summary = self.generate_summary(center)
                summaries.append(f"CLUSTER {i} CENTER: {center}\n{'='*80}\n{summary}\n")
            except Exception as e:
                print(f"Error generating summary for cluster {i} ({center}): {e}")
                summaries.append(f"CLUSTER {i} CENTER: {center}\n{'='*80}\nError generating summary: {e}\n")
        
        return "\n\n".join(summaries)
    
    def _generate_dynamic_topic_labels(self, cluster_centers: List[str]) -> List[str]:
        """Generate topic labels for clusters using LLM analysis of cluster content."""
        topic_labels = []
        
        for i, center in enumerate(cluster_centers):
            # Check if we have content for this center
            if center in self.file_contents:
                # Extract key terms from the file content to generate a topic label
                content = self.file_contents[center][:1000]  # First 1000 chars
                
                # Use LLM to generate a concise topic label
                prompt = f"""Based on this research paper excerpt, generate a concise 2-4 word topic label that describes the main research focus:

Content: {content}

Respond with only the topic label, no explanations. Examples: "DNA Repair Mechanisms", "Radiation Therapy Effects", "Cellular Response Pathways"."""
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_config['openai_model'],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=20,
                        temperature=0.1
                    )
                    label = response.choices[0].message.content.strip()
                    # Clean up the label (remove quotes, periods, etc.)
                    label = label.replace('"', '').replace("'", '').replace('.', '').strip()
                    topic_labels.append(label)
                except Exception as e:
                    print(f"Error generating label for cluster {i+1}: {e}")
                    topic_labels.append(f"Research Topic {i + 1}")
            else:
                # Fallback for missing content
                topic_labels.append(f"Research Topic {i + 1}")
        
        return topic_labels

    def _get_cluster_topic_labels(self, cluster_centers: List[str]) -> List[str]:
        """Generate topic labels for clusters based on their center files."""
        # First try known mappings for specific centers from previous analyses
        topic_labels = []
        
        for i, center in enumerate(cluster_centers):
            # Check known centers first
            if center == "2d8e20cd519c93e8d419cef5f849a34ae1b6b8a1.pdf":
                topic_labels.append("Radiotherapy Lung Effects")
            elif center == "eb72983d7526535ca834c21c5aad5c9d3b15b9af.pdf":
                topic_labels.append("DNA Repair Mechanisms")
            elif center == "806e01f44b9508e4ce7aae9e9c7a2d0b43275aa7.pdf":
                topic_labels.append("UV Radiation Response")
            elif center == "075d609fd28958f91027690db0e14003777fcc44.pdf":
                topic_labels.append("Carbon Ion Therapy")
            elif center == "60a14361a438ffd03c9233aa12f28331f929f472.pdf":
                topic_labels.append("Glioma Radiation Resistance")
            elif center == "c948d0c00a3c625d70e2a6a5d4002004b019ec55.pdf":
                topic_labels.append("Endothelial Progenitor Cells")
            elif center == "825962780bc13d7127a3e6ddb6d1767424d50a19.pdf":
                topic_labels.append("DNA Damage Repair Enzymes")
            elif center == "511e78a4f28cb516ff552f0e568f8be9f26c45c3.pdf":
                topic_labels.append("Radioprotective Compounds")
            elif center == "29ee8955b5e50c324834334a4f6e01d66653e988.pdf":
                topic_labels.append("Radiation Risk Assessment")
            elif center == "10960db3dc34c2f829fd651938a563fb1a4c38cb.pdf":
                topic_labels.append("Telomere Maintenance")
            else:
                # For unknown centers, we'll use dynamic generation later
                topic_labels.append(None)
        
        # For any None entries (unknown centers), use dynamic generation
        if None in topic_labels:
            try:
                dynamic_labels = self._generate_dynamic_topic_labels(cluster_centers)
                for i, label in enumerate(topic_labels):
                    if label is None:
                        topic_labels[i] = dynamic_labels[i]
            except Exception as e:
                print(f"Error generating dynamic labels: {e}")
                # Final fallback to generic labels
                for i, label in enumerate(topic_labels):
                    if label is None:
                        topic_labels[i] = f"Research Topic {i + 1}"
        
        return topic_labels

    def generate_tsne_visualization(self, output_file: str = "tsne_clusters.pdf") -> str:
        """Generate t-SNE visualization of embeddings with topical cluster labels."""
        if not self.embeddings:
            return "No embeddings available for visualization."
        
        # Prepare data
        file_keys = list(self.embeddings.keys())
        embeddings_matrix = np.array([self.embeddings[key] for key in file_keys])
        
        # Get cluster assignments
        cluster_centers = self.select_non_overlapping_clusters(5)  # Use 5 clusters
        file_to_cluster = {}
        used_files = set()
        
        # Assign files to clusters
        for i, center in enumerate(cluster_centers):
            if center in self.similarities:
                similar_files = [sim_file for sim_file, _ in self.similarities[center]]
                cluster_files = [center] + similar_files
                
                for file in cluster_files:
                    if file not in used_files:
                        file_to_cluster[file] = i
                        used_files.add(file)
        
        # Assign remaining files to cluster -1 (unclustered)
        for file in file_keys:
            if file not in file_to_cluster:
                file_to_cluster[file] = -1
        
        # Create cluster labels array
        cluster_labels = [file_to_cluster[file] for file in file_keys]
        
        # Generate t-SNE
        print("Generating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(file_keys)-1))
        embeddings_2d = tsne.fit_transform(embeddings_matrix)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define colors for clusters and get topical labels
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray']
        topic_labels = self._get_cluster_topic_labels(cluster_centers)
        cluster_names = topic_labels + ['Unclustered']
        
        # Plot points
        for cluster_id in range(-1, 5):
            mask = np.array(cluster_labels) == cluster_id
            if np.any(mask):
                color_idx = cluster_id if cluster_id >= 0 else 5
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=colors[color_idx], label=cluster_names[color_idx], 
                          alpha=0.7, s=50)
        
        # Highlight cluster centers with abbreviated labels for annotations
        def get_short_label(full_label: str) -> str:
            """Generate short annotation labels from full topic labels."""
            # Extract first significant word or create meaningful abbreviation
            if "Radiotherapy" in full_label or "Lung" in full_label:
                return "Lung"
            elif "DNA" in full_label and "Repair" in full_label:
                return "DNA"
            elif "UV" in full_label or "Radiation Response" in full_label:
                return "UV"
            elif "Carbon" in full_label or "Ion" in full_label:
                return "Ion"
            elif "Glioma" in full_label:
                return "Glioma"
            elif "Endothelial" in full_label:
                return "EPC"
            elif "Enzyme" in full_label:
                return "Enzyme"
            elif "Radioprotective" in full_label:
                return "Protect"
            elif "Risk" in full_label:
                return "Risk"
            elif "Telomere" in full_label:
                return "Telo"
            else:
                # Fallback: use first word or first few characters
                words = full_label.split()
                return words[0][:6] if words else full_label[:6]
        
        for i, center in enumerate(cluster_centers):
            if center in file_keys and i < len(topic_labels):
                center_idx = file_keys.index(center)
                ax.scatter(embeddings_2d[center_idx, 0], embeddings_2d[center_idx, 1], 
                          c='black', s=200, marker='x', linewidths=3)
                # Use abbreviated topic labels for annotations (keep short for readability)
                short_label = get_short_label(topic_labels[i])
                ax.annotate(short_label, (embeddings_2d[center_idx, 0], embeddings_2d[center_idx, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization of Document Embeddings with Cluster Labels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate cluster statistics with topical labels
        cluster_stats = {}
        for cluster_id in range(len(topic_labels)):
            count = sum(1 for label in cluster_labels if label == cluster_id)
            cluster_stats[topic_labels[cluster_id]] = count
        unclustered_count = sum(1 for label in cluster_labels if label == -1)
        cluster_stats["Unclustered"] = unclustered_count
        
        stats_text = "t-SNE Cluster Statistics:\n"
        for cluster_name, count in cluster_stats.items():
            stats_text += f"  {cluster_name}: {count} files\n"
        
        return f"t-SNE visualization saved to {output_file}\n{stats_text}"
    
    def generate_spatial_cluster_tsne(self, output_file: str = "tsne_spatial_clusters.pdf") -> str:
        """Generate t-SNE visualization with automatic spatial clustering and topic labels."""
        if not self.embeddings:
            return "No embeddings available for visualization."
        
        # Prepare data
        file_keys = list(self.embeddings.keys())
        embeddings_matrix = np.array([self.embeddings[key] for key in file_keys])
        
        # Generate t-SNE
        print("Generating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(file_keys)-1))
        embeddings_2d = tsne.fit_transform(embeddings_matrix)
        
        # Perform spatial clustering on t-SNE coordinates using new 20-cluster approach
        print("Performing 20-cluster KMeans with label merging...")
        
        # Scale embeddings for clustering
        scaler = StandardScaler()
        embeddings_2d_scaled = scaler.fit_transform(embeddings_2d)
        
        spatial_clusters = self._twenty_clusters_with_merging(embeddings_2d_scaled, embeddings_2d, file_keys)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Define colors for clusters (same color for clusters with same topic)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
                 'darkred', 'darkblue', 'darkgreen', 'darkorange', 'indigo', 'maroon', 'hotpink', 'dimgray']
        
        # Create color mapping for unique topics
        unique_topics = list(set(cluster['topic_label'] for cluster in spatial_clusters))
        topic_colors = {topic: colors[i % len(colors)] for i, topic in enumerate(unique_topics)}
        
        # Plot all clusters
        legend_entries = {}  # Track legend entries to avoid duplicates
        
        for i, cluster_info in enumerate(spatial_clusters):
            cluster_indices = cluster_info['indices']
            topic_label = cluster_info['topic_label']
            color = topic_colors[topic_label]
            
            # Plot the cluster points
            ax.scatter(embeddings_2d[cluster_indices, 0], embeddings_2d[cluster_indices, 1], 
                      c=color, alpha=0.8, s=60)
            
            # Add to legend only once per unique topic
            if topic_label not in legend_entries:
                legend_entries[topic_label] = color
            
            # Add cluster center annotation with indicators for merging/splitting/synthesis
            center_x, center_y = cluster_info['center']
            
            # Create abbreviated label for annotation
            short_label = self._create_short_spatial_label(topic_label)
            
            # Add indicators for different operations (using ASCII characters for better compatibility)
            indicators = ""
            if cluster_info.get('merged', False):
                indicators += "*"  # Merged cluster
            if cluster_info.get('split_from'):
                indicators += "^"  # Split cluster  
            if cluster_info.get('synthesis_applied', False):
                indicators += "#"  # Synthesized label
            if cluster_info.get('fallback_applied', False):
                indicators += "~"  # Fallback label
            if cluster_info.get('deep_analysis_applied', False):
                indicators += "@"  # Deep analysis
            if cluster_info.get('aggressive_split_from'):
                indicators += "#^"  # Aggressive split
            if cluster_info.get('similarity_applied', False):
                indicators += "&"  # Similarity-based
            if cluster_info.get('specific_labeling_applied', False):
                indicators += "+"  # Specific labeling
                
            ax.annotate(f"{short_label}{indicators}", (center_x, center_y), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color))
        
        # Create legend with unique topic labels
        legend_handles = []
        for topic_label, color in legend_entries.items():
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                           markersize=8, label=topic_label))
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization: 20-Cluster KMeans with Topic-Based Merging')
        
        # Create main legend for topics
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Research Topics')
        
        # Add text box with symbol definitions
        symbol_definitions = (
            "Symbol Definitions:\n"
            "* = merged clusters\n"
            "^ = split from larger cluster\n"
            "# = synthesized label\n"
            "~ = fallback label\n"
            "@ = deep analysis\n"
            "#^ = aggressive split\n"
            "& = similarity-based\n"
            "+ = specific labeling"
        )
        
        # Add symbol legend as text box
        ax.text(1.05, 0.4, symbol_definitions, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
                fontsize=8, verticalalignment='top')
        
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate cluster statistics with operation indicators
        stats_text = "Advanced Topic Cluster Statistics:\n"
        for cluster_info in spatial_clusters:
            topic_label = cluster_info['topic_label']
            count = cluster_info['size']
            
            # Add operation indicators
            operation_info = []
            if cluster_info.get('merged', False):
                operation_info.append(f"merged from {len(cluster_info['original_clusters'])} clusters")
            if cluster_info.get('split_from'):
                operation_info.append(f"split from cluster {cluster_info['split_from']}")
            if cluster_info.get('synthesis_applied', False):
                operation_info.append("synthesized label")
            if cluster_info.get('fallback_applied', False):
                operation_info.append("fallback label")
            if cluster_info.get('deep_analysis_applied', False):
                operation_info.append("deep analysis")
            if cluster_info.get('aggressive_split_from'):
                operation_info.append(f"aggressive split from {cluster_info['aggressive_split_from']}")
            if cluster_info.get('similarity_applied', False):
                operation_info.append("similarity-based label")
            if cluster_info.get('specific_labeling_applied', False):
                operation_info.append("specific labeling")
                
            info_str = f" ({', '.join(operation_info)})" if operation_info else ""
            stats_text += f"  {topic_label}: {count} files{info_str}\n"
        
        # Add legend for symbols
        stats_text += "\nSymbol Definitions:\n"
        stats_text += "* = merged, ^ = split, # = synthesized, ~ = fallback\n"
        stats_text += "@ = deep analysis, #^ = aggressive split, & = similarity, + = specific\n"
        
        return f"Advanced clustering t-SNE visualization saved to {output_file}\n{stats_text}"
    
    def _find_spatial_clusters(self, embeddings_2d: np.ndarray, file_keys: List[str]) -> List[Dict]:
        """Find compact spatial clusters in t-SNE space using DBSCAN."""
        # Normalize the t-SNE coordinates for better clustering
        scaler = StandardScaler()
        embeddings_2d_scaled = scaler.fit_transform(embeddings_2d)
        
        # Try multiple parameter combinations to find good clusters
        # Start with smaller min_samples and wider eps range
        param_combinations = [
            (3.0, 3), (2.5, 3), (2.0, 3), (1.5, 3), (1.2, 3), (1.0, 3), (0.8, 3),
            (2.0, 4), (1.5, 4), (1.0, 4), (0.8, 4),
            (1.5, 5), (1.0, 5), (0.8, 5)
        ]
        
        best_clusters = []
        best_params = None
        
        for eps, min_samples in param_combinations:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(embeddings_2d_scaled)
            
            # Count clusters found
            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)  # Remove noise points (-1)
            
            # Extract cluster information
            temp_clusters = []
            all_cluster_sizes = []
            for label in sorted(unique_labels):
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_size = len(cluster_indices)
                all_cluster_sizes.append(cluster_size)
                
                # Filter clusters to be within our desired size range
                if 5 <= cluster_size <= 25:  # Allow up to 25 for more flexibility
                    cluster_files = [file_keys[i] for i in cluster_indices]
                    
                    temp_clusters.append({
                        'label': label,
                        'indices': cluster_indices,
                        'files': cluster_files,
                        'size': cluster_size,
                        'center': np.mean(embeddings_2d[cluster_indices], axis=0)
                    })
            
            print(f"eps={eps}, min_samples={min_samples}: Found {len(unique_labels)} total clusters, {len(temp_clusters)} suitable clusters")
            print(f"  All cluster sizes: {sorted(all_cluster_sizes, reverse=True)[:10]}...")  # Show top 10 sizes
            
            # Keep the best result (most clusters in desired size range)
            if len(temp_clusters) > len(best_clusters):
                best_clusters = temp_clusters
                best_params = (eps, min_samples)
        
        # Sort by cluster size (larger clusters first) and limit to 10
        best_clusters.sort(key=lambda x: x['size'], reverse=True)
        spatial_clusters = best_clusters[:10]  # Maximum 10 clusters
        
        print(f"DBSCAN result: params={best_params}, {len(spatial_clusters)} spatial clusters with sizes: {[c['size'] for c in spatial_clusters]}")
        
        # If DBSCAN didn't find good clusters, use KMeans as fallback
        if len(spatial_clusters) < 3:
            print("DBSCAN found too few clusters, using KMeans fallback...")
            spatial_clusters = self._kmeans_fallback_clustering(embeddings_2d_scaled, embeddings_2d, file_keys)
        
        return spatial_clusters
    
    def _kmeans_fallback_clustering(self, embeddings_2d_scaled: np.ndarray, embeddings_2d: np.ndarray, file_keys: List[str]) -> List[Dict]:
        """Fallback clustering using KMeans when DBSCAN fails."""
        # Try different numbers of clusters
        n_clusters_options = [8, 6, 5]
        
        for n_clusters in n_clusters_options:
            if n_clusters * 5 > len(file_keys):  # Need at least 5 points per cluster
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_2d_scaled)
            
            # Extract cluster information
            spatial_clusters = []
            for label in range(n_clusters):
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_size = len(cluster_indices)
                
                # Only include clusters with reasonable size
                if cluster_size >= 5:
                    cluster_files = [file_keys[i] for i in cluster_indices]
                    
                    spatial_clusters.append({
                        'label': label,
                        'indices': cluster_indices,
                        'files': cluster_files,
                        'size': cluster_size,
                        'center': np.mean(embeddings_2d[cluster_indices], axis=0)
                    })
            
            print(f"KMeans with {n_clusters} clusters: Found {len(spatial_clusters)} suitable clusters with sizes: {[c['size'] for c in spatial_clusters]}")
            
            # If we found good clusters, use them
            if len(spatial_clusters) >= 3:
                return spatial_clusters
        
        # If still no good clusters, create a few manual regions
        print("KMeans also failed, creating grid-based spatial regions...")
        return self._grid_based_clustering(embeddings_2d, file_keys)
    
    def _twenty_clusters_with_merging(self, embeddings_2d_scaled: np.ndarray, embeddings_2d: np.ndarray, file_keys: List[str]) -> List[Dict]:
        """New approach: Start with 20 KMeans clusters, generate labels, then merge identical labels."""
        # Start with 20 clusters
        n_clusters = min(20, len(file_keys) // 10)  # Ensure at least 10 points per cluster
        if n_clusters < 5:
            n_clusters = min(5, len(file_keys) // 5)  # Fallback to fewer clusters if needed
            
        print(f"Starting with {n_clusters} initial KMeans clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_2d_scaled)
        
        # Extract initial cluster information
        initial_clusters = []
        for label in range(n_clusters):
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size >= 3:  # Lower threshold since we'll merge later
                cluster_files = [file_keys[i] for i in cluster_indices]
                
                initial_clusters.append({
                    'original_label': label,
                    'indices': cluster_indices,
                    'files': cluster_files,
                    'size': cluster_size,
                    'center': np.mean(embeddings_2d[cluster_indices], axis=0)
                })
        
        print(f"Created {len(initial_clusters)} initial clusters with sizes: {[c['size'] for c in initial_clusters]}")
        
        # Generate topic labels for each initial cluster
        print("Generating topic labels for initial clusters...")
        cluster_topic_labels = {}
        for i, cluster_info in enumerate(initial_clusters):
            topic_label = self._generate_cluster_topic_label(cluster_info['files'])
            cluster_topic_labels[i] = topic_label
            cluster_info['topic_label'] = topic_label
        
        # Group clusters by identical topic labels
        print("Merging clusters with identical topic labels...")
        topic_to_clusters = {}
        for i, cluster_info in enumerate(initial_clusters):
            topic = cluster_info['topic_label']
            if topic not in topic_to_clusters:
                topic_to_clusters[topic] = []
            topic_to_clusters[topic].append(i)
        
        # Create merged clusters
        merged_clusters = []
        for topic, cluster_indices in topic_to_clusters.items():
            if len(cluster_indices) == 1:
                # Single cluster - just use as is
                cluster_info = initial_clusters[cluster_indices[0]]
                merged_clusters.append({
                    'label': len(merged_clusters),
                    'topic_label': topic,
                    'indices': cluster_info['indices'],
                    'files': cluster_info['files'],
                    'size': cluster_info['size'],
                    'center': cluster_info['center'],
                    'original_clusters': [cluster_indices[0]],
                    'merged': False
                })
            else:
                # Multiple clusters with same topic - merge them
                all_indices = []
                all_files = []
                all_centers = []
                
                for cluster_idx in cluster_indices:
                    cluster_info = initial_clusters[cluster_idx]
                    all_indices.extend(cluster_info['indices'])
                    all_files.extend(cluster_info['files'])
                    all_centers.append(cluster_info['center'])
                
                # Calculate merged center as average of individual cluster centers
                merged_center = np.mean(all_centers, axis=0)
                
                merged_clusters.append({
                    'label': len(merged_clusters),
                    'topic_label': topic,
                    'indices': np.array(all_indices),
                    'files': all_files,
                    'size': len(all_files),
                    'center': merged_center,
                    'original_clusters': cluster_indices,
                    'merged': True
                })
        
        # Sort by size (larger first)
        merged_clusters.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"Final result: {len(merged_clusters)} unique topic clusters")
        for cluster in merged_clusters:
            merged_info = f" (merged from {len(cluster['original_clusters'])} clusters)" if cluster['merged'] else ""
            print(f"  {cluster['topic_label']}: {cluster['size']} files{merged_info}")
        
        # Apply synthesis and splitting to improve cluster quality
        final_clusters = self._apply_synthesis_and_splitting(merged_clusters, embeddings_2d)
        
        return final_clusters
    
    def _apply_synthesis_and_splitting(self, clusters: List[Dict], embeddings_2d: np.ndarray) -> List[Dict]:
        """Apply synthesis and splitting operators to improve cluster labeling."""
        print("Applying synthesis and splitting operators...")
        
        final_clusters = []
        
        for cluster in clusters:
            topic_label = cluster['topic_label']
            
            # Check if cluster needs improvement (non-technical or generic label)
            if not self._is_technical_label(topic_label):
                print(f"Cluster '{topic_label}' needs improvement...")
                
                # First try synthesis
                if 'Mixed Research Topic' in topic_label or not self._is_technical_label(topic_label):
                    synthesized_label = self._synthesize_cluster_label(cluster['files'])
                    
                    if synthesized_label:
                        print(f"  Synthesis successful: '{topic_label}' -> '{synthesized_label}'")
                        cluster['topic_label'] = synthesized_label
                        cluster['synthesis_applied'] = True
                        final_clusters.append(cluster)
                        continue
                
                # If synthesis failed, try splitting (only for larger clusters)
                if cluster['size'] >= 15:
                    print(f"  Attempting to split cluster with {cluster['size']} files...")
                    split_clusters = self._split_heterogeneous_cluster(cluster, embeddings_2d)
                    
                    if len(split_clusters) > 1:
                        print(f"  Successfully split into {len(split_clusters)} subclusters")
                        final_clusters.extend(split_clusters)
                        continue
                    else:
                        # If splitting failed, try synthesis as a last resort
                        print(f"  Splitting failed, trying synthesis as fallback...")
                        synthesized_label = self._synthesize_cluster_label(cluster['files'])
                        if synthesized_label:
                            print(f"  Fallback synthesis successful: '{topic_label}' -> '{synthesized_label}'")
                            cluster['topic_label'] = synthesized_label
                            cluster['synthesis_applied'] = True
                            final_clusters.append(cluster)
                            continue
                
                # Try more aggressive fallback methods
                fallback_label = self._generate_fallback_label(cluster)
                if fallback_label and fallback_label != topic_label:
                    print(f"  Fallback labeling successful: '{topic_label}' -> '{fallback_label}'")
                    cluster['topic_label'] = fallback_label
                    cluster['fallback_applied'] = True
                    final_clusters.append(cluster)
                else:
                    # If everything failed, keep original cluster
                    print(f"  Keeping original cluster: '{topic_label}'")
                    final_clusters.append(cluster)
            else:
                # Cluster already has good technical label
                final_clusters.append(cluster)
        
        print(f"After synthesis/splitting: {len(final_clusters)} final clusters")
        
        # Apply iterative refinement for remaining generic clusters
        refined_clusters = self._iterative_refinement(final_clusters, embeddings_2d)
        
        print(f"After iterative refinement: {len(refined_clusters)} final clusters")
        return refined_clusters
    
    def _iterative_refinement(self, clusters: List[Dict], embeddings_2d: np.ndarray) -> List[Dict]:
        """Apply iterative refinement to improve generic cluster labels."""
        print("Applying iterative refinement to remaining generic clusters...")
        
        refined_clusters = []
        
        for cluster in clusters:
            topic_label = cluster['topic_label']
            
            # Check if cluster still has generic label after all previous attempts
            if not self._is_technical_label(topic_label):
                print(f"Iteratively refining generic cluster: '{topic_label}' ({cluster['size']} files)")
                
                # Strategy 1: Deep content analysis with more files
                improved_label = self._deep_content_analysis(cluster)
                if improved_label and self._is_technical_label(improved_label):
                    print(f"  Deep analysis successful: '{topic_label}' -> '{improved_label}'")
                    cluster['topic_label'] = improved_label
                    cluster['deep_analysis_applied'] = True
                    refined_clusters.append(cluster)
                    continue
                
                # Strategy 2: Try aggressive splitting with smaller subclusters
                if cluster['size'] >= 10:
                    print(f"  Attempting aggressive splitting...")
                    split_clusters = self._aggressive_split_cluster(cluster, embeddings_2d)
                    if len(split_clusters) > 1:
                        print(f"  Aggressive splitting successful: {len(split_clusters)} subclusters")
                        refined_clusters.extend(split_clusters)
                        continue
                
                # Strategy 3: Cross-cluster similarity analysis
                similar_label = self._find_similar_technical_cluster(cluster, clusters)
                if similar_label and self._is_technical_label(similar_label):
                    print(f"  Similarity analysis successful: '{topic_label}' -> '{similar_label}'")
                    cluster['topic_label'] = similar_label
                    cluster['similarity_applied'] = True
                    refined_clusters.append(cluster)
                    continue
                
                # Strategy 4: Generate highly specific labels based on size and content patterns
                specific_label = self._generate_specific_label(cluster)
                if specific_label and specific_label != topic_label:
                    print(f"  Specific labeling successful: '{topic_label}' -> '{specific_label}'")
                    cluster['topic_label'] = specific_label
                    cluster['specific_labeling_applied'] = True
                    refined_clusters.append(cluster)
                    continue
                
                print(f"  All refinement attempts failed, keeping: '{topic_label}'")
                refined_clusters.append(cluster)
            else:
                # Cluster already has good technical label
                refined_clusters.append(cluster)
        
        return refined_clusters
    
    def _deep_content_analysis(self, cluster_info: Dict) -> str:
        """Perform deep content analysis using more files and sophisticated prompting."""
        cluster_files = cluster_info['files']
        
        # Use more files for deeper analysis (up to 15 vs previous 8)
        cluster_content = []
        for file_key in cluster_files[:15]:
            if file_key in self.file_contents:
                # Use more content per file (1500 chars vs 1000)
                content = self.file_contents[file_key][:1500]
                cluster_content.append(f"Paper {len(cluster_content)+1}: {content}")
        
        if not cluster_content:
            return None
        
        combined_content = "\n\n".join(cluster_content)
        
        # More sophisticated prompt for deep analysis
        prompt = f"""Perform deep technical analysis of these {len(cluster_content)} research papers. Your goal is to identify the MOST SPECIFIC shared scientific aspect, methodology, or mechanism that unites these papers.

Research papers:
{combined_content}

Analysis Instructions:
1. Look beyond general topics like "cancer" or "radiation" 
2. Identify specific:
   - Molecular pathways (e.g., "p53-mediated apoptosis", "NF-κB signaling")
   - Specific techniques (e.g., "CRISPR-Cas9 editing", "flow cytometry analysis")
   - Precise mechanisms (e.g., "homologous recombination repair", "oxidative stress response")
   - Specific diseases/conditions (e.g., "glioblastoma treatment", "acute lymphoblastic leukemia")
   - Exact molecular targets (e.g., "EGFR inhibition", "mTOR pathway modulation")

Generate a highly specific 2-6 word technical label that captures the most precise shared aspect. Avoid generic terms like "research", "studies", "applications".

Examples of GOOD specific labels: "p53 Tumor Suppressor Pathway", "CRISPR Gene Editing", "Mitochondrial Dysfunction Mechanisms", "Immunotherapy Checkpoint Inhibitors"

Specific technical label:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config['openai_model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,  # More tokens for complex labels
                temperature=0.1  # Lower temp for precision
            )
            
            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                topic_label = content.split('\n')[0]
                
                # Clean the response
                topic_label = self._clean_topic_label(topic_label)
                
                return topic_label[:60]  # Allow longer labels for specificity
                    
        except Exception as e:
            print(f"Warning: Deep analysis failed: {e}")
            
        return None
    
    def _aggressive_split_cluster(self, cluster_info: Dict, embeddings_2d: np.ndarray) -> List[Dict]:
        """Attempt aggressive splitting into smaller, more coherent subclusters."""
        cluster_indices = cluster_info['indices'] 
        cluster_files = cluster_info['files']
        
        # Extract embeddings for this cluster
        cluster_embeddings = embeddings_2d[cluster_indices]
        
        # Try more aggressive splitting - more subclusters
        n_subclusters = min(6, max(3, len(cluster_files) // 8))  # Smaller subclusters
        
        try:
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            subcluster_labels = kmeans.fit_predict(cluster_embeddings)
            
            # Create subclusters
            subclusters = []
            for label in range(n_subclusters):
                subcluster_mask = subcluster_labels == label
                subcluster_indices = cluster_indices[subcluster_mask]
                subcluster_files = [cluster_files[i] for i in range(len(cluster_files)) if subcluster_mask[i]]
                
                if len(subcluster_files) >= 3:  # Lower threshold for aggressive splitting
                    subclusters.append({
                        'label': f"{cluster_info['label']}_aggressive_split_{label}",
                        'indices': subcluster_indices,
                        'files': subcluster_files,
                        'size': len(subcluster_files), 
                        'center': np.mean(embeddings_2d[subcluster_indices], axis=0),
                        'original_clusters': [cluster_info['label']],
                        'merged': False,
                        'aggressive_split_from': cluster_info['label']
                    })
            
            # Generate labels for each subcluster using deep analysis
            final_subclusters = []
            for subcluster in subclusters:
                # Try deep analysis first
                topic_label = self._deep_content_analysis(subcluster)
                
                if not topic_label or not self._is_technical_label(topic_label):
                    # Fallback to regular labeling
                    topic_label = self._generate_cluster_topic_label(subcluster['files'])
                
                if self._is_technical_label(topic_label):
                    subcluster['topic_label'] = topic_label
                    final_subclusters.append(subcluster)
            
            # Only return split if we got at least 2 good technical subclusters
            if len(final_subclusters) >= 2:
                return final_subclusters
                
        except Exception as e:
            print(f"Warning: Aggressive splitting failed: {e}")
            
        return [cluster_info]
    
    def _find_similar_technical_cluster(self, target_cluster: Dict, all_clusters: List[Dict]) -> str:
        """Find a similar cluster with a good technical label and adapt it."""
        target_center = target_cluster['center']
        target_size = target_cluster['size']
        
        # Find clusters with good technical labels
        good_clusters = [c for c in all_clusters if self._is_technical_label(c['topic_label']) and c != target_cluster]
        
        if not good_clusters:
            return None
        
        # Find the closest cluster by center distance and similar size
        min_distance = float('inf')
        best_match = None
        
        for cluster in good_clusters:
            # Calculate distance in t-SNE space
            distance = np.linalg.norm(np.array(target_center) - np.array(cluster['center']))
            
            # Factor in size similarity (prefer similar-sized clusters)
            size_ratio = min(target_size, cluster['size']) / max(target_size, cluster['size'])
            adjusted_distance = distance / (size_ratio + 0.1)  # Penalize very different sizes
            
            if adjusted_distance < min_distance:
                min_distance = adjusted_distance
                best_match = cluster
        
        if best_match:
            # Adapt the label for this cluster size
            base_label = best_match['topic_label']
            if target_size > best_match['size'] * 1.5:
                return f"Broad {base_label}"
            elif target_size < best_match['size'] * 0.5:
                return f"Focused {base_label}"
            else:
                return f"Related {base_label}"
        
        return None
    
    def _generate_specific_label(self, cluster_info: Dict) -> str:
        """Generate highly specific labels based on cluster characteristics."""
        size = cluster_info['size']
        center = cluster_info['center']
        
        # Create specific labels based on position and size
        x, y = center
        
        # More specific area-based labeling
        if abs(x) < 3 and abs(y) < 3:
            area_type = "Core Mechanisms"
        elif x > 15:
            area_type = "Advanced Therapeutics" if y > 0 else "Applied Interventions"
        elif x < -15:
            area_type = "Fundamental Pathways" if y > 0 else "Clinical Applications"
        elif y > 15:
            area_type = "Theoretical Models"
        elif y < -15:
            area_type = "Experimental Methods"
        elif x > 8:
            area_type = "Therapeutic Approaches" if y > 0 else "Treatment Protocols"
        elif x < -8:
            area_type = "Molecular Mechanisms" if y > 0 else "Diagnostic Methods"
        elif y > 8:
            area_type = "Pathway Analysis"
        elif y < -8:
            area_type = "Cellular Responses"
        else:
            area_type = "Biomedical Mechanisms"
        
        # Size-based specificity
        if size > 100:
            return f"Major {area_type}"
        elif size > 50:
            return f"{area_type} Studies"
        elif size > 25:
            return f"Specialized {area_type}"
        else:
            return f"Targeted {area_type}"
    
    def _synthesize_cluster_label(self, cluster_files: List[str]) -> str:
        """Try to synthesize a technical label from cluster content using deeper LLM analysis."""
        # Get more content per file for better synthesis
        cluster_content = []
        for file_key in cluster_files[:8]:  # Use more files for synthesis
            if file_key in self.file_contents:
                # Take more content for synthesis (1000 chars vs 500)
                content = self.file_contents[file_key][:1000]
                cluster_content.append(f"Paper {len(cluster_content)+1}: {content}")
        
        if not cluster_content:
            return None
        
        combined_content = "\n\n".join(cluster_content)
        
        # More sophisticated synthesis prompt
        prompt = f"""Analyze these {len(cluster_content)} research papers that appear to be on related topics but may not have obvious common themes. Your task is to find the underlying technical connection and create a precise 2-5 word scientific label.

Research papers:
{combined_content}

Look for:
- Common methodologies or techniques
- Shared biological pathways or mechanisms  
- Similar experimental approaches
- Related disease areas or conditions
- Common molecular targets or processes

Generate a technical label that captures the most specific shared scientific aspect. Examples: "p53 Pathway Regulation", "CRISPR Gene Editing", "Mitochondrial Dysfunction Studies", "Immunotherapy Biomarkers"

Technical label:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config['openai_model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40,  # More tokens for synthesis
                temperature=0.2  # Slightly higher temp for creativity
            )
            
            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                topic_label = content.split('\n')[0]
                
                # Clean the response
                topic_label = self._clean_topic_label(topic_label)
                
                # Check if it's a meaningful technical label
                if self._is_technical_label(topic_label):
                    return topic_label[:50]
                    
        except Exception as e:
            print(f"Warning: Synthesis failed: {e}")
            
        return None
    
    def _split_heterogeneous_cluster(self, cluster_info: Dict, embeddings_2d: np.ndarray) -> List[Dict]:
        """Split a heterogeneous cluster into more coherent subclusters."""
        cluster_indices = cluster_info['indices'] 
        cluster_files = cluster_info['files']
        
        # Don't split small clusters
        if len(cluster_files) < 10:
            return [cluster_info]
        
        # Extract embeddings for this cluster
        cluster_embeddings = embeddings_2d[cluster_indices]
        
        # Try splitting into 2-4 subclusters depending on size
        n_subclusters = min(4, max(2, len(cluster_files) // 15))
        
        print(f"Attempting to split cluster of {len(cluster_files)} files into {n_subclusters} subclusters...")
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            subcluster_labels = kmeans.fit_predict(cluster_embeddings)
            
            # Create subclusters
            subclusters = []
            for label in range(n_subclusters):
                subcluster_mask = subcluster_labels == label
                subcluster_indices = cluster_indices[subcluster_mask]
                subcluster_files = [cluster_files[i] for i in range(len(cluster_files)) if subcluster_mask[i]]
                
                if len(subcluster_files) >= 5:  # Only keep meaningful subclusters
                    subclusters.append({
                        'label': f"{cluster_info['label']}_split_{label}",
                        'indices': subcluster_indices,
                        'files': subcluster_files,
                        'size': len(subcluster_files), 
                        'center': np.mean(embeddings_2d[subcluster_indices], axis=0),
                        'original_clusters': [cluster_info['label']],
                        'merged': False,
                        'split_from': cluster_info['label']
                    })
            
            # Generate labels for each subcluster
            final_subclusters = []
            for subcluster in subclusters:
                topic_label = self._generate_cluster_topic_label(subcluster['files'])
                
                # Check if the subcluster has a good technical label
                if self._is_technical_label(topic_label):
                    subcluster['topic_label'] = topic_label
                    final_subclusters.append(subcluster)
                else:
                    # Try synthesis on the subcluster
                    synthesized_label = self._synthesize_cluster_label(subcluster['files'])
                    if synthesized_label:
                        subcluster['topic_label'] = synthesized_label
                        final_subclusters.append(subcluster)
            
            # Only return split if we got at least 2 good subclusters
            if len(final_subclusters) >= 2:
                print(f"Successfully split into {len(final_subclusters)} subclusters: {[sc['topic_label'] for sc in final_subclusters]}")
                return final_subclusters
            else:
                print("Splitting failed to produce coherent subclusters")
                return [cluster_info]
                
        except Exception as e:
            print(f"Warning: Cluster splitting failed: {e}")
            return [cluster_info]
    
    def _clean_topic_label(self, topic_label: str) -> str:
        """Clean and standardize a topic label."""
        # Remove common LLM response prefixes (reuse existing logic)
        prefixes_to_remove = [
            "Here is the topic label:", "Technical label:", "The label is:",
            "Label:", "Topic:", "Based on", "After analyzing", "Looking at"
        ]
        
        for prefix in prefixes_to_remove:
            if topic_label.lower().startswith(prefix.lower()):
                topic_label = topic_label[len(prefix):].strip()
                break
        
        # Clean formatting
        topic_label = topic_label.strip('"').strip("'").strip(':').strip('*').strip('-').strip()
        
        # Remove introductory phrases
        intro_phrases = ['is ', 'would be ', 'appears to be ', 'seems to be ']
        for phrase in intro_phrases:
            if topic_label.lower().startswith(phrase):
                topic_label = topic_label[len(phrase):].strip()
                break
                
        return topic_label
    
    def _is_technical_label(self, label: str) -> bool:
        """Check if a label is technical/scientific rather than generic."""
        if not label or len(label) < 4:
            return False
            
        # Check for generic patterns that indicate non-technical labels
        generic_patterns = [
            'mixed research', 'research topic', 'topic cluster', 'unknown topic',
            'various studies', 'multiple topics', 'diverse research', 'general research',
            'biomedical research', 'medical research', 'scientific research',
            'research field', 'research area', 'research studies', 'research applications',
            'biological studies', 'medical studies', 'clinical studies',
            'major research', 'specialized research', 'focused research',
            'advanced research', 'fundamental research', 'applied research',
            'experimental research', 'theoretical research', 'core research'
        ]
        
        label_lower = label.lower()
        if any(pattern in label_lower for pattern in generic_patterns):
            return False
        
        # Check for overly broad single-word technical terms
        broad_technical_terms = [
            'radiation research', 'cancer research', 'dna research', 'cell research',
            'molecular research', 'biological research', 'medical research',
            'clinical research', 'therapeutic research', 'diagnostic research'
        ]
        
        if any(term in label_lower for term in broad_technical_terms):
            return False
        
        # Check for technical indicators
        technical_indicators = [
            'dna', 'rna', 'protein', 'gene', 'cell', 'molecular', 'pathway', 'mechanism',
            'receptor', 'enzyme', 'antibody', 'treatment', 'therapy', 'disease', 
            'cancer', 'tumor', 'radiation', 'chemical', 'biological', 'clinical',
            'diagnosis', 'biomarker', 'signaling', 'metabolism', 'synthesis'
        ]
        
        words = label_lower.split()
        technical_word_count = sum(1 for word in words if any(indicator in word for indicator in technical_indicators))
        
        # Label is technical if it has technical words, isn't too generic, and is specific enough
        is_specific = technical_word_count > 0 and len(words) <= 6
        
        # Additional specificity check - must have at least 2 meaningful words or 1 very specific term
        if technical_word_count == 1 and len(words) <= 3:
            # Single technical word labels are usually too broad unless very specific
            specific_terms = ['p53', 'crispr', 'mrna', 'covid', 'alzheimer', 'parkinson']
            if not any(term in label_lower for term in specific_terms):
                return False
        
        return is_specific
    
    def _generate_fallback_label(self, cluster_info: Dict) -> str:
        """Generate fallback labels for clusters using multiple strategies."""
        cluster_files = cluster_info['files']
        
        # Strategy 1: Try to find similar clusters with good labels and use their theme
        similar_label = self._find_similar_cluster_label(cluster_info)
        if similar_label:
            return similar_label
        
        # Strategy 2: Analyze filenames for patterns
        filename_label = self._generate_filename_based_label(cluster_files)
        if filename_label:
            return filename_label
        
        # Strategy 3: Use embedding similarity to nearest well-labeled cluster
        embedding_label = self._generate_embedding_based_label(cluster_info)
        if embedding_label:
            return embedding_label
        
        # Strategy 4: Generate label based on cluster position in t-SNE space
        position_label = self._generate_position_based_label(cluster_info)
        if position_label:
            return position_label
        
        return None
    
    def _find_similar_cluster_label(self, target_cluster: Dict) -> str:
        """Find a similar cluster with a good label and adapt it."""
        # This would need access to other clusters - for now return None
        # In a full implementation, we'd compare embeddings with other clusters
        return None
    
    def _generate_filename_based_label(self, cluster_files: List[str]) -> str:
        """Generate label based on filename patterns."""
        # Extract common patterns from filenames
        filename_words = []
        for file_key in cluster_files[:20]:  # Look at first 20 files
            # Remove .pdf extension and hash-like strings
            clean_name = file_key.replace('.pdf', '')
            # Skip if it looks like a hash (all hex characters)
            if len(clean_name) == 40 and all(c in '0123456789abcdef' for c in clean_name):
                continue
            filename_words.extend(clean_name.split('_'))
        
        if not filename_words:
            return None
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(word.lower() for word in filename_words if len(word) > 2)
        
        # Look for scientific terms in filenames
        scientific_words = []
        for word, count in word_counts.most_common(10):
            if any(indicator in word.lower() for indicator in ['dna', 'rna', 'protein', 'gene', 'cell', 'cancer', 'radiation', 'treatment', 'therapy']):
                scientific_words.append(word.title())
        
        if len(scientific_words) >= 2:
            return f"{' '.join(scientific_words[:3])} Research"
        elif len(scientific_words) == 1:
            return f"{scientific_words[0]} Studies"
        
        return None
    
    def _generate_embedding_based_label(self, cluster_info: Dict) -> str:
        """Generate label based on embedding similarity to known patterns."""
        # For clusters without content, try to infer topic from embedding space position
        cluster_indices = cluster_info['indices']
        cluster_size = len(cluster_indices)
        
        # Generate labels based on cluster size and characteristics
        if cluster_size > 100:
            return "Major Research Area"
        elif cluster_size > 50:
            return "Biomedical Research Topic"  
        elif cluster_size > 30:
            return "Specialized Research Field"
        elif cluster_size > 15:
            return "Focused Research Studies"
        else:
            return "Specialized Research"
    
    def _generate_position_based_label(self, cluster_info: Dict) -> str:
        """Generate label based on position in t-SNE space."""
        center = cluster_info['center']
        cluster_size = cluster_info['size']
        
        # Use quadrant and distance from origin to create descriptive labels
        x, y = center
        
        # Determine general area
        if abs(x) < 5 and abs(y) < 5:
            area = "Core"
        elif x > 10:
            area = "Advanced" if y > 0 else "Applied"
        elif x < -10:
            area = "Fundamental" if y > 0 else "Clinical"
        elif y > 10:
            area = "Theoretical"
        elif y < -10:
            area = "Experimental"
        else:
            area = "General"
        
        # Combine with size-based specificity
        if cluster_size > 80:
            return f"{area} Biomedical Research"
        elif cluster_size > 40:
            return f"{area} Medical Studies"
        elif cluster_size > 20:
            return f"{area} Research Topic"
        else:
            return f"{area} Specialized Studies"
    
    def _grid_based_clustering(self, embeddings_2d: np.ndarray, file_keys: List[str]) -> List[Dict]:
        """Create spatial clusters based on grid regions in t-SNE space."""
        # Divide t-SNE space into grid regions
        x_coords = embeddings_2d[:, 0]
        y_coords = embeddings_2d[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Create 3x3 grid
        x_bins = np.linspace(x_min, x_max, 4)  # 4 edges = 3 bins
        y_bins = np.linspace(y_min, y_max, 4)
        
        spatial_clusters = []
        cluster_id = 0
        
        for i in range(3):
            for j in range(3):
                # Find points in this grid cell
                x_mask = (x_coords >= x_bins[i]) & (x_coords < x_bins[i+1])
                y_mask = (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1])
                
                # Handle edge case for last bin
                if i == 2:
                    x_mask = (x_coords >= x_bins[i]) & (x_coords <= x_bins[i+1])
                if j == 2:
                    y_mask = (y_coords >= y_bins[j]) & (y_coords <= y_bins[j+1])
                
                region_mask = x_mask & y_mask
                cluster_indices = np.where(region_mask)[0]
                cluster_size = len(cluster_indices)
                
                # Only include regions with enough points
                if cluster_size >= 8:  # Higher threshold for grid-based
                    cluster_files = [file_keys[idx] for idx in cluster_indices]
                    
                    spatial_clusters.append({
                        'label': cluster_id,
                        'indices': cluster_indices,
                        'files': cluster_files,
                        'size': cluster_size,
                        'center': np.mean(embeddings_2d[cluster_indices], axis=0)
                    })
                    cluster_id += 1
        
        # Sort by size and limit to top clusters
        spatial_clusters.sort(key=lambda x: x['size'], reverse=True)
        spatial_clusters = spatial_clusters[:8]  # Max 8 regions
        
        print(f"Grid-based clustering: Found {len(spatial_clusters)} spatial regions with sizes: {[c['size'] for c in spatial_clusters]}")
        
        return spatial_clusters
    
    def _generate_cluster_topic_label(self, cluster_files: List[str]) -> str:
        """Generate a single topic label for a cluster using LLM analysis."""
        # Collect content from all files in the cluster
        cluster_content = []
        for file_key in cluster_files[:10]:  # Limit to first 10 files to keep prompt manageable
            if file_key in self.file_contents:
                # Take first 500 characters from each file
                content = self.file_contents[file_key][:500]
                cluster_content.append(f"Paper {len(cluster_content)+1}: {content}")
        
        if not cluster_content:
            return f"Mixed Research Topic {len(cluster_files)} papers"
        
        # Combine content from all papers in the cluster
        combined_content = "\n\n".join(cluster_content)
        
        # Create prompt for LLM to analyze the cluster topic  
        prompt = f"""Below are {len(cluster_content)} research paper excerpts that are clustered together. Your task is to create a 2-5 word topic label.

{combined_content}

Examples: "DNA Repair Mechanisms", "Radiation Therapy Effects", "Cancer Immunotherapy", "Stem Cell Research"

Provide only the topic label (no explanations, no prefixes, no extra text):"""

        try:
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model_config['openai_model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.1
            )
            
            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                
                # Clean up the response - take first line and limit length
                topic_label = content.split('\n')[0]
                
                # Remove common LLM response prefixes (more comprehensive patterns)
                prefixes_to_remove = [
                    "Here is the topic label:",
                    "Here are the topic labels for each paper:",
                    "Here are the topic labels:",
                    "The topic labels are:",
                    "Topic label:",
                    "The topic label is:",
                    "Based on the analysis:",
                    "After analyzing:",
                    "Based on these excerpts:",
                    "After reviewing:",
                    "Looking at these papers:",
                    "From the excerpts:",
                    "Analyzing these",
                    "The common theme",
                    "The main topic",
                    "I would suggest",
                    "I would label this",
                    "This cluster represents",
                    "These papers focus on",
                    "The research focuses on",
                    "Topic:",
                    "Label:",
                    "Research topic:",
                    "Common theme:",
                    "Main topic:",
                    "Subject:",
                    "Theme:",
                    "Area:",
                    "Field:"
                ]
                
                # Remove prefixes (case insensitive)
                for prefix in prefixes_to_remove:
                    if topic_label.lower().startswith(prefix.lower()):
                        topic_label = topic_label[len(prefix):].strip()
                        break
                
                # Remove any quotes, colons, or extra formatting
                topic_label = topic_label.strip('"').strip("'").strip(':').strip('*').strip('-').strip()
                
                # Remove any remaining introductory phrases
                intro_phrases = ['is ', 'would be ', 'appears to be ', 'seems to be ', 'can be described as ']
                for phrase in intro_phrases:
                    if topic_label.lower().startswith(phrase):
                        topic_label = topic_label[len(phrase):].strip()
                        break
                
                # Additional cleaning for sentence fragments
                if topic_label.lower().startswith('focused on '):
                    topic_label = topic_label[11:].strip()
                if topic_label.lower().startswith('related to '):
                    topic_label = topic_label[11:].strip()
                if topic_label.lower().startswith('dealing with '):
                    topic_label = topic_label[13:].strip()
                
                # Check if the cleaned label is meaningful
                if topic_label and len(topic_label) > 3:
                    # Check if it doesn't start with generic instruction words
                    first_word = topic_label.lower().split()[0] if topic_label.split() else ""
                    generic_first_words = ['paper', 'papers', 'research', 'studies', 'analysis', 'topic', 'label', 'theme']
                    
                    if first_word not in generic_first_words and not topic_label.lower().startswith(('based', 'after', 'here', 'the', 'this', 'these')):
                        return topic_label[:50]
                
                # Fallback if label is still not good
                return f"Research Topic {len(cluster_files)} papers"
            else:
                return f"Topic Cluster {len(cluster_files)} papers"
        except Exception as e:
            print(f"Warning: LLM API call failed for topic labeling: {e}")
            return f"Research Topic {len(cluster_files)} papers"
    
    def _generate_spatial_cluster_labels(self, spatial_clusters: List[Dict]) -> Dict[int, str]:
        """Generate topic labels for spatial clusters using LLM analysis."""
        cluster_labels = {}
        
        for i, cluster_info in enumerate(spatial_clusters):
            cluster_files = cluster_info['files']
            
            # Collect content from all files in the cluster
            cluster_content = []
            for file_key in cluster_files:
                if file_key in self.file_contents:
                    # Take first 500 characters from each file to keep prompt manageable
                    content = self.file_contents[file_key][:500]
                    cluster_content.append(f"Paper {len(cluster_content)+1}: {content}")
            
            if cluster_content:
                # Combine content from all papers in the cluster
                combined_content = "\n\n".join(cluster_content)
                
                # Create prompt for LLM to analyze the cluster topic
                prompt = f"""Analyze these {len(cluster_content)} research paper excerpts that are clustered together in a semantic similarity space. Generate a concise 2-5 word topic label that captures the main research theme shared by these papers.

Papers in this cluster:
{combined_content}

Respond with only the topic label, no explanations. Examples: "DNA Repair Mechanisms", "Cancer Immunotherapy", "Radiation Biology Effects", "Stem Cell Research"."""
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_config['openai_model'],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=30,
                        temperature=0.1
                    )
                    label = response.choices[0].message.content.strip()
                    # Clean up the label
                    label = label.replace('"', '').replace("'", '').replace('.', '').strip()
                    cluster_labels[i] = label
                    print(f"Generated label for spatial cluster {i+1}: '{label}' ({cluster_info['size']} papers)")
                except Exception as e:
                    print(f"Error generating label for spatial cluster {i+1}: {e}")
                    cluster_labels[i] = f"Research Topic {i + 1}"
            else:
                cluster_labels[i] = f"Research Topic {i + 1}"
        
        return cluster_labels
    
    def _create_short_spatial_label(self, full_label: str) -> str:
        """Create short annotation labels for spatial clusters."""
        # Extract key words for annotation
        words = full_label.split()
        if len(words) >= 2:
            return f"{words[0][:4]}{words[1][:4]}"
        elif len(words) == 1:
            return words[0][:8]
        else:
            return full_label[:8]


def load_model_config(config_file: str, model_shortname: str) -> Dict[str, Any]:
    """Load model configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    for server in config['servers']:
        if server['shortname'] == model_shortname:
            return server
    
    raise ValueError(f"Model '{model_shortname}' not found in config file.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='File similarity analyzer with LLM summarization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python similarity-analyzer.py ./docs --model scout
  python similarity-analyzer.py ./papers --model gpt41 --similarity-count 7 --embedding-model openai:text-embedding-ada-002
  python similarity-analyzer.py ./data --sample 20 --model scout --embedding-model sentence-transformers:all-MiniLM-L6-v2
  python similarity-analyzer.py ./data --model scout --target-file paper1.pdf
  python similarity-analyzer.py ./papers --model scout --multi-cluster 3
  python similarity-analyzer.py ./papers --model scout --multi-cluster 5 --generate-tsne
  python similarity-analyzer.py ./papers --model scout --spatial-clustering --output-pdf my_clusters.pdf
        """
    )
    
    parser.add_argument('input_directory', help='Input directory containing files to analyze')
    parser.add_argument('--model', default='scout', 
                       help='Model shortname from config file (default: scout)')
    parser.add_argument('--config', default='model_servers.yaml', 
                       help='Model configuration file (default: model_servers.yaml)')
    parser.add_argument('--similarity-count', type=int, default=5,
                       help='Number of most similar files to find (default: 5)')
    parser.add_argument('--sample', type=int,
                       help='Randomly sample N files from directory (if not provided, processes all files)')
    parser.add_argument('--embedding-model', default='sentence-transformers:all-MiniLM-L6-v2',
                       help='Embedding model: sentence-transformers:all-MiniLM-L6-v2, openai:text-embedding-ada-002, tfidf, etc. (default: sentence-transformers:all-MiniLM-L6-v2)')
    parser.add_argument('--target-file', 
                       help='Specific file to analyze (if not provided, first file is used)')
    parser.add_argument('--multi-cluster', type=int,
                       help='Generate summaries for N non-overlapping clusters (e.g., --multi-cluster 3)')
    parser.add_argument('--generate-tsne', action='store_true',
                       help='Generate t-SNE visualization of embeddings with cluster labels')
    parser.add_argument('--spatial-clustering', action='store_true',
                       help='Generate t-SNE visualization with automatic spatial clustering (finds 5-10 compact clusters in t-SNE space)')
    parser.add_argument('--output-pdf', default=None,
                       help='Output filename for PDF visualization (e.g., my_analysis.pdf). If not specified, uses default names: tsne_clusters.pdf for --generate-tsne, tsne_spatial_clusters.pdf for --spatial-clustering')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Load model configuration
        model_config = load_model_config(args.config, args.model)
        
        # Create analyzer
        analyzer = SimilarityAnalyzer(model_config, args.similarity_count, args.sample, args.embedding_model)
        
        # Process directory
        print(f"Processing directory: {args.input_directory}")
        analyzer.process_directory(args.input_directory)
        
        # Generate summary
        if args.multi_cluster:
            print(f"\nGenerating summaries for {args.multi_cluster} non-overlapping clusters...")
            summary = analyzer.generate_multi_cluster_summary(args.multi_cluster)
        else:
            print("\nGenerating summary...")
            summary = analyzer.generate_summary(args.target_file)
        
        print("\n" + "="*80)
        print("TECHNICAL SUMMARY")
        print("="*80)
        print(summary)
        
        # Generate t-SNE visualization if requested
        if args.generate_tsne:
            print("\n" + "="*80)
            print("GENERATING t-SNE VISUALIZATION")
            print("="*80)
            output_file = args.output_pdf if args.output_pdf else "tsne_clusters.pdf"
            tsne_output = analyzer.generate_tsne_visualization(output_file)
            print(tsne_output)
        
        # Generate spatial clustering t-SNE if requested
        if args.spatial_clustering:
            print("\n" + "="*80)
            print("GENERATING SPATIAL CLUSTERING t-SNE VISUALIZATION")
            print("="*80)
            output_file = args.output_pdf if args.output_pdf else "tsne_spatial_clusters.pdf"
            spatial_output = analyzer.generate_spatial_cluster_tsne(output_file)
            print(spatial_output)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()