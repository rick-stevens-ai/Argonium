#!/usr/bin/env python3
"""
master_extract.py - Master Bioinformatics Workflow Extraction Script

This script combines the best features from multiple extraction scripts:
- YAML-based model configuration from model_servers.yaml
- Rich UI and progress tracking from make_v21.py
- Workflow extraction prompts from new_extract_v3.py
- Workflow Calculus notation from l33_process_summary_to_workflow.py

Usage:
    python master_extract.py input_file [options]
"""

import os
import sys
import argparse
import time
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Rich console imports
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table

# OpenAI and text processing imports
from openai import OpenAI
import tiktoken

# LaTeX processing (optional)
try:
    from pylatexenc.latex2text import LatexNodes2Text
    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False

# PDF processing (optional)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Global console instance
console = Console()

class ExtractionMode(Enum):
    """Extraction mode options."""
    WORKFLOW = "workflow"
    PROBLEM_SUMMARY = "problem_summary"
    BVBRC_MAPPING = "bvbrc_mapping"
    WORKFLOW_CALCULUS = "workflow_calculus"
    CLASSIFICATION = "classification"
    EXPERIMENTAL_PROTOCOL = "experimental_protocol"
    AUTOMATED_PROTOCOL = "automated_protocol"

@dataclass
class ModelConfig:
    """Model configuration from YAML file."""
    server: str
    shortname: str
    openai_api_key: str
    openai_api_base: str
    openai_model: str

@dataclass
class ExtractionConfig:
    """Configuration for extraction process."""
    input_file: str
    output_file: str
    mode: ExtractionMode
    model_config: ModelConfig
    chunk_size: int
    max_tokens: int
    temperature: float
    character_limit: int
    guidance_files: Dict[str, str]
    require_problem: bool = False
    require_tool: bool = False
    batch_mode: bool = False
    input_files: List[str] = None
    worker_count: int = 1

class MasterExtractor:
    """Main extraction class that handles all extraction modes."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.console = Console()
        self.client = None
        self.tokenizer = None
        self.total_chunks = 0
        self.processed_chunks = 0
        self.results = []
        self.results_lock = Lock()
        self.progress_lock = Lock()
        
        # Initialize OpenAI client
        self._initialize_client()
        
        # Initialize tokenizer
        self._initialize_tokenizer()
        
        # Load guidance files
        self.guidance = self._load_guidance_files()
    
    def _initialize_client(self):
        """Initialize OpenAI client with model configuration."""
        try:
            self.client = OpenAI(
                api_key=self.config.model_config.openai_api_key,
                base_url=self.config.model_config.openai_api_base
            )
            self.console.print(f"✅ Initialized client for {self.config.model_config.shortname}")
        except Exception as e:
            self.console.print(f"❌ Failed to initialize client: {e}")
            sys.exit(1)
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for text chunking."""
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            self.console.print("✅ Initialized tokenizer")
        except Exception as e:
            self.console.print(f"❌ Failed to initialize tokenizer: {e}")
            # Fallback to basic chunking
            self.tokenizer = None
    
    def _load_guidance_files(self) -> Dict[str, str]:
        """Load guidance files for different extraction modes."""
        guidance = {}
        
        for name, path in self.config.guidance_files.items():
            if path and os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        guidance[name] = f.read()
                    self.console.print(f"✅ Loaded {name} guidance from {path}")
                except Exception as e:
                    self.console.print(f"⚠️  Failed to load {name} guidance: {e}")
            else:
                self.console.print(f"⚠️  {name} guidance file not found: {path}")
        
        return guidance
    
    def _read_input_file(self) -> str:
        """Read and preprocess input file."""
        return self._read_single_file(self.config.input_file)
    
    def _read_single_file(self, file_path: str) -> str:
        """Read and preprocess a single file based on its type."""
        try:
            file_type = get_file_type(file_path)
            
            if file_type == 'pdf':
                content = read_pdf_file(file_path)
            else:
                # Read text/markdown files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if not content:
                self.console.print(f"⚠️  No content extracted from {file_path}")
                return ""
            
            # Apply character limit
            if len(content) > self.config.character_limit:
                content = content[:self.config.character_limit]
                self.console.print(f"⚠️  Text truncated to {self.config.character_limit} characters for {file_path}")
            
            # Clean LaTeX if available
            if LATEX_AVAILABLE:
                content = LatexNodes2Text().latex_to_text(content)
            
            return content
        except Exception as e:
            self.console.print(f"❌ Failed to read file {file_path}: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        if self.tokenizer:
            # Token-based chunking
            tokens = self.tokenizer.encode(text)
            chunks = []
            for i in range(0, len(tokens), self.config.chunk_size):
                chunk_tokens = tokens[i:i + self.config.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
        else:
            # Character-based chunking fallback
            chunk_size_chars = self.config.chunk_size * 4  # Rough estimate
            chunks = [text[i:i + chunk_size_chars] for i in range(0, len(text), chunk_size_chars)]
        
        return chunks
    
    def _get_prompt_for_mode(self, mode: ExtractionMode, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate appropriate prompt based on extraction mode."""
        
        if mode == ExtractionMode.WORKFLOW:
            return self._get_workflow_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.PROBLEM_SUMMARY:
            return self._get_problem_summary_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.BVBRC_MAPPING:
            return self._get_bvbrc_mapping_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.WORKFLOW_CALCULUS:
            return self._get_workflow_calculus_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.CLASSIFICATION:
            return self._get_classification_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.EXPERIMENTAL_PROTOCOL:
            return self._get_experimental_protocol_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.AUTOMATED_PROTOCOL:
            return self._get_automated_protocol_prompt(chunk_text, chunk_idx, total_chunks)
        else:
            raise ValueError(f"Unknown extraction mode: {mode}")
    
    def _get_workflow_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate workflow extraction prompt (from extract_workflow.py)."""
        return f"""Please extract the bioinformatics workflow used in the following text (part {chunk_idx + 1} of {total_chunks}). 
Identify the primary objective of the author in using the tool, and indicate which services in BV-BRC would be needed to replicate that workflow.

Text:
{chunk_text}"""
    
    def _get_problem_summary_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate problem summary prompt (from extract_problem_summary.py)."""
        return f"""
Identify the primary scientific objective(s) of the author(s) and identify the key bioinformatics analysis needed to meet these objectives.

Please rephrase the goal as a series of questions that could be answered using a powerful bioinformatics environment such as BV-BRC. These questions should be specific search terms and analysis that could be done to meet the goal.

Consider if this problem would make a good tutorial for using the system or a good demonstration for an AI to demonstrate using the tools in an automatic fashion. If you think this is a good example for that, then finish the output with the words "GOOD EXAMPLE".

If the analysis mentioned in the workflow can be done without uploading new data to BV-BRC, then finish the output with the words "POSSIBLY GREAT EXAMPLE"

Text:
{chunk_text}"""
    
    def _get_bvbrc_mapping_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate BV-BRC mapping prompt (from compare_to_bvbrc.py)."""
        bvbrc_services = """
BV-BRC contains the following services and tools:

1. Genome Assembly Service: Allows assembly of bacterial genomes using multiple assemblers for comparison.
2. Genome Annotation Service: Annotates bacterial genomes with RASTtk and viral genomes with VIGOR4.
3. Comprehensive Genome Analysis Service: Performs assembly, annotation, and comparative analysis of bacterial genomes.
4. BLAST Service: Searches against public or private genomes using DNA or protein sequences to find matches.
5. Primer Design Service: Designs primers from input sequences using Primer3 for PCR, hybridization, and sequencing.
6. Similar Genome Finder Service: Finds similar bacterial genomes based on genome distance estimation using Mash/MinHash.
7. Genome Alignment Service: Produces whole-genome alignments of bacterial genomes using progressiveMauve.
8. Variation Analysis Service: Identifies sequence variations by comparing samples to a reference genome.
9. Tn-Seq Analysis Service: Identifies essential genomic regions from transposon insertion sequencing data.
10. Phylogenetic Tree Service: Constructs custom phylogenetic trees using Codon Tree and RAxML methods.
11. Gene Tree Service: Builds phylogenetic trees based on user-selected genomes, genes, or proteins using FastTree or RAxML.
12. MSA and SNP/Variation Analysis Service: Aligns sequences and analyzes SNP variations from input data.
13. Meta-CATS: Compares aligned sequence positions and identifies statistically significant variations.
14. Proteome Comparison Service: Compares protein sequences across multiple genomes using BLASTP.
15. Comparative Systems Service: Compares protein families and pathways across up to 500 bacterial genomes.
16. Taxonomic Classification Service: Classifies metagenomic reads into taxonomic bins using Kraken 2.
17. Metagenomic Binning Service: Bins metagenomic reads or contigs into genome sets from environmental samples.
18. Metagenomic Read Mapping Service: Maps metagenomic reads to genes related to antibiotic resistance and virulence.
19. RNA-Seq Analysis Service: Analyzes RNA-Seq data for differential gene expression using Tuxedo or HISAT2.
20. Expression Import Service: Uploads and analyzes pre-processed differential gene expression datasets.
21. Fastq Utilities Service: Provides FASTQ file operations including trimming, quality checking, and alignment.
22. ID Mapper Tool: Maps BV-BRC identifiers to external databases or vice versa.
23. SARS-CoV-2 Genome Assembly and Annotation Service: Performs assembly, annotation, and variation analysis of SARS-CoV-2 genomes.
24. SARS-CoV-2 Wastewater Analysis Service: Analyzes wastewater samples for SARS-CoV-2 variants using Freyja.
25. Sequence Submission Service: Validates and submits virus sequences to NCBI Genbank.
26. HA Subtype Numbering Conversion Service: Converts HA protein sequence numbering to align with subtype references.
27. Subspecies Classification Service: Assigns viral genotypes/subtypes based on reference tree positions.
28. Genome Browser: Provides a graphical representation of genomic feature alignments.
29. Circular Genome Viewer: Visualizes genome alignments in an interactive circular format.
30. Compare Region Viewer: Identifies and displays proteins from the same family across different genomes.
31. Archaeopteryx.js Phylogenetic Tree Viewer: Interactive display of phylogenetic trees with customization options.
32. Multiple Sequence Alignment Viewer: Visualizes multiple sequence alignments with linked phylogenetic trees.
33. Protein Family Sorter: Examines distribution of protein families across selected genomes for pan-genome analysis.
34. Pathway Comparison Tool: Identifies metabolic pathways across genomes and visualizes them using KEGG maps.
35. Subsystems Data and Viewer: Summarizes gene functionality across genomes with heatmaps and pie charts.
"""
        
        return f"""
{bvbrc_services}

Please analyze the scientific problem and the specific questions in the following text, and make specific suggestions for how the BV-BRC collection of tools could be used to answer the questions. Be as specific as possible about the queries and prompts that would be used to answer the questions using the tools contained in BV-BRC. Where the questions need other tools please indicate so.

Text:
{chunk_text}"""
    
    def _get_workflow_calculus_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate workflow calculus prompt (from l33_process_summary_to_workflow.py)."""
        notation_guidance = self.guidance.get('notation', '[Notation guidance not available]')
        rubric_guidance = self.guidance.get('rubric', '[Rubric guidance not available]')
        
        return f"""
Please read and process the following section of a scientific article.

**Task A – High‑level summary**
1. Summarise the main goal of the paper.
2. Explain briefly how the workflow you will design addresses that goal.

**Task B – Workflow extraction**
For *each* research question or analysis described (even implicitly) in the text, emit a **compact workflow** using the Workflow Calculus v0.9 **exactly** as specified in the Notation Guidance below.
Follow the Extraction **Rubric** strictly – fill the mandatory fields and strive for rubric score 3 where information allows.

* Keep the notation pure UTF‑8 (no LaTeX, no code‑blocks).
* Compose transformations with ∘ or ⇒, parallelism with ∥, iteration with […] as needed.
* Provide an enumerated list *after* the workflows that defines every function / tool symbol you introduced (one‑line each).

---
### Rubric Guidance
{rubric_guidance}
---
### Notation Guidance
{notation_guidance}
---

#### Text chunk {chunk_idx + 1}/{total_chunks}
{chunk_text}
""".strip()
    
    def _get_classification_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate classification prompt (from classify.py)."""
        return f"""Your task is to analyze the provided text and output either 'TOOL' or 'PROBLEM' based on the criteria provided.

- If the text is about a bioinformatics tool or database (not a specific use of that on a problem), output 'TOOL'.
- If the text is about using one or more tools in a scientific workflow to solve a specific problem, output 'PROBLEM'.

Only output 'TOOL' or 'PROBLEM', nothing else.

Here is the text:

{chunk_text}"""
    
    def _get_experimental_protocol_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate experimental protocol extraction prompt."""
        return f"""Please extract all experimental protocols and laboratory procedures mentioned in the following text (part {chunk_idx + 1} of {total_chunks}). 

Focus ONLY on wet lab experimental protocols and procedures, NOT computational steps or bioinformatics analyses.

Use standard protocol names when possible. Examples include:
- PCR amplification
- DNA extraction
- Gel electrophoresis
- Bacterial culture
- Protein purification
- Western blot
- ELISA
- Flow cytometry
- Microscopy
- Sequencing library preparation
- Cell transformation
- Plasmid isolation
- Restriction enzyme digestion
- Ligation
- Transfection
- Immunofluorescence
- RT-PCR
- qPCR
- Cloning
- Mutagenesis

Return the results as a compact bulleted list using standard protocol names. If no experimental protocols are found, return "No experimental protocols identified."

Text:
{chunk_text}"""
    
    def _get_automated_protocol_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate automated protocol mapping prompt."""
        return f"""Please analyze the following scientific text (part {chunk_idx + 1} of {total_chunks}) and create a detailed automation plan for reproducing this research using three types of automated systems:

**FIXED ROBOT SYSTEMS** (Liquid handlers, plate readers, automated pipettes, etc.):
- What wet lab steps could be performed by fixed laboratory automation?
- Specify equipment needed (e.g., Hamilton STAR, Tecan EVO, automated plate readers)
- Detail liquid handling volumes, plate formats, incubation times
- Include quality control checkpoints

**HUMANOID ROBOT SYSTEMS** (General-purpose robots with human-like manipulation):
- What physical manipulations require human-like dexterity?
- Specify manual procedures that could be automated (microscopy setup, gel loading, equipment operation)
- Detail the required sensing and manipulation capabilities
- Include safety protocols and error handling

**AGI COMPUTATIONAL SYSTEMS** (Advanced AI for analysis and decision-making):
- What computational workflows and data analysis could be fully automated?
- Specify bioinformatics pipelines, statistical analyses, and interpretation steps
- Detail decision trees for experimental parameter optimization
- Include automated hypothesis generation and experimental design

For each system type, provide:
1. **Specific Steps**: Numbered list of automated procedures
2. **Required Capabilities**: Technical specifications needed
3. **Integration Points**: How the three systems coordinate
4. **Quality Control**: Automated validation and error detection
5. **Timeline**: Estimated automation time vs. manual time

**Integration Plan**: Describe how all three systems would work together to fully automate the research from start to publication-ready results.

Text:
{chunk_text}"""
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to extract information."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes scientific texts for bioinformatics workflows."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.console.print(f"❌ API call failed: {e}")
            return f"ERROR: {e}"
    
    def _process_chunk_parallel(self, chunk_data: Tuple[int, str, int, int]) -> Tuple[int, str]:
        """Process a single chunk in parallel execution."""
        chunk_idx, chunk_text, chunk_num, total_chunks = chunk_data
        
        try:
            # Generate prompt for current mode
            prompt = self._get_prompt_for_mode(self.config.mode, chunk_text, chunk_num, total_chunks)
            
            # Make API call
            result = self._call_api(prompt)
            
            # Update progress counter (thread-safe)
            with self.progress_lock:
                self.processed_chunks += 1
            
            return (chunk_idx, result)
        except Exception as e:
            error_msg = f"ERROR processing chunk {chunk_idx}: {e}"
            with self.progress_lock:
                self.processed_chunks += 1
            return (chunk_idx, error_msg)
    
    def _classify_document(self, text: str) -> str:
        """Classify document as TOOL or PROBLEM."""
        try:
            # Use a sample of the document for classification (first 2000 characters)
            sample_text = text[:2000] if len(text) > 2000 else text
            
            prompt = f"""Your task is to analyze the provided text and output either 'TOOL' or 'PROBLEM' based on the criteria provided.

- If the text is about a bioinformatics tool or database (not a specific use of that on a problem), output 'TOOL'.
- If the text is about using one or more tools in a scientific workflow to solve a specific problem, output 'PROBLEM'.

Only output 'TOOL' or 'PROBLEM', nothing else.

Here is the text:

{sample_text}"""
            
            response = self.client.chat.completions.create(
                model=self.config.model_config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies scientific texts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0,
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            # Extract TOOL or PROBLEM from the response
            if 'TOOL' in result:
                return 'TOOL'
            elif 'PROBLEM' in result:
                return 'PROBLEM'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            self.console.print(f"❌ Classification failed: {e}")
            return 'ERROR'
    
    def extract(self) -> List[str]:
        """Main extraction method."""
        if self.config.batch_mode:
            return self._extract_batch()
        else:
            return self._extract_single()
    
    def _extract_single(self) -> List[str]:
        """Extract from a single file."""
        self.console.print(f"🚀 Starting extraction in {self.config.mode.value} mode")
        
        # Read input file
        text = self._read_input_file()
        
        # Check if document classification is required
        if self.config.require_problem or self.config.require_tool:
            self.console.print("🔍 Classifying document first...")
            classification = self._classify_document(text)
            
            if self.config.require_problem:
                if classification == 'TOOL':
                    self.console.print("❌ Document classified as TOOL - skipping extraction (use --require-problem flag only for PROBLEM papers)")
                    return ["Document classified as TOOL - extraction skipped"]
                elif classification == 'PROBLEM':
                    self.console.print("✅ Document classified as PROBLEM - proceeding with extraction")
                elif classification == 'ERROR':
                    self.console.print("⚠️  Classification failed - proceeding with extraction anyway")
                else:
                    self.console.print(f"⚠️  Unknown classification result '{classification}' - proceeding with extraction anyway")
                    
            elif self.config.require_tool:
                if classification == 'PROBLEM':
                    self.console.print("❌ Document classified as PROBLEM - skipping extraction (use --require-tool flag only for TOOL papers)")
                    return ["Document classified as PROBLEM - extraction skipped"]
                elif classification == 'TOOL':
                    self.console.print("✅ Document classified as TOOL - proceeding with extraction")
                elif classification == 'ERROR':
                    self.console.print("⚠️  Classification failed - proceeding with extraction anyway")
                else:
                    self.console.print(f"⚠️  Unknown classification result '{classification}' - proceeding with extraction anyway")
        
        # Chunk text
        chunks = self._chunk_text(text)
        self.total_chunks = len(chunks)
        
        self.console.print(f"📄 Processing {self.total_chunks} chunks from {self.config.input_file}")
        
        if self.config.worker_count > 1:
            # Parallel processing
            self.console.print(f"🔄 Using {self.config.worker_count} workers for parallel processing")
            return self._process_chunks_parallel(chunks)
        else:
            # Sequential processing
            return self._process_chunks_sequential(chunks)
    
    def _process_chunks_sequential(self, chunks: List[str]) -> List[str]:
        """Process chunks sequentially (original behavior)."""
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=self.total_chunks)
            
            for idx, chunk in enumerate(chunks):
                progress.update(task, description=f"Processing chunk {idx + 1}/{self.total_chunks}")
                
                # Generate prompt for current mode
                prompt = self._get_prompt_for_mode(self.config.mode, chunk, idx, self.total_chunks)
                
                # Make API call
                result = self._call_api(prompt)
                
                # Store result
                self.results.append(result)
                self.processed_chunks += 1
                
                progress.advance(task)
        
        return self.results
    
    def _process_chunks_parallel(self, chunks: List[str]) -> List[str]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        # Prepare chunk data for parallel processing
        chunk_data = [(idx, chunk, idx, self.total_chunks) for idx, chunk in enumerate(chunks)]
        
        # Initialize results list with placeholders
        results = [None] * len(chunks)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=self.total_chunks)
            
            with ThreadPoolExecutor(max_workers=self.config.worker_count) as executor:
                # Submit all tasks
                future_to_chunk = {executor.submit(self._process_chunk_parallel, data): data for data in chunk_data}
                
                # Process completed tasks
                for future in as_completed(future_to_chunk):
                    chunk_idx, result = future.result()
                    results[chunk_idx] = result
                    progress.advance(task)
        
        # Store results in the main results list
        self.results = results
        return self.results
    
    def _extract_batch(self) -> List[str]:
        """Extract from multiple files in batch mode."""
        self.console.print(f"🚀 Starting batch extraction in {self.config.mode.value} mode")
        self.console.print(f"📁 Processing {len(self.config.input_files)} files")
        
        if self.config.worker_count > 1:
            # Parallel file processing
            self.console.print(f"🔄 Using {self.config.worker_count} workers for parallel processing")
            return self._process_files_parallel()
        else:
            # Sequential file processing
            return self._process_files_sequential()
    
    def _process_files_sequential(self) -> List[str]:
        """Process files sequentially (original behavior)."""
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            files_task = progress.add_task("Processing files", total=len(self.config.input_files))
            
            for file_idx, file_path in enumerate(self.config.input_files):
                progress.update(files_task, description=f"Processing {Path(file_path).name}")
                
                # Read file
                text = self._read_single_file(file_path)
                
                if not text:
                    self.console.print(f"⚠️  Skipping empty file: {file_path}")
                    progress.advance(files_task)
                    continue
                
                # Check if document classification is required
                if self.config.require_problem or self.config.require_tool:
                    classification = self._classify_document(text)
                    
                    if self.config.require_problem and classification == 'TOOL':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as TOOL")
                        progress.advance(files_task)
                        continue
                    elif self.config.require_tool and classification == 'PROBLEM':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as PROBLEM")
                        progress.advance(files_task)
                        continue
                
                # Add file header to results
                file_header = f"\n{'='*60}\nFILE: {Path(file_path).name}\n{'='*60}\n"
                all_results.append(file_header)
                
                # Chunk text
                chunks = self._chunk_text(text)
                
                # Process chunks for this file
                for idx, chunk in enumerate(chunks):
                    # Generate prompt for current mode
                    prompt = self._get_prompt_for_mode(self.config.mode, chunk, idx, len(chunks))
                    
                    # Make API call
                    result = self._call_api(prompt)
                    
                    # Store result
                    all_results.append(result)
                    self.processed_chunks += 1
                
                progress.advance(files_task)
        
        self.results = all_results
        return all_results
    
    def _process_files_parallel(self) -> List[str]:
        """Process files in parallel using ThreadPoolExecutor."""
        # Prepare all chunk data from all files
        all_chunk_data = []
        file_results_mapping = {}
        chunk_idx = 0
        
        for file_idx, file_path in enumerate(self.config.input_files):
            # Read file
            text = self._read_single_file(file_path)
            
            if not text:
                self.console.print(f"⚠️  Skipping empty file: {file_path}")
                continue
            
            # Check if document classification is required
            if self.config.require_problem or self.config.require_tool:
                classification = self._classify_document(text)
                
                if self.config.require_problem and classification == 'TOOL':
                    self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as TOOL")
                    continue
                elif self.config.require_tool and classification == 'PROBLEM':
                    self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as PROBLEM")
                    continue
            
            # Chunk text
            chunks = self._chunk_text(text)
            
            # Record file info and chunk ranges
            file_start_idx = chunk_idx
            file_results_mapping[file_idx] = {
                'file_path': file_path,
                'start_idx': file_start_idx,
                'end_idx': file_start_idx + len(chunks),
                'chunk_count': len(chunks)
            }
            
            # Add chunk data for this file
            for local_idx, chunk in enumerate(chunks):
                all_chunk_data.append((chunk_idx, chunk, local_idx, len(chunks)))
                chunk_idx += 1
        
        if not all_chunk_data:
            self.console.print("⚠️  No files to process after filtering")
            return []
        
        # Initialize results list with placeholders
        results = [None] * len(all_chunk_data)
        
        # Update total chunks for progress tracking
        self.total_chunks = len(all_chunk_data)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=self.total_chunks)
            
            with ThreadPoolExecutor(max_workers=self.config.worker_count) as executor:
                # Submit all tasks
                future_to_chunk = {executor.submit(self._process_chunk_parallel, data): data for data in all_chunk_data}
                
                # Process completed tasks
                for future in as_completed(future_to_chunk):
                    chunk_idx, result = future.result()
                    results[chunk_idx] = result
                    progress.advance(task)
        
        # Reconstruct results organized by file
        all_results = []
        for file_idx in sorted(file_results_mapping.keys()):
            file_info = file_results_mapping[file_idx]
            
            # Add file header
            file_header = f"\n{'='*60}\nFILE: {Path(file_info['file_path']).name}\n{'='*60}\n"
            all_results.append(file_header)
            
            # Add results for this file
            for i in range(file_info['start_idx'], file_info['end_idx']):
                all_results.append(results[i])
        
        self.results = all_results
        return all_results
    
    def save_results(self):
        """Save extraction results to output file."""
        try:
            # Combine all results
            combined_result = "\n\n".join(self.results)
            
            # Save to file
            with open(self.config.output_file, 'w', encoding='utf-8') as f:
                f.write(combined_result)
            
            self.console.print(f"✅ Results saved to {self.config.output_file}")
            
            # Display preview
            self.console.print(Panel(
                Markdown(combined_result[:1000] + "..." if len(combined_result) > 1000 else combined_result),
                title="Extraction Results Preview",
                expand=False
            ))
            
        except Exception as e:
            self.console.print(f"❌ Failed to save results: {e}")

def discover_files(input_path: str, recursive: bool = True) -> List[str]:
    """Discover .txt, .md, and .pdf files in a directory or return single file."""
    supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
    
    if os.path.isfile(input_path):
        # Single file - check if supported
        file_ext = Path(input_path).suffix.lower()
        if file_ext in supported_extensions:
            return [input_path]
        else:
            console.print(f"❌ Unsupported file type: {file_ext}")
            return []
    
    elif os.path.isdir(input_path):
        # Directory - find all supported files
        files = []
        
        if recursive:
            # Recursive search using Path.rglob()
            path_obj = Path(input_path)
            for ext in supported_extensions:
                pattern = f"**/*{ext}"
                files.extend([str(p) for p in path_obj.rglob(f"*{ext}")])
        else:
            # Non-recursive search using glob
            for ext in supported_extensions:
                pattern = os.path.join(input_path, f"*{ext}")
                files.extend(glob.glob(pattern))
        
        # Sort for consistent processing order
        files.sort()
        
        if not files:
            search_type = "recursively" if recursive else "in directory"
            console.print(f"❌ No supported files found {search_type}: {input_path}")
            console.print(f"Supported extensions: {', '.join(supported_extensions)}")
        
        return files
    
    else:
        console.print(f"❌ Path not found: {input_path}")
        return []

def get_file_type(file_path: str) -> str:
    """Determine file type based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.md', '.markdown']:
        return 'markdown'
    else:
        return 'text'

def read_pdf_file(file_path: str) -> str:
    """Extract text from PDF file."""
    if not PDF_AVAILABLE:
        console.print("❌ PDF processing requires PyPDF2: pip install PyPDF2")
        return ""
    
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        console.print(f"❌ Failed to read PDF {file_path}: {e}")
        return ""

def load_model_config(config_file: str, model_shortname: str) -> ModelConfig:
    """Load model configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find model by shortname
        for server in config['servers']:
            if server['shortname'] == model_shortname:
                # Handle environment variables in API key (following PullR pattern)
                openai_api_key_config = server['openai_api_key']
                openai_api_key = None
                
                if openai_api_key_config == "${OPENAI_API_KEY}":
                    openai_api_key = os.environ.get('OPENAI-API-KEY') or os.environ.get('OPENAI_API_KEY')
                    if not openai_api_key:
                        console.print("❌ OpenAI API key is configured to use environment variable "
                                    "'OPENAI-API-KEY' or 'OPENAI_API_KEY', but neither is set.")
                        sys.exit(1)
                elif openai_api_key_config:
                    openai_api_key = openai_api_key_config
                else:
                    console.print(f"❌ 'openai_api_key' not specified for model {model_shortname}")
                    sys.exit(1)
                
                return ModelConfig(
                    server=server['server'],
                    shortname=server['shortname'],
                    openai_api_key=openai_api_key,
                    openai_api_base=server['openai_api_base'],
                    openai_model=server['openai_model']
                )
        
        # If not found, list available models
        available_models = [server['shortname'] for server in config['servers']]
        raise ValueError(f"Model '{model_shortname}' not found. Available models: {available_models}")
        
    except Exception as e:
        console.print(f"❌ Failed to load model config: {e}")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Master bioinformatics workflow extraction script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extraction Modes:
  workflow         - Extract bioinformatics workflows and BV-BRC service requirements
  problem_summary  - Identify scientific objectives and generate research questions
  bvbrc_mapping    - Map scientific problems to specific BV-BRC tools
  workflow_calculus - Generate compact Workflow Calculus descriptions
  classification   - Classify text as TOOL or PROBLEM
  experimental_protocol - Extract experimental protocols and laboratory procedures
  automated_protocol - Map research to automated protocol using fixed robots, humanoid robots, and AGI

Examples:
  python master_extract.py paper.txt --mode workflow --model scout
  python master_extract.py papers/ --mode workflow --model scout --output batch_results.txt
  python master_extract.py papers/ --mode workflow --model scout --no-recursive
  python master_extract.py paper.pdf --mode bvbrc_mapping --model llama --output results.txt
  python master_extract.py paper.txt --mode workflow_calculus --notation notation.txt --rubric rubric.txt
  python master_extract.py papers/ --mode experimental_protocol --model scout --output protocols.txt
  python master_extract.py paper.txt --mode automated_protocol --model scout --output automation_plan.txt
  python master_extract.py paper.txt --mode workflow --require-problem --model scout
  python master_extract.py papers/ --mode workflow --require-tool --model scout
  python master_extract.py paper.txt --mode workflow --model scout --workers 4
  python master_extract.py papers/ --mode workflow --model scout --workers 8 --output batch_results.txt
        """
    )
    
    parser.add_argument('input_path', help='Input file or directory to process (.txt, .md, .pdf files)')
    parser.add_argument('--mode', type=str, 
                       choices=[mode.value for mode in ExtractionMode], 
                       default='workflow', help='Extraction mode (default: workflow)')
    parser.add_argument('--output', default='extraction_results.txt', 
                       help='Output file (default: extraction_results.txt)')
    parser.add_argument('--model', default='scout', 
                       help='Model shortname from config file (default: scout)')
    parser.add_argument('--config', default='model_servers.yaml', 
                       help='Model configuration file (default: model_servers.yaml)')
    parser.add_argument('--chunk-size', type=int, default=3000, 
                       help='Chunk size in tokens/characters (default: 3000)')
    parser.add_argument('--max-tokens', type=int, default=2000, 
                       help='Maximum tokens per API response (default: 2000)')
    parser.add_argument('--temperature', type=float, default=0.0, 
                       help='API temperature (default: 0.0)')
    parser.add_argument('--character-limit', type=int, default=100000, 
                       help='Maximum characters to process (default: 100000)')
    parser.add_argument('--notation', default='notation.txt', help='Notation guidance file (default: notation.txt)')
    parser.add_argument('--rubric', default='rubric.txt', help='Rubric guidance file (default: rubric.txt)')
    parser.add_argument('--require-problem', action='store_true', 
                       help='Only proceed with extraction if paper is classified as PROBLEM (runs classification first)')
    parser.add_argument('--require-tool', action='store_true', 
                       help='Only proceed with extraction if paper is classified as TOOL (runs classification first)')
    parser.add_argument('--no-recursive', action='store_true', 
                       help='Do not search subdirectories recursively (default: recursive search enabled)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker threads for parallel processing (default: 1)')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate mutually exclusive flags
    if args.require_problem and args.require_tool:
        console.print("❌ Error: --require-problem and --require-tool flags cannot be used together")
        sys.exit(1)
    
    # Discover input files
    recursive = not args.no_recursive
    input_files = discover_files(args.input_path, recursive=recursive)
    if not input_files:
        sys.exit(1)
    
    # Determine if batch mode
    batch_mode = len(input_files) > 1
    
    # Welcome message
    classification_filter = "None"
    if args.require_problem:
        classification_filter = "PROBLEM only"
    elif args.require_tool:
        classification_filter = "TOOL only"
    
    console.print(Panel(
        "[bold blue]Master Bioinformatics Workflow Extraction Script[/bold blue]\n"
        f"Mode: {args.mode}\n"
        f"Model: {args.model}\n"
        f"Input: {args.input_path}\n"
        f"Files to process: {len(input_files)}\n"
        f"Batch mode: {'Yes' if batch_mode else 'No'}\n"
        f"Recursive search: {'Yes' if recursive else 'No'}\n"
        f"Workers: {args.workers}\n"
        f"Output: {args.output}\n"
        f"Classification Filter: {classification_filter}",
        title="Configuration",
        expand=False
    ))
    
    # Show discovered files if batch mode
    if batch_mode:
        console.print("📁 Discovered files:")
        for file_path in input_files:
            file_type = get_file_type(file_path)
            console.print(f"  • {Path(file_path).name} ({file_type})")
    
    # Load model configuration
    model_config = load_model_config(args.config, args.model)
    
    # Setup guidance files
    guidance_files = {
        'notation': args.notation,
        'rubric': args.rubric
    }
    
    # Create extraction configuration
    config = ExtractionConfig(
        input_file=input_files[0] if not batch_mode else "",
        output_file=args.output,
        mode=ExtractionMode(args.mode),
        model_config=model_config,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        character_limit=args.character_limit,
        guidance_files=guidance_files,
        require_problem=args.require_problem,
        require_tool=args.require_tool,
        batch_mode=batch_mode,
        input_files=input_files,
        worker_count=args.workers
    )
    
    # Create extractor and run
    extractor = MasterExtractor(config)
    results = extractor.extract()
    extractor.save_results()
    
    # Summary
    if batch_mode:
        console.print(Panel(
            f"✅ Batch extraction complete!\n"
            f"📁 Processed {len(input_files)} files\n"
            f"📊 Processed {extractor.processed_chunks} chunks total\n"
            f"💾 Results saved to {args.output}",
            title="Summary",
            expand=False
        ))
    else:
        console.print(Panel(
            f"✅ Extraction complete!\n"
            f"📊 Processed {extractor.processed_chunks} chunks\n"
            f"💾 Results saved to {args.output}",
            title="Summary",
            expand=False
        ))

if __name__ == "__main__":
    main()