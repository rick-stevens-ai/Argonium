# Argonium Research Pipeline Workflows

This document describes the comprehensive workflows supported by the Argonium research pipeline tools. The system is designed to support end-to-end research workflows from paper discovery and classification to AI model evaluation and reasoning analysis.

## Overview

The Argonium toolkit consists of 14 specialized Python scripts that work together to support various research workflows in scientific literature analysis, AI model evaluation, and automated reasoning systems. The tools are organized around several main workflow categories:

1. **Literature Discovery & Management**
2. **Content Classification & Organization** 
3. **Data Processing & Analysis**
4. **AI Model Evaluation & Benchmarking**
5. **Reasoning Analysis & Validation**

---

## 1. Literature Discovery & Management Workflow

### Primary Tools
- `download_papers_v8.py` - Paper discovery and download
- `analyze_resources.py` - Content analysis and summarization

### Workflow Description
This workflow enables automated discovery, downloading, and initial analysis of academic papers from Semantic Scholar.

#### Process Steps:
1. **Keyword Generation**: Use OpenAI models to generate research keywords from a base topic
2. **Paper Search**: Query Semantic Scholar API with generated or provided keywords
3. **Relevance Filtering**: Use AI models to assess paper relevance to research topics
4. **Content Extraction**: Download and extract text from PDFs and abstracts
5. **Directory Organization**: Automatically organize papers by keyword/topic
6. **Resource Analysis**: Generate comprehensive summaries of paper collections

#### Key Features:
- **AI-Enhanced Search**: Generate related keywords using OpenAI models for comprehensive coverage
- **Intelligent Filtering**: Use AI models to filter papers by relevance before download
- **Bulk Processing**: Handle up to 1000 papers per keyword with rate limiting
- **Multi-format Support**: Process PDFs, text files, and abstracts
- **Automated Organization**: Create topic-based directory structures
- **Progress Tracking**: Resume interrupted downloads and track processed papers

#### Typical Usage:
```bash
# Generate keywords and download papers
python download_papers_v8.py --generate-keywords "antibiotic resistance" --model gpt41 --relevance-model scout --max-relevant-papers 50

# Analyze downloaded paper collections
python analyze_resources.py
```

---

## 2. Content Classification & Organization Workflow

### Primary Tools
- `classify_papers.py` - Document classification
- `split_pdf.py` - PDF processing

### Workflow Description
This workflow provides sophisticated document classification and organization capabilities for research paper collections.

#### Process Steps:
1. **Content Extraction**: Extract text from PDFs, markdown, and text files
2. **AI Classification**: Use predefined topic taxonomies with AI models for classification
3. **Directory Organization**: Create topic-based folder structures
4. **File Management**: Copy classified papers to appropriate directories and move originals to processed folder
5. **Synopsis Generation**: Create summary files with first 500 words for quick reference

#### Supported Classifications:
- Molecular Biology & Microbiology
- Antibiotic Resistance & Development
- Origins of Life & RNA World
- Radiation Biology & Cancer Origins
- Quantum Computing & Materials
- Theoretical Physics & Mathematics
- Computational Science & Methods

#### Key Features:
- **Predefined Taxonomies**: 25+ scientific topic categories
- **AI-Powered Classification**: Use any OpenAI-compatible model for classification
- **Batch Processing**: Handle entire directories with progress tracking
- **File Organization**: Automated copying and archival systems
- **Synopsis Generation**: Create searchable summaries of papers

#### Typical Usage:
```bash
# Classify papers in a directory
python classify_papers.py /path/to/papers --classification-model scout --organize-files

# Split large PDFs for processing
python split_pdf.py large_document.pdf 5
```

---

## 3. Data Processing & Analysis Workflow

### Primary Tools
- `merge_json.py` - Data aggregation
- `merge_incorrect_answers.py` - Error analysis
- `randomize_json.py` - Data randomization
- `sample_json.py` - Data sampling
- `render_json.py` - Report generation
- `merge.py` - Q&A processing

### Workflow Description
This workflow provides comprehensive data processing capabilities for JSON-based datasets, particularly focused on question-answer pairs and model evaluation results.

#### Process Steps:
1. **Data Aggregation**: Merge multiple JSON files into unified datasets
2. **Error Analysis**: Identify and analyze incorrect model responses
3. **Data Preparation**: Randomize, sample, and format data for analysis
4. **Report Generation**: Create formatted markdown reports from JSON data
5. **Quality Control**: Process and clean Q&A pairs

#### Key Features:
- **Flexible Merging**: Combine JSON arrays with error handling
- **Intersection Analysis**: Find questions that multiple models answered incorrectly
- **Statistical Sampling**: Random sampling with configurable sizes
- **Format Conversion**: Convert between different data formats
- **Markdown Generation**: Create readable reports from JSON data

#### Typical Usage:
```bash
# Merge multiple result files
python merge_json.py file1.json file2.json file3.json > combined.json

# Analyze incorrect answers across models
python merge_incorrect_answers.py model1_results.json model2_results.json --mode intersection --format argonium

# Create random sample for testing
python sample_json.py large_dataset.json 100 > test_sample.json
```

---

## 4. AI Model Evaluation & Benchmarking Workflow

### Primary Tools
- `run_all_models.py` - Multi-model benchmarking
- `reprocess_results.py` - Results reanalysis

### Workflow Description
This workflow enables comprehensive evaluation of multiple AI models against standardized benchmarks with detailed performance analysis.

#### Process Steps:
1. **Model Discovery**: Load available models from configuration files
2. **Availability Testing**: Check model accessibility before benchmarking
3. **Parallel Evaluation**: Run benchmarks across multiple models simultaneously
4. **Performance Analysis**: Extract accuracy, confidence, and timing metrics
5. **Comparative Reporting**: Generate ranked performance summaries
6. **Result Reprocessing**: Re-analyze existing results with updated parsing

#### Key Features:
- **Multi-Model Support**: Test any OpenAI-compatible models
- **Availability Checking**: Pre-validate model access before testing
- **Configurable Benchmarks**: Support for multiple-choice and free-form Q&A
- **Parallel Processing**: Configurable worker pools for efficiency
- **Comprehensive Metrics**: Accuracy, confidence, runtime analysis
- **Reproducible Testing**: Seed-based randomization for consistent results

#### Typical Usage:
```bash
# Run comprehensive benchmark across all models
python run_all_models.py questions.json --grader gpt41 --random 100 --parallel 5 --save-incorrect

# Reprocess existing results with updated parsing
python reprocess_results.py benchmark_results.json
```

---

## 5. Reasoning Analysis & Validation Workflow

### Primary Tools
- `reasoning_traces_v6.py` - Detailed reasoning analysis
- `make_v21.py` - Question generation (referenced but not available)

### Workflow Description
This workflow provides deep analysis of AI model reasoning processes, generating detailed internal monologues and meta-cognitive assessments.

#### Process Steps:
1. **Question Processing**: Extract and analyze multiple-choice questions
2. **Persona Generation**: Create expert personas for specialized reasoning
3. **Reasoning Trace Generation**: Generate detailed thought processes for each option
4. **AI-Powered Grading**: Use dedicated models for semantic answer verification
5. **Prediction Analysis**: Extract and validate model predictions with grading results
6. **Confidence Assessment**: Analyze prediction confidence and calibration
7. **Meta-Analysis**: Generate comprehensive reasoning performance reports
8. **Stream-of-Consciousness**: Create coherent internal dialogue narratives

#### Specialized Features:
- **Expert Personas**: Support for microbiologist, physicist, historian, and other specialist perspectives
- **AI-Powered Grading**: Dedicated models for semantic answer verification instead of regex matching
- **Detailed Reasoning**: 150-200 word analysis per option with technical depth
- **Semantic Answer Matching**: Understands answer equivalence ("third option" = "option 3")
- **Dual Prediction Analysis**: Compare detailed vs. quick reasoning approaches
- **Confidence Calibration**: Analysis of prediction confidence vs. actual performance
- **Whole-Trace Analysis**: Meta-cognitive assessment of reasoning patterns
- **Error Recovery**: Robust parsing with fallback text extraction

#### Key Capabilities:
- **Multiple Output Formats**: JSON structured data with fallback text extraction
- **Scientific Depth**: Domain-specific terminology and expert-level reasoning
- **Accuracy Tracking**: Detailed performance metrics with confidence breakdowns
- **Resumable Processing**: Continue interrupted analysis sessions
- **Stream Analysis**: Generate natural internal dialogue from structured reasoning

#### Typical Usage:
```bash
# Generate reasoning traces with microbiologist persona
python reasoning_traces_v6.py questions.json --model gpt41 --specialty microbiologist --max-questions 50 --whole-trace-analysis

# With AI grading model for answer verification
python reasoning_traces_v6.py questions.json --model gpt41 --grading-model scout --specialty physicist --dual-prediction

# Multiple models for different tasks
python reasoning_traces_v6.py questions.json --model claude3 --grading-model gpt41 --whole-trace-model scout --specialty microbiologist

# Continue previous analysis
python reasoning_traces_v6.py questions.json --continue-from previous_results.json --specialty physicist
```

---

## Integrated Workflow Examples

### Complete Research Pipeline
1. **Discovery**: Use `download_papers_v8.py` to find and download papers on a topic
2. **Classification**: Use `classify_papers.py` to organize papers by research area
3. **Analysis**: Use `analyze_resources.py` to generate summaries of paper collections
4. **Question Generation**: Create evaluation questions from the literature
5. **Model Evaluation**: Use `run_all_models.py` to benchmark models on the questions
6. **Reasoning Analysis**: Use `reasoning_traces_v6.py` for detailed cognitive analysis

### Model Comparison Study
1. **Benchmark Preparation**: Use data processing tools to prepare test sets
2. **Multi-Model Testing**: Run `run_all_models.py` for comprehensive evaluation
3. **Error Analysis**: Use `merge_incorrect_answers.py` to identify challenging questions
4. **Deep Reasoning**: Use `reasoning_traces_v6.py` on difficult cases
5. **Report Generation**: Use `render_json.py` to create formatted analysis reports

### Literature Meta-Analysis
1. **Bulk Collection**: Use `download_papers_v8.py` with keyword generation
2. **Smart Classification**: Use `classify_papers.py` with AI-powered categorization
3. **Content Analysis**: Use `analyze_resources.py` for comprehensive summaries
4. **Quality Assessment**: Use sampling tools to validate classification accuracy
5. **Trend Analysis**: Use data processing tools to identify research patterns

---

## Configuration and Extensibility

### Model Configuration
All tools support flexible model configuration through `model_servers.yaml`, enabling:
- Custom OpenAI-compatible endpoints
- Environment variable API key management
- Model-specific parameter handling
- Reasoning model support (o3, o4mini)

### Workflow Customization
The modular design allows for:
- Custom classification taxonomies
- Specialized expert personas
- Configurable processing parameters
- Extensible data formats
- Custom evaluation metrics

### Integration Points
Tools are designed to work together through:
- Standardized JSON data formats
- Consistent directory structures
- Shared configuration systems
- Composable command-line interfaces
- Progress tracking and resumption capabilities

---

## Technical Requirements

### Dependencies
- Python 3.7+
- OpenAI API access
- PyPDF2 for PDF processing
- NLTK for text processing
- scikit-learn for TF-IDF analysis
- NetworkX for TextRank algorithms
- YAML for configuration
- tqdm for progress tracking

### Performance Considerations
- Rate limiting for API calls
- Parallel processing support
- Progress saving and resumption
- Configurable timeouts
- Memory-efficient processing
- Batch operation support

This comprehensive toolkit enables sophisticated research workflows combining literature discovery, AI evaluation, and cognitive analysis in a unified, extensible framework.