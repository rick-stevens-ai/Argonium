# Argonium Research Pipeline

A comprehensive toolkit for scientific literature analysis, AI model evaluation, and cognitive reasoning assessment. The Argonium pipeline enables end-to-end research workflows from paper discovery to detailed model reasoning analysis.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **📚 Literature Discovery**: Automated paper search, download, and AI-powered relevance filtering with PDF validation
- **🏷️ Content Classification**: AI-powered categorization using 25+ predefined scientific taxonomies
- **📊 Data Processing**: Comprehensive JSON manipulation, sampling, and analysis tools with reproducible randomization
- **🤖 Model Benchmarking**: Multi-model evaluation with parallel processing and availability checks
- **⚡ Advanced Scoring**: High-performance parallel scoring engine with AI-powered grading
- **🧠 Reasoning Analysis**: Deep cognitive assessment with expert persona modeling and meta-analysis
- **📄 Document Processing**: PDF validation, splitting, and integrity checking
- **🔧 Utility Tools**: Data cleaning, format conversion, and report generation
- **🔄 Integrated Workflows**: Complete research pipelines from discovery to analysis with automation scripts

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- OpenAI API key or compatible model endpoints
- Internet connection for paper downloads

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/argonium.git
   cd argonium
   ```

2. **Install dependencies**:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Configure your models**:
   ```bash
   cp model_servers.yaml.example model_servers.yaml
   # Edit model_servers.yaml with your API endpoints and keys
   ```

4. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export SS_API_KEY="your-semantic-scholar-api-key"  # Optional but recommended
   ```

### Basic Usage

**Complete Research Pipeline** (Recommended for new users):
```bash
./scripts/complete_research_pipeline.sh \
    --topic "antibiotic resistance" \
    --specialty microbiologist \
    --quick
```

**Individual Workflows**:

```bash
# Literature discovery and analysis
./scripts/literature_discovery.sh \
    --keyword "quantum computing" \
    --generate-keywords \
    --max-papers 50 \
    --organize

# Papers to reasoning traces (simple)
./scripts/papers_to_reasoning_simple.sh \
    --papers-dir ./research_papers \
    --specialty microbiologist \
    --model gpt41

# Multi-model benchmarking
./scripts/model_benchmark.sh \
    --questions questions.json \
    --sample-size 100 \
    --save-incorrect

# Detailed reasoning analysis
./scripts/reasoning_analysis.sh \
    --questions questions.json \
    --specialty physicist \
    --whole-trace
```

## 📋 Workflow Overview

### 1. Literature Discovery & Management
- **Primary Tool**: `download_papers_v8.py`
- **Workflow Script**: `scripts/literature_discovery.sh`
- **Purpose**: Discover, download, and organize academic papers from Semantic Scholar

**Key Features**:
- AI-enhanced keyword generation
- Relevance filtering using language models
- Automatic paper organization by topic
- Progress tracking and resumable downloads

**Example**:
```bash
python download_papers_v8.py \
    --generate-keywords "machine learning" \
    --model gpt41 \
    --relevance-model scout \
    --max-relevant-papers 50
```

### 2. Content Classification & Organization
- **Primary Tool**: `classify_papers.py`
- **Purpose**: AI-powered classification using scientific taxonomies

**Supported Classifications**:
- Molecular Biology & Microbiology
- Antibiotic Resistance & Development
- Quantum Computing & Materials
- Theoretical Physics & Mathematics
- And 20+ more categories

**Example**:
```bash
python classify_papers.py /path/to/papers \
    --classification-model scout \
    --organize-files
```

### 3. Model Evaluation & Benchmarking
- **Primary Tool**: `run_all_models.py`
- **Workflow Script**: `scripts/model_benchmark.sh`
- **Purpose**: Comprehensive multi-model evaluation

**Example**:
```bash
python run_all_models.py questions.json \
    --grader gpt41 \
    --random 100 \
    --parallel 10 \
    --save-incorrect
```

### 4. Advanced Model Scoring & Evaluation
- **Primary Tool**: `argonium_score_parallel_v9.py`
- **Purpose**: Advanced multi-model scoring with parallel processing and detailed grading

**Key Features**:
- **Parallel Processing**: Multi-threaded evaluation for faster processing
- **Auto-Format Detection**: Automatically detects multiple-choice vs. free-form questions
- **Advanced Grading**: Uses AI models to grade responses with detailed evaluation
- **Error Analysis**: Saves incorrect answers for detailed analysis
- **Reproducible Sampling**: Random sampling with seed support

**Example**:
```bash
# Basic parallel scoring
python argonium_score_parallel_v9.py questions.json \
    --model gpt41 \
    --grader scout \
    --parallel 8 \
    --save-incorrect

# Random sampling with reproducible results
python argonium_score_parallel_v9.py questions.json \
    --model claude3 \
    --grader gpt41 \
    --random 50 \
    --seed 42
```

### 5. Reasoning Analysis & Validation
- **Primary Tool**: `reasoning_traces_v6.py`
- **Workflow Script**: `scripts/reasoning_analysis.sh`
- **Purpose**: Deep cognitive analysis with expert personas and AI-powered answer verification

**Key Features**:
- **AI-Powered Grading**: Use separate models for answer verification instead of regex matching
- **Expert Persona Modeling**: Detailed reasoning from domain specialists
- **Semantic Answer Matching**: Understands answer equivalence ("third option" = "option 3")
- **Dual Prediction Analysis**: Compare detailed vs. quick reasoning approaches
- **Flexible Verification**: Falls back to regex if grading model not available

**Expert Specialties**:
- Microbiologist
- Physicist (General & Quantum)
- Historian
- Generic Expert

**Examples**:
```bash
# Basic reasoning analysis
python reasoning_traces_v6.py questions.json \
    --model gpt41 \
    --specialty microbiologist \
    --whole-trace-analysis

# With AI grading model for answer verification
python reasoning_traces_v6.py questions.json \
    --model gpt41 \
    --grading-model scout \
    --specialty physicist \
    --dual-prediction

# Multiple models for different tasks
python reasoning_traces_v6.py questions.json \
    --model claude3 \
    --grading-model gpt41 \
    --whole-trace-model scout \
    --specialty microbiologist
```

### 6. Document Similarity & Clustering Analysis
- **Primary Tool**: `similarity_analyzer.py`
- **Purpose**: Advanced document similarity analysis with semantic embeddings and AI-powered clustering

**Key Features**:
- **Multiple Embedding Models**: Support for Sentence Transformers, OpenAI embeddings, and TF-IDF
- **Semantic Clustering**: Automatic document clustering with t-SNE visualization
- **AI-Powered Topic Labeling**: LLM-generated cluster labels and summaries
- **Multi-Format Support**: Process PDF, TXT, MD, and JSON files
- **Advanced Visualization**: Interactive t-SNE plots with spatial clustering
- **Comprehensive Analysis**: Generate detailed technical summaries of document clusters

**Example**:
```bash
# Basic similarity analysis with visualization
python similarity_analyzer.py ./research_papers \
    --model scout \
    --embedding-model sentence-transformers:all-MiniLM-L6-v2 \
    --spatial-clustering \
    --output-pdf cluster_analysis.pdf

# Multi-cluster analysis with OpenAI embeddings
python similarity_analyzer.py ./documents \
    --model gpt41 \
    --embedding-model openai:text-embedding-ada-002 \
    --multi-cluster 5 \
    --generate-tsne
```

### 7. Data Processing & Analysis
**JSON Manipulation & Reporting:**
- `merge_json.py` - Combine multiple JSON files with advanced options
- `merge_incorrect_answers.py` - Analyze model errors and find intersections
- `sample_json.py` - Random sampling from JSON datasets
- `render_json.py` - Generate formatted markdown reports
- `randomize_json.py` - Randomize JSON data for testing
- `reprocess_results.py` - Post-process evaluation results

**Document & PDF Processing:**
- `split_pdf.py` - Split large PDF files into smaller chunks
- `validate_pdf_quick.py` - Fast PDF validation and integrity checks
- `paper_syn_org.py` - Organize paper synopses and metadata
- `select_interesting_papers.py` - Filter and select relevant papers
- `similarity_analyzer.py` - Advanced document similarity analysis and clustering

**Data Cleaning & Utilities:**
- `cleanup_mc.py` - Clean and standardize multiple choice questions
- `merge.py` - Basic file merging operations

## 🛠️ Core Tools Reference

### Primary Pipeline Tools
| Tool | Purpose | Key Features |
|------|---------|-------------|
| `download_papers_v8.py` | Paper discovery | AI keyword generation, relevance filtering, PDF validation |
| `classify_papers.py` | Content classification | 25+ scientific categories, AI-powered organization |
| `analyze_resources.py` | Content analysis | TF-IDF, TextRank, comprehensive summaries |
| `make_v21.py` | Question generation | Extract questions from papers, chunk processing |
| `run_all_models.py` | Model benchmarking | Multi-model, parallel evaluation, availability checks |
| `argonium_score_parallel_v9.py` | Advanced scoring | Parallel processing, AI grading, error analysis |
| `reasoning_traces_v6.py` | Reasoning analysis | Expert personas, AI grading, dual prediction, meta-analysis |
| `similarity_analyzer.py` | Document clustering | Semantic embeddings, t-SNE visualization, AI topic labeling |

### Data Processing & Utilities
| Tool | Purpose | Key Features |
|------|---------|-------------|
| `merge_incorrect_answers.py` | Error analysis | Intersection analysis, difficulty assessment |
| `merge_json.py` | JSON manipulation | Combine datasets, filtering options |
| `sample_json.py` | Data sampling | Random sampling with seed support |
| `render_json.py` | Report generation | Markdown formatting, structured output |
| `reprocess_results.py` | Results processing | Post-evaluation analysis and cleanup |
| `split_pdf.py` | PDF processing | Split large documents, batch processing |
| `validate_pdf_quick.py` | PDF validation | Fast integrity checks, corruption detection |
| `similarity_analyzer.py` | Document similarity | Multiple embeddings, clustering, visualization |
| `cleanup_mc.py` | Data cleaning | Multiple choice standardization |

## 📁 Project Structure

```
argonium/
├── scripts/                        # Workflow automation scripts
│   ├── complete_research_pipeline.sh
│   ├── literature_discovery.sh
│   ├── model_benchmark.sh
│   ├── reasoning_analysis.sh
│   └── papers_to_reasoning_simple.sh
├── # Core Pipeline Tools
├── download_papers_v8.py          # Paper discovery and download with PDF validation
├── classify_papers.py             # AI-powered classification and organization
├── analyze_resources.py           # Literature analysis and summarization
├── make_v21.py                     # Question generation from papers
├── run_all_models.py              # Multi-model benchmarking with availability checks
├── argonium_score_parallel_v9.py  # Advanced parallel scoring engine
├── reasoning_traces_v6.py         # Detailed reasoning analysis with expert personas
├── similarity_analyzer.py         # Document similarity analysis and clustering
├── # Data Processing & Analysis Tools
├── merge_json.py                   # Advanced JSON file combination
├── merge_incorrect_answers.py     # Model error analysis and intersections
├── sample_json.py                  # Random sampling with reproducible seeds
├── render_json.py                  # Formatted report generation
├── randomize_json.py              # JSON data randomization
├── reprocess_results.py           # Post-evaluation result processing
├── # Document & PDF Processing
├── split_pdf.py                    # PDF splitting and batch processing
├── validate_pdf_quick.py          # Fast PDF validation and integrity checks
├── paper_syn_org.py               # Paper synopsis organization
├── select_interesting_papers.py   # Paper filtering and selection
├── # Utilities & Cleanup
├── cleanup_mc.py                   # Multiple choice question standardization
├── merge.py                        # Basic file merging operations
├── # Configuration & Setup
├── model_servers.yaml.example     # Model configuration template
├── requirements.txt               # Python dependencies
├── install.sh                     # Installation script
└── workflow.md                    # Detailed workflow documentation
```

## ⚙️ Configuration

### Model Configuration

Create `model_servers.yaml` from the example template:

```yaml
servers:
  - shortname: "gpt41"
    server: "OpenAI GPT-4"
    openai_api_key: "${OPENAI_API_KEY}"
    openai_api_base: "https://api.openai.com/v1"
    openai_model: "gpt-4"
  
  - shortname: "scout"
    server: "Custom endpoint"
    openai_api_key: "${CUSTOM_API_KEY}"
    openai_api_base: "https://your-endpoint.com/v1"
    openai_model: "your-model-name"
```

### Environment Variables

```bash
# Required for most functionality
export OPENAI_API_KEY="your-openai-api-key"

# Optional but recommended for literature discovery
export SS_API_KEY="your-semantic-scholar-api-key"

# For custom model endpoints
export CUSTOM_API_KEY="your-custom-api-key"
```

## 🔬 Example Research Workflows

### Academic Literature Review
```bash
# 1. Discover and organize papers
./scripts/literature_discovery.sh \
    --keyword "CRISPR gene editing" \
    --generate-keywords \
    --max-papers 100 \
    --organize

# 2. Generate questions from literature
# (Manual step - create questions.json from papers)

# 3. Evaluate models on domain knowledge
./scripts/model_benchmark.sh \
    --questions questions.json \
    --sample-size 50 \
    --save-incorrect
```

### Advanced Model Evaluation
```bash
# 1. High-throughput parallel scoring
python argonium_score_parallel_v9.py questions.json \
    --model gpt41 \
    --grader scout \
    --parallel 12 \
    --save-incorrect

# 2. Random sampling for quick evaluation
python argonium_score_parallel_v9.py questions.json \
    --model claude3 \
    --grader gpt41 \
    --random 100 \
    --seed 42

# 3. Compare different grading approaches
python argonium_score_parallel_v9.py questions.json \
    --model llama2 \
    --grader gpt41 \
    --format mc \
    --parallel 8
```

### Model Reasoning Assessment
```bash
# 1. Analyze reasoning with domain expertise and AI grading
python reasoning_traces_v6.py questions.json \
    --model gpt41 \
    --grading-model scout \
    --specialty microbiologist \
    --max-questions 100 \
    --whole-trace-analysis

# 2. Compare detailed vs. quick reasoning approaches
python reasoning_traces_v6.py questions.json \
    --model claude3 \
    --grading-model gpt41 \
    --specialty physicist \
    --dual-prediction \
    --max-questions 100

# 3. Multi-model reasoning analysis
python reasoning_traces_v6.py questions.json \
    --model llama2 \
    --grading-model gpt41 \
    --whole-trace-model scout \
    --specialty microbiologist
```

### Complete Research Pipeline
```bash
# End-to-end workflow in one command
./scripts/complete_research_pipeline.sh \
    --topic "quantum machine learning" \
    --specialty "quantum physicist" \
    --max-papers 50 \
    --sample-questions 75
```

### Data Processing & Utilities
```bash
# Combine multiple JSON datasets
python merge_json.py dataset1.json dataset2.json --output combined.json

# Random sampling for testing
python sample_json.py large_dataset.json 100 --seed 42 > sample.json

# Generate formatted reports
python render_json.py results.json --output report.md

# Validate PDF integrity
python validate_pdf_quick.py papers/*.pdf

# Split large PDFs for processing
python split_pdf.py large_paper.pdf --max-pages 20

# Analyze document similarity and clustering
python similarity_analyzer.py ./research_papers --model scout --spatial-clustering

# Clean multiple choice questions
python cleanup_mc.py messy_questions.json --output clean_questions.json
```

## 📊 Output Formats

### Literature Analysis
- **README.md**: Comprehensive literature summaries
- **Topic directories**: Organized paper collections
- **Synopsis files**: Quick paper overviews

### Model Evaluation
- **JSON results**: Detailed performance metrics with confidence scores
- **Markdown reports**: Human-readable summaries with accuracy breakdowns
- **Error analysis**: Incorrect answer collections for pattern analysis
- **Parallel scoring logs**: Real-time progress tracking and performance stats

### Reasoning Analysis
- **Detailed traces**: Complete thought processes
- **Meta-analysis**: Cognitive pattern assessment
- **Sample extracts**: Notable reasoning examples

### Document Similarity Analysis
- **t-SNE visualizations**: Interactive cluster plots with topic labels
- **Similarity matrices**: Cosine similarity between documents
- **Cluster summaries**: AI-generated technical summaries of document groups
- **Embedding caches**: Persistent storage for faster re-analysis

## 🚨 Common Issues & Solutions

### Installation Issues
```bash
# If pip install fails, try:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# For PDF processing issues:
# macOS: brew install poppler
# Ubuntu: sudo apt-get install poppler-utils

# For similarity analysis dependencies:
pip install sentence-transformers scikit-learn matplotlib
```

### API Configuration
```bash
# Verify your API keys are set:
echo $OPENAI_API_KEY
echo $SS_API_KEY

# Test model accessibility:
python -c "import openai; print('OpenAI configured')"
```

### Rate Limiting
- Use `--parallel` parameter to control concurrent requests
- Add delays between API calls for high-volume processing
- Consider using multiple API keys for increased throughput

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```bash
# Clone for development
git clone https://github.com/yourusername/argonium.git
cd argonium

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Semantic Scholar API** for literature access
- **OpenAI** for language model capabilities
- **Scientific community** for open research practices
- **Contributors** who help improve this toolkit

## 📞 Support

- **Documentation**: See `workflow.md` for detailed workflow descriptions
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and ideas

## 🔮 Roadmap

- [ ] **Enhanced visualizations** for analysis results
- [ ] **Integration with more literature databases**
- [ ] **Advanced reasoning pattern analysis**
- [ ] **Web interface** for easier workflow management
- [ ] **Docker containerization** for easier deployment
- [ ] **Integration with research management tools**

---

**Ready to transform your research workflow?** Start with the quick installation and try the complete research pipeline on your topic of interest!

```bash
./scripts/complete_research_pipeline.sh --topic "your research area" --quick
```