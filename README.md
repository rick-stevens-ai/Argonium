# Argonium Research Pipeline

A comprehensive toolkit for scientific literature analysis, AI model evaluation, and cognitive reasoning assessment. The Argonium pipeline enables end-to-end research workflows from paper discovery to detailed model reasoning analysis.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **📚 Literature Discovery**: Automated paper search, download, and AI-powered relevance filtering
- **🏷️ Content Classification**: AI-powered categorization using predefined scientific taxonomies
- **📊 Data Processing**: Comprehensive JSON manipulation and analysis tools
- **🤖 Model Benchmarking**: Multi-model evaluation with detailed performance analysis
- **🧠 Reasoning Analysis**: Deep cognitive assessment with expert persona modeling
- **🔄 Integrated Workflows**: Complete research pipelines from discovery to analysis

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

### 4. Reasoning Analysis & Validation
- **Primary Tool**: `reasoning_traces_v6.py`
- **Workflow Script**: `scripts/reasoning_analysis.sh`
- **Purpose**: Deep cognitive analysis with expert personas

**Expert Specialties**:
- Microbiologist
- Physicist (General & Quantum)
- Historian
- Generic Expert

**Example**:
```bash
python reasoning_traces_v6.py questions.json \
    --model gpt41 \
    --specialty microbiologist \
    --whole-trace-analysis
```

### 5. Data Processing & Analysis
Tools for JSON manipulation, sampling, and report generation:
- `merge_json.py` - Combine multiple JSON files
- `merge_incorrect_answers.py` - Analyze model errors
- `sample_json.py` - Random sampling
- `render_json.py` - Generate formatted reports

## 🛠️ Core Tools Reference

| Tool | Purpose | Key Features |
|------|---------|-------------|
| `download_papers_v8.py` | Paper discovery | AI keyword generation, relevance filtering |
| `classify_papers.py` | Content classification | 25+ scientific categories, AI-powered |
| `analyze_resources.py` | Content analysis | TF-IDF, TextRank, comprehensive summaries |
| `run_all_models.py` | Model benchmarking | Multi-model, parallel evaluation |
| `reasoning_traces_v6.py` | Reasoning analysis | Expert personas, detailed traces |
| `merge_incorrect_answers.py` | Error analysis | Intersection analysis, difficulty assessment |

## 📁 Project Structure

```
argonium/
├── scripts/                      # Workflow automation scripts
│   ├── complete_research_pipeline.sh
│   ├── literature_discovery.sh
│   ├── model_benchmark.sh
│   └── reasoning_analysis.sh
├── analyze_resources.py          # Literature analysis
├── classify_papers.py           # AI-powered classification
├── download_papers_v8.py        # Paper discovery and download
├── reasoning_traces_v6.py       # Detailed reasoning analysis
├── run_all_models.py            # Multi-model benchmarking
├── merge_incorrect_answers.py   # Error analysis
├── [other processing tools]     # Data manipulation utilities
├── model_servers.yaml.example   # Model configuration template
├── requirements.txt             # Python dependencies
├── install.sh                   # Installation script
└── workflow.md                  # Detailed workflow documentation
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

### Model Reasoning Assessment
```bash
# 1. Analyze reasoning with domain expertise
./scripts/reasoning_analysis.sh \
    --questions questions.json \
    --specialty microbiologist \
    --max-questions 100 \
    --whole-trace

# 2. Compare different expert perspectives
./scripts/reasoning_analysis.sh \
    --questions questions.json \
    --specialty physicist \
    --max-questions 100
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

## 📊 Output Formats

### Literature Analysis
- **README.md**: Comprehensive literature summaries
- **Topic directories**: Organized paper collections
- **Synopsis files**: Quick paper overviews

### Model Evaluation
- **JSON results**: Detailed performance metrics
- **Markdown reports**: Human-readable summaries
- **Error analysis**: Incorrect answer collections

### Reasoning Analysis
- **Detailed traces**: Complete thought processes
- **Meta-analysis**: Cognitive pattern assessment
- **Sample extracts**: Notable reasoning examples

## 🚨 Common Issues & Solutions

### Installation Issues
```bash
# If pip install fails, try:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# For PDF processing issues:
# macOS: brew install poppler
# Ubuntu: sudo apt-get install poppler-utils
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