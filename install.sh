#!/bin/bash

# Argonium Research Pipeline Installation Script

set -e  # Exit on any error

echo "=========================================="
echo "Argonium Research Pipeline Installation"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.7"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"; then
    echo "✓ Python $PYTHON_VERSION detected (>= $REQUIRED_VERSION required)"
else
    echo "❌ Python $REQUIRED_VERSION or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✓ pip3 is available"

# Upgrade pip, setuptools, and wheel
echo ""
echo "Upgrading pip, setuptools, and wheel..."
pip3 install --upgrade pip setuptools wheel

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Download required NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    print('✓ NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: NLTK download failed: {e}')
    print('You may need to download NLTK data manually later')
"

# Check for optional system dependencies
echo ""
echo "Checking optional system dependencies..."

# Check for poppler (needed for advanced PDF processing)
if command -v pdftotext &> /dev/null; then
    echo "✓ Poppler detected (advanced PDF processing available)"
else
    echo "⚠️  Poppler not found (basic PDF processing only)"
    echo "   To install:"
    echo "   - macOS: brew install poppler"
    echo "   - Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "   - CentOS/RHEL: sudo yum install poppler-utils"
fi

# Check for jq (useful for JSON processing in scripts)
if command -v jq &> /dev/null; then
    echo "✓ jq detected (enhanced JSON processing available)"
else
    echo "⚠️  jq not found (basic JSON processing only)"
    echo "   To install:"
    echo "   - macOS: brew install jq"
    echo "   - Ubuntu/Debian: sudo apt-get install jq"
    echo "   - CentOS/RHEL: sudo yum install jq"
fi

# Create example configuration file
echo ""
echo "Setting up configuration files..."

if [ ! -f "model_servers.yaml" ]; then
    cat > model_servers.yaml.example << 'EOF'
# Argonium Model Configuration
# Copy this file to model_servers.yaml and configure your endpoints

servers:
  # OpenAI GPT-4
  - shortname: "gpt41"
    server: "OpenAI GPT-4"
    openai_api_key: "${OPENAI_API_KEY}"
    openai_api_base: "https://api.openai.com/v1"
    openai_model: "gpt-4"

  # OpenAI GPT-3.5 Turbo
  - shortname: "gpt35"
    server: "OpenAI GPT-3.5 Turbo"
    openai_api_key: "${OPENAI_API_KEY}"
    openai_api_base: "https://api.openai.com/v1"
    openai_model: "gpt-3.5-turbo"

  # Example custom endpoint
  - shortname: "scout"
    server: "Custom Model Endpoint"
    openai_api_key: "${CUSTOM_API_KEY}"
    openai_api_base: "https://your-endpoint.com/v1"
    openai_model: "your-model-name"

  # Claude (via API)
  - shortname: "claude3"
    server: "Anthropic Claude"
    openai_api_key: "${ANTHROPIC_API_KEY}"
    openai_api_base: "https://api.anthropic.com/v1"
    openai_model: "claude-3-sonnet-20240229"

  # Add more models as needed
EOF
    echo "✓ Created model_servers.yaml.example"
    echo "  → Copy to model_servers.yaml and configure your API endpoints"
else
    echo "✓ model_servers.yaml already exists"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.sh
echo "✓ All scripts are now executable"

# Create basic directory structure
echo ""
echo "Creating directory structure..."
mkdir -p logs
mkdir -p data
mkdir -p results
echo "✓ Basic directories created"

# Verify installation
echo ""
echo "Verifying installation..."

# Test basic imports
python3 -c "
import sys
required_modules = [
    'numpy', 'tqdm', 'yaml', 'openai', 'requests', 
    'PyPDF2', 'pdfminer', 'nltk', 'sklearn', 'networkx'
]

failed_imports = []
for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        failed_imports.append(module)

if failed_imports:
    print(f'❌ Failed to import: {failed_imports}')
    sys.exit(1)
else:
    print('✓ All required Python modules imported successfully')
"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your API endpoints:"
echo "   cp model_servers.yaml.example model_servers.yaml"
echo "   # Edit model_servers.yaml with your API keys and endpoints"
echo ""
echo "2. Set environment variables:"
echo "   export OPENAI_API_KEY='your-openai-api-key'"
echo "   export SS_API_KEY='your-semantic-scholar-api-key'  # Optional"
echo ""
echo "3. Test the installation:"
echo "   ./scripts/complete_research_pipeline.sh --help"
echo ""
echo "4. Run a quick test:"
echo "   ./scripts/complete_research_pipeline.sh \\"
echo "       --topic 'machine learning' \\"
echo "       --quick"
echo ""
echo "Documentation:"
echo "  - README.md: Quick start guide"
echo "  - workflow.md: Detailed workflow documentation"
echo "  - scripts/: Workflow automation scripts"
echo ""
echo "Support:"
echo "  - GitHub Issues: Report bugs and request features"
echo "  - GitHub Discussions: Ask questions and share ideas"
echo ""
echo "Happy researching! 🚀"
echo "=========================================="