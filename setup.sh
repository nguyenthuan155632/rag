#!/bin/bash

echo "🚀 Setting up RAG Codebase Search App"
echo "===================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Additional setup for LangChain if needed
echo "🔧 Ensuring LangChain modules are properly installed..."
pip install --upgrade langchain langchain-community

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Set your API keys:"
echo "   export OPENAI_API_KEY='your-openai-key'"
echo "   export CURSOR_API_KEY='your-cursor-key'  # Optional"
echo ""
echo "2. Run the app:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🌐 The app will be available at: http://localhost:5555"
echo "===================================="