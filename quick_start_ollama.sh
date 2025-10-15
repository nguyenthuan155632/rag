#!/bin/bash

# Cleanup function to kill processes on port 5555
cleanup() {
    echo ""
    echo "🛑 Stopping processes on port 5555..."

    # Kill Flask app process
    if [ ! -z "$APP_PID" ]; then
        kill $APP_PID 2>/dev/null
        echo "✅ Stopped Flask app (PID: $APP_PID)"
    fi

    # Kill any other processes using port 5555
    PIDS=$(lsof -ti:5555 2>/dev/null)
    if [ ! -z "$PIDS" ]; then
        echo "🔄 Killing additional processes on port 5555..."
        echo "$PIDS" | xargs kill 2>/dev/null
        echo "✅ Killed processes: $PIDS"
    fi

    exit 0
}

# Set trap to catch Ctrl+C (SIGINT)
trap cleanup SIGINT

echo "🚀 Quick Start Ollama RAG System"
echo "================================="

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
else
    echo "❌ Ollama not found"
    echo "📥 Installing Ollama..."

    # Detect OS and install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "❌ Homebrew not found. Please install Ollama manually:"
            echo "   Visit: https://ollama.ai/download"
        fi
    else
        echo "💡 Please install Ollama manually:"
        echo "   Visit: https://ollama.ai/download"
        exit 1
    fi
fi

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "✅ Ollama is running"
else
    echo "🔄 Starting Ollama..."
    ollama serve &
    sleep 5

    # Check again
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✅ Ollama started successfully"
    else
        echo "❌ Failed to start Ollama"
        echo "💡 Please start manually: ollama serve"
        exit 1
    fi
fi

# Check if CodeLlama is available
if ollama list | grep -q "codellama"; then
    echo "✅ CodeLlama model is available"
else
    echo "📥 Pulling CodeLlama model (this may take a few minutes)..."
    ollama pull codellama

    if ollama list | grep -q "codellama"; then
        echo "✅ CodeLlama model downloaded successfully"
    else
        echo "❌ Failed to download CodeLlama"
        echo "💡 Try manually: ollama pull codellama"
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    echo "💡 Try: pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo "🌐 Starting the RAG app..."
echo ""
echo "📝 What's happening:"
echo "   • Using Ollama CodeLlama for LLM"
echo "   • Using local sentence transformers for embeddings"
echo "   • Completely offline - no API costs!"
echo ""
echo "🚀 Opening browser..."

# Start the app
python app.py &
APP_PID=$!

# Wait a moment for the server to start
sleep 3

# Open browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:5555
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:5555
fi

echo "✅ App started at: http://localhost:5555"
echo "💡 Upload a repomix file to start searching your codebase!"
echo ""
echo "🛑 To stop: Press Ctrl+C (will clean up all processes on port 5555)"

# Wait for the app
wait $APP_PID