# 🚀 Quick Start Guide

## ⚠️ Getting "Ollama not available" Error?

This error means Ollama isn't running yet. Here's how to fix it:

### 🎯 Option 1: Automated Setup (Easiest)
```bash
./quick_start_ollama.sh
```
This will:
- Install Ollama (if needed)
- Pull CodeLlama model
- Start Ollama server
- Install Python dependencies
- Launch the app

### 🔧 Option 2: Manual Setup

#### Step 1: Install Ollama
```bash
# macOS
brew install ollama

# Or download from: https://ollama.ai/download
```

#### Step 2: Pull Code Model
```bash
ollama pull codellama
```

#### Step 3: Start Ollama
```bash
ollama serve
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Start App
```bash
python app.py
```

### 🔍 Check Your Setup
```bash
python check_setup.py
```

This will tell you exactly what's configured and what you need to install.

## 📱 What You'll See

When everything is working, you'll see:
```
🚀 Starting RAG Codebase Search App
==================================================
📊 Configuration:
  Ollama: ✅ Available (Local LLM)
  Local Embeddings: ✅ Enabled (all-MiniLM-L6-v2)
  🤖 Mode: Fully Local (Ollama + Local Embeddings)

🎉 Local RAG system ready!
   ✅ Ollama LLM: codellama
   ✅ Local Embeddings: all-MiniLM-L6-v2
   🌐 Completely offline - no API costs!

🌐 Starting server at: http://localhost:5555
```

## 🌐 Access the App

Open your browser and go to: **http://localhost:5555**

You'll see a green badge saying: **"🏠 Fully Local: Ollama (codellama) + Local (all-MiniLM-L6-v2)"**

## 📁 Upload Your Codebase

1. Generate a repomix of your codebase
2. Upload the file (.json, .txt, or .md)
3. Start asking questions about your code!

## 💡 Pro Tips

- **No internet required** once Ollama is installed
- **Zero API costs** - completely free
- **Your code stays private** - never leaves your machine
- **Works offline** - perfect for sensitive projects

## 🆘 Still Having Issues?

1. **Check Ollama is running**: `ollama list`
2. **Check model is installed**: `ollama list` (should show codellama)
3. **Check dependencies**: `python check_setup.py`
4. **Try the automated script**: `./quick_start_ollama.sh`

That's it! You'll have a fully local RAG system running in minutes! 🎉