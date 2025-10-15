# Ollama Setup Guide

This guide will help you set up a completely local RAG system using Ollama and local embeddings - no API costs!

## 🚀 Quick Setup (Recommended)

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

### 2. Start Ollama

```bash
ollama serve
```

### 3. Pull a Code Model

**For Code Analysis (Recommended):**
```bash
ollama pull codellama
```

**Alternative Models:**
```bash
ollama pull deepseek-coder     # Excellent for code
ollama pull llama2             # General purpose
ollama pull mistral            # Fast and efficient
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the App

```bash
python app.py
```

That's it! You now have a fully local RAG system running. 🎉

## 📋 Available Models

### Code-Specific Models (Best for Codebases)
- **codellama** (7B/13B/34B/70B) - Excellent for code understanding
- **deepseek-coder** (6.7B/33B) - Specialized for code analysis
- **starcoder** (15B) - Trained on GitHub code

### General Purpose Models
- **llama2** (7B/13B/70B) - Good all-around performance
- **mistral** (7B) - Fast and efficient
- **qwen** (7B/14B) - Strong reasoning capabilities

### Model Size Recommendations
- **7B models**: Good for most codebases, faster response
- **13B+ models**: Better understanding, slower response
- **34B+ models**: Best quality, requires more RAM

## ⚙️ Configuration Options

### Environment Variables

```bash
# Enable Ollama (default: true)
export USE_OLLAMA=true

# Choose your model (default: codellama)
export OLLAMA_MODEL=codellama

# Ollama server URL (default: http://localhost:11434)
export OLLAMA_BASE_URL=http://localhost:11434

# Use local embeddings (default: true)
export USE_LOCAL_EMBEDDINGS=true

# Embedding model (default: all-MiniLM-L6-v2)
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Model Examples

```bash
# Use DeepSeek Coder for better code analysis
export OLLAMA_MODEL=deepseek-coder

# Use a smaller model for faster responses
export OLLAMA_MODEL=mistral

# Use a larger model for better quality
export OLLAMA_MODEL=codellama:13b
```

## 🔧 Alternative Setups

### Option 1: Fully Local (No Internet Required)
```bash
export USE_OLLAMA=true
export USE_LOCAL_EMBEDDINGS=true
# No API keys needed!
```

### Option 2: Local LLM + Cloud Embeddings
```bash
export USE_OLLAMA=true
export USE_LOCAL_EMBEDDINGS=false
export OPENAI_API_KEY=sk-your-key-here  # For embeddings only
```

### Option 3: Hybrid (Multiple LLMs)
```bash
export USE_OLLAMA=true
export CURSOR_API_KEY=your-cursor-key  # Fallback option
export OPENAI_API_KEY=sk-your-key      # Another fallback
```

## 🎯 Performance Tips

### 1. Choose the Right Model Size
- **< 8GB RAM**: Use 7B models (codellama, mistral)
- **8-16GB RAM**: Use 13B models (codellama:13b)
- **16GB+ RAM**: Use 34B+ models (codellama:34b)

### 2. Optimize Embeddings
```bash
# Smaller embedding model (faster)
export EMBEDDING_MODEL=all-MiniLM-L6-v2

# Larger embedding model (better quality)
export EMBEDDING_MODEL=all-mpnet-base-v2
```

### 3. GPU Acceleration (if available)
Ollama automatically uses GPU if supported. For optimal performance:
- **NVIDIA GPUs**: CUDA support is automatic
- **Apple Silicon (M1/M2/M3)**: Metal support is automatic
- **AMD GPUs**: ROCm support may need manual setup

## 🛠️ Troubleshooting

### Ollama Not Found
```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve
```

### Model Not Downloaded
```bash
# Download the specific model
ollama pull codellama
ollama pull deepseek-coder
```

### Connection Errors
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Verify Ollama is running on correct port
netstat -an | grep 11434
```

### Memory Issues
```bash
# Use a smaller model
export OLLAMA_MODEL=mistral

# Or use quantized versions
export OLLAMA_MODEL=codellama:7b-q4_K_M
```

## 📊 Model Comparison

| Model | Size | Best For | Speed | RAM Required |
|-------|------|----------|-------|-------------|
| mistral | 7B | General purpose | ⚡ Fast | 4-8GB |
| codellama | 7B | Code analysis | ⚡ Fast | 4-8GB |
| deepseek-coder | 6.7B | Code analysis | ⚡ Fast | 4-8GB |
| codellama | 13B | Complex code | 🐢 Medium | 8-16GB |
| codellama | 34B | Best quality | 🐌 Slow | 16-32GB |

## 🎉 Benefits of Local Setup

- ✅ **No API Costs** - Completely free
- ✅ **Privacy** - Your code never leaves your machine
- ✅ **Offline** - Works without internet connection
- ✅ **Customizable** - Use any model you want
- ✅ **Fast** - No network latency
- ✅ **Unlimited** - No rate limits or quotas

## 📚 Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Model Library](https://ollama.ai/library)
- [Sentence Transformers Models](https://huggingface.co/models?library=sentence-transformers)