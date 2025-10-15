# RAG Codebase Search Web App

A Flask-based web application that uses RAG (Retrieval-Augmented Generation) to enable intelligent searching through codebases using natural language queries.

## Features

- 📁 **File Upload**: Support for repomix codebase files (.txt, .md, .json)
- 🔍 **Intelligent Search**: Natural language queries against your codebase
- 🏠 **Local AI Support**: Use Ollama for completely offline RAG (no API costs!)
- 🤖 **Hybrid AI**: Combines local and cloud models (GLM, Ollama, OpenAI)
- 💫 **Modern UI**: Clean, responsive interface with real-time feedback
- 🔧 **Flexible Configuration**: Mix and match local and cloud models

## Installation

1. **Clone and navigate to the project**:
   ```bash
   cd rag-search-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter LangChain import errors, try:
   ```bash
   pip install --upgrade langchain langchain-community
   ```

3. **Configure your preferred setup**:

   **🏠 Option 1: Fully Local (Recommended - No API Costs!)**
   ```bash
   # Install Ollama first: https://ollama.ai/download
   brew install ollama  # macOS
   ollama pull codellama
   ollama serve

   # No API keys required!
   python app.py
   ```

   **🤖 Option 2: GLM (Z.AI) Setup**
   ```bash
   export GLM_API_KEY='your-glm-api-key-here'
   export USE_GLM=true
   python app.py
   ```

   **☁️ Option 3: Cloud-based Setup**
   ```bash
   export OPENAI_API_KEY='sk-your-openai-key-here'
   python app.py
   ```

   **🔀 Option 4: Hybrid Setup**
   ```bash
   # Use Ollama for LLM, OpenAI for embeddings
   export OPENAI_API_KEY='sk-your-openai-key-here'
   ollama pull codellama && ollama serve
   python app.py
   ```

   **🔍 Check your setup:**
   ```bash
   python check_setup.py
   ```

   📖 **Detailed Ollama Setup**: See [OLLAMA_SETUP.md](OLLAMA_SETUP.md)

## Usage

1. **Start the Flask app**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5555`

3. **Upload your repomix file**:
   - Generate a repomix of your codebase using the repomix tool
   - Upload the generated file (.json, .txt, or .md)

4. **Start searching**:
   - Ask questions about your codebase in natural language
   - Get AI-powered answers with relevant code snippets
   - Explore source documents for detailed information

## Example Queries

- "How does the authentication system work?"
- "Where is the database connection configured?"
- "Show me the error handling implementation"
- "What are the main API endpoints?"
- "How are user sessions managed?"

## File Structure

```
rag-search-app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   ├── index.html        # File upload page
│   └── search.html       # Search interface and results
├── static/
│   └── style.css         # Styling
├── uploads/              # Directory for uploaded files
└── README.md            # This file
```

## Supported File Formats

- **JSON**: Repomix-generated codebase files
- **TXT**: Plain text files
- **MD**: Markdown files

## Technology Stack

- **Backend**: Flask
- **AI/ML**: LangChain, GLM (Z.AI), Ollama, OpenAI
- **Vector Store**: FAISS
- **Frontend**: HTML5, CSS3, JavaScript
- **Syntax Highlighting**: Prism.js

## Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GLM_API_KEY` | Optional | - | Z.AI GLM API key for LLM generation |
| `USE_GLM` | No | `false` | Enable GLM LLM for generation |
| `GLM_MODEL` | No | `glm-4.5-air` | GLM model to use |
| `GLM_MAX_TOKENS` | No | `32768` | Maximum tokens for GLM responses |
| `GLM_TEMPERATURE` | No | `0.7` | GLM temperature (0.0-1.0) |
| `OPENAI_API_KEY` | Optional | - | OpenAI API key for embeddings and fallback LLM |
| `USE_OLLAMA` | No | `true` | Enable Ollama for local LLM |
| `OLLAMA_MODEL` | No | `qwen2.5-coder:7b` | Ollama model to use |
| `USE_LOCAL_EMBEDDINGS` | No | `true` | Use local sentence transformers |

### LLM Priority Order

The app automatically selects the best available LLM in this order:
1. **GLM (Z.AI)** - High-performance cloud model with excellent reasoning
2. **Ollama** - Local models (no API costs)
3. **OpenAI** - Cloud fallback option

### Hybrid Mode Benefits

- **Best of Both Worlds**: OpenAI's excellent embeddings + Cursor's code understanding
- **Fallback Protection**: Automatically falls back to OpenAI if Cursor is unavailable
- **Cost Optimization**: Use the most appropriate model for each task
- **Flexibility**: Easily switch between models based on your needs

## Configuration

The app can be configured by modifying the following in `app.py`:

- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)
- `UPLOAD_FOLDER`: Directory for uploaded files
- Text splitter parameters (`chunk_size`, `chunk_overlap`)

## Error Handling

The app includes comprehensive error handling for:
- Invalid file types
- File size limits
- API connection issues
- Malformed repomix files

## Security Notes

- Files are uploaded to a secure directory
- File types are validated
- API keys should be set via environment variables
- The app runs in development mode by default

## Development

To run in development mode with debug features:

```bash
export FLASK_ENV=development
python app.py
```

## Production Deployment

For production deployment, consider:

1. Using a production WSGI server (Gunicorn, uWSGI)
2. Setting proper environment variables
3. Configuring HTTPS
4. Setting up proper logging
5. Using a production-grade vector store

## Troubleshooting

**Common Issues:**

1. **OpenAI API Key Error**: Make sure your API key is set correctly
2. **File Upload Fails**: Check file size and format
3. **Memory Issues**: Reduce `chunk_size` for large codebases
4. **Slow Response Times**: Consider using a smaller `k` value for retrieval

## License

This project is for educational and demonstration purposes.