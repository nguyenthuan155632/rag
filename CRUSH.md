# RAG Codebase Search Web App - Development Guidelines

## Quick Commands
- **Start development**: `python app.py` (use python3)
- **Check setup**: `python check_setup.py`
- **Quick Ollama setup**: `./quick_start_ollama.sh`
- **Install dependencies**: `pip install -r requirements.txt` (use virtual environment)

## LLM Configuration Options
- **GLM (Z.AI)**: `export USE_GLM=true` and `export GLM_API_KEY='your-key'`
- **Ollama (Local)**: `export USE_OLLAMA=true` (default)
- **OpenAI (Cloud)**: `export OPENAI_API_KEY='sk-your-key'`

Priority order: GLM → Ollama → OpenAI

## Code Style Guidelines

### Python Conventions
- Use snake_case for variables and functions
- Use PascalCase for classes (e.g., `OllamaLLM`, `LocalEmbeddings`)
- Import order: standard library → third-party → local imports
- Type hints required for function signatures: `def func(param: str) -> bool:`
- Maximum line length: 100 characters

### Flask Application Structure
- Route decorators directly above function definitions
- Use flash() for user feedback messages
- Global variables should be explicitly managed (e.g., `current_rag`)
- Error handling with try/except blocks and user-friendly messages

### AI/ML Integration
- Environment variables for configuration: `USE_OLLAMA`, `OLLAMA_MODEL`, `EMBEDDING_MODEL`
- Fallback patterns: Ollama → OpenAI → Error
- Class-based LLM wrappers inheriting from LangChain base classes
- Document processing with proper metadata handling

### File Handling
- Use `secure_filename()` for uploaded files
- Allowed extensions: `{'txt', 'md', 'json'}`
- Cleanup on errors: remove uploaded files if processing fails
- Directory existence checks: `os.makedirs(path, exist_ok=True)`

### Configuration Management
- Load environment variables early with `load_dotenv()`
- Default values in code, override via environment
- Boolean parsing: `os.getenv("VAR", "default").lower() == "true"`

### Error Handling
- Specific error messages for different failure modes
- Graceful degradation when services unavailable
- User-facing error messages via flash() and JSON responses
- Cleanup resources in error paths