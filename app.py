"""
RAG Codebase Search App - Main Application
A Flask-based application for semantic search on codebases using RAG (Retrieval-Augmented Generation)
"""
import os
import ollama
from flask import Flask

from config import (
    SECRET_KEY, UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
    USE_GLM, GLM_API_KEY, GLM_MODEL, GLM_MAX_TOKENS, GLM_TEMPERATURE,
    USE_OLLAMA, OLLAMA_MODEL,
    USE_LOCAL_EMBEDDINGS, EMBEDDING_MODEL
)
from routes import init_routes

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize routes
init_routes(app)

if __name__ == '__main__':
    print("🚀 Starting RAG Codebase Search App")
    print("=" * 50)

    # Check configuration
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    glm_configured = bool(GLM_API_KEY)

    # Check Ollama availability
    ollama_available = False
    if USE_OLLAMA:
        try:
            ollama.list()
            ollama_available = True
        except Exception as e:
            print(f"⚠️ Ollama not available: {e}")

    print(f"📊 Configuration:")
    print(f"  GLM: {'✅ Configured' if glm_configured else '❌ Not Configured'} ({GLM_MODEL}, max_tokens={GLM_MAX_TOKENS}, temp={GLM_TEMPERATURE})")
    print(f"  Ollama: {'✅ Available' if ollama_available else '❌ Not Available'} (Local LLM)")
    print(f"  Local Embeddings: {'✅ Enabled' if USE_LOCAL_EMBEDDINGS else '❌ Disabled'} ({EMBEDDING_MODEL})")
    print(f"  OpenAI API: {'✅ Configured' if openai_configured else '❌ Missing (Optional fallback)'}")

    # Determine mode
    if USE_GLM and glm_configured:
        if USE_LOCAL_EMBEDDINGS:
            print(f"  🤖 Mode: Fully Local (GLM + Local Embeddings)")
        else:
            print(f"  🤖 Mode: GLM LLM + OpenAI Embeddings")
    elif USE_OLLAMA and ollama_available:
        if USE_LOCAL_EMBEDDINGS:
            print(f"  🤖 Mode: Fully Local (Ollama + Local Embeddings)")
        else:
            print(f"  🤖 Mode: Local LLM + OpenAI Embeddings")
    elif openai_configured:
        if USE_LOCAL_EMBEDDINGS:
            print(f"  🤖 Mode: OpenAI LLM + Local Embeddings")
        else:
            print(f"  🤖 Mode: OpenAI Only (LLM + Embeddings)")
    else:
        print(f"  ⚠️  Mode: Not Configured - GLM, Ollama or API keys needed!")

    print("\n📝 Setup Instructions:")
    if USE_GLM and not glm_configured:
        print(f"  🔑 Get GLM API key: https://z.ai/manage-apikey/apikey-list")
        print(f"  🔑 Set GLM_API_KEY='your-key'")
        print(f"  🔑 Set USE_GLM=true")
    
    if not ollama_available and USE_OLLAMA:
        print(f"  🔧 Install Ollama: https://ollama.ai/download")
        print(f"  🔧 Pull model: ollama pull {OLLAMA_MODEL}")
        print(f"  🔧 Start Ollama: ollama serve")

    if openai_configured:
        print(f"  ✅ OpenAI API configured (fallback option)")
    else:
        print(f"  🔑 Optional: export OPENAI_API_KEY='sk-your-key' (for fallback)")

    
    # Determine if ready to start
    ready = glm_configured or ollama_available or openai_configured

    if ready:
        if USE_GLM and glm_configured:
            print(f"\n🎉 GLM RAG system ready!")
            if USE_LOCAL_EMBEDDINGS:
                print(f"   ✅ GLM LLM: {GLM_MODEL}")
                print(f"   ✅ Local Embeddings: {EMBEDDING_MODEL}")
            else:
                print(f"   ✅ GLM LLM: {GLM_MODEL}")
                print(f"   ✅ OpenAI Embeddings")
        elif USE_OLLAMA and ollama_available:
            print(f"\n🎉 Local RAG system ready!")
            if USE_LOCAL_EMBEDDINGS:
                print(f"   ✅ Ollama LLM: {OLLAMA_MODEL}")
                print(f"   ✅ Local Embeddings: {EMBEDDING_MODEL}")
                print(f"   🌐 Completely offline - no API costs!")
            else:
                print(f"   ✅ Ollama LLM: {OLLAMA_MODEL}")
                print(f"   ✅ OpenAI Embeddings")
                print(f"   💡 Local LLM + cloud embeddings")
        elif openai_configured:
            print(f"\n✅ Cloud-based RAG system ready!")

        print(f"🌐 Starting server at: http://localhost:5555")
        print(f"💡 Open your browser and upload a repomix codebase file!")
    else:
        print(f"\n❌ Cannot start - no LLM configured")
        print(f"💡 Please install Ollama or configure API keys")

    print("=" * 50)

    # Only start the app if we have at least one LLM configured
    if ready:
        app.run(debug=True, host='0.0.0.0', port=5555)
    else:
        print(f"\n🛑 App not started. Please configure an LLM first.")
        print(f"📚 See README.md for detailed setup instructions.")
