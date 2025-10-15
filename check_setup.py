#!/usr/bin/env python3

import os
import sys
import subprocess

def check_ollama():
    """Check if Ollama is available and running"""
    try:
        # Check if Ollama command exists
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "Ollama not installed"

        # Check if Ollama is running
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            return True, "Ollama is running"
        else:
            return False, "Ollama not running"

    except Exception as e:
        return False, f"Ollama not accessible: {str(e)}"

def main():
    print("🔍 RAG System Setup Checker")
    print("=" * 40)

    # Check Ollama
    ollama_ok, ollama_msg = check_ollama()
    if ollama_ok:
        print(f"✅ Ollama: {ollama_msg}")

        # Check available models
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags')
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    print(f"📦 Available models:")
                    for model in models:
                        print(f"   • {model['name']}")
                else:
                    print(f"⚠️ No models installed")
                    print(f"💡 Run: ollama pull codellama")
        except:
            print(f"⚠️ Could not list models")
    else:
        print(f"❌ Ollama: {ollama_msg}")

    # Check Python dependencies
    try:
        import ollama
        print(f"✅ Ollama Python package")
    except ImportError:
        print(f"❌ Ollama Python package (pip install ollama)")

    try:
        from sentence_transformers import SentenceTransformer
        print(f"✅ Sentence transformers")
    except ImportError:
        print(f"❌ Sentence transformers (pip install sentence-transformers)")

    # Check configuration
    use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "codellama")

    print(f"\n📊 Configuration:")
    print(f"  USE_OLLAMA: {use_ollama}")
    print(f"  OLLAMA_MODEL: {ollama_model}")

    # Provide recommendations
    print(f"\n💡 Recommendations:")
    if not ollama_ok:
        print(f"  🚀 Quick start with Ollama:")
        print(f"     1. Install: https://ollama.ai/download")
        print(f"     2. Pull model: ollama pull {ollama_model}")
        print(f"     3. Start: ollama serve")
        print(f"     4. Or run: ./quick_start_ollama.sh")
    else:
        print(f"  ✅ Ollama is ready!")
        print(f"  🚀 Start the app: python app.py")

    print(f"\n🌐 App will be available at: http://localhost:5555")
    print("=" * 40)

if __name__ == "__main__":
    main()