"""
Routes module
Contains all Flask route handlers
"""
import os
import shutil
import ollama
from flask import render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from config import (
    USE_GLM, GLM_API_KEY, GLM_MODEL, GLM_MAX_TOKENS, GLM_TEMPERATURE,
    USE_OLLAMA, OLLAMA_MODEL,
    USE_LOCAL_EMBEDDINGS, EMBEDDING_MODEL,
    allowed_file
)
from rag_system import create_rag_system


# Global variable to store the current RAG system
current_rag = None


def init_routes(app):
    """Initialize all routes for the Flask app"""
    
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        global current_rag

        if 'file' not in request.files:
            flash('No file selected')
            return redirect(url_for('index'))

        file = request.files['file']

        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('index'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Create RAG system
                qa_chain, doc_count = create_rag_system(file_path)
                current_rag = qa_chain

                flash(f'Successfully uploaded and processed {filename} ({doc_count} documents)')
                return redirect(url_for('search'))

            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                # Clean up uploaded file on error
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect(url_for('index'))
        else:
            flash('Invalid file type. Please upload .txt, .md, or .json files')
            return redirect(url_for('index'))

    @app.route('/search')
    def search():
        return render_template('search.html')

    @app.route('/query', methods=['POST'])
    def query():
        global current_rag

        if current_rag is None:
            return jsonify({'error': 'No file uploaded yet. Please upload a file first.'})

        data = request.get_json()
        query = data.get('query', '').strip()
        history = data.get('history', [])

        if not query:
            return jsonify({'error': 'Please enter a query'})

        try:
            # Build context from conversation history
            context_query = query
            if history:
                # Add recent conversation context
                recent_history = history[-4:]  # Last 4 messages for context
                history_context = "\n\nRecent conversation:\n"
                for msg in recent_history:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    history_context += f"{role}: {msg['content']}\n"
                
                context_query = f"{history_context}\n\nCurrent question: {query}"

            # Perform RAG query with context
            result = current_rag({"query": context_query})

            return jsonify({
                'answer': result["result"]
            })

        except Exception as e:
            return jsonify({'error': f'Error processing query: {str(e)}'})

    @app.route('/reset')
    def reset():
        global current_rag
        current_rag = None

        # Clean up uploaded files
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        flash('System reset successfully')
        return redirect(url_for('index'))

    @app.route('/api/config')
    def get_config():
        """Get current configuration info"""
        # Check Ollama availability
        ollama_available = False
        try:
            ollama.list()  # Test connection
            ollama_available = True
        except:
            pass

        config = {
            "use_glm": USE_GLM,
            "glm_configured": bool(GLM_API_KEY),
            "glm_model": GLM_MODEL,
            "glm_max_tokens": GLM_MAX_TOKENS,
            "glm_temperature": GLM_TEMPERATURE,
            "use_ollama": USE_OLLAMA,
            "ollama_available": ollama_available,
            "ollama_model": OLLAMA_MODEL,
            "use_local_embeddings": USE_LOCAL_EMBEDDINGS,
            "embedding_model": EMBEDDING_MODEL,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "model_info": {}
        }

        # Determine model configuration
        if USE_GLM and GLM_API_KEY:
            if USE_LOCAL_EMBEDDINGS:
                config["model_info"] = {
                    "type": "Local",
                    "llm": f"GLM ({GLM_MODEL})",
                    "embeddings": f"Local ({EMBEDDING_MODEL})"
                }
            else:
                config["model_info"] = {
                    "type": "Hybrid Local",
                    "llm": f"GLM ({GLM_MODEL})",
                    "embeddings": "OpenAI"
                }
        elif USE_OLLAMA and ollama_available:
            if USE_LOCAL_EMBEDDINGS:
                config["model_info"] = {
                    "type": "Local",
                    "llm": f"Ollama ({OLLAMA_MODEL})",
                    "embeddings": f"Local ({EMBEDDING_MODEL})"
                }
            else:
                config["model_info"] = {
                    "type": "Hybrid Local",
                    "llm": f"Ollama ({OLLAMA_MODEL})",
                    "embeddings": "OpenAI"
                }
        elif os.getenv("OPENAI_API_KEY"):
            if USE_LOCAL_EMBEDDINGS:
                config["model_info"] = {
                    "type": "Hybrid Local",
                    "llm": "OpenAI",
                    "embeddings": f"Local ({EMBEDDING_MODEL})"
                }
            else:
                config["model_info"] = {
                    "type": "OpenAI Only",
                    "llm": "OpenAI",
                    "embeddings": "OpenAI"
                }
        else:
            config["model_info"] = {
                "type": "Not Configured",
                "llm": "None",
                "embeddings": "None"
            }

        return jsonify(config)

