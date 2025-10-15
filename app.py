import ollama
import os
import json
import requests

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.embeddings import Embeddings
from typing import Optional, List
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the current RAG system
current_rag = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'md', 'json'}

# Ollama configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embeddings configuration
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Ollama generation configuration
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "8000"))

# GLM configuration
USE_GLM = os.getenv("USE_GLM", "false").lower() == "true"
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4.5-air")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
GLM_MAX_TOKENS = int(os.getenv("GLM_MAX_TOKENS", "32768"))
GLM_TEMPERATURE = float(os.getenv("GLM_TEMPERATURE", "0.7"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class OllamaLLM(LLM):
    """Custom LLM class for Ollama integration"""

    def __init__(self, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434", temperature: float = 0.7, num_predict: int = 8000):
        super().__init__()
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._num_predict = num_predict

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def temperature(self) -> float:
        return self._temperature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the Ollama API"""
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self._temperature,
                    "num_predict": self._num_predict,  # Maximum number of tokens to generate
                    "stop": stop
                }
            )
            return response["message"]["content"]
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """Generate responses for multiple prompts"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

class LocalEmbeddings(Embeddings):
    """Local embeddings using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.model.encode([text])[0].tolist()

    def __call__(self, text: str) -> List[float]:
        """Make the embeddings callable (required by LangChain)"""
        return self.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._dimension


class GLMLLM(LLM):
    """Custom LLM class for GLM (Z.AI) integration"""

    def __init__(self, model: str = "glm-4.5-air", api_key: str = "", 
                 base_url: str = "https://api.z.ai/api/coding/paas/v4",
                 temperature: float = 0.7, max_tokens: int = 32768):
        super().__init__()
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "glm"

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the GLM API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Accept-Language": "en-US,en"
            }
            
            data = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "thinking": {
                    "type": "enabled"
                }
            }
            
            if stop:
                data["stop"] = stop

            response = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception("Invalid response format from GLM API")
                
        except requests.exceptions.Timeout as e:
            raise Exception(f"GLM API timeout - the request took too long. This is normal for complex queries. Please try again or use a simpler query.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"GLM API request error: {str(e)}")
        except Exception as e:
            raise Exception(f"GLM API error: {str(e)}")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """Generate responses for multiple prompts"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)


def create_rag_system(file_path):
    """Create RAG system from uploaded file"""
    try:
        # Determine file type and load accordingly
        if file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
        elif file_path.endswith('.json'):
            # Handle repomix JSON files
            documents = load_repomix_json(file_path)
        else:
            loader = TextLoader(file_path)
            documents = loader.load()

        # Validate documents
        if not documents:
            raise Exception("No documents found in the uploaded file")

        print(f"📄 Loaded {len(documents)} documents from {file_path}")

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        if not texts:
            raise Exception("No text chunks created from documents")

        print(f"🔤 Created {len(texts)} text chunks")

        # Create embeddings and vector store
        if USE_LOCAL_EMBEDDINGS:
            print(f"🔧 Using local embeddings: {EMBEDDING_MODEL}")
            embeddings = LocalEmbeddings(EMBEDDING_MODEL)
        else:
            print("🔧 Using OpenAI embeddings")
            embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.from_documents(texts, embeddings)

        # Create retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Create LLM based on configuration
        llm = None
        llm_source = ""

        # Try GLM first if enabled
        if USE_GLM and GLM_API_KEY:
            print(f"🤖 Attempting to use GLM LLM: {GLM_MODEL}")
            try:
                llm = GLMLLM(model=GLM_MODEL, api_key=GLM_API_KEY, base_url=GLM_BASE_URL, temperature=GLM_TEMPERATURE, max_tokens=GLM_MAX_TOKENS)
                llm_source = f"GLM ({GLM_MODEL})"
                print(f"✅ GLM LLM initialized successfully")
            except Exception as e:
                print(f"⚠️ GLM not available: {e}")
                print(f"💡 To use GLM:")
                print(f"   1. Get API key: https://z.ai/manage-apikey/apikey-list")
                print(f"   2. Set environment: export GLM_API_KEY='your-key'")
        elif USE_GLM and not GLM_API_KEY:
            print(f"⚠️ GLM enabled but no API key provided")
            print(f"💡 Set GLM_API_KEY environment variable")

        # Try Ollama if GLM failed
        if llm is None and USE_OLLAMA:
            print(f"🤖 Attempting to use Ollama LLM: {OLLAMA_MODEL}")
            try:
                # Test Ollama connection first
                import ollama
                ollama.list()  # Test connection
                llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7, num_predict=OLLAMA_NUM_PREDICT)
                llm_source = f"Ollama ({OLLAMA_MODEL})"
                print(f"✅ Ollama LLM connected successfully")
            except Exception as e:
                print(f"⚠️ Ollama not available: {e}")
                print(f"💡 To use Ollama:")
                print(f"   1. Install Ollama: https://ollama.ai/download")
                print(f"   2. Pull model: ollama pull {OLLAMA_MODEL}")
                print(f"   3. Start Ollama: ollama serve")

        # Try OpenAI as fallback
        if llm is None and os.getenv("OPENAI_API_KEY"):
            print(f"🔄 Falling back to OpenAI LLM")
            try:
                llm = OpenAI(temperature=0.7)
                llm_source = "OpenAI"
                print(f"✅ OpenAI LLM initialized successfully")
            except Exception as e:
                print(f"⚠️ OpenAI not available: {e}")

        # If no LLM is available, provide helpful error message
        if llm is None:
            print(f"\n❌ No LLM available!")
            print(f"\n🔧 Quick Setup Options:")
            print(f"   1. Use GLM (Z.AI):")
            print(f"      • Get API key: https://z.ai/manage-apikey/apikey-list")
            print(f"      • Set: export GLM_API_KEY='your-key'")
            print(f"      • Set: export USE_GLM=true")
            print(f"   \n   2. Use Ollama (Recommended - Free):")
            print(f"      • Install: https://ollama.ai/download")
            print(f"      • Pull model: ollama pull {OLLAMA_MODEL}")
            print(f"      • Start: ollama serve")
            print(f"   \n   3. Use the quick start script:")
            print(f"      ./quick_start_ollama.sh")
            print(f"   \n   4. Or use OpenAI:")
            print(f"      export OPENAI_API_KEY='sk-your-key'")
            raise Exception("No LLM available. Please configure GLM, Ollama, or OpenAI.")

        # Confirm LLM is ready
        if llm:
            print(f"✅ LLM ready: {llm_source}")
        else:
            raise Exception("Failed to initialize any LLM")

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        return qa_chain, len(texts)

    except Exception as e:
        raise Exception(f"Error creating RAG system: {str(e)}")

def load_repomix_json(file_path):
    """Load and parse repomix JSON file"""
    from langchain_core.documents import Document

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        processed_files = 0

        # Handle different repomix JSON structures
        if isinstance(data, dict):
            if 'files' in data:
                # Standard repomix format
                print(f"📂 Processing repomix JSON with {len(data['files'])} files")
                for file_info in data['files']:
                    content = file_info.get('content', '')
                    path = file_info.get('path', 'unknown')

                    if content and content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={'source': path, 'file_type': 'repomix'}
                        )
                        documents.append(doc)
                        processed_files += 1

            elif 'chunks' in data:
                # Alternative repomix format
                print(f"📂 Processing repomix JSON with {len(data['chunks'])} chunks")
                for chunk in data['chunks']:
                    content = chunk.get('content', '')
                    path = chunk.get('path', 'unknown')

                    if content and content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={'source': path, 'file_type': 'repomix'}
                        )
                        documents.append(doc)
                        processed_files += 1

            else:
                # Single file in dict format
                content = data.get('content', '')
                path = data.get('path', data.get('filename', 'unknown'))
                if content and content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={'source': path, 'file_type': 'repomix'}
                    )
                    documents.append(doc)
                    processed_files = 1

        elif isinstance(data, list):
            # List format
            print(f"📂 Processing repomix JSON list with {len(data)} items")
            for item in data:
                if isinstance(item, dict):
                    content = item.get('content', '')
                    path = item.get('path', item.get('file', 'unknown'))

                    if content and content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={'source': path, 'file_type': 'repomix'}
                        )
                        documents.append(doc)
                        processed_files += 1

        else:
            # Try to treat as raw content
            content = str(data)
            if content.strip():
                doc = Document(
                    page_content=content,
                    metadata={'source': 'raw_data', 'file_type': 'repomix'}
                )
                documents.append(doc)
                processed_files = 1

        print(f"✅ Successfully processed {processed_files} files from repomix JSON")
        return documents

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        # Fallback: try to load as regular text file
        try:
            loader = TextLoader(file_path)
            return loader.load()
        except Exception as fallback_error:
            raise Exception(f"Failed to parse as JSON and failed to load as text: {fallback_error}")

    except Exception as e:
        print(f"⚠️ Error processing repomix JSON, falling back to text loader: {e}")
        # Fallback: try to load as regular text file
        try:
            loader = TextLoader(file_path)
            return loader.load()
        except Exception as fallback_error:
            raise Exception(f"Failed to process repomix JSON: {e}, fallback also failed: {fallback_error}")

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
    import shutil
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
        import ollama
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