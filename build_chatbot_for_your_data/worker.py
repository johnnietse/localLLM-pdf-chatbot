# import os
# from langchain.llms import CTransformers
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
# from ctransformers import AutoModelForCausalLM  # Import directly from ctransformers
#
# # Initialize global variables
# qa_chain = None
# chat_history = []
# llm = None
# llm_embeddings = None
#
#
# def init_llm():
#     global llm, llm_embeddings
#     # Initialize embeddings
#     llm_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
#     # Initialize local LLM using direct ctransformers method
#     model_path = "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#
#     # Load model directly from local file
#     llm = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         model_type="llama",  # TinyLlama is Llama-based
#         config={
#             'max_new_tokens': 512,
#             'temperature': 0.1,
#             'context_length': 2048,
#             'gpu_layers': 0,  # Set to >0 if you have GPU
#             'threads': os.cpu_count() // 2  # Use half of available CPU cores
#         }
#     )
#
#
# def process_document(document_path):
#     global qa_chain, llm, llm_embeddings
#
#     # Load and split document
#     loader = PyPDFLoader(document_path)
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = text_splitter.split_documents(documents)
#
#     # Create vector store
#     db = Chroma.from_documents(texts, llm_embeddings)
#     retriever = db.as_retriever(search_kwargs={"k": 3})
#
#     # Create QA chain - need to use the direct model with RetrievalQA
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )
#
#
# def process_prompt(prompt):
#     global qa_chain, chat_history
#     if not qa_chain:
#         return "Please upload a PDF document first!"
#
#     result = qa_chain({"query": prompt})
#     answer = result["result"]
#
#     # Update chat history
#     chat_history.append((prompt, answer))
#     return answer
#
#
# # Initialize the language model
# init_llm()











#############################################################################################################






# import os
# from ctransformers import AutoModelForCausalLM
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# # from langchain.vectorstores import Chroma

# # Replace these imports at the top of worker.py
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma

# # Initialize global variables
# vector_store = None
# llm = None
# llm_embeddings = None
# # model_path = "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# # Change this line in worker.py
# model_path = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Absolute path in container


# def init_llm():
#     global llm, llm_embeddings
#     llm_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Load model directly with correct parameters
#     llm = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         model_type="llama",
#         max_new_tokens=512,
#         temperature=0.1,
#         context_length=2048
#     )


# def process_document(document_path):
#     global vector_store
#     loader = PyPDFLoader(document_path)
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = text_splitter.split_documents(documents)
#     vector_store = Chroma.from_documents(texts, llm_embeddings)


# def process_prompt(prompt):
#     if not vector_store:
#         return "Please upload a PDF document first!"

#     # Simple retrieval
#     docs = vector_store.similarity_search(prompt,   k=2)
#     context = "\n".join([d.page_content for d in docs])

#     # Create prompt with context
#     full_prompt = f"""Use the following context to answer the question:
#     {context}

#     Question: {prompt}
#     Answer:"""

#     # Generate response
#     answer = llm(full_prompt)
#     return answer


# # Initialize the language model
# init_llm()




######################################################################################################


# import os
# from ctransformers import AutoModelForCausalLM
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter

# # Initialize global variables
# vector_store = None
# llm = None
# llm_embeddings = None
# model_path = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# def init_llm():
#     global llm, llm_embeddings
    
#     # Configure cache directories with proper permissions
#     cache_dir = "/app/.cache/huggingface"
#     os.makedirs(cache_dir, exist_ok=True)
#     os.environ['HF_HOME'] = cache_dir
#     os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
#     # Initialize embeddings with proper configuration
#     llm_embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},  # or 'cuda' if available
#         encode_kwargs={'normalize_embeddings': False}
#     )

#     # Load LLM model
#     llm = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         model_type="llama",
#         max_new_tokens=512,
#         temperature=0.1,
#         context_length=2048
#     )

# def process_document(document_path):
#     global vector_store
#     try:
#         loader = PyPDFLoader(document_path)
#         documents = loader.load()
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         texts = text_splitter.split_documents(documents)
        
#         # Create persistent Chroma database
#         persist_dir = "/app/.chroma_db"
#         vector_store = Chroma.from_documents(
#             documents=texts,
#             embedding=llm_embeddings,
#             persist_directory=persist_dir
#         )
#     except Exception as e:
#         print(f"Error processing document: {str(e)}")
#         raise

# def process_prompt(prompt):
#     if not vector_store:
#         return "Please upload a PDF document first!"

#     try:
#         # Retrieve relevant context
#         docs = vector_store.similarity_search(prompt, k=2)
#         context = "\n".join([d.page_content for d in docs])

#         # Create enhanced prompt template
#         full_prompt = f"""Context information is below.
#         ---------------------
#         {context}
#         ---------------------
#         Given the context information and no prior knowledge, answer the question.
#         Question: {prompt}
#         Answer:"""

#         # Generate response
#         answer = llm(full_prompt)
#         return str(answer).strip()
#     except Exception as e:
#         print(f"Error processing prompt: {str(e)}")
#         return "Sorry, an error occurred while processing your request."

# # Initialize the language model
# init_llm()


import os
import logging
from pathlib import Path
from ctransformers import AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from huggingface_hub import snapshot_download, try_to_load_from_cache

# OpenShift-specific configuration
# class Config:
#     # Change from /app to /tmp for OpenShift compatibility
#     BASE_DIR = "/app/data"  # OpenShift allows writing to /tmp
#     MODEL_PATH = os.getenv("MODEL_PATH", f"{BASE_DIR}/models/tinyllama.gguf")
#     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#     CACHE_DIR = f"{BASE_DIR}/cache"
#     CHROMA_DIR = f"{BASE_DIR}/chroma_db"
    
#     # Resource-constrained settings
#     CHUNK_SIZE = 800  # Reduced for OpenShift memory limits
#     CHUNK_OVERLAP = 50
#     LLM_CONFIG = {
#         "model_type": "llama",
#         "max_new_tokens": 256,  # Reduced for OpenShift
#         "temperature": 0.1,
#         "context_length": 1024,  # Reduced context window
#         "gpu_layers": 0  # Force CPU-only
#     }
class Config:
    # Use absolute paths consistent with container
    BASE_DIR = "/app/data"  # Persistent storage path
    # MODEL_PATH = os.path.join(BASE_DIR, "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    MODEL_PATH = "/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Changed path

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # CACHE_DIR = os.path.join(BASE_DIR, "cache")
    # CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
    CACHE_DIR = "/app/data/cache"
    CHROMA_DIR = "/app/data/chroma_db"
    
    # Resource-constrained settings
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 50
    LLM_CONFIG = {
        "model_type": "llama",
        "max_new_tokens": 256,
        "temperature": 0.1,
        "context_length": 1024,
        "gpu_layers": 0
    }


# Initialize globals
vector_store = None
llm = None
llm_embeddings = None

# def setup_openshift_environment():
#     """Configure paths with OpenShift permissions"""
#     try:
#         # Create writable directories
#         Path(Config.BASE_DIR).mkdir(parents=True, exist_ok=True)
        
#         # Set cache locations
#         os.environ['HF_HOME'] = Config.CACHE_DIR
#         os.environ['TRANSFORMERS_CACHE'] = Config.CACHE_DIR
#         os.environ['HUGGINGFACE_HUB_CACHE'] = Config.CACHE_DIR
        
#         # OpenShift requires specific permissions
#         os.chmod(Config.BASE_DIR, 0o777)
#         logger.info(f"OpenShift environment configured at {Config.BASE_DIR}")
#     except Exception as e:
#         logger.error(f"OpenShift setup failed: {str(e)}")
#         raise


def setup_openshift_environment():
    """Configure paths with OpenShift permissions"""
    try:
        # Create directories if they don't exist (with exist_ok=True)
        Path(Config.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        
        # Set cache locations
        os.environ['HF_HOME'] = Config.CACHE_DIR
        os.environ['TRANSFORMERS_CACHE'] = Config.CACHE_DIR
        os.environ['HUGGINGFACE_HUB_CACHE'] = Config.CACHE_DIR
        
        logger.info(f"OpenShift environment configured at {Config.BASE_DIR}")
    except Exception as e:
        logger.error(f"OpenShift setup failed: {str(e)}")
        raise

def download_model_with_fallback():
    """Handle model downloads with OpenShift restrictions"""
    try:
        # Check if model exists in cache first
        if try_to_load_from_cache(Config.EMBEDDING_MODEL, "model.safetensors"):
            logger.info("Found cached embedding model")
            return

        # Download with OpenShift-friendly settings
        snapshot_download(
            repo_id=Config.EMBEDDING_MODEL,
            cache_dir=Config.CACHE_DIR,
            local_dir=f"{Config.CACHE_DIR}/{Config.EMBEDDING_MODEL.replace('/', '_')}",
            resume_download=True,
            max_workers=1,  # Reduced for OpenShift
            local_files_only=False,
            token=os.getenv("HF_TOKEN")  # Use env var if needed
        )
        logger.info("Embedding model downloaded")
    except Exception as e:
        logger.warning(f"Model download may be incomplete: {str(e)}")
        if not Path(Config.CACHE_DIR).exists():
            raise RuntimeError("Cache directory inaccessible in OpenShift")

def initialize_llm_openshift():
    """LLM initialization with OpenShift constraints"""
    global llm, llm_embeddings
    
    try:
        # Embeddings with low-memory settings
        llm_embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            cache_folder=Config.CACHE_DIR,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 4  # Reduced for OpenShift
            }
        )

        # Verify GGUF model exists
        if not Path(Config.MODEL_PATH).exists():
            raise FileNotFoundError(f"Model missing at {Config.MODEL_PATH}")

        # CPU-only, low-memory configuration
        llm = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_PATH,
            **Config.LLM_CONFIG
        )
        logger.info("Components initialized in OpenShift mode")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

def process_document_openshift(document_path):
    """Document processing with OpenShift limits"""
    global vector_store
    
    try:
        # Verify document exists
        if not Path(document_path).exists():
            raise FileNotFoundError(f"Document not accessible at {document_path}")

        # Load with page limit for OpenShift
        loader = PyPDFLoader(document_path)
        documents = loader.load()[:20]  # Limit to first 20 pages
        
        text_splitter = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        
        # Persistent storage with cleanup handling
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=llm_embeddings,
            persist_directory=Config.CHROMA_DIR,
            client_settings={
                "chroma_db_impl": "duckdb+parquet",
                "persist_directory": Config.CHROMA_DIR
            }
        )
        logger.info(f"Processed {len(texts)} chunks in OpenShift")
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise

def process_prompt_openshift(prompt):
    """Prompt handling with OpenShift restrictions"""
    try:
        if not vector_store:
            return "Please upload a document first (OpenShift-ready)"

        # Memory-constrained search
        docs = vector_store.similarity_search(
            prompt, 
            k=1  # Only 1 result for OpenShift
        )
        context = "\n".join([d.page_content[:500] for d in docs])  # Truncated

        # Optimized prompt template
        response = llm(
            f"Context: {context}\n\nQuestion: {prompt[:200]}\nAnswer:"
        )
        return str(response).strip()[:500]  # Response length limit
    except Exception as e:
        logger.error(f"OpenShift prompt error: {str(e)}")
        return "System busy (OpenShift-limited)"

# Configure logging for OpenShift
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/app.log')
    ]
)
logger = logging.getLogger('openshift_worker')

# Initialize in OpenShift mode
try:
    setup_openshift_environment()
    download_model_with_fallback()
    initialize_llm_openshift()
except Exception as e:
    logger.critical(f"OpenShift initialization failed: {str(e)}")
    raise