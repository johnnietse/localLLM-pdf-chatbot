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


import os
from ctransformers import AutoModelForCausalLM
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma

# Replace these imports at the top of worker.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# Initialize global variables
vector_store = None
llm = None
llm_embeddings = None
# model_path = "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Change this line in worker.py
model_path = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Absolute path in container


def init_llm():
    global llm, llm_embeddings
    llm_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load model directly with correct parameters
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.1,
        context_length=2048
    )


def process_document(document_path):
    global vector_store
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(texts, llm_embeddings)


def process_prompt(prompt):
    if not vector_store:
        return "Please upload a PDF document first!"

    # Simple retrieval
    docs = vector_store.similarity_search(prompt,   k=2)
    context = "\n".join([d.page_content for d in docs])

    # Create prompt with context
    full_prompt = f"""Use the following context to answer the question:
    {context}

    Question: {prompt}
    Answer:"""

    # Generate response
    answer = llm(full_prompt)
    return answer


# Initialize the language model
init_llm()