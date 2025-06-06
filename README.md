# Local LLM PDF Chatbot

## Web Framework
![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?logo=flask)
![Flask-CORS](https://img.shields.io/badge/Flask_CORS-4.0.0-000000)

## AI/ML Libraries
![LangChain](https://img.shields.io/badge/LangChain-0.1.16-00A67E)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-EF4784)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-3.0.0-FF6B6B)
![CTransformers](https://img.shields.io/badge/CTransformers-0.2.27-8A2BE2)

## System Utilities
![py-cpuinfo](https://img.shields.io/badge/py--cpuinfo-9.0.0-5D8AA8)

This project is a chatbot that allows you to upload PDF documents and ask questions about their content. It uses a local Large Language Model (LLM) to generate responses, avoiding the need for expensive API keys.

## Features

- 📄 Upload and process PDF documents
- 💬 Chat with your documents using natural language
- 🧠 Powered by local LLMs (no API costs)
- 🌙 Light/dark mode toggle
- 🔄 Conversation history
- ♻️ Reset chat functionality

## Technologies Used

- **Python** (Flask backend)
- **JavaScript** (Frontend interface)
- **LangChain** (Document processing)
- **ChromaDB** (Vector storage)
- **Sentence Transformers** (Text embeddings)
- **CTransformers** (Local LLM interface)
- **TinyLlama** (1.1B parameter LLM)

## Prerequisites

- Python 3.10+
- 8GB RAM (minimum)
- 2GB free disk space

## Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/johnnietse/localLLM-pdf-chatbot.git
cd localLLM-pdf-chatbot
```

2. Create a virtual environment
```bash
python -m venv .venv
```

- For Linux/Mac:
```bash
source .venv/bin/activate
```

- For Windows:
```bash
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install flask flask-cors langchain chromadb sentence-transformers ctransformers py-cpuinfo
```
or by installing the required packages by
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download the TinyLlama model


```bash
# You can certainly change this to your project's directory, but if you clone this repository you would run this command: `cd build_chatbot_for_your_data`
cd build_chatbot_for_your_data

# Create models directory
mkdir models

# Download TinyLlama (Windows PowerShell)
# by this
curl -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" `
     -OutFile "models\tinyllama.gguf"

# or by this through running the PowerShell script file (download_models.ps1) to download the local LLM models into your project's directory
.\download_models.ps1
```
5. Project Structure

<pre>
chatbot-pdf/
│         
├── build_chatbot_for_your_data/   # Main application folder
│   ├── static/                    # Frontend assets: Static files (JS, CSS)
│   │   ├── script.js              # Client-side JavaScript (chat logic)
│   │   └── style.css              # CSS styling
│   │
│   ├── templates/                 # HTML templates
│   │   └── index.html             # Main chat interface
│   │
│   ├── Dockerfile                 # Docker - Containerization config
│   ├── requirements.txt           # Python dependencies
│   ├── server.py                  # Main server logic (Flask server)
│   ├── server_exercise.py         # For testing or development - Development/test version
│   └── worker.py                  # Background task processing, particularly LLM and document processing
│
├── models/                        # Folder/Directory for local LLM models (currently empty)
│
├── .gitattributes                 # Git attributes config
├── .gitignore                     # Files and folders to ignore in Git
├── LICENSE                        # Usage rights
├── README.md                      # Project documentation
├── deployment.yaml                # Kubernetes deployment config
├── download_models.ps1            # PowerShell script to download models
├── pvc.yaml                       # Kubernetes storage config
└── service.yaml                   # Kubernetes service definition
</pre>

6. Run the Application
```bash
python build_chatbot_for_your_data/server.py
```
Open your web browser and visit: `http://localhost:8000`


## Usage Guide
1. Start Chatting: The chatbot will greet you and ask for a PDF upload
2. Upload PDF: Click "Upload File" and select a PDF document
3. Ask Questions: Type your questions in the input field
4.Reset Chat: Use the refresh button to start a new conversation
5. Toggle Theme: Switch between light/dark mode using the toggle

## Customization Options
### Using Different Models
1. Download any GGUF format model from TheBloke's Hugging Face
2. Place it in the `models/` directory
3. Update `worker.py` with the new model path and type:

```python
# For Mistral model
llm = AutoModelForCausalLM.from_pretrained(
    "../models/mistral-7b.Q4_K_M.gguf",
    model_type="mistral",
    max_new_tokens=1024,
    temperature=0.1
    context_length=2048
)

# For Llama2 model
llm = AutoModelForCausalLM.from_pretrained(
    "../models/llama-2-7b.Q4_K_M.gguf",
    model_type="llama",
    max_new_tokens=1024,
    temperature=0.1
    context_length=2048
)

# For tinyllama model
llm = AutoModelForCausalLM.from_pretrained(
    "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    model_type="llama",
    max_new_tokens=1024,
    temperature=0.1
    context_length=2048
)
```

### Performance Tuning
Adjust these parameters in `worker.py` for better performance:

```python
llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type="llama",
    max_new_tokens=1024,      # Increase for longer responses
    temperature=0.3,          # Increase for more creative responses
    context_length=4096,      # Increase for larger documents
    gpu_layers=40,            # Enable if you have NVIDIA GPU
    threads=8                 # Use more CPU cores
)
```

## Troubleshooting
### Common Issues
1. Model not found:
- Verify model is in `models/` directory
- Check filename in `worker.py`

2. Slow responses:
- Reduce `max_new_tokens`
- Use smaller model
- Add `gpu_layers=40` if you have NVIDIA GPU

3. Memory errors:
- Reduce `chunk_size` in `process_document()`
- Use smaller model
- Close other memory-intensive applications

### Error Messages
- **"Please upload a PDF document first!"**: Upload a document before asking questions
- **"File not uploaded correctly"**: Try a different PDF file
- **"Error processing document"**: The PDF might be corrupted or encrypted

## 📸 Screenshots
![Screenshot (2135)](https://github.com/user-attachments/assets/f67c7123-6d35-4ad6-9468-b03bfb373094)

![Screenshot (2136)](https://github.com/user-attachments/assets/36bd5927-bedc-44b5-982a-9cad26139755)
