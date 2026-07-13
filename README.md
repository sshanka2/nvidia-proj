# Government Document Agent

Agentic RAG proof-of-concept for Indian government documents in Tamil and Hindi.

The app ingests a PDF, chunks the extracted text, embeds it with NVIDIA NIM, stores the chunks in a local FAISS index, and answers questions using retrieved context.

## Why this project

Government departments publish large volumes of circulars, orders, notices, and citizen-facing scheme documents in regional languages. Citizens and frontline staff often need quick answers, but keyword search is weak for long PDFs, mixed scripts, and multilingual questions.

This prototype demonstrates how NVIDIA's software stack can support a practical Indian-language document assistant.

## Stack

- Python 3.11 for broad AI-library compatibility
- NVIDIA NIM API for hosted, NVIDIA-optimized model inference
- RAG blueprint pattern for grounded document question answering
- `nvidia/nv-embedqa-e5-v5` for semantic retrieval
- `meta/llama-3.1-8b-instruct` for answer generation
- PyMuPDF for PDF text extraction
- FAISS for local vector search
- Gradio for a simple demo UI

The embedding client sends `input_type="passage"` for document chunks and `input_type="query"` for user questions because NVIDIA's asymmetric embedding models require that distinction.

## Run Instructions

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure NVIDIA API access

Copy the example environment file:

```powershell
copy .env.example .env
```

Edit `.env` and add your NVIDIA API key:

```env
NVIDIA_API_KEY=your_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_LLM_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_EMBED_MODEL=nvidia/nv-embedqa-e5-v5
```

Do not commit the `.env` file.

### 4. Test NVIDIA NIM connectivity

```powershell
python smoke_test.py
```

Expected result: the script prints a short response from the configured NVIDIA-hosted model.

### 5. Ingest a PDF

Place a Tamil or Hindi government PDF inside the `data/` folder, then run:

```powershell
python -m agent.ingest data\your-document.pdf
```

This extracts text, chunks it, creates embeddings through NVIDIA NIM, and stores a local FAISS index under `storage/`.

### 6. Ask a question from the command line

```powershell
python -m agent.agent "What is this document about?"
```

### 7. Run the browser demo

```powershell
python app.py
```

Open the local Gradio URL shown in the terminal.

## Limitations

- The current version works best with text-based PDFs.
- Scanned or image-only PDFs may fail because OCR is not included yet.
- The FAISS vector index is stored locally and is intended for a single-user demo.
- Ingesting a new document replaces the active local index.
- The app requires internet access to call NVIDIA NIM endpoints.
- External users need their own NVIDIA API key from build.nvidia.com.
- Model availability may depend on the user's NVIDIA account access.
- Answer quality depends on PDF text extraction quality, retrieval quality, and the selected NIM model.
- The prototype does not yet include formal evaluation metrics for Tamil vs Hindi performance.
- The first version uses a lightweight custom agent flow; AgentIQ integration is a recommended next step.
