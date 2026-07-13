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

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and add your NVIDIA API key.

## Smoke test

```powershell
python smoke_test.py
```

## Ingest a Document

Put a Tamil or Hindi government PDF in `data/`, then run:

```powershell
python -m agent.ingest data\your-document.pdf
```

## Ask a Question

```powershell
python -m agent.agent "What is this document about?"
```

## Run the Demo UI

```powershell
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Startup Pitch Customization

### Sarvam AI

Sarvam's opportunity is Indian-language document intelligence at enterprise and government scale. This demo can be positioned as a lightweight application layer on top of Indic models, with NVIDIA NIM, TensorRT-LLM, and Triton helping reduce inference latency and cost for production document workflows.

### Gnani.ai

Gnani focuses on multilingual voice AI for Indian enterprises. This demo can be adapted into a voice-enabled citizen service flow: a user asks a question by phone, the system retrieves an answer from a government document, and responds in the user's language. NVIDIA's pitch centers on low-latency ASR, LLM, and TTS inference.

### AI4Bharat

AI4Bharat's work on Indic datasets, OCR, and language resources fits the data and evaluation layer. This demo can use AI4Bharat-style datasets to benchmark retrieval and answer quality across Indian languages, while NVIDIA provides accelerated training and inference infrastructure.
