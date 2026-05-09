# DALIL - Road Safety Guide AI Chatbot

DALIL is a bilingual AI-powered road safety chatbot for Saudi Arabia built with
FastAPI, LangChain, FAISS, and Vertex AI embeddings. It uses a
Retrieval-Augmented Generation (RAG) pipeline over a curated Saudi road-safety
knowledge base to answer questions about:

- Saudi traffic law
- accidents and post-accident procedures
- roundabouts and right of way
- licensing and violations
- vehicle modification rules
- phone use while driving
- general road-safety guidance

The system is designed to stay grounded in the indexed source documents rather
than generating unsupported legal or safety advice.

## Core behavior

Prompt behavior is centralized in `backend/prompts.py`.

The prompt enforces:

- grounded answers from retrieved sources
- clarification for underspecified questions when needed
- fallback replies for unsupported or out-of-scope questions
- structured answers with short sections and bullet points
- Arabic/English answer-language matching

Intent routing is centralized in `backend/intent_router.py`.

The router handles:

- greetings and general conversational messages
- thanks and simple social messages
- meta/project questions such as team and supervisor
- out-of-scope questions
- follow-up detection before retrieval
- topic-aware routing for domains such as accident, roundabout, phone use, and
  vehicle modification

## Models

### Primary configuration

- Embeddings: `gemini-embedding-001`
- Embedding dimensions: `3072`
- LLM: `gemini-3.1-pro-preview`

### Comparison baseline

- Embeddings: `text-multilingual-embedding-002`
- Embedding dimensions: `768`
- LLM: `gemini-3.1-pro-preview`

The active profile is configurable through `backend/.env` or the comparison
settings files under `comparison_versions/`.

## Embedding comparison support

The project includes a controlled embedding comparison workflow between:

- `gemini-embedding-001`
- `text-multilingual-embedding-002`

Comparison assets live under `comparison_versions/`, including:

- per-model `settings.env`
- PowerShell helpers to build vector stores and run the API
- the evaluation question sheet
- the report-ready comparison write-up
- saved test-result files

The frontend also includes an embedding-model switch so both retrieval profiles
can be tested from the same UI.

## Project structure

```text
GAI_Pdf's/
├── backend/                    FastAPI app, RAG chain, prompts, router, config
├── frontend/                   React/Vite chat interface
├── data/                       Knowledge base JSON, FAISS stores, PDFs, credentials
├── comparison_versions/        Embedding comparison configs, reports, results
├── scripts/                    PDF-to-JSON conversion script
├── logs/                       Local server logs
└── README.md
```

## Backend setup

Install backend dependencies:

```powershell
cd .\backend
python -m pip install -r requirements.txt
```

The local `backend/.env` is configured for Vertex AI using the service-account
JSON under `data/credentials`.

If you prefer an API-key configuration instead, set `GEMINI_API_KEY` in
`backend/.env` and switch:

```env
GOOGLE_GENAI_USE_VERTEXAI=false
```

## Build the default vector store

```powershell
cd .\backend
python .\build_vector_store.py
```

This builds the local FAISS index in `data/vector_store/`.

## Run the backend

```powershell
cd .\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8011
```

Open:

- [http://127.0.0.1:8011/docs](http://127.0.0.1:8011/docs)
- [http://127.0.0.1:8011/health](http://127.0.0.1:8011/health)

## Run the frontend

Install frontend dependencies once:

```powershell
cd .\frontend
npm install
```

Start the Vite frontend:

```powershell
cd .\frontend
npm run dev -- --host 127.0.0.1 --port 5174
```

Open:

- [http://127.0.0.1:5174](http://127.0.0.1:5174)

The frontend is configured to call the backend on `http://127.0.0.1:8011`.

## Comparison workflow

The comparison setup uses separate folders and vector stores so the multilingual
baseline does not overwrite the main Gemini configuration.

### Gemini comparison profile

- `comparison_versions/gemini-embedding-001/settings.env`
- vector store path: `data/vector_store_gemini_embedding_001`

### Multilingual comparison profile

- `comparison_versions/text-multilingual-embedding-002/settings.env`
- vector store path: `data/vector_store_text_multilingual_embedding_002`

### Build a profile-specific vector store

Example for the multilingual profile:

```powershell
cd .\comparison_versions\text-multilingual-embedding-002
.\build_vector_store.ps1
```

### Run the API with a profile-specific env file

Example for the multilingual profile:

```powershell
cd .\backend
$env:RAG_ENV_FILE="C:\Users\fatom\OneDrive\Desktop\GAI_Pdf's\comparison_versions\text-multilingual-embedding-002\settings.env"
python -m uvicorn main:app --host 127.0.0.1 --port 8011
```

## API behavior

The main endpoints are:

- `POST /ask`
- `POST /ask/stream`

The response metadata includes fields such as:

- `intent`
- `intent_detail`
- `used_rag`
- `needs_clarification`
- `suggested_questions`
- `embedding_profile`

This makes it easier to inspect whether a response came from retrieval, router
logic, clarification logic, or a non-RAG path.

## Example request

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8011/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"What should a driver do when approaching a roundabout?","top_k":6}'
```

## Example follow-up request with chat history

```json
{
  "question": "What about priority?",
  "top_k": 3,
  "chat_history": [
    {
      "role": "user",
      "content": "What should a driver do when approaching a roundabout?"
    },
    {
      "role": "assistant",
      "content": "A driver should choose the correct lane and give way to vehicles already inside the roundabout."
    }
  ]
}
```

