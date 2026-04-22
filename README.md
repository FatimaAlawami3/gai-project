# Road Safety Assistant

Road Safety Assistant is a LangChain + FastAPI RAG system over
`saudi_road_safety_kb.json`.

Prompt behavior is centralized in `prompts.py`. The prompt enforces grounded answers,
fallback replies for unsupported questions, strict citation use, and Arabic/English
answer language matching.

Intent routing is centralized in `intent_router.py`. General messages such as
greetings, thanks, capability questions, and off-topic questions return a generic
answer immediately without vector retrieval or an LLM call. Project questions about
the team, supervisor, course, university, or project purpose are answered from the
static project details in the same router.

The `/ask` response includes `intent` and `used_rag` so you can see whether the
request went through retrieval or was answered by the general router. It also
includes `intent_detail`, `suggested_questions`, and formatted source citations.

Chat history is database-free. The frontend can send recent messages in
`chat_history`, and the backend uses them only to understand follow-up questions.

## Models

- Embeddings: `gemini-embedding-001`
- Embedding dimensions: `3072`
- LLM: `gemini-3.1-pro-preview`

The model names are configurable in `backend/.env`.

## Project structure

```text
GAI_Pdf's/
├── backend/      FastAPI, LangChain, prompts, intent router, and backend .env
├── frontend/     React/Vite chat interface
├── data/         Knowledge base JSON, FAISS vector store, PDFs, and credentials
├── scripts/      PDF-to-JSON conversion script
├── logs/         Local server logs
└── README.md
```

## Setup

Install backend dependencies:

```powershell
cd .\backend
python -m pip install -r requirements.txt
```

The local `backend/.env` is already configured for the service-account JSON in
`data/credentials`. If you prefer a Gemini API key, set `GEMINI_API_KEY` in
`backend/.env` and change
`GOOGLE_GENAI_USE_VERTEXAI=false`.

## Build the vector store

```powershell
cd .\backend
python .\build_vector_store.py
```

This creates a local FAISS index in `data/vector_store/`.

## Run the API

```powershell
cd .\backend
python -m uvicorn main:app --reload --port 8000
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Run the frontend

Install Node dependencies once:

```powershell
cd .\frontend
npm install
```

Start the React/Vite frontend:

```powershell
cd .\frontend
npm run dev -- --host 127.0.0.1 --port 5174
```

Open:

- `http://127.0.0.1:5174`

The frontend sends recent messages as `chat_history`, so follow-up questions such
as `What about priority?` can be understood in the previous roundabout context.

Example request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"What should a driver do when approaching a roundabout?","top_k":6}'
```

Follow-up request with chat history:

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
      "content": "A driver should choose the correct lane and signal clearly."
    }
  ]
}
```
