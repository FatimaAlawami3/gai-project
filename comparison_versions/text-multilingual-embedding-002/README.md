# Multilingual Embedding Experiment

This version keeps DALIL on:

- Embedding model: `text-multilingual-embedding-002`
- Embedding dimensions: `768`
- LLM: `gemini-3.1-pro-preview`
- Vector store path: `data/vector_store_text_multilingual_embedding_002`

Use this as the comparison baseline against the Gemini embedding setup.

## Run

Build the vector store:

```powershell
.\build_vector_store.ps1
```

Run the backend:

```powershell
.\run_api.ps1
```
