# Gemini Embedding Experiment

This version keeps DALIL on:

- Embedding model: `gemini-embedding-001`
- Embedding dimensions: `3072`
- LLM: `gemini-3.1-pro-preview`
- Vector store path: `data/vector_store_gemini_embedding_001`

Use this as the main system configuration in the comparison.

## Run

Build the vector store:

```powershell
.\build_vector_store.ps1
```

Run the backend:

```powershell
.\run_api.ps1
```
