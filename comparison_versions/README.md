# Embedding Comparison Setup

This folder keeps the embedding-model comparison setup separate from the main DALIL backend configuration.

It includes two experiment versions:

- `gemini-embedding-001/`
- `text-multilingual-embedding-002/`

Both versions share the same:

- backend code
- frontend code
- PDF corpus
- JSON knowledge base
- prompt logic
- intent router
- evaluation question set

Only the embedding configuration and vector-store destination differ between the two versions.

This keeps the comparison fair while protecting the original project setup from accidental changes.

## Recommended comparison procedure

1. Build or reuse the Gemini vector store.
2. Build the multilingual vector store.
3. Use the same user questions for both systems.
4. Keep the same `top_k`, LLM model, and prompts.
5. Compare:
   - retrieval relevance
   - answer correctness
   - Arabic question quality
   - follow-up handling quality
   - citation usefulness
