from rag_chain import build_vector_store, load_kb_documents
from rag_config import get_settings


def main() -> None:
    settings = get_settings()
    documents = load_kb_documents(settings.knowledge_base_path)
    print(f"Loaded {len(documents)} chunks from {settings.knowledge_base_path}")
    print(
        "Embedding with "
        f"{settings.embedding_model} ({settings.embedding_dimensions} dimensions)"
    )
    build_vector_store(settings)
    print(f"Vector store saved to {settings.vector_store_path}")


if __name__ == "__main__":
    main()
