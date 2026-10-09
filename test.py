import faiss

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load the FAISS vector store
vector_store = FaissVectorStore.from_persist_dir(
    persist_dir="vector_store"
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir="vector_store"
)

index = load_index_from_storage(storage_context)

retriever = index.as_retriever(similarity_top_k=3)

question = input("Ask a question about SPIT: ")
results = retriever.retrieve(question)

for i, result in enumerate(results, 1):
    print(f"\n{'=' * 50}")
    print(f"RESULT {i}")
    print(f"Similarity score: {result.score}")
    print(f"Source: {result.node.metadata.get('source_title', 'Unknown')}")
    print(f"URL: {result.node.metadata.get('source_url', 'Unknown')}")
    print("\nRetrieved text:")
    print(result.node.get_content() or "[No text available]")