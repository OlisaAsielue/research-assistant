"""Build or load the Chroma vector index over processed papers and expose a query engine."""
import argparse

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

PROCESSED_DIR = "data/processed"
PERSIST_DIR = "./chroma_db"
COLLECTION = "research_papers"

# Small local embedding model; Mistral-7B served locally by Ollama for generation.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=120.0)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


def build_index(rebuild: bool = False) -> VectorStoreIndex:
    """Load the persisted index, or embed data/processed into Chroma if empty or rebuild=True."""
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    if rebuild:
        try:
            db.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = db.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store)

    documents = SimpleDirectoryReader(PROCESSED_DIR, required_exts=[".txt"]).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )


def initialize_rag(similarity_top_k: int = 4, rebuild: bool = False):
    """Query engine that retrieves the top-k chunks and answers from them."""
    return build_index(rebuild=rebuild).as_query_engine(similarity_top_k=similarity_top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the index, optionally ask one question.")
    parser.add_argument("--rebuild", action="store_true", help="re-embed all processed papers")
    parser.add_argument("--query", help="question to answer from the indexed papers")
    args = parser.parse_args()

    engine = initialize_rag(rebuild=args.rebuild)
    print(f"Index ready in {PERSIST_DIR} (collection '{COLLECTION}').")
    if args.query:
        response = engine.query(args.query)
        print(response)
        for node in response.source_nodes:
            print(f"  - {node.metadata.get('file_name', '?')} (score {node.score:.3f})")
