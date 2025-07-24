from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os

# Load your documents from /data folder
def load_documents():
    return SimpleDirectoryReader("data").load_data()

# Build the vector index
def build_index():
    documents = load_documents()

    # Use Hugging Face embedding model (runs locally)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use Gemma via Ollama
    Settings.llm = Ollama(model="gemma:2b")  # <-- Fix here

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist("vector_store")
    return index

# Load saved vector index
def load_index():
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir="vector_store")
    return load_index_from_storage(storage_context)

# Ask a question
def ask(query, index=None):
    if not index:
        index = load_index()
    return index.as_query_engine().query(query)
