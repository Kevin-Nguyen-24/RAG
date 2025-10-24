"""Configuration management for the Financial ESG RAG system."""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OllamaConfig(BaseModel):
    """Ollama LLM API configuration."""
    api_url: str = os.getenv("OLLAMA_API_URL", "https://ollama-gemma-235329445359.us-central1.run.app")
    model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    timeout: int = 180
    max_tokens: int = 2048
    temperature: float = 0.3  # Lower for more factual responses

class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "esg_financial_reports")
    vector_size: int = 384  # for all-MiniLM-L6-v2

class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    batch_size: int = 32

class MemoryConfig(BaseModel):
    """Memory system configuration."""
    memory_type: str = os.getenv("MEMORY_TYPE", "sqlite")
    db_path: Path = Path(os.getenv("MEMORY_DB_PATH", "./data/memory.db"))
    max_conversation_history: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
    short_term_window: int = 8  # Last N messages for immediate context

class AppConfig(BaseModel):
    """Main application configuration."""
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    app_port: int = int(os.getenv("APP_PORT", "8501"))
    data_dir: Path = Path("./data")
    
    ollama: OllamaConfig = OllamaConfig()
    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    memory: MemoryConfig = MemoryConfig()

# Global config instance
config = AppConfig()

# Ensure data directory exists
config.data_dir.mkdir(parents=True, exist_ok=True)
config.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
