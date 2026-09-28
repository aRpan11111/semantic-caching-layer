import os
from typing import Literal
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_DEFAULT_MODEL: str = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.1-8b-instant")
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_DEFAULT_MODEL: str = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Provider fallback strategy
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "groq")
    
    # Vector store configuration: chroma | memory
    VECTOR_STORE_TYPE: Literal["chroma", "memory", "qdrant", "redis"] = os.getenv("VECTOR_STORE_TYPE", "chroma")
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db_semantic_cache")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "llm_semantic_cache")
    
    # Cache Policy defaults
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))
    DEFAULT_TTL_SECONDS: int = int(os.getenv("DEFAULT_TTL_SECONDS", "86400"))
    
    # Embedding Model selection
    EMBEDDING_PROVIDER: Literal["openai", "local"] = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384 if os.getenv("EMBEDDING_PROVIDER", "local") == "local" else 1536

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
