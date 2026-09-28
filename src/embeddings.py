import logging
from typing import List
import numpy as np
from src.config import settings

logger = logging.getLogger("semantic_cache.embeddings")

class EmbeddingEngine:
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self.openai_client = None
        self.fastembed_model = None
        
        if self.provider == "openai" and settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.dimension = 1536
                logger.info("Initialized OpenAI text-embedding-3-small engine.")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embedding engine: {e}. Falling back to local.")
        
        # Fallback to Local Embedding Engine (FastEmbed or lightweight sentence transformer / hashing fallback)
        try:
            from fastembed import TextEmbedding
            logger.info("Initializing FastEmbed (BAAI/bge-small-en-v1.5)...")
            self.fastembed_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
            self.dimension = 384
            self.provider = "local"
            logger.info("FastEmbed engine ready.")
        except Exception as e:
            logger.warning(f"FastEmbed not available ({e}). Using lightweight ONNX/vector fallback.")
            self.provider = "fallback"
            self.dimension = 384

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a normalized float vector."""
        if not text:
            return [0.0] * self.dimension
        
        vecs = self.embed_batch([text])
        return vecs[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into L2-normalized float vectors."""
        if not texts:
            return []
            
        if self.provider == "openai" and self.openai_client:
            try:
                res = self.openai_client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                raw_vecs = [item.embedding for item in res.data]
                return [self._normalize(v) for v in raw_vecs]
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}. Using fallback.")

        if self.fastembed_model:
            embeddings_generator = self.fastembed_model.embed(texts)
            raw_vecs = [list(vec) for vec in embeddings_generator]
            return [self._normalize(v) for v in raw_vecs]

        # Lightweight Fallback Deterministic Embedder (for offline dev testing without external model weights download)
        return [self._fallback_embed(t) for t in texts]

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic lightweight pseudo-embedding using n-gram hashing and SVD/random projection."""
        np.random.seed(42)
        words = text.lower().strip().split()
        vec = np.zeros(self.dimension)
        for w in words:
            # Hash word to seed
            seed = sum(ord(c) * (i + 1) for i, c in enumerate(w)) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            w_vec = rng.normal(0, 1, self.dimension)
            vec += w_vec
        
        if np.linalg.norm(vec) == 0:
            vec = np.random.normal(0, 1, self.dimension)
        return self._normalize(vec.tolist())

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

# Singleton instance
embedding_engine = EmbeddingEngine()
