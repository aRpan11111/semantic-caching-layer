import time
import uuid
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger("semantic_cache.vector_store")

class CacheEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float]
    prompt_text: str
    system_prompt_hash: str
    model: str
    params_hash: str
    response: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
    ttl_seconds: int = 86400
    expires_at: float = Field(default_factory=lambda: time.time() + 86400)
    hit_count: int = 0
    tags: List[str] = Field(default_factory=list)

class VectorSearchResult(BaseModel):
    entry: CacheEntry
    similarity: float

class BaseVectorStore:
    def search_nearest(
        self,
        query_vector: List[float],
        system_prompt_hash: str,
        params_hash: str,
        threshold: float = 0.95
    ) -> Optional[VectorSearchResult]:
        raise NotImplementedError

    def store_entry(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    def invalidate_by_system_prompt(self, system_prompt_hash: str) -> int:
        raise NotImplementedError

    def invalidate_by_model(self, model: str) -> int:
        raise NotImplementedError

    def invalidate_by_tag(self, tag: str) -> int:
        raise NotImplementedError

    def find_near_misses(
        self,
        query_vector: List[float],
        system_prompt_hash: str,
        params_hash: str,
        threshold: float = 0.95,
        lower_bound: float = 0.80
    ) -> List[VectorSearchResult]:
        raise NotImplementedError

    def clear_all(self) -> int:
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """Production ChromaDB Vector Store with persistent storage and metadata filtering."""
    def __init__(self):
        try:
            import chromadb
            self.chroma_path = getattr(settings, "CHROMA_PATH", "./chroma_db_semantic_cache")
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.client.get_or_create_collection(
                name=getattr(settings, "CHROMA_COLLECTION", "llm_semantic_cache"),
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaVectorStore at '{self.chroma_path}'.")
            self.fallback = None
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB ({e}). Falling back to InMemoryVectorStore.")
            self.fallback = InMemoryVectorStore()

    def search_nearest(self, query_vector: List[float], system_prompt_hash: str, params_hash: str, threshold: float = 0.95) -> Optional[VectorSearchResult]:
        if self.fallback:
            return self.fallback.search_nearest(query_vector, system_prompt_hash, params_hash, threshold)
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=5,
                where={
                    "$and": [
                        {"system_prompt_hash": {"$eq": system_prompt_hash}},
                        {"params_hash": {"$eq": params_hash}}
                    ]
                }
            )
            
            if not results or not results.get("ids") or not results["ids"][0]:
                return None

            now = time.time()
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = round(1.0 - distance, 4)
                meta = results["metadatas"][0][i]
                
                # Check expiration
                if now > meta.get("expires_at", 0):
                    continue

                if similarity >= threshold:
                    response_dict = json.loads(meta["response_json"])
                    entry = CacheEntry(
                        id=doc_id,
                        vector=query_vector,
                        prompt_text=meta["prompt_text"],
                        system_prompt_hash=meta["system_prompt_hash"],
                        model=meta["model"],
                        params_hash=meta["params_hash"],
                        response=response_dict,
                        ttl_seconds=meta["ttl_seconds"],
                        expires_at=meta["expires_at"],
                        hit_count=meta.get("hit_count", 0) + 1
                    )
                    
                    # Update hit count in ChromaDB
                    meta["hit_count"] = entry.hit_count
                    self.collection.update(
                        ids=[doc_id],
                        metadatas=[meta]
                    )
                    return VectorSearchResult(entry=entry, similarity=similarity)

        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
        
        return None

    def store_entry(self, entry: CacheEntry) -> None:
        if self.fallback:
            return self.fallback.store_entry(entry)
        
        try:
            metadata = {
                "prompt_text": entry.prompt_text,
                "system_prompt_hash": entry.system_prompt_hash,
                "model": entry.model,
                "params_hash": entry.params_hash,
                "response_json": json.dumps(entry.response),
                "ttl_seconds": entry.ttl_seconds,
                "expires_at": entry.expires_at,
                "hit_count": entry.hit_count,
                "tags": ",".join(entry.tags)
            }
            
            self.collection.add(
                ids=[entry.id],
                embeddings=[entry.vector],
                documents=[entry.prompt_text],
                metadatas=[metadata]
            )
            logger.debug(f"Stored entry {entry.id} in ChromaDB.")
        except Exception as e:
            logger.error(f"ChromaDB store error: {e}")

    def find_near_misses(self, query_vector: List[float], system_prompt_hash: str, params_hash: str, threshold: float = 0.95, lower_bound: float = 0.80) -> List[VectorSearchResult]:
        if self.fallback:
            return self.fallback.find_near_misses(query_vector, system_prompt_hash, params_hash, threshold, lower_bound)
        
        near_misses = []
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=10,
                where={
                    "$and": [
                        {"system_prompt_hash": {"$eq": system_prompt_hash}},
                        {"params_hash": {"$eq": params_hash}}
                    ]
                }
            )
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = round(1.0 - distance, 4)
                    meta = results["metadatas"][0][i]
                    
                    if lower_bound <= similarity < threshold:
                        response_dict = json.loads(meta["response_json"])
                        entry = CacheEntry(
                            id=doc_id,
                            vector=query_vector,
                            prompt_text=meta["prompt_text"],
                            system_prompt_hash=meta["system_prompt_hash"],
                            model=meta["model"],
                            params_hash=meta["params_hash"],
                            response=response_dict,
                            ttl_seconds=meta["ttl_seconds"],
                            expires_at=meta["expires_at"]
                        )
                        near_misses.append(VectorSearchResult(entry=entry, similarity=similarity))
        except Exception as e:
            logger.error(f"ChromaDB near-miss query error: {e}")

        near_misses.sort(key=lambda x: x.similarity, reverse=True)
        return near_misses

    def invalidate_by_system_prompt(self, system_prompt_hash: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_system_prompt(system_prompt_hash)
        try:
            res = self.collection.get(where={"system_prompt_hash": {"$eq": system_prompt_hash}})
            ids = res.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"ChromaDB invalidation error: {e}")
            return 0

    def invalidate_by_model(self, model: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_model(model)
        try:
            res = self.collection.get(where={"model": {"$eq": model}})
            ids = res.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"ChromaDB invalidation error: {e}")
            return 0

    def invalidate_by_tag(self, tag: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_tag(tag)
        return 0

    def clear_all(self) -> int:
        if self.fallback:
            return self.fallback.clear_all()
        try:
            count = self.collection.count()
            self.client.delete_collection(name=getattr(settings, "CHROMA_COLLECTION", "llm_semantic_cache"))
            self.collection = self.client.get_or_create_collection(
                name=getattr(settings, "CHROMA_COLLECTION", "llm_semantic_cache"),
                metadata={"hnsw:space": "cosine"}
            )
            return count
        except Exception as e:
            logger.error(f"ChromaDB clear_all error: {e}")
            return 0


class InMemoryVectorStore(BaseVectorStore):
    """Fast, thread-safe in-memory vector store using NumPy cosine similarity."""
    def __init__(self):
        self.entries: Dict[str, CacheEntry] = {}
        logger.info("Initialized InMemoryVectorStore (NumPy Cosine Engine).")

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _filter_valid_entries(self, system_prompt_hash: str, params_hash: str) -> List[CacheEntry]:
        now = time.time()
        valid = []
        expired_ids = []
        for entry_id, entry in list(self.entries.items()):
            if now > entry.expires_at:
                expired_ids.append(entry_id)
                continue
            if entry.system_prompt_hash == system_prompt_hash and entry.params_hash == params_hash:
                valid.append(entry)
        
        for eid in expired_ids:
            self.entries.pop(eid, None)
            
        return valid

    def search_nearest(
        self,
        query_vector: List[float],
        system_prompt_hash: str,
        params_hash: str,
        threshold: float = 0.95
    ) -> Optional[VectorSearchResult]:
        candidates = self._filter_valid_entries(system_prompt_hash, params_hash)
        if not candidates:
            return None

        best_similarity = -1.0
        best_entry = None

        for entry in candidates:
            sim = self._cosine_similarity(query_vector, entry.vector)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_entry and best_similarity >= threshold:
            best_entry.hit_count += 1
            return VectorSearchResult(entry=best_entry, similarity=round(best_similarity, 4))
        
        return None

    def find_near_misses(
        self,
        query_vector: List[float],
        system_prompt_hash: str,
        params_hash: str,
        threshold: float = 0.95,
        lower_bound: float = 0.80
    ) -> List[VectorSearchResult]:
        candidates = self._filter_valid_entries(system_prompt_hash, params_hash)
        near_misses = []
        for entry in candidates:
            sim = self._cosine_similarity(query_vector, entry.vector)
            if lower_bound <= sim < threshold:
                near_misses.append(VectorSearchResult(entry=entry, similarity=round(sim, 4)))
        
        near_misses.sort(key=lambda x: x.similarity, reverse=True)
        return near_misses

    def store_entry(self, entry: CacheEntry) -> None:
        self.entries[entry.id] = entry

    def invalidate_by_system_prompt(self, system_prompt_hash: str) -> int:
        to_remove = [k for k, v in self.entries.items() if v.system_prompt_hash == system_prompt_hash]
        for k in to_remove:
            del self.entries[k]
        return len(to_remove)

    def invalidate_by_model(self, model: str) -> int:
        to_remove = [k for k, v in self.entries.items() if v.model.lower() == model.lower()]
        for k in to_remove:
            del self.entries[k]
        return len(to_remove)

    def invalidate_by_tag(self, tag: str) -> int:
        to_remove = [k for k, v in self.entries.items() if tag in v.tags]
        for k in to_remove:
            del self.entries[k]
        return len(to_remove)

    def clear_all(self) -> int:
        count = len(self.entries)
        self.entries.clear()
        return count


class QdrantVectorStore(BaseVectorStore):
    """High-performance Qdrant Vector Store implementation."""
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
            self.host = getattr(settings, "QDRANT_HOST", "localhost")
            self.port = getattr(settings, "QDRANT_PORT", 6333)
            self.collection_name = getattr(settings, "QDRANT_COLLECTION", "llm_semantic_cache")
            
            self.client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
            # Create collection if missing
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            logger.info(f"Initialized QdrantVectorStore at '{self.host}:{self.port}'.")
            self.fallback = None
        except Exception as e:
            logger.warning(f"Failed to initialize QdrantDB ({e}). Falling back to InMemoryVectorStore.")
            self.fallback = InMemoryVectorStore()

    def search_nearest(self, query_vector: List[float], system_prompt_hash: str, params_hash: str, threshold: float = 0.95) -> Optional[VectorSearchResult]:
        if self.fallback:
            return self.fallback.search_nearest(query_vector, system_prompt_hash, params_hash, threshold)
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=5
            )
            for hit in hits:
                payload = hit.payload or {}
                if payload.get("system_prompt_hash") == system_prompt_hash and payload.get("params_hash") == params_hash:
                    if hit.score >= threshold:
                        entry = CacheEntry(**payload["cache_entry"])
                        entry.hit_count += 1
                        return VectorSearchResult(entry=entry, similarity=float(hit.score))
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
        return None

    def store_entry(self, entry: CacheEntry) -> None:
        if self.fallback:
            return self.fallback.store_entry(entry)
        try:
            from qdrant_client.models import PointStruct
            point = PointStruct(
                id=entry.id,
                vector=entry.vector,
                payload={
                    "system_prompt_hash": entry.system_prompt_hash,
                    "params_hash": entry.params_hash,
                    "model": entry.model,
                    "cache_entry": entry.dict()
                }
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
        except Exception as e:
            logger.error(f"Qdrant store error: {e}")

    def find_near_misses(self, query_vector: List[float], system_prompt_hash: str, params_hash: str, threshold: float = 0.95, lower_bound: float = 0.80) -> List[VectorSearchResult]:
        if self.fallback:
            return self.fallback.find_near_misses(query_vector, system_prompt_hash, params_hash, threshold, lower_bound)
        return []

    def invalidate_by_system_prompt(self, system_prompt_hash: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_system_prompt(system_prompt_hash)
        return 0

    def invalidate_by_model(self, model: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_model(model)
        return 0

    def invalidate_by_tag(self, tag: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_tag(tag)
        return 0

    def clear_all(self) -> int:
        if self.fallback:
            return self.fallback.clear_all()
        return 0


class RedisVectorStore(BaseVectorStore):
    """High-performance Redis Vector Store implementation using Redis HSET & vector search."""
    def __init__(self):
        try:
            import redis
            self.host = getattr(settings, "REDIS_HOST", "localhost")
            self.port = getattr(settings, "REDIS_PORT", 6379)
            self.client = redis.Redis(host=self.host, port=self.port, socket_timeout=3.0)
            self.client.ping()
            logger.info(f"Initialized RedisVectorStore at '{self.host}:{self.port}'.")
            self.fallback = None
        except Exception as e:
            logger.warning(f"Failed to initialize RedisDB ({e}). Falling back to InMemoryVectorStore.")
            self.fallback = InMemoryVectorStore()

    def search_nearest(self, query_vector: List[float], system_prompt_hash: str, params_hash: str, threshold: float = 0.95) -> Optional[VectorSearchResult]:
        if self.fallback:
            return self.fallback.search_nearest(query_vector, system_prompt_hash, params_hash, threshold)
        try:
            # Query keys
            keys = self.client.keys("cache_entry:*")
            best_match = None
            best_sim = -1.0
            query_arr = np.array(query_vector, dtype=np.float32)
            norm_q = np.linalg.norm(query_arr)

            for k in keys:
                raw_data = self.client.get(k)
                if not raw_data:
                    continue
                entry_dict = json.loads(raw_data.decode("utf-8"))
                if entry_dict.get("system_prompt_hash") == system_prompt_hash and entry_dict.get("params_hash") == params_hash:
                    vec_arr = np.array(entry_dict["vector"], dtype=np.float32)
                    norm_v = np.linalg.norm(vec_arr)
                    if norm_q > 0 and norm_v > 0:
                        sim = float(np.dot(query_arr, vec_arr) / (norm_q * norm_v))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = CacheEntry(**entry_dict)

            if best_match and best_sim >= threshold:
                best_match.hit_count += 1
                return VectorSearchResult(entry=best_match, similarity=best_sim)
        except Exception as e:
            logger.error(f"Redis search error: {e}")
        return None

    def store_entry(self, entry: CacheEntry) -> None:
        if self.fallback:
            return self.fallback.store_entry(entry)
        try:
            key = f"cache_entry:{entry.id}"
            self.client.set(key, json.dumps(entry.dict()), ex=entry.ttl_seconds)
        except Exception as e:
            logger.error(f"Redis store error: {e}")

    def find_near_misses(self, query_vector: List[float], system_prompt_hash: str, params_hash: str, threshold: float = 0.95, lower_bound: float = 0.80) -> List[VectorSearchResult]:
        if self.fallback:
            return self.fallback.find_near_misses(query_vector, system_prompt_hash, params_hash, threshold, lower_bound)
        return []

    def invalidate_by_system_prompt(self, system_prompt_hash: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_system_prompt(system_prompt_hash)
        return 0

    def invalidate_by_model(self, model: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_model(model)
        return 0

    def invalidate_by_tag(self, tag: str) -> int:
        if self.fallback:
            return self.fallback.invalidate_by_tag(tag)
        return 0

    def clear_all(self) -> int:
        if self.fallback:
            return self.fallback.clear_all()
        try:
            keys = self.client.keys("cache_entry:*")
            if keys:
                return self.client.delete(*keys)
        except Exception:
            pass
        return 0


def get_vector_store() -> BaseVectorStore:
    vtype = getattr(settings, "VECTOR_STORE_TYPE", "chroma").lower()
    if vtype == "redis":
        return RedisVectorStore()
    elif vtype == "qdrant":
        return QdrantVectorStore()
    elif vtype == "chroma" or vtype == "chromadb":
        return ChromaVectorStore()
    return InMemoryVectorStore()

vector_store = get_vector_store()


