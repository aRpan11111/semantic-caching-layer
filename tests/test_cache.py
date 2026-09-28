import pytest
import time
from fastapi.testclient import TestClient

from src.main import app
from src.cache_key import CacheKeyGenerator
from src.embeddings import embedding_engine
from src.vector_store import InMemoryVectorStore, CacheEntry
from src.cache_policy import TTLClassifier, AdaptiveThreshold

client = TestClient(app)

def test_cache_key_isolation():
    sys1 = "You are a helpful coding assistant."
    sys2 = "You are a concise financial advisor."
    
    hash1 = CacheKeyGenerator.generate_system_hash(sys1)
    hash2 = CacheKeyGenerator.generate_system_hash(sys2)
    
    assert hash1 != hash2
    
    param_hash1 = CacheKeyGenerator.generate_params_hash("gpt-4o", temperature=0.7)
    param_hash2 = CacheKeyGenerator.generate_params_hash("gpt-4o", temperature=0.2)
    
    assert param_hash1 != param_hash2

def test_ttl_classifier():
    ttl_temp, tier_temp = TTLClassifier.classify("What is the stock price of Apple today?")
    assert ttl_temp == 3600
    assert "temporal" in tier_temp
    
    ttl_fact, tier_fact = TTLClassifier.classify("Explain what a binary search tree is in computer science.")
    assert ttl_fact == 604800
    assert "evergreen" in tier_fact

def test_adaptive_threshold():
    thresh_code = AdaptiveThreshold.get_threshold("Write a python function to solve fibonacci", 0.95)
    assert thresh_code == 0.98
    
    thresh_class = AdaptiveThreshold.get_threshold("Classify the sentiment of this text", 0.95)
    assert thresh_class == 0.90

def test_vector_store_cosine_hit():
    store = InMemoryVectorStore()
    
    # Store entry
    vec1 = embedding_engine.embed_text("What is Python programming language?")
    entry = CacheEntry(
        vector=vec1,
        prompt_text="What is Python programming language?",
        system_prompt_hash="sys_test",
        model="mock-model",
        params_hash="params_test",
        response={"choices": [{"message": {"content": "Python is a high-level programming language."}}]}
    )
    store.store_entry(entry)
    
    # Search with exact vector
    hit = store.search_nearest(vec1, "sys_test", "params_test", threshold=0.95)
    assert hit is not None
    assert hit.similarity >= 0.99
    
    # Search with different system prompt hash -> MUST BE MISS (isolated)
    miss_sys = store.search_nearest(vec1, "sys_other", "params_test", threshold=0.95)
    assert miss_sys is None

def test_fastapi_chat_completions_hit_miss():
    from src.vector_store import vector_store
    vector_store.clear_all()
    
    payload = {

        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
    
    # First request -> MISS
    res1 = client.post("/v1/chat/completions", json=payload)
    assert res1.status_code == 200
    assert res1.headers.get("X-Cache-Hit") == "false"
    
    # Second request -> HIT
    res2 = client.post("/v1/chat/completions", json=payload)
    assert res2.status_code == 200
    assert res2.headers.get("X-Cache-Hit") == "true"
    assert float(res2.headers.get("X-Cache-Similarity")) >= 0.95
    assert float(res2.headers.get("X-Response-Time-Ms")) < 100.0
