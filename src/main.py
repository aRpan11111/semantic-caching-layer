import time
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import settings
from src.embeddings import embedding_engine
from src.cache_key import CacheKeyGenerator
from src.vector_store import vector_store, CacheEntry
from src.cache_policy import TTLClassifier, AdaptiveThreshold
from src.providers import provider_router
from src.streaming import StreamProcessor
from src.metrics import stats_tracker
from src.management_api import router as management_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("semantic_cache.proxy")

app = FastAPI(
    title="Semantic Cache Service",
    description="Middleware caching proxy for LLM APIs (Groq, OpenAI, Anthropic).",
    version="1.0.0"
)

app.include_router(management_router)

Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {
        "service": "Semantic Cache Service",
        "status": "ok",
        "vector_store": settings.VECTOR_STORE_TYPE,
        "default_provider": settings.DEFAULT_PROVIDER,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "vector_store": settings.VECTOR_STORE_TYPE,
        "groq_configured": bool(settings.GROQ_API_KEY)
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions proxy endpoint."""
    start_time = time.time()
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Field 'messages' is required.")

    model = body.get("model", settings.GROQ_DEFAULT_MODEL if settings.GROQ_API_KEY else "gpt-4o-mini")
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")
    is_stream = body.get("stream", False)

    # Compute key hashes
    user_prompt = CacheKeyGenerator.extract_user_prompt(messages)
    system_prompt = CacheKeyGenerator.extract_system_prompt(messages)
    
    sys_hash = CacheKeyGenerator.generate_system_hash(system_prompt)
    params_hash = CacheKeyGenerator.generate_params_hash(model, temperature, max_tokens)

    prompt_vector = embedding_engine.embed_text(user_prompt)

    # Dynamic similarity threshold selection
    active_threshold = AdaptiveThreshold.get_threshold(user_prompt, settings.SIMILARITY_THRESHOLD)

    # Vector store lookup
    cache_result = vector_store.search_nearest(
        query_vector=prompt_vector,
        system_prompt_hash=sys_hash,
        params_hash=params_hash,
        threshold=active_threshold
    )

    # Cache Hit
    if cache_result:
        latency_s = time.time() - start_time
        latency_ms = round(latency_s * 1000, 2)
        cached_entry = cache_result.entry
        similarity = cache_result.similarity
        response_dict = cached_entry.response

        usage = response_dict.get("usage", {})
        stats_tracker.record_hit(model=model, latency=latency_s, similarity=similarity, usage=usage)

        logger.info(f"Cache hit [sim={similarity:.4f}, lat={latency_ms:.1f}ms] prompt='{user_prompt[:40]}...'")

        if is_stream:
            return StreamingResponse(
                StreamProcessor.simulate_stream_cached(response_dict),
                media_type="text/event-stream",
                headers={
                    "X-Cache-Hit": "true",
                    "X-Cache-Similarity": str(similarity),
                    "X-Cache-TTL": str(cached_entry.ttl_seconds),
                    "X-Response-Time-Ms": str(latency_ms)
                }
            )

        return JSONResponse(
            content=response_dict,
            headers={
                "X-Cache-Hit": "true",
                "X-Cache-Similarity": str(similarity),
                "X-Cache-TTL": str(cached_entry.ttl_seconds),
                "X-Response-Time-Ms": str(latency_ms),
                "X-Cost-Saved-USD": str(round((usage.get("prompt_tokens", 0) * 0.00000015) + (usage.get("completion_tokens", 0) * 0.00000060), 6))
            }
        )

    # Cache Miss -> Forward to Upstream LLM Provider
    near_misses = vector_store.find_near_misses(
        query_vector=prompt_vector,
        system_prompt_hash=sys_hash,
        params_hash=params_hash,
        threshold=active_threshold,
        lower_bound=0.80
    )
    best_near_miss_score = near_misses[0].similarity if near_misses else 0.0

    logger.info(f"Cache miss [near_miss={best_near_miss_score:.4f}] forwarding '{user_prompt[:40]}...'")

    provider = provider_router.get_provider(model)

    try:
        provider_response = await provider.generate_chat_completion(body)
    except Exception as e:
        logger.error(f"Provider LLM error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM Provider Error: {str(e)}")

    latency_s = time.time() - start_time
    latency_ms = round(latency_s * 1000, 2)

    ttl_seconds, ttl_tier = TTLClassifier.classify(user_prompt, settings.DEFAULT_TTL_SECONDS)

    if ttl_seconds > 0:
        new_entry = CacheEntry(
            vector=prompt_vector,
            prompt_text=user_prompt,
            system_prompt_hash=sys_hash,
            model=model,
            params_hash=params_hash,
            response=provider_response,
            ttl_seconds=ttl_seconds,
            expires_at=time.time() + ttl_seconds,
            tags=[ttl_tier, model]
        )
        vector_store.store_entry(new_entry)

    provider_name = provider.__class__.__name__.replace("Provider", "").lower()
    stats_tracker.record_miss(model=model, provider=provider_name, latency=latency_s, best_similarity=best_near_miss_score, prompt_text=user_prompt)


    if is_stream:
        return StreamingResponse(
            StreamProcessor.simulate_stream_cached(provider_response),
            media_type="text/event-stream",
            headers={
                "X-Cache-Hit": "false",
                "X-Cache-Similarity": str(best_near_miss_score),
                "X-Response-Time-Ms": str(latency_ms)
            }
        )

    return JSONResponse(
        content=provider_response,
        headers={
            "X-Cache-Hit": "false",
            "X-Cache-Similarity": str(best_near_miss_score),
            "X-Response-Time-Ms": str(latency_ms)
        }
    )
