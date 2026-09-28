from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.vector_store import vector_store
from src.metrics import stats_tracker
from src.config import settings

router = APIRouter(prefix="/v1/cache", tags=["Cache Management"])

class InvalidationRequest(BaseModel):
    system_prompt_hash: Optional[str] = None
    model: Optional[str] = None
    tag: Optional[str] = None

class ThresholdTuneRequest(BaseModel):
    test_queries: List[str]
    target_answers: List[str]
    thresholds_to_test: List[float] = [0.85, 0.90, 0.93, 0.95, 0.97, 0.98]

@router.post("/invalidate")
async def invalidate_cache(req: InvalidationRequest):
    """Invalidate cache entries by system prompt hash, model ID, or custom tag."""
    invalidated_count = 0
    if req.system_prompt_hash:
        invalidated_count += vector_store.invalidate_by_system_prompt(req.system_prompt_hash)
    if req.model:
        invalidated_count += vector_store.invalidate_by_model(req.model)
    if req.tag:
        invalidated_count += vector_store.invalidate_by_tag(req.tag)
        
    return {
        "status": "success",
        "invalidated_count": invalidated_count,
        "filters": req.dict(exclude_none=True)
    }

@router.get("/stats")
async def get_cache_stats():
    """Retrieve operational metrics, hit rates, cost savings, and latency reduction."""
    return stats_tracker.get_summary()

@router.delete("/clear")
async def clear_cache():
    """Flushes all stored entries from vector cache."""
    cleared = vector_store.clear_all()
    return {"status": "success", "cleared_entries": cleared}

@router.post("/tune-threshold")
async def tune_threshold(req: ThresholdTuneRequest):
    """
    Simulates different similarity thresholds against test data to visualize 
    tradeoff between cache hit rate and precision.
    """
    from src.embeddings import embedding_engine
    
    if len(req.test_queries) != len(req.target_answers):
        raise HTTPException(status_code=400, detail="test_queries and target_answers must have equal length.")

    results = []
    # Seed embeddings
    query_vecs = embedding_engine.embed_batch(req.test_queries)
    
    for thresh in req.thresholds_to_test:
        hits = 0
        near_hits = 0
        misses = 0
        
        for vec in query_vecs:
            search_res = vector_store.search_nearest(vec, "default_system", "default_params", threshold=thresh)
            if search_res:
                hits += 1
            else:
                misses += 1
                
        hit_rate = (hits / len(query_vecs)) * 100 if query_vecs else 0
        results.append({
            "threshold": thresh,
            "hit_rate_pct": round(hit_rate, 2),
            "hits": hits,
            "misses": misses,
            "tradeoff_notes": "High accuracy, lower hit rate" if thresh >= 0.96 else ("Balanced hit rate & precision" if thresh >= 0.92 else "Higher hit rate, potential semantic drift")
        })

    return {
        "dataset_size": len(req.test_queries),
        "current_active_threshold": settings.SIMILARITY_THRESHOLD,
        "simulation_results": results
    }

@router.get("/near-misses")
async def get_near_misses():
    """
    Retrieve queries that fell just below similarity threshold (0.80 - 0.94).
    Useful for identifying threshold tuning and prompt normalization candidates.
    """
    return {
        "near_miss_count": len(stats_tracker.near_miss_log),
        "near_misses": stats_tracker.near_miss_log
    }

