from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from typing import Dict, Any

# Prometheus Metrics Definitions

REQUEST_COUNTER = Counter(
    "llm_cache_requests_total",
    "Total LLM proxy requests received",
    ["status", "model", "provider"]  # status: hit | miss
)

LATENCY_HISTOGRAM = Histogram(
    "llm_cache_latency_seconds",
    "Request latency in seconds",
    ["status", "model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

COST_SAVED_COUNTER = Counter(
    "llm_cache_cost_saved_usd_total",
    "Estimated total LLM API cost saved in USD",
    ["model"]
)

TOKENS_SAVED_COUNTER = Counter(
    "llm_cache_tokens_saved_total",
    "Total token count served directly from semantic cache",
    ["type", "model"]  # type: prompt | completion
)

SIMILARITY_HISTOGRAM = Histogram(
    "llm_cache_similarity_scores",
    "Distribution of similarity search scores",
    ["outcome"],  # outcome: hit | near_miss | miss
    buckets=[0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0]
)

CACHE_SIZE_GAUGE = Gauge(
    "llm_cache_entries_total",
    "Current total active cache entries stored"
)

# In-memory tracking for statistics reporting endpoints
class CacheStatsTracker:
    def __init__(self):
        self.total_requests = 0
        self.hits = 0
        self.misses = 0
        self.total_cost_saved_usd = 0.0
        self.total_prompt_tokens_saved = 0
        self.total_completion_tokens_saved = 0
        self.hit_latencies = []
        self.miss_latencies = []
        self.near_miss_log = []

    def record_hit(self, model: str, latency: float, similarity: float, usage: Dict[str, Any]):
        self.total_requests += 1
        self.hits += 1
        self.hit_latencies.append(latency)
        
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        self.total_prompt_tokens_saved += prompt_tokens
        self.total_completion_tokens_saved += completion_tokens
        
        # Calculate cost savings (approximate benchmark: $0.15/1M prompt, $0.60/1M completion)
        cost_saved = (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060)
        self.total_cost_saved_usd += cost_saved

        # Update Prometheus
        REQUEST_COUNTER.labels(status="hit", model=model, provider="cache").inc()
        LATENCY_HISTOGRAM.labels(status="hit", model=model).observe(latency)
        COST_SAVED_COUNTER.labels(model=model).inc(cost_saved)
        TOKENS_SAVED_COUNTER.labels(type="prompt", model=model).inc(prompt_tokens)
        TOKENS_SAVED_COUNTER.labels(type="completion", model=model).inc(completion_tokens)
        SIMILARITY_HISTOGRAM.labels(outcome="hit").observe(similarity)

    def record_miss(self, model: str, provider: str, latency: float, best_similarity: float, prompt_text: str = ""):
        self.total_requests += 1
        self.misses += 1
        self.miss_latencies.append(latency)
        
        REQUEST_COUNTER.labels(status="miss", model=model, provider=provider).inc()
        LATENCY_HISTOGRAM.labels(status="miss", model=model).observe(latency)
        
        outcome = "near_miss" if best_similarity >= 0.80 else "miss"
        SIMILARITY_HISTOGRAM.labels(outcome=outcome).observe(best_similarity)

        if best_similarity >= 0.80:
            self.near_miss_log.append({
                "timestamp": time.time(),
                "prompt": prompt_text[:100],
                "model": model,
                "similarity": round(best_similarity, 4),
                "suggestion": "Candidate for threshold tuning or prompt normalization"
            })
            # Keep log bounded
            if len(self.near_miss_log) > 200:
                self.near_miss_log.pop(0)

    def get_summary(self) -> Dict[str, Any]:
        hit_rate = (self.hits / self.total_requests * 100) if self.total_requests > 0 else 0.0
        avg_hit_lat = (sum(self.hit_latencies) / len(self.hit_latencies) * 1000) if self.hit_latencies else 0.0
        avg_miss_lat = (sum(self.miss_latencies) / len(self.miss_latencies) * 1000) if self.miss_latencies else 0.0

        return {
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "total_cost_saved_usd": round(self.total_cost_saved_usd, 4),
            "total_prompt_tokens_saved": self.total_prompt_tokens_saved,
            "total_completion_tokens_saved": self.total_completion_tokens_saved,
            "total_tokens_saved": self.total_prompt_tokens_saved + self.total_completion_tokens_saved,
            "near_miss_count": len(self.near_miss_log),
            "avg_hit_latency_ms": round(avg_hit_lat, 2),
            "avg_miss_latency_ms": round(avg_miss_lat, 2),
            "latency_reduction_factor": round(avg_miss_lat / avg_hit_lat, 1) if avg_hit_lat > 0 else 0.0
        }

stats_tracker = CacheStatsTracker()

