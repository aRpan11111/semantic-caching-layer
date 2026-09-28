import json
import time
from typing import AsyncGenerator, Dict, Any, List
import logging

logger = logging.getLogger("semantic_cache.streaming")

class StreamProcessor:
    """Manages SSE streaming pass-through and token chunk accumulation for cache insertion."""
    
    @staticmethod
    def format_sse_chunk(chunk_dict: Dict[str, Any]) -> str:
        return f"data: {json.dumps(chunk_dict)}\n\n"

    @classmethod
    async def simulate_stream_cached(cls, cached_response: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Convert a cached full response object into rapid SSE stream chunks for streaming clients."""
        choices = cached_response.get("choices", [])
        if not choices:
            yield "data: [DONE]\n\n"
            return
            
        full_text = choices[0].get("message", {}).get("content", "")
        words = full_text.split(" ")
        
        chunk_id = cached_response.get("id", f"chatcmpl-stream-{int(time.time())}")
        model = cached_response.get("model", "cached-model")

        for i, word in enumerate(words):
            piece = word + (" " if i < len(words) - 1 else "")
            chunk_obj = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }
                ]
            }
            yield cls.format_sse_chunk(chunk_obj)
        yield "data: [DONE]\n\n"
