import hashlib
import json
from typing import Dict, Any, List, Optional

class CacheKeyGenerator:
    @staticmethod
    def extract_system_prompt(messages: List[Dict[str, Any]]) -> str:
        """Extract all system prompt contents from OpenAI formatted messages."""
        system_chunks = []
        for msg in messages:
            if msg.get("role") == "system":
                system_chunks.append(str(msg.get("content", "")))
        return "\n".join(system_chunks).strip()

    @staticmethod
    def extract_user_prompt(messages: List[Dict[str, Any]]) -> str:
        """Extract latest user prompt content from OpenAI formatted messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        # Fallback to concatenate non-system messages
        non_system = [str(m.get("content", "")) for m in messages if m.get("role") != "system"]
        return " ".join(non_system).strip()

    @classmethod
    def generate_system_hash(cls, system_prompt: str) -> str:
        """Generate SHA256 hash of system prompt."""
        if not system_prompt:
            return "default_system"
        return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def generate_params_hash(cls, model: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate parameter signature hash to isolate cache entries between different generation params."""
        param_dict = {
            "model": model.lower() if model else "default",
            "temperature": round(float(temperature), 2) if temperature is not None else 1.0,
            "max_tokens": max_tokens if max_tokens is not None else -1
        }
        encoded = json.dumps(param_dict, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def generate_cache_key(
        cls,
        user_prompt: str,
        system_prompt: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Create composite cache metadata key for lookup & invalidation scoping."""
        sys_hash = cls.generate_system_hash(system_prompt)
        param_hash = cls.generate_params_hash(model, temperature, max_tokens)
        prompt_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:16]
        return f"{sys_hash}:{param_hash}:{prompt_hash}"
