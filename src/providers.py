import logging
import time
from typing import Dict, Any, List, Optional
import httpx
from src.config import settings

logger = logging.getLogger("semantic_cache.providers")

class BaseLLMProvider:
    async def generate_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class GroqProvider(BaseLLMProvider):
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        if self.api_key:
            try:
                from groq import AsyncGroq
                self.client = AsyncGroq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not load AsyncGroq SDK: {e}")
                self.client = None
        else:
            self.client = None

    async def generate_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        
        messages = request_data.get("messages", [])
        model = request_data.get("model", settings.GROQ_DEFAULT_MODEL)
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 1024)

        if self.client:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.model_dump()
            
        # HTTP client fallback
        async with httpx.AsyncClient() as http_client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            res = await http_client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30.0)
            res.raise_for_status()
            return res.json()


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        if self.api_key:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None

    async def generate_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is not configured.")
            
        messages = request_data.get("messages", [])
        model = request_data.get("model", settings.OPENAI_DEFAULT_MODEL)
        res = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=request_data.get("temperature", 0.7),
            max_tokens=request_data.get("max_tokens")
        )
        return res.model_dump()


class MockProvider(BaseLLMProvider):
    """Fallback high-performance synthetic provider for testing and offline execution."""
    async def generate_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = request_data.get("messages", [])
        model = request_data.get("model", "mock-llm-v1")
        
        last_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_msg = m.get("content", "")
                break
                
        prompt_len = len(last_msg.split())
        reply_content = f"Synthetic cached/uncached LLM response for query: '{last_msg}'. This reply demonstrates semantic caching proxy functionality with low latency."
        reply_tokens = len(reply_content.split())
        
        return {
            "id": f"chatcmpl-mock-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": max(5, prompt_len * 2),
                "completion_tokens": reply_tokens * 2,
                "total_tokens": max(5, prompt_len * 2) + (reply_tokens * 2)
            }
        }


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider for offline/local model testing (llama3, mistral, etc.)."""
    def __init__(self):
        self.base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434/v1")

    async def generate_chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = request_data.get("messages", [])
        model = request_data.get("model", "llama3")
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens")

        async with httpx.AsyncClient() as http_client:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": False
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            res = await http_client.post(f"{self.base_url}/chat/completions", json=payload, timeout=60.0)
            res.raise_for_status()
            return res.json()


class ProviderRouter:
    def __init__(self):
        self.groq = GroqProvider()
        self.openai = OpenAIProvider()
        self.ollama = OllamaProvider()
        self.mock = MockProvider()

    def get_provider(self, model_name: str) -> BaseLLMProvider:
        model_lower = model_name.lower()
        if "ollama" in model_lower or "mistral" in model_lower or "phi" in model_lower:
            return self.ollama
        if "llama" in model_lower or "mixtral" in model_lower or "gemma" in model_lower or "groq" in model_lower:
            if settings.GROQ_API_KEY:
                return self.groq
        elif ("gpt" in model_lower or "text-embedding" in model_lower) and settings.OPENAI_API_KEY:
            return self.openai
            
        # Default provider check
        if settings.DEFAULT_PROVIDER == "groq" and settings.GROQ_API_KEY:
            return self.groq
        elif settings.DEFAULT_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            return self.openai
        elif settings.DEFAULT_PROVIDER == "ollama":
            return self.ollama
            
        # Fallback to mock provider if no keys set or mock requested
        return self.mock

provider_router = ProviderRouter()

