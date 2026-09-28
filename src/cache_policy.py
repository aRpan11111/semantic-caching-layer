import re
from typing import Tuple

class TTLClassifier:
    """Classifies user prompts into TTL tiers based on content stability."""
    
    TEMPORAL_PATTERNS = [
        r"\btoday\b", r"\byesterday\b", r"\btomorrow\b", r"\bnow\b", r"\bcurrent(ly)?\b",
        r"\blatest\b", r"\brecent(ly)?\b", r"\bnews\b", r"\bweather\b", r"\bstock price\b",
        r"\bmarket price\b", r"\bscore\b", r"\blive\b", r"\b2026\b", r"\bthis week\b"
    ]
    
    FACTUAL_PATTERNS = [
        r"\bwhat is\b", r"\bexplain\b", r"\bdefine\b", r"\bhow to\b", r"\bcode for\b",
        r"\bwrite a (function|script|class)\b", r"\bdifference between\b", r"\bhistory of\b"
    ]

    @classmethod
    def classify(cls, prompt: str, default_ttl: int = 86400) -> Tuple[int, str]:
        """
        Returns (ttl_seconds, tier_name)
        Tiers:
          - 'bypass': 0s (no caching)
          - 'short': 3600s (1h)
          - 'medium': 43200s (12h)
          - 'long': 86400s (24h)
          - 'extended': 604800s (7 days)
        """
        lower_prompt = prompt.lower()
        
        # Check temporal indicators
        for pattern in cls.TEMPORAL_PATTERNS:
            if re.search(pattern, lower_prompt):
                return 3600, "short (temporal intent)"
                
        # Check evergreen/factual indicators
        for pattern in cls.FACTUAL_PATTERNS:
            if re.search(pattern, lower_prompt):
                return 604800, "extended (evergreen factual)"
                
        return default_ttl, "default"


class AdaptiveThreshold:
    """Adjusts similarity threshold based on prompt sensitivity & intent."""
    
    @classmethod
    def get_threshold(cls, prompt: str, base_threshold: float = 0.95) -> float:
        lower_prompt = prompt.lower()
        
        # Code generation and math require exact precision (0.98 threshold)
        if any(w in lower_prompt for w in ["code", "function", "def ", "class ", "equation", "formula", "sql", "regex"]):
            return max(base_threshold, 0.98)
            
        # Classification & summarization tolerate higher variance (0.90 threshold)
        if any(w in lower_prompt for w in ["classify", "sentiment", "summarize", "extract", "categorize"]):
            return min(base_threshold, 0.90)
            
        return base_threshold
