"""LLM Response Caching for RLM"""

import hashlib
from collections import OrderedDict
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CacheEntry:
    """A cached LLM response"""
    response: str
    model: str
    timestamp: datetime = field(default_factory=datetime.now)
    hit_count: int = 0


class LLMCache:
    """
    Simple LRU cache for LLM responses.

    Useful for repeated sub-calls with identical prompts
    during chunk processing.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _get_key(self, prompt: str, model: str) -> str:
        """Generate a cache key from prompt and model"""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, prompt: str, model: str) -> Optional[str]:
        """
        Get a cached response if available.

        Args:
            prompt: The prompt that was sent
            model: The model that was used

        Returns:
            Cached response or None if not found
        """
        key = self._get_key(prompt, model)

        if key in self.cache:
            self.hits += 1
            entry = self.cache[key]
            entry.hit_count += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return entry.response

        self.misses += 1
        return None

    def set(self, prompt: str, model: str, response: str):
        """
        Cache a response.

        Args:
            prompt: The prompt that was sent
            model: The model that was used
            response: The response to cache
        """
        key = self._get_key(prompt, model)

        # If key exists, update and move to end
        if key in self.cache:
            self.cache[key].response = response
            self.cache[key].timestamp = datetime.now()
            self.cache.move_to_end(key)
            return

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        # Add new entry
        self.cache[key] = CacheEntry(response=response, model=model)

    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def invalidate(self, prompt: str, model: str):
        """Invalidate a specific cache entry"""
        key = self._get_key(prompt, model)
        if key in self.cache:
            del self.cache[key]


# Global cache instance
_global_cache: Optional[LLMCache] = None


def get_cache(max_size: int = 1000) -> LLMCache:
    """Get or create the global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache(max_size)
    return _global_cache


def clear_cache():
    """Clear the global cache"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
