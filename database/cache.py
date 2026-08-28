"""
database/cache.py

Caches prediction results by input hash, so identical requests skip
recomputation. Redis is optional at runtime - if it's down, the API
should still work (just without caching), not crash. That fallback is a
deliberate design choice worth mentioning in an interview: caching is a
performance optimization, not a hard dependency for correctness.
"""
import hashlib
import json
import logging
import os

import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = 300  # 5 minutes

try:
    redis_client = redis.from_url(REDIS_URL, socket_connect_timeout=1)
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    logger.warning("Redis unavailable - predictions will not be cached.")
    redis_client = None
    REDIS_AVAILABLE = False


def _cache_key(payload: dict) -> str:
    """Deterministic hash of the input, so identical requests always map
    to the same cache key regardless of dict key ordering."""
    serialized = json.dumps(payload, sort_keys=True)
    return "prediction:" + hashlib.sha256(serialized.encode()).hexdigest()


def get_cached_prediction(payload: dict) -> dict | None:
    if not REDIS_AVAILABLE:
        return None
    try:
        cached = redis_client.get(_cache_key(payload))
        if cached:
            logger.info("Cache hit")
            return json.loads(cached)
        logger.info("Cache miss")
        return None
    except Exception:
        logger.warning("Redis read failed, falling back to live prediction.")
        return None


def set_cached_prediction(payload: dict, result: dict) -> None:
    if not REDIS_AVAILABLE:
        return
    try:
        redis_client.setex(_cache_key(payload), CACHE_TTL_SECONDS, json.dumps(result))
    except Exception:
        logger.warning("Redis write failed, continuing without caching.")
