import time
from collections import defaultdict

from fastapi import HTTPException

from app.config import settings

_WINDOW_SECONDS = 60
_hits: dict[int, list[float]] = defaultdict(list)


def enforce_rate_limit(user_id: int) -> None:
    """Simple in-memory fixed-window limiter, per user, for AI endpoints only.
    Sufficient for a single-process demo; a real deployment would use Redis.
    """
    now = time.time()
    window_start = now - _WINDOW_SECONDS
    _hits[user_id] = [t for t in _hits[user_id] if t > window_start]

    if len(_hits[user_id]) >= settings.ai_rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="AI rate limit exceeded, try again shortly")

    _hits[user_id].append(now)
