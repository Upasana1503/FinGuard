"""
Simple in-memory sliding-window rate limiter, keyed by API key id.

Deliberately not DB-backed: fine for a single-process portfolio deployment,
resets on restart, and won't stay correct across multiple worker processes
or instances. If this ever needs to scale past one process, move the
counters to Redis -- noted here rather than silently pretending this scales.
"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException

_WINDOW_SECONDS = 60
_MAX_REQUESTS_PER_WINDOW = 30

_request_log: dict[str, deque] = defaultdict(deque)


def check_rate_limit(api_key_id: str) -> None:
    now = time.time()
    log = _request_log[api_key_id]

    while log and now - log[0] > _WINDOW_SECONDS:
        log.popleft()

    if len(log) >= _MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {_MAX_REQUESTS_PER_WINDOW} requests per {_WINDOW_SECONDS}s",
        )

    log.append(now)
