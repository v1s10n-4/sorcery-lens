"""
Per-key rate limiting via slowapi (wraps limits library).
Default: 60 requests/minute per API key.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request


def _key_func(request: Request) -> str:
    """Use the Bearer token as the rate-limit bucket key."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func)
