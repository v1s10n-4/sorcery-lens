"""
Bearer token authentication.

Keys are stored as plaintext env vars during development.
In production, rotate to hashed keys (bcrypt) — swap the comparator here.
"""
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings

_bearer = HTTPBearer(auto_error=True)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> str:
    if not settings.api_key_set:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEYS not configured",
        )
    token = credentials.credentials
    if token not in settings.api_key_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
