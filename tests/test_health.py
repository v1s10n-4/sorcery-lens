"""
Basic smoke tests — no CLIP/FAISS required.
"""
import os
import asyncio
import pytest

# Minimal env so config loads without crashing
os.environ.setdefault("API_KEYS", "test-key")
os.environ.setdefault("EMBEDDINGS_PATH", "data/embeddings.npz")
os.environ.setdefault("INDEX_PATH", "data/index.json")

from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import HTTPException
from app.main import app
from app import auth

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_identify_requires_auth():
    """Without a Bearer token, /identify must return 403."""
    resp = client.post("/identify", files={"image": ("test.jpg", b"\xff\xd8\xff", "image/jpeg")})
    assert resp.status_code in (401, 403)


def test_identify_bad_token():
    resp = client.post(
        "/identify",
        headers={"Authorization": "Bearer invalid-token"},
        files={"image": ("test.jpg", b"\xff\xd8\xff", "image/jpeg")},
    )
    assert resp.status_code == 401


def test_missing_api_keys_returns_503(monkeypatch):
    class DummySettings:
        api_key_set = set()

    monkeypatch.setattr(auth, "settings", DummySettings())

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-key")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.require_api_key(credentials))

    assert exc_info.value.status_code == 503
