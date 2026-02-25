"""
Basic smoke tests — no CLIP/FAISS required.
"""
import os
import pytest

# Minimal env so config loads without crashing
os.environ.setdefault("API_KEYS", "test-key")
os.environ.setdefault("EMBEDDINGS_PATH", "data/embeddings.npz")
os.environ.setdefault("INDEX_PATH", "data/index.json")

from fastapi.testclient import TestClient
from app.main import app

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
