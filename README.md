# sorcery-lens

Private image recognition service for [Sorcery: Contested Realm](https://sorcery-tcg.com) TCG cards.

Consumed by **sorcery-companion** via an authenticated REST API.

---

## Architecture

```
Client (camera frame)
      │ POST /identify (Bearer token)
      ▼
FastAPI service
      │
      ├─ Auth middleware (per-key rate limiting)
      ├─ CLIP ViT-B/32 inference (open_clip)
      └─ FAISS cosine similarity search
              │
              └─ embeddings.npz + index.json
                   (generated offline, mounted at runtime)
```

## Stack

| Layer | Technology |
|-------|-----------|
| HTTP server | FastAPI + Uvicorn |
| Inference | open_clip (ViT-B/32, OpenAI weights) |
| Similarity search | faiss-cpu |
| Container | Docker (non-root, python:3.12-slim) |
| Deployment | Railway / Render |

---

## Quickstart (local dev)

```bash
# 1. Create venv + install deps
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure secrets (never hardcode)
cp .env.example .env
# Edit .env: set API_KEYS, COMPANION_DB_URL, etc.

# 3. Generate embeddings (requires DB access)
export COMPANION_DB_URL=postgresql://user:pass@host:5432/db
python scripts/generate_embeddings.py

# 4. Run the service
uvicorn app.main:app --reload
```

## Docker

```bash
docker compose up --build
```

## Embedding generation

```bash
# Full run
COMPANION_DB_URL=postgresql://... python scripts/generate_embeddings.py

# Dry run (metadata only, no CLIP)
COMPANION_DB_URL=postgresql://... python scripts/generate_embeddings.py --dry-run

# Limit to 50 cards (dev/testing)
COMPANION_DB_URL=postgresql://... python scripts/generate_embeddings.py --limit 50
```

Outputs: `data/embeddings.npz` + `data/index.json`

---

## API

### `POST /identify`

```
Authorization: Bearer {API_KEY}
Content-Type: multipart/form-data
Body: image (JPEG/PNG/WebP, max 2 MB)
```

**Response 200:**
```json
{
  "match": {
    "cardId": "clxyz...",
    "name": "Glacial Peak",
    "slug": "glacial-peak-alpha",
    "confidence": 0.92
  },
  "candidates": [
    { "cardId": "...", "name": "Glacial Peak", "slug": "glacial-peak-alpha", "confidence": 0.92 },
    { "cardId": "...", "name": "Frozen Tundra", "slug": "frozen-tundra-beta", "confidence": 0.71 },
    { "cardId": "...", "name": "Ice Cave", "slug": "ice-cave-alpha", "confidence": 0.68 }
  ],
  "latencyMs": 45
}
```

**Error codes:** `401` auth · `422` bad image · `429` rate limited · `503` service unavailable

---

## Security

- All secrets via environment variables — never in code
- No image data is stored or logged
- Embeddings are never exposed via any endpoint
- API keys are per-caller; rotate independently
- Docker: non-root user, minimal base image
- HTTPS only (reverse proxy handles TLS)

---

## License

Proprietary — Romain Daniel. All rights reserved.
