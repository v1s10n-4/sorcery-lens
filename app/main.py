"""
sorcery-lens FastAPI application.

Design constraints:
  - No image data is ever stored or logged
  - All config comes from env vars
  - Auth is required on every endpoint
  - Rate limiting is per API key
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.auth import require_api_key
from app.config import settings
from app.identify import identify_image, _load_model, _load_index
from app.rate_limit import limiter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sorcery-lens")


# ── Lifespan: pre-warm model + index ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-warming CLIP model and FAISS index...")
    # Both calls are blocking/CPU-bound — run in thread pool so the event loop
    # stays free. Model load is fatal; index load logs but doesn't crash
    # (allows local dev without embeddings mounted).
    await asyncio.to_thread(_load_model)
    logger.info("CLIP model ready.")
    try:
        await asyncio.to_thread(_load_index)
        logger.info("FAISS index ready.")
    except Exception as exc:
        logger.error("Index load failed (embeddings not mounted?): %s", exc)
    yield
    # shutdown — nothing to clean up (FAISS index is in-memory)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="sorcery-lens",
    description="Private card recognition API for Sorcery: Contested Realm",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,   # No public Swagger UI
    redoc_url=None,  # No public ReDoc
    openapi_url=None,  # No public schema endpoint — IP protection
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    content={"detail": "Rate limit exceeded"},
))
app.add_middleware(SlowAPIMiddleware)

# CORS — tightly scoped to companion origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Health check (no auth — used by platform health probes) ──────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Core endpoint ─────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB

@app.post("/identify")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def identify(
    request: Request,
    image: UploadFile = File(...),
    _key: str = Depends(require_api_key),
):
    """
    Identify a Sorcery card from an image.

    - **image**: JPEG or PNG, max 2 MB
    - Returns top-3 candidates with confidence scores
    """
    # Validate content type
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image must be JPEG, PNG, or WebP",
        )

    # Read bytes — enforced size limit
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image exceeds maximum size of {MAX_IMAGE_BYTES // 1024 // 1024} MB",
        )

    # Process — image bytes stay in memory, never written anywhere.
    # identify_image is CPU-bound (CLIP inference); run in thread pool to avoid
    # blocking the event loop and crashing the anyio TaskGroup.
    try:
        result = await asyncio.to_thread(identify_image, data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recognition service temporarily unavailable",
        )

    return JSONResponse(content=result)
