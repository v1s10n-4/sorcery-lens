"""
sorcery-lens FastAPI application — pHash edition.

Design constraints:
  - No image data stored or logged
  - All config via env vars
  - Auth required on every endpoint
  - Rate limiting per API key
  - CV2 inference runs in thread pool (CPU-bound, never blocks event loop)
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
from app.identify import identify_image, _load_hash_db
from app.rate_limit import limiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sorcery-lens")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading hash DB...")
    # Blocking I/O — run in thread pool so the event loop stays free
    await asyncio.to_thread(_load_hash_db)
    logger.info("Hash DB ready.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="sorcery-lens",
    description="Private card recognition API for Sorcery: Contested Realm",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,  # No public schema — IP protection
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda req, exc: JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded"},
    ),
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Health (no auth — platform probes) ───────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Identify ──────────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB — phone photos can be 3-5 MB


@app.post("/identify")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def identify(
    request: Request,
    image: UploadFile = File(...),
    _key: str = Depends(require_api_key),
):
    """
    Identify a Sorcery: Contested Realm card from a photo.

    - **image**: JPEG, PNG, or WebP, max 10 MB (resized internally to ≤1000px)
    - Returns top-5 candidates sorted by pHash distance (lower = better match)
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image must be JPEG, PNG, or WebP",
        )

    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image exceeds 10 MB limit",
        )

    # CV2 inference is CPU-bound — run in thread pool, never block the event loop
    try:
        result = await asyncio.to_thread(identify_image, data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Identify error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recognition service temporarily unavailable",
        )

    return JSONResponse(content=result)
