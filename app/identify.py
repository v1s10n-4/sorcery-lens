"""
Core identification logic:
  1. Load CLIP model + FAISS index once at startup (cached on module)
  2. Preprocess uploaded image
  3. Embed with CLIP
  4. Cosine similarity search via FAISS
  5. Return top-k candidates

Nothing from this module is ever exposed publicly.
Image bytes are processed in memory and immediately discarded.
"""
import json
import time
import io
import logging
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import faiss
import open_clip
import torch
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

class CardEntry:
    __slots__ = ("card_id", "name", "slug")

    def __init__(self, card_id: str, name: str, slug: str):
        self.card_id = card_id
        self.name = name
        self.slug = slug


# ── Startup-time singletons ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model():
    """Load CLIP model once. Cached — not reloaded between requests."""
    logger.info("Loading CLIP model %s / %s", settings.clip_model, settings.clip_pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model,
        pretrained=settings.clip_pretrained,
    )
    model.eval()
    return model, preprocess


@lru_cache(maxsize=1)
def _load_index() -> Tuple[faiss.Index, List[CardEntry]]:
    """Load FAISS index + card metadata once at startup."""
    logger.info("Loading FAISS index from %s", settings.embeddings_path)
    data = np.load(settings.embeddings_path)
    embeddings: np.ndarray = data["embeddings"].astype("float32")

    # Normalise for cosine similarity (inner product on unit vectors)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    with open(settings.index_path, "r") as f:
        meta = json.load(f)

    cards = [
        CardEntry(
            card_id=m["cardId"],
            name=m["name"],
            slug=m["slug"],
        )
        for m in meta
    ]

    logger.info("Index ready: %d vectors, %d card entries", index.ntotal, len(cards))
    return index, cards


# ── Public interface ──────────────────────────────────────────────────────────

def _center_crop(img: Image.Image, ratio: float = 0.70) -> Image.Image:
    """Crop to the center portion of the image.

    Phone photos typically have lots of background around the card.
    Cropping to the center ~70% removes most of the noise and
    dramatically improves CLIP embedding quality.
    """
    w, h = img.size
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def identify_image(image_bytes: bytes, top_k: int = 3) -> dict:
    """
    Given raw image bytes, return the top-k card matches.
    Raises ValueError for unreadable images.
    Raises RuntimeError if the index is not loaded.
    """
    t0 = time.monotonic()

    # Decode image in memory — never touch disk
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    # Center-crop to remove background noise from phone photos
    img = _center_crop(img, ratio=0.70)

    model, preprocess = _load_model()
    index, cards = _load_index()

    # Preprocess + embed
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    query = features.cpu().numpy().astype("float32")

    # FAISS inner-product search (vectors are L2-normalised → cosine sim)
    distances, indices = index.search(query, top_k)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    candidates = []
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(cards):
            continue
        card = cards[idx]
        candidates.append(
            {
                "cardId": card.card_id,
                "name": card.name,
                "slug": card.slug,
                "confidence": round(float(dist), 4),
            }
        )

    latency_ms = round((time.monotonic() - t0) * 1000)

    return {
        "match": candidates[0] if candidates else None,
        "candidates": candidates,
        "latencyMs": latency_ms,
    }
