"""
Core pHash card identification pipeline.

Pipeline per request:
  1. Decode image bytes in memory (never touches disk)
  2. Detect card contour via multi-strategy OpenCV (CLAHE/Canny, adaptive, morphology)
  3. Perspective-correct to standard card size
  4. Try all 4 rotations (0°/90°/180°/270°) — handles any real-world orientation
  5. Crop art region (percentage-based, orientation-agnostic)
  6. Compute 256-bit pHash per rotation
  7. Match against pre-computed hash DB, minimum distance across rotations
  8. Fallback to center-crop if detection fails

No ML model. No GPU. No network calls at runtime.
IP protected — hash DB is never exposed via any endpoint.
"""

import io
import json
import logging
import time
from functools import lru_cache
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# ── Card geometry ──────────────────────────────────────────────────────────────

CARD_W = 380
CARD_H = 531

# Art region as fractions of the standard portrait card
ART_LEFT   = 0.05
ART_TOP    = 0.13
ART_RIGHT  = 0.95
ART_BOTTOM = 0.70

HASH_SIZE  = 16   # 256-bit hash (16×16)

# Confidence thresholds
DIST_CONFIDENT  = 70   # < 70 → confident match
DIST_LOW        = 90   # 70-90 → low confidence
                       # > 90 → no match


# ── Hash DB (loaded once at startup) ─────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_hash_db() -> list[dict]:
    """
    Load pre-computed pHash database from disk.
    Returns list of {slug, hash} sorted by slug for deterministic ordering.
    Cached — only runs once per process.
    """
    logger.info("Loading hash DB from %s", settings.hashes_path)
    with open(settings.hashes_path) as f:
        raw: dict[str, str] = json.load(f)

    # Load index to get cardId + name per slug
    logger.info("Loading index from %s", settings.index_path)
    with open(settings.index_path) as f:
        index: list[dict] = json.load(f)

    slug_to_meta = {e["slug"]: e for e in index}

    db = []
    for slug, hex_hash in raw.items():
        meta = slug_to_meta.get(slug, {})
        db.append({
            "slug":   slug,
            "cardId": meta.get("cardId", ""),
            "name":   meta.get("name", slug),
            "hash":   imagehash.hex_to_hash(hex_hash),
        })

    logger.info("Hash DB ready: %d entries", len(db))
    return db


# ── Image geometry helpers ────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _crop_art_cv(card_cv: np.ndarray) -> np.ndarray:
    h, w = card_cv.shape[:2]
    return card_cv[
        int(h * ART_TOP):int(h * ART_BOTTOM),
        int(w * ART_LEFT):int(w * ART_RIGHT),
    ]


def _crop_art_pil(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * ART_LEFT), int(h * ART_TOP),
                     int(w * ART_RIGHT), int(h * ART_BOTTOM)))


# ── Card detection ────────────────────────────────────────────────────────────

def _detect_card(img_cv: np.ndarray) -> Optional[np.ndarray]:
    """
    Multi-strategy contour detection. Returns 4 ordered corners or None.

    Priority:
      1. CLAHE + Canny (multiple thresholds)
      2. Bilateral + adaptive threshold
      3. Simple threshold + morphology
    """
    h, w = img_cv.shape[:2]
    scale = min(1.0, 1000 / max(h, w))
    small = cv2.resize(img_cv, None, fx=scale, fy=scale) if scale < 1 else img_cv.copy()
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    img_area = sh * sw
    best_cnt  = None
    best_area = 0

    def _check_contours(contours, method_tag):
        nonlocal best_cnt, best_area
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.08 or area > img_area * 0.75:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) != 4 or area <= best_area:
                    continue
                pts = approx.reshape(-1, 2)
                margin = max(sw, sh) * 0.03
                # Reject contours that span the entire image edge
                if (np.any(pts[:, 0] < margin) and np.any(pts[:, 0] > sw - margin) and
                        np.any(pts[:, 1] < margin) and np.any(pts[:, 1] > sh - margin)):
                    continue
                r = cv2.minAreaRect(cnt)
                rw, rh = r[1]
                if rw == 0 or rh == 0:
                    continue
                aspect = min(rw, rh) / max(rw, rh)
                if 0.55 < aspect < 0.85:
                    best_cnt  = approx
                    best_area = area

    # Strategy 1: CLAHE + Canny
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel   = np.ones((3, 3), np.uint8)
    for low, high in [(20, 80), (30, 100), (50, 150), (10, 50)]:
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges   = cv2.Canny(blurred, low, high)
        edges   = cv2.dilate(edges, kernel, iterations=2)
        edges   = cv2.erode(edges, kernel, iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _check_contours(cnts, f"canny({low},{high})")

    # Strategy 2: Bilateral + adaptive threshold
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    for bs in [11, 15, 21]:
        binary  = cv2.adaptiveThreshold(bilateral, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, bs, 2)
        edges   = cv2.Canny(binary, 30, 100)
        edges   = cv2.dilate(edges, kernel, iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _check_contours(cnts, f"adaptive(bs={bs})")

    # Strategy 3: Simple threshold + morphology
    kernel5 = np.ones((5, 5), np.uint8)
    for tv in [80, 100, 120, 140, 160]:
        _, binary = cv2.threshold(enhanced, tv, 255, cv2.THRESH_BINARY)
        binary    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)
        cnts, _   = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _check_contours(cnts, f"thresh({tv})")

    if best_cnt is None:
        return None

    pts = best_cnt.reshape(4, 2).astype("float32") / scale
    return _order_points(pts)


# ── Perspective correction ────────────────────────────────────────────────────

def _perspective_correct(img_cv: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Warp detected card to CARD_W × CARD_H. Always returns portrait orientation.
    """
    tl, tr, br, bl = corners
    avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    if avg_w > avg_h * 1.05:
        # Landscape — warp to landscape, then rotate portrait
        dst = np.array([[0, 0], [CARD_H - 1, 0],
                        [CARD_H - 1, CARD_W - 1], [0, CARD_W - 1]], dtype="float32")
        M   = cv2.getPerspectiveTransform(corners, dst)
        out = cv2.warpPerspective(img_cv, M, (CARD_H, CARD_W))
        return cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        dst = np.array([[0, 0], [CARD_W - 1, 0],
                        [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype="float32")
        M   = cv2.getPerspectiveTransform(corners, dst)
        return cv2.warpPerspective(img_cv, M, (CARD_W, CARD_H))


# ── Matching ──────────────────────────────────────────────────────────────────

def _rank_db(query_hashes: dict[str, "imagehash.ImageHash"],
             db: list[dict], top_k: int) -> list[dict]:
    """Rank all DB entries by minimum pHash distance across all query rotations."""
    results = []
    for entry in db:
        best_d   = 999
        best_rot = ""
        for rot, h in query_hashes.items():
            d = h - entry["hash"]
            if d < best_d:
                best_d   = d
                best_rot = rot
        confidence = round(max(0.0, 1.0 - best_d / 128), 4)
        results.append({
            "cardId":     entry["cardId"],
            "name":       entry["name"],
            "slug":       entry["slug"],
            "distance":   best_d,
            "confidence": confidence,
            "rotation":   best_rot,
        })
    results.sort(key=lambda x: x["distance"])
    return results[:top_k]


# ── Public interface ──────────────────────────────────────────────────────────

def identify_image(image_bytes: bytes, top_k: int = 5) -> dict:
    """
    Full pHash identification pipeline.

    Args:
        image_bytes: Raw JPEG/PNG/WebP bytes (never written to disk)
        top_k:       Number of candidates to return

    Returns:
        {
          "results": [...],     # top_k candidates, sorted by distance
          "method": "detected|fallback",
          "time_ms": int
        }

    Raises:
        ValueError: if image bytes are unreadable
        RuntimeError: if hash DB failed to load
    """
    t0  = time.monotonic()
    db  = _load_hash_db()

    # Decode image — stays in memory
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError("cv2 could not decode image")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    corners = _detect_card(img_cv)
    method  = "detected" if corners is not None else "fallback"

    if corners is not None:
        card = _perspective_correct(img_cv, corners)

        # 4 rotations — handles portrait, landscape, upside-down, sideways
        rotations = {
            "0deg":   card,
            "90deg":  cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE),
            "180deg": cv2.rotate(card, cv2.ROTATE_180),
            "270deg": cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        query_hashes = {}
        for rot, rotated in rotations.items():
            art     = _crop_art_cv(rotated)
            art_pil = Image.fromarray(cv2.cvtColor(art, cv2.COLOR_BGR2RGB))
            query_hashes[rot] = imagehash.phash(art_pil, hash_size=HASH_SIZE)

    else:
        # Fallback: center crop at 70%, try portrait + landscape
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot decode image for fallback: {exc}") from exc

        w, h   = img_pil.size
        nw, nh = int(w * 0.70), int(h * 0.70)
        crop   = img_pil.crop(((w - nw) // 2, (h - nh) // 2,
                                (w + nw) // 2, (h + nh) // 2))
        query_hashes = {
            "0deg":  imagehash.phash(_crop_art_pil(crop), hash_size=HASH_SIZE),
            "90deg": imagehash.phash(_crop_art_pil(crop.rotate(90, expand=True)),
                                     hash_size=HASH_SIZE),
        }

    results  = _rank_db(query_hashes, db, top_k)
    time_ms  = round((time.monotonic() - t0) * 1000)

    return {
        "results": results,
        "method":  method,
        "time_ms": time_ms,
    }
