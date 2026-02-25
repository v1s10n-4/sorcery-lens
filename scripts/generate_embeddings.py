#!/usr/bin/env python3
"""
generate_embeddings.py — Offline embedding generation pipeline.

Fetches card variant metadata from the sorcery-companion Postgres DB,
downloads card images from Cloudflare R2, encodes with CLIP, and saves:
  - data/embeddings.npz   (float32 matrix, shape [N, 512])
  - data/index.json       (list of {cardId, name, slug})

Usage:
  COMPANION_DB_URL=postgresql://... python scripts/generate_embeddings.py [options]

Options:
  --model       CLIP model name       (default: ViT-B-32)
  --pretrained  CLIP pretrained name  (default: openai)
  --out-dir     Output directory      (default: data)
  --workers     Download concurrency  (default: 8)
  --limit       Max cards to process  (default: all)
  --dry-run     Print plan, no encode

Environment variables (required):
  DATABASE_URL  Postgres connection string (never hardcode)

Environment variables (optional):
  R2_PUBLIC_BASE    R2 CDN base URL (default: see DEFAULT_R2_BASE below)
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import open_clip
import psycopg2
import psycopg2.extras
import torch
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_R2_BASE = "https://pub-fbad7d695b084411b42bdff03adbffd5.r2.dev/cards"
DOWNLOAD_TIMEOUT_S = 30
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate-embeddings")


# ── DB query ──────────────────────────────────────────────────────────────────

def fetch_card_variants(db_url: str, limit: Optional[int] = None) -> list[dict]:
    """
    Query sorcery-companion Postgres DB for card variants.

    Returns list of {cardId, name, slug} sorted by card name then finish.
    Each variant has its own image on R2.
    """
    logger.info("Connecting to companion DB...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Actual Postgres table names: Prisma keeps PascalCase (must be quoted)
    query = """
        SELECT
            cv.slug     AS slug,
            c.id        AS card_id,
            c.name      AS name
        FROM "CardVariant" cv
        JOIN "Card" c ON cv."cardId" = c.id
        WHERE cv.slug IS NOT NULL
          AND cv.slug != ''
        ORDER BY c.name, cv.finish
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    cur.execute(query)
    rows = [
        {"cardId": r["card_id"], "name": r["name"], "slug": r["slug"]}
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()

    logger.info("Fetched %d card variants from DB", len(rows))
    return rows


# ── Image download ─────────────────────────────────────────────────────────────

def download_image(slug: str, r2_base: str, client: httpx.Client) -> Optional[bytes]:
    """
    Download a card image from Cloudflare R2 CDN.
    Returns raw bytes, or None on failure after retries.
    """
    url = f"{r2_base}/{slug}.png"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, timeout=DOWNLOAD_TIMEOUT_S)
            if resp.status_code == 200:
                return resp.content
            elif resp.status_code == 404:
                logger.warning("Image not found: %s", url)
                return None
            else:
                logger.warning(
                    "HTTP %d for %s (attempt %d/%d)",
                    resp.status_code, url, attempt, MAX_RETRIES,
                )
        except httpx.RequestError as exc:
            logger.warning("Request error for %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)

    return None


def download_batch(
    variants: list[dict],
    r2_base: str,
    workers: int,
) -> dict[str, bytes]:
    """
    Concurrently download all card images.
    Returns {slug: bytes} for successful downloads.
    """
    results: dict[str, bytes] = {}
    total = len(variants)

    logger.info("Downloading %d images (concurrency=%d)...", total, workers)

    with httpx.Client(follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_image, v["slug"], r2_base, client): v["slug"]
                for v in variants
            }
            done = 0
            for future in as_completed(futures):
                slug = futures[future]
                done += 1
                data = future.result()
                if data is not None:
                    results[slug] = data
                if done % 100 == 0 or done == total:
                    logger.info("  Downloaded %d/%d", done, total)

    failed = total - len(results)
    if failed:
        logger.warning("%d images failed to download and will be skipped", failed)

    return results


# ── CLIP inference ────────────────────────────────────────────────────────────

def build_embeddings(
    variants: list[dict],
    image_data: dict[str, bytes],
    model_name: str,
    pretrained: str,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run CLIP inference on all successfully downloaded images.

    Returns:
      embeddings  np.ndarray [N, D] float32
      index_meta  list of {cardId, name, slug} aligned to embeddings rows
    """
    logger.info("Loading CLIP model %s / %s...", model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()

    device = torch.device("cpu")  # faiss-cpu; no GPU dependency in prod
    model = model.to(device)

    embed_dim = model.visual.output_dim
    logger.info("Embedding dimension: %d", embed_dim)

    embeddings = []
    index_meta = []
    skipped = 0

    total = len(variants)
    for i, variant in enumerate(variants):
        slug = variant["slug"]
        raw = image_data.get(slug)
        if raw is None:
            skipped += 1
            continue

        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot decode image for slug=%s: %s", slug, exc)
            skipped += 1
            continue

        try:
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy().astype("float32")[0])
            index_meta.append(
                {
                    "cardId": variant["cardId"],
                    "name": variant["name"],
                    "slug": slug,
                }
            )
        except Exception as exc:
            logger.warning("Inference failed for slug=%s: %s", slug, exc)
            skipped += 1
            continue

        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info("  Encoded %d/%d (skipped=%d)", len(embeddings), total, skipped)

    logger.info(
        "Embedding complete: %d encoded, %d skipped", len(embeddings), skipped
    )
    return np.stack(embeddings, axis=0), index_meta


# ── Save artifacts ────────────────────────────────────────────────────────────

def save_artifacts(
    embeddings: np.ndarray,
    index_meta: list[dict],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npz"
    idx_path = out_dir / "index.json"

    np.savez_compressed(str(emb_path), embeddings=embeddings)
    with open(idx_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    size_mb = emb_path.stat().st_size / 1024 / 1024
    logger.info(
        "Saved %s (%.1f MB) and %s (%d entries)",
        emb_path, size_mb, idx_path, len(index_meta),
    )
    return emb_path, idx_path


# ── Verification ──────────────────────────────────────────────────────────────

def verify_artifacts(emb_path: Path, idx_path: Path):
    """Quick sanity check on saved artifacts."""
    data = np.load(str(emb_path))
    embs = data["embeddings"]
    with open(idx_path) as f:
        meta = json.load(f)

    assert embs.ndim == 2, f"Expected 2D array, got {embs.ndim}D"
    assert embs.shape[0] == len(meta), (
        f"Row count mismatch: embeddings={embs.shape[0]}, meta={len(meta)}"
    )
    assert embs.dtype == np.float32, f"Expected float32, got {embs.dtype}"

    # Spot-check norms (should be ≈1.0 after normalisation)
    norms = np.linalg.norm(embs[:10], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Embeddings not L2-normalised: {norms}"

    logger.info(
        "✓ Verification passed: shape=%s dtype=%s norms≈1.0",
        embs.shape, embs.dtype,
    )
    logger.info("  Sample entries:")
    for entry in meta[:3]:
        logger.info("    %s", entry)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    p.add_argument("--pretrained", default="openai", help="CLIP pretrained weights")
    p.add_argument("--out-dir", default="data", help="Output directory for artifacts")
    p.add_argument("--workers", type=int, default=8, help="Download concurrency")
    p.add_argument("--limit", type=int, default=None, help="Max variants to process (dev/testing)")
    p.add_argument("--r2-base", default=None, help="Override R2 CDN base URL")
    p.add_argument("--dry-run", action="store_true", help="Fetch metadata only, no inference")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Secrets from env — never from args or hardcoded ──────────────────────
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error(
            "DATABASE_URL environment variable is required.\n"
            "Set it in your .env file or export it before running this script.\n"
            "Example: export DATABASE_URL=postgresql://user:pass@host:5432/db"
        )
        sys.exit(1)

    r2_base = args.r2_base or os.environ.get("R2_PUBLIC_BASE", DEFAULT_R2_BASE)
    out_dir = Path(args.out_dir)

    t_start = time.monotonic()
    logger.info("=== sorcery-lens embedding generation ===")
    logger.info("Model: %s / %s", args.model, args.pretrained)
    logger.info("R2 base: %s", r2_base)
    logger.info("Output: %s", out_dir.resolve())

    # Step 1: Fetch card variants
    variants = fetch_card_variants(db_url, limit=args.limit)
    if not variants:
        logger.error("No card variants returned from DB. Check schema or query.")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — stopping after metadata fetch.")
        for v in variants[:5]:
            logger.info("  %s", v)
        logger.info("  ... and %d more", max(0, len(variants) - 5))
        return

    # Step 2: Download images
    image_data = download_batch(variants, r2_base, workers=args.workers)
    if not image_data:
        logger.error("No images downloaded. Check R2 connectivity.")
        sys.exit(1)

    # Step 3: CLIP inference
    embeddings, index_meta = build_embeddings(
        variants, image_data, args.model, args.pretrained
    )

    # Step 4: Save
    emb_path, idx_path = save_artifacts(embeddings, index_meta, out_dir)

    # Step 5: Verify
    verify_artifacts(emb_path, idx_path)

    elapsed = time.monotonic() - t_start
    logger.info("=== Done in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
