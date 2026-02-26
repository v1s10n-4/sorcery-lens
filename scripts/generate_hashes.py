#!/usr/bin/env python3
"""
generate_hashes.py — Offline pHash generation pipeline.

Downloads card variant images from Cloudflare R2, crops the art region,
computes a 256-bit perceptual hash, and saves:
  - data/hashes.json   ({slug: hex_hash_string}, ~200 KB)

Usage:
  python scripts/generate_hashes.py [options]

Options:
  --index     Path to index.json     (default: data/index.json)
  --out       Output path            (default: data/hashes.json)
  --workers   Download concurrency   (default: 12)
  --limit     Max slugs to process   (dev/testing)
  --hash-size pHash size             (default: 16 → 256-bit)
  --dry-run   Print plan, no hashing

No secrets required — all card images are on a public CDN.
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
import imagehash
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate-hashes")

R2_BASE = "https://pub-fbad7d695b084411b42bdff03adbffd5.r2.dev/cards"
USER_AGENT = "sorcery-lens/1.0 (hash-gen)"
DOWNLOAD_TIMEOUT = 20
MAX_RETRIES = 3

# Art region — percentage coordinates on a standard portrait card
ART_LEFT   = 0.05
ART_TOP    = 0.13
ART_RIGHT  = 0.95
ART_BOTTOM = 0.70


def crop_art_region(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((
        int(w * ART_LEFT),
        int(h * ART_TOP),
        int(w * ART_RIGHT),
        int(h * ART_BOTTOM),
    ))


def download_image(slug: str, client: httpx.Client) -> Optional[bytes]:
    url = f"{R2_BASE}/{slug}.png"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = client.get(url, timeout=DOWNLOAD_TIMEOUT)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                logger.warning("404: %s", url)
                return None
            logger.warning("HTTP %d for %s (attempt %d)", r.status_code, url, attempt)
        except httpx.RequestError as exc:
            logger.warning("RequestError for %s (attempt %d): %s", url, attempt, exc)
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
    return None


def hash_slug(slug: str, client: httpx.Client, hash_size: int) -> Optional[str]:
    raw = download_image(slug, client)
    if raw is None:
        return None
    try:
        img = Image.open(__import__("io").BytesIO(raw)).convert("RGB")
        art = crop_art_region(img)
        h = imagehash.phash(art, hash_size=hash_size)
        return str(h)
    except Exception as exc:
        logger.warning("Hash failed for %s: %s", slug, exc)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index",     default="data/index.json",  help="Path to index.json")
    ap.add_argument("--out",       default="data/hashes.json", help="Output path")
    ap.add_argument("--workers",   type=int, default=12,        help="Download concurrency")
    ap.add_argument("--limit",     type=int, default=None,      help="Max slugs (dev)")
    ap.add_argument("--hash-size", type=int, default=16,        help="pHash size (bits=size²)")
    ap.add_argument("--dry-run",   action="store_true",         help="Plan only, no hashing")
    args = ap.parse_args()

    with open(args.index) as f:
        index = json.load(f)

    slugs = [e["slug"] for e in index if e.get("slug")]
    if args.limit:
        slugs = slugs[: args.limit]

    logger.info("=== sorcery-lens hash generation ===")
    logger.info("Slugs: %d | Workers: %d | hash_size: %d | out: %s",
                len(slugs), args.workers, args.hash_size, args.out)

    if args.dry_run:
        logger.info("DRY RUN — first 5 slugs: %s", slugs[:5])
        return

    t0 = time.monotonic()
    hashes: dict[str, str] = {}
    failed = 0

    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(hash_slug, slug, client, args.hash_size): slug
                       for slug in slugs}
            done = 0
            for future in as_completed(futures):
                slug = futures[future]
                done += 1
                result = future.result()
                if result is not None:
                    hashes[slug] = result
                else:
                    failed += 1
                if done % 200 == 0 or done == len(slugs):
                    logger.info("  %d/%d (failed=%d)", done, len(slugs), failed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(hashes, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    elapsed = time.monotonic() - t0
    logger.info("Saved %s (%.1f KB, %d entries) in %.1fs", out_path, size_kb, len(hashes), elapsed)

    if failed:
        logger.warning("%d slugs failed — rerun to retry", failed)
    if failed > len(slugs) * 0.05:
        logger.error("Too many failures (>5%%) — something is wrong")
        sys.exit(1)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
