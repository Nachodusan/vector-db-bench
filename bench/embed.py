"""Embed the corpus once, cache it, never think about it again.

The vectors are the one thing every engine must share. If two engines see even
slightly different inputs the comparison is void, so they are computed once,
written to a single `.npy`, and keyed by the corpus fingerprint.

Vectors are L2-normalised on the way out. Cosine similarity and inner product
then coincide, which lets every engine be configured with the same metric
rather than each with its own near-equivalent.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

from bench.corpus import CACHE, Corpus

MODEL = os.getenv("BENCH_EMBED_MODEL", "BAAI/bge-m3")
BATCH = 128
WORKERS = 6


def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Point it at any OpenAI-compatible "
            "embeddings endpoint (OPENAI_BASE_URL), or pass --env."
        )
    return OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    for attempt in range(4):
        try:
            resp = client.embeddings.create(model=MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def embed(corpus: Corpus, *, progress: bool = True) -> np.ndarray:
    """Return an (n, d) float32 array of unit vectors, cached on disk."""
    path = CACHE / f"vectors-{corpus.fingerprint}-{MODEL.replace('/', '_')}.npy"
    if path.exists():
        return np.load(path)

    client = _client()
    batches = [corpus.chunks[i:i + BATCH] for i in range(0, len(corpus), BATCH)]
    done = 0
    out: list[list[list[float]]] = [None] * len(batches)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_embed_batch, client, b): i for i, b in enumerate(batches)}
        for future, i in futures.items():
            out[i] = future.result()
            done += 1
            if progress and done % 10 == 0:
                print(f"  {done}/{len(batches)} batches", end="\r", flush=True)

    vectors = np.asarray([v for batch in out for v in batch], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    if progress:
        print(f"  {len(batches)}/{len(batches)} batches · saved {path.name}")
    return vectors


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    from bench.corpus import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="Path to a .env holding the API key.")
    ap.add_argument("--limit", type=int, default=60_000)
    args = ap.parse_args()
    load_dotenv(args.env) if args.env else load_dotenv()

    corpus = load(args.limit)
    print(f"embedding {len(corpus):,} chunks with {MODEL}")
    t0 = time.perf_counter()
    vectors = embed(corpus)
    print(f"  shape {vectors.shape} · {time.perf_counter() - t0:.1f}s")
    print(f"  unit norm: {np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)}")
