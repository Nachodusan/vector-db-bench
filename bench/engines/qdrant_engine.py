"""Qdrant, against a real server.

Not the embedded (`path=`) mode. qdrant-client's local implementation says so
itself: "Local mode performs exact (brute-force) search, so `search_params`
has no effect." Benchmarking HNSW through it measures brute force and prints
a flat recall of 1.000 at every ef_search.

The server is a single native binary — see scripts/qdrant.sh. No Docker.
"""

from __future__ import annotations

import os
import time

import numpy as np
from qdrant_client import QdrantClient, models

from bench.engines.base import EF_CONSTRUCTION, M, BuildStats

COLLECTION = "bench"


class QdrantEngine:
    name = "qdrant"
    knob = "hnsw_ef"

    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.getenv("BENCH_QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.url, timeout=120)
        try:
            self.client.get_collections()
        except Exception as exc:
            raise RuntimeError(
                f"no Qdrant server at {self.url}. Start one with scripts/qdrant.sh "
                f"(a single native binary, no Docker). Embedded mode is not an "
                f"option here: it is brute force by design."
            ) from exc

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        self.client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=vectors.shape[1],
                distance=models.Distance.DOT,      # vectors are unit length
            ),
            hnsw_config=models.HnswConfigDiff(
                m=M,
                ef_construct=EF_CONSTRUCTION,
                # Qdrant has *two* thresholds, and both have to be lowered.
                # `indexing_threshold` below decides whether an HNSW index is
                # built at all; `full_scan_threshold` (10,000 KB by default)
                # decides whether a segment is searched with it or scanned
                # exhaustively anyway. Leave this at its default on a small
                # collection and recall comes back a flat 1.000 at every
                # ef_search — an index that exists and is never used.
                full_scan_threshold=10,          # KB; the minimum Qdrant accepts
            ),
            # Qdrant will not build an HNSW index below `indexing_threshold`,
            # which defaults to 20,000 vectors — it quietly serves exact
            # results instead. A benchmark that does not set this measures
            # brute force and reports it as HNSW: recall comes out at a flat
            # 1.000 for every ef_search, which is the tell.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
        )
        self.client.upload_collection(
            collection_name=COLLECTION,
            vectors=vectors,
            ids=[int(i) for i in ids],
            batch_size=512,
        )
        self._wait_for_index(len(vectors))
        return BuildStats(time.perf_counter() - t0)

    def _wait_for_index(self, expected: int, timeout_s: float = 300.0) -> None:
        """Block until the HNSW index actually covers the vectors.

        Indexing is asynchronous. Querying before it finishes measures a
        half-built index, which is neither brute force nor HNSW.
        """
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            info = self.client.get_collection(COLLECTION)
            indexed = info.indexed_vectors_count or 0
            if str(info.status).endswith("green") and indexed >= expected:
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"qdrant indexed {indexed}/{expected} vectors before the timeout"
        )

    def search(self, queries: np.ndarray, k: int, ef_search: int) -> list[list[int]]:
        params = models.SearchParams(hnsw_ef=ef_search)
        out: list[list[int]] = []
        for q in queries:
            hits = self.client.query_points(
                collection_name=COLLECTION,
                query=q.tolist(),
                limit=k,
                search_params=params,
                with_payload=False,
            ).points
            out.append([int(h.id) for h in hits])
        return out

    def close(self) -> None:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.client.close()
