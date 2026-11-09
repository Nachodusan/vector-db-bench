"""Chroma, embedded.

One wrinkle worth knowing before you benchmark it: Chroma's `ef_search` lives
on the *collection*, not on the query. Sweeping it means mutating the
collection between sweeps rather than passing a parameter, which is handled
here so the sweep code does not have to care.
"""

from __future__ import annotations

import time

import chromadb
import numpy as np
from chromadb.config import Settings

from bench.engines.base import EF_CONSTRUCTION, M, BuildStats

COLLECTION = "bench"


class ChromaEngine:
    name = "chroma"
    knob = "hnsw:search_ef"

    def __init__(self) -> None:
        self.client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        self.collection = None
        self._current_ef: int | None = None

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> BuildStats:
        t0 = time.perf_counter()
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=COLLECTION,
            configuration={
                "hnsw": {
                    "space": "ip",                 # vectors are unit length
                    "max_neighbors": M,
                    "ef_construction": EF_CONSTRUCTION,
                }
            },
        )
        str_ids = [str(int(i)) for i in ids]
        step = 2000
        for i in range(0, len(vectors), step):
            self.collection.add(
                ids=str_ids[i:i + step],
                embeddings=vectors[i:i + step].tolist(),
            )
        return BuildStats(time.perf_counter() - t0)

    def _set_ef(self, ef_search: int) -> None:
        if self._current_ef == ef_search:
            return
        self.collection.modify(configuration={"hnsw": {"ef_search": ef_search}})
        # Re-fetch the handle. The change persists on the collection, but the
        # object in hand keeps serving the configuration it was created with,
        # which silently pins every sweep to the first value.
        self.collection = self.client.get_collection(COLLECTION)
        self._current_ef = ef_search
        # Chroma reports the new value faithfully. As of 1.5.9 it does not
        # act on it: results are byte-identical across ef_search 10, 40 and
        # 320 for every query. The sweep detects that for itself rather than
        # trusting this setter — see `EngineResult.knob_effective`.

    def search(self, queries: np.ndarray, k: int, ef_search: int) -> list[list[int]]:
        self._set_ef(ef_search)
        out: list[list[int]] = []
        for q in queries:
            res = self.collection.query(
                query_embeddings=[q.tolist()],
                n_results=k,
                include=[],
            )
            out.append([int(i) for i in res["ids"][0]])
        return out

    def close(self) -> None:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
