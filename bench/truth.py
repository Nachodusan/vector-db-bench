"""Exact nearest neighbours, by brute force.

Recall has to be measured against the true answer, which means exact search
over the same vectors — not against another approximate index, and not against
human relevance labels.

Those are different questions. Whether the *right passage* comes back is a
property of the embedding model. Whether the index returns what exact search
would have returned is a property of the index. Only the second one is a fair
thing to compare databases on, and it is the only thing measured here.

At this corpus size brute force is a single matrix multiply, so the ground
truth is cheap and exact rather than sampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUERY_COUNT = 500
SEED = 20261109


@dataclass(frozen=True)
class Split:
    index_vectors: np.ndarray     # (n, d) what goes into each database
    index_ids: np.ndarray         # (n,)   stable ids, shared across engines
    queries: np.ndarray           # (q, d) held out, never indexed
    truth: np.ndarray             # (q, k) exact top-k index_ids per query

    @property
    def k(self) -> int:
        return self.truth.shape[1]


def make_split(vectors: np.ndarray, *, k: int = 10,
               query_count: int = QUERY_COUNT, seed: int = SEED) -> Split:
    """Hold out queries, then compute exact top-k over everything else.

    Queries are held-out corpus vectors, which is what ann-benchmarks does and
    what keeps the query distribution identical to the indexed distribution.
    Querying with vectors that are *in* the index would make top-1 trivially
    itself and hide exactly the differences being measured.
    """
    rng = np.random.default_rng(seed)
    n = len(vectors)
    query_idx = rng.choice(n, size=query_count, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[query_idx] = False

    index_vectors = np.ascontiguousarray(vectors[mask])
    index_ids = np.flatnonzero(mask).astype(np.int64)
    queries = np.ascontiguousarray(vectors[query_idx])

    # Vectors are unit length, so inner product is cosine similarity.
    sims = queries @ index_vectors.T                       # (q, n)
    top = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    ordered = np.take_along_axis(
        top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1
    )
    truth = index_ids[ordered]

    return Split(index_vectors, index_ids, queries, truth)


def recall_at_k(truth: np.ndarray, got: list[list[int]]) -> float:
    """Mean fraction of the true top-k that the engine actually returned.

    Order within the k is not scored. An index that returns the right ten in
    the wrong order has found them; ranking them is the scorer's job upstream.
    """
    if len(got) != len(truth):
        raise ValueError(f"expected {len(truth)} result rows, got {len(got)}")
    k = truth.shape[1]
    total = 0.0
    for expected, actual in zip(truth, got):
        total += len(set(expected.tolist()) & set(actual)) / k
    return total / len(truth)


if __name__ == "__main__":
    from dotenv import load_dotenv

    from bench.corpus import load
    from bench.embed import embed

    load_dotenv()
    split = make_split(embed(load(), progress=False))
    print(f"index   {split.index_vectors.shape}")
    print(f"queries {split.queries.shape}")
    print(f"truth   {split.truth.shape} (top-{split.k})")
    print(f"perfect recall sanity: {recall_at_k(split.truth, [list(r) for r in split.truth]):.3f}")
    print(f"empty   recall sanity: {recall_at_k(split.truth, [[] for _ in split.truth]):.3f}")
