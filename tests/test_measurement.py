"""Tests for the measuring apparatus.

An instrument that can be wrong about the thing it measures needs its own
tests more than the thing does.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.sweep import EngineResult, Point
from bench.truth import make_split, recall_at_k


# --- recall ---------------------------------------------------------------

def test_perfect_and_empty_recall():
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(truth, [[1, 2, 3], [4, 5, 6]]) == 1.0
    assert recall_at_k(truth, [[], []]) == 0.0


def test_recall_ignores_order_within_k():
    """Returning the right ten in the wrong order is finding them. Ranking
    them is the scorer's job upstream, not the index's."""
    truth = np.array([[1, 2, 3]])
    assert recall_at_k(truth, [[3, 1, 2]]) == 1.0


def test_partial_recall_is_a_fraction_of_k():
    truth = np.array([[1, 2, 3, 4]])
    assert recall_at_k(truth, [[1, 2, 99, 98]]) == 0.5


def test_recall_rejects_a_mismatched_row_count():
    with pytest.raises(ValueError):
        recall_at_k(np.array([[1], [2]]), [[1]])


# --- split ----------------------------------------------------------------

@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(7)
    v = rng.normal(size=(400, 32)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_queries_are_held_out_of_the_index(vectors):
    split = make_split(vectors, k=5, query_count=40)
    assert len(split.index_vectors) == len(vectors) - 40
    assert len(split.queries) == 40
    # No query vector may also sit in the index, or top-1 is trivially itself.
    index_rows = {v.tobytes() for v in split.index_vectors}
    assert not any(q.tobytes() in index_rows for q in split.queries)


def test_truth_ids_refer_to_indexed_vectors(vectors):
    split = make_split(vectors, k=5, query_count=40)
    assert split.truth.shape == (40, 5)
    assert set(split.truth.flatten().tolist()) <= set(split.index_ids.tolist())


def test_truth_really_is_the_exact_nearest_neighbours(vectors):
    """Recompute independently and compare. If this drifts, every recall
    number in the repository is measured against the wrong answer."""
    split = make_split(vectors, k=5, query_count=20)
    sims = split.queries @ split.index_vectors.T
    expected = split.index_ids[np.argsort(-sims, axis=1)[:, :5]]
    assert np.array_equal(expected, split.truth)


def test_split_is_deterministic(vectors):
    a = make_split(vectors, k=5, query_count=40)
    b = make_split(vectors, k=5, query_count=40)
    assert np.array_equal(a.truth, b.truth)
    assert np.array_equal(a.queries, b.queries)


# --- latency at matched recall -------------------------------------------

def result_with(points: list[tuple[int, float, float]]) -> EngineResult:
    r = EngineResult("e", "ef", 1.0, "", 100, 8)
    r.points = [Point(ef, rec, p95, p95, p95, 1000 / p95) for ef, rec, p95 in points]
    return r


def test_latency_at_recall_interpolates_between_points():
    r = result_with([(10, 0.90, 1.0), (20, 1.00, 2.0)])
    assert r.latency_at_recall(0.95) == pytest.approx(1.5)


def test_latency_at_recall_hits_an_exact_point():
    r = result_with([(10, 0.90, 1.0), (20, 0.95, 1.7), (40, 0.99, 3.0)])
    assert r.latency_at_recall(0.95) == pytest.approx(1.7)


def test_unreachable_recall_returns_none_rather_than_extrapolating():
    """An engine that never reaches the target has not reached it. Guessing
    where its curve would have gone invents a number."""
    r = result_with([(10, 0.80, 1.0), (20, 0.88, 2.0)])
    assert r.latency_at_recall(0.95) is None


def test_target_below_the_whole_curve_uses_the_cheapest_point():
    r = result_with([(10, 0.97, 1.0), (20, 0.99, 2.0)])
    assert r.latency_at_recall(0.90) == pytest.approx(1.0)


# --- the inert-knob guard -------------------------------------------------

def test_a_flat_recall_curve_means_the_knob_did_nothing():
    """Qdrant below its full_scan_threshold and Chroma 1.5.9's ef_search both
    produce this: six measurements of one setting, wearing the shape of a
    finished sweep."""
    inert = result_with([(10, 0.976, 1.4), (40, 0.976, 1.4), (320, 0.976, 1.4)])
    assert not inert.knob_effective


def test_a_flat_perfect_curve_is_also_inert():
    """The brute-force tell: recall pinned at 1.000 whatever you ask for."""
    exact = result_with([(10, 1.0, 1.3), (80, 1.0, 1.3), (320, 1.0, 1.2)])
    assert not exact.knob_effective


def test_a_real_curve_is_effective():
    real = result_with([(10, 0.70, 1.2), (80, 0.95, 2.4), (320, 1.00, 4.7)])
    assert real.knob_effective


def test_an_inert_knob_has_no_matched_recall_latency():
    """Refusing to report a number is the point. An engine that cannot be
    moved along the curve cannot be compared at a point on it."""
    inert = result_with([(10, 0.976, 1.4), (320, 0.976, 1.4)])
    assert inert.latency_at_recall(0.90) is None
    assert inert.latency_at_recall(0.95) is None


def test_a_curve_where_latency_falls_as_recall_rises_is_flagged():
    """HNSW cannot be both more accurate and faster: higher ef_search visits
    more of the graph. This shape means a point was still warming up."""
    noisy = result_with([(10, 0.83, 4.14), (20, 0.92, 1.31), (320, 1.00, 3.29)])
    assert not noisy.monotonic


def test_a_well_behaved_curve_is_monotonic():
    clean = result_with([(10, 0.76, 1.13), (80, 0.98, 3.64), (320, 0.998, 11.17)])
    assert clean.monotonic
