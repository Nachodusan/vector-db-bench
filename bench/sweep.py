"""Sweep ef_search, measure recall and latency, interpolate the comparison.

Reporting one latency number per engine is the mistake this module exists to
avoid. Every HNSW implementation trades accuracy for speed on a knob, so a
single measurement says only "this engine, at whatever ef_search its author
happened to default to". Two engines can be a factor of three apart on latency
and identical once you hold recall fixed.

So: sweep the knob, record the whole curve, and compare at matched recall.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from bench.engines.base import EF_SEARCH, Engine
from bench.truth import Split, recall_at_k

WARMUP = 50


@dataclass
class Point:
    ef_search: int
    recall: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    qps: float


@dataclass
class EngineResult:
    engine: str
    knob: str
    build_seconds: float
    build_note: str
    vectors: int
    dim: int
    points: list[Point] = field(default_factory=list)

    @property
    def monotonic(self) -> bool:
        """Does latency rise with recall, as HNSW requires?

        Higher ef_search searches more of the graph: it cannot be both more
        accurate and faster. A curve that says otherwise is measurement
        noise — usually a point that was still warming up — and the
        matched-recall figures interpolated across it are not trustworthy.
        """
        curve = sorted(self.points, key=lambda p: p.recall)
        return all(a.p95_ms <= b.p95_ms * 1.15 for a, b in zip(curve, curve[1:]))

    @property
    def knob_effective(self) -> bool:
        """Did sweeping the knob change anything at all?

        Two separate ways to end up measuring one setting six times:

        * the engine silently serves exact results (Qdrant below its
          `full_scan_threshold`), so recall is a flat 1.000; or
        * the engine accepts and reports the new value but does not act on it
          (Chroma 1.5.9's `hnsw:ef_search`), so recall is flat at whatever it
          happens to be.

        Both look like a finished benchmark. Neither is one, and an engine
        whose knob is inert cannot be placed on a recall/latency curve, so it
        is excluded from matched-recall comparison rather than quietly
        averaged in.
        """
        recalls = {p.recall for p in self.points}
        return len(recalls) > 1

    def latency_at_recall(self, target: float) -> float | None:
        """Interpolated p95 latency at a recall target.

        Returns None when the engine never reaches the target anywhere on its
        curve — which is itself a result, and more honest than extrapolating.
        """
        if not self.knob_effective:
            return None
        curve = sorted(self.points, key=lambda p: p.recall)
        if not curve or curve[-1].recall < target:
            return None
        below = [p for p in curve if p.recall <= target]
        above = [p for p in curve if p.recall >= target]
        if not below:
            return above[0].p95_ms
        lo, hi = below[-1], above[0]
        if hi.recall == lo.recall:
            return lo.p95_ms
        t = (target - lo.recall) / (hi.recall - lo.recall)
        return lo.p95_ms + t * (hi.p95_ms - lo.p95_ms)


def measure(engine: Engine, split: Split, *, k: int = 10,
            ef_values: tuple[int, ...] = EF_SEARCH,
            passes: int = 3,
            on_point=None) -> EngineResult:
    """Sweep the knob and measure, discarding everything but the last pass.

    The first pass after a build is not comparable to the ones after it. On
    Qdrant, measured three times in a row at an identical setting:

        pass 1   p50 1.11 ms   p95 4.67 ms   max 13.27 ms
        pass 2   p50 1.04 ms   p95 1.34 ms   max  1.75 ms
        pass 3   p50 1.12 ms   p95 1.36 ms   max  1.77 ms

    It is not the ef value — the same spike appears at whatever is measured
    first, and 50 warm-up queries do not absorb it. It is a one-off cost the
    engine pays settling after the index is built, and it lands entirely on
    whichever point the sweep happens to visit first, bending that end of the
    curve upward. The symptom is a curve where latency falls as recall rises,
    which HNSW cannot do.

    Discarding the first pass fixed that, and left a second kind of noise:
    isolated points where p50 stayed at 1.5 ms while p95 jumped to 4 ms, from
    whatever else the machine was doing for a moment.

    So each query is timed in every pass and **the minimum across passes is
    kept**. Interference only ever adds time, so the per-query minimum is the
    closest estimate of what the engine actually costs, and percentiles taken
    over those minima describe the engine rather than the laptop. Recall is
    identical across passes — the index does not change — so nothing is lost.
    """
    stats = engine.build(split.index_vectors, split.index_ids)
    result = EngineResult(
        engine=engine.name,
        knob=engine.knob,
        build_seconds=round(stats.seconds, 2),
        build_note=stats.note,
        vectors=len(split.index_vectors),
        dim=int(split.index_vectors.shape[1]),
    )

    # Warm up caches, JIT, connection pools, page cache — whatever each engine
    # has. Timing the first queries after a build measures the build.
    engine.search(split.queries[:WARMUP], k, ef_values[-1])

    for ef in ef_values:
        engine.search(split.queries[:WARMUP], k, ef)

        best = np.full(len(split.queries), np.inf)
        got: list[list[int]] = []
        for pass_no in range(passes):
            got = []
            for j, q in enumerate(split.queries):
                t0 = time.perf_counter()
                hits = engine.search(q[None, :], k, ef)[0]
                elapsed = (time.perf_counter() - t0) * 1000.0
                if pass_no > 0 or passes == 1:
                    best[j] = min(best[j], elapsed)
                got.append(hits)
            if pass_no == 0:
                # The settling pass. Timed but never counted.
                best[:] = np.inf

        arr = best
        point = Point(
            ef_search=ef,
            recall=round(recall_at_k(split.truth, got), 4),
            p50_ms=round(float(np.percentile(arr, 50)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            mean_ms=round(float(arr.mean()), 3),
            qps=round(1000.0 / float(arr.mean()), 1),
        )
        result.points.append(point)
        if on_point:
            on_point(result, point)

    return result


def to_dict(result: EngineResult) -> dict:
    d = asdict(result)
    d["knob_effective"] = result.knob_effective
    d["monotonic"] = result.monotonic
    d["latency_at_recall"] = {
        str(target): (round(v, 3) if (v := result.latency_at_recall(target)) else None)
        for target in (0.90, 0.95, 0.99)
    }
    return d
