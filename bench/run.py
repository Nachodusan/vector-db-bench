"""Run the benchmark.

    python -m bench.run                        # all engines
    python -m bench.run --engines qdrant chroma
    python -m bench.run --limit 20000          # smaller corpus
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from bench.corpus import load
from bench.embed import MODEL, embed
from bench.engines.base import EF_CONSTRUCTION, EF_SEARCH, M
from bench.sweep import measure, to_dict
from bench.truth import make_split

RESULTS = Path(__file__).resolve().parents[1] / "results"


def build_engine(name: str):
    if name == "qdrant":
        from bench.engines.qdrant_engine import QdrantEngine
        return QdrantEngine()
    if name == "chroma":
        from bench.engines.chroma_engine import ChromaEngine
        return ChromaEngine()
    if name == "pgvector":
        from bench.engines.pgvector_engine import PgvectorEngine
        return PgvectorEngine()
    raise SystemExit(f"unknown engine: {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="*", default=["qdrant", "chroma", "pgvector"])
    ap.add_argument("--limit", type=int, default=60_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS / "results.json"))
    args = ap.parse_args(argv)

    corpus = load(args.limit)
    vectors = embed(corpus, progress=False)
    split = make_split(vectors, k=args.k)

    print(f"corpus   {len(corpus):,} chunks · {MODEL} · {vectors.shape[1]}d")
    print(f"indexed  {len(split.index_vectors):,} · queries {len(split.queries)} · top-{args.k}")
    print(f"hnsw     M={M} ef_construction={EF_CONSTRUCTION} · sweeping ef_search {list(EF_SEARCH)}")
    print()

    results = []
    for name in args.engines:
        engine = build_engine(name)
        print(f"{name}")
        try:
            def show(result, point, name=name):
                print(f"  ef={point.ef_search:<4} recall {point.recall:.4f}  "
                      f"p50 {point.p50_ms:>7.2f} ms  p95 {point.p95_ms:>7.2f} ms  "
                      f"{point.qps:>7.1f} q/s")

            result = measure(engine, split, k=args.k, on_point=show)
        finally:
            engine.close()

        note = f" ({result.build_note})" if result.build_note else ""
        print(f"  build {result.build_seconds:.1f}s{note}")
        if not result.knob_effective:
            only = result.points[0]
            print(f"  {result.knob} is inert: recall {only.recall:.4f} at every value "
                  f"in the sweep.")
            print(f"  One fixed operating point — recall {only.recall:.4f} at "
                  f"p95 {only.p95_ms:.2f} ms — with no way to trade along the curve,")
            print("  so it cannot be compared at matched recall.")
        else:
            if not result.monotonic:
                print("  WARNING: latency does not rise with recall on this curve; "
                      "a point was still warming up and the figures below are noisy.")
            for target in (0.90, 0.95, 0.99):
                at = result.latency_at_recall(target)
                print(f"  p95 @ recall {target:.2f}: " +
                      (f"{at:.2f} ms" if at else "out of reach on this curve"))
        print()
        results.append(to_dict(result))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": f"{platform.machine()} · {platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "corpus": {
            "chunks": len(corpus),
            "fingerprint": corpus.fingerprint,
            "embedding_model": MODEL,
            "dim": int(vectors.shape[1]),
        },
        "setup": {
            "indexed": len(split.index_vectors),
            "queries": len(split.queries),
            "k": args.k,
            "m": M,
            "ef_construction": EF_CONSTRUCTION,
            "ef_search_sweep": list(EF_SEARCH),
        },
        "engines": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"results → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
