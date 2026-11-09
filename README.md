# vector-db-bench

Comparing vector databases at default settings measures the defaults.

Every HNSW implementation trades accuracy for speed on a knob. Quote one
latency number per engine and you have reported where each vendor happened to
set `ef_search`, not how the engines compare. So this sweeps the knob, measures
recall against exact brute force, and compares **latency at matched recall**.

41,896 chunks of public-domain prose, embedded with BGE-M3 (1024d), indexed in
Qdrant 1.20, Chroma 1.5.9 and pgvector 0.8 with identical HNSW parameters
(`M=16`, `ef_construction=200`), on an Apple M4.

---

## The result

| recall target | qdrant | pgvector | ratio |
|---|---|---|---|
| 0.90 | **1.39 ms** | 1.90 ms | 1.4× |
| 0.95 | **1.52 ms** | 2.79 ms | 1.8× |
| 0.99 | **1.84 ms** | 6.19 ms | 3.4× |

p95, interpolated along each engine's own curve.

**The gap widens with accuracy.** At recall 0.90 the two are within 40% of each
other; at 0.99 Qdrant is 3.4× faster. "Which is faster" has no answer until you
name a recall target — and a single-point benchmark picks one of those three
numbers without telling you which.

Index build, same data, same parameters:

| | qdrant | chroma | pgvector |
|---|---|---|---|
| build | **11.3 s** | 21.3 s | 187.8 s |

Full sweeps in [`results/results.json`](results/results.json).

### Chroma is not in the table

Chroma 1.5.9 accepts `hnsw:ef_search`, persists it, and reports the new value
back when asked. It does not act on it. Across `ef_search` 10 → 320 the results
are byte-identical for all 500 queries:

```
  ef=10   recall 0.9716  p50 2.04 ms  p95 2.38 ms
  ef=40   recall 0.9716  p50 2.05 ms  p95 2.40 ms
  ef=320  recall 0.9716  p50 2.00 ms  p95 2.35 ms
```

So Chroma has one operating point — recall 0.9716 at p95 2.38 ms — and no way
to trade along the curve. That is a perfectly reasonable place to sit. It just
cannot be compared *at matched recall*, because it cannot be matched. Averaging
it in anyway would have produced a number with no meaning, so it is reported
separately and excluded from the comparison.

## Five ways this benchmark was wrong before it was right

Every one of these produced a complete, plausible-looking table. None of them
is visible in the output.

**1. Qdrant was doing brute force.** It has *two* thresholds.
`indexing_threshold` decides whether an HNSW index gets built;
`full_scan_threshold` (10,000 KB by default) decides whether a segment is
searched with it or scanned exhaustively anyway. Lower only the first and you
get an index that exists and is never used. The tell is recall pinned at a flat
1.000 for every `ef_search`.

Also: qdrant-client's embedded mode cannot be benchmarked at all. Its own
source says so — *"Local mode performs exact (brute-force) search, so
`search_params` has no effect."* This repo runs a real server from the native
binary; [`scripts/qdrant.sh`](scripts/qdrant.sh) fetches it. No Docker.

**2. pgvector's index build was measured at the default
`maintenance_work_mem`.** 64 MB, against an index an order of magnitude larger,
so it built on disk: **448 s instead of 188 s**. Letting a default mislead
*against* an engine is the same error as letting it mislead for one.

**3. The client was charging pgvector for its own overhead.** Query vectors
were being sent as 1024-float text literals, which Postgres then parsed before
it could search. Switching to the binary protocol moved p95 at recall 0.95 from
**6.80 ms to 2.79 ms**. Published as-was, the headline would have read "Qdrant
is 4.7× faster"; the real figure at that recall is 1.8×.

**4. The first point of every sweep was still warming up.** Measured three
times in a row at an identical setting:

```
  pass 1   p50 1.11 ms   p95 4.67 ms   max 13.27 ms
  pass 2   p50 1.04 ms   p95 1.34 ms   max  1.75 ms
  pass 3   p50 1.12 ms   p95 1.36 ms   max  1.77 ms
```

Not the ef value — the spike followed whatever was measured first, and 50
warm-up queries did not absorb it. It bent one end of the curve upward, which
showed up as **latency falling as recall rose**. HNSW cannot do that: a higher
`ef_search` visits more of the graph, so it cannot be both more accurate and
faster. That impossibility is the only reason this one got caught, and it is
now a guard.

**5. Isolated points were picking up system noise.** p50 steady at 1.5 ms while
p95 jumped to 4 ms, from whatever else the machine was doing for a moment. Each
query is now timed in three passes; the first is discarded entirely and the
**minimum across the rest** is kept. Interference only ever adds time, so the
per-query minimum is the closest estimate of what the engine costs. Recall is
identical across passes, so nothing is lost.

## The guards

Three of the five above are now checked automatically, because every one of
them produced output that looked finished.

| Guard | Catches |
|---|---|
| `knob_effective` | recall identical across the whole sweep — either brute force in disguise, or a setter that lies |
| `monotonic` | latency falling as recall rises, which the algorithm forbids |
| `latency_at_recall` returns `None` | an engine that never reaches the target, instead of extrapolating one |

An engine that fails `knob_effective` is excluded from matched-recall
comparison rather than quietly averaged in.

## Method

**Ground truth is exact.** Brute force over the same vectors — a single matrix
multiply at this scale — not another approximate index and not human relevance
labels. Those are different questions: whether the *right passage* comes back
is a property of the embedding model, while whether the index returns what
exact search would have is a property of the index. Only the second is a fair
thing to compare databases on.

**Queries are held out.** 500 corpus vectors, never indexed. Querying with
vectors that are in the index makes top-1 trivially itself and hides exactly
the differences being measured.

**Real embeddings, not synthetic vectors.** Gaussian noise flatters every index
equally. Real embeddings are clustered, anisotropic and full of
near-duplicates, which is where approximate search starts making different
mistakes in different implementations.

**Recall ignores order within k.** An index that returns the right ten in the
wrong order has found them; ranking them is the scorer's job upstream.

**Build parameters are pinned across engines.** `M` and `ef_construction` are
set once in [`bench/engines/base.py`](bench/engines/base.py) and apply
everywhere. Changing them changes every engine at once, which is the only
honest way to change them.

## Running it

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"

scripts/qdrant.sh &                 # native binary, no Docker
createdb bench 2>/dev/null; export BENCH_PG_DSN=postgresql:///bench

cp .env.example .env                # an embeddings endpoint
python -m bench.corpus              # download and chunk
python -m bench.embed               # embed once, cached
python -m bench.run                 # the benchmark
```

The corpus and vectors are cached under `.cache/`, keyed by a fingerprint of
the book list and chunking parameters, so adding a book invalidates them
properly. (It did not, at first. That is bug zero.)

Python 3.13 rather than 3.15: `onnxruntime`, which Chroma depends on, has no
wheels for 3.15, and pip resolves Chroma back to a 2023 release that then fails
to import.

### Tests

```bash
pytest
```

Eighteen tests, no network, 0.1 s — on the measuring apparatus rather than the
databases. An instrument that can be wrong about the thing it measures needs
its own tests more than the thing does. Six of them cover the guards.

## What this is not

- **Not a scale benchmark.** 41k vectors on a laptop. Index behaviour at 10M
  vectors on server hardware is a different question, and the differences here
  would not transfer unexamined.
- **Not a feature comparison.** Filtering, hybrid search, replication,
  durability and operational cost decide most real choices and none are
  measured. pgvector's actual argument is that your vectors live next to your
  rows in a database you already run; that does not appear in a latency table.
- **Not a verdict.** Qdrant is faster at matched recall here, by between 1.4×
  and 3.4× depending entirely on where you need to sit.

## Licence

[MIT](LICENSE).
