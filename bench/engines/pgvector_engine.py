"""pgvector, against a real PostgreSQL server.

The only engine here that is not embedded, which shows up in the numbers:
every query pays client/server round-trip and parse cost that Qdrant and
Chroma do not. That is not a flaw in the measurement, it is the thing you are
buying when you keep your vectors next to your rows.

Rows are loaded with COPY rather than INSERT, and the HNSW index is built
after the load, which is what pgvector's own documentation recommends and what
anyone operating it would do.
"""

from __future__ import annotations

import os
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from bench.engines.base import EF_CONSTRUCTION, M, BuildStats

TABLE = "bench_vectors"
DSN = os.getenv("BENCH_PG_DSN", "postgresql:///postgres")

# Build-time tuning. Reported in the results so it is never mistaken for a
# default.
MAINTENANCE_WORK_MEM = os.getenv("BENCH_PG_MAINTENANCE_WORK_MEM", "2GB")
PARALLEL_WORKERS = int(os.getenv("BENCH_PG_PARALLEL_WORKERS", "4"))


class PgvectorEngine:
    name = "pgvector"
    knob = "hnsw.ef_search"

    def __init__(self, dsn: str | None = None) -> None:
        self.conn = psycopg.connect(dsn or DSN, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Send and receive vectors in the binary protocol. Without this the
        # client ships a 1024-float text literal per query and the server
        # parses it before it can search — client-side overhead that would
        # have been charged to pgvector's search time.
        register_vector(self.conn)

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> BuildStats:
        dim = vectors.shape[1]
        t0 = time.perf_counter()
        self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        self.conn.execute(f"CREATE TABLE {TABLE} (id bigint PRIMARY KEY, v vector({dim}))")

        with self.conn.cursor().copy(f"COPY {TABLE} (id, v) FROM STDIN") as copy:
            for i, vec in zip(ids, vectors):
                copy.write_row((int(i), "[" + ",".join(f"{x:.6f}" for x in vec) + "]"))
        load_s = time.perf_counter() - t0

        # pgvector builds HNSW in memory when the graph fits in
        # maintenance_work_mem and spills to disk when it does not. The
        # default is 64 MB; this index is an order of magnitude larger, and
        # leaving the default in place made the build take 448 s instead of
        # the time below. Measuring that number would have been measuring the
        # default, which is the thing this whole benchmark argues against.
        self.conn.execute(f"SET maintenance_work_mem = '{MAINTENANCE_WORK_MEM}'")
        self.conn.execute(
            f"SET max_parallel_maintenance_workers = {PARALLEL_WORKERS}"
        )

        # Inner product ops, matching the unit-length vectors.
        self.conn.execute(
            f"CREATE INDEX ON {TABLE} USING hnsw (v vector_ip_ops) "
            f"WITH (m = {M}, ef_construction = {EF_CONSTRUCTION})"
        )
        self.conn.execute(f"ANALYZE {TABLE}")
        total = time.perf_counter() - t0
        return BuildStats(
            total,
            note=(f"copy {load_s:.1f}s, index {total - load_s:.1f}s, "
                  f"maintenance_work_mem={MAINTENANCE_WORK_MEM}, "
                  f"parallel_workers={PARALLEL_WORKERS}"),
        )

    def search(self, queries: np.ndarray, k: int, ef_search: int) -> list[list[int]]:
        self.conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        sql = f"SELECT id FROM {TABLE} ORDER BY v <#> %b LIMIT {int(k)}"
        out: list[list[int]] = []
        for q in queries:
            out.append([r[0] for r in self.conn.execute(sql, (q,)).fetchall()])
        return out

    def close(self) -> None:
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        finally:
            self.conn.close()
