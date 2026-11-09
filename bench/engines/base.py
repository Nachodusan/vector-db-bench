"""What every engine has to provide, and the parameters they all share.

The index parameters below are fixed across engines on purpose. HNSW's recall
is governed by `M` and `ef_construction` at build time and `ef_search` at query
time; letting each engine pick its own defaults would measure the defaults.
Here the build parameters are pinned and `ef_search` is swept, so what is left
to differ is the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Shared HNSW build parameters. Changing these changes every engine at once,
# which is the only honest way to change them.
M = 16
EF_CONSTRUCTION = 200

# The query-time knob, swept for every engine.
EF_SEARCH = (10, 20, 40, 80, 160, 320)


@dataclass
class BuildStats:
    seconds: float
    note: str = ""


class Engine(Protocol):
    name: str
    knob: str

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> BuildStats: ...
    def search(self, queries: np.ndarray, k: int, ef_search: int) -> list[list[int]]: ...
    def close(self) -> None: ...
