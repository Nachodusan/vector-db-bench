"""Corpus: real text, embedded with a real model.

Synthetic Gaussian vectors are the usual shortcut in ANN benchmarks and they
flatter every index equally. Real embeddings are clustered, anisotropic and
have a long tail of near-duplicates, which is exactly where approximate search
starts making different mistakes in different implementations.

Public-domain books from Project Gutenberg, chunked by sentence groups. Cached
on disk, so the download and the chunking happen once.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / ".cache"

# Long, prose-heavy, out of copyright, and stable URLs.
BOOKS = {
    2701: "Moby-Dick (Melville)",
    1342: "Pride and Prejudice (Austen)",
    84:   "Frankenstein (Shelley)",
    2600: "War and Peace (Tolstoy)",
    1399: "Anna Karenina (Tolstoy)",
    2554: "Crime and Punishment (Dostoevsky)",
    345:  "Dracula (Stoker)",
    1400: "Great Expectations (Dickens)",
    768:  "Wuthering Heights (Brontë)",
    4300: "Ulysses (Joyce)",
}

WORDS_PER_CHUNK = 60
STRIDE = 45  # overlapping, as a real RAG pipeline would


@dataclass(frozen=True)
class Corpus:
    chunks: list[str]
    source: list[int]       # book id per chunk

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def fingerprint(self) -> str:
        """Identifies this exact corpus, so cached embeddings cannot drift."""
        h = hashlib.sha256()
        h.update(f"{len(self.chunks)}|{WORDS_PER_CHUNK}|{STRIDE}".encode())
        for c in self.chunks[::97]:          # sample, not the whole thing
            h.update(c.encode("utf-8"))
        return h.hexdigest()[:16]


def _download(book_id: int) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"gutenberg-{book_id}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "vector-db-bench"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed host
        text = resp.read().decode("utf-8", errors="replace")
    path.write_text(text, encoding="utf-8")
    return text


def _strip_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF TH[EIS].*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF TH[EIS].*?\*\*\*", text)
    return text[(start.end() if start else 0):(end.start() if end else len(text))]


def _chunk(text: str) -> list[str]:
    words = re.sub(r"\s+", " ", _strip_boilerplate(text)).split()
    out: list[str] = []
    for i in range(0, max(0, len(words) - WORDS_PER_CHUNK), STRIDE):
        chunk = " ".join(words[i:i + WORDS_PER_CHUNK]).strip()
        if len(chunk) > 80:
            out.append(chunk)
    return out


def load(limit: int | None = 60_000) -> Corpus:
    """Build (or reuse) the corpus. `limit` caps total chunks."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # The cache key covers everything that changes the output: which books,
    # how they are chunked, and the cap. Keying on the cap alone silently
    # served a stale corpus when books were added.
    key = hashlib.sha256(
        f"{sorted(BOOKS)}|{WORDS_PER_CHUNK}|{STRIDE}|{limit}".encode()
    ).hexdigest()[:12]
    cached = CACHE / f"corpus-{key}.json"
    if cached.exists():
        data = json.loads(cached.read_text(encoding="utf-8"))
        return Corpus(data["chunks"], data["source"])

    chunks: list[str] = []
    source: list[int] = []
    per_book = (limit // len(BOOKS)) if limit else None

    for book_id in BOOKS:
        got = _chunk(_download(book_id))
        if per_book:
            got = got[:per_book]
        chunks.extend(got)
        source.extend([book_id] * len(got))

    if limit:
        chunks, source = chunks[:limit], source[:limit]

    cached.write_text(json.dumps({"chunks": chunks, "source": source}), encoding="utf-8")
    return Corpus(chunks, source)


if __name__ == "__main__":
    c = load()
    print(f"{len(c):,} chunks · fingerprint {c.fingerprint}")
    for book_id, name in BOOKS.items():
        print(f"  {name:<34} {c.source.count(book_id):>6,} chunks")
    print()
    print("sample:", c.chunks[0][:160], "…")
