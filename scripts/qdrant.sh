#!/usr/bin/env bash
# Fetch and run a real Qdrant server. No Docker.
#
# The embedded mode (QdrantClient(path=...)) is not an option for a benchmark:
# qdrant-client's own source says "Local mode performs exact (brute-force)
# search, so `search_params` has no effect."
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.cache/qdrant"
mkdir -p "$DIR/storage"
cd "$DIR"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) ASSET=qdrant-aarch64-apple-darwin.tar.gz ;;
  Darwin-x86_64) ASSET=qdrant-x86_64-apple-darwin.tar.gz ;;
  Linux-x86_64) ASSET=qdrant-x86_64-unknown-linux-gnu.tar.gz ;;
  Linux-aarch64) ASSET=qdrant-aarch64-unknown-linux-musl.tar.gz ;;
  *) echo "no prebuilt qdrant for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

if [ ! -x ./qdrant ]; then
  echo "fetching $ASSET"
  curl -sSL -o q.tar.gz "https://github.com/qdrant/qdrant/releases/latest/download/$ASSET"
  tar xzf q.tar.gz && rm q.tar.gz
fi

echo "qdrant on http://localhost:6333"
exec env QDRANT__STORAGE__STORAGE_PATH=./storage QDRANT__SERVICE__HTTP_PORT=6333 ./qdrant
