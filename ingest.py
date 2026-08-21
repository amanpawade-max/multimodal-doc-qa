#!/usr/bin/env python3
"""Usage: python ingest.py path/to/document.pdf [more.pdf ...]"""
import sys

from src.pipeline import ingest
from src.embed_store import VectorStore


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    store = VectorStore()
    for path in sys.argv[1:]:
        ingest(path, store=store)
    print(f"\nTotal chunks in index: {store.count()}")


if __name__ == "__main__":
    main()
