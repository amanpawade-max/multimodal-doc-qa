#!/usr/bin/env python3
"""Usage: python query.py "your question here" """
import sys

from src.pipeline import answer


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(answer(question))


if __name__ == "__main__":
    main()
