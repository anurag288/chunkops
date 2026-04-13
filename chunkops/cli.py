"""
chunkops CLI

Commands:
    chunkops chunk  <file>  [--strategy recursive] [--max-tokens 400]
    chunkops bench  <file>  [--strategies recursive,semantic]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_chunk(args):
    from chunkops import Chunker
    text = Path(args.file).read_text(encoding="utf-8")
    chunker = Chunker(strategy=args.strategy, min_tokens=args.min_tokens, max_tokens=args.max_tokens)
    chunks = chunker.chunk(text, doc_id=args.file)
    print(f"\nFile: {args.file}")
    print(f"Strategy: {args.strategy}  |  min_tokens: {args.min_tokens}  |  max_tokens: {args.max_tokens}")
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        preview = c.text[:80].replace("\n", " ")
        print(f"  [{c.id}] tokens={c.token_count:>4}  span={c.span}  {preview!r}")
    print()


def cmd_bench(args):
    from chunkops import benchmark
    text = Path(args.file).read_text(encoding="utf-8")
    strategies = [s.strip() for s in args.strategies.split(",")]
    print(f"\nBenchmarking {len(strategies)} strategies on {args.file} ...")
    result = benchmark([text], strategies=strategies, metric=args.metric)
    result.report()


def main():
    parser = argparse.ArgumentParser(
        prog="chunkops",
        description="Semantic-aware chunking with provenance tracking",
    )
    sub = parser.add_subparsers(dest="command")

    # chunk subcommand
    p_chunk = sub.add_parser("chunk", help="Chunk a text file")
    p_chunk.add_argument("file", help="Path to input .txt file")
    p_chunk.add_argument("--strategy", default="recursive",
                         choices=["fixed", "recursive", "structural", "semantic", "adaptive"])
    p_chunk.add_argument("--min-tokens", type=int, default=50)
    p_chunk.add_argument("--max-tokens", type=int, default=400)

    # bench subcommand
    p_bench = sub.add_parser("bench", help="Benchmark strategies on a file")
    p_bench.add_argument("file", help="Path to input .txt file")
    p_bench.add_argument("--strategies", default="fixed,recursive,structural",
                         help="Comma-separated list of strategies")
    p_bench.add_argument("--metric", default="coherence",
                         choices=["coherence", "chunk_count", "avg_tokens"])

    args = parser.parse_args()
    if args.command == "chunk":
        cmd_chunk(args)
    elif args.command == "bench":
        cmd_bench(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
