from __future__ import annotations

"""Standalone worker: runs CrossEncoderReranker in an isolated process.

Trên máy Windows này, load CrossEncoder (sentence-transformers) trong cùng process đã
load SentenceTransformer (dùng bởi DenseSearch) gây segfault native (OpenMP/torch runtime
conflict). Worker này chạy rerank trong process riêng qua subprocess, nhận input/trả kết
quả qua JSON file để tránh xung đột.

Usage: python -m src._rerank_worker <input_json_path> <output_json_path>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, encoding="utf-8") as f:
        payload = json.load(f)

    from src.m3_rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    result = reranker.rerank(payload["query"], payload["documents"], top_k=payload["top_k"])

    output = [
        {"text": r.text, "original_score": r.original_score, "rerank_score": r.rerank_score,
         "metadata": r.metadata, "rank": r.rank}
        for r in result
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
