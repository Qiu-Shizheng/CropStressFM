from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from cropstressfm import CropStressFM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input", default="examples/input/example_genes.csv")
    parser.add_argument("--embeddings", default="examples/input/example_esm2_embeddings.npz")
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    with np.load(args.embeddings, allow_pickle=False) as payload:
        source_embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    indices = np.arange(args.genes) % len(source)
    ids = [f"benchmark_{index}" for index in range(args.genes)]
    promoters = [source[index]["promoter_sequence"] for index in indices]
    embeddings = source_embeddings[indices]
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictor = CropStressFM(device=args.device, batch_size=args.batch_size)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictor.predict(promoters, embeddings, ids=ids)
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - started
    timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        predictor.predict(promoters, embeddings, ids=ids)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    inference_seconds = float(np.median(timings))
    result = {
        "device": str(predictor.device),
        "genes": args.genes,
        "batch_size": args.batch_size,
        "model_load_seconds": load_seconds,
        "cold_inference_seconds": cold_seconds,
        "median_warm_inference_seconds": inference_seconds,
        "median_warm_milliseconds_per_gene": inference_seconds * 1000.0 / args.genes,
        "peak_process_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    if predictor.device.type == "cuda":
        result["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 1024.0**2
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
