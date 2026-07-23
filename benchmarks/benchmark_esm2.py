from __future__ import annotations

import argparse
import json
import resource
import time

import torch

from cropstressfm import embed_proteins, read_fasta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="examples/input/example_proteins.fasta")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    records = read_fasta(args.fasta)
    records = dict(list(records.items())[: args.limit])
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    ids, embeddings = embed_proteins(records, device=args.device, checkpoint=args.checkpoint)
    seconds = time.perf_counter() - started
    result = {
        "device": args.device,
        "proteins": len(ids),
        "embedding_shape": list(embeddings.shape),
        "seconds": seconds,
        "peak_process_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        result["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / 1024.0**2
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
