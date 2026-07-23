from __future__ import annotations

import numpy as np


DNA_CODES = {"A": 0, "C": 1, "G": 2, "T": 3}


def normalize_dna(sequence: str) -> str:
    return "".join(base if base in DNA_CODES else "N" for base in str(sequence).upper())


def reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return str(sequence).translate(table)[::-1].upper()


def exact_kmer_frequencies(sequence: str, k: int = 5) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    counts = np.zeros(4**k, dtype=np.float32)
    mask = (1 << (2 * k)) - 1
    code = 0
    run = 0
    for base in normalize_dna(sequence):
        value = DNA_CODES.get(base)
        if value is None:
            code = 0
            run = 0
            continue
        code = ((code << 2) | value) & mask
        run += 1
        if run >= k:
            counts[code] += 1.0
    total = float(counts.sum())
    if total > 0:
        counts /= total
    return counts


def promoter_views(sequence: str, maximum_length: int = 2000) -> tuple[str, str, str, str]:
    cleaned = normalize_dna(sequence)
    if not cleaned:
        raise ValueError("Promoter sequence is empty")
    full = cleaned[-maximum_length:]
    return full, full[-200:], full[-500:], full[-2000:]


def build_promoter_features(sequences: list[str], k: int = 5, maximum_length: int = 2000) -> np.ndarray:
    matrix = np.zeros((len(sequences), 4 * (4**k)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        views = promoter_views(sequence, maximum_length=maximum_length)
        matrix[row] = np.concatenate([exact_kmer_frequencies(view, k=k) for view in views])
    return matrix
