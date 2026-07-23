from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


def read_fasta(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    identifier = ""
    chunks: list[str] = []

    def store() -> None:
        if not identifier:
            return
        sequence = "".join(chunks).replace("*", "").replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"Empty protein sequence for {identifier}")
        if identifier in records:
            raise ValueError(f"Duplicate protein identifier: {identifier}")
        records[identifier] = sequence

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                store()
                identifier = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    store()
    if not records:
        raise ValueError("Protein FASTA contains no records")
    return records


def _local_esm_model(esm_module, checkpoint: Path):
    model_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return esm_module.pretrained.load_model_and_alphabet_core(checkpoint.stem, model_data, None)


def _load_esm(checkpoint: str | Path | None):
    try:
        import esm
    except ImportError as error:
        raise RuntimeError("Install the protein extra with: pip install 'cropstressfm[protein]'") from error
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return _local_esm_model(esm, path)
    cached = Path(torch.hub.get_dir()) / "checkpoints" / "esm2_t33_650M_UR50D.pt"
    if cached.exists():
        return _local_esm_model(esm, cached)
    return esm.pretrained.esm2_t33_650M_UR50D()


def embed_proteins(
    sequences: Mapping[str, str],
    device: str = "auto",
    checkpoint: str | Path | None = None,
    max_residues: int = 1000,
    batch_size: int = 1,
) -> tuple[list[str], np.ndarray]:
    if not 1 <= max_residues <= 1022:
        raise ValueError("max_residues must be between 1 and 1022")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if str(target).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model, alphabet = _load_esm(checkpoint)
    model = model.eval().to(torch.device(target))
    converter = alphabet.get_batch_converter()
    layer = int(model.num_layers)
    dimension = int(model.embed_dim)
    identifiers = list(sequences)
    output = []
    with torch.inference_mode():
        for identifier in identifiers:
            sequence = str(sequences[identifier]).replace("*", "").replace(" ", "").upper()
            if not sequence:
                raise ValueError(f"Empty protein sequence for {identifier}")
            chunks = [sequence[start : start + max_residues] for start in range(0, len(sequence), max_residues)]
            total = np.zeros(dimension, dtype=np.float64)
            maximum = np.full(dimension, -np.inf, dtype=np.float32)
            observed = 0
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                labels = [f"{identifier}_{start + offset}" for offset in range(len(batch))]
                _, _, tokens = converter(list(zip(labels, batch, strict=True)))
                representation = model(tokens.to(target), repr_layers=[layer], return_contacts=False)["representations"][layer]
                for index, chunk in enumerate(batch):
                    residues = representation[index, 1 : len(chunk) + 1].float()
                    total += residues.sum(dim=0).cpu().numpy().astype(np.float64, copy=False)
                    maximum = np.maximum(maximum, residues.max(dim=0).values.cpu().numpy().astype(np.float32, copy=False))
                    observed += len(chunk)
            vector = np.concatenate([(total / observed).astype(np.float32), maximum.astype(np.float32)])
            output.append(vector)
    return identifiers, np.asarray(output, dtype=np.float32)


def write_embeddings(path: str | Path, ids: list[str], embeddings: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.asarray(ids),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        model_name="ESM-2 t33 650M UR50D",
        pooling="residue-length-weighted mean plus global maximum",
    )
