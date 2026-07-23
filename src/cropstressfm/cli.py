from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .plotting import plot_predictions
from .predictor import CropStressFM
from .prepare import prepare_inputs
from .protein import embed_proteins, read_fasta, write_embeddings


def load_prediction_input(path: str | Path) -> tuple[list[str], list[str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input CSV contains no rows")
    id_column = "id" if "id" in rows[0] else "gene_id"
    promoter_column = "promoter_sequence" if "promoter_sequence" in rows[0] else "promoter_seq"
    if id_column not in rows[0] or promoter_column not in rows[0]:
        raise ValueError("Input CSV requires id and promoter_sequence columns")
    identifiers = [row[id_column].strip() for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Input identifiers must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Input identifiers must be unique")
    return identifiers, [row[promoter_column] for row in rows]


def load_embedding_input(path: str | Path, wanted_ids: list[str]) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "embeddings" not in payload:
            raise ValueError("Embedding NPZ requires an embeddings array")
        values = np.asarray(payload["embeddings"], dtype=np.float32)
        id_key = next((key for key in ("ids", "gene_id", "gene_key") if key in payload), None)
        if id_key is None:
            if len(values) != len(wanted_ids):
                raise ValueError("Row-aligned embeddings have a different row count from the input CSV")
            return values
        observed = [str(value) for value in payload[id_key]]
    if len(set(observed)) != len(observed):
        raise ValueError("Embedding identifiers are not unique")
    lookup = {identifier: index for index, identifier in enumerate(observed)}
    missing = [identifier for identifier in wanted_ids if identifier not in lookup]
    if missing:
        raise ValueError(f"Embeddings are missing {len(missing)} input identifiers")
    return values[[lookup[identifier] for identifier in wanted_ids]]


def write_predictions(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_predict(args: argparse.Namespace) -> None:
    ids, promoters = load_prediction_input(args.input)
    embeddings = load_embedding_input(args.embeddings, ids)
    started = time.perf_counter()
    predictor = CropStressFM(device=args.device, batch_size=args.batch_size)
    predictions = predictor.predict(promoters, embeddings, ids=ids)
    elapsed = time.perf_counter() - started
    write_predictions(args.output, predictions)
    if args.plot:
        plot_predictions(predictions, args.plot)
    print(json.dumps({"output": str(args.output), "genes": len(ids), "device": str(predictor.device), "seconds": elapsed}, indent=2))


def command_embed(args: argparse.Namespace) -> None:
    records = read_fasta(args.fasta)
    ids, embeddings = embed_proteins(
        records,
        device=args.device,
        checkpoint=args.checkpoint,
        max_residues=args.max_residues,
        batch_size=args.batch_size,
    )
    write_embeddings(args.output, ids, embeddings)
    print(json.dumps({"output": str(args.output), "proteins": len(ids), "embedding_dimension": int(embeddings.shape[1])}, indent=2))


def command_prepare(args: argparse.Namespace) -> None:
    report = prepare_inputs(
        genome_path=args.genome,
        annotation_path=args.annotation,
        output_directory=args.output_dir,
        protein_fasta=args.proteins,
        gene_list=args.gene_list,
        upstream=args.upstream,
        feature_type=args.feature_type,
        gene_id_key=args.gene_id_key,
    )
    print(json.dumps(report, indent=2))


def command_info(args: argparse.Namespace) -> None:
    predictor = CropStressFM(device=args.device, batch_size=1)
    print(json.dumps(predictor.model_info(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cropstressfm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input", required=True)
    predict.add_argument("--embeddings", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--plot")
    predict.add_argument("--device", default="auto")
    predict.add_argument("--batch-size", type=int, default=64)
    predict.set_defaults(function=command_predict)
    embed = subparsers.add_parser("embed-protein")
    embed.add_argument("--fasta", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--checkpoint")
    embed.add_argument("--device", default="auto")
    embed.add_argument("--max-residues", type=int, default=1000)
    embed.add_argument("--batch-size", type=int, default=1)
    embed.set_defaults(function=command_embed)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--genome", required=True)
    prepare.add_argument("--annotation", required=True)
    prepare.add_argument("--proteins")
    prepare.add_argument("--gene-list")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--upstream", type=int, default=2000)
    prepare.add_argument("--feature-type", default="gene")
    prepare.add_argument("--gene-id-key", default="")
    prepare.set_defaults(function=command_prepare)
    info = subparsers.add_parser("info")
    info.add_argument("--device", default="cpu")
    info.set_defaults(function=command_info)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
