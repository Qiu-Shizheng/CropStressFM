from __future__ import annotations

import bisect
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from .features import reverse_complement
from .protein import read_fasta


@dataclass(frozen=True)
class Gene:
    identifier: str
    chromosome: str
    start: int
    end: int
    strand: str


def parse_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for field in text.strip().strip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        if "=" in field:
            key, value = field.split("=", 1)
        else:
            match = re.match(r"([^\s]+)\s+[\"']?(.+?)[\"']?$", field)
            if match is None:
                continue
            key, value = match.groups()
        attributes[key.strip()] = unquote(value.strip().strip("\"'"))
    return attributes


def read_genes(path: str | Path, feature_type: str = "gene", gene_id_key: str = "") -> list[Gene]:
    genes = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != feature_type:
                continue
            attributes = parse_attributes(fields[8])
            keys = [gene_id_key] if gene_id_key else ["gene_id", "ID", "Name", "locus_tag"]
            identifier = next((attributes.get(key, "") for key in keys if attributes.get(key, "")), "")
            if not identifier:
                continue
            genes.append(Gene(identifier, fields[0], int(fields[3]), int(fields[4]), fields[6]))
    unique = {gene.identifier: gene for gene in genes}
    return list(unique.values())


def prepare_inputs(
    genome_path: str | Path,
    annotation_path: str | Path,
    output_directory: str | Path,
    protein_fasta: str | Path | None = None,
    gene_list: str | Path | None = None,
    upstream: int = 2000,
    feature_type: str = "gene",
    gene_id_key: str = "",
) -> dict[str, object]:
    try:
        from pyfaidx import Fasta
    except ImportError as error:
        raise RuntimeError("Install pyfaidx to prepare inputs") from error
    if upstream <= 0:
        raise ValueError("upstream must be positive")
    selected_ids = None
    if gene_list is not None:
        selected_ids = {line.strip() for line in Path(gene_list).read_text(encoding="utf-8").splitlines() if line.strip()}
    genes = read_genes(annotation_path, feature_type=feature_type, gene_id_key=gene_id_key)
    by_chromosome: dict[str, list[Gene]] = {}
    for gene in genes:
        by_chromosome.setdefault(gene.chromosome, []).append(gene)
    previous_ends = {chromosome: sorted(gene.end for gene in group) for chromosome, group in by_chromosome.items()}
    next_starts = {chromosome: sorted(gene.start for gene in group) for chromosome, group in by_chromosome.items()}
    genome = Fasta(str(genome_path), as_raw=True, sequence_always_upper=True)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    for gene in genes:
        if selected_ids is not None and gene.identifier not in selected_ids:
            continue
        if gene.chromosome not in genome:
            skipped.append({"id": gene.identifier, "reason": "chromosome_not_found"})
            continue
        chromosome_length = len(genome[gene.chromosome])
        if gene.strand == "+":
            tss0 = max(0, gene.start - 1)
            ends = previous_ends[gene.chromosome]
            position = bisect.bisect_left(ends, gene.start) - 1
            boundary = ends[position] if position >= 0 else 0
            start0 = max(boundary, tss0 - upstream)
            end0 = tss0
            sequence = str(genome[gene.chromosome][start0:end0])
        elif gene.strand == "-":
            tss0 = min(chromosome_length, gene.end)
            starts = next_starts[gene.chromosome]
            position = bisect.bisect_right(starts, gene.end)
            boundary = starts[position] - 1 if position < len(starts) else chromosome_length
            start0 = tss0
            end0 = min(boundary, tss0 + upstream)
            sequence = reverse_complement(str(genome[gene.chromosome][start0:end0]))
        else:
            skipped.append({"id": gene.identifier, "reason": "invalid_strand"})
            continue
        if not sequence:
            skipped.append({"id": gene.identifier, "reason": "empty_intergenic_promoter"})
            continue
        rows.append(
            {
                "id": gene.identifier,
                "promoter_sequence": sequence,
                "promoter_length": len(sequence),
                "chromosome": gene.chromosome,
                "strand": gene.strand,
                "interval_start_0based": start0,
                "interval_end_0based": end0,
            }
        )
    genes_csv = output / "genes.csv"
    with genes_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id", "promoter_sequence"])
        writer.writeheader()
        writer.writerows(rows)
    protein_count = 0
    missing_proteins = []
    if protein_fasta is not None:
        proteins = read_fasta(protein_fasta)
        protein_output = output / "proteins.fasta"
        with protein_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                identifier = row["id"]
                if identifier not in proteins:
                    missing_proteins.append(identifier)
                    continue
                sequence = proteins[identifier]
                handle.write(f">{identifier}\n")
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start : start + 80] + "\n")
                protein_count += 1
    report = {
        "genes_in_annotation": len(genes),
        "genes_requested": len(selected_ids) if selected_ids is not None else len(genes),
        "promoters_written": len(rows),
        "proteins_written": protein_count,
        "missing_protein_ids": missing_proteins,
        "skipped": skipped,
        "promoter_definition": f"strand-aware upstream intergenic sequence clipped by the nearest annotated gene and a {upstream} bp maximum",
        "annotation_feature_type": feature_type,
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
