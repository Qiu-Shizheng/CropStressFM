import csv
from pathlib import Path

from cropstressfm.prepare import prepare_inputs


def test_prepare_strand_aware_promoters(tmp_path: Path) -> None:
    genome = tmp_path / "genome.fa"
    annotation = tmp_path / "genes.gff3"
    genome.write_text(">chr1\nAAAACCCCGGGGTTTT\n>chr2\nAAAACCCCGGGGTTTT\n", encoding="utf-8")
    annotation.write_text(
        "##gff-version 3\n"
        "chr1\ttest\tgene\t9\t10\t.\t+\t.\tID=gene_plus\n"
        "chr2\ttest\tgene\t7\t8\t.\t-\t.\tID=gene_minus\n",
        encoding="utf-8",
    )
    report = prepare_inputs(genome, annotation, tmp_path / "prepared", upstream=4)
    with (tmp_path / "prepared" / "genes.csv").open(encoding="utf-8") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    assert rows["gene_plus"]["promoter_sequence"] == "CCCC"
    assert rows["gene_minus"]["promoter_sequence"] == "CCCC"
    assert report["promoters_written"] == 2
