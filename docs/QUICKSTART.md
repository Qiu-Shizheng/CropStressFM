# Five-minute quick start

This guide runs the complete CropStressFM prediction path using the pretrained ESM-2 embeddings included in the repository.

## Install

```bash
git clone https://github.com/Qiu-Shizheng/CropStressFM.git
cd CropStressFM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Confirm that the release weights load:

```bash
cropstressfm info --device cpu
```

The response reports the release version, three ensemble seeds, input dimensions and supported stress types.

## Run the real example

```bash
cropstressfm predict \
  --input examples/input/example_genes.csv \
  --embeddings examples/input/example_esm2_embeddings.npz \
  --output examples/output/quickstart_predictions.csv \
  --plot examples/output/quickstart_predictions.png \
  --device cpu
```

The command reads five experimentally supported crop genes and writes one row per gene. `stress_gene_probability` answers the general prioritization question. The six `<type>_probability` columns answer independent heat, drought, salt, cold, osmotic and other-abiotic questions.

## Run one new gene

Create `one_gene.csv`:

```csv
id,promoter_sequence
candidate_1,ACTGACTGACTGACTGACTG
```

Create `one_protein.fasta`:

```fasta
>candidate_1
MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRG
```

Generate the protein embedding:

```bash
cropstressfm embed-protein \
  --fasta one_protein.fasta \
  --output one_protein_esm2.npz \
  --device cuda
```

Run the predictor:

```bash
cropstressfm predict \
  --input one_gene.csv \
  --embeddings one_protein_esm2.npz \
  --output one_gene_prediction.json \
  --device auto
```

The short DNA and protein strings above demonstrate file syntax only. Biological inference requires the correctly oriented promoter and a representative full-length protein sequence.

## Rank a candidate list

CSV output can be sorted directly:

```python
import csv

with open("predictions.csv", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

ranked = sorted(rows, key=lambda row: float(row["stress_gene_probability"]), reverse=True)
for row in ranked[:20]:
    print(row["id"], row["stress_gene_probability"], row["top_stress_type"])
```

For stress-specific ranking, sort by `drought_probability`, `salt_probability` or the corresponding task column. Preserve `*_probability_sd` to identify unstable rankings.
