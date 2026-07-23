# API reference

## Python

### `CropStressFM`

```python
from cropstressfm import CropStressFM

predictor = CropStressFM(device="auto", batch_size=64)
```

Arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `auto` | `cpu`, `cuda`, `cuda:0` or automatic selection |
| `batch_size` | `int` | `64` | Number of genes per core-model forward pass |

### `predict`

```python
rows = predictor.predict(promoter_sequences, protein_embeddings, ids=gene_ids)
```

| Input | Required shape | Description |
|---|---:|---|
| `promoter_sequences` | `n` strings | TSS-oriented promoter DNA |
| `protein_embeddings` | `(n, 2560)` or `(n, 2561)` | Raw pooled ESM-2 representation, optionally with availability indicator |
| `ids` | `n` strings | Optional output identifiers |

The return value is a list of dictionaries. Output probabilities are Python floats and classification calls are Python booleans.

### `read_fasta`

```python
from cropstressfm import read_fasta

records = read_fasta("proteins.fasta")
```

Returns an insertion-ordered dictionary from the first FASTA header token to the normalized protein sequence.

### `embed_proteins`

```python
from cropstressfm import embed_proteins

ids, embeddings = embed_proteins(
    records,
    device="cuda",
    checkpoint=None,
    max_residues=1000,
    batch_size=1,
)
```

The function returns identifiers and a float32 matrix with 2,560 columns. If `checkpoint` is omitted, it first checks the PyTorch cache and then uses the fair-esm download mechanism.

### `write_embeddings`

```python
from cropstressfm import write_embeddings

write_embeddings("embeddings.npz", ids, embeddings)
```

## Command line

### Model information

```bash
cropstressfm info --device cpu
```

### Prediction

```bash
cropstressfm predict \
  --input genes.csv \
  --embeddings esm2_embeddings.npz \
  --output predictions.csv \
  --plot predictions.png \
  --device auto \
  --batch-size 64
```

The output suffix selects CSV or JSON. Plot generation requires the `plot` optional dependency.

### Protein embedding

```bash
cropstressfm embed-protein \
  --fasta proteins.fasta \
  --output esm2_embeddings.npz \
  --checkpoint /optional/local/checkpoint.pt \
  --device cuda \
  --max-residues 1000 \
  --batch-size 1
```

### Genome input preparation

```bash
cropstressfm prepare \
  --genome genome.fa \
  --annotation annotation.gff3 \
  --proteins proteins.fa \
  --gene-list genes.txt \
  --output-dir prepared \
  --upstream 2000 \
  --feature-type gene \
  --gene-id-key ID
```

`--proteins` and `--gene-list` are optional. When `--gene-id-key` is omitted, the parser checks `gene_id`, `ID`, `Name` and `locus_tag` in that order.
