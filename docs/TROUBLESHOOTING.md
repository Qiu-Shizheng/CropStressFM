# Troubleshooting

## MKL symbol or threading error

Some mixed Conda/PyTorch installations expose incompatible MKL and OpenMP libraries. Run CropStressFM with sequential MKL threading:

```bash
export MKL_THREADING_LAYER=sequential
export OMP_NUM_THREADS=2
cropstressfm predict --input genes.csv --embeddings embeddings.npz --output predictions.csv
```

A clean virtual environment is preferred when the error persists.

## CUDA out of memory during ESM-2 embedding

Reduce the chunk batch size:

```bash
cropstressfm embed-protein \
  --fasta proteins.fasta \
  --output embeddings.npz \
  --device cuda \
  --batch-size 1 \
  --max-residues 500
```

The pooling result remains residue weighted. A smaller chunk size increases runtime but reduces activation memory.

## ESM-2 cannot download on an offline node

Download `esm2_t33_650M_UR50D.pt` on a networked machine, transfer it to shared storage and use:

```bash
cropstressfm embed-protein \
  --fasta proteins.fasta \
  --output embeddings.npz \
  --checkpoint /path/to/esm2_t33_650M_UR50D.pt \
  --device cuda
```

## Missing embedding identifiers

The prediction CSV and NPZ must use identical IDs. Inspect them with:

```python
import csv
import numpy as np

with open("genes.csv", encoding="utf-8") as handle:
    gene_ids = [row["id"] for row in csv.DictReader(handle)]

with np.load("embeddings.npz", allow_pickle=False) as payload:
    embedding_ids = [str(value) for value in payload["ids"]]

print(set(gene_ids) - set(embedding_ids))
```

## Unexpected prediction after changing annotation release

Confirm that genome, GFF/GTF and protein FASTA came from the same release. Check strand orientation, TSS location, protein isoform and whether identifiers were remapped between releases.

## Stress-type scores are all high

The six tasks are independent and share a strong general stress-gene representation. They are not normalized against each other. Compare each score with its own threshold, retain the ensemble standard deviation and treat the highest type as a prioritization aid.
