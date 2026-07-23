# Input preparation

Input quality is the main determinant of whether a CropStressFM prediction is interpretable. The promoter, annotation and protein sequence must refer to the same gene model release.

## Input contract

CropStressFM requires two aligned biological inputs per gene:

1. A promoter DNA sequence in transcriptional orientation, with the TSS-proximal base at the right end.
2. A 2,560-dimensional ESM-2 representation of one representative protein isoform.

An identifier aligns these two inputs. It is never encoded by the model. A species label is not required.

## Starting files

Use a coordinated set of files from one annotation release:

- reference genome FASTA
- gene annotation in GFF3 or GTF format
- representative or longest protein FASTA
- optional plain-text list of target gene identifiers

Chromosome names must match between the genome FASTA and annotation. Gene identifiers must match between the selected annotation attribute and protein FASTA headers.

## Automated promoter extraction

```bash
cropstressfm prepare \
  --genome reference_genome.fa \
  --annotation genes.gff3 \
  --proteins representative_proteins.fa \
  --gene-list target_gene_ids.txt \
  --output-dir prepared_inputs \
  --upstream 2000 \
  --feature-type gene \
  --gene-id-key ID
```

For each annotated gene, the command:

1. Locates the annotated TSS from gene coordinates and strand.
2. Selects up to 2,000 upstream bases.
3. Clips the interval at the closest upstream gene boundary to avoid including an adjacent annotated gene body.
4. Reverse-complements minus-strand intervals.
5. Writes the TSS-proximal base at the right end of the final sequence.

If the annotation stores genes under another feature type or attribute, change `--feature-type` and `--gene-id-key`. Common GFF3 identifiers use `ID`; common GTF files use `gene_id`.

## Experimentally defined promoters

If a curated promoter or experimentally supported TSS is available, it may be supplied directly in `genes.csv`. Apply these rules:

- retain the original 5-prime to 3-prime transcriptional orientation
- place the TSS-proximal base at the right end
- include only `A`, `C`, `G`, `T` and optional `N`
- keep the available sequence even if it is shorter than 2,000 bp
- sequences longer than 2,000 bp are accepted, but only the rightmost 2,000 bp are used

Ambiguous characters are converted to `N`. Any 5-mer that overlaps an `N` is excluded from the frequency calculation.

## Protein selection

Use a biologically representative protein isoform. When no canonical isoform is designated, a reproducible longest-protein rule is acceptable. Record that choice in the analysis metadata.

FASTA headers must begin with the identifier in `genes.csv`:

```fasta
>gene_001 optional description
MAEAPQTVEELKQLAAAGVEVVVDD
```

The parser uses the first whitespace-delimited token. Terminal `*` characters are removed. Duplicate identifiers should be resolved before embedding.

## Generate ESM-2 representations

```bash
cropstressfm embed-protein \
  --fasta prepared_inputs/proteins.fasta \
  --output prepared_inputs/esm2_embeddings.npz \
  --device cuda \
  --max-residues 1000 \
  --batch-size 1
```

The embedding output contains:

| NPZ key | Shape | Description |
|---|---:|---|
| `ids` | `(n,)` | FASTA identifiers in embedding order |
| `embeddings` | `(n, 2560)` | ESM-2 mean and maximum pooled representation |
| `model_name` | scalar | Protein encoder identity |
| `pooling` | scalar | Pooling method |

Long proteins are processed as non-overlapping chunks. The mean is weighted by the number of residues in each chunk, and the maximum is taken across all residues.

## Quality-control checklist

- Genome and annotation releases are identical.
- FASTA chromosome names exactly match the GFF/GTF sequence names.
- Every prediction row has one unique identifier.
- Every identifier has one promoter and one protein embedding.
- Minus-strand promoters were reverse-complemented.
- The TSS-proximal base is at the right end.
- Promoter sequences are not genomic intervals copied in mixed strand orientations.
- Protein isoform selection is documented and reproducible.
- Empty or extremely short promoter intervals are reviewed.
- The embedding NPZ identifier order is checked; CropStressFM will realign by ID when `ids` is present.

## External tools

Large production studies may use dedicated genome-annotation utilities such as BEDTools or AGAT to build promoter intervals. The same orientation and clipping contract must be preserved. Export the final sequence into the two-column `id,promoter_sequence` format before prediction.
