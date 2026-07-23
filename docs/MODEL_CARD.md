# CropStressFM model card

## Model summary

CropStressFM is a sequence-based ensemble for general abiotic-stress gene prioritization and six stress-type binary scores. It uses promoter DNA and a frozen protein-language-model representation. Species names, gene identifiers, dataset identifiers and target stress names are not encoder inputs.

| Property | Release value |
|---|---|
| Release | 1.0.0 |
| Ensemble | Three independent models, seeds 41, 42 and 43 |
| Promoter input | TSS-oriented, maximum 2,000 bp |
| DNA representation | Exact normalized 5-mer frequencies in four views |
| Protein representation | ESM-2 t33 650M, weighted mean plus global maximum |
| Fusion | Five tokens, four Transformer layers, eight heads |
| Shared dimension | 192 |
| Fusion hidden dimension | 768 |
| Parameters | Approximately 3.30 million per inference member, excluding ESM-2 |
| Main threshold | 0.5395221295 |

## Training design

Training used two supervision stages with separate evidence roles.

The first stage used 21,653 transcriptional-response records from foxtail millet, rice, sorghum and switchgrass. Of these, 18,345 were used for training and 3,308 for validation. This stage supplied weak auxiliary supervision for sequence representation learning.

The functional stage contained 1,441 experimentally or literature-supported T1/T2 stress genes and 5,764 matched background genes from Arabidopsis, rice, wheat, maize and soybean. The train split contained 834 T1/T2 positives and 3,201 backgrounds; the validation split contained 350 positives and 1,288 backgrounds. Arabidopsis, rice and wheat contributed functional training data. Rice and wheat evaluation components were separated by protein-family groups. Maize and soybean functional sets were held out as unseen-species tests.

Background genes represent matched genes without positive evidence in the assembled evidence catalogue. They are not guaranteed biological non-stress genes.

## Locked evaluation

The table reports the frozen three-seed ensemble on high-confidence T1/T2 crop tests. Threshold-dependent metrics use the validation-only main threshold.

| Crop test | Evaluation design | n | Positive | AUROC | AP | Recall | MCC | Top-20% recall | AP lift over prevalence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rice | Held-out protein families | 427 | 90 | 0.844 | 0.642 | 0.733 | 0.426 | 0.567 | 3.05 |
| Wheat | Held-out protein families | 68 | 14 | 0.952 | 0.728 | 1.000 | 0.556 | 0.786 | 3.54 |
| Maize | Unseen species | 500 | 100 | 0.776 | 0.539 | 0.720 | 0.341 | 0.500 | 2.70 |
| Soybean | Unseen species | 265 | 53 | 0.744 | 0.499 | 0.755 | 0.264 | 0.491 | 2.50 |

Top-20% recall measures the fraction of known positives recovered when only the highest-scoring 20% of genes are selected. AP lift divides average precision by the positive prevalence.

## Decision thresholds

| Output | Threshold |
|---|---:|
| General stress gene | 0.539522 |
| Heat | 0.896569 |
| Drought | 0.971635 |
| Salt | 0.896885 |
| Cold | 0.979948 |
| Osmotic | 0.972847 |
| Other abiotic | 0.792956 |

Thresholds were selected from validation predictions before locked evaluation. Stress-type heads are separate one-versus-background binary tasks. Their outputs are not a probability simplex.

## Limitations

Functional evidence is sparse and uneven among species and stress types. Gene-family composition may influence transfer. Promoter definitions depend on annotation quality and TSS choice. ESM-2 embeddings depend on protein isoform selection. The model does not directly represent tissue, developmental stage, dosage, stress duration, chromatin accessibility, environmental interaction or field management.

The general score has stronger supervision than individual type heads. Closely related stress programs can receive similar scores, and multiple type calls are permitted. Use score vectors, seed standard deviations and external biological evidence when selecting experiments.
