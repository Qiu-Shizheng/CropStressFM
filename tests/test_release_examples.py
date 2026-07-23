import csv
from pathlib import Path

import numpy as np

from cropstressfm import CropStressFM


ROOT = Path(__file__).resolve().parents[1]


def test_release_examples_match_frozen_predictions() -> None:
    with (ROOT / "examples/input/example_genes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(ROOT / "examples/input/example_esm2_embeddings.npz", allow_pickle=False) as payload:
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    predictor = CropStressFM(device="cpu", batch_size=16)
    predictions = predictor.predict(
        [row["promoter_sequence"] for row in rows],
        embeddings,
        ids=[row["id"] for row in rows],
    )
    expected = {
        "ZM00001EB051510": (0.9081193805, "Salt"),
        "Os07g0581700": (0.9777100682, "Cold"),
        "Csa_3G008920": (0.9205991626, "Heat"),
        "TRAESCS4B02G278100": (0.9590619206, "Drought"),
        "GLYMA_05G045800": (0.8299835324, "Salt"),
    }
    for row in predictions:
        probability, stress_type = expected[str(row["id"])]
        assert abs(float(row["stress_gene_probability"]) - probability) < 2e-6
        assert row["top_stress_type"] == stress_type
