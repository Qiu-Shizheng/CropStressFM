from __future__ import annotations

from pathlib import Path


def plot_predictions(predictions: list[dict[str, object]], path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install the plot extra with: pip install 'cropstressfm[plot]'") from error
    labels = [str(row["id"]) for row in predictions]
    tasks = ["stress_gene", "heat", "drought", "salt", "cold", "osmotic", "other_abiotic"]
    display = ["Stress gene", "Heat", "Drought", "Salt", "Cold", "Osmotic", "Other abiotic"]
    values = []
    for row in predictions:
        values.append([float(row["stress_gene_probability"])] + [float(row[f"{task}_probability"]) for task in tasks[1:]])
    matrix = np.asarray(values, dtype=float)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.7})
    figure, axis = plt.subplots(figsize=(8.2, max(3.4, len(labels) * 0.58 + 1.3)), layout="constrained")
    image = axis.imshow(matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(display)), display, rotation=28, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("CropStressFM predictions", loc="left", fontweight="bold")
    axis.tick_params(length=0)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            axis.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.35 else "#111111", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Predicted probability")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
