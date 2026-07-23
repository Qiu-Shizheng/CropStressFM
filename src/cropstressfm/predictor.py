from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .assets import load_config, weights_directory
from .features import build_promoter_features
from .model import CropStressFMNetwork


class CropStressFM:
    def __init__(self, device: str = "auto", batch_size: int = 64) -> None:
        self.config = load_config()
        self.device = self._resolve_device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with np.load(weights_directory() / "esm2_scaler.npz") as scaler:
            self.esm_mean = np.asarray(scaler["mean"], dtype=np.float32)
            self.esm_scale = np.asarray(scaler["scale"], dtype=np.float32)
        self.models: list[tuple[int, CropStressFMNetwork, dict[str, dict[str, torch.Tensor]]]] = []
        for seed in self.config["seeds"]:
            payload = torch.load(
                weights_directory() / f"cropstressfm_seed{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
            network = CropStressFMNetwork(**self.config["architecture"])
            network.load_state_dict(payload["state_dict"], strict=True)
            network.eval().to(self.device)
            heads = {
                name: {key: value.to(self.device) for key, value in parameters.items()}
                for name, parameters in payload["stress_heads"].items()
            }
            self.models.append((int(seed), network, heads))

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(device)

    def _prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("Protein embeddings must be a two-dimensional array")
        raw_dim = int(self.config["input"]["esm2_raw_dimension"])
        model_dim = int(self.config["input"]["esm2_model_dimension"])
        if values.shape[1] == raw_dim:
            values = np.hstack([values, np.ones((len(values), 1), dtype=np.float32)])
        if values.shape[1] != model_dim:
            raise ValueError(f"Expected {raw_dim} or {model_dim} ESM-2 features, received {values.shape[1]}")
        if not np.isfinite(values).all():
            raise ValueError("Protein embeddings contain non-finite values")
        scale = np.where(self.esm_scale > 0, self.esm_scale, 1.0)
        return ((values - self.esm_mean) / scale).astype(np.float32)

    def predict(
        self,
        promoter_sequences: Sequence[str],
        protein_embeddings: np.ndarray,
        ids: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        promoters = [str(sequence) for sequence in promoter_sequences]
        if not promoters:
            raise ValueError("At least one promoter sequence is required")
        if ids is None:
            identifiers = [f"gene_{index + 1}" for index in range(len(promoters))]
        else:
            identifiers = [str(value) for value in ids]
        if len(identifiers) != len(promoters):
            raise ValueError("ids and promoter_sequences must have the same length")
        protein = self._prepare_embeddings(protein_embeddings)
        if len(protein) != len(promoters):
            raise ValueError("Protein embeddings and promoter_sequences must have the same number of rows")
        kmer = build_promoter_features(
            promoters,
            k=int(self.config["input"]["k"]),
            maximum_length=int(self.config["input"]["promoter_maximum_bp"]),
        )
        stress_names = list(self.config["stress_types"])
        probabilities = []
        type_probabilities = []
        gate_values = []
        with torch.inference_mode():
            for _, network, heads in self.models:
                seed_any = []
                seed_types = []
                seed_gates = []
                for start in range(0, len(promoters), self.batch_size):
                    stop = min(start + self.batch_size, len(promoters))
                    kmer_batch = torch.as_tensor(kmer[start:stop], dtype=torch.float32, device=self.device)
                    protein_batch = torch.as_tensor(protein[start:stop], dtype=torch.float32, device=self.device)
                    shared, gates = network.encode(kmer_batch, protein_batch)
                    seed_any.append(torch.sigmoid(network.any_head(shared).squeeze(-1)).cpu())
                    seed_types.append(
                        torch.stack(
                            [
                                torch.sigmoid(F.linear(shared, heads[name]["weight"], heads[name]["bias"]).squeeze(-1))
                                for name in stress_names
                            ],
                            dim=1,
                        ).cpu()
                    )
                    seed_gates.append(gates.cpu())
                probabilities.append(torch.cat(seed_any).numpy())
                type_probabilities.append(torch.cat(seed_types).numpy())
                gate_values.append(torch.cat(seed_gates).numpy())
        any_array = np.stack(probabilities, axis=0)
        type_array = np.stack(type_probabilities, axis=0)
        gate_array = np.stack(gate_values, axis=0)
        any_mean = any_array.mean(axis=0)
        any_sd = any_array.std(axis=0, ddof=1)
        type_mean = type_array.mean(axis=0)
        type_sd = type_array.std(axis=0, ddof=1)
        gate_mean = gate_array.mean(axis=0)
        thresholds = self.config["thresholds"]
        display = self.config["stress_display_names"]
        gate_names = self.config["gate_names"]
        results = []
        for row, identifier in enumerate(identifiers):
            top_index = int(np.argmax(type_mean[row]))
            top_key = stress_names[top_index]
            item: dict[str, object] = {
                "id": identifier,
                "stress_gene_probability": float(any_mean[row]),
                "stress_gene_probability_sd": float(any_sd[row]),
                "stress_gene_prediction": bool(any_mean[row] >= float(thresholds["stress_gene"])),
                "stress_gene_threshold": float(thresholds["stress_gene"]),
                "top_stress_type": str(display[top_key]),
                "top_stress_probability": float(type_mean[row, top_index]),
            }
            for column, key in enumerate(stress_names):
                item[f"{key}_probability"] = float(type_mean[row, column])
                item[f"{key}_probability_sd"] = float(type_sd[row, column])
                item[f"{key}_prediction"] = bool(type_mean[row, column] >= float(thresholds[key]))
                item[f"{key}_threshold"] = float(thresholds[key])
            for column, name in enumerate(gate_names):
                item[f"gate_{name}"] = float(gate_mean[row, column])
            results.append(item)
        return results

    def model_info(self) -> dict[str, object]:
        backbone_count = sum(parameter.numel() for _, model, _ in self.models for parameter in model.parameters())
        head_count = sum(parameter.numel() for _, _, heads in self.models for head in heads.values() for parameter in head.values())
        return {
            "name": self.config["name"],
            "version": self.config["version"],
            "device": str(self.device),
            "ensemble_seeds": list(self.config["seeds"]),
            "loaded_backbone_parameters": int(backbone_count),
            "loaded_stress_head_parameters": int(head_count),
            "loaded_parameters": int(backbone_count + head_count),
            "input": self.config["input"],
            "stress_types": [self.config["stress_display_names"][name] for name in self.config["stress_types"]],
        }
