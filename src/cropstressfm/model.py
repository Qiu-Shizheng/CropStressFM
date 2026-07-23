from __future__ import annotations

import torch
from torch import nn


class ViewwiseKmerEncoder(nn.Module):
    def __init__(self, total_kmer_dim: int, n_views: int, d_model: int, rank: int, dropout: float) -> None:
        super().__init__()
        if n_views <= 0 or total_kmer_dim % n_views:
            raise ValueError("Invalid viewwise k-mer dimensions")
        self.n_views = int(n_views)
        self.per_view_dim = int(total_kmer_dim // n_views)
        self.rank = int(rank)
        self.weight = nn.Parameter(torch.empty(self.per_view_dim, self.rank))
        self.bias = nn.Parameter(torch.zeros(self.rank))
        self.post = nn.Sequential(
            nn.LayerNorm(self.rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.rank, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shaped = values.reshape(values.shape[0], self.n_views, self.per_view_dim)
        return self.post(shaped @ self.weight + self.bias)


class CropStressFMNetwork(nn.Module):
    def __init__(
        self,
        kmer_dim: int = 4096,
        numeric_dim: int = 2561,
        d_model: int = 192,
        hidden_dim: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.4,
        kmer_rank: int = 96,
        n_views: int = 4,
    ) -> None:
        super().__init__()
        self.n_views = int(n_views)
        self.kmer_encoder = ViewwiseKmerEncoder(kmer_dim, n_views, d_model, kmer_rank, dropout)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_embed = nn.Parameter(torch.zeros(4, d_model))
        self.view_embed = nn.Parameter(torch.zeros(n_views, d_model))
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.any_head = nn.Linear(hidden_dim // 2, 1)

    def encode(self, kmer: torch.Tensor, protein: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        promoter = self.kmer_encoder(kmer)
        promoter = promoter + self.modality_embed[1].view(1, 1, -1) + self.view_embed.view(1, self.n_views, -1)
        protein_token = (self.numeric_encoder(protein) + self.modality_embed[3]).unsqueeze(1)
        tokens = torch.cat([promoter, protein_token], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1) + self.modality_embed[0].view(1, 1, -1)
        encoded = self.transformer(torch.cat([cls, tokens], dim=1))
        cls_out = encoded[:, 0]
        body = encoded[:, 1:]
        gates = torch.softmax(self.gate(body).squeeze(-1), dim=1)
        pooled = (body * gates.unsqueeze(-1)).sum(dim=1)
        mean_pool = body.mean(dim=1)
        max_pool = body.max(dim=1).values
        shared = self.shared(torch.cat([cls_out, pooled, mean_pool, max_pool], dim=1))
        return shared, gates

    def forward(self, kmer: torch.Tensor, protein: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared, gates = self.encode(kmer, protein)
        return self.any_head(shared).squeeze(-1), gates
