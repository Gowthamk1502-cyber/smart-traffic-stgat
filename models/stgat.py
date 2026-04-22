"""STGAT-like traffic forecasting model."""

from __future__ import annotations

import torch
from torch import nn

from models.layers import DepthwiseTemporalConvNet, PositionalTimeEncoding, STGATBlock, TemporalAttention


class STGATModel(nn.Module):
    """
    Fuller spatio-temporal graph attention forecaster.

    Input shape:
        (batch, input_steps, num_nodes, input_dim)
    Output shape:
        (batch, output_steps, num_nodes)
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        input_steps: int,
        output_steps: int,
        lstm_hidden_dim: int,
        gat_hidden_dim: int,
        lstm_layers: int = 1,
        temporal_mode: str = "tcn",
        tcn_layers: int = 3,
        num_blocks: int = 2,
        spatial_heads: int = 4,
        temporal_heads: int = 4,
        max_neighbors: int = 24,
        time_embedding_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_encoder = PositionalTimeEncoding(input_dim=2, output_dim=time_embedding_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(1 + time_embedding_dim, lstm_hidden_dim),
            nn.LayerNorm(lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_mode = temporal_mode
        if temporal_mode == "lstm":
            self.temporal_encoder = nn.LSTM(
                input_size=lstm_hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0.0,
                batch_first=True,
            )
        elif temporal_mode == "tcn":
            self.temporal_encoder = DepthwiseTemporalConvNet(
                hidden_dim=lstm_hidden_dim,
                num_layers=tcn_layers,
                dropout=dropout,
            )
        else:
            raise ValueError("temporal_mode must be either 'tcn' or 'lstm'")
        self.blocks = nn.ModuleList(
            [
                STGATBlock(
                    hidden_dim=lstm_hidden_dim,
                    spatial_heads=spatial_heads,
                    temporal_heads=temporal_heads,
                    max_neighbors=max_neighbors,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.temporal_attention = TemporalAttention(lstm_hidden_dim)
        self.spatial_projection = nn.Sequential(
            nn.Linear(lstm_hidden_dim, gat_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.horizon_embedding = nn.Parameter(torch.empty(output_steps, gat_hidden_dim))
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim + gat_hidden_dim, gat_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.horizon_embedding)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
        values = x[..., :1]
        time_features = x[..., 1:]
        encoded_time = self.time_encoder(time_features)
        features = self.input_projection(torch.cat([values, encoded_time], dim=-1))

        batch_size, time_steps, num_nodes, feat_dim = features.shape
        lstm_input = features.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, feat_dim)
        if self.temporal_mode == "lstm":
            temporal_sequence, _ = self.temporal_encoder(lstm_input)
        else:
            temporal_sequence = self.temporal_encoder(lstm_input)
        hidden = temporal_sequence.reshape(batch_size, num_nodes, time_steps, -1).permute(0, 2, 1, 3)

        spatial_weights = None
        for block in self.blocks:
            hidden, spatial_weights = block(hidden, adjacency)

        temporal_context, temporal_weights = self.temporal_attention(hidden.permute(0, 2, 1, 3))
        spatial_context = self.spatial_projection(temporal_context)
        horizon_context = spatial_context.unsqueeze(1) + self.horizon_embedding.view(1, self.output_steps, 1, -1)
        temporal_context = temporal_context.unsqueeze(1).expand(-1, self.output_steps, -1, -1)
        fused = self.dropout(torch.cat([temporal_context, horizon_context], dim=-1))
        preds = self.decoder(fused).squeeze(-1).contiguous()
        return {
            "predictions": preds,
            "temporal_attention": temporal_weights,
            "spatial_attention": spatial_weights,
        }
