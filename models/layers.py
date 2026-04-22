"""Reusable neural network layers for the STGAT-like model."""

from __future__ import annotations

import math

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Learns the relative importance of historical time steps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor with shape (batch, nodes, time, hidden_dim).

        Returns:
            context: Weighted temporal summary with shape (batch, nodes, hidden_dim).
            weights: Attention weights with shape (batch, nodes, time).
        """
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=2)
        return context, weights


class GraphAttentionLayer(nn.Module):
    """Dense GAT-style layer that consumes a binary adjacency matrix."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, alpha: float = 0.2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(out_dim))
        self.attn_dst = nn.Parameter(torch.empty(out_dim))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features with shape (batch, nodes, in_dim).
            adjacency: Binary adjacency with shape (nodes, nodes).

        Returns:
            node_embeddings: Updated node features with shape (batch, nodes, out_dim).
            attention: Pairwise attention scores with shape (batch, nodes, nodes).
        """
        h = self.linear(x)
        src_scores = torch.matmul(h, self.attn_src)
        dst_scores = torch.matmul(h, self.attn_dst)
        logits = self.leaky_relu(src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2))

        mask = adjacency.bool().unsqueeze(0)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(masked_logits, dim=-1)
        attention = self.dropout(attention)
        node_embeddings = torch.matmul(attention, h)
        return node_embeddings, attention


class MultiHeadGraphAttentionLayer(nn.Module):
    """Multi-head dense graph attention for batched spatio-temporal tensors."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features with shape (batch, nodes, dim) or (batch, time, nodes, dim).
            adjacency: Binary adjacency with shape (nodes, nodes).
        """
        original_shape = x.shape
        has_time_dim = x.dim() == 4
        if has_time_dim:
            batch_size, time_steps, num_nodes, _ = x.shape
            x = x.reshape(batch_size * time_steps, num_nodes, -1)

        h = self.proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)
        src_scores = torch.sum(h * self.attn_src.view(1, self.num_heads, 1, self.head_dim), dim=-1)
        dst_scores = torch.sum(h * self.attn_dst.view(1, self.num_heads, 1, self.head_dim), dim=-1)
        logits = self.leaky_relu(src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2))

        mask = adjacency.bool().view(1, 1, adjacency.size(0), adjacency.size(1))
        logits = logits.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, h)
        out = out.permute(0, 2, 1, 3).reshape(x.size(0), x.size(1), -1)
        out = self.out_proj(out)

        if has_time_dim:
            out = out.reshape(original_shape[0], original_shape[1], original_shape[2], -1)
            attention = attention.reshape(original_shape[0], original_shape[1], self.num_heads, original_shape[2], original_shape[2])
            attention_summary = attention.mean(dim=2)
        else:
            attention_summary = attention.mean(dim=1)

        return out, attention_summary


class SparseTopKGraphAttentionLayer(nn.Module):
    """Custom sparse graph attention that only attends to top-k graph neighbors."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        max_neighbors: int = 24,
        dropout: float = 0.0,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.max_neighbors = max_neighbors
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        has_time_dim = x.dim() == 4
        if has_time_dim:
            batch_size, time_steps, num_nodes, _ = x.shape
            x = x.reshape(batch_size * time_steps, num_nodes, -1)

        num_nodes = adjacency.size(0)
        k = min(self.max_neighbors, num_nodes)
        neighbor_idx = torch.topk(adjacency, k=k, dim=-1).indices
        neighbor_mask = torch.gather(adjacency, 1, neighbor_idx).bool()

        h = self.proj(x).view(x.size(0), num_nodes, self.num_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)
        src_scores = torch.sum(h * self.attn_src.view(1, self.num_heads, 1, self.head_dim), dim=-1)
        dst_scores = torch.sum(h * self.attn_dst.view(1, self.num_heads, 1, self.head_dim), dim=-1)

        h_neighbors = h[:, :, neighbor_idx, :]
        dst_neighbors = dst_scores[:, :, neighbor_idx]
        logits = self.leaky_relu(src_scores.unsqueeze(-1) + dst_neighbors)
        logits = logits.masked_fill(~neighbor_mask.view(1, 1, num_nodes, k), float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)

        out = torch.sum(attention.unsqueeze(-1) * h_neighbors, dim=-2)
        out = out.permute(0, 2, 1, 3).reshape(x.size(0), num_nodes, -1)
        out = self.out_proj(out)

        if has_time_dim:
            out = out.reshape(original_shape[0], original_shape[1], original_shape[2], -1)
            attention_summary = attention.mean(dim=1).reshape(original_shape[0], original_shape[1], num_nodes, k)
        else:
            attention_summary = attention.mean(dim=1)

        return out, attention_summary


class STGATBlock(nn.Module):
    """One full spatio-temporal attention block with residual normalization."""

    def __init__(
        self,
        hidden_dim: int,
        spatial_heads: int,
        temporal_heads: int,
        max_neighbors: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_attention = SparseTopKGraphAttentionLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=spatial_heads,
            max_neighbors=max_neighbors,
            dropout=dropout,
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=temporal_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_out, spatial_weights = self.spatial_attention(x, adjacency)
        x = self.spatial_norm(x + self.dropout(spatial_out))

        batch_size, time_steps, num_nodes, hidden_dim = x.shape
        temporal_input = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, hidden_dim)
        temporal_out, _ = self.temporal_attention(temporal_input, temporal_input, temporal_input, need_weights=False)
        temporal_out = temporal_out.reshape(batch_size, num_nodes, time_steps, hidden_dim).permute(0, 2, 1, 3)
        x = self.temporal_norm(x + self.dropout(temporal_out))
        x = self.ffn_norm(x + self.dropout(self.feed_forward(x)))
        return x, spatial_weights


class DepthwiseTemporalConvNet(nn.Module):
    """Fast temporal encoder using depthwise-separable dilated convolutions."""

    def __init__(self, hidden_dim: int, num_layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            padding = dilation
            layers.extend(
                [
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden_dim,
                    ),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape (batch * nodes, time, hidden_dim).
        """
        residual = x
        y = x.transpose(1, 2)
        y = self.net(y)
        y = y[..., : x.size(1)].transpose(1, 2)
        return self.norm(residual + y)


class PositionalTimeEncoding(nn.Module):
    """Optional learnable projection for cyclic time-of-day features."""

    def __init__(self, input_dim: int = 2, output_dim: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.scale = math.sqrt(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) / self.scale
