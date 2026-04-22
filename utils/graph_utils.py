"""Graph construction helpers for the METR-LA traffic network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphArtifacts:
    adjacency: np.ndarray
    edge_index: np.ndarray
    correlation: np.ndarray


def build_correlation_graph(signal_matrix: np.ndarray, threshold: float = 0.7, top_k: int | None = None) -> GraphArtifacts:
    """
    Builds a symmetric binary adjacency matrix from Pearson correlation.

    Args:
        signal_matrix: Array with shape (num_observations, num_nodes).
        threshold: Minimum absolute correlation required to create an edge.
        top_k: Optional upper bound on outgoing neighbors per node.
    """
    correlation = np.corrcoef(signal_matrix, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)

    adjacency = (np.abs(correlation) >= threshold).astype(np.float32)

    if top_k is not None and top_k > 0:
        pruned = np.zeros_like(adjacency)
        for node_idx in range(adjacency.shape[0]):
            neighbor_scores = np.abs(correlation[node_idx])
            if np.count_nonzero(neighbor_scores) == 0:
                continue
            top_idx = np.argsort(neighbor_scores)[-top_k:]
            pruned[node_idx, top_idx] = adjacency[node_idx, top_idx]
        adjacency = np.maximum(pruned, pruned.T)

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    edge_index = np.vstack(np.nonzero(adjacency)).astype(np.int64)
    return GraphArtifacts(adjacency=adjacency, edge_index=edge_index, correlation=correlation)


def adjacency_to_tensor(adjacency: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(adjacency, dtype=torch.float32, device=device)


def maybe_refresh_graph(
    epoch: int,
    refresh_every: int,
    history_matrix: np.ndarray,
    threshold: float,
    top_k: int | None = None,
) -> GraphArtifacts | None:
    """
    Recomputes the graph from a rolling slice of training history.

    Using a moving window makes the adjacency change over time instead of
    rebuilding the exact same correlation graph each refresh.
    """
    if refresh_every <= 0 or epoch == 0 or epoch % refresh_every != 0:
        return None
    window = max(history_matrix.shape[0] // 4, 512)
    max_start = max(history_matrix.shape[0] - window, 0)
    start = min((epoch // refresh_every) * window // 2, max_start)
    dynamic_slice = history_matrix[start : start + window]
    return build_correlation_graph(dynamic_slice, threshold=threshold, top_k=top_k)
