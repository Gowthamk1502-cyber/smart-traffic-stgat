"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from utils.graph_utils import GraphArtifacts, build_correlation_graph


LOGGER = logging.getLogger(__name__)


class StandardScaler:
    """Simple standard scaler for NumPy arrays."""

    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / self.std

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return array * self.std + self.mean

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> "StandardScaler":
        return cls(mean=state["mean"], std=state["std"])


class TrafficSequenceDataset(Dataset):
    """Torch dataset for normalized traffic sequences and time features."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        time_of_day: np.ndarray,
        scaler: StandardScaler,
    ) -> None:
        self.x = scaler.transform(x).astype(np.float32)
        self.y = scaler.transform(y).astype(np.float32)
        self.mask = (y != 0).astype(np.float32)
        self.time_of_day = time_of_day.astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_values = self.x[idx]
        target = self.y[idx]
        phase = 2 * np.pi * self.time_of_day[idx]
        time_features = np.stack([np.sin(phase), np.cos(phase)], axis=-1)
        time_features = np.repeat(time_features[:, None, :], x_values.shape[1], axis=1)
        inputs = np.concatenate([x_values[..., None], time_features], axis=-1).astype(np.float32)
        return torch.from_numpy(inputs), torch.from_numpy(target), torch.from_numpy(self.mask[idx])


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    graph: GraphArtifacts
    train_matrix: np.ndarray


def _split_to_numpy(split) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = len(set(split["node_id"][:207]))
    num_samples = split.num_rows // num_nodes

    history_cols = [f"x_t-{step}_d0" for step in range(11, 0, -1)] + ["x_t+0_d0"]
    target_cols = [f"y_t+{step}_d0" for step in range(1, 13)]
    time_cols = [f"x_t-{step}_d1" for step in range(11, 0, -1)] + ["x_t+0_d1"]
    table = split.data

    x = np.stack(
        [np.asarray(table.column(column), dtype=np.float32).reshape(num_samples, num_nodes) for column in history_cols],
        axis=1,
    )
    y = np.stack(
        [np.asarray(table.column(column), dtype=np.float32).reshape(num_samples, num_nodes) for column in target_cols],
        axis=1,
    )
    time_of_day = np.stack(
        [np.asarray(table.column(column), dtype=np.float32).reshape(num_samples, num_nodes)[:, 0] for column in time_cols],
        axis=1,
    )
    return x, y, time_of_day


def prepare_dataloaders(config: dict) -> PreparedData:
    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    cache_dir = Path(data_cfg["cache_dir"])
    graph_cache_path = Path(data_cfg["graph_cache_path"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataset %s", dataset_name)
    ds = load_dataset(dataset_name, cache_dir=str(cache_dir))
    LOGGER.info("Converting HuggingFace splits into graph tensors")

    train_x, train_y, train_time = _split_to_numpy(ds["train"])
    val_x, val_y, val_time = _split_to_numpy(ds["validation"])
    test_x, test_y, test_time = _split_to_numpy(ds["test"])
    LOGGER.info(
        "Prepared arrays | train=%s val=%s test=%s",
        train_x.shape,
        val_x.shape,
        test_x.shape,
    )

    scaler = StandardScaler(mean=train_x.mean(), std=train_x.std())
    train_matrix = train_x.reshape(-1, train_x.shape[-1])
    if graph_cache_path.exists():
        cached = np.load(graph_cache_path)
        graph = GraphArtifacts(
            adjacency=cached["adjacency"],
            edge_index=cached["edge_index"],
            correlation=cached["correlation"],
        )
        LOGGER.info("Loaded cached graph from %s", graph_cache_path)
    else:
        graph = build_correlation_graph(
            train_matrix,
            threshold=data_cfg["corr_threshold"],
            top_k=data_cfg.get("graph_top_k"),
        )
        np.savez_compressed(
            graph_cache_path,
            adjacency=graph.adjacency,
            edge_index=graph.edge_index,
            correlation=graph.correlation,
        )
        LOGGER.info("Cached graph artifacts at %s", graph_cache_path)

    train_ds = TrafficSequenceDataset(
        train_x,
        train_y,
        time_of_day=train_time,
        scaler=scaler,
    )
    val_ds = TrafficSequenceDataset(
        val_x,
        val_y,
        time_of_day=val_time,
        scaler=scaler,
    )
    test_ds = TrafficSequenceDataset(
        test_x,
        test_y,
        time_of_day=test_time,
        scaler=scaler,
    )

    loader_kwargs = {
        "batch_size": data_cfg["batch_size"],
        "num_workers": data_cfg["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        graph=graph,
        train_matrix=train_matrix,
    )
