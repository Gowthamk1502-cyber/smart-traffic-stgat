"""Evaluation utilities for test-set forecasting."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.stgat import STGATModel
from train import compute_metrics, resolve_device, set_seed, setup_logging
from utils.data_loader import StandardScaler, prepare_dataloaders
from utils.graph_utils import adjacency_to_tensor


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[STGATModel, dict]:
    """
    Load a checkpoint created by this project.

    The checkpoint stores metadata in addition to tensors, so PyTorch 2.6+
    needs weights_only=False for this trusted local artifact.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train the upgraded STGAT first with `python train.py`, then run `python test.py`."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = STGATModel(
        num_nodes=config["model"]["num_nodes"],
        input_dim=config["model"]["input_dim"],
        input_steps=config["model"]["input_steps"],
        output_steps=config["model"]["output_steps"],
        lstm_hidden_dim=config["model"]["lstm_hidden_dim"],
        gat_hidden_dim=config["model"]["gat_hidden_dim"],
        lstm_layers=config["model"]["lstm_layers"],
        temporal_mode=config["model"].get("temporal_mode", "tcn"),
        tcn_layers=config["model"].get("tcn_layers", 3),
        num_blocks=config["model"].get("num_blocks", 2),
        spatial_heads=config["model"].get("spatial_heads", 4),
        temporal_heads=config["model"].get("temporal_heads", 4),
        max_neighbors=config["model"].get("max_neighbors", 24),
        time_embedding_dim=config["model"].get("time_embedding_dim", 16),
        dropout=config["model"]["dropout"],
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "This checkpoint is incompatible with the current full STGAT architecture. "
            "Retrain with `python train.py` so a fresh upgraded checkpoint is saved."
        ) from exc
    model.eval()
    return model, checkpoint


def plot_predictions(predictions: np.ndarray, targets: np.ndarray, output_dir: Path) -> None:
    """Save forecast-vs-actual plots for quick visual inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sensor_idx = 0
    sample_idx = 0
    pred_series = predictions[sample_idx, :, sensor_idx]
    true_series = targets[sample_idx, :, sensor_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(true_series, label="Actual", marker="o")
    plt.plot(pred_series, label="Predicted", marker="x")
    plt.title("Traffic Forecast for Sensor 0")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Traffic Speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "timeseries_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(targets.flatten(), predictions.flatten(), s=6, alpha=0.35)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()


def _masked_values(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return values[masks.astype(bool)]


def _safe_percentage(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def compute_operational_metrics(predictions: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    """Translate regression errors into traffic-operations style metrics."""
    valid_preds = _masked_values(predictions, masks)
    valid_targets = _masked_values(targets, masks)
    errors = np.abs(valid_preds - valid_targets)
    target_mean = float(np.mean(valid_targets))

    metrics = {
        "valid_points": float(valid_targets.size),
        "target_mean": target_mean,
        "within_3": float(np.mean(errors <= 3.0) * 100.0),
        "within_5": float(np.mean(errors <= 5.0) * 100.0),
        "within_10": float(np.mean(errors <= 10.0) * 100.0),
        "normalized_accuracy": max(0.0, 100.0 * (1.0 - float(np.mean(errors)) / max(target_mean, 1e-6))),
    }

    peak_threshold = np.percentile(valid_targets, 85)
    low_speed_threshold = np.percentile(valid_targets, 15)
    peak_mask = (targets >= peak_threshold) & masks.astype(bool)
    congestion_mask = (targets <= low_speed_threshold) & masks.astype(bool)

    for name, segment_mask in [("peak", peak_mask), ("congestion", congestion_mask)]:
        segment_errors = np.abs(predictions[segment_mask] - targets[segment_mask])
        metrics[f"{name}_points"] = float(segment_errors.size)
        metrics[f"{name}_mae"] = float(np.mean(segment_errors)) if segment_errors.size else 0.0
        metrics[f"{name}_within_5"] = float(np.mean(segment_errors <= 5.0) * 100.0) if segment_errors.size else 0.0

    actual_direction = np.sign(targets[:, -1, :] - targets[:, 0, :])
    pred_direction = np.sign(predictions[:, -1, :] - predictions[:, 0, :])
    direction_mask = (masks[:, -1, :] > 0) & (masks[:, 0, :] > 0) & (actual_direction != 0)
    metrics["trend_accuracy"] = _safe_percentage(
        float(np.sum((actual_direction == pred_direction) & direction_mask)),
        float(np.sum(direction_mask)),
    )
    return metrics


def build_terminal_report(metrics: dict[str, float], output_dir: Path) -> str:
    """Create a clear terminal report for traffic-management evaluation."""
    if metrics["mae"] <= 3.0:
        verdict = "EXCELLENT - ready for strong real-time traffic forecasting demos"
    elif metrics["mae"] <= 4.5:
        verdict = "STRONG - practical forecasting quality for traffic monitoring"
    elif metrics["mae"] <= 6.0:
        verdict = "PROMISING - useful baseline, but more tuning can improve reliability"
    else:
        verdict = "NEEDS TUNING - model learned traffic patterns but is not yet deployment-grade"

    return f"""
Smart Traffic STGAT Test Report
============================================================
Operational Verdict : {verdict}

Core Forecast Quality
  MAE                : {metrics['mae']:.4f}
  RMSE               : {metrics['rmse']:.4f}
  MSE                : {metrics['mse']:.4f}
  Normalized Accuracy: {metrics['normalized_accuracy']:.2f}%
  Test Points Scored : {int(metrics['valid_points']):,}

Traffic-Control Reliability
  Within +/-3 speed units : {metrics['within_3']:.2f}%
  Within +/-5 speed units : {metrics['within_5']:.2f}%
  Within +/-10 speed units: {metrics['within_10']:.2f}%
  60-min trend accuracy   : {metrics['trend_accuracy']:.2f}%

Forecast Horizon Stability
  MAE @ 15 minutes : {metrics['horizon_3_mae']:.4f}
  MAE @ 30 minutes : {metrics['horizon_6_mae']:.4f}
  MAE @ 60 minutes : {metrics['horizon_12_mae']:.4f}

Critical Traffic Segments
  Peak-flow MAE             : {metrics['peak_mae']:.4f}
  Peak-flow within +/-5     : {metrics['peak_within_5']:.2f}%
  Congestion-zone MAE       : {metrics['congestion_mae']:.4f}
  Congestion-zone within +/-5: {metrics['congestion_within_5']:.2f}%

Generated Visual Evidence
  {output_dir / 'timeseries_comparison.png'}
  {output_dir / 'predicted_vs_actual.png'}
============================================================
"""


def test_model(config: dict) -> dict[str, float]:
    """Run the trained model on the METR-LA test split."""
    output_dir = Path(config["data"]["output_dir"])
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    set_seed(config["seed"])
    device = resolve_device(config["device"])
    checkpoint_path = Path(config["training"]["checkpoint_path"])

    data_bundle = prepare_dataloaders(config)
    model, checkpoint = load_model(checkpoint_path, device)
    scaler = StandardScaler.from_state_dict(checkpoint["scaler"])
    adjacency = adjacency_to_tensor(checkpoint["adjacency"], device)

    preds_list = []
    targets_list = []
    masks_list = []

    with torch.no_grad():
        for inputs, targets, mask in data_bundle.test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs, adjacency)["predictions"].cpu().numpy()
            preds_list.append(outputs)
            targets_list.append(targets.numpy())
            masks_list.append(mask.numpy())

    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    predictions = scaler.inverse_transform(predictions)
    targets = scaler.inverse_transform(targets)

    metrics = compute_metrics(
        torch.from_numpy(predictions.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
        torch.from_numpy(masks.astype(np.float32)),
    )
    horizon_metrics = {}
    for horizon_idx in [2, 5, 11]:
        horizon_metrics[f"horizon_{horizon_idx + 1}_mae"] = compute_metrics(
            torch.from_numpy(predictions[:, horizon_idx].astype(np.float32)),
            torch.from_numpy(targets[:, horizon_idx].astype(np.float32)),
            torch.from_numpy(masks[:, horizon_idx].astype(np.float32)),
        )["mae"]
    metrics.update(horizon_metrics)
    metrics.update(compute_operational_metrics(predictions, targets, masks))
    plot_predictions(predictions, targets, output_dir=output_dir)
    logger.info(
        "Test metrics | MAE=%.4f RMSE=%.4f MSE=%.4f | MAE@15min=%.4f MAE@30min=%.4f MAE@60min=%.4f",
        metrics["mae"],
        metrics["rmse"],
        metrics["mse"],
        metrics["horizon_3_mae"],
        metrics["horizon_6_mae"],
        metrics["horizon_12_mae"],
    )
    return metrics
