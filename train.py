"""Training entry point for the STGAT traffic forecasting project."""

from __future__ import annotations

import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from models.stgat import STGATModel
from utils.data_loader import PreparedData, prepare_dataloaders
from utils.graph_utils import adjacency_to_tensor, maybe_refresh_graph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", mode="w"),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_config_paths(config: dict, base_dir: Path) -> dict:
    """Resolve project-relative paths so scripts work from any shell cwd."""
    resolved = deepcopy(config)
    for section, key in [
        ("data", "cache_dir"),
        ("data", "graph_cache_path"),
        ("data", "output_dir"),
        ("training", "checkpoint_path"),
    ]:
        path = Path(resolved[section][key])
        if not path.is_absolute():
            resolved[section][key] = str(base_dir / path)
    return resolved


def masked_huber_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    congestion_threshold: float | None = None,
    congestion_weight: float = 0.0,
    horizon_weight: float = 1.0,
) -> torch.Tensor:
    """Huber loss over valid readings, optionally weighted for congestion and long horizons."""
    mask = mask.to(dtype=preds.dtype)
    error = preds - targets
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta, dtype=preds.dtype, device=preds.device))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear

    weights = torch.ones_like(loss)
    if congestion_threshold is not None and congestion_weight > 0:
        congestion_mask = (targets <= congestion_threshold).to(dtype=preds.dtype)
        weights = weights + congestion_mask * congestion_weight
    if horizon_weight > 1.0:
        horizon_weights = torch.linspace(1.0, horizon_weight, preds.size(1), device=preds.device, dtype=preds.dtype)
        weights = weights * horizon_weights.view(1, -1, 1)

    weighted_mask = mask * weights
    return torch.sum(loss * weighted_mask) / torch.clamp(weighted_mask.sum(), min=1.0)


def compute_congestion_threshold(train_matrix: np.ndarray, scaler_mean: float, scaler_std: float, quantile: float) -> float:
    """Return the standardized low-speed threshold used for congestion-aware training."""
    valid_values = train_matrix[train_matrix != 0]
    raw_threshold = float(np.percentile(valid_values, quantile))
    return (raw_threshold - scaler_mean) / max(scaler_std, 1e-6)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    if mask is None:
        mask = torch.ones_like(targets)
    mask = mask.to(dtype=preds.dtype, device=preds.device)
    denom = torch.clamp(mask.sum(), min=1.0)
    squared_error = (preds - targets) ** 2
    absolute_error = torch.abs(preds - targets)
    mse = (torch.sum(squared_error * mask) / denom).item()
    mae = (torch.sum(absolute_error * mask) / denom).item()
    rmse = mse ** 0.5
    return {"mae": mae, "rmse": rmse, "mse": mse}


def run_epoch(
    model: STGATModel,
    dataloader,
    adjacency: torch.Tensor,
    criterion: nn.Module | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    loss_config: dict | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_items = 0

    progress = tqdm(dataloader, leave=False, disable=not sys.stderr.isatty())
    loss_config = loss_config or {}
    for batch in progress:
        inputs, targets, mask = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs, adjacency)["predictions"]
                loss = masked_huber_loss(outputs, targets, mask, **loss_config)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        batch_metrics = compute_metrics(outputs.detach(), targets, mask)
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_mae += batch_metrics["mae"] * batch_size
        total_rmse += batch_metrics["rmse"] * batch_size
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "mae": total_mae / max(total_items, 1),
        "rmse": total_rmse / max(total_items, 1),
    }


def save_checkpoint(path: Path, model: STGATModel, optimizer, scaler_state: dict, config: dict, adjacency: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler": scaler_state,
            "config": config,
            "adjacency": adjacency,
        },
        path,
    )


def train_model(config: dict) -> tuple[STGATModel, PreparedData, torch.device]:
    output_dir = Path(config["data"]["output_dir"])
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    set_seed(config["seed"])
    device = resolve_device(config["device"])
    logger.info("Using device: %s", device)

    data_bundle = prepare_dataloaders(config)
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

    adjacency_np = data_bundle.graph.adjacency
    adjacency = adjacency_to_tensor(adjacency_np, device)
    criterion = None
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"].get("lr_factor", 0.5),
        patience=config["training"].get("lr_patience", 3),
    )
    use_amp = bool(config["training"].get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    logger.info("Mixed precision: %s", use_amp)
    congestion_threshold = compute_congestion_threshold(
        train_matrix=data_bundle.train_matrix,
        scaler_mean=data_bundle.scaler.mean,
        scaler_std=data_bundle.scaler.std,
        quantile=config["training"].get("congestion_quantile", 15),
    )
    train_loss_config = {
        "delta": config["training"].get("huber_delta", 1.0),
        "congestion_threshold": congestion_threshold,
        "congestion_weight": config["training"].get("congestion_weight", 0.0),
        "horizon_weight": config["training"].get("horizon_weight", 1.0),
    }
    eval_loss_config = {
        "delta": config["training"].get("huber_delta", 1.0),
    }
    logger.info(
        "Loss focus | congestion_quantile=%s congestion_threshold=%.4f congestion_weight=%.2f horizon_weight=%.2f",
        config["training"].get("congestion_quantile", 15),
        congestion_threshold,
        train_loss_config["congestion_weight"],
        train_loss_config["horizon_weight"],
    )

    best_val_loss = float("inf")
    checkpoint_path = Path(config["training"]["checkpoint_path"])
    refresh_every = config["data"]["refresh_graph_every"]
    validate_every = max(1, int(config["training"].get("validate_every", 1)))
    early_stopping_patience = config["training"].get("early_stopping_patience")
    epochs_without_improvement = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        refreshed_graph = maybe_refresh_graph(
            epoch=epoch,
            refresh_every=refresh_every,
            history_matrix=data_bundle.train_matrix,
            threshold=config["data"]["corr_threshold"],
            top_k=config["data"].get("graph_top_k"),
        )
        if refreshed_graph is not None:
            adjacency_np = refreshed_graph.adjacency
            adjacency = adjacency_to_tensor(adjacency_np, device)
            logger.info("Refreshed dynamic graph at epoch %d", epoch)

        train_metrics = run_epoch(
            model=model,
            dataloader=data_bundle.train_loader,
            adjacency=adjacency,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=config["training"]["grad_clip"],
            scaler=scaler,
            use_amp=use_amp,
            loss_config=train_loss_config,
        )
        should_validate = epoch == 1 or epoch % validate_every == 0 or epoch == config["training"]["epochs"]
        if should_validate:
            val_metrics = run_epoch(
                model=model,
                dataloader=data_bundle.val_loader,
                adjacency=adjacency,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                loss_config=eval_loss_config,
            )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_mae=%.4f train_rmse=%.4f | val_loss=%.4f val_mae=%.4f val_rmse=%.4f",
                epoch,
                config["training"]["epochs"],
                train_metrics["loss"],
                train_metrics["mae"],
                train_metrics["rmse"],
                val_metrics["loss"],
                val_metrics["mae"],
                val_metrics["rmse"],
            )
            scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler_state=data_bundle.scaler.state_dict(),
                    config=config,
                    adjacency=adjacency_np,
                )
                logger.info("Saved new best model to %s", checkpoint_path)
            else:
                epochs_without_improvement += validate_every
                if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                    logger.info("Early stopping after %d epochs without validation improvement", epochs_without_improvement)
                    break
        else:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_mae=%.4f train_rmse=%.4f | validation skipped",
                epoch,
                config["training"]["epochs"],
                train_metrics["loss"],
                train_metrics["mae"],
                train_metrics["rmse"],
            )

    return model, data_bundle, device


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_config_paths(cfg, project_dir)
    train_model(cfg)
