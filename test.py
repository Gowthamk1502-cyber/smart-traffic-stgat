"""Test-set entry point for the trained STGAT traffic forecasting model."""

from __future__ import annotations

from pathlib import Path

import yaml

from train import resolve_config_paths
from utils.evaluation import build_terminal_report, test_model


def main() -> None:
    """Run the saved checkpoint on the HuggingFace METR-LA test split."""
    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = resolve_config_paths(yaml.safe_load(f), project_dir)

    metrics = test_model(config)
    print(build_terminal_report(metrics, Path(config["data"]["output_dir"])))


if __name__ == "__main__":
    main()
