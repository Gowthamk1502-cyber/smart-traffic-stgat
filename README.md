# Smart Traffic Prediction with Efficient STGAT

An end-to-end PyTorch project for **traffic speed forecasting** on the HuggingFace `witgaw/METR-LA` dataset using an efficient STGAT-style architecture. The project loads raw sensor time-series data, reconstructs graph samples, builds a dynamic correlation graph, trains a spatio-temporal graph attention model, and evaluates traffic prediction quality with practical traffic-operations metrics.

## Project Goal

The goal is to predict future traffic sensor values across the METR-LA road network using both temporal history and spatial sensor relationships.

The model takes:

```text
Input : 12 historical time steps x 207 sensors
Output: 12 future time steps x 207 sensors
```

This corresponds to multi-step traffic forecasting across all sensors, including short-term and 60-minute horizon prediction.

## Key Features

- Loads the HuggingFace `witgaw/METR-LA` dataset with `train`, `validation`, and `test` splits.
- Reconstructs the real dataset schema into tensors shaped `(samples, 12, 207)`.
- Builds a graph dynamically from Pearson correlation between traffic sensors.
- Uses time-of-day cyclic encoding to capture daily traffic patterns.
- Implements an efficient STGAT-style model with sparse top-k graph attention.
- Uses a depthwise dilated temporal convolution encoder for faster GPU training.
- Supports CUDA mixed precision training for lower memory usage and faster epochs.
- Uses masked loss and masked metrics to ignore invalid zero sensor readings.
- Includes congestion-aware and long-horizon weighted training loss.
- Saves the best validation checkpoint automatically.
- Produces a terminal traffic-operations report and prediction visualizations.

## Architecture Overview

The model is designed to balance forecasting quality and practical training speed on modest GPUs.

Pipeline:

```text
Traffic history + time encoding
        |
Input projection
        |
Depthwise dilated temporal encoder
        |
Stacked spatio-temporal graph attention blocks
        |
Temporal attention pooling
        |
Horizon-aware decoder
        |
12-step traffic forecast
```

Core components:

- **Dynamic Graph Construction**: Pearson correlation graph built from training data.
- **Sparse Top-K Graph Attention**: Each sensor attends to selected correlated neighbors instead of the full dense graph.
- **Temporal Conv Encoder**: Fast dilated TCN-style encoder for historical traffic patterns.
- **Temporal Attention**: Learns which historical time steps matter most.
- **Congestion-Aware Loss**: Gives extra weight to low-speed congested regimes.
- **Horizon Weighting**: Gives more importance to later forecast steps.

## Project Structure

```text
smart-traffic-stgat/
|-- config.yaml
|-- requirements.txt
|-- train.py
|-- test.py
|-- data/
|   `-- .gitkeep
|-- models/
|   |-- __init__.py
|   |-- layers.py
|   `-- stgat.py
|-- outputs/
|   `-- .gitkeep
|-- utils/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- evaluation.py
|   `-- graph_utils.py
`-- README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build that matches your system. This project was tested with:

```text
PyTorch 2.11.0+cu130
NVIDIA RTX 3050 Laptop GPU
```

## Training

Run:

```bash
python train.py
```

Training will:

- Download/load the METR-LA dataset.
- Convert HuggingFace rows into graph time-series tensors.
- Normalize traffic values using training statistics.
- Build/cache the correlation graph.
- Train the efficient STGAT model.
- Validate every configured interval.
- Save the best checkpoint to `outputs/best_full_stgat.pt`.

The checkpoint is intentionally not committed to GitHub. After cloning the repo, train first before running tests.

## Testing

After training, run:

```bash
python test.py
```

The test script loads `outputs/best_full_stgat.pt`, evaluates the METR-LA test split, and prints a traffic-operations report.

Example report format:

```text
Smart Traffic STGAT Test Report
============================================================
Operational Verdict : STRONG - practical forecasting quality for traffic monitoring

Core Forecast Quality
  MAE                : 3.9555
  RMSE               : 6.9001
  MSE                : 47.6116
  Normalized Accuracy: 93.15%

Traffic-Control Reliability
  Within +/-3 speed units : 63.39%
  Within +/-5 speed units : 78.42%
  Within +/-10 speed units: 91.15%

Forecast Horizon Stability
  MAE @ 15 minutes : 3.3320
  MAE @ 30 minutes : 3.9660
  MAE @ 60 minutes : 4.8551
============================================================
```

Generated plots:

```text
outputs/timeseries_comparison.png
outputs/predicted_vs_actual.png
```

## Results

Recent METR-LA test run:

```text
MAE                : 3.9555
RMSE               : 6.9001
MSE                : 47.6116
Normalized Accuracy: 93.15%
MAE @ 15 minutes   : 3.3320
MAE @ 30 minutes   : 3.9660
MAE @ 60 minutes   : 4.8551
```

Operational segment performance:

```text
Peak-flow MAE              : 2.9478
Peak-flow within +/-5      : 89.86%
Congestion-zone MAE        : 8.1796
Congestion-zone within +/-5: 47.87%
```

Interpretation:

- The model is strong overall for traffic forecasting and peak-flow monitoring.
- Congestion forecasting remains the hardest regime and is explicitly targeted with weighted training loss.
- The project is best described as a practical research-grade STGAT-style traffic forecasting pipeline.

## Configuration

Main hyperparameters are stored in `config.yaml`.

Important settings:

```yaml
model:
  temporal_mode: tcn
  lstm_hidden_dim: 64
  gat_hidden_dim: 64
  num_blocks: 2
  max_neighbors: 12

training:
  epochs: 50
  mixed_precision: true
  congestion_weight: 1.5
  horizon_weight: 1.3
  early_stopping_patience: 10
```

## Reproducibility

The project sets random seeds and keeps configuration centralized. Some GPU operations may still have small nondeterministic differences depending on CUDA, PyTorch, and hardware.

## GitHub Notes

The repository excludes generated artifacts:

```text
data/hf_cache/
data/cached_graph.npz
outputs/*.pt
outputs/*.log
outputs/*.png
__pycache__/
```

Users should regenerate these by running:

```bash
python train.py
python test.py
```

## License

This project is intended for research and educational use.
