from __future__ import annotations

import json
import math
import random
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path("data") / "processed_1h" / "aligned_dataset.csv"
RESULT_DIR = Path("results") / "stgnn_1h_forecast"
FIGURE_DIR = RESULT_DIR / "figures"

INPUT_HOURS = 48
FORECAST_HOURS = 24
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
DROPOUT = 0.1
PATIENCE = 10
RANDOM_SEED = 2026
EPS = 1e-6

TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-09-30T23:00:00Z")
TEST_START = pd.Timestamp("2023-10-01T00:00:00Z")
TEST_END = pd.Timestamp("2023-12-31T23:00:00Z")

REPRESENTATIVE_ORIGINS = [
    pd.Timestamp("2023-10-15T00:00:00Z"),
    pd.Timestamp("2023-11-20T00:00:00Z"),
    pd.Timestamp("2023-12-10T00:00:00Z"),
]


@dataclass
class WindowDataset:
    x: np.ndarray
    y: np.ndarray
    origins: pd.DatetimeIndex
    target_timestamps: list[pd.DatetimeIndex]


class GraphConv(nn.Module):
    """Dense graph convolution with a fixed normalized adjacency matrix."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, node, channel]
        x_graph = torch.einsum("ij,btjc->btic", adjacency, x)
        return self.linear(x_graph)


class STGNNForecast(nn.Module):
    """Simple STGNN: graph convolution per hour + node-wise temporal GRU."""

    def __init__(
        self,
        num_nodes: int,
        hidden_size: int,
        dropout: float,
        horizon: int,
        adjacency: np.ndarray,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.register_buffer("adjacency", torch.from_numpy(adjacency.astype(np.float32)))
        self.gconv_in = GraphConv(1, hidden_size)
        self.gconv_out = GraphConv(hidden_size, hidden_size)
        self.temporal = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_hours, num_nodes]
        h = x.unsqueeze(-1)
        h = F.gelu(self.gconv_in(h, self.adjacency))
        h = self.dropout(h)
        h = F.gelu(self.gconv_out(h, self.adjacency))
        batch, time, nodes, hidden = h.shape
        h = h.permute(0, 2, 1, 3).reshape(batch * nodes, time, hidden)
        _, last = self.temporal(h)
        node_state = last[-1].reshape(batch, nodes, hidden)
        out = self.head(self.dropout(node_state))
        return out.permute(0, 2, 1).contiguous()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in FIGURE_DIR.glob("*.png"):
        old_png.unlink()


def load_dataset() -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "timestamp" not in df.columns:
        raise ValueError("aligned_dataset.csv must contain timestamp")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("aligned_dataset.csv contains invalid timestamps")

    netload_cols = [f"netload_bus_{i}" for i in range(1, 34)]
    # This script now focuses on source/load forecasting only.
    # Electricity price is not a prediction target; it is used later as a known
    # exogenous dispatch parameter by the optimization scripts.
    feature_cols = netload_cols
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"aligned_dataset.csv missing required columns: {missing}")
    values = df[feature_cols].copy()
    for col in feature_cols:
        values[col] = pd.to_numeric(values[col], errors="coerce")
    if values.isna().any().any():
        raise ValueError("Feature columns contain NaN or non-numeric values")
    values.index = pd.DatetimeIndex(timestamps, name="timestamp")
    values = values.sort_index()
    return values, values.index, feature_cols


def compute_scaler(values: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    train_rows = values.loc[(values.index >= TRAIN_START) & (values.index <= TRAIN_END)]
    if train_rows.empty:
        raise ValueError("No rows found in training period for scaler statistics")
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0, ddof=0).replace(0.0, 1.0)
    std = std.mask(std < EPS, 1.0)
    return mean, std


def constant_zero_columns(values: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Columns that are physically zero in the dataset should stay zero."""
    return [col for col in feature_cols if float(values[col].abs().max()) <= EPS]


def compute_pearson_adjacency(values: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute abs-Pearson adjacency on the training period and GCN-normalize it."""
    train_rows = values.loc[(values.index >= TRAIN_START) & (values.index <= TRAIN_END), feature_cols]
    if train_rows.empty:
        raise ValueError("No rows found in training period for Pearson adjacency")
    corr = train_rows.corr(method="pearson").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    adj = np.abs(corr.to_numpy(dtype=np.float32))
    np.fill_diagonal(adj, 1.0)
    degree = np.sum(adj, axis=1)
    degree = np.maximum(degree, EPS)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    adj_norm = (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)
    return pd.DataFrame(adj, index=feature_cols, columns=feature_cols), adj_norm


def build_windows(
    standardized_values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> WindowDataset:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    origins: list[pd.Timestamp] = []
    target_times: list[pd.DatetimeIndex] = []

    max_start = len(timestamps) - INPUT_HOURS - FORECAST_HOURS + 1
    for start_idx in range(max_start):
        input_end = start_idx + INPUT_HOURS
        target_end_idx = input_end + FORECAST_HOURS
        y_times = timestamps[input_end:target_end_idx]
        if y_times[0] < target_start or y_times[-1] > target_end:
            continue
        x_list.append(standardized_values[start_idx:input_end])
        y_list.append(standardized_values[input_end:target_end_idx])
        origins.append(timestamps[input_end - 1])
        target_times.append(y_times)

    if not x_list:
        raise ValueError(f"No windows found for target period {target_start} to {target_end}")
    return WindowDataset(
        x=np.stack(x_list).astype(np.float32),
        y=np.stack(y_list).astype(np.float32),
        origins=pd.DatetimeIndex(origins, name="forecast_origin"),
        target_timestamps=target_times,
    )


def split_train_validation(dataset: WindowDataset, validation_ratio: float = 0.2) -> tuple[WindowDataset, WindowDataset]:
    n_total = len(dataset.x)
    n_val = max(1, int(math.ceil(n_total * validation_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Not enough training windows for validation split")

    train = WindowDataset(
        x=dataset.x[:n_train],
        y=dataset.y[:n_train],
        origins=dataset.origins[:n_train],
        target_timestamps=dataset.target_timestamps[:n_train],
    )
    val = WindowDataset(
        x=dataset.x[n_train:],
        y=dataset.y[n_train:],
        origins=dataset.origins[n_train:],
        target_timestamps=dataset.target_timestamps[n_train:],
    )
    return train, val


def make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(dataset.x), torch.from_numpy(dataset.y))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]
    return total_loss / max(total_count, 1)


def train_model(
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    input_size: int,
    adjacency: np.ndarray,
) -> tuple[nn.Module, list[dict], float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGNNForecast(
        num_nodes=input_size,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        horizon=FORECAST_HOURS,
        adjacency=adjacency,
    ).to(device)
    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_ds, BATCH_SIZE, shuffle=False)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    log_rows: list[dict] = []

    print("[INFO] Predictor: stgnn")
    print(f"[INFO] Training on device: {device}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate_mae(model, val_loader, device, loss_fn)
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        log_rows.append(
            {
                "epoch": epoch,
                "train_mae_standardized": train_loss,
                "val_mae_standardized": val_loss,
                "best_val_mae_standardized": best_val,
                "improved": bool(improved),
            }
        )
        print(
            f"epoch={epoch:03d} train_MAE={train_loss:.6f} "
            f"val_MAE={val_loss:.6f} best={best_val:.6f}"
        )
        if bad_epochs >= PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state")
    model.load_state_dict(best_state)
    return model, log_rows, best_val


def predict(model: nn.Module, dataset: WindowDataset) -> np.ndarray:
    device = next(model.parameters()).device
    loader = make_loader(dataset, BATCH_SIZE, shuffle=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def inverse_transform(array: np.ndarray, mean: pd.Series, std: pd.Series) -> np.ndarray:
    return array * std.to_numpy(dtype=np.float32)[None, None, :] + mean.to_numpy(dtype=np.float32)[None, None, :]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_cols: list[str]) -> dict:
    """Metrics for net-load forecasting only.

    Electricity price is intentionally excluded from the forecasting task.
    """
    metrics = {
        "overall": {
            "mae": mae(y_true, y_pred),
            "mape_percent": mape(y_true, y_pred),
        },
        "netload_overall": {
            "mae": mae(y_true, y_pred),
            "mape_percent": mape(y_true, y_pred),
        },
        "per_bus_netload": {},
        "per_horizon": {},
    }

    for i, bus_name in enumerate(feature_cols):
        metrics["per_bus_netload"][bus_name] = {
            "mae": mae(y_true[:, :, i], y_pred[:, :, i]),
            "mape_percent": mape(y_true[:, :, i], y_pred[:, :, i]),
        }

    for h in range(FORECAST_HOURS):
        metrics["per_horizon"][f"horizon_{h + 1}"] = {
            "mae": mae(y_true[:, h, :], y_pred[:, h, :]),
            "mape_percent": mape(y_true[:, h, :], y_pred[:, h, :]),
        }
    return metrics


def save_predictions_long(
    dataset: WindowDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: list[str],
    path: Path,
) -> None:
    fieldnames = [
        "sample_id",
        "forecast_origin",
        "target_timestamp",
        "horizon",
        "variable",
        "y_true",
        "y_pred",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        for sample_id in range(y_true.shape[0]):
            origin = dataset.origins[sample_id].strftime("%Y-%m-%dT%H:%M:%SZ")
            target_times = dataset.target_timestamps[sample_id]
            for h in range(FORECAST_HOURS):
                target_ts = target_times[h].strftime("%Y-%m-%dT%H:%M:%SZ")
                for j, variable in enumerate(feature_cols):
                    writer.writerow(
                        [
                            sample_id,
                            origin,
                            target_ts,
                            h + 1,
                            variable,
                            float(y_true[sample_id, h, j]),
                            float(y_pred[sample_id, h, j]),
                        ]
                    )


def nearest_sample_indices(origins: pd.DatetimeIndex) -> list[int]:
    indices: list[int] = []
    origin_ns = origins.view("int64")
    for requested in REPRESENTATIVE_ORIGINS:
        requested_ns = requested.value
        idx = int(np.argmin(np.abs(origin_ns - requested_ns)))
        if idx not in indices:
            indices.append(idx)
    return indices


def plot_variable_examples(
    dataset: WindowDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: list[str],
    variable: str,
    path: Path,
) -> None:
    var_idx = feature_cols.index(variable)
    sample_indices = nearest_sample_indices(dataset.origins)
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 3.2 * len(sample_indices)), sharex=False)
    if len(sample_indices) == 1:
        axes = [axes]
    for ax, sample_id in zip(axes, sample_indices):
        target_times = dataset.target_timestamps[sample_id]
        x = [ts.to_pydatetime() for ts in target_times]
        ax.plot(x, y_true[sample_id, :, var_idx], label="true", linewidth=2)
        ax.plot(x, y_pred[sample_id, :, var_idx], label="pred", linewidth=2, linestyle="--")
        ax.set_title(f"{variable}, origin={dataset.origins[sample_id].strftime('%Y-%m-%d %H:%M UTC')}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_total_netload_examples(
    dataset: WindowDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    sample_indices = nearest_sample_indices(dataset.origins)
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 3.2 * len(sample_indices)), sharex=False)
    if len(sample_indices) == 1:
        axes = [axes]
    for ax, sample_id in zip(axes, sample_indices):
        target_times = dataset.target_timestamps[sample_id]
        x = [ts.to_pydatetime() for ts in target_times]
        true_total = y_true[sample_id, :, :33].sum(axis=1)
        pred_total = y_pred[sample_id, :, :33].sum(axis=1)
        ax.plot(x, true_total, label="true", linewidth=2)
        ax.plot(x, pred_total, label="pred", linewidth=2, linestyle="--")
        ax.set_title(f"total_netload, origin={dataset.origins[sample_id].strftime('%Y-%m-%d %H:%M UTC')}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_figures(dataset: WindowDataset, y_true: np.ndarray, y_pred: np.ndarray, feature_cols: list[str]) -> None:
    for bus in [6, 18, 30]:
        plot_variable_examples(
            dataset,
            y_true,
            y_pred,
            feature_cols,
            f"netload_bus_{bus}",
            FIGURE_DIR / f"netload_bus_{bus}_example.png",
        )
    plot_total_netload_examples(dataset, y_true, y_pred, FIGURE_DIR / "total_netload_example.png")


def save_model(
    model: nn.Module,
    feature_cols: list[str],
    best_val: float,
    adjacency_path: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "predictor": "stgnn",
            "feature_cols": feature_cols,
            "input_hours": INPUT_HOURS,
            "forecast_hours": FORECAST_HOURS,
            "input_size": len(feature_cols),
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "best_val_mae_standardized": best_val,
            "adjacency": {
                "type": "abs_pearson_training_period_gcn_normalized",
                "raw_abs_pearson_csv": adjacency_path,
            },
        },
        RESULT_DIR / "model_best.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train distribution-feeder 1H net-load forecaster.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATA_PATH, RESULT_DIR, FIGURE_DIR
    global EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN_SIZE, PATIENCE, RANDOM_SEED

    DATA_PATH = args.data_path
    RESULT_DIR = args.output_dir
    FIGURE_DIR = RESULT_DIR / "figures"
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    LEARNING_RATE = float(args.learning_rate)
    HIDDEN_SIZE = int(args.hidden_size)
    PATIENCE = int(args.patience)
    RANDOM_SEED = int(args.seed)

    set_seed(RANDOM_SEED)
    ensure_dirs()

    values, timestamps, feature_cols = load_dataset()
    mean, std = compute_scaler(values)
    zero_cols = constant_zero_columns(values, feature_cols)
    pearson_adj_df, pearson_adj_norm = compute_pearson_adjacency(values, feature_cols)
    adjacency_path = RESULT_DIR / "pearson_adjacency_abs.csv"
    pearson_adj_df.to_csv(adjacency_path)
    standardized = ((values - mean) / std).to_numpy(dtype=np.float32)

    train_all = build_windows(standardized, timestamps, TRAIN_START, TRAIN_END)
    test_ds = build_windows(standardized, timestamps, TEST_START, TEST_END)
    train_ds, val_ds = split_train_validation(train_all, validation_ratio=0.2)

    scaler_stats = {
        "feature_cols": feature_cols,
        "predictor": "stgnn",
        "mean": {col: float(mean[col]) for col in feature_cols},
        "std": {col: float(std[col]) for col in feature_cols},
        "pearson_adjacency": {
            "raw_abs_csv": str(adjacency_path),
            "normalization": "D^-1/2 |corr| D^-1/2 with diagonal self-loops",
            "training_period_only": True,
        },
        "constant_zero_columns": zero_cols,
        "train_period": {
            "start": TRAIN_START.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": TRAIN_END.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "forecast_output_period": {
            "start": TEST_START.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": TEST_END.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "correction_train_validate_months": ["2023-10", "2023-11"],
            "test_month": "2023-12",
        },
    }
    (RESULT_DIR / "scaler_stats.json").write_text(json.dumps(scaler_stats, indent=2), encoding="utf-8")

    model, log_rows, best_val = train_model(
        train_ds,
        val_ds,
        input_size=len(feature_cols),
        adjacency=pearson_adj_norm,
    )
    pd.DataFrame(log_rows).to_csv(RESULT_DIR / "train_log.csv", index=False)
    save_model(model, feature_cols, best_val, str(adjacency_path))

    pred_std = predict(model, test_ds)
    y_true = inverse_transform(test_ds.y, mean, std)
    y_pred = inverse_transform(pred_std, mean, std)
    if zero_cols:
        zero_idx = [feature_cols.index(col) for col in zero_cols]
        y_true[:, :, zero_idx] = 0.0
        y_pred[:, :, zero_idx] = 0.0

    metrics = compute_metrics(y_true, y_pred, feature_cols)
    metrics["predictor"] = "stgnn"
    metrics["constant_zero_columns"] = zero_cols
    metrics["pearson_adjacency"] = {
        "raw_abs_csv": str(adjacency_path),
        "used_by_model": True,
    }
    metrics["counts"] = {
        "train_windows_total": int(len(train_all.x)),
        "train_windows_used": int(len(train_ds.x)),
        "validation_windows": int(len(val_ds.x)),
        "test_windows": int(len(test_ds.x)),
    }
    metrics["best_val_mae_standardized"] = float(best_val)
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_predictions_long(test_ds, y_true, y_pred, feature_cols, RESULT_DIR / "test_predictions.csv")
    save_figures(test_ds, y_true, y_pred, feature_cols)

    print("\n[OK] STGNN 1H forecast training completed.")
    print(f"train_windows={len(train_ds.x)}")
    print(f"validation_windows={len(val_ds.x)}")
    print(f"test_windows={len(test_ds.x)}")
    print(f"best_validation_MAE_standardized={best_val:.6f}")
    print(
        f"test MAE / MAPE: {metrics['overall']['mae']:.6f} / "
        f"{metrics['overall']['mape_percent']:.3f}%"
    )
    print(
        f"net-load MAE / MAPE: {metrics['netload_overall']['mae']:.6f} / "
        f"{metrics['netload_overall']['mape_percent']:.3f}%"
    )
    print("Price is not predicted; dispatch scripts use the realized day-ahead price as an exogenous input.")
    print(f"results={RESULT_DIR}")

if __name__ == "__main__":
    main()

