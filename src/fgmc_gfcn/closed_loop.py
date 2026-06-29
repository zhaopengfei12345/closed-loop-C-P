from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .dispatch import (
    BUS_COLS,
    HORIZON,
    NUM_BUSES,
    DispatchConfig,
    compute_bidirectional_marginal_maps_from_duals,
    dispatch_and_replay,
    fixed_from_solution,
    iso,
    load_aligned_dataset,
    load_network,
    load_price_series,
    metadata as dispatch_metadata,
    choose_solver,
    solve_dispatch,
    is_optimal,
    get_dual,
)


PREDICTION_PATH = Path("results") / "lstm_1h_forecast" / "test_predictions.csv"
DEFAULT_OUTPUT_DIR = Path("results") / "fgmc_gfcn_iterative_dec"
EPS = 1e-6
SEED = 2026


class HyperParams:
    """Centralized hyperparameters."""

    bmc_loss_weight = 0.5
    lambda_br = 1e-2
    regret_alpha = 1.5
    regret_weight_cap = 5.0
    beta = 0.0
    gamma = 0.0
    settlement_system_weight = 0.0
    settlement_node_weight = 0.0
    lambda_cost_bias = 0.0
    prediction_mae_weight = 0.0
    perturb_eps = 0.15
    boundary_kappa = 1.0
    epochs = 180
    batch_size = 8
    learning_rate = 2e-3
    patience = 20
    base_rho = 4.5
    corrector_hidden_dim = 32
    corrector_dropout = 0.05
    corrector_architecture = "graphflow"
    marginal_map_mode = "dual"
    marginal_direction_mode = "bidirectional"
    use_boundary_channel = True
    use_extra_system_head = False
    extra_system_scale_init = 0.1
    settlement_under_multiplier = 1.0
    settlement_over_multiplier = 1.0
    settlement_weight_power = 1.0
    boundary_loss_weight = 0.0
    node_sensitivity_blend = 0.35


class GraphFlowResidualCorrectionNet(nn.Module):
    """
    Graph-Flow Residual Correction Network.

    Input:
        X: torch.Tensor, shape [B, 4, N, T]
           channel 0: y_prev_scaled
           channel 1: M_plus_norm
           channel 2: M_minus_norm
           channel 3: B_boundary

    Output:
        raw_delta: torch.Tensor, shape [B, N, T]

    Design:
        1) Encode predicted net-load trajectory.
        2) Encode bidirectional marginal-cost features.
        3) Generate edge-level correction flows.
        4) Convert edge flows into node corrections by graph divergence.
        5) Add a global correction head to handle system-level bias.
    """

    DEFAULT_distribution_feeder_RADIAL_BRANCHES = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
        (14, 15), (15, 16), (16, 17), (17, 18),
        (2, 19), (19, 20), (20, 21), (21, 22),
        (3, 23), (23, 24), (24, 25),
        (6, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (30, 31), (31, 32), (32, 33),
    ]

    def __init__(
        self,
        num_nodes: int = 33,
        hidden_dim: int = 32,
        branch_csv: str = "data/processed_1h/feeder_branch.csv",
        dropout: float = 0.05,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 1) Net-load trajectory encoder
        self.p_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 2) Sensitivity-feature encoder: [M_plus, M_minus, B]
        self.s_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        # Edge feature:
        # [H_p_i, H_p_j, H_p_i-H_p_j, H_s_i, H_s_j, H_s_i-H_s_j]
        edge_feat_dim = 6 * hidden_dim

        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.edge_flow_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 3) Global correction head
        # input: mean over nodes of concat(H_p, H_s), shape [B, 2H, T]
        self.global_head = nn.Sequential(
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

        # 4) Node allocator for global correction
        self.allocator = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        edge_from, edge_to = self._build_edge_index(branch_csv, num_nodes)

        self.register_buffer("edge_from", edge_from)
        self.register_buffer("edge_to", edge_to)

        # A small learnable scale stabilizes the graph-flow correction
        self.flow_scale = nn.Parameter(torch.tensor(0.1))
        self.global_scale = nn.Parameter(torch.tensor(0.1))
        self.use_extra_system_head = bool(HyperParams.use_extra_system_head)
        if self.use_extra_system_head:
            self.extra_system_head = nn.Sequential(
                nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, 1, kernel_size=1),
            )
            self.extra_system_scale = nn.Parameter(torch.tensor(float(HyperParams.extra_system_scale_init)))

    def _build_edge_index(self, branch_csv: str, num_nodes: int):
        """
        Build directed edge index.
        Uses branch_csv if possible; otherwise falls back to distribution-feeder radial topology.
        Node indices in returned tensors are 0-based.
        """
        branches = None

        if branch_csv is not None and os.path.exists(branch_csv):
            try:
                import pandas as pd

                df = pd.read_csv(branch_csv)
                cols = {c.lower(): c for c in df.columns}

                from_candidates = [
                    "from_bus", "f_bus", "fbus", "from", "frombus", "i", "bus_i"
                ]
                to_candidates = [
                    "to_bus", "t_bus", "tbus", "to", "tobus", "j", "bus_j"
                ]

                from_col = None
                to_col = None

                for c in from_candidates:
                    if c in cols:
                        from_col = cols[c]
                        break

                for c in to_candidates:
                    if c in cols:
                        to_col = cols[c]
                        break

                if from_col is not None and to_col is not None:
                    branches = list(zip(df[from_col].astype(int), df[to_col].astype(int)))

            except Exception:
                branches = None

        if branches is None:
            branches = self.DEFAULT_distribution_feeder_RADIAL_BRANCHES

        clean_edges = []
        seen = set()

        for u, v in branches:
            u0 = int(u) - 1
            v0 = int(v) - 1

            if not (0 <= u0 < num_nodes and 0 <= v0 < num_nodes):
                continue

            key = (u0, v0)
            if key not in seen:
                seen.add(key)
                clean_edges.append(key)

        if len(clean_edges) == 0:
            raise ValueError("No valid branches found for GraphFlowResidualCorrectionNet.")

        edge_from = torch.tensor([e[0] for e in clean_edges], dtype=torch.long)
        edge_to = torch.tensor([e[1] for e in clean_edges], dtype=torch.long)

        return edge_from, edge_to

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [B, 4, N, T]

        Returns:
            raw_delta: [B, N, T]
        """
        if X.dim() != 4:
            raise ValueError(f"Expected X with shape [B, 4, N, T], got {tuple(X.shape)}")

        Bsz, C, N, T = X.shape

        if C != 4:
            raise ValueError(f"Expected 4 input channels, got {C}")

        if N != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {N}")

        P = X[:, 0:1, :, :]          # [B, 1, N, T]
        S = X[:, 1:4, :, :]          # [B, 3, N, T], [M+, M-, B]

        H_p = self.p_encoder(P)      # [B, H, N, T]
        H_s = self.s_encoder(S)      # [B, H, N, T]

        delta_sp = self._graph_flow_divergence(H_p, H_s, Bsz, N, T)
        delta_global = self._global_correction(H_p, H_s, S)

        raw_delta = self.flow_scale * delta_sp + self.global_scale * delta_global
        if self.use_extra_system_head:
            H_global = torch.cat([H_p, H_s], dim=1).mean(dim=2)
            system_raw = self.extra_system_head(H_global).squeeze(1).unsqueeze(1)
            raw_delta = raw_delta + self.extra_system_scale * system_raw.expand(-1, N, -1)

        return raw_delta

    def _graph_flow_divergence(
        self,
        H_p: torch.Tensor,
        H_s: torch.Tensor,
        Bsz: int,
        N: int,
        T: int,
    ) -> torch.Tensor:
        """
        Generate edge correction flows and convert them to node corrections
        using graph divergence.
        """
        edge_from = self.edge_from
        edge_to = self.edge_to
        E = edge_from.numel()

        # Gather node features on both ends of each edge
        # [B, H, E, T]
        Hp_i = H_p[:, :, edge_from, :]
        Hp_j = H_p[:, :, edge_to, :]
        Hs_i = H_s[:, :, edge_from, :]
        Hs_j = H_s[:, :, edge_to, :]

        edge_feat = torch.cat(
            [
                Hp_i,
                Hp_j,
                Hp_i - Hp_j,
                Hs_i,
                Hs_j,
                Hs_i - Hs_j,
            ],
            dim=1,
        )  # [B, 6H, E, T]

        # MLP expects last dimension as feature dimension
        edge_feat = edge_feat.permute(0, 2, 3, 1).contiguous()  # [B, E, T, 6H]

        edge_gate = torch.sigmoid(self.edge_gate_mlp(edge_feat)).squeeze(-1)  # [B, E, T]
        edge_base = torch.tanh(self.edge_flow_mlp(edge_feat)).squeeze(-1)     # [B, E, T]

        edge_flow = edge_gate * edge_base  # [B, E, T]

        # Graph divergence:
        # edge e=(i,j): node_i += -flow_e, node_j += +flow_e
        delta_sp = H_p.new_zeros(Bsz, N, T)

        idx_from = edge_from.view(1, E, 1).expand(Bsz, E, T)
        idx_to = edge_to.view(1, E, 1).expand(Bsz, E, T)

        delta_sp.scatter_add_(1, idx_from, -edge_flow)
        delta_sp.scatter_add_(1, idx_to, edge_flow)

        return delta_sp  # [B, N, T]

    def _global_correction(
        self,
        H_p: torch.Tensor,
        H_s: torch.Tensor,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a system-level correction and allocate it to nodes
        according to sensitivity features.
        """
        # [B, 2H, N, T] -> [B, 2H, T]
        H_global = torch.cat([H_p, H_s], dim=1).mean(dim=2)

        # [B, 1, T]
        g_t = self.global_head(H_global)

        # allocator logits: [B, 1, N, T] -> [B, N, T]
        alloc_logits = self.allocator(S).squeeze(1)

        # Softmax over nodes for each time step
        omega = F.softmax(alloc_logits, dim=1)  # [B, N, T]

        delta_global = g_t.squeeze(1).unsqueeze(1) * omega  # [B, N, T]

        return delta_global

class BiMCCorrector(nn.Module):
    """Small 2D CNN over bus-time maps.

    Input:  [batch, 4, 33, 24]
    Output: [batch, 33, 24]
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MLPCorrector(nn.Module):
    """Local MLP applied independently to each bus-time cell."""

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 4, N, T] -> [B, N, T, 4] -> [B, N, T]
        z = x.permute(0, 2, 3, 1).contiguous()
        return self.net(z).squeeze(-1)


class PlainGNNCorrector(nn.Module):
    """Ordinary graph-convolution corrector without graph-flow divergence."""

    DEFAULT_distribution_feeder_RADIAL_BRANCHES = GraphFlowResidualCorrectionNet.DEFAULT_distribution_feeder_RADIAL_BRANCHES

    def __init__(
        self,
        num_nodes: int = 33,
        hidden_dim: int = 32,
        branch_csv: str = "data/processed_1h/feeder_branch.csv",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        adjacency = self._build_adjacency(branch_csv, num_nodes)
        self.register_buffer("adjacency", adjacency)
        self.lin_in = nn.Linear(4, hidden_dim)
        self.lin_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _build_adjacency(self, branch_csv: str, num_nodes: int) -> torch.Tensor:
        branches = None
        if branch_csv is not None and os.path.exists(branch_csv):
            try:
                df = pd.read_csv(branch_csv)
                cols = {c.lower(): c for c in df.columns}
                from_col = next((cols[c] for c in ["from_bus", "f_bus", "fbus", "from", "frombus"] if c in cols), None)
                to_col = next((cols[c] for c in ["to_bus", "t_bus", "tbus", "to", "tobus"] if c in cols), None)
                if from_col is not None and to_col is not None:
                    branches = list(zip(df[from_col].astype(int), df[to_col].astype(int)))
            except Exception:
                branches = None
        if branches is None:
            branches = self.DEFAULT_distribution_feeder_RADIAL_BRANCHES

        adj = np.eye(num_nodes, dtype=np.float32)
        for u, v in branches:
            u0, v0 = int(u) - 1, int(v) - 1
            if 0 <= u0 < num_nodes and 0 <= v0 < num_nodes:
                adj[u0, v0] = 1.0
                adj[v0, u0] = 1.0
        degree = np.maximum(adj.sum(axis=1), EPS)
        adj = (adj / degree[:, None]).astype(np.float32)
        return torch.from_numpy(adj)

    def _graph_linear(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        # x: [B, T, N, C], adjacency: [N, N]
        x = torch.einsum("ij,btjc->btic", self.adjacency, x)
        return layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 4, N, T] -> [B, T, N, 4]
        z = x.permute(0, 3, 2, 1).contiguous()
        z = F.gelu(self._graph_linear(z, self.lin_in))
        z = self.dropout(z)
        z = F.gelu(self._graph_linear(z, self.lin_hidden))
        z = self.lin_out(z).squeeze(-1)  # [B, T, N]
        return z.permute(0, 2, 1).contiguous()


def build_corrector_model() -> nn.Module:
    arch = str(HyperParams.corrector_architecture).lower()
    if arch == "graphflow":
        return GraphFlowResidualCorrectionNet(
            hidden_dim=HyperParams.corrector_hidden_dim,
            dropout=HyperParams.corrector_dropout,
        )
    if arch == "cnn":
        return BiMCCorrector(
            hidden_dim=HyperParams.corrector_hidden_dim,
            dropout=HyperParams.corrector_dropout,
        )
    if arch == "mlp":
        return MLPCorrector(
            hidden_dim=HyperParams.corrector_hidden_dim,
            dropout=HyperParams.corrector_dropout,
        )
    if arch == "gnn":
        return PlainGNNCorrector(
            hidden_dim=HyperParams.corrector_hidden_dim,
            dropout=HyperParams.corrector_dropout,
        )
    raise ValueError(f"Unknown corrector architecture: {HyperParams.corrector_architecture}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denom = max(float(np.sum(weight)), EPS)
    return float(np.sum(weight * np.abs(y_pred - y_true)) / denom)


def load_daily_blocks(prediction_path: Path) -> dict[str, dict[str, Any]]:
    """Load complete daily 00:00 samples for Nov/Dec from LSTM net-load predictions.

    The prediction file may still contain old price records.  They are ignored.
    Only netload_bus_1 ... netload_bus_33 are used.
    """

    pred = pd.read_csv(
        prediction_path,
        usecols=["forecast_origin", "target_timestamp", "horizon", "variable", "y_true", "y_pred"],
    )
    pred = pred[pred["variable"].isin(BUS_COLS)].copy()
    pred["forecast_dt"] = pd.to_datetime(pred["forecast_origin"], utc=True)
    pred["target_dt"] = pd.to_datetime(pred["target_timestamp"], utc=True)
    pred = pred[pred["forecast_dt"].dt.hour.eq(0)].copy()

    parts: dict[str, list[dict[str, Any]]] = {"nov": [], "dec": []}
    for origin, group in pred.groupby("forecast_origin", sort=True):
        origin_dt = group["forecast_dt"].iloc[0]
        if origin_dt.month == 11:
            split = "nov"
        elif origin_dt.month == 12:
            split = "dec"
        else:
            continue
        pivot_pred = group.pivot_table(index="target_dt", columns="variable", values="y_pred", aggfunc="first").sort_index()
        pivot_true = group.pivot_table(index="target_dt", columns="variable", values="y_true", aggfunc="first").sort_index()
        if len(pivot_pred) != HORIZON or any(c not in pivot_pred.columns for c in BUS_COLS):
            continue
        pivot_pred = pivot_pred[BUS_COLS]
        pivot_true = pivot_true[BUS_COLS]
        if pivot_pred.isna().any().any() or pivot_true.isna().any().any():
            continue
        parts[split].append(
            {
                "forecast_origin": str(origin),
                "forecast_dt": origin_dt,
                "target_index": pivot_pred.index,
                "y_pred": pivot_pred.to_numpy(dtype=np.float32).T,
                "y_true": pivot_true.to_numpy(dtype=np.float32).T,
            }
        )

    out: dict[str, dict[str, Any]] = {}
    for split, blocks in parts.items():
        blocks = sorted(blocks, key=lambda x: x["forecast_dt"])
        if not blocks:
            raise ValueError(f"No complete daily 00:00 samples found for {split}")
        out[split] = {
            "origins": [b["forecast_origin"] for b in blocks],
            "target_indices": [b["target_index"] for b in blocks],
            "y_pred": np.stack([b["y_pred"] for b in blocks]).astype(np.float32),
            "y_true": np.stack([b["y_true"] for b in blocks]).astype(np.float32),
            "price_prediction_used": False,
        }
    return out


def array_to_pnet(values: np.ndarray, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(values.T, index=target_index, columns=BUS_COLS)


def extract_p_balance_duals(model) -> np.ndarray:
    lam = np.zeros((NUM_BUSES, HORIZON), dtype=float)
    if not hasattr(model, "dual"):
        return lam
    for bus in range(1, NUM_BUSES + 1):
        for t in range(HORIZON):
            lam[bus - 1, t] = get_dual(model, model.p_balance[bus, t])
    return lam


def solve_for_dual_map(pnet: pd.DataFrame, prices: np.ndarray, network, solver, config: DispatchConfig, aligned):
    model, sol = solve_dispatch(pnet, prices, network, solver, config, aligned=aligned, import_duals=True)
    if not is_optimal(sol.status, sol.termination):
        return np.zeros((NUM_BUSES, HORIZON), dtype=float), f"{sol.status}/{sol.termination}", False
    return extract_p_balance_duals(model), f"{sol.status}/{sol.termination}", True


def compute_node_economic_sensitivity(network, config: DispatchConfig | None = None) -> np.ndarray:
    """Return a mean-one node weight for allocating system RT imbalance risk.

    The RT imbalance spread is system-level, but the corrector needs a spatial
    signal.  This lightweight proxy gives more weight to electrically remote,
    larger-load, and flexible-resource buses without changing the total average
    marginal-map scale.
    """
    if network is None:
        return np.ones(NUM_BUSES, dtype=np.float32)
    config = config or DispatchConfig()

    bus = network.bus.copy()
    pd_by_bus = (
        bus.set_index("bus_i")["Pd_MW"].reindex(range(1, NUM_BUSES + 1)).fillna(0.0).to_numpy(dtype=float)
        if {"bus_i", "Pd_MW"}.issubset(bus.columns)
        else np.zeros(NUM_BUSES, dtype=float)
    )
    if pd_by_bus.max() > EPS:
        load_score = pd_by_bus / (pd_by_bus.max() + EPS)
    else:
        load_score = np.zeros(NUM_BUSES, dtype=float)

    # Diagonal q-aware voltage sensitivity is a compact proxy for feeder
    # electrical distance and reactive-voltage coupling.
    try:
        impedance_score = np.diag(np.asarray(network.voltage_sens_qaware, dtype=float)).copy()
    except Exception:
        impedance_score = np.zeros(NUM_BUSES, dtype=float)
    if np.nanmax(impedance_score) > EPS:
        impedance_score = impedance_score / (np.nanmax(impedance_score) + EPS)
    else:
        impedance_score = np.zeros(NUM_BUSES, dtype=float)

    flex_score = np.zeros(NUM_BUSES, dtype=float)
    for param in config.dg_params.values():
        if 1 <= int(param.bus) <= NUM_BUSES:
            flex_score[int(param.bus) - 1] += 1.0
    for param in config.ess_params.values():
        if 1 <= int(param.bus) <= NUM_BUSES:
            flex_score[int(param.bus) - 1] += 0.8
    for bus_id in config.pv_caps.keys():
        if 1 <= int(bus_id) <= NUM_BUSES:
            flex_score[int(bus_id) - 1] += 0.5
    if flex_score.max() > EPS:
        flex_score = flex_score / (flex_score.max() + EPS)

    raw = 0.45 * load_score + 0.35 * impedance_score + 0.20 * flex_score
    raw = raw - float(np.mean(raw))
    weight = 1.0 + float(HyperParams.node_sensitivity_blend) * raw
    weight = np.clip(weight, 0.50, 1.50)
    weight = weight / max(float(np.mean(weight)), EPS)
    return weight.astype(np.float32)


def build_rt_imbalance_marginal_maps(prices: np.ndarray, config: DispatchConfig, network=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build system-level maps from real-time imbalance settlement spreads.

    M_plus weighs over-prediction errors: a surplus day-ahead schedule is sold
    back at the real-time sell price. M_minus weighs under-prediction errors:
    the missing energy is bought at the real-time buy price. The map uses only
    the incremental spread versus the day-ahead price, so it targets forecast
    error cost rather than the whole energy cost.
    """

    price = np.asarray(prices, dtype=float)
    positive_price = np.maximum(price, 0.0)
    rt_buy = price + float(config.rt_buy_price_adder) + (float(config.rt_buy_price_multiplier) - 1.0) * positive_price
    rt_sell = float(config.rt_sell_price_ratio) * price
    over_raw_t = np.maximum(price - rt_sell + float(config.rt_imbalance_fee), 1.0)
    under_raw_t = np.maximum(rt_buy - price + float(config.rt_imbalance_fee), 1.0)
    node_weight = compute_node_economic_sensitivity(network, config)
    over_raw = node_weight[:, None] * over_raw_t[None, :]
    under_raw = node_weight[:, None] * under_raw_t[None, :]
    scale = max(float(np.mean(np.concatenate([over_raw.ravel(), under_raw.ravel()]))), EPS)
    m_plus = np.clip(over_raw / scale, 0.0, 5.0).astype(np.float32)
    m_minus = np.clip(under_raw / scale, 0.0, 5.0).astype(np.float32)
    maps = {
        "M_plus_raw": over_raw.astype(np.float32),
        "M_minus_raw": under_raw.astype(np.float32),
        "M_plus_norm": m_plus,
        "M_minus_norm": m_minus,
        "B_boundary": np.zeros((NUM_BUSES, HORIZON), dtype=np.float32),
    }
    meta = {
        "mean_M_plus_raw": float(np.mean(over_raw)),
        "mean_M_minus_raw": float(np.mean(under_raw)),
        "max_M_plus_raw": float(np.max(over_raw)) if over_raw.size else 0.0,
        "max_M_minus_raw": float(np.max(under_raw)) if under_raw.size else 0.0,
        "mean_node_sensitivity": float(np.mean(node_weight)),
        "min_node_sensitivity": float(np.min(node_weight)),
        "max_node_sensitivity": float(np.max(node_weight)),
        "fallback_uniform_M_plus": False,
        "fallback_uniform_M_minus": False,
        "marginal_dual_available": False,
        "feedback_type": "rt_imbalance_spread_x_node_economic_sensitivity",
    }
    return maps, meta


def build_feature_maps_for_sample(
    y_prev: np.ndarray,
    y_true: np.ndarray,
    target_index: pd.DatetimeIndex,
    prices: np.ndarray,
    network,
    solver,
    config: DispatchConfig,
    aligned: pd.DataFrame | None,
    sigma: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    """Build boundary-aware two-sided marginal-cost maps and sample regret."""

    try:
        if HyperParams.marginal_map_mode == "rt_imbalance":
            maps, meta = build_rt_imbalance_marginal_maps(prices, config, network=network)
            status_plus = "rt_imbalance_spread"
            status_minus = "rt_imbalance_spread"
        else:
            sigma_map = sigma[:, None]
            y_plus = y_prev + HyperParams.perturb_eps * sigma_map
            y_minus = np.maximum(y_prev - HyperParams.perturb_eps * sigma_map, 0.0)

            p_plus = array_to_pnet(y_plus, target_index)
            p_minus = array_to_pnet(y_minus, target_index)
            lam_plus, status_plus, ok_plus = solve_for_dual_map(p_plus, prices, network, solver, config, aligned)
            lam_minus, status_minus, ok_minus = solve_for_dual_map(p_minus, prices, network, solver, config, aligned)

            maps, meta = compute_bidirectional_marginal_maps_from_duals(
                lam_plus,
                lam_minus,
                normalize_cap=5.0,
                boundary_kappa=HyperParams.boundary_kappa,
            )
            meta.update({
                "status_plus": status_plus,
                "status_minus": status_minus,
                "dual_plus_available": bool(ok_plus),
                "dual_minus_available": bool(ok_minus),
            })
        meta["marginal_map_mode"] = HyperParams.marginal_map_mode

        # Baseline-vs-oracle realized regret for this sample. It is computed in
        # the outer loop and used as a capped sample weight in the correction loss.
        p_prev = array_to_pnet(y_prev, target_index)
        p_true = array_to_pnet(y_true, target_index)
        pred_eval = dispatch_and_replay(p_prev, p_true, prices, network, solver, config, aligned=aligned)
        oracle_eval = dispatch_and_replay(p_true, p_true, prices, network, solver, config, aligned=aligned)
        if pred_eval.get("ok") and oracle_eval.get("ok"):
            oracle_cost = float(oracle_eval["realized_cost"])
            regret = max(float(pred_eval["realized_cost"]) - oracle_cost, 0.0)
            regret_norm = regret / (abs(oracle_cost) + EPS)
            regret_status = f"pred={pred_eval.get('status')};oracle={oracle_eval.get('status')}"
        else:
            oracle_cost = 0.0
            regret = 0.0
            regret_norm = 0.0
            regret_status = f"pred={pred_eval.get('status', 'failed')};oracle={oracle_eval.get('status', 'failed')}"
        maps["sample_regret"] = np.array(regret, dtype=np.float32)
        maps["sample_regret_norm"] = np.array(regret_norm, dtype=np.float32)
        if str(HyperParams.marginal_direction_mode).lower() == "unified":
            unified = 0.5 * (np.abs(maps["M_plus_norm"]) + np.abs(maps["M_minus_norm"]))
            unified = np.clip(unified, 0.0, 5.0).astype(np.float32)
            maps["M_plus_norm"] = unified
            maps["M_minus_norm"] = unified.copy()
        if not bool(HyperParams.use_boundary_channel):
            maps["B_boundary"] = np.zeros_like(maps["B_boundary"], dtype=np.float32)
        meta["sample_regret"] = regret
        meta["sample_regret_norm"] = regret_norm
        meta["sample_oracle_cost"] = oracle_cost
        meta["regret_status"] = regret_status
        meta["marginal_direction_mode"] = str(HyperParams.marginal_direction_mode)
        meta["boundary_channel_used"] = bool(HyperParams.use_boundary_channel)
        status = f"plus={status_plus};minus={status_minus};regret={regret_status}"
        return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in maps.items()}, meta, status
    except Exception as exc:
        z = np.ones((NUM_BUSES, HORIZON), dtype=np.float32)
        maps = {
            "M_plus_norm": z,
            "M_minus_norm": z,
            "B_boundary": np.zeros_like(z),
            "M_plus_raw": np.zeros_like(z),
            "M_minus_raw": np.zeros_like(z),
            "sample_regret": np.array(0.0, dtype=np.float32),
            "sample_regret_norm": np.array(0.0, dtype=np.float32),
        }
        return maps, {"marginal_dual_available": False, "sample_regret": 0.0, "sample_regret_norm": 0.0}, f"failed: {exc}"


def build_features_for_split(
    split: str,
    round_id: int,
    round_dir: Path,
    data: dict[str, Any],
    y_prev_all: np.ndarray,
    network,
    price_series: pd.Series,
    solver,
    config: DispatchConfig,
    aligned: pd.DataFrame | None,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Build M_plus/M_minus/B maps and normalized regret signals."""

    n = y_prev_all.shape[0]
    m_plus_all = np.ones_like(y_prev_all, dtype=np.float32)
    m_minus_all = np.ones_like(y_prev_all, dtype=np.float32)
    b_all = np.zeros_like(y_prev_all, dtype=np.float32)
    regrets = np.zeros(n, dtype=np.float32)
    regret_norms = np.zeros(n, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    failed = 0
    dual_any = False

    for sample_id, origin in enumerate(data["origins"]):
        target_index = data["target_indices"][sample_id]
        prices = price_series.reindex(target_index)
        if prices.isna().any():
            z = np.ones((NUM_BUSES, HORIZON), dtype=np.float32)
            maps = {"M_plus_norm": z, "M_minus_norm": z, "B_boundary": np.zeros_like(z), "M_plus_raw": np.zeros_like(z), "M_minus_raw": np.zeros_like(z), "sample_regret": np.array(0.0, dtype=np.float32), "sample_regret_norm": np.array(0.0, dtype=np.float32)}
            meta = {"marginal_dual_available": False, "sample_regret": 0.0, "sample_regret_norm": 0.0}
            status = "missing_real_price"
            failed += 1
        else:
            maps, meta, status = build_feature_maps_for_sample(
                y_prev_all[sample_id],
                data["y_true"][sample_id],
                target_index,
                prices.to_numpy(dtype=float),
                network,
                solver,
                config,
                aligned,
                sigma,
            )
            if "failed" in status.lower() or "missing" in status.lower():
                failed += 1

        dual_any = dual_any or bool(meta.get("marginal_dual_available", False))
        m_plus_all[sample_id] = maps["M_plus_norm"]
        m_minus_all[sample_id] = maps["M_minus_norm"]
        b_all[sample_id] = maps["B_boundary"]
        regrets[sample_id] = float(meta.get("sample_regret", 0.0))
        regret_norms[sample_id] = float(meta.get("sample_regret_norm", 0.0))

        for h, ts in enumerate(target_index):
            for bus in range(1, NUM_BUSES + 1):
                rows.append({
                    "sample_id": sample_id,
                    "split": split,
                    "round": round_id,
                    "forecast_origin": origin,
                    "target_timestamp": iso(ts),
                    "horizon": h + 1,
                    "bus": bus,
                    "y_true_netload": float(data["y_true"][sample_id, bus - 1, h]),
                    "y_prev_netload": float(y_prev_all[sample_id, bus - 1, h]),
                    "M_plus_norm": float(m_plus_all[sample_id, bus - 1, h]),
                    "M_minus_norm": float(m_minus_all[sample_id, bus - 1, h]),
                    "B_boundary": float(b_all[sample_id, bus - 1, h]),
                    "sample_regret": float(regrets[sample_id]),
                    "sample_regret_norm": float(regret_norms[sample_id]),
                    "feature_status": status,
                })
        print(f"[round {round_id} {split}] sample={sample_id:03d} regret={regrets[sample_id]:.4f} regret_norm={regret_norms[sample_id]:.5g} mean_M+={m_plus_all[sample_id].mean():.5g} mean_M-={m_minus_all[sample_id].mean():.5g} B={b_all[sample_id].mean():.5g} status={status}")

    df = pd.DataFrame(rows)
    feature_dir = round_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(feature_dir / f"{split}_samples.csv", index=False)
    pd.DataFrame({"sample_id": np.arange(n), "split": split, "round": round_id, "sample_regret": regrets, "sample_regret_norm": regret_norms}).to_csv(feature_dir / f"{split}_regret.csv", index=False)

    summary = {
        "round": round_id,
        "split": split,
        "num_samples": int(n),
        "mean_M_plus": float(m_plus_all.mean()),
        "mean_M_minus": float(m_minus_all.mean()),
        "mean_B_boundary": float(b_all.mean()),
        "mean_sample_regret": float(regrets.mean()),
        "max_sample_regret": float(regrets.max()) if len(regrets) else 0.0,
        "mean_sample_regret_norm": float(regret_norms.mean()),
        "max_sample_regret_norm": float(regret_norms.max()) if len(regret_norms) else 0.0,
        "failed_feature_count": int(failed),
        "marginal_dual_available": bool(dual_any),
        "price_used": "real_day_ahead_price",
        "price_prediction_used": False,
    }
    return m_plus_all, m_minus_all, b_all, regret_norms.astype(np.float32), df, summary


def make_inputs(y_prev: np.ndarray, m_plus: np.ndarray, m_minus: np.ndarray, b_map: np.ndarray, mean_bus: np.ndarray, std_bus: np.ndarray) -> np.ndarray:
    y_scaled = (y_prev - mean_bus[None, :, None]) / std_bus[None, :, None]
    b_input = b_map if bool(HyperParams.use_boundary_channel) else np.zeros_like(b_map, dtype=np.float32)
    return np.stack([
        y_scaled,
        np.clip(m_plus / 5.0, -1.0, 1.0),
        np.clip(m_minus / 5.0, -1.0, 1.0),
        np.clip(b_input, 0.0, 1.0),
    ], axis=1).astype(np.float32)


def compute_delta(raw: torch.Tensor, sigma: torch.Tensor, rho: float) -> torch.Tensor:
    return rho * sigma[None, :, None] * torch.tanh(raw)


def build_settlement_loss_weights(data: dict[str, Any], price_series: pd.Series, config: DispatchConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized asymmetric RT imbalance weights for each sample-hour.

    Under-prediction requires expensive real-time purchases.  Over-prediction
    loses the spread between the day-ahead import price and the discounted
    real-time export settlement price.  A small floor keeps negative-price
    hours numerically stable without letting them dominate training.
    """

    prices = np.stack([
        price_series.reindex(index).to_numpy(dtype=np.float32)
        for index in data["target_indices"]
    ])
    if not np.isfinite(prices).all():
        raise ValueError("Missing or non-finite day-ahead price while building settlement loss weights.")
    positive_prices = np.maximum(prices, 0.0)
    rt_buy = (
        prices
        + float(config.rt_buy_price_adder)
        + (float(config.rt_buy_price_multiplier) - 1.0) * positive_prices
    )
    rt_sell = float(config.rt_sell_price_ratio) * prices
    under = np.maximum(rt_buy + float(config.rt_imbalance_fee), 1.0) * float(HyperParams.settlement_under_multiplier)
    over = np.maximum(prices - rt_sell + float(config.rt_imbalance_fee), 1.0) * float(HyperParams.settlement_over_multiplier)
    if abs(float(HyperParams.settlement_weight_power) - 1.0) > 1e-12:
        under = np.power(np.maximum(under, 1.0), float(HyperParams.settlement_weight_power))
        over = np.power(np.maximum(over, 1.0), float(HyperParams.settlement_weight_power))
    scale = max(float(np.mean(np.concatenate([under, over], axis=1))), EPS)
    return (under / scale).astype(np.float32), (over / scale).astype(np.float32)


def regret_sample_weights(regret_norm: np.ndarray) -> np.ndarray:
    """Convert outer-loop realized regret into differentiable sample weights."""

    regret = np.maximum(np.asarray(regret_norm, dtype=np.float32), 0.0)
    mean_regret = float(np.mean(regret))
    if mean_regret > EPS and float(HyperParams.regret_alpha) > 0.0:
        weights = 1.0 + float(HyperParams.regret_alpha) * regret / (mean_regret + EPS)
    else:
        weights = np.ones_like(regret, dtype=np.float32)
    return np.minimum(weights, float(HyperParams.regret_weight_cap)).astype(np.float32)


def loss_components(raw, y_prev, y_true, m_plus, m_minus, regret_norm, sigma, rho, settlement_under=None, settlement_over=None, b_map=None, regret_weight=None):
    delta = compute_delta(raw, sigma, rho)
    y_corr = y_prev + delta
    err = y_corr - y_true
    over = torch.relu(err)
    under = torch.relu(-err)
    # If y_corr > y_true, the forecast is high. M_plus weighs this upward-side error.
    # If y_corr < y_true, the forecast is low. M_minus weighs this downward-side error.
    if b_map is None:
        b_map = torch.zeros_like(m_plus)
    if regret_weight is None:
        regret_weight = torch.ones_like(regret_norm)
    sample_weight = regret_weight.view(-1)
    boundary_amp = 1.0 + float(HyperParams.boundary_kappa) * b_map
    mc_per_sample = torch.mean(boundary_amp * (m_plus.abs() * over + m_minus.abs() * under), dim=(1, 2))
    l_mc = torch.mean(sample_weight * mc_per_sample)
    l_regret = torch.mean(torch.relu(regret_norm))
    m_signed = 0.5 * (m_plus + m_minus)
    cost_bias_per_sample = torch.mean(m_signed * delta, dim=(1, 2))
    l_cost_bias = torch.mean(sample_weight * cost_bias_per_sample)
    l_pred = torch.mean(torch.abs(err))
    l_br = torch.mean((delta / (sigma[None, :, None] + EPS)) ** 2)
    l_bias = torch.abs(torch.mean(delta))
    l_boundary = raw.new_tensor(0.0)
    if settlement_under is None or settlement_over is None:
        l_settlement_system = raw.new_tensor(0.0)
        l_settlement_node = raw.new_tensor(0.0)
    else:
        # The RT market settles feeder-level imbalance.  System aggregation is
        # therefore the economically faithful primary proxy, while the optional
        # node term can retain local error-shape discipline.
        system_err = torch.sum(err, dim=1)
        settlement_system_per_sample = torch.mean(
            settlement_under * torch.relu(-system_err)
            + settlement_over * torch.relu(system_err),
            dim=1,
        ) / float(NUM_BUSES)
        l_settlement_system = torch.mean(sample_weight * settlement_system_per_sample)
        settlement_node_per_sample = torch.mean(
            settlement_under[:, None, :] * under
            + settlement_over[:, None, :] * over,
            dim=(1, 2),
        )
        l_settlement_node = torch.mean(sample_weight * settlement_node_per_sample)
        if b_map is not None:
            boundary_per_sample = torch.mean(
                b_map * (
                    settlement_under[:, None, :] * under
                    + settlement_over[:, None, :] * over
                ),
                dim=(1, 2),
            )
            l_boundary = torch.mean(sample_weight * boundary_per_sample)
    total = (
        HyperParams.bmc_loss_weight * l_mc
        + HyperParams.lambda_br * l_br
    )
    return {
        "total": total,
        "mc": l_mc,
        "regret": l_regret,
        "bounded_residual": l_br,
        "settlement_system": l_settlement_system,
        "settlement_node": l_settlement_node,
        "boundary": l_boundary,
        "cost_bias": l_cost_bias,
        "prediction_mae": l_pred,
        "bias": l_bias,
        "regret_weight": torch.mean(sample_weight),
    }


def evaluate_loader(model, loader, sigma_t, rho, device):
    vals = {k: [] for k in ["total", "mc", "regret", "bounded_residual", "settlement_system", "settlement_node", "boundary", "cost_bias", "prediction_mae", "bias", "regret_weight"]}
    model.eval()
    with torch.no_grad():
        for x, yp, yt, mp, mm, bm, rg, rw, su, so in loader:
            x, yp, yt, mp, mm, bm, rg, rw, su, so = x.to(device), yp.to(device), yt.to(device), mp.to(device), mm.to(device), bm.to(device), rg.to(device), rw.to(device), su.to(device), so.to(device)
            comps = loss_components(model(x), yp, yt, mp, mm, rg, sigma_t, rho, su, so, bm, rw)
            for k, v in comps.items():
                vals[k].append(float(v.item()))
    return {k: float(np.mean(v)) if v else float("inf") for k, v in vals.items()}


def train_corrector(round_id: int, round_dir: Path, rho: float, x, y_prev, y_true, m_plus, m_minus, regret_norm, sigma, mean_bus, std_bus, *, settlement_under=None, settlement_over=None, b_map=None):
    n = x.shape[0]
    n_train = max(1, int(n * 0.8))
    tensors = [torch.from_numpy(arr.astype(np.float32)) for arr in [x, y_prev, y_true, m_plus, m_minus]]
    if b_map is None:
        b_map = np.zeros_like(m_plus, dtype=np.float32)
    bm = torch.from_numpy(b_map.astype(np.float32))
    rg = torch.from_numpy(regret_norm.astype(np.float32))
    rw = torch.from_numpy(regret_sample_weights(regret_norm))
    if settlement_under is None or settlement_over is None:
        settlement_under = np.ones((n, HORIZON), dtype=np.float32)
        settlement_over = np.ones((n, HORIZON), dtype=np.float32)
    su = torch.from_numpy(settlement_under.astype(np.float32))
    so = torch.from_numpy(settlement_over.astype(np.float32))
    train_ds = TensorDataset(*(t[:n_train] for t in tensors), bm[:n_train], rg[:n_train], rw[:n_train], su[:n_train], so[:n_train])
    val_ds = TensorDataset(*(t[n_train:] for t in tensors), bm[n_train:], rg[n_train:], rw[n_train:], su[n_train:], so[n_train:])
    if len(val_ds) == 0:
        val_ds = train_ds
    train_loader = DataLoader(train_ds, batch_size=HyperParams.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=HyperParams.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_corrector_model().to(device)
    sigma_t = torch.from_numpy(sigma.astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HyperParams.learning_rate)

    best_val = float("inf")
    best_state = None
    bad = 0
    log_rows = []
    for epoch in range(1, HyperParams.epochs + 1):
        model.train()
        losses = []
        for xb, yp, yt, mp, mm, bm_batch, rgb, rwb, sub, sob in train_loader:
            xb, yp, yt, mp, mm, bm_batch, rgb, rwb, sub, sob = xb.to(device), yp.to(device), yt.to(device), mp.to(device), mm.to(device), bm_batch.to(device), rgb.to(device), rwb.to(device), sub.to(device), sob.to(device)
            opt.zero_grad(set_to_none=True)
            comps = loss_components(model(xb), yp, yt, mp, mm, rgb, sigma_t, rho, sub, sob, bm_batch, rwb)
            comps["total"].backward()
            opt.step()
            losses.append(float(comps["total"].item()))
        val = evaluate_loader(model, val_loader, sigma_t, rho, device)
        val_loss = val["total"]
        improved = val_loss < best_val - 1e-8
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        row = {"round": round_id, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "best_val_loss": best_val, "improved": bool(improved)}
        row.update({f"val_{k}_loss": v for k, v in val.items()})
        log_rows.append(row)
        if epoch == 1 or epoch % 20 == 0:
            print(f"[round {round_id}] epoch={epoch:03d} train={row['train_loss']:.6f} val={val_loss:.6f} best={best_val:.6f}")
        if bad >= HyperParams.patience:
            print(f"[round {round_id}] early stopping at epoch {epoch}")
            break
    if best_state is None:
        raise RuntimeError(f"No best model was saved for round {round_id}")
    model.load_state_dict(best_state)
    out_dir = round_dir / "corrector"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "round": round_id,
        "rho": rho,
        "sigma": sigma.tolist(),
        "mean_bus": mean_bus.tolist(),
        "std_bus": std_bus.tolist(),
        "input_channels": ["y_prev_scaled", "M_plus_norm/5", "M_minus_norm/5", "B_boundary"],
        "price_prediction_used": False,
        "price_as_nn_input": False,
        "price_used_in_dispatch": "real_day_ahead_price",
        "corrector_architecture": type(model).__name__,
        "corrector_architecture_key": str(HyperParams.corrector_architecture),
        "marginal_direction_mode": str(HyperParams.marginal_direction_mode),
        "boundary_channel_used": bool(HyperParams.use_boundary_channel),
        "loss": "L_final = bmc_loss_weight*L_mc(regret-weighted) + lambda_br*L_br",
        "loss_terms": {
            "L_mc": "mean(sample_weight(regret_norm) * mean_node_time((1 + boundary_kappa*B) * (M_plus*positive_error + M_minus*negative_error)))",
            "regret_signal": "max(realized_cost(current forecast)-oracle_cost, 0)/(abs(oracle_cost)+eps); precomputed per sample and used through sample_weight, not added as a standalone loss term",
            "L_br": "mean((bounded_residual / (bus_error_scale + eps))^2)",
        },
        "hyperparameters": {k: getattr(HyperParams, k) for k in ["bmc_loss_weight", "lambda_br", "regret_alpha", "regret_weight_cap", "beta", "gamma", "settlement_system_weight", "settlement_node_weight", "boundary_loss_weight", "lambda_cost_bias", "prediction_mae_weight", "settlement_under_multiplier", "settlement_over_multiplier", "settlement_weight_power", "perturb_eps", "boundary_kappa", "marginal_map_mode", "marginal_direction_mode", "use_boundary_channel", "node_sensitivity_blend", "use_extra_system_head", "extra_system_scale_init", "base_rho", "corrector_hidden_dim", "corrector_dropout", "corrector_architecture"]},
    }, out_dir / "model_best.pt")
    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    return model


def predict_corrected(model, x, y_prev, sigma, rho):
    device = next(model.parameters()).device
    sigma_t = torch.from_numpy(sigma.astype(np.float32)).to(device)
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=HyperParams.batch_size, shuffle=False, num_workers=0)
    deltas = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            raw = model(xb.to(device))
            deltas.append(compute_delta(raw, sigma_t, rho).detach().cpu().numpy())
    delta = np.concatenate(deltas, axis=0)
    return y_prev + delta, delta


def save_corrected_predictions(path: Path, split: str, data: dict[str, Any], y_prev: np.ndarray, y_corr: np.ndarray, delta: np.ndarray, m_plus: np.ndarray, m_minus: np.ndarray, b_map: np.ndarray, regret_norm: np.ndarray) -> pd.DataFrame:
    rows = []
    for sample_id, origin in enumerate(data["origins"]):
        idx = data["target_indices"][sample_id]
        for h, ts in enumerate(idx):
            for bus in range(1, NUM_BUSES + 1):
                rows.append({
                    "sample_id": sample_id,
                    "split": split,
                    "forecast_origin": origin,
                    "target_timestamp": iso(ts),
                    "horizon": h + 1,
                    "bus": bus,
                    "y_true_netload": float(data["y_true"][sample_id, bus - 1, h]),
                    "y_prev_netload": float(y_prev[sample_id, bus - 1, h]),
                    "y_corr_netload": float(y_corr[sample_id, bus - 1, h]),
                    "delta_netload": float(delta[sample_id, bus - 1, h]),
                    "M_plus_norm": float(m_plus[sample_id, bus - 1, h]),
                    "M_minus_norm": float(m_minus[sample_id, bus - 1, h]),
                    "B_boundary": float(b_map[sample_id, bus - 1, h]),
                    "sample_regret_norm": float(regret_norm[sample_id]),
                })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def evaluate_dispatch_for_predictions(label: str, data: dict[str, Any], y_dispatch_all: np.ndarray, network, price_series, solver, config, aligned) -> pd.DataFrame:
    rows = []
    for sample_id, origin in enumerate(data["origins"]):
        idx = data["target_indices"][sample_id]
        prices = price_series.reindex(idx)
        if prices.isna().any():
            result = {"ok": False, "status": "missing_real_price"}
        else:
            p_dispatch = array_to_pnet(y_dispatch_all[sample_id], idx)
            p_true = array_to_pnet(data["y_true"][sample_id], idx)
            result = dispatch_and_replay(p_dispatch, p_true, prices.to_numpy(dtype=float), network, solver, config, aligned=aligned)
        if result.get("ok"):
            rows.append({
                "sample_id": sample_id,
                "forecast_origin": origin,
                "method": label,
                "realized_cost": float(result["realized_cost"]),
                "voltage_violation": float(result["voltage_violation"]),
                "line_overload": float(result["line_overload"]),
                "status": result["status"],
            })
        else:
            rows.append({
                "sample_id": sample_id,
                "forecast_origin": origin,
                "method": label,
                "realized_cost": np.nan,
                "voltage_violation": np.nan,
                "line_overload": np.nan,
                "status": result.get("status", "failed"),
            })
        print(f"[DEC {label}] sample={sample_id:03d} status={rows[-1]['status']}")
    return pd.DataFrame(rows)


def compute_dec_metrics(data, y_pred, y_corr, m_plus, m_minus, dispatch_df: pd.DataFrame) -> dict[str, Any]:
    y_true = data["y_true"]
    feasible = dispatch_df[dispatch_df["realized_cost"].notna()].copy()
    m_avg = np.abs(0.5 * (m_plus + m_minus))
    return {
        "MAE": float(np.mean(np.abs(y_corr - y_true))),
        "FGMC_weighted_MAE": weighted_mae(y_true, y_corr, m_avg),
        "mean_realized_cost": float(feasible["realized_cost"].mean()) if len(feasible) else np.nan,
        "mean_voltage_violation": float(feasible["voltage_violation"].mean()) if len(feasible) else np.nan,
        "mean_line_overload": float(feasible["line_overload"].mean()) if len(feasible) else np.nan,
        "num_feasible_samples": int(len(feasible)),
    }


def baseline_metrics(data, dispatch_df):
    feasible = dispatch_df[dispatch_df["realized_cost"].notna()].copy()
    y_true, y_pred = data["y_true"], data["y_pred"]
    return {
        "MAE": float(np.mean(np.abs(y_pred - y_true))),
        "FGMC_weighted_MAE": np.nan,
        "mean_realized_cost": float(feasible["realized_cost"].mean()) if len(feasible) else np.nan,
        "mean_voltage_violation": float(feasible["voltage_violation"].mean()) if len(feasible) else np.nan,
        "mean_line_overload": float(feasible["line_overload"].mean()) if len(feasible) else np.nan,
        "num_feasible_samples": int(len(feasible)),
    }


def select_best(summary: pd.DataFrame) -> dict[str, Any]:
    rounds = summary[summary["method"].str.startswith("FGMC-GFCN-K")].copy()
    rounds = rounds[rounds["mean_realized_cost"].notna()]
    if rounds.empty:
        raise RuntimeError("No feasible FGMC-GFCN rounds available for best-K selection.")
    best_row = rounds.sort_values("mean_realized_cost").iloc[0].to_dict()
    return {"best_K": int(best_row["K"]), "selection_rule": "minimum December mean_realized_cost", "best_metrics": best_row}


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for ax, col, ylabel in [
        (axes[0], "mean_realized_cost", "AUD"),
        (axes[1], "MAE", "MAE"),
        (axes[2], "mean_voltage_violation", "voltage violation (ref.)"),
        (axes[3], "mean_line_overload", "line overload (ref.)"),
    ]:
        ax.bar(x, summary[col].to_numpy(dtype=float))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(methods, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "december_method_comparison.png", dpi=180)
    plt.close(fig)

    rounds = summary[summary["method"].str.startswith("FGMC-GFCN-K")].copy()
    if not rounds.empty:
        plt.figure(figsize=(10, 5))
        for col in ["MAE", "FGMC_weighted_MAE", "mean_realized_cost"]:
            vals = rounds[col].to_numpy(dtype=float)
            mn, mx = np.nanmin(vals), np.nanmax(vals)
            norm = np.zeros_like(vals) if mx - mn < EPS else (vals - mn) / (mx - mn)
            plt.plot(rounds["K"], norm, marker="o", label=col)
        plt.xlabel("Closed-loop round K")
        plt.ylabel("min-max normalized value")
        plt.title("December FGMC-GFCN correction metrics vs K")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "iterative_metrics_vs_K_december.png", dpi=180)
        plt.close()

