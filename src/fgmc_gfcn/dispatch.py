from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo


HORIZON = 24
NUM_BUSES = 33
SBASE_MVA = 1.0
VMIN = 0.95**2
VMAX = 1.05**2
EPS = 1e-8
BUS_COLS = [f"netload_bus_{i}" for i in range(1, NUM_BUSES + 1)]
LOAD_COLS = [f"load_bus_{i}" for i in range(1, NUM_BUSES + 1)]
PV_COLS = [f"pv_bus_{i}" for i in range(1, NUM_BUSES + 1)]


@dataclass(frozen=True)
class DGParam:
    bus: int
    Pmin: float
    Pmax: float
    Smax: float
    ramp: float
    a: float
    b: float


@dataclass(frozen=True)
class ESSParam:
    bus: int
    Pmax: float
    Emax: float
    Emin: float
    E0: float
    Smax: float
    eta_ch: float = 0.95
    eta_dis: float = 0.95


@dataclass
class DispatchConfig:
    data_dir: Path = Path("data") / "processed_1h"
    line_limit_default_mva: float = 7.0
    storage_degradation_cost: float = 30.0
    sell_price_ratio: float = 0.7
    rt_buy_price_adder: float = 50.0
    rt_buy_price_multiplier: float = 1.0
    rt_sell_price_ratio: float = 0.5
    rt_imbalance_fee: float = 0.0
    tiny_grid_split_cost: float = 1e-4
    # Realized replay / real-time recourse settings.
    # In replay, the day-ahead grid buy/sell schedule is fixed, while local
    # flexible resources are allowed to redispatch around their day-ahead values.
    enable_replay_recourse: bool = True
    rt_dg_adjustment_cost: float = 5.0
    rt_ess_adjustment_cost: float = 5.0
    rt_q_adjustment_cost: float = 0.1
    pv_curtailment_cost: float = 300.0
    time_limit: int = 120
    threads: int = 1
    qcp_dual: int = 1
    dg_params: dict[str, DGParam] = field(default_factory=lambda: {
        "DG1": DGParam(bus=6, Pmin=0.00, Pmax=0.80, Smax=0.90, ramp=0.30, a=18.0, b=45.0),
        "DG2": DGParam(bus=14, Pmin=0.00, Pmax=0.60, Smax=0.70, ramp=0.25, a=25.0, b=55.0),
        "DG3": DGParam(bus=30, Pmin=0.00, Pmax=0.50, Smax=0.60, ramp=0.20, a=35.0, b=65.0),
    })
    ess_params: dict[str, ESSParam] = field(default_factory=lambda: {
        "ESS1": ESSParam(bus=18, Pmax=0.40, Emax=1.20, Emin=0.12, E0=0.60, Smax=0.45),
        "ESS2": ESSParam(bus=25, Pmax=0.35, Emax=1.00, Emin=0.10, E0=0.50, Smax=0.40),
        "ESS3": ESSParam(bus=33, Pmax=0.30, Emax=0.80, Emin=0.08, E0=0.40, Smax=0.35),
    })
    pv_caps: dict[int, float] = field(default_factory=lambda: {
        6: 0.30,
        13: 0.25,
        18: 0.30,
        25: 0.40,
        30: 0.35,
    })

    @property
    def pv_smax(self) -> dict[int, float]:
        return {bus: 1.1 * cap for bus, cap in self.pv_caps.items()}


@dataclass
class NetworkData:
    bus: pd.DataFrame
    branch_all: pd.DataFrame
    active_branch: pd.DataFrame
    q_ratio: dict[int, float]
    downstream_by_line: dict[int, set[int]]
    voltage_sens: np.ndarray
    voltage_sens_qaware: np.ndarray


@dataclass
class DispatchSolution:
    status: str
    termination: str
    objective: float
    economic_cost: float
    P_DG: dict[str, list[float]]
    Q_DG: dict[str, list[float]]
    P_ch: dict[str, list[float]]
    P_dis: dict[str, list[float]]
    Q_ESS: dict[str, list[float]]
    SOC: dict[str, list[float]]
    Q_PV: dict[int, list[float]]
    P_PV_curt: dict[int, list[float]]
    P_grid_buy: list[float]
    P_grid_sell: list[float]
    P_rt_buy: list[float]
    P_rt_sell: list[float]
    P_grid: list[float]
    Q_grid: list[float]
    V: dict[int, list[float]]
    P_line: dict[int, list[float]]
    Q_line: dict[int, list[float]]
    S_line: dict[int, list[float]]
    I_sq_line: dict[int, list[float]]
    I_line: dict[int, list[float]]
    current_limit: dict[int, float]
    soc_relax_gap: dict[int, list[float]]
    max_soc_relax_gap: float
    max_voltage_violation: float
    max_line_overload: float
    raw: dict[str, Any] = field(default_factory=dict)


def iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _value(v: Any) -> float:
    val = pyo.value(v, exception=False)
    try:
        out = float(val)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def load_network(config: DispatchConfig | None = None) -> NetworkData:
    config = config or DispatchConfig()
    bus = pd.read_csv(config.data_dir / "feeder_bus.csv")
    branch_all = pd.read_csv(config.data_dir / "feeder_branch.csv")
    active = branch_all.loc[branch_all["status"].eq(1)].copy().reset_index(drop=True)
    if len(bus) != NUM_BUSES:
        raise ValueError(f"Expected {NUM_BUSES} buses, found {len(bus)}")
    if len(active) != NUM_BUSES - 1:
        raise ValueError(f"Expected {NUM_BUSES - 1} active radial branches, found {len(active)}")

    base_kv = float(bus["baseKV"].replace(0, np.nan).dropna().iloc[0])
    z_base_ohm = base_kv**2 / SBASE_MVA
    active["line_id"] = np.arange(len(active))
    active["r_pu"] = active["r_ohm"].astype(float) / z_base_ohm
    active["x_pu"] = active["x_ohm"].astype(float) / z_base_ohm
    active["limit_MVA"] = active["rateA"].where(active["rateA"] > 0, config.line_limit_default_mva).astype(float)
    # Current-limit form used by the DistFlow/SOCP model.  With per-unit voltage
    # and power in MW on SBASE_MVA, l_ij = |I_ij|^2 is bounded by (Smax/Sbase)^2
    # when no explicit ampacity is available.
    active["limit_I_sq"] = (active["limit_MVA"] / SBASE_MVA) ** 2

    q_ratio = {}
    for row in bus.itertuples(index=False):
        pd_mw = float(getattr(row, "Pd_MW"))
        qd_mvar = float(getattr(row, "Qd_MVAr"))
        q_ratio[int(getattr(row, "bus_i"))] = 0.0 if pd_mw <= EPS else qd_mvar / pd_mw

    downstream_by_line, voltage_sens, voltage_sens_qaware = build_tree_maps(active, q_ratio)
    return NetworkData(
        bus=bus,
        branch_all=branch_all,
        active_branch=active,
        q_ratio=q_ratio,
        downstream_by_line=downstream_by_line,
        voltage_sens=voltage_sens,
        voltage_sens_qaware=voltage_sens_qaware,
    )


def build_tree_maps(
    branch: pd.DataFrame,
    q_ratio: dict[int, float] | None = None,
) -> tuple[dict[int, set[int]], np.ndarray, np.ndarray]:
    q_ratio = q_ratio or {i: 0.0 for i in range(1, NUM_BUSES + 1)}
    children: dict[int, list[tuple[int, int]]] = {bus: [] for bus in range(1, NUM_BUSES + 1)}
    parent_line: dict[int, int] = {}
    parent_bus: dict[int, int] = {}
    r_by_line: dict[int, float] = {}
    x_by_line: dict[int, float] = {}
    for row in branch.itertuples(index=False):
        lid = int(row.line_id)
        fbus = int(row.fbus)
        tbus = int(row.tbus)
        children[fbus].append((tbus, lid))
        parent_bus[tbus] = fbus
        parent_line[tbus] = lid
        r_by_line[lid] = float(row.r_pu)
        x_by_line[lid] = float(row.x_pu)

    def descendants(start_bus: int) -> set[int]:
        out = {start_bus}
        stack = [start_bus]
        while stack:
            bus = stack.pop()
            for child, _ in children.get(bus, []):
                if child not in out:
                    out.add(child)
                    stack.append(child)
        return out

    downstream_by_line: dict[int, set[int]] = {}
    for row in branch.itertuples(index=False):
        downstream_by_line[int(row.line_id)] = descendants(int(row.tbus))

    path_lines: dict[int, list[int]] = {1: []}
    for bus in range(2, NUM_BUSES + 1):
        path: list[int] = []
        cur = bus
        while cur != 1 and cur in parent_line:
            path.append(parent_line[cur])
            cur = parent_bus[cur]
        path_lines[bus] = list(reversed(path))

    voltage_sens = np.zeros((NUM_BUSES, NUM_BUSES), dtype=float)
    voltage_sens_qaware = np.zeros_like(voltage_sens)
    for j in range(1, NUM_BUSES + 1):
        path_j = set(path_lines[j])
        for i in range(1, NUM_BUSES + 1):
            common = path_j.intersection(path_lines[i])
            voltage_sens[j - 1, i - 1] = 2.0 * sum(r_by_line[lid] for lid in common) / SBASE_MVA
            voltage_sens_qaware[j - 1, i - 1] = 2.0 * sum(
                r_by_line[lid] + x_by_line[lid] * q_ratio.get(i, 0.0) for lid in common
            ) / SBASE_MVA
    return downstream_by_line, voltage_sens, voltage_sens_qaware


def load_price_series(config: DispatchConfig | None = None) -> pd.Series:
    config = config or DispatchConfig()
    df = pd.read_csv(config.data_dir / "price_timeseries.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["price_AUD_per_MWh"].sort_index()


def load_aligned_dataset(config: DispatchConfig | None = None) -> pd.DataFrame | None:
    config = config or DispatchConfig()
    path = config.data_dir / "aligned_dataset.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def choose_solver(config: DispatchConfig | None = None) -> tuple[str, pyo.SolverFactory]:
    config = config or DispatchConfig()
    solver = pyo.SolverFactory("gurobi")
    if not solver.available(False):
        raise RuntimeError("Gurobi is required for the active-reactive SOCP/QCP dispatch model.")
    solver.options["TimeLimit"] = int(config.time_limit)
    solver.options["Threads"] = int(config.threads)
    solver.options["QCPDual"] = int(config.qcp_dual)
    # Keep the model convex.  Do not set NonConvex=2 unless you intentionally add nonconvex constraints.
    return "gurobi", solver


def _standardize_bus_frame(df: pd.DataFrame | None, prefix: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = [f"{prefix}_bus_{i}" for i in range(1, NUM_BUSES + 1)]
    if df is None:
        return pd.DataFrame(0.0, index=index, columns=cols)
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.set_index("timestamp")
    out = out.sort_index()
    # Accept both load_bus_i and bus_i style columns.
    if all(col in out.columns for col in cols):
        return out.reindex(index)[cols].astype(float).fillna(0.0)
    bus_cols = [f"bus_{i}" for i in range(1, NUM_BUSES + 1)]
    if all(col in out.columns for col in bus_cols):
        renamed = out[bus_cols].rename(columns={f"bus_{i}": f"{prefix}_bus_{i}" for i in range(1, NUM_BUSES + 1)})
        return renamed.reindex(index)[cols].astype(float).fillna(0.0)
    return pd.DataFrame(0.0, index=index, columns=cols)


def lookup_load_pv(
    index: pd.DatetimeIndex,
    pnet: pd.DataFrame,
    aligned: pd.DataFrame | None,
    load: pd.DataFrame | None = None,
    pv: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if load is not None or pv is not None:
        load_df = _standardize_bus_frame(load, "load", index)
        pv_df = _standardize_bus_frame(pv, "pv", index)
        if load is None:
            # fallback load = netload + pv
            load_df = pnet.copy()
            load_df.columns = [f"load_bus_{i}" for i in range(1, NUM_BUSES + 1)]
            load_df = load_df + pv_df.to_numpy()
        return load_df, pv_df
    if aligned is not None:
        load_cols = [f"load_bus_{i}" for i in range(1, NUM_BUSES + 1)]
        pv_cols = [f"pv_bus_{i}" for i in range(1, NUM_BUSES + 1)]
        if all(c in aligned.columns for c in load_cols + pv_cols):
            load_df = aligned.reindex(index)[load_cols].astype(float)
            pv_df = aligned.reindex(index)[pv_cols].astype(float)
            if not load_df.isna().any().any() and not pv_df.isna().any().any():
                return load_df, pv_df
    # Minimal fallback: infer nonnegative load from netload and set PV to zero.
    load_df = pd.DataFrame(
        np.maximum(pnet.to_numpy(dtype=float), 0.0),
        index=index,
        columns=[f"load_bus_{i}" for i in range(1, NUM_BUSES + 1)],
    )
    pv_df = pd.DataFrame(0.0, index=index, columns=[f"pv_bus_{i}" for i in range(1, NUM_BUSES + 1)])
    return load_df, pv_df


def build_dispatch_model(
    pnet: pd.DataFrame,
    prices: np.ndarray,
    network: NetworkData,
    config: DispatchConfig | None = None,
    *,
    fixed: dict[str, Any] | None = None,
    aligned: pd.DataFrame | None = None,
    load: pd.DataFrame | None = None,
    pv: pd.DataFrame | None = None,
    import_duals: bool = False,
) -> pyo.ConcreteModel:
    config = config or DispatchConfig()
    if len(pnet) != HORIZON:
        raise ValueError(f"Expected {HORIZON} pnet rows, got {len(pnet)}")
    pnet = pnet.copy()
    pnet.index = pd.DatetimeIndex(pd.to_datetime(pnet.index, utc=True), name="timestamp")
    pnet = pnet[BUS_COLS].astype(float)
    prices = np.asarray(prices, dtype=float)
    if len(prices) != HORIZON:
        raise ValueError(f"Expected {HORIZON} prices, got {len(prices)}")
    positive_prices = np.maximum(prices, 0.0)
    rt_buy_prices = prices + config.rt_buy_price_adder + (config.rt_buy_price_multiplier - 1.0) * positive_prices
    rt_sell_prices = config.rt_sell_price_ratio * prices
    effective_rt_buy_prices = rt_buy_prices + config.rt_imbalance_fee
    effective_rt_sell_prices = rt_sell_prices - config.rt_imbalance_fee
    if np.any(effective_rt_buy_prices <= effective_rt_sell_prices):
        raise ValueError(
            "Real-time imbalance prices permit unbounded simultaneous buy/sell. "
            "Increase rt_buy_price_adder/rt_imbalance_fee or reduce rt_sell_price_ratio."
        )

    load_df, pv_df = lookup_load_pv(pnet.index, pnet, aligned, load=load, pv=pv)
    # PV inverter capacities must cover observed PV power.
    pv_caps = dict(config.pv_caps)
    for bus in pv_caps:
        col = f"pv_bus_{bus}"
        if col in pv_df.columns:
            pv_caps[bus] = max(float(pv_caps[bus]), float(pv_df[col].max()))
    pv_smax = {bus: 1.1 * cap for bus, cap in pv_caps.items()}

    bus_ids = list(range(1, NUM_BUSES + 1))
    times = list(range(HORIZON))
    line_ids = network.active_branch["line_id"].astype(int).tolist()
    dg_names = list(config.dg_params)
    ess_names = list(config.ess_params)
    pv_buses = sorted(pv_caps)

    pnet_dict = {(bus, t): float(pnet.iloc[t][f"netload_bus_{bus}"]) for bus in bus_ids for t in times}
    qload_dict = {}
    ppv_dict = {}
    for bus in bus_ids:
        qratio = network.q_ratio.get(bus, 0.0)
        for t in times:
            qload_dict[(bus, t)] = qratio * max(float(load_df.iloc[t][f"load_bus_{bus}"]), 0.0)
            ppv_dict[(bus, t)] = float(pv_df.iloc[t][f"pv_bus_{bus}"]) if f"pv_bus_{bus}" in pv_df.columns else 0.0

    incoming = {bus: [] for bus in bus_ids}
    outgoing = {bus: [] for bus in bus_ids}
    branch_lookup = {}
    for row in network.active_branch.itertuples(index=False):
        lid = int(row.line_id)
        outgoing[int(row.fbus)].append(lid)
        incoming[int(row.tbus)].append(lid)
        branch_lookup[lid] = row

    dg_at_bus = {bus: [] for bus in bus_ids}
    for name, p in config.dg_params.items():
        dg_at_bus[p.bus].append(name)
    ess_at_bus = {bus: [] for bus in bus_ids}
    for name, p in config.ess_params.items():
        ess_at_bus[p.bus].append(name)
    pv_at_bus = {bus: (bus in pv_buses) for bus in bus_ids}

    m = pyo.ConcreteModel()
    if import_duals:
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.T = pyo.Set(initialize=times, ordered=True)
    m.TE = pyo.Set(initialize=list(range(HORIZON + 1)), ordered=True)
    m.B = pyo.Set(initialize=bus_ids, ordered=True)
    m.L = pyo.Set(initialize=line_ids, ordered=True)
    m.G = pyo.Set(initialize=dg_names, ordered=True)
    m.E = pyo.Set(initialize=ess_names, ordered=True)
    m.PV = pyo.Set(initialize=pv_buses, ordered=True)

    m.P = pyo.Var(m.L, m.T)
    m.Q = pyo.Var(m.L, m.T)
    # Squared voltage magnitude v_i,t.  A nonnegative bound helps solvers
    # recognize the rotated-SOC branch-flow relaxation as convex.
    m.V = pyo.Var(m.B, m.T, bounds=(0, None))
    # Squared branch current l_ij,t = |I_ij,t|^2 for DistFlow/SOCP.
    m.I_sq = pyo.Var(m.L, m.T, bounds=(0, None))
    # Day-ahead import/export schedule and real-time imbalance settlement.
    # The RT components are fixed to zero during day-ahead optimization.  During
    # replay, the DA schedule is fixed and only RT imbalance can balance errors.
    m.P_grid_buy = pyo.Var(m.T, bounds=(0, None))
    m.P_grid_sell = pyo.Var(m.T, bounds=(0, None))
    m.P_rt_buy = pyo.Var(m.T, bounds=(0, None))
    m.P_rt_sell = pyo.Var(m.T, bounds=(0, None))
    m.Q_grid = pyo.Var(m.T)
    m.P_DG = pyo.Var(m.G, m.T)
    m.Q_DG = pyo.Var(m.G, m.T)
    m.P_ch = pyo.Var(m.E, m.T)
    m.P_dis = pyo.Var(m.E, m.T)
    m.Q_ESS = pyo.Var(m.E, m.T)
    m.SOC = pyo.Var(m.E, m.TE)
    m.Q_PV = pyo.Var(m.PV, m.T)
    # Active PV curtailment. Since pnet = load - available PV, curtailing PV
    # increases the effective net load by P_PV_curt.
    m.P_PV_curt = pyo.Var(m.PV, m.T, bounds=(0, None))
    def pv_curtailment_bounds_rule(model, b, t):
        return model.P_PV_curt[b, t] <= max(ppv_dict[(int(b), t)], 0.0)
    m.pv_curtailment_bounds = pyo.Constraint(m.PV, m.T, rule=pv_curtailment_bounds_rule)

    def dg_bounds_rule(model, g, t):
        p = config.dg_params[g]
        return pyo.inequality(p.Pmin, model.P_DG[g, t], p.Pmax)
    m.dg_bounds = pyo.Constraint(m.G, m.T, rule=dg_bounds_rule)

    def dg_ramp_up_rule(model, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return model.P_DG[g, t] - model.P_DG[g, t - 1] <= config.dg_params[g].ramp
    def dg_ramp_down_rule(model, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return model.P_DG[g, t - 1] - model.P_DG[g, t] <= config.dg_params[g].ramp
    m.dg_ramp_up = pyo.Constraint(m.G, m.T, rule=dg_ramp_up_rule)
    m.dg_ramp_down = pyo.Constraint(m.G, m.T, rule=dg_ramp_down_rule)

    def dg_apparent_rule(model, g, t):
        p = config.dg_params[g]
        return model.P_DG[g, t] ** 2 + model.Q_DG[g, t] ** 2 <= p.Smax**2
    m.dg_apparent = pyo.Constraint(m.G, m.T, rule=dg_apparent_rule)

    def ess_ch_bounds_rule(model, e, t):
        return pyo.inequality(0, model.P_ch[e, t], config.ess_params[e].Pmax)
    def ess_dis_bounds_rule(model, e, t):
        return pyo.inequality(0, model.P_dis[e, t], config.ess_params[e].Pmax)
    def ess_soc_bounds_rule(model, e, t):
        p = config.ess_params[e]
        return pyo.inequality(p.Emin, model.SOC[e, t], p.Emax)
    m.ess_ch_bounds = pyo.Constraint(m.E, m.T, rule=ess_ch_bounds_rule)
    m.ess_dis_bounds = pyo.Constraint(m.E, m.T, rule=ess_dis_bounds_rule)
    m.soc_bounds = pyo.Constraint(m.E, m.TE, rule=ess_soc_bounds_rule)

    def soc_initial_rule(model, e):
        return model.SOC[e, 0] == config.ess_params[e].E0
    def soc_terminal_rule(model, e):
        return model.SOC[e, HORIZON] == config.ess_params[e].E0
    def soc_dynamic_rule(model, e, t):
        p = config.ess_params[e]
        return model.SOC[e, t + 1] == model.SOC[e, t] + p.eta_ch * model.P_ch[e, t] - model.P_dis[e, t] / p.eta_dis
    m.soc_initial = pyo.Constraint(m.E, rule=soc_initial_rule)
    m.soc_terminal = pyo.Constraint(m.E, rule=soc_terminal_rule)
    m.soc_dynamic = pyo.Constraint(m.E, m.T, rule=soc_dynamic_rule)

    def ess_apparent_rule(model, e, t):
        p = config.ess_params[e]
        return (model.P_dis[e, t] - model.P_ch[e, t]) ** 2 + model.Q_ESS[e, t] ** 2 <= p.Smax**2
    m.ess_apparent = pyo.Constraint(m.E, m.T, rule=ess_apparent_rule)

    def pv_apparent_rule(model, b, t):
        p_pv_used = ppv_dict[(int(b), t)] - model.P_PV_curt[b, t]
        return p_pv_used ** 2 + model.Q_PV[b, t] ** 2 <= pv_smax[int(b)] ** 2
    m.pv_apparent = pyo.Constraint(m.PV, m.T, rule=pv_apparent_rule)

    def p_balance_rule(model, b, t):
        # DistFlow active-power balance: the power received from an incoming
        # branch equals sending-end power minus r*l losses.
        inflow = sum(
            model.P[l, t] - float(branch_lookup[int(l)].r_pu) * model.I_sq[l, t] * SBASE_MVA
            for l in incoming[b]
        )
        outflow = sum(model.P[l, t] for l in outgoing[b])
        dg = sum(model.P_DG[g, t] for g in dg_at_bus[b])
        ess = sum(model.P_dis[e, t] - model.P_ch[e, t] for e in ess_at_bus[b])
        pv_curt = model.P_PV_curt[b, t] if pv_at_bus[b] else 0.0
        grid = (
            model.P_grid_buy[t] - model.P_grid_sell[t]
            + model.P_rt_buy[t] - model.P_rt_sell[t]
        ) if b == 1 else 0.0
        return inflow + grid + dg + ess - pnet_dict[(b, t)] - pv_curt - outflow == 0

    def q_balance_rule(model, b, t):
        # DistFlow reactive-power balance with x*l reactive losses.
        inflow = sum(
            model.Q[l, t] - float(branch_lookup[int(l)].x_pu) * model.I_sq[l, t] * SBASE_MVA
            for l in incoming[b]
        )
        outflow = sum(model.Q[l, t] for l in outgoing[b])
        dg = sum(model.Q_DG[g, t] for g in dg_at_bus[b])
        ess = sum(model.Q_ESS[e, t] for e in ess_at_bus[b])
        pv_q = model.Q_PV[b, t] if pv_at_bus[b] else 0.0
        grid = model.Q_grid[t] if b == 1 else 0.0
        return inflow + grid + dg + ess + pv_q - qload_dict[(b, t)] - outflow == 0

    m.p_balance = pyo.Constraint(m.B, m.T, rule=p_balance_rule)
    m.q_balance = pyo.Constraint(m.B, m.T, rule=q_balance_rule)

    def slack_voltage_rule(model, t):
        return model.V[1, t] == 1.0
    m.slack_voltage = pyo.Constraint(m.T, rule=slack_voltage_rule)

    def voltage_drop_rule(model, l, t):
        row = branch_lookup[int(l)]
        r = float(row.r_pu)
        x = float(row.x_pu)
        return model.V[int(row.tbus), t] == model.V[int(row.fbus), t] - 2.0 * (
            r * model.P[l, t] / SBASE_MVA + x * model.Q[l, t] / SBASE_MVA
        ) + (r**2 + x**2) * model.I_sq[l, t]
    m.voltage_drop = pyo.Constraint(m.L, m.T, rule=voltage_drop_rule)

    def voltage_lower_rule(model, b, t):
        return model.V[b, t] >= VMIN
    def voltage_upper_rule(model, b, t):
        return model.V[b, t] <= VMAX
    m.voltage_lower = pyo.Constraint(m.B, m.T, rule=voltage_lower_rule)
    m.voltage_upper = pyo.Constraint(m.B, m.T, rule=voltage_upper_rule)

    def branch_soc_rule(model, l, t):
        # SOCP relaxation of P^2+Q^2 = v*l, written in quadratic form.
        row = branch_lookup[int(l)]
        fbus = int(row.fbus)
        return (model.P[l, t] / SBASE_MVA) ** 2 + (model.Q[l, t] / SBASE_MVA) ** 2 <= model.V[fbus, t] * model.I_sq[l, t]
    m.branch_soc = pyo.Constraint(m.L, m.T, rule=branch_soc_rule)

    def current_limit_rule(model, l, t):
        limit_i_sq = float(branch_lookup[int(l)].limit_I_sq)
        return model.I_sq[l, t] <= limit_i_sq
    m.current_limit = pyo.Constraint(m.L, m.T, rule=current_limit_rule)

    # With negative prices, simultaneous unbounded buy/sell could create an
    # artificial arbitrage direction.  Export is therefore disabled in those
    # hours while import remains available to serve the feeder.
    def no_sell_negative_price_rule(model, t):
        if prices[t] < 0:
            return model.P_grid_sell[t] == 0
        return pyo.Constraint.Skip
    m.no_sell_negative_price = pyo.Constraint(m.T, rule=no_sell_negative_price_rule)

    is_replay = bool(fixed and "P_grid_buy" in fixed and "P_grid_sell" in fixed)
    if is_replay:
        for t in times:
            m.P_grid_buy[t].fix(float(fixed["P_grid_buy"][t]))
            m.P_grid_sell[t].fix(float(fixed["P_grid_sell"][t]))
    else:
        # Real-time imbalance is unavailable while constructing the day-ahead
        # schedule.  It is enabled only when replaying realized net load.
        for t in times:
            m.P_rt_buy[t].fix(0.0)
            m.P_rt_sell[t].fix(0.0)

    if fixed and not config.enable_replay_recourse:
        # Backward-compatible replay mode: all local dispatch decisions are fixed
        # at their day-ahead values, and only P_rt_buy/P_rt_sell can balance active
        # forecast errors.
        for g in dg_names:
            for t in times:
                if "P_DG" in fixed:
                    m.P_DG[g, t].fix(float(fixed["P_DG"][g][t]))
                if "Q_DG" in fixed:
                    m.Q_DG[g, t].fix(float(fixed["Q_DG"][g][t]))
        for e in ess_names:
            for t in times:
                if "P_ch" in fixed:
                    m.P_ch[e, t].fix(float(fixed["P_ch"][e][t]))
                if "P_dis" in fixed:
                    m.P_dis[e, t].fix(float(fixed["P_dis"][e][t]))
                if "Q_ESS" in fixed:
                    m.Q_ESS[e, t].fix(float(fixed["Q_ESS"][e][t]))
        for b in pv_buses:
            for t in times:
                if "Q_PV" in fixed:
                    # fixed may use int or str keys.
                    qpv = fixed["Q_PV"].get(b, fixed["Q_PV"].get(str(b), None))
                    if qpv is not None:
                        m.Q_PV[b, t].fix(float(qpv[t]))
                if "P_PV_curt" in fixed:
                    curt = fixed["P_PV_curt"].get(b, fixed["P_PV_curt"].get(str(b), None))
                    if curt is not None:
                        m.P_PV_curt[b, t].fix(float(curt[t]))

    dg_cost = sum(
        config.dg_params[g].a * m.P_DG[g, t] ** 2 + config.dg_params[g].b * m.P_DG[g, t]
        for g in dg_names for t in times
    )
    grid_cost = sum(
        float(prices[t]) * m.P_grid_buy[t]
        - config.sell_price_ratio * float(prices[t]) * m.P_grid_sell[t]
        for t in times
    )
    rt_imbalance_cost = sum(
        float(rt_buy_prices[t]) * m.P_rt_buy[t]
        - float(rt_sell_prices[t]) * m.P_rt_sell[t]
        for t in times
    )
    imbalance_fee_cost = config.rt_imbalance_fee * sum(m.P_rt_buy[t] + m.P_rt_sell[t] for t in times)
    storage_deg = config.storage_degradation_cost * sum(m.P_ch[e, t] + m.P_dis[e, t] for e in ess_names for t in times)
    pv_curt_cost = config.pv_curtailment_cost * sum(m.P_PV_curt[b, t] for b in pv_buses for t in times)
    grid_split = config.tiny_grid_split_cost * sum(
        m.P_grid_buy[t] + m.P_grid_sell[t] + m.P_rt_buy[t] + m.P_rt_sell[t]
        for t in times
    )

    # When replaying a realized day, the DA grid schedule is fixed but local
    # resources can redispatch. These convex quadratic terms discourage arbitrary
    # deviation from the DA plan and represent real-time regulation effort.
    rt_adjustment_cost = 0.0
    if is_replay and config.enable_replay_recourse and fixed:
        if "P_DG" in fixed:
            rt_adjustment_cost += config.rt_dg_adjustment_cost * sum(
                (m.P_DG[g, t] - float(fixed["P_DG"][g][t])) ** 2
                for g in dg_names for t in times
            )
        if "Q_DG" in fixed:
            rt_adjustment_cost += config.rt_q_adjustment_cost * sum(
                (m.Q_DG[g, t] - float(fixed["Q_DG"][g][t])) ** 2
                for g in dg_names for t in times
            )
        if "P_ch" in fixed:
            rt_adjustment_cost += config.rt_ess_adjustment_cost * sum(
                (m.P_ch[e, t] - float(fixed["P_ch"][e][t])) ** 2
                for e in ess_names for t in times
            )
        if "P_dis" in fixed:
            rt_adjustment_cost += config.rt_ess_adjustment_cost * sum(
                (m.P_dis[e, t] - float(fixed["P_dis"][e][t])) ** 2
                for e in ess_names for t in times
            )
        if "Q_ESS" in fixed:
            rt_adjustment_cost += config.rt_q_adjustment_cost * sum(
                (m.Q_ESS[e, t] - float(fixed["Q_ESS"][e][t])) ** 2
                for e in ess_names for t in times
            )
        if "Q_PV" in fixed:
            rt_adjustment_cost += config.rt_q_adjustment_cost * sum(
                (m.Q_PV[b, t] - float(fixed["Q_PV"].get(b, fixed["Q_PV"].get(str(b)))[t])) ** 2
                for b in pv_buses for t in times
                if fixed["Q_PV"].get(b, fixed["Q_PV"].get(str(b), None)) is not None
            )

    m.objective = pyo.Objective(
        expr=dg_cost + grid_cost + rt_imbalance_cost + storage_deg
        + imbalance_fee_cost + pv_curt_cost + rt_adjustment_cost + grid_split
    )

    m._dispatch_context = {
        "pnet_dict": pnet_dict,
        "qload_dict": qload_dict,
        "ppv_dict": ppv_dict,
        "branch_lookup": branch_lookup,
        "pv_caps": pv_caps,
        "pv_smax": pv_smax,
        "config": config,
    }
    return m


def solve_model(model: pyo.ConcreteModel, solver) -> tuple[str, str]:
    result = solver.solve(model, tee=False)
    return str(result.solver.status), str(result.solver.termination_condition)


def is_optimal(status: str, termination: str) -> bool:
    return "optimal" in f"{status}/{termination}".lower()


def economic_cost_from_solution(sol: dict[str, Any], prices: np.ndarray, config: DispatchConfig) -> float:
    total = 0.0
    for t in range(HORIZON):
        total += float(prices[t]) * sol["P_grid_buy"][t]
        total -= config.sell_price_ratio * float(prices[t]) * sol["P_grid_sell"][t]
        positive_price = max(float(prices[t]), 0.0)
        rt_buy_price = float(prices[t]) + config.rt_buy_price_adder + (config.rt_buy_price_multiplier - 1.0) * positive_price
        total += rt_buy_price * sol["P_rt_buy"][t]
        total -= config.rt_sell_price_ratio * float(prices[t]) * sol["P_rt_sell"][t]
        total += config.rt_imbalance_fee * (sol["P_rt_buy"][t] + sol["P_rt_sell"][t])
    for g, p in config.dg_params.items():
        for val in sol["P_DG"][g]:
            total += p.a * val**2 + p.b * val
    for e in config.ess_params:
        for t in range(HORIZON):
            total += config.storage_degradation_cost * (sol["P_ch"][e][t] + sol["P_dis"][e][t])
    if "P_PV_curt" in sol:
        for b, vals in sol["P_PV_curt"].items():
            for val in vals:
                total += config.pv_curtailment_cost * val
    return float(total)


def extract_solution(
    model: pyo.ConcreteModel,
    network: NetworkData,
    prices: np.ndarray,
    config: DispatchConfig | None = None,
    status: str = "unknown",
    termination: str = "unknown",
) -> DispatchSolution:
    config = config or DispatchConfig()
    times = range(HORIZON)
    line_ids = network.active_branch["line_id"].astype(int).tolist()
    P_DG = {g: [_value(model.P_DG[g, t]) for t in times] for g in config.dg_params}
    Q_DG = {g: [_value(model.Q_DG[g, t]) for t in times] for g in config.dg_params}
    P_ch = {e: [_value(model.P_ch[e, t]) for t in times] for e in config.ess_params}
    P_dis = {e: [_value(model.P_dis[e, t]) for t in times] for e in config.ess_params}
    Q_ESS = {e: [_value(model.Q_ESS[e, t]) for t in times] for e in config.ess_params}
    SOC = {e: [_value(model.SOC[e, t + 1]) for t in times] for e in config.ess_params}
    Q_PV = {int(b): [_value(model.Q_PV[b, t]) for t in times] for b in model.PV}
    P_PV_curt = {int(b): [_value(model.P_PV_curt[b, t]) for t in times] for b in model.PV}
    P_grid_buy = [_value(model.P_grid_buy[t]) for t in times]
    P_grid_sell = [_value(model.P_grid_sell[t]) for t in times]
    P_rt_buy = [_value(model.P_rt_buy[t]) for t in times]
    P_rt_sell = [_value(model.P_rt_sell[t]) for t in times]
    P_grid = [P_grid_buy[t] - P_grid_sell[t] + P_rt_buy[t] - P_rt_sell[t] for t in times]
    Q_grid = [_value(model.Q_grid[t]) for t in times]
    V = {b: [_value(model.V[b, t]) for t in times] for b in range(1, NUM_BUSES + 1)}
    P_line = {int(l): [_value(model.P[l, t]) for t in times] for l in line_ids}
    Q_line = {int(l): [_value(model.Q[l, t]) for t in times] for l in line_ids}
    I_sq_line = {int(l): [_value(model.I_sq[l, t]) for t in times] for l in line_ids}
    I_line = {l: [float(math.sqrt(max(I_sq_line[l][t], 0.0))) for t in times] for l in line_ids}
    S_line = {l: [float(math.sqrt(max(P_line[l][t] ** 2 + Q_line[l][t] ** 2, 0.0))) for t in times] for l in line_ids}
    current_limit = {int(row.line_id): float(math.sqrt(max(float(row.limit_I_sq), 0.0))) for row in network.active_branch.itertuples(index=False)}
    max_line_overload = max(max(I_line[l][t] - current_limit[l], 0.0) for l in line_ids for t in times)
    soc_relax_gap: dict[int, list[float]] = {}
    max_soc_relax_gap = 0.0
    branch_by_line = {int(row.line_id): row for row in network.active_branch.itertuples(index=False)}
    for l in line_ids:
        fbus = int(branch_by_line[l].fbus)
        soc_relax_gap[l] = []
        for t in times:
            lhs = (P_line[l][t] / SBASE_MVA) ** 2 + (Q_line[l][t] / SBASE_MVA) ** 2
            rhs = V[fbus][t] * I_sq_line[l][t]
            gap = max(float(rhs - lhs), 0.0) if math.isfinite(rhs) and math.isfinite(lhs) else float("nan")
            soc_relax_gap[l].append(gap)
            if math.isfinite(gap):
                max_soc_relax_gap = max(max_soc_relax_gap, gap)
    max_voltage_violation = max(
        max(VMIN - V[b][t], 0.0) + max(V[b][t] - VMAX, 0.0)
        for b in range(1, NUM_BUSES + 1) for t in times
    )
    simple = {
        "P_DG": P_DG,
        "Q_DG": Q_DG,
        "P_ch": P_ch,
        "P_dis": P_dis,
        "Q_ESS": Q_ESS,
        "Q_PV": Q_PV,
        "P_PV_curt": P_PV_curt,
        "P_grid": P_grid,
        "P_grid_buy": P_grid_buy,
        "P_grid_sell": P_grid_sell,
        "P_rt_buy": P_rt_buy,
        "P_rt_sell": P_rt_sell,
    }
    # Use the model objective as the realized economic cost so that PV
    # curtailment and replay redispatch effort are included consistently.
    economic_cost = _value(model.objective)
    return DispatchSolution(
        status=status,
        termination=termination,
        objective=_value(model.objective),
        economic_cost=economic_cost,
        P_DG=P_DG,
        Q_DG=Q_DG,
        P_ch=P_ch,
        P_dis=P_dis,
        Q_ESS=Q_ESS,
        SOC=SOC,
        Q_PV=Q_PV,
        P_PV_curt=P_PV_curt,
        P_grid_buy=P_grid_buy,
        P_grid_sell=P_grid_sell,
        P_rt_buy=P_rt_buy,
        P_rt_sell=P_rt_sell,
        P_grid=P_grid,
        Q_grid=Q_grid,
        V=V,
        P_line=P_line,
        Q_line=Q_line,
        S_line=S_line,
        I_sq_line=I_sq_line,
        I_line=I_line,
        current_limit=current_limit,
        soc_relax_gap=soc_relax_gap,
        max_soc_relax_gap=float(max_soc_relax_gap),
        max_voltage_violation=float(max_voltage_violation),
        max_line_overload=float(max_line_overload),
    )


def fixed_from_solution(sol: DispatchSolution) -> dict[str, Any]:
    return {
        "P_DG": sol.P_DG,
        "Q_DG": sol.Q_DG,
        "P_ch": sol.P_ch,
        "P_dis": sol.P_dis,
        "Q_ESS": sol.Q_ESS,
        "Q_PV": sol.Q_PV,
        "P_PV_curt": sol.P_PV_curt,
        "P_grid_buy": sol.P_grid_buy,
        "P_grid_sell": sol.P_grid_sell,
    }


def solve_dispatch(
    pnet: pd.DataFrame,
    prices: np.ndarray,
    network: NetworkData,
    solver,
    config: DispatchConfig | None = None,
    *,
    fixed: dict[str, Any] | None = None,
    aligned: pd.DataFrame | None = None,
    load: pd.DataFrame | None = None,
    pv: pd.DataFrame | None = None,
    import_duals: bool = False,
) -> tuple[pyo.ConcreteModel, DispatchSolution]:
    config = config or DispatchConfig()
    model = build_dispatch_model(
        pnet,
        prices,
        network,
        config,
        fixed=fixed,
        aligned=aligned,
        load=load,
        pv=pv,
        import_duals=import_duals,
    )
    status, termination = solve_model(model, solver)
    sol = extract_solution(model, network, prices, config, status=status, termination=termination)
    return model, sol


def get_dual(model: pyo.ConcreteModel, con) -> float:
    try:
        return float(model.dual.get(con, 0.0) or 0.0)
    except Exception:
        return 0.0



def compute_marginal_cost_map(
    model: pyo.ConcreteModel,
    normalize_cap: float = 5.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Build the economic marginal-cost map M from active-power-balance duals.

    M_raw[i,t] = |lambda_P[i,t]|, where lambda_P is the dual variable of the
    nodal active-power balance constraint.  This quantity is used as a local
    marginal economic attribution: it measures how sensitive the optimized
    dispatch objective is to one additional MW of net load at bus i and time t.

    If all duals are zero or unavailable, the function falls back to a uniform
    map so that the corrector training does not collapse to a zero loss.
    """
    if not hasattr(model, "dual"):
        raise ValueError("Model does not have a dual suffix. Build/solve with import_duals=True.")

    M_raw = np.zeros((NUM_BUSES, HORIZON), dtype=float)
    for bus in range(1, NUM_BUSES + 1):
        for t in range(HORIZON):
            M_raw[bus - 1, t] = abs(get_dual(model, model.p_balance[bus, t]))

    mean_m = float(M_raw.mean())
    if mean_m > EPS:
        M_norm = np.minimum(M_raw / (mean_m + EPS), normalize_cap)
        fallback_uniform = False
    else:
        # Robust fallback: if Gurobi/Pyomo fails to import LP/QCP duals, use a
        # uniform weight.  This recovers ordinary MAE correction while recording
        # that marginal-cost duals were unavailable.
        M_norm = np.ones_like(M_raw)
        fallback_uniform = True

    components = {
        "M_raw": M_raw,
        "M_norm": M_norm,
    }
    meta = {
        "mean_M_raw": mean_m,
        "max_M_raw": float(M_raw.max()),
        "marginal_dual_available": not fallback_uniform,
        "fallback_uniform_M": bool(fallback_uniform),
    }
    return M_norm.astype(np.float32), components, meta



def normalize_map(raw: np.ndarray, normalize_cap: float = 5.0, fallback_value: float = 1.0) -> tuple[np.ndarray, bool, float]:
    """Normalize a nonnegative node-time map by its mean with robust fallback."""
    raw = np.asarray(raw, dtype=float)
    mean_raw = float(raw.mean())
    if mean_raw > EPS and np.isfinite(mean_raw):
        return np.minimum(raw / (mean_raw + EPS), normalize_cap).astype(np.float32), False, mean_raw
    return (np.ones_like(raw, dtype=np.float32) * float(fallback_value)), True, mean_raw


def normalize_signed_map(raw: np.ndarray, normalize_cap: float = 5.0) -> tuple[np.ndarray, bool, float]:
    """Normalize a signed economic marginal map by its mean absolute value."""
    raw = np.asarray(raw, dtype=float)
    mean_abs = float(np.mean(np.abs(raw)))
    if mean_abs > EPS and np.isfinite(mean_abs):
        return np.clip(raw / (mean_abs + EPS), -normalize_cap, normalize_cap).astype(np.float32), False, mean_abs
    return np.zeros_like(raw, dtype=np.float32), True, mean_abs


def compute_bidirectional_marginal_maps_from_duals(
    lambda_plus: np.ndarray,
    lambda_minus: np.ndarray,
    normalize_cap: float = 5.0,
    boundary_kappa: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct two-sided marginal-cost maps and a boundary/switching map.

    Parameters
    ----------
    lambda_plus, lambda_minus:
        Raw active-power balance shadow prices under upward/downward net-load
        perturbations.  They should have shape [NUM_BUSES, HORIZON].

    Returns
    -------
    maps:
        M_plus, M_minus and B_boundary, all as node-time maps.  M_plus weighs
        over-prediction errors, M_minus weighs under-prediction errors, and
        B_boundary marks local dispatch-state switching sensitivity.
    meta:
        Diagnostics for dual availability and normalization.
    """
    lp = np.asarray(lambda_plus, dtype=float)
    lm = np.asarray(lambda_minus, dtype=float)
    raw_plus = np.abs(lp)
    raw_minus = np.abs(lm)
    M_plus_norm, fb_plus, mean_plus = normalize_map(raw_plus, normalize_cap=normalize_cap)
    M_minus_norm, fb_minus, mean_minus = normalize_map(raw_minus, normalize_cap=normalize_cap)
    B = np.abs(lp - lm) / (np.abs(lp) + np.abs(lm) + EPS)
    B = np.clip(B, 0.0, 1.0).astype(np.float32)
    maps = {
        "M_plus_raw": raw_plus.astype(np.float32),
        "M_minus_raw": raw_minus.astype(np.float32),
        "M_plus_norm": M_plus_norm.astype(np.float32),
        "M_minus_norm": M_minus_norm.astype(np.float32),
        "B_boundary": B,
    }
    meta = {
        "mean_M_plus_raw": mean_plus,
        "mean_M_minus_raw": mean_minus,
        "max_M_plus_raw": float(np.max(np.abs(raw_plus))) if raw_plus.size else 0.0,
        "max_M_minus_raw": float(np.max(np.abs(raw_minus))) if raw_minus.size else 0.0,
        "fallback_uniform_M_plus": bool(fb_plus),
        "fallback_uniform_M_minus": bool(fb_minus),
        "marginal_dual_available": not (fb_plus and fb_minus),
        "boundary_kappa": float(boundary_kappa),
        "feedback_type": "two_sided_marginal_cost_with_separate_boundary_map",
    }
    return maps, meta


def compute_attribution_maps(
    model: pyo.ConcreteModel,
    sol: DispatchSolution,
    network: NetworkData,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    bus_ids = range(1, NUM_BUSES + 1)
    line_ids = network.active_branch["line_id"].astype(int).tolist()
    A_p = np.zeros((NUM_BUSES, HORIZON), dtype=float)
    A_q = np.zeros_like(A_p)
    A_v = np.zeros_like(A_p)
    A_line = np.zeros_like(A_p)
    qcp_line_dual_available = False

    for bus in bus_ids:
        qratio = network.q_ratio.get(bus, 0.0)
        for t in range(HORIZON):
            A_p[bus - 1, t] = abs(get_dual(model, model.p_balance[bus, t]))
            A_q[bus - 1, t] = abs(get_dual(model, model.q_balance[bus, t])) * abs(qratio)

    for i in bus_ids:
        for t in range(HORIZON):
            total = 0.0
            for j in bus_ids:
                dual_v = abs(get_dual(model, model.voltage_lower[j, t])) + abs(get_dual(model, model.voltage_upper[j, t]))
                total += dual_v * abs(network.voltage_sens_qaware[j - 1, i - 1])
            A_v[i - 1, t] = total

    for row in network.active_branch.itertuples(index=False):
        lid = int(row.line_id)
        for t in range(HORIZON):
            dual_l = get_dual(model, model.current_limit[lid, t])
            if abs(dual_l) > EPS:
                qcp_line_dual_available = True
            # Approximate downstream attribution of active/reactive current-limit pressure.
            for bus in network.downstream_by_line[lid]:
                qratio = network.q_ratio.get(bus, 0.0)
                A_line[bus - 1, t] += abs(dual_l) * (1.0 + abs(qratio))

    A_raw = A_p + A_q + A_v + A_line
    mean_a = float(A_raw.mean())
    A_norm = np.minimum(A_raw / (mean_a + EPS), 5.0) if mean_a > EPS else np.zeros_like(A_raw)
    components = {
        "A_P_balance": A_p,
        "A_Q_balance": A_q,
        "A_voltage": A_v,
        "A_line": A_line,
        "A_raw": A_raw,
        "A_norm": A_norm,
    }
    meta = {"qcp_line_dual_available": bool(qcp_line_dual_available), "mean_A_raw": mean_a, "max_A_raw": float(A_raw.max())}
    return A_norm, components, meta


def compute_risk_maps(
    real_sol: DispatchSolution,
    network: NetworkData,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    r_line = np.zeros((NUM_BUSES, HORIZON), dtype=float)
    current_limit_by_line = {int(row.line_id): float(math.sqrt(max(float(row.limit_I_sq), 0.0))) for row in network.active_branch.itertuples(index=False)}
    for lid in current_limit_by_line:
        I = np.asarray(real_sol.I_line[lid], dtype=float)
        overload = np.maximum(I - current_limit_by_line[lid], 0.0)
        for bus in network.downstream_by_line[lid]:
            r_line[bus - 1, :] += overload

    v_real = np.zeros((NUM_BUSES, HORIZON), dtype=float)
    for bus in range(1, NUM_BUSES + 1):
        v_real[bus - 1, :] = np.asarray(real_sol.V[bus], dtype=float)
    v_violation = np.maximum(VMIN - v_real, 0.0) + np.maximum(v_real - VMAX, 0.0)
    r_voltage = np.zeros_like(v_real)
    for t in range(HORIZON):
        r_voltage[:, t] = np.abs(network.voltage_sens_qaware).T @ v_violation[:, t]
    R_raw = r_line + r_voltage
    mean_r = float(R_raw.mean())
    R_norm = np.minimum(R_raw / (mean_r + EPS), 5.0) if mean_r > EPS else np.zeros_like(R_raw)
    components = {"R_line": r_line, "R_voltage": r_voltage, "R_raw": R_raw, "R_norm": R_norm}
    return R_norm, components


def dispatch_and_replay(
    pnet_dispatch: pd.DataFrame,
    pnet_real: pd.DataFrame,
    prices: np.ndarray,
    network: NetworkData,
    solver,
    config: DispatchConfig | None = None,
    *,
    aligned: pd.DataFrame | None = None,
    import_duals: bool = False,
) -> dict[str, Any]:
    config = config or DispatchConfig()
    try:
        pred_model, pred_sol = solve_dispatch(
            pnet_dispatch,
            prices,
            network,
            solver,
            config,
            aligned=aligned,
            import_duals=import_duals,
        )
        if not is_optimal(pred_sol.status, pred_sol.termination):
            return {"ok": False, "status": f"pred_{pred_sol.status}/{pred_sol.termination}"}
        fixed = fixed_from_solution(pred_sol)
        real_model, real_sol = solve_dispatch(
            pnet_real,
            prices,
            network,
            solver,
            config,
            fixed=fixed,
            aligned=aligned,
            import_duals=False,
        )
        if not is_optimal(real_sol.status, real_sol.termination):
            return {
                "ok": False,
                "status": f"real_{real_sol.status}/{real_sol.termination}",
                "pred_model": pred_model,
                "pred_sol": pred_sol,
            }
        return {
            "ok": True,
            "status": f"{pred_sol.status}/{pred_sol.termination};{real_sol.status}/{real_sol.termination}",
            "pred_model": pred_model,
            "pred_sol": pred_sol,
            "real_model": real_model,
            "real_sol": real_sol,
            "realized_cost": real_sol.economic_cost,
            "voltage_violation": real_sol.max_voltage_violation,
            "line_overload": real_sol.max_line_overload,
            "max_soc_relax_gap": real_sol.max_soc_relax_gap,
        }
    except Exception as exc:
        return {"ok": False, "status": f"failed: {exc}"}


def metadata(config: DispatchConfig, qcp_line_dual_available: bool | None = None) -> dict[str, Any]:
    out = {
        "reactive_dispatch_enabled": True,
        "storage_degradation_enabled": True,
        "power_flow_model": "DistFlow with SOCP relaxation",
        "voltage_drop_constraint": "v_j = v_i - 2(rP+xQ) + (r^2+x^2)l",
        "branch_flow_soc_constraint": "P^2 + Q^2 <= v_i*l_ij",
        "line_capacity_constraint": "0 <= l_ij <= Imax_ij^2",
        "dg_capacity_constraint": "P_DG^2 + Q_DG^2 <= S_DG^2",
        "pv_inverter_constraint": "P_PV^2 + Q_PV^2 <= S_PV^2",
        "ess_converter_constraint": "(P_dis - P_ch)^2 + Q_ESS^2 <= S_ESS^2",
        "storage_degradation_cost_AUD_per_MWh": config.storage_degradation_cost,
        "objective_terms": "day-ahead grid trading cost + real-time imbalance settlement + DG generation cost + storage degradation cost + PV curtailment cost + replay redispatch effort",
        "voltage_security_constraint": "hard constraint: VMIN <= V_i,t <= VMAX in both day-ahead dispatch and realized replay",
        "grid_exchange_constraint": "day-ahead P_grid_buy/P_grid_sell are fixed during realized replay; nonnegative P_rt_buy/P_rt_sell can balance active-power forecast error at the substation; Q_grid remains an unbounded signed recourse variable",
        "replay_recourse": "when enabled, DG, ESS, PV curtailment, and reactive-power decisions are re-optimized under realized net load with deviation penalties from the day-ahead plan",
        "day_ahead_grid_buy_price": "real day-ahead price",
        "day_ahead_grid_sell_price": f"{config.sell_price_ratio} * real day-ahead price",
        "real_time_imbalance_buy_price": f"real day-ahead price + {config.rt_buy_price_adder} AUD/MWh + ({config.rt_buy_price_multiplier} - 1) * max(real day-ahead price, 0)",
        "real_time_imbalance_sell_price": f"{config.rt_sell_price_ratio} * real day-ahead price",
        "real_time_imbalance_fee_AUD_per_MWh": config.rt_imbalance_fee,
        "negative_price_export_rule": "P_grid_sell = 0 when buy price < 0 to prevent artificial simultaneous buy/sell arbitrage",
        "line_limit_default_MVA": config.line_limit_default_mva,
        "economic_feedback": "two-sided boundary-aware marginal cost maps from p_balance shadow prices",
        "M_definition": "M_plus/M_minus from active-power balance duals under upward/downward net-load perturbations; B_boundary from dual variation",
        "price_used": "real_day_ahead_price",
        "price_prediction_used": False,
        "correction_target": "netload_only",
        "line_overload_definition": "sqrt(l_ij) - Imax_ij",
        "soc_relaxation_gap_definition": "v_i*l_ij - (P_ij^2+Q_ij^2)",
        "QCPDual": config.qcp_dual,
    }
    if qcp_line_dual_available is not None:
        out["qcp_line_dual_available"] = bool(qcp_line_dual_available)
    return out


__all__ = [
    "HORIZON", "NUM_BUSES", "BUS_COLS", "VMIN", "VMAX", "DispatchConfig", "NetworkData",
    "DispatchSolution", "load_network", "load_price_series", "load_aligned_dataset", "choose_solver",
    "build_dispatch_model", "solve_model", "solve_dispatch", "extract_solution", "fixed_from_solution", "get_dual",
    "compute_marginal_cost_map", "compute_attribution_maps", "compute_risk_maps", "dispatch_and_replay", "is_optimal", "iso", "metadata",
]

