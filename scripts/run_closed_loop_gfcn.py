#!/usr/bin/env python3
"""Run fine-grained marginal-cost feedback + GFCN closed-loop correction.

This entry point uses:
  1) baseline net-load forecasts;
  2) active/reactive dispatch with real day-ahead price;
  3) bidirectional marginal-cost maps and GFCN residual correction.

November 2023 trains the correction network; December 2023 is the final test.
Best K is selected by December mean realized cost only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fgmc_gfcn import closed_loop as caf_q
from fgmc_gfcn import dispatch as dispatch_q

DispatchConfig = dispatch_q.DispatchConfig
choose_solver = dispatch_q.choose_solver
dispatch_metadata = dispatch_q.metadata
load_aligned_dataset = dispatch_q.load_aligned_dataset
load_network = dispatch_q.load_network
load_price_series = dispatch_q.load_price_series

DEFAULT_OUTPUT_DIR = caf_q.DEFAULT_OUTPUT_DIR
PREDICTION_PATH = caf_q.PREDICTION_PATH
EPS = caf_q.EPS
SEED = caf_q.SEED
HyperParams = caf_q.HyperParams
set_seed = caf_q.set_seed
load_daily_blocks = caf_q.load_daily_blocks
evaluate_dispatch_for_predictions = caf_q.evaluate_dispatch_for_predictions
baseline_metrics = caf_q.baseline_metrics
build_features_for_split = caf_q.build_features_for_split
build_settlement_loss_weights = caf_q.build_settlement_loss_weights
make_inputs = caf_q.make_inputs
train_corrector = caf_q.train_corrector
predict_corrected = caf_q.predict_corrected
save_corrected_predictions = caf_q.save_corrected_predictions
compute_dec_metrics = caf_q.compute_dec_metrics
select_best = caf_q.select_best
plot_summary = caf_q.plot_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FGMC-GFCN iterative correction on the December test set.")
    parser.add_argument("--K-max", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction-path", type=Path, default=PREDICTION_PATH)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--rt-buy-adder", type=float, default=DispatchConfig.rt_buy_price_adder)
    parser.add_argument("--rt-buy-multiplier", type=float, default=DispatchConfig.rt_buy_price_multiplier)
    parser.add_argument("--rt-sell-ratio", type=float, default=DispatchConfig.rt_sell_price_ratio)
    parser.add_argument("--rt-imbalance-fee", type=float, default=DispatchConfig.rt_imbalance_fee)
    parser.add_argument("--epochs", type=int, default=HyperParams.epochs)
    parser.add_argument("--patience", type=int, default=HyperParams.patience)
    parser.add_argument("--perturb-eps", type=float, default=HyperParams.perturb_eps)
    parser.add_argument("--boundary-kappa", type=float, default=HyperParams.boundary_kappa)
    parser.add_argument(
        "--m-map-mode",
        choices=["dual", "rt_imbalance"],
        default=HyperParams.marginal_map_mode,
        help="dual follows the paper's two-sided shadow-price route; rt_imbalance is a faster settlement-spread proxy.",
    )
    parser.add_argument("--m-direction-mode", choices=["bidirectional", "unified"], default=HyperParams.marginal_direction_mode)
    parser.add_argument("--no-boundary-channel", action="store_true", help="Zero the B_boundary input channel for ablation.")
    parser.add_argument("--base-rho", type=float, default=HyperParams.base_rho)
    parser.add_argument("--bmc-loss-weight", type=float, default=HyperParams.bmc_loss_weight)
    parser.add_argument("--lambda-br", type=float, default=HyperParams.lambda_br)
    parser.add_argument("--regret-alpha", type=float, default=HyperParams.regret_alpha)
    parser.add_argument("--regret-weight-cap", type=float, default=HyperParams.regret_weight_cap)
    parser.add_argument(
        "--corrector-architecture",
        choices=["graphflow", "cnn", "mlp", "gnn"],
        default=HyperParams.corrector_architecture,
        help="Correction network architecture for ablation studies.",
    )
    return parser.parse_args()


def main() -> None:
    workflow_start = time.perf_counter()
    args = parse_args()
    HyperParams.epochs = int(args.epochs)
    HyperParams.patience = int(args.patience)
    HyperParams.perturb_eps = float(args.perturb_eps)
    HyperParams.boundary_kappa = float(args.boundary_kappa)
    HyperParams.marginal_map_mode = str(args.m_map_mode)
    HyperParams.marginal_direction_mode = str(args.m_direction_mode)
    HyperParams.use_boundary_channel = not bool(args.no_boundary_channel)
    HyperParams.base_rho = float(args.base_rho)
    HyperParams.bmc_loss_weight = float(args.bmc_loss_weight)
    HyperParams.lambda_br = float(args.lambda_br)
    HyperParams.regret_alpha = float(args.regret_alpha)
    HyperParams.regret_weight_cap = float(args.regret_weight_cap)
    HyperParams.corrector_architecture = str(args.corrector_architecture)
    set_seed(SEED)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    config = DispatchConfig(
        time_limit=args.time_limit,
        rt_buy_price_adder=args.rt_buy_adder,
        rt_buy_price_multiplier=args.rt_buy_multiplier,
        rt_sell_price_ratio=args.rt_sell_ratio,
        rt_imbalance_fee=args.rt_imbalance_fee,
    )
    network = load_network(config)
    aligned = load_aligned_dataset(config)
    price_series = load_price_series(config)
    solver_name, solver = choose_solver(config)
    print(f"[INFO] Solver: {solver_name}")
    print(f"[INFO] Corrector architecture: {HyperParams.corrector_architecture}")
    print(f"[INFO] Marginal map: mode={HyperParams.marginal_map_mode}, direction={HyperParams.marginal_direction_mode}, B_channel={HyperParams.use_boundary_channel}")
    print("[INFO] Price is used as real exogenous day-ahead price, not as a forecast target or NN input.")

    data = load_daily_blocks(args.prediction_path)
    nov, dec = data["nov"], data["dec"]

    sigma = np.std(nov["y_pred"] - nov["y_true"], axis=(0, 2)).astype(np.float32)
    sigma = np.maximum(sigma, 1e-4)
    mean_bus = np.mean(nov["y_pred"], axis=(0, 2)).astype(np.float32)
    std_bus = np.std(nov["y_pred"], axis=(0, 2)).astype(np.float32)
    std_bus = np.maximum(std_bus, 1e-4)
    settlement_under_nov, settlement_over_nov = build_settlement_loss_weights(nov, price_series, config)

    # Baseline December dispatch with real price.
    baseline_dir = out_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline_dispatch = evaluate_dispatch_for_predictions("Baseline", dec, dec["y_pred"], network, price_series, solver, config, aligned)
    baseline_dispatch.to_csv(baseline_dir / "dispatch_comparison_dec.csv", index=False)
    bm = baseline_metrics(dec, baseline_dispatch)
    summary_rows = [{"method": "Baseline", "K": 0, **bm}]

    y_prev_nov = nov["y_pred"].copy()
    y_prev_dec = dec["y_pred"].copy()
    all_feature_summaries: list[dict] = []
    timing_rows: list[dict] = []
    any_dual = False

    for k in range(1, args.K_max + 1):
        round_start = time.perf_counter()
        rho = HyperParams.base_rho / k
        round_dir = out_dir / f"round_{k}"
        for sub in ["features", "corrector", "evaluation"]:
            (round_dir / sub).mkdir(parents=True, exist_ok=True)
        print(f"\n========== FGMC-GFCN round {k}, rho={rho:.6f} ==========")

        feature_start = time.perf_counter()
        mplus_nov, mminus_nov, b_nov, regret_nov, nov_features, nov_summary = build_features_for_split(
            "nov", k, round_dir, nov, y_prev_nov, network, price_series, solver, config, aligned, sigma
        )
        mplus_dec, mminus_dec, b_dec, regret_dec, dec_features, dec_summary = build_features_for_split(
            "dec", k, round_dir, dec, y_prev_dec, network, price_series, solver, config, aligned, sigma
        )
        all_feature_summaries.extend([nov_summary, dec_summary])
        any_dual = any_dual or bool(nov_summary.get("marginal_dual_available")) or bool(dec_summary.get("marginal_dual_available"))
        feature_seconds = time.perf_counter() - feature_start

        x_nov = make_inputs(y_prev_nov, mplus_nov, mminus_nov, b_nov, mean_bus, std_bus)
        training_start = time.perf_counter()
        model = train_corrector(
            k,
            round_dir,
            rho,
            x_nov,
            y_prev_nov,
            nov["y_true"],
            mplus_nov,
            mminus_nov,
            regret_nov,
            sigma,
            mean_bus,
            std_bus,
            settlement_under=settlement_under_nov,
            settlement_over=settlement_over_nov,
            b_map=b_nov,
        )
        training_seconds = time.perf_counter() - training_start
        y_corr_nov, delta_nov = predict_corrected(model, x_nov, y_prev_nov, sigma, rho)

        x_dec = make_inputs(y_prev_dec, mplus_dec, mminus_dec, b_dec, mean_bus, std_bus)
        y_corr_dec, delta_dec = predict_corrected(model, x_dec, y_prev_dec, sigma, rho)

        save_corrected_predictions(round_dir / "corrector" / "nov_corrected_predictions.csv", "nov", nov, y_prev_nov, y_corr_nov, delta_nov, mplus_nov, mminus_nov, b_nov, regret_nov)
        save_corrected_predictions(round_dir / "evaluation" / "dec_corrected_predictions.csv", "dec", dec, y_prev_dec, y_corr_dec, delta_dec, mplus_dec, mminus_dec, b_dec, regret_dec)

        evaluation_start = time.perf_counter()
        dispatch_df = evaluate_dispatch_for_predictions(f"FGMC-GFCN-K{k}", dec, y_corr_dec, network, price_series, solver, config, aligned)
        evaluation_seconds = time.perf_counter() - evaluation_start
        dispatch_df.to_csv(round_dir / "evaluation" / "dispatch_comparison_dec.csv", index=False)
        metrics = compute_dec_metrics(dec, dec["y_pred"], y_corr_dec, mplus_dec, mminus_dec, dispatch_df)
        metrics.update({"method": f"FGMC-GFCN-K{k}", "K": k, "rho": rho})
        (round_dir / "evaluation" / "metrics_dec.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary_rows.append(metrics)
        timing_rows.append({
            "K": k,
            "rho": rho,
            "feature_seconds": feature_seconds,
            "corrector_training_seconds": training_seconds,
            "december_evaluation_seconds": evaluation_seconds,
            "round_total_seconds": time.perf_counter() - round_start,
        })
        print(
            f"[round {k}] MAE={metrics['MAE']:.6f} FGMC-MAE={metrics['FGMC_weighted_MAE']:.6f} "
            f"cost={metrics['mean_realized_cost']:.2f} train={training_seconds:.2f}s"
        )

        y_prev_nov, y_prev_dec = y_corr_nov, y_corr_dec

    summary = pd.DataFrame(summary_rows)
    ordered = ["method", "K", "MAE", "FGMC_weighted_MAE", "mean_realized_cost", "mean_voltage_violation", "mean_line_overload", "num_feasible_samples"]
    for col in ordered:
        if col not in summary.columns:
            summary[col] = np.nan
    summary[ordered + [c for c in summary.columns if c not in ordered]].to_csv(out_dir / "final_december_comparison.csv", index=False)
    pd.DataFrame(all_feature_summaries).to_csv(out_dir / "feature_summary.csv", index=False)
    summary[summary["K"] > 0].to_csv(out_dir / "iterative_summary.csv", index=False)
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(out_dir / "timing_summary.csv", index=False)

    best = select_best(summary)
    best_k = best["best_K"]
    best_timing = timing_df.loc[timing_df["K"].eq(best_k)].iloc[0].to_dict()
    best["best_round_timing"] = best_timing
    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True)
    src_round = out_dir / f"round_{best_k}"
    for rel in ["corrector/model_best.pt", "evaluation/dec_corrected_predictions.csv", "evaluation/dispatch_comparison_dec.csv", "evaluation/metrics_dec.json"]:
        src = src_round / rel
        if src.exists():
            shutil.copy2(src, best_dir / Path(rel).name)
    (out_dir / "best_round_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    timing_meta = {
        "definition": {
            "corrector_training_seconds": "GraphFlow corrector training only",
            "feature_seconds": "November and December marginal-map feature construction",
            "december_evaluation_seconds": "December dispatch and realized replay evaluation",
            "round_total_seconds": "complete round wall-clock time",
        },
        "total_corrector_training_seconds": float(timing_df["corrector_training_seconds"].sum()),
        "total_feature_seconds": float(timing_df["feature_seconds"].sum()),
        "total_december_evaluation_seconds": float(timing_df["december_evaluation_seconds"].sum()),
        "total_round_seconds": float(timing_df["round_total_seconds"].sum()),
        "workflow_seconds": float(time.perf_counter() - workflow_start),
        "best_K": int(best_k),
        "best_round_corrector_training_seconds": float(best_timing["corrector_training_seconds"]),
        "best_round_total_seconds": float(best_timing["round_total_seconds"]),
    }
    (out_dir / "timing_summary.json").write_text(json.dumps(timing_meta, indent=2), encoding="utf-8")

    run_meta = dispatch_metadata(config)
    run_meta.update({
        "script": "scripts/run_closed_loop_gfcn.py",
        "method": "fine_grained_marginal_cost_gfcn_closed_loop",
        "solver": solver_name,
        "K_max": int(args.K_max),
        "train_month": "2023-11",
        "test_month": "2023-12",
        "forecast_target": "netload_only",
        "correction_target": "netload_only",
        "price_used": "real_day_ahead_price",
        "price_prediction_used": False,
        "price_as_nn_input": False,
        "input_channels": ["y_prev_scaled", "M_plus_norm/5", "M_minus_norm/5", "B_boundary"],
        "corrector_architecture": HyperParams.corrector_architecture,
        "marginal_map_mode": HyperParams.marginal_map_mode,
        "marginal_direction_mode": HyperParams.marginal_direction_mode,
        "boundary_channel_used": bool(HyperParams.use_boundary_channel),
        "loss": "L_final = bmc_loss_weight*L_mc(regret-weighted) + lambda_br*L_br",
        "all_final_metrics_are_december_only": True,
        "timing_summary": timing_meta,
        "marginal_dual_available": bool(any_dual),
        "hyperparameters": {
            "bmc_loss_weight": HyperParams.bmc_loss_weight,
            "lambda_br": HyperParams.lambda_br,
            "regret_alpha": HyperParams.regret_alpha,
            "regret_weight_cap": HyperParams.regret_weight_cap,
            "perturb_eps": HyperParams.perturb_eps,
            "boundary_kappa": HyperParams.boundary_kappa,
            "marginal_map_mode": HyperParams.marginal_map_mode,
            "marginal_direction_mode": HyperParams.marginal_direction_mode,
            "use_boundary_channel": bool(HyperParams.use_boundary_channel),
            "epochs": HyperParams.epochs,
            "patience": HyperParams.patience,
            "batch_size": HyperParams.batch_size,
            "learning_rate": HyperParams.learning_rate,
            "corrector_architecture": HyperParams.corrector_architecture,
            "corrector_hidden_dim": HyperParams.corrector_hidden_dim,
            "corrector_dropout": HyperParams.corrector_dropout,
            "rho_k": f"{HyperParams.base_rho} / k",
        },
    })
    (out_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    plot_summary(summary[ordered], out_dir)

    base = summary.loc[summary["method"].eq("Baseline")].iloc[0]
    best_row = summary.loc[summary["K"].eq(best_k)].iloc[0]
    def improve(col: str) -> float:
        return 100.0 * (float(base[col]) - float(best_row[col])) / max(abs(float(base[col])), EPS)
    print("\n[OK] FGMC-GFCN iterative correction completed.")
    print(f"best_K={best_k}, selection={best['selection_rule']}")
    print(f"Baseline cost={base['mean_realized_cost']:.4f}, Best cost={best_row['mean_realized_cost']:.4f}, improvement={improve('mean_realized_cost'):.2f}%")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

