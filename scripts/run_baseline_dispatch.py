#!/usr/bin/env python3
"""Run the predict-then-optimize baseline on the distribution-feeder test month."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fgmc_gfcn import closed_loop as feedback
from fgmc_gfcn import dispatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline forecast by day-ahead dispatch and realized replay."
    )
    parser.add_argument("--prediction-path", type=Path, default=feedback.PREDICTION_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "baseline_dispatch_q_dec")
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--rt-buy-adder", type=float, default=dispatch.DispatchConfig.rt_buy_price_adder)
    parser.add_argument("--rt-buy-multiplier", type=float, default=dispatch.DispatchConfig.rt_buy_price_multiplier)
    parser.add_argument("--rt-sell-ratio", type=float, default=dispatch.DispatchConfig.rt_sell_price_ratio)
    parser.add_argument("--rt-imbalance-fee", type=float, default=dispatch.DispatchConfig.rt_imbalance_fee)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = dispatch.DispatchConfig(
        time_limit=args.time_limit,
        rt_buy_price_adder=args.rt_buy_adder,
        rt_buy_price_multiplier=args.rt_buy_multiplier,
        rt_sell_price_ratio=args.rt_sell_ratio,
        rt_imbalance_fee=args.rt_imbalance_fee,
    )
    network = dispatch.load_network(config)
    aligned = dispatch.load_aligned_dataset(config)
    price_series = dispatch.load_price_series(config)
    solver_name, solver = dispatch.choose_solver(config)

    data = feedback.load_daily_blocks(args.prediction_path)
    dec = data["dec"]
    dispatch_df = feedback.evaluate_dispatch_for_predictions(
        "Baseline", dec, dec["y_pred"], network, price_series, solver, config, aligned
    )
    dispatch_df.to_csv(out_dir / "dispatch_comparison_dec.csv", index=False)

    metrics = feedback.baseline_metrics(dec, dispatch_df)
    metrics.update({"method": "Baseline", "K": 0, "solver": solver_name})
    (out_dir / "metrics_dec.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_dir / "final_december_comparison.csv", index=False)

    meta = dispatch.metadata(config)
    meta.update(
        {
            "script": "scripts/run_baseline_dispatch.py",
            "solver": solver_name,
            "forecast_target": "netload_only",
            "price_used": "real_day_ahead_price",
            "price_prediction_used": False,
            "prediction_path": str(args.prediction_path),
            "test_month": "2023-12",
        }
    )
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[OK] Baseline dispatch completed.")
    print(f"December samples: {len(dec['origins'])}")
    print(f"MAE={metrics['MAE']:.6f}")
    print(f"realized_cost={metrics['mean_realized_cost']:.4f}")
    print(f"results={out_dir}")


if __name__ == "__main__":
    main()

