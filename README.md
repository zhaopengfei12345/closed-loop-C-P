# Closed-Loop Predict-and-Optimize

This folder contains a implementation of the paper:

**Fine-Grained Marginal Cost Feedback-Based Closed-Loop Predict-and-Optimize Framework for Distribution System Scheduling**.


## What Is Included

- `src/fgmc_gfcn_ieee33/dispatch.py`
  - IEEE 33-bus active/reactive dispatch model.
  - DG, PV, ESS, voltage, branch-current and real-time replay constraints.
  - Shadow-price extraction from nodal active-power balance constraints.

- `src/fgmc_gfcn_ieee33/closed_loop.py`
  - Bidirectional marginal-cost map construction.
  - Boundary/switching map construction.
  - Graph-Flow Correction Network (GFCN).
  - Closed-loop residual training utilities and dispatch evaluation.

- `scripts/train_forecaster.py`
  - 48-hour-to-24-hour net-load forecaster.
  - Supports `lstm` and a lightweight `stgnn` baseline.

- `scripts/run_baseline_dispatch.py`
  - Predict-then-optimize baseline evaluation on December.

- `scripts/run_closed_loop_gfcn.py`
  - Closed-loop FGMC-GFCN correction and evaluation.

## Data Scope

The experiment uses the processed IEEE 33-bus hourly dataset under `data/processed_1h`.

Data provenance:

- Load profiles: Ausgrid Distribution Zone Substation Data.
- PV profiles: AEMO NEMWeb Dispatch SCADA.
- Solar-unit identification: AEMO CDEII available-generator metadata, filtered by `REGIONID=NSW1` and `CO2E_ENERGY_SOURCE=Solar`.
- Prices: AEMO NSW1 DispatchIS reports.

Important limitation: the CDEII generator metadata identifies solar DUIDs but does not provide plant-level latitude/longitude or meteorological variables. The current experiments do not use weather, irradiance, temperature, cloud-cover, or plant-coordinate features.

## Install

From the repository root:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r open_source/fgmc_gfcn_ieee33/requirements.txt
```

Gurobi is required for the dispatch runs. Make sure `gurobipy` can find a valid local license.

## Run

Run from the repository root so the default `data/processed_1h` paths resolve.

Train the statistical forecaster:

```bash
python open_source/fgmc_gfcn_ieee33/scripts/train_forecaster.py \
  --predictor stgnn \
  --data-path data/processed_1h/aligned_dataset.csv \
  --output-dir results/fgmc_gfcn_forecast
```

Evaluate the PTO baseline:

```bash
python open_source/fgmc_gfcn_ieee33/scripts/run_baseline_dispatch.py \
  --prediction-path results/fgmc_gfcn_forecast/test_predictions.csv \
  --output-dir results/fgmc_gfcn_baseline
```

Run the closed-loop GFCN correction:

```bash
python open_source/fgmc_gfcn_ieee33/scripts/run_closed_loop_gfcn.py \
  --prediction-path results/fgmc_gfcn_forecast/test_predictions.csv \
  --output-dir results/fgmc_gfcn_iterative_dec \
  --K-max 5 \
  --m-map-mode dual
```

`--m-map-mode dual` follows the paper route by constructing two-sided shadow-price maps. `--m-map-mode rt_imbalance` is retained as a faster settlement-spread proxy for debugging and ablation.

## Main Outputs

- `test_predictions.csv`: long-format baseline forecasts.
- `dispatch_comparison_dec.csv`: realized replay cost and constraint metrics.
- `features/*_samples.csv`: node-time marginal-cost and boundary maps.
- `corrector/model_best.pt`: trained GFCN checkpoint for each round.
- `evaluation/dec_corrected_predictions.csv`: corrected net-load forecasts.
- `final_december_comparison.csv`: baseline and closed-loop comparison.
- `best_round_summary.json`: selected correction round and metrics.

## Differences From the Manuscript Text

The code follows the manuscript route, with these explicit implementation choices:

- The base forecaster supports both LSTM and a lightweight STGNN. No weather features are used because the processed dataset does not contain meteorological variables.
- The default open-source setting uses bidirectional dual/shadow-price feedback. A faster real-time settlement proxy remains available through `--m-map-mode rt_imbalance`.
- The default correction loss has two terms: a regret-weighted marginal-cost error and bounded residual regularization. Dispatch regret is computed by dispatch-and-replay in the outer loop, then used only as a capped sample weight for the differentiable marginal-cost loss. The default weights use the selected search setting `bmc_loss_weight=0.5`, `lambda_br=0.01`, `regret_alpha=1.5`, and `regret_weight_cap=5.0`.
- The PV-to-bus mapping is synthetic for the IEEE test feeder and should not be interpreted as geographic matching.

## Citation Metadata To Report

When describing the PV source, use language such as:

> AEMO CDEII available-generator metadata are used to identify solar DUIDs in the NSW1 region, and AEMO Dispatch SCADA provides the corresponding generation time series.

Do not state that the AEMO metadata provide PV plant coordinates or meteorological data.
