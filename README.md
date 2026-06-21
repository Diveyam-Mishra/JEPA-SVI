## Overview

JEPA IV Surface provides an end-to-end pipeline for:

* Option chain ingestion and validation
* Implied volatility surface construction
* Classical forecasting baselines (PCA, VAR, DNS, SVI)
* Self-supervised JEPA training
* Latent-space forecasting
* Arbitrage diagnostics
* Hedging evaluation

The project is organized into a multi-phase workflow that takes raw option data through surface generation, representation learning, forecasting, and financial evaluation.

## Project Structure

```text
src/jepa_iv/
├── data.py              # Data loading and validation
├── black_scholes.py     # Pricing, Greeks, IV solver
├── surface.py           # Surface construction
├── baselines.py         # PCA, VAR, DNS baselines
├── svi.py               # SVI calibration
├── models.py            # JEPA architecture
├── training.py          # Training utilities
├── dynamics.py          # Latent forecasting
├── metrics.py           # Evaluation metrics
├── experiments.py       # Experiment workflows
├── hedging.py           # Hedging simulations
└── cli.py               # Command-line interface
```

## Quick Start

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
$env:PYTHONPATH="$PWD\src"

uv run --no-sync python -m unittest discover -s tests
uv run --no-sync python -m jepa_iv.cli smoke-test
```

## Workflow

```text
Option Data
    ↓
IV Surface Construction
    ↓
Baseline Models
    ↓
JEPA Pretraining
    ↓
Latent Dynamics
    ↓
Forecasting Evaluation
    ↓
Arbitrage Analysis
    ↓
Hedging Backtests
```

## Example Commands

Pull option-chain data:

```powershell
uv run --no-sync python -m jepa_iv.cli pull-yfinance \
    --symbol SPY \
    --output data/raw/options.parquet
```

Build IV surfaces:

```powershell
uv run --no-sync python -m jepa_iv.cli build-surfaces \
    --input data/raw/options.parquet \
    --output data/processed/surfaces.npz
```

Train JEPA:

```powershell
uv run --no-sync python -m jepa_iv.cli train-jepa \
    --surfaces data/processed/surfaces.npz \
    --output runs/jepa
```

Evaluate baselines:

```powershell
uv run --no-sync python -m jepa_iv.cli evaluate \
    --surfaces data/processed/surfaces.npz \
    --run-dir runs/baselines
```

## Research Goals

* Learn robust representations of implied volatility surfaces
* Forecast future surface dynamics from latent representations
* Compare against established quantitative baselines
* Study arbitrage consistency of generated forecasts
* Evaluate economic value through hedging performance
