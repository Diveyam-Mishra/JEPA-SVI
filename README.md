# JEPA IV Surface

This repository is organized around the 10 phases in `JEPA_IV_Surface_Checklist.xlsx`.
The goal of this README is not to describe the theory. It is to tell you what to run,
in what order, what each step produces, and when to move to the next phase.

You said the near-term plan is:

1. use `yfinance` only
2. see whether the results are good enough
3. only then move to CBOE data

That is a sensible sequencing for development. `yfinance` is enough to validate the
pipeline, the numerical code, the baselines, the JEPA training loop, and the experiment
logic. It is not enough to claim institutional-grade data quality or a full historical
surface study without extra work.

## Environment

Every command below assumes PowerShell in this project root:

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
$env:PYTHONPATH="$PWD\src"
```

Quick checks:

```powershell
uv run --no-sync python -m unittest discover -s tests
uv run --no-sync python -m jepa_iv.cli smoke-test
```

## What Exists In Code

The repository already contains the core building blocks:

- data adapters and validation: [data.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/data.py>)
- Black-Scholes pricing, Greeks, IV solver: [black_scholes.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/black_scholes.py>)
- filtering, interpolation, scaling, arbitrage checks: [surface.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/surface.py>)
- PCA, VAR, DNS-style baselines: [baselines.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/baselines.py>)
- SVI fitting: [svi.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/svi.py>)
- JEPA model and training loop: [models.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/models.py>) and [training.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/training.py>)
- latent dynamics: [dynamics.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/dynamics.py>)
- forecasting metrics and DM test: [metrics.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/metrics.py>) and [experiments.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/experiments.py>)
- hedging backtest logic: [hedging.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/hedging.py>)
- CLI entry points: [cli.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/cli.py>)

## Phase-By-Phase Run Order

### Phase 1: Data Acquisition And Surface Construction

This phase gets you from raw option chains to a clean tensor of daily IV surfaces.

What this phase should produce:

- raw option chain file in `data/raw/`
- filtered option dataset with IVs
- interpolated daily surface tensor in `data/processed/surfaces.npz`

With `yfinance`, the intended workflow is:

1. pull option chains for your chosen underlying
2. store them as Parquet
3. build IV surfaces from the Parquet file

Step 1 and step 2 are now a real CLI command:

```powershell
uv run --no-sync python -m jepa_iv.cli pull-yfinance --symbol SPY --output data/raw/options.parquet
```

What this command does:

- pulls the current Yahoo option chain snapshot for the symbol
- collects all available expiries
- normalizes the columns into the repository schema
- writes `data/raw/options.parquet`

Important limitation:

- Yahoo does not give you a true historical option-chain archive through this path
- this command gives you a current snapshot, which is enough to validate the pipeline
- for a real multi-day or multi-month study with Yahoo, you would need to run and store snapshots repeatedly over time

After that, step 3 is:

```powershell
uv run --no-sync python -m jepa_iv.cli build-surfaces --input data/raw/options.parquet --output data/processed/surfaces.npz
```

What this command does:

- validates the raw option table
- computes mid prices
- computes implied vols
- filters bad rows
- interpolates daily surfaces onto the fixed `(20 x 12)` grid
- saves `surfaces` and `dates` into `data/processed/surfaces.npz`

If you have purchased historical parquet files in the vendor EOD schema, you can point
the same command at one file, a directory, or a glob pattern. Example:

```powershell
uv run --no-sync python -m jepa_iv.cli build-surfaces --input "spy_eod_*.parquet" --output data/processed/surfaces_full_history.npz
```

That path now works directly. The loader will:

- read all matching Parquet files
- convert the vendor call/put schema into one normalized row per option side
- use vendor-supplied `C_IV` and `P_IV` when available
- fall back to solving IV only when supplied IV is missing or unusable

What you should inspect before moving on:

- enough daily surfaces were built
- no obvious NaN/Inf issues
- retention after filtering is acceptable
- surfaces look plausible across strikes and maturities

For `yfinance`, use this phase to validate engineering quality, not to make strong claims
about production data completeness.

### Phase 2: Baselines

This phase gives you the benchmarks JEPA must beat.

What this phase should produce:

- PCA explained variance results
- PCA+VAR forecasts
- DNS forecasts
- random walk forecasts
- historical mean forecasts
- SVI parameter sets and optional SVI-based surface comparisons

The current CLI gives you a minimal baseline evaluation file:

```powershell
uv run --no-sync python -m jepa_iv.cli evaluate --surfaces data/processed/surfaces.npz --run-dir runs/baselines
```

What this currently writes:

- `runs/baselines/baseline_scores.csv`

What you should additionally run from the library for a full Phase 2 study:

- `PCAVARBaseline` from [baselines.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/baselines.py>)
- `NelsonSiegelTermStructure` from [baselines.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/baselines.py>)
- `fit_svi_surface` from [svi.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/svi.py>)

Interpretation:

- this phase tells you whether the dataset is too easy, too noisy, or rich enough for JEPA
- if PCA already explains nearly everything and forecasts extremely well, JEPA has a higher bar
- if SVI or DNS performs poorly, that is itself a useful result

### Phase 3: JEPA Architecture

This phase is the model construction phase.

What this phase should produce:

- working JEPA encoder pair
- EMA target updates
- masking logic
- predictor network
- end-to-end forward pass

This is already implemented. The fastest sanity check is:

```powershell
uv run --no-sync python -m jepa_iv.cli smoke-test
```

What that confirms:

- Black-Scholes and IV inversion are consistent
- patch masking works
- JEPA forward pass runs
- prediction and target tensor shapes match

### Phase 4: Self-Supervised Pretraining

This phase trains JEPA on the daily surfaces.

What this phase should produce:

- JEPA checkpoints
- final trained model
- scaler file
- training loss history
- latent-collapse diagnostics

Run:

```powershell
uv run --no-sync python -m jepa_iv.cli train-jepa --surfaces data/processed/surfaces.npz --output runs/jepa --epochs 200 --batch-size 64 --embed-dim 128 --mask-ratio 0.60
```

What this writes:

- `runs/jepa/model.pt`
- periodic checkpoints like `runs/jepa/checkpoint_epoch_0020.pt`
- `runs/jepa/scaler.npz`

What to check before Phase 5:

- loss decreases
- no NaN during training
- representation health warnings are not showing repeated collapse

### Phase 5: Temporal Dynamics On Latent Space

This phase turns JEPA from a representation learner into a forecaster.

What this phase should produce:

- latent vectors for all surfaces
- VAR fit on JEPA latents
- decoder from latent space back to surface space
- next-day and multi-horizon surface forecasts

Relevant modules:

- latent extraction: [training.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/training.py>)
- VAR dynamics: [dynamics.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/dynamics.py>)
- decoder class: [models.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/models.py>)

Conceptually, this is the order:

1. train JEPA
2. extract latent vectors for train, validation, and test
3. fit `LatentVARDynamics` on training latents
4. train `SurfaceDecoder` from latent vectors to normalized surfaces
5. forecast next latent state
6. decode back to surface space
7. inverse-normalize

If the latent VAR is weak, that does not necessarily kill the project. It may mean the
latent space is nonlinear and you should replace VAR with GRU/LSTM later.

### Phase 6: Forecasting Experiments

This phase is the main predictive comparison.

What this phase should produce:

- MSE table across all methods
- QLIKE table across all methods
- Diebold-Mariano test results
- region-level results
- regime-level results

Relevant functions:

- `mse`, `qlike`, `diebold_mariano` in [metrics.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/metrics.py>)
- region scoring in [experiments.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/experiments.py>)

This is the decision phase for `yfinance`:

- if JEPA is clearly unstable, the issue is probably model/pipeline quality, not data source
- if JEPA is stable and competitive, then it is worth upgrading the data source to CBOE
- if results are inconclusive because the data is too sparse, that is the strongest signal to move to CBOE

### Phase 7: Interpretability

This phase explains what JEPA learned.

What this phase should produce:

- PCA on JEPA latent vectors
- linear probe results against classical factors
- reverse-probe results from PCA to JEPA
- latent traversal plots
- UMAP or t-SNE regime plots
- Koopman-style clustering evidence

Relevant modules:

- latent extraction in [training.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/training.py>)
- metrics scaffolding in [experiments.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/experiments.py>)

This phase is where you answer:

- did JEPA just rediscover PCA
- did JEPA find nonlinear structure PCA misses
- do latent clusters align with volatility regimes

### Phase 8: No-Arbitrage Emergence

This phase checks whether JEPA predictions are structurally better behaved than simpler methods.

What this phase should produce:

- butterfly violation rates
- calendar violation rates
- comparison table across JEPA, PCA+VAR, DNS, and SVI

Relevant functions:

- `butterfly_violation_rate` in [surface.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/surface.py>)
- `calendar_violation_rate` in [surface.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/surface.py>)

Interpretation:

- if JEPA beats PCA+VAR on arbitrage consistency, that is a strong structural result
- if SVI still dominates, that is expected because it is explicitly parametric

### Phase 9: Hedging

This phase tests whether JEPA forecasts are economically useful.

What this phase should produce:

- delta-hedging P&L series
- P&L variance comparison
- turnover and transaction-cost impact
- latent-factor risk decomposition

Relevant module:

- [hedging.py](</C:/Users/fixadmin/JEPA -STYLE SVI/src/jepa_iv/hedging.py>)

This is the practical value test:

- if JEPA improves forecast error but does not improve hedging, then the gain may not be tradeable
- if JEPA survives transaction costs, the result is much stronger

### Phase 10: Write-Up And Submission

This phase is not about code execution. It is about packaging the results.

What this phase should produce:

- ablation table
- final results tables
- interpretability figures
- arbitrage comparison figures
- hedging result tables
- paper draft and submission package

You should only move to this phase once Phases 1 through 9 produce stable, repeatable outputs.

## Recommended Actual Order For You Right Now

Given your current plan, this is the run order I would use:

1. run tests and smoke test
2. run `pull-yfinance` to create `data/raw/options.parquet`
3. run `build-surfaces`
4. inspect the resulting number and quality of daily surfaces
5. run `evaluate` to get quick random-walk and historical-mean benchmarks
6. run `train-jepa`
7. extract latents and fit latent VAR
8. compare JEPA forecasts against PCA+VAR, DNS, SVI, random walk, and historical mean
9. run arbitrage checks
10. run hedging backtests
11. if the pipeline is stable and the results look promising, replace the raw-data source with CBOE and rerun the full sequence

## When To Stay On YFinance And When To Move

Stay on `yfinance` while:

- debugging ingestion
- validating IV inversion
- validating surface interpolation
- checking that training converges
- testing experiment code paths

Move to CBOE when:

- you need richer history
- you need denser chains across expiries and strikes
- filtering removes too much Yahoo data
- your JEPA vs baseline conclusions depend too heavily on sparse or noisy observations

## Important Limitation

The repository already has the core model, surface, SVI, metrics, and hedging code, but the
CLI is still intentionally thin. It currently exposes:

- `smoke-test`
- `pull-yfinance`
- `build-surfaces`
- `train-jepa`
- `evaluate`

For the later phases, the modules are in place but the full end-to-end experiment runners are
not yet exposed as dedicated CLI commands. The correct next engineering step, if you want it,
is to add explicit commands such as:

- `pull-yfinance`
- `fit-baselines`
- `fit-svi`
- `extract-latents`
- `forecast-latents`
- `run-experiments`
- `run-hedging`

That would make the Phase 1 to Phase 10 workflow executable without writing any extra runner scripts.
#   J E P A - S V I  
 