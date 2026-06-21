from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jepa_iv.baselines import HistoricalMeanBaseline, PCAVARBaseline
from jepa_iv.config import DataConfig, JEPAConfig, SurfaceGridConfig, TrainingConfig
from jepa_iv.data import (
    YFinanceOptionProvider,
    chronological_split,
    load_parquet_input,
    resolve_parquet_inputs,
    store_raw_options,
)
from jepa_iv.dynamics import LatentVARDynamics
from jepa_iv.experiments import compare_to_best_baseline, score_by_region, score_forecasts
from jepa_iv.metrics import mse, qlike
from jepa_iv.models import SurfaceDecoder, SurfaceJEPA
from jepa_iv.surface import SurfaceScaler, build_surface_tensor
from jepa_iv.training import extract_latents, train_jepa


def _cmd_smoke_test(_: argparse.Namespace) -> None:
    from jepa_iv.black_scholes import black_scholes_price, implied_volatility
    from jepa_iv.masking import block_mask_indices

    price = black_scholes_price("call", 100.0, 100.0, 0.5, 0.03, 0.2)
    iv = implied_volatility(price, "call", 100.0, 100.0, 0.5, 0.03)
    config = JEPAConfig(embed_dim=16, encoder_depth=1, encoder_heads=4, predictor_depth=1)
    model = SurfaceJEPA(config)
    context, target = block_mask_indices(2, (5, 4), config.mask_ratio)
    x = np.random.default_rng(0).normal(size=(2, 20, 12)).astype("float32")
    pred, tgt = model.forward(torch.as_tensor(x), context, target)
    print({"iv": round(iv, 10), "prediction_shape": tuple(pred.shape), "target_shape": tuple(tgt.shape)})


def _cmd_pull_yfinance(args: argparse.Namespace) -> None:
    provider = YFinanceOptionProvider()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    frame = provider.fetch(args.symbol, today, today + pd.Timedelta(days=1))
    if frame.empty:
        raise RuntimeError(f"no option rows returned for symbol {args.symbol}")
    output = store_raw_options(frame, args.output)
    print(
        {
            "symbol": args.symbol,
            "rows": int(len(frame)),
            "expiries": int(frame["expiry"].nunique()),
            "timestamp": str(frame["timestamp"].iloc[0]),
            "output": str(output),
        }
    )


def _cmd_build_surfaces(args: argparse.Namespace) -> None:
    frame = load_parquet_input(args.input)
    surfaces, dates = build_surface_tensor(frame, SurfaceGridConfig(), DataConfig())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, surfaces=surfaces, dates=dates.astype("datetime64[D]"))
    source_count = len(resolve_parquet_inputs(args.input))
    print(f"wrote {len(surfaces)} surfaces to {output} from {source_count} parquet file(s)")


def _load_surface_payload(path: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    surfaces = payload["surfaces"]
    dates = payload["dates"]
    if len(surfaces) < 3:
        raise ValueError(
            f"{path} contains only {len(surfaces)} surface(s). "
            "Temporal train/validation/test splitting requires at least 3 surfaces. "
            "Use a historical surface file such as data/processed/surfaces_full_history.npz."
        )
    return surfaces, dates


def _cmd_train_jepa(args: argparse.Namespace) -> None:
    surfaces, dates = _load_surface_payload(args.surfaces)
    split = chronological_split(surfaces, dates)
    scaler = SurfaceScaler.fit(split.train)
    train = scaler.transform(split.train).astype("float32")
    val = scaler.transform(split.validation).astype("float32")
    model_config = JEPAConfig(embed_dim=args.embed_dim, mask_ratio=args.mask_ratio)
    train_config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, output_dir=Path(args.output))
    train_jepa(train, val, model_config, train_config)
    np.savez_compressed(Path(args.output) / "scaler.npz", mean=scaler.mean, std=scaler.std)
    print(f"training artifacts written to {args.output}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    surfaces, dates = _load_surface_payload(args.surfaces)
    split = chronological_split(surfaces, dates)
    actual = split.test
    random_walk = surfaces[-len(actual) - 1 : -1]
    historical = HistoricalMeanBaseline().fit(split.train).predict(len(actual))
    scores = {
        "random_walk": {"mse": mse(actual, random_walk), "qlike": qlike(actual, random_walk)},
        "historical_mean": {"mse": mse(actual, historical), "qlike": qlike(actual, historical)},
    }
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(scores, orient="index").to_csv(run_dir / "baseline_scores.csv")
    print(scores)


# ---------------------------------------------------------------------------
# Phase 5+6: run-experiments
# ---------------------------------------------------------------------------

def _train_decoder(
    decoder: SurfaceDecoder,
    latents: np.ndarray,
    surfaces: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> SurfaceDecoder:
    """Train the surface decoder to map latent vectors back to surfaces."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = decoder.to(device)
    flat_surfaces = surfaces.reshape(len(surfaces), -1)
    dataset = TensorDataset(
        torch.as_tensor(latents, dtype=torch.float32),
        torch.as_tensor(flat_surfaces, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        epoch_losses = []
        for z_batch, s_batch in loader:
            z_batch, s_batch = z_batch.to(device), s_batch.to(device)
            pred = decoder(z_batch).reshape(z_batch.shape[0], -1)
            loss = loss_fn(pred, s_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = np.mean(epoch_losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
    if best_state is not None:
        decoder.load_state_dict(best_state)
    print(f"  decoder training complete — best loss: {best_loss:.6f}")
    return decoder


def _rolling_var_forecast(var_model, all_latents: np.ndarray, n_test: int) -> np.ndarray:
    """Rolling one-step-ahead VAR forecast using actual latents at each step.

    For each test index i, uses the actual encoded latents up to t-1
    (not previously predicted values) to forecast exactly one step.
    This puts VAR on equal footing with random walk.
    """
    k_ar = var_model.k_ar
    full_latents = all_latents  # train+val+test latents
    n_total = len(full_latents)
    n_before_test = n_total - n_test
    forecasts = []
    for i in range(n_test):
        idx = n_before_test + i  # index of the test surface we're predicting
        lag_start = idx - k_ar
        if lag_start < 0:
            lag_start = 0
        lag_data = full_latents[lag_start:idx]
        # Pad if not enough lags (shouldn't happen normally)
        if len(lag_data) < k_ar:
            pad = np.repeat(lag_data[:1], k_ar - len(lag_data), axis=0)
            lag_data = np.concatenate([pad, lag_data], axis=0)
        pred = var_model.forecast(lag_data, steps=1)
        forecasts.append(pred[0])
    return np.array(forecasts)


def _rolling_pca_var_forecast(
    pca_var: PCAVARBaseline,
    all_surfaces: np.ndarray,
    n_test: int,
) -> np.ndarray:
    """Rolling one-step-ahead PCA+VAR forecast using actual surfaces at each step."""
    flat_all = all_surfaces.reshape(len(all_surfaces), -1)
    all_scores = pca_var.pca.transform(flat_all)
    k_ar = pca_var.var_.k_ar
    n_total = len(all_scores)
    n_before_test = n_total - n_test
    forecasts = []
    for i in range(n_test):
        idx = n_before_test + i
        lag_start = max(0, idx - k_ar)
        lag_data = all_scores[lag_start:idx]
        if len(lag_data) < k_ar:
            pad = np.repeat(lag_data[:1], k_ar - len(lag_data), axis=0)
            lag_data = np.concatenate([pad, lag_data], axis=0)
        pred = pca_var.var_.forecast(lag_data, steps=1)
        forecasts.append(pred[0])
    score_forecasts_arr = np.array(forecasts)
    reconstructed = pca_var.pca.inverse_transform(score_forecasts_arr)
    return reconstructed.reshape((n_test, *pca_var.surface_shape_))


def _cmd_run_experiments(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load surfaces and split ──────────────────────────────────────
    print("loading surfaces …")
    surfaces, dates = _load_surface_payload(args.surfaces)
    split = chronological_split(surfaces, dates)
    actual = split.test
    n_test = len(actual)
    print(f"  train={len(split.train)}  val={len(split.validation)}  test={n_test}")

    # ── 2. Load scaler ──────────────────────────────────────────────────
    scaler_path = Path(args.model_dir) / "scaler.npz"
    scaler_data = np.load(scaler_path)
    scaler = SurfaceScaler(mean=scaler_data["mean"], std=scaler_data["std"])
    train_scaled = scaler.transform(split.train).astype("float32")
    val_scaled = scaler.transform(split.validation).astype("float32")
    test_scaled = scaler.transform(split.test).astype("float32")

    # ── 3. Load trained JEPA model ──────────────────────────────────────
    print("loading JEPA model …")
    model_path = Path(args.model_dir) / "model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    jepa_config = JEPAConfig(**checkpoint["jepa_config"])
    model = SurfaceJEPA(jepa_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # ── 4. Extract latents for ALL surfaces ─────────────────────────────
    print("extracting latents …")
    all_surfaces_scaled = np.concatenate([train_scaled, val_scaled, test_scaled], axis=0)
    all_latents = extract_latents(model, all_surfaces_scaled, device=device)
    n_train = len(split.train)
    n_val = len(split.validation)
    train_latents = all_latents[:n_train]
    val_latents = all_latents[n_train : n_train + n_val]
    test_latents = all_latents[n_train + n_val :]
    trainval_latents = all_latents[: n_train + n_val]
    print(f"  latent dim = {train_latents.shape[1]}")

    np.savez_compressed(
        run_dir / "latents.npz",
        train=train_latents,
        validation=val_latents,
        test=test_latents,
    )

    # ── 5. Fit VAR on train+val latents ─────────────────────────────────
    print("fitting latent VAR dynamics …")
    var_dynamics = LatentVARDynamics(maxlags=args.var_lags)
    var_dynamics.fit(trainval_latents)
    print(f"  VAR order selected: {var_dynamics.model_.k_ar}")

    # ── 6. Rolling one-step-ahead JEPA forecasts ────────────────────────
    print(f"rolling one-step-ahead forecast for {n_test} test days …")
    forecast_latents = _rolling_var_forecast(var_dynamics.model_, all_latents, n_test)

    # ── 7. Train surface decoder ────────────────────────────────────────
    print("training surface decoder …")
    trainval_surfaces_scaled = np.concatenate([train_scaled, val_scaled], axis=0)
    decoder = SurfaceDecoder(jepa_config.embed_dim, jepa_config.surface_shape)
    decoder = _train_decoder(
        decoder,
        trainval_latents,
        trainval_surfaces_scaled,
        epochs=args.decoder_epochs,
        device=device,
    )

    # ── 8. Decode forecasted latents → surfaces ─────────────────────────
    print("decoding forecasts …")
    decoder.eval()
    with torch.no_grad():
        jepa_forecast_scaled = (
            decoder(torch.as_tensor(forecast_latents, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
    jepa_forecast = scaler.inverse_transform(jepa_forecast_scaled)

    # Also decode actual test latents to measure reconstruction quality
    with torch.no_grad():
        recon_scaled = (
            decoder(torch.as_tensor(test_latents, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
    recon = scaler.inverse_transform(recon_scaled)
    recon_mse = mse(actual, recon)
    print(f"  decoder reconstruction MSE on test: {recon_mse:.6f}")

    # ── 9. Baselines ────────────────────────────────────────────────────
    print("running baselines …")
    random_walk = surfaces[-n_test - 1 : -1]
    historical = HistoricalMeanBaseline().fit(split.train).predict(n_test)

    # PCA+VAR baseline — also rolling one-step-ahead
    pca_var = PCAVARBaseline(n_components=args.pca_components, maxlags=args.var_lags)
    pca_var.fit(split.train)
    pca_evr = pca_var.explained_variance_ratio
    print(f"  PCA explained variance (top {len(pca_evr)}): {np.round(pca_evr, 4).tolist()}")
    print(f"  PCA cumulative: {np.round(np.cumsum(pca_evr), 4).tolist()}")
    pca_var_forecast = _rolling_pca_var_forecast(pca_var, surfaces, n_test)

    # ── 10. Score all methods ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FORECASTING RESULTS  (rolling one-step-ahead)")
    print("=" * 60)

    forecasts = {
        "jepa_latent_var": jepa_forecast,
        "random_walk": random_walk,
        "historical_mean": historical,
        "pca_var": pca_var_forecast,
    }

    method_scores = score_forecasts(actual, forecasts)
    rows = {}
    for ms in method_scores:
        rows[ms.method] = {"mse": ms.mse, "qlike": ms.qlike}
        print(f"  {ms.method:20s}  MSE={ms.mse:.6f}  QLIKE={ms.qlike:.4f}")

    scores_df = pd.DataFrame.from_dict(rows, orient="index")
    scores_df.to_csv(run_dir / "experiment_scores.csv")

    # ── 11. Region-level analysis ───────────────────────────────────────
    print("\n" + "-" * 60)
    print("REGION-LEVEL SCORES (JEPA)")
    print("-" * 60)
    region_scores = score_by_region(actual, jepa_forecast)
    region_rows = {}
    for region, rs in region_scores.items():
        region_rows[region] = {"mse": rs.mse, "qlike": rs.qlike}
        print(f"  {region:15s}  MSE={rs.mse:.6f}  QLIKE={rs.qlike:.4f}")
    pd.DataFrame.from_dict(region_rows, orient="index").to_csv(run_dir / "jepa_region_scores.csv")

    # ── 12. Diebold-Mariano test ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("DIEBOLD-MARIANO TEST (JEPA vs best baseline)")
    print("-" * 60)
    baseline_forecasts = {k: v for k, v in forecasts.items() if k != "jepa_latent_var"}
    try:
        best_name, dm_stat, dm_p = compare_to_best_baseline(actual, jepa_forecast, baseline_forecasts)
        print(f"  best baseline: {best_name}")
        print(f"  DM statistic:  {dm_stat:.4f}")
        print(f"  p-value:       {dm_p:.4f}")
        if dm_p < 0.05:
            winner = "JEPA" if dm_stat < 0 else best_name
            print(f"  → statistically significant at 5%: {winner} is better")
        else:
            print("  → not statistically significant at 5%")
        dm_results = {"best_baseline": best_name, "dm_statistic": dm_stat, "p_value": dm_p}
    except ValueError as exc:
        print(f"  DM test skipped: {exc}")
        dm_results = {"error": str(exc)}

    with open(run_dir / "dm_test.json", "w") as f:
        json.dump(dm_results, f, indent=2)

    # ── 13. Save all forecasts for later use ────────────────────────────
    np.savez_compressed(
        run_dir / "forecasts.npz",
        actual=actual,
        jepa=jepa_forecast,
        random_walk=random_walk,
        historical_mean=historical,
        pca_var=pca_var_forecast,
    )

    print("\n" + "=" * 60)
    print(f"all results saved to {run_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Phase 8: No-Arbitrage Emergence
# ---------------------------------------------------------------------------

def _cmd_arbitrage_check(args: argparse.Namespace) -> None:
    from jepa_iv.surface import butterfly_violation_rate, calendar_violation_rate

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load forecasts from run-experiments
    forecast_path = Path(args.experiment_dir) / "forecasts.npz"
    data = np.load(forecast_path)
    grid = SurfaceGridConfig()
    tenors = grid.tenor_years
    rate = 0.05
    spot = 1.0  # surfaces are on moneyness grid

    methods = {
        "actual": data["actual"],
        "jepa": data["jepa"],
        "random_walk": data["random_walk"],
        "historical_mean": data["historical_mean"],
        "pca_var": data["pca_var"],
    }

    print("=" * 70)
    print("NO-ARBITRAGE EMERGENCE TEST")
    print("=" * 70)

    results = {}
    for name, surfaces in methods.items():
        butterfly_rates = []
        calendar_rates = []
        for surf in surfaces:
            butterfly_rates.append(butterfly_violation_rate(surf, spot, rate, tenors))
            calendar_rates.append(calendar_violation_rate(surf, tenors))
        results[name] = {
            "butterfly_mean": float(np.mean(butterfly_rates)),
            "butterfly_std": float(np.std(butterfly_rates)),
            "calendar_mean": float(np.mean(calendar_rates)),
            "calendar_std": float(np.std(calendar_rates)),
        }
        print(f"\n  {name}:")
        print(f"    butterfly violations: {results[name]['butterfly_mean']:.4f} ± {results[name]['butterfly_std']:.4f}")
        print(f"    calendar  violations: {results[name]['calendar_mean']:.4f} ± {results[name]['calendar_std']:.4f}")

    # Comparison table
    print("\n" + "-" * 70)
    print(f"  {'method':20s}  {'butterfly':>12s}  {'calendar':>12s}")
    print("  " + "-" * 50)
    for name, r in results.items():
        print(f"  {name:20s}  {r['butterfly_mean']:12.4f}  {r['calendar_mean']:12.4f}")

    # Highlight JEPA vs baselines
    print("\n" + "-" * 70)
    jepa_b = results["jepa"]["butterfly_mean"]
    jepa_c = results["jepa"]["calendar_mean"]
    for baseline in ["random_walk", "pca_var", "historical_mean"]:
        b_diff = results[baseline]["butterfly_mean"] - jepa_b
        c_diff = results[baseline]["calendar_mean"] - jepa_c
        b_word = "fewer" if b_diff > 0 else "more"
        c_word = "fewer" if c_diff > 0 else "more"
        print(f"  JEPA vs {baseline}: {abs(b_diff):.4f} {b_word} butterfly, {abs(c_diff):.4f} {c_word} calendar")

    pd.DataFrame.from_dict(results, orient="index").to_csv(run_dir / "arbitrage_violations.csv")
    print(f"\n  results saved to {run_dir / 'arbitrage_violations.csv'}")


# ---------------------------------------------------------------------------
# Phase 7: Interpretability
# ---------------------------------------------------------------------------

def _cmd_interpretability(args: argparse.Namespace) -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load latents ─────────────────────────────────────────────────
    latent_data = np.load(Path(args.experiment_dir) / "latents.npz")
    train_latents = latent_data["train"]
    val_latents = latent_data["validation"]
    test_latents = latent_data["test"]
    all_latents = np.concatenate([train_latents, val_latents, test_latents], axis=0)

    # Load surfaces and dates for alignment
    surface_data = np.load(args.surfaces, allow_pickle=False)
    surfaces = surface_data["surfaces"]
    dates = surface_data["dates"]
    split = chronological_split(surfaces, dates)
    all_dates = np.concatenate([split.train_dates, split.validation_dates, split.test_dates])
    test_dates = split.test_dates
    n_train = len(split.train)
    n_val = len(split.validation)
    n_test = len(split.test)

    # Load forecasts for regime analysis
    forecast_data = np.load(Path(args.experiment_dir) / "forecasts.npz")

    # ── 2. PCA on JEPA latents ──────────────────────────────────────────
    print("=" * 70)
    print("JEPA LATENT SPACE ANALYSIS")
    print("=" * 70)

    n_pcs = min(10, all_latents.shape[1])
    pca_latent = PCA(n_components=n_pcs)
    latent_pcs = pca_latent.fit_transform(all_latents)

    print(f"\n  PCA on JEPA latents (dim={all_latents.shape[1]}):")
    cumvar = np.cumsum(pca_latent.explained_variance_ratio_)
    for i, (ev, cv) in enumerate(zip(pca_latent.explained_variance_ratio_, cumvar)):
        print(f"    PC{i+1:2d}: {ev:.4f}  (cumulative: {cv:.4f})")

    latent_pca_results = {
        "explained_variance_ratio": pca_latent.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumvar.tolist(),
    }

    # ── 3. PCA rediscovery test ─────────────────────────────────────────
    #    Does JEPA just learn PCA? Regress JEPA latent PCs onto surface PCA scores.
    print("\n" + "-" * 70)
    print("PCA REDISCOVERY TEST")
    print("  (How much of JEPA's latent space is explained by surface PCA?)")
    print("-" * 70)

    surface_pca = PCA(n_components=min(10, surfaces.shape[0]))
    flat_surfaces = surfaces.reshape(len(surfaces), -1)
    surface_scores = surface_pca.fit_transform(flat_surfaces)

    print(f"\n  Surface PCA explained variance:")
    surf_cumvar = np.cumsum(surface_pca.explained_variance_ratio_)
    for i in range(min(5, len(surface_pca.explained_variance_ratio_))):
        ev = surface_pca.explained_variance_ratio_[i]
        cv = surf_cumvar[i]
        print(f"    Surface PC{i+1:2d}: {ev:.4f}  (cumulative: {cv:.4f})")

    # Forward probe: predict JEPA latent PCs from surface PCA scores
    print(f"\n  Forward probe (surface PCA → JEPA latent PC):  R²")
    rediscovery_r2 = []
    for i in range(min(5, n_pcs)):
        reg = LinearRegression().fit(surface_scores, latent_pcs[:, i])
        r2 = reg.score(surface_scores, latent_pcs[:, i])
        rediscovery_r2.append(float(r2))
        marker = "≈ PCA" if r2 > 0.8 else "novel" if r2 < 0.3 else "partial"
        print(f"    JEPA PC{i+1} from surface PCA:  R²={r2:.4f}  [{marker}]")

    # Reverse probe: predict surface PCA scores from JEPA latent PCs
    print(f"\n  Reverse probe (JEPA latent PCs → surface PCA):  R²")
    reverse_r2 = []
    for i in range(min(5, len(surface_pca.explained_variance_ratio_))):
        reg = LinearRegression().fit(latent_pcs, surface_scores[:, i])
        r2 = reg.score(latent_pcs, surface_scores[:, i])
        reverse_r2.append(float(r2))
        print(f"    Surface PC{i+1} from JEPA latent:  R²={r2:.4f}")

    # ── 4. VIX factor regression ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("VIX FACTOR REGRESSION")
    print("-" * 70)

    vix_results = {}
    vix_values = None
    try:
        import yfinance as yf

        date_min = str(pd.Timestamp(str(all_dates[0])).date())
        date_max = str((pd.Timestamp(str(all_dates[-1])) + pd.Timedelta(days=5)).date())
        print(f"\n  downloading VIX from {date_min} to {date_max} …")
        vix_df = yf.download("^VIX", start=date_min, end=date_max, progress=False)
        if vix_df.empty:
            raise ValueError("VIX download returned empty DataFrame")

        vix_close = vix_df["Close"].squeeze()
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]

        # Align VIX to surface dates using nearest available
        surface_dates_pd = pd.to_datetime([str(d) for d in all_dates])
        vix_aligned = []
        for d in surface_dates_pd:
            # Find nearest VIX date within 3 business days
            diffs = abs(vix_close.index - d)
            nearest_idx = diffs.argmin()
            if diffs[nearest_idx] <= pd.Timedelta(days=5):
                vix_aligned.append(float(vix_close.iloc[nearest_idx]))
            else:
                vix_aligned.append(np.nan)
        vix_values = np.array(vix_aligned)
        valid_mask = np.isfinite(vix_values)
        coverage = valid_mask.sum() / len(vix_values)
        print(f"  VIX alignment coverage: {coverage:.1%} ({valid_mask.sum()}/{len(vix_values)})")

        if valid_mask.sum() > 30:
            # Regress each latent PC onto VIX
            vix_valid = vix_values[valid_mask].reshape(-1, 1)
            print(f"\n  JEPA latent PC vs VIX:  R²")
            vix_r2_per_pc = []
            for i in range(min(5, n_pcs)):
                pc_valid = latent_pcs[valid_mask, i]
                reg = LinearRegression().fit(vix_valid, pc_valid)
                r2 = reg.score(vix_valid, pc_valid)
                vix_r2_per_pc.append(float(r2))
                strength = "STRONG" if r2 > 0.6 else "moderate" if r2 > 0.3 else "weak"
                print(f"    PC{i+1} vs VIX:  R²={r2:.4f}  [{strength}]")

            # Full multivariate: predict VIX from all latent PCs
            reg_full = LinearRegression().fit(latent_pcs[valid_mask], vix_valid)
            r2_full = reg_full.score(latent_pcs[valid_mask], vix_valid)
            print(f"\n    Full latent → VIX:  R²={r2_full:.4f}")

            vix_results = {
                "per_pc_r2": vix_r2_per_pc,
                "full_r2": float(r2_full),
                "coverage": float(coverage),
            }
        else:
            print("  insufficient VIX alignment for regression, skipping")

    except ImportError:
        print("\n  yfinance not installed — skipping VIX analysis")
        print("  install with: uv sync --extra data")
    except Exception as exc:
        print(f"\n  VIX analysis failed: {exc}")

    # ── 5. Regime analysis ──────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("REGIME ANALYSIS (MSE by VIX tercile)")
    print("-" * 70)

    regime_results = {}
    if vix_values is not None:
        # Get VIX for test period only
        test_vix = vix_values[n_train + n_val :]
        test_valid = np.isfinite(test_vix)

        if test_valid.sum() > 10:
            test_vix_clean = test_vix[test_valid]
            tercile_low = np.percentile(test_vix_clean, 33)
            tercile_high = np.percentile(test_vix_clean, 67)
            print(f"\n  VIX terciles: low < {tercile_low:.1f}, mid {tercile_low:.1f}–{tercile_high:.1f}, high > {tercile_high:.1f}")

            regime_masks = {
                f"low_vix(<{tercile_low:.0f})": test_valid & (test_vix <= tercile_low),
                f"mid_vix({tercile_low:.0f}-{tercile_high:.0f})": test_valid & (test_vix > tercile_low) & (test_vix <= tercile_high),
                f"high_vix(>{tercile_high:.0f})": test_valid & (test_vix > tercile_high),
            }

            actual_test = forecast_data["actual"]
            forecast_methods = {
                "jepa": forecast_data["jepa"],
                "random_walk": forecast_data["random_walk"],
                "historical_mean": forecast_data["historical_mean"],
                "pca_var": forecast_data["pca_var"],
            }

            print(f"\n  {'method':20s}", end="")
            for regime_name in regime_masks:
                print(f"  {regime_name:>18s}", end="")
            print()
            print("  " + "-" * 80)

            for method_name, method_forecast in forecast_methods.items():
                regime_results[method_name] = {}
                print(f"  {method_name:20s}", end="")
                for regime_name, mask in regime_masks.items():
                    if mask.sum() > 0:
                        regime_mse = mse(actual_test[mask], method_forecast[mask])
                        regime_results[method_name][regime_name] = float(regime_mse)
                        print(f"  {regime_mse:18.6f}", end="")
                    else:
                        print(f"  {'N/A':>18s}", end="")
                print()

            # Highlight where JEPA wins/loses per regime
            print()
            for regime_name in regime_masks:
                jepa_mse = regime_results.get("jepa", {}).get(regime_name)
                rw_mse = regime_results.get("random_walk", {}).get(regime_name)
                if jepa_mse is not None and rw_mse is not None:
                    if jepa_mse < rw_mse:
                        print(f"  ✓ JEPA beats random walk in {regime_name} ({jepa_mse:.6f} vs {rw_mse:.6f})")
                    else:
                        print(f"  ✗ Random walk beats JEPA in {regime_name} ({rw_mse:.6f} vs {jepa_mse:.6f})")
        else:
            print("  insufficient VIX data for test period, skipping regime analysis")
    else:
        print("  VIX data not available, skipping regime analysis")

    # ── 6. Save all results ─────────────────────────────────────────────
    summary = {
        "latent_pca": latent_pca_results,
        "rediscovery_r2": rediscovery_r2,
        "reverse_probe_r2": reverse_r2,
        "vix_regression": vix_results,
        "regime_mse": regime_results,
    }
    with open(run_dir / "interpretability.json", "w") as f:
        json.dump(summary, f, indent=2)

    if regime_results:
        pd.DataFrame.from_dict(regime_results, orient="index").to_csv(run_dir / "regime_mse.csv")

    print("\n" + "=" * 70)
    print(f"all interpretability results saved to {run_dir}")
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jepa-iv")
    sub = parser.add_subparsers(required=True)

    smoke = sub.add_parser("smoke-test")
    smoke.set_defaults(func=_cmd_smoke_test)

    pull = sub.add_parser("pull-yfinance")
    pull.add_argument("--symbol", required=True)
    pull.add_argument("--output", default="data/raw/options.parquet")
    pull.set_defaults(func=_cmd_pull_yfinance)

    build = sub.add_parser("build-surfaces")
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(func=_cmd_build_surfaces)

    train = sub.add_parser("train-jepa")
    train.add_argument("--surfaces", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--embed-dim", type=int, default=128)
    train.add_argument("--mask-ratio", type=float, default=0.60)
    train.set_defaults(func=_cmd_train_jepa)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--surfaces", required=True)
    eval_parser.add_argument("--run-dir", required=True)
    eval_parser.set_defaults(func=_cmd_evaluate)

    exp = sub.add_parser("run-experiments", help="Phase 5+6: latent dynamics, forecasting, and full comparison")
    exp.add_argument("--surfaces", required=True, help="Path to surfaces.npz")
    exp.add_argument("--model-dir", required=True, help="Directory with model.pt and scaler.npz (e.g. runs/jepa)")
    exp.add_argument("--run-dir", required=True, help="Output directory for experiment results")
    exp.add_argument("--pca-components", type=int, default=5, help="PCA components for PCA+VAR baseline")
    exp.add_argument("--var-lags", type=int, default=5, help="Max VAR lag order")
    exp.add_argument("--decoder-epochs", type=int, default=200, help="Epochs to train the surface decoder")
    exp.set_defaults(func=_cmd_run_experiments)

    arb = sub.add_parser("arbitrage-check", help="Phase 8: no-arbitrage emergence test")
    arb.add_argument("--experiment-dir", required=True, help="Directory with forecasts.npz (from run-experiments)")
    arb.add_argument("--run-dir", required=True, help="Output directory for arbitrage results")
    arb.set_defaults(func=_cmd_arbitrage_check)

    interp = sub.add_parser("interpretability", help="Phase 7: latent analysis, VIX regression, regime breakdown")
    interp.add_argument("--experiment-dir", required=True, help="Directory with latents.npz and forecasts.npz")
    interp.add_argument("--surfaces", required=True, help="Path to surfaces.npz (for dates and surface PCA)")
    interp.add_argument("--run-dir", required=True, help="Output directory for interpretability results")
    interp.set_defaults(func=_cmd_interpretability)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

