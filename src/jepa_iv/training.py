from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from jepa_iv.config import JEPAConfig, TrainingConfig
from jepa_iv.masking import block_mask_indices
from jepa_iv.metrics import effective_rank
from jepa_iv.models import SurfaceJEPA


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_tau(base: float, final: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return final
    progress = step / (total_steps - 1)
    return final - (final - base) * (0.5 * (1.0 + np.cos(np.pi * progress)))


def train_jepa(
    train_surfaces: np.ndarray,
    validation_surfaces: np.ndarray | None,
    jepa_config: JEPAConfig = JEPAConfig(),
    training_config: TrainingConfig = TrainingConfig(),
) -> SurfaceJEPA:
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    device = resolve_device(training_config.device)
    model = SurfaceJEPA(jepa_config).to(device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    dataset = TensorDataset(torch.as_tensor(train_surfaces, dtype=torch.float32))
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        drop_last=False,
    )
    total_steps = max(1, training_config.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = nn.MSELoss()
    patch_grid = (
        jepa_config.surface_shape[0] // jepa_config.patch_shape[0],
        jepa_config.surface_shape[1] // jepa_config.patch_shape[1],
    )
    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(training_config.epochs):
        model.train()
        losses = []
        for (batch,) in tqdm(loader, desc=f"epoch {epoch + 1}", leave=False):
            batch = batch.to(device)
            context_idx, target_idx = block_mask_indices(
                len(batch), patch_grid, jepa_config.mask_ratio, device=device
            )
            pred, target = model(batch, context_idx, target_idx)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tau = cosine_tau(jepa_config.ema_tau, jepa_config.ema_tau_final, step, total_steps)
            model.update_target_encoder(tau)
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
        if validation_surfaces is not None and len(validation_surfaces):
            stats = representation_health(model, validation_surfaces, device=device)
            if stats["min_std"] < training_config.collapse_std_floor or stats["effective_rank"] < training_config.effective_rank_floor:
                print(f"representation collapse warning: {stats}")
        if (epoch + 1) % training_config.checkpoint_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "jepa_config": asdict(jepa_config),
                    "training_config": asdict(training_config),
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                },
                output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
            )
    torch.save({"model": model.state_dict(), "jepa_config": asdict(jepa_config)}, output_dir / "model.pt")
    return model


@torch.no_grad()
def extract_latents(
    model: SurfaceJEPA,
    surfaces: np.ndarray,
    *,
    batch_size: int = 256,
    device: torch.device | str | None = None,
    mean_pool: bool = True,
) -> np.ndarray:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    loader = DataLoader(TensorDataset(torch.as_tensor(surfaces, dtype=torch.float32)), batch_size=batch_size)
    latents = []
    for (batch,) in loader:
        encoded = model.context_encoder(batch.to(device))
        if mean_pool:
            encoded = encoded.mean(dim=1)
        latents.append(encoded.cpu().numpy())
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def representation_health(model: SurfaceJEPA, surfaces: np.ndarray, *, device: torch.device) -> dict[str, float]:
    z = extract_latents(model, surfaces, device=device, mean_pool=True)
    return {"min_std": float(z.std(axis=0).min()), "effective_rank": effective_rank(z)}
