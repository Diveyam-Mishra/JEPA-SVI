from __future__ import annotations

import torch


def block_mask_indices(
    batch_size: int,
    patch_grid_shape: tuple[int, int],
    mask_ratio: float,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate contiguous block masks and return context/target indices."""

    rows, cols = patch_grid_shape
    num_patches = rows * cols
    target_count = max(1, min(num_patches - 1, round(num_patches * mask_ratio)))
    context_count = num_patches - target_count
    all_context = []
    all_target = []
    for _ in range(batch_size):
        block_h = max(1, round((target_count * rows / cols) ** 0.5))
        block_w = max(1, round(target_count / block_h))
        block_h = min(block_h, rows)
        block_w = min(block_w, cols)
        start_r = int(torch.randint(0, rows - block_h + 1, (1,), device=device).item())
        start_c = int(torch.randint(0, cols - block_w + 1, (1,), device=device).item())
        mask = torch.zeros(rows, cols, dtype=torch.bool, device=device)
        mask[start_r : start_r + block_h, start_c : start_c + block_w] = True
        target = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        if len(target) < target_count:
            remaining = torch.nonzero(~mask.flatten(), as_tuple=False).flatten()
            perm = remaining[torch.randperm(len(remaining), device=device)]
            target = torch.cat([target, perm[: target_count - len(target)]])
        target = target[:target_count]
        context = torch.tensor(
            [idx for idx in range(num_patches) if idx not in set(target.detach().cpu().tolist())],
            device=device,
            dtype=torch.long,
        )
        if len(context) > context_count:
            context = context[:context_count]
        all_context.append(context)
        all_target.append(target)
    return torch.stack(all_context), torch.stack(all_target)
