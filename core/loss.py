"""Loss functions for the optimization loop.

All rendered images are (H, W, C) float32 tensors in [0, 1].
Both views compare rendered pixels against a target image (MSE-style
terms plus silhouette and negative-space penalties).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _match_size(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bilinearly resize ``target`` to match ``rendered`` spatial dims if needed."""
    if rendered.shape[:2] == target.shape[:2]:
        return target
    # (H, W, C) → (1, C, H, W) → resize → (H, W, C)
    t = target.permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(
        t,
        size=(rendered.shape[0], rendered.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return t.squeeze(0).permute(1, 2, 0)


# ---------------------------------------------------------------------------
# MSE loss
# ---------------------------------------------------------------------------


def mse_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared error between a rendered image and a target.

    Args:
        rendered: (H, W, C) float32 — output from the differentiable renderer.
                  May have 4 channels (RGBA); the alpha channel is ignored.
        target:   (H, W, 3) float32 — ground-truth RGB image in [0, 1].
                  Resized automatically if spatial dims differ.
        mask:     (H, W) or (H, W, 1) float32 optional weight map.
                  Useful for ignoring background pixels.

    Returns:
        Scalar loss tensor, differentiable w.r.t. ``rendered``.
    """
    r = rendered[..., :3]              # drop alpha if present → (H, W, 3)
    t = _match_size(r, target.to(r.device))

    diff = (r - t) ** 2               # (H, W, 3)

    if mask is not None:
        m = mask.to(r.device)
        if m.dim() == 2:
            m = m.unsqueeze(-1)        # (H, W, 1)  broadcast over C
        diff = diff * m
        return diff.sum() / (m.sum() * 3.0 + 1e-8)

    return diff.mean()


def silhouette_loss(
    rendered: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error between rendered alpha and a target foreground mask."""
    if rendered.shape[-1] >= 4:
        alpha = rendered[..., 3:4]
    else:
        alpha = rendered[..., :3].amax(dim=-1, keepdim=True)
    mask = _match_size(alpha, target_mask.to(alpha.device))
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    return ((alpha - mask.clamp(0.0, 1.0)) ** 2).mean()


def negative_space_loss(
    rendered: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize rendered coverage in target background/transparent regions."""
    if rendered.shape[-1] >= 4:
        alpha = rendered[..., 3:4]
    else:
        alpha = rendered[..., :3].amax(dim=-1, keepdim=True)
    mask = _match_size(alpha, target_mask.to(alpha.device))
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    background = (1.0 - mask.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return (alpha.square() * background).sum() / (background.sum() + 1e-8)


def masked_rgb_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """RGB loss weighted to the target foreground."""
    if target_mask.dim() == 2:
        target_mask = target_mask.unsqueeze(-1)
    mask = _match_size(rendered[..., :1], target_mask.to(rendered.device)).clamp(0.0, 1.0)
    return mse_loss(rendered, target, mask)
