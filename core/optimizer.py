"""Optimization loop for the anamorphic sculpture.

SceneOptimizer wraps the render → loss → backward → step cycle.  It is a
plain Python object with no Qt dependency so it can be used from scripts,
notebooks, or the UI worker thread equally.

Typical use (from a script)
---------------------------
    optimizer = SceneOptimizer(patches, renderer, cam1, cam2, t1, t2, lr=1e-3)
    for step, metrics in optimizer.run(n_steps=500):
        print(step, metrics["loss"])

Typical use (from the UI worker)
---------------------------------
    See ui/worker.py — the worker wraps ``run()`` in a QThread and emits
    Qt signals for each step.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from core.loss import masked_rgb_loss, negative_space_loss, silhouette_loss
from core.overlap import OVERLAP_MODES, overlap_loss, planar_overlap_repair
from core.renderer import DiffRenderer
from optimizer.srd import StochasticRewriteDescent

if TYPE_CHECKING:
    from core.patch import Patch
    from core.swept_volume import SweptVolume
    from scene.camera import Camera
    from scene.scene import Mesh


DEFAULT_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.95, 0.95),
    (0.05, 0.05, 0.05),
)
THETA_CAMERA_MARGIN: float = np.deg2rad(15.0)
DEFAULT_HANGING_PLANE_Y: float = 3.5


def parse_palette(text: str | Sequence[str] | Sequence[Sequence[float]] | None) -> torch.Tensor:
    """Parse user-selected colours into an (K, 3) float tensor in [0, 1].

    Accepted forms:
      - "#111111, #f4d35e, #2f6690"
      - ["#111111", "#f4d35e"]
      - [[0.1, 0.2, 0.3], [255, 128, 0]]
    """
    if text is None or text == "":
        return torch.tensor(DEFAULT_PALETTE, dtype=torch.float32)

    if isinstance(text, str):
        raw_items: Sequence[Any] = [p.strip() for p in text.replace(";", ",").split(",")]
    else:
        raw_items = text

    colors: list[list[float]] = []
    for item in raw_items:
        if item is None or item == "":
            continue

        if isinstance(item, str):
            value = item.strip()
            if value.startswith("#"):
                value = value[1:]
            if len(value) == 3:
                value = "".join(ch * 2 for ch in value)
            if len(value) != 6:
                raise ValueError(f"Palette colour {item!r} must be #RGB or #RRGGBB.")
            colors.append([
                int(value[0:2], 16) / 255.0,
                int(value[2:4], 16) / 255.0,
                int(value[4:6], 16) / 255.0,
            ])
            continue

        vals = [float(v) for v in item]
        if len(vals) != 3:
            raise ValueError("Palette RGB entries must contain exactly 3 values.")
        if max(vals) > 1.0:
            vals = [v / 255.0 for v in vals]
        colors.append(vals)

    if not colors:
        colors = [list(c) for c in DEFAULT_PALETTE]

    return torch.tensor(colors, dtype=torch.float32).clamp(0.0, 1.0)


def image_to_tensor(image: str | Path | np.ndarray | torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Load/convert an image to (H, W, 3/4) float32 in [0, 1]."""
    if isinstance(image, torch.Tensor):
        t = image.detach().to(device=device, dtype=torch.float32)
        if t.max() > 1.0:
            t = t / 255.0
        return t[..., :4].clamp(0.0, 1.0)

    if isinstance(image, (str, Path)):
        from PIL import Image

        arr = np.array(Image.open(image).convert("RGBA"))
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = arr[..., :4]
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    if t.max() > 1.0:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def fit_image_to_resolution(
    image: str | Path | np.ndarray | torch.Tensor,
    resolution: tuple[int, int],
    device: str = "cpu",
) -> torch.Tensor:
    """Scale an image into a fixed canvas without changing its aspect ratio."""
    img = image_to_tensor(image, device)
    target_h, target_w = resolution
    src_h, src_w = img.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return torch.zeros(target_h, target_w, img.shape[-1], device=device)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    corners = torch.stack([
        img[0, 0],
        img[0, -1],
        img[-1, 0],
        img[-1, -1],
    ])
    background = corners.median(dim=0).values
    if img.shape[-1] >= 4:
        background = torch.zeros_like(background)
    canvas = background.view(1, 1, -1).expand(target_h, target_w, img.shape[-1]).clone()

    resized = img.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        resized,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas.clamp(0.0, 1.0)


def quantize_to_palette(image: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Map every pixel to the nearest user-selected colour."""
    img = image[..., :3]
    pal = palette.to(device=img.device, dtype=img.dtype).clamp(0.0, 1.0)
    flat = img.reshape(-1, 3)
    distances = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(dim=-1)
    nearest = distances.argmin(dim=1)
    return pal[nearest].reshape_as(img)


def foreground_mask_from_image(
    image: str | Path | np.ndarray | torch.Tensor,
    palette: torch.Tensor,
    device: str = "cpu",
    resolution: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Estimate target foreground as pixels that differ from the corner background."""
    img = (
        fit_image_to_resolution(image, resolution, device)
        if resolution is not None else image_to_tensor(image, device)
    )
    if img.shape[-1] >= 4:
        return img[..., 3:4].clamp(0.0, 1.0)

    q = quantize_to_palette(img, palette)
    h, w = q.shape[:2]
    band = max(1, min(h, w) // 20)
    corners = torch.cat([
        img[:band, :band].reshape(-1, 3),
        img[:band, -band:].reshape(-1, 3),
        img[-band:, :band].reshape(-1, 3),
        img[-band:, -band:].reshape(-1, 3),
    ])
    bg_rgb = corners.median(dim=0).values
    pal = palette.to(device=img.device, dtype=img.dtype)
    bg_idx = ((pal - bg_rgb.unsqueeze(0)) ** 2).sum(dim=1).argmin()
    bg_color = pal[bg_idx]
    mask = (((q - bg_color) ** 2).sum(dim=-1, keepdim=True) > 1e-6).float()
    if float(mask.mean().detach().cpu()) < 1e-4:
        distances = ((img - bg_rgb) ** 2).sum(dim=-1, keepdim=True)
        mask = (distances > 0.02 ** 2).float()
    return mask


def snap_patches_to_palette(
    patches: Sequence["Patch"],
    palette: str | Sequence[str] | Sequence[Sequence[float]] | torch.Tensor | None,
) -> torch.Tensor:
    """Snap each patch albedo to the nearest palette colour."""
    pal = palette if isinstance(palette, torch.Tensor) else parse_palette(palette)
    with torch.no_grad():
        for patch in patches:
            patch_palette = pal.to(device=patch.albedo.device, dtype=patch.albedo.dtype)
            rgb = patch.albedo.detach().clamp(0.0, 1.0)
            idx = ((patch_palette - rgb.unsqueeze(0)) ** 2).sum(dim=1).argmin()
            patch.albedo.copy_(patch_palette[idx])
            patch.albedo.requires_grad_(False)
    return pal


def _parameter_groups(patches: Sequence["Patch"]) -> list[torch.nn.Parameter]:
    """Return learnable shape/orientation parameters, excluding albedo."""
    params: list[torch.nn.Parameter] = []
    for patch in patches:
        params.extend([patch.center, patch.theta])
        for cp in patch.control_points:
            cp.z.requires_grad_(False)
            cp.z.grad = None
            params.extend([
                cp.x,
                cp.y,
                cp.handle_scale,
                cp.handle_rotation,
            ])
    return params


def patch_overlap_loss(
    patches: Sequence["Patch"],
    margin: float = 0.005,
    mode: str = "sphere",
) -> torch.Tensor:
    """Soft penalty for overlapping patches under the selected overlap test."""
    return overlap_loss(patches, margin=margin, mode=mode)


def constrain_patch_to_square_xz_bounds(
    patch: "Patch",
    half_size: float,
    n_per_segment: int = 20,
) -> None:
    """Shift a patch so its sampled outline stays inside square XZ bounds."""
    half = max(float(half_size), 1e-4)
    pts = patch.sample_spline_world(n_per_segment)

    min_x = float(pts[:, 0].min().detach().cpu())
    max_x = float(pts[:, 0].max().detach().cpu())
    min_z = float(pts[:, 2].min().detach().cpu())
    max_z = float(pts[:, 2].max().detach().cpu())

    shift_x = 0.0
    if min_x < -half:
        shift_x = -half - min_x
    if max_x + shift_x > half:
        shift_x += half - (max_x + shift_x)

    shift_z = 0.0
    if min_z < -half:
        shift_z = -half - min_z
    if max_z + shift_z > half:
        shift_z += half - (max_z + shift_z)

    patch.center.data[0].add_(shift_x)
    patch.center.data[2].add_(shift_z)
    patch.center.data[0].clamp_(-half, half)
    patch.center.data[2].clamp_(-half, half)


def _wrap_theta_half_turn(theta: float) -> float:
    """Wrap a Y rotation to [-pi/2, pi/2), treating theta and theta+pi as equivalent."""
    return ((theta + np.pi * 0.5) % np.pi) - np.pi * 0.5


def _theta_distance(a: float, b: float) -> float:
    """Shortest angular distance when opposite patch normals are equivalent."""
    return abs(_wrap_theta_half_turn(a - b))


def _camera_yaw_angles(cameras: Sequence["Camera"]) -> list[float]:
    """Camera yaw angles in the same theta convention used by Patch."""
    angles: list[float] = []
    for camera in cameras:
        offset = camera.position - camera.target
        angles.append(_wrap_theta_half_turn(float(np.arctan2(offset[0], offset[2]))))
    return angles


def theta_allowed(
    theta: float,
    camera_angles: Sequence[float],
    margin: float = THETA_CAMERA_MARGIN,
) -> bool:
    """Return True when theta is at least margin radians away from every camera yaw."""
    return all(_theta_distance(theta, angle) >= margin for angle in camera_angles)


def constrain_theta_to_camera_band(
    theta: float,
    camera_angles: Sequence[float],
    margin: float = THETA_CAMERA_MARGIN,
) -> float:
    """Project theta to the nearest orientation outside the camera edge-on margin."""
    theta = _wrap_theta_half_turn(theta)
    if theta_allowed(theta, camera_angles, margin):
        return theta

    candidates: list[float] = []
    for angle in camera_angles:
        candidates.append(_wrap_theta_half_turn(angle - margin))
        candidates.append(_wrap_theta_half_turn(angle + margin))

    valid = [
        candidate for candidate in candidates
        if theta_allowed(candidate, camera_angles, margin * 0.999)
    ]
    if not valid:
        valid = candidates
    return min(valid, key=lambda candidate: _theta_distance(theta, candidate))


class SceneOptimizer:
    """Render, compare to quantized target images, and Adam-step patches."""

    def __init__(
        self,
        patches: list["Patch"],
        camera1: "Camera",
        camera2: "Camera",
        target1: str | Path | np.ndarray | torch.Tensor,
        target2: str | Path | np.ndarray | torch.Tensor | None = None,
        *,
        palette: str | Sequence[str] | Sequence[Sequence[float]] | None = None,
        renderer: DiffRenderer | None = None,
        resolution: tuple[int, int] = (192, 256),
        lr: float = 1e-3,
        device: str = "cpu",
        n_per_segment: int = 20,
        silhouette_weight: float = 2.0,
        negative_space_weight: float = 2.5,  # used to be 4.0
        #overlap_weight: float = 0.05,
        overlap_weight: float = 0.7,
        overlap_margin: float = 0.005,
        overlap_mode: str = "sphere",
        overlap_repair: bool = False,
        overlap_repair_interval: int = 5,
        theta_camera_margin: float = THETA_CAMERA_MARGIN,
        hanging_plane_size: float = 5.0,
        hanging_plane_y: float = DEFAULT_HANGING_PLANE_Y,
        min_patch_area: float = 0.001,
        srd_config: dict[str, object] | None = None,
        swept_volume: "SweptVolume | None" = None,
    ) -> None:
        if not patches:
            raise ValueError("SceneOptimizer requires at least one patch.")

        self.patches = patches
        self.camera1 = camera1
        self.camera2 = camera2
        self.device = device
        self.resolution = resolution
        self.render_resolutions = resolution
        self.silhouette_weight = silhouette_weight
        self.negative_space_weight = negative_space_weight
        self.overlap_weight = overlap_weight
        self.overlap_margin = overlap_margin
        if overlap_mode not in OVERLAP_MODES:
            raise ValueError(
                f"Unknown overlap_mode {overlap_mode!r}; expected one of {OVERLAP_MODES}."
            )
        self.overlap_mode = overlap_mode
        # Repair only exists for the planar test: it needs the exact
        # plane-intersection criterion to know a pair really interpenetrates.
        self.overlap_repair = bool(overlap_repair) and overlap_mode == "planar"
        self.overlap_repair_interval = max(1, int(overlap_repair_interval))
        self._repair_counter = 0
        self.last_repaired_pairs = 0
        self.last_repair_shift = 0.0
        self.theta_camera_margin = theta_camera_margin
        self.theta_camera_angles = _camera_yaw_angles((camera1, camera2))
        self.hanging_plane_size = hanging_plane_size
        self.hanging_plane_y = hanging_plane_y
        self.min_patch_area = min_patch_area
        self.swept_volume = swept_volume

        self.palette = parse_palette(palette).to(device)
        target1_fit = fit_image_to_resolution(target1, self.resolution, device)
        self.target1_is_mask = target1_fit.shape[-1] >= 4
        self.target1 = quantize_to_palette(target1_fit, self.palette)
        self.target1_mask = foreground_mask_from_image(
            target1_fit,
            self.palette,
            device,
        )
        target2_fit = (
            fit_image_to_resolution(target2, self.resolution, device)
            if target2 is not None else None
        )
        self.target2_is_mask = target2_fit is not None and target2_fit.shape[-1] >= 4
        self.target2 = (
            quantize_to_palette(target2_fit, self.palette)
            if target2_fit is not None else None
        )
        self.target2_mask = (
            foreground_mask_from_image(target2_fit, self.palette, device)
            if target2_fit is not None else None
        )

        self.renderer = renderer or DiffRenderer(device=device, n_per_segment=n_per_segment)
        self.optim = torch.optim.Adam(_parameter_groups(patches), lr=lr)
        snap_patches_to_palette(self.patches, self.palette)
        self._post_step_constraints()
        if srd_config is not None:
            srd_kwargs = dict(srd_config)
            srd_kwargs["swept_volume"] = swept_volume
            self.srd: StochasticRewriteDescent | None = StochasticRewriteDescent(**srd_kwargs)
        else:
            self.srd = None

    def rebuild_optim(self) -> None:
        """Recreate the optimizer after the patch list was edited externally.

        Required when pieces are added or deleted outside the optimization
        loop (e.g. UI edits while paused) so Adam tracks the current
        parameter set. No-op when there are no patches left.
        """
        if not self.patches:
            return
        defaults = self.optim.defaults.copy()
        self.optim = self.optim.__class__(_parameter_groups(self.patches), **defaults)
        self._post_step_constraints()

    def step(self, step_idx: int = 1, total_steps: int = 1) -> dict[str, float]:
        smallest_area = self._smallest_patch_area()
        metrics = self._continuous_step_with_optimizer(self.optim)
        metrics["smallest_patch_area"] = smallest_area
        metrics["tiny_patches_deleted"] = 0.0
        if self.srd is not None:
            stats = self.srd.step(
                self,
                self.optim,
                metrics["loss"],
                (self.camera1, self.camera2),
                (self.target1, self.target2),
                step_idx,
            )
            metrics.update({
                "srd_active_patches": float(stats.active),
                "srd_added": float(stats.added),
                "srd_deleted": float(stats.deleted),
                "srd_total_adds": float(stats.total_adds),
                "srd_total_splits": float(stats.total_splits),
                "srd_total_growth": float(stats.total_added),
                "srd_total_deletes": float(stats.total_deleted),
                "srd_evaluated": float(stats.evaluated),
                "srd_promising": float(stats.promising),
                "srd_accepted": float(stats.accepted),
            })
            metrics["tiny_patches_deleted"] = float(stats.deleted)
        metrics["self_intersections_prevented"] = float(
            sum(getattr(patch, "self_intersect_counter", 0) for patch in self.patches)
        )
        return metrics

    def _smallest_patch_area(self) -> float:
        if not self.patches:
            return 0.0
        return min(float(patch.compute_area().detach().cpu()) for patch in self.patches)

    def _continuous_step_with_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        valid_shape_states = self._capture_patch_shape_states()
        optimizer.zero_grad(set_to_none=True)
        render1, render2 = self.renderer.render_both(
            self.patches,
            self.camera1,
            self.camera2,
            self.resolution,
        )
        loss, components = self._loss_from_renders(render1, render2, self.patches)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self._post_step_constraints(valid_shape_states, optimizer)
        return self._metrics_from_components(loss, components)

    def _metrics_from_components(
        self,
        loss: torch.Tensor,
        components: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        loss_value = float(loss.detach().cpu())
        return {
            "loss": loss_value,
            "patches": float(len(self.patches)),
            "view1_mse": float(components["loss1"].detach().cpu()),
            "view2_loss": float(components["loss2"].detach().cpu()),
            "view1_silhouette": float(components["loss1_silhouette"].detach().cpu()),
            "view2_silhouette": float(components["loss2_silhouette"].detach().cpu()),
            "view1_negative_space": float(components["loss1_negative_space"].detach().cpu()),
            "view2_negative_space": float(components["loss2_negative_space"].detach().cpu()),
            "overlap": float(components["overlap"].detach().cpu()),
            "negative_space_weighted": float(
                (
                    self.negative_space_weight
                    * (components["loss1_negative_space"] + components["loss2_negative_space"])
                ).detach().cpu()
            ),
            "overlap_weighted": float((self.overlap_weight * components["overlap"]).detach().cpu()),
            "overlap_repaired_pairs": float(self.last_repaired_pairs),
            "overlap_repair_shift": float(self.last_repair_shift),
        }

    def _silhouette_stats(
        self,
        render: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, float]:
        """IoU and its false-positive/false-negative decomposition.

        `target_mask` must be a foreground mask as produced by
        `foreground_mask_from_image` (shape (H, W, 1), float in 0..1), not a
        colour image. Deriving the target silhouette from colour is wrong here:
        the targets are flat shapes whose silhouette lives in the alpha channel,
        so a black shape quantises to an all-zero RGB image and would score a
        constant IoU of 0 regardless of how well the render matched it.

        IoU already charges for negative space: area the render covers but the
        target does not enters the union, so spilling outside the silhouette
        lowers it just as failing to fill it does. What IoU cannot tell you is
        *which* of the two happened, so the decomposition is reported alongside:

            coverage   |R & T| / |T|   how much of the target got filled
            precision  |R & T| / |R|   how much of the render landed on target
            spill      |R & ~T| / |T|  area covered that should be empty,
                                       as a fraction of the target's own area

        `spill` is the negative-space number: 0 means the render never paints
        outside the silhouette, 0.5 means it wrongly covers an extra half of a
        target's worth of background. It is normalised by the target area
        rather than by the frame so that it stays sensitive -- the background
        is most of the image, so an IoU taken over the complement masks would
        sit near 1.0 for every run and separate nothing.
        """
        # Convert to CPU numpy if needed
        if isinstance(render, torch.Tensor):
            render_np = render.numpy() if render.is_cpu else render.cpu().numpy()
        else:
            render_np = render

        if isinstance(target_mask, torch.Tensor):
            target_np = (
                target_mask.numpy() if target_mask.is_cpu else target_mask.cpu().numpy()
            )
        else:
            target_np = target_mask

        # Render is always RGBA; its silhouette is the alpha channel.
        render_binary = render_np[..., 3] > 0.5

        # Drop the trailing singleton channel so the masks broadcast as (H, W).
        if target_np.ndim == render_binary.ndim + 1:
            target_np = target_np[..., 0]
        target_binary = target_np > 0.5

        if render_binary.shape != target_binary.shape:
            raise ValueError(
                f"IoU shape mismatch: render {render_binary.shape} vs "
                f"target mask {target_binary.shape}"
            )

        intersection = float((render_binary & target_binary).sum())
        union = float((render_binary | target_binary).sum())
        render_area = float(render_binary.sum())
        target_area = float(target_binary.sum())
        false_positive = render_area - intersection

        return {
            "iou": intersection / union if union else 0.0,
            "coverage": intersection / target_area if target_area else 0.0,
            "precision": intersection / render_area if render_area else 0.0,
            "spill": false_positive / target_area if target_area else 0.0,
        }

    def _calculate_iou(self, render: torch.Tensor, target_mask: torch.Tensor) -> float:
        """Intersection over Union between the rendered and target silhouettes."""
        return self._silhouette_stats(render, target_mask)["iou"]

    def _loss_from_renders(
        self,
        render1: torch.Tensor,
        render2: torch.Tensor,
        patches: Sequence["Patch"],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss1_rgb = (
            torch.zeros((), device=render1.device)
            if self.target1_is_mask else masked_rgb_loss(render1, self.target1, self.target1_mask)
        )
        loss1_silhouette = silhouette_loss(render1, self.target1_mask)
        loss1_negative_space = negative_space_loss(render1, self.target1_mask)
        loss1 = (
            loss1_rgb
            + self.silhouette_weight * loss1_silhouette
            + self.negative_space_weight * loss1_negative_space
        )
        loss2 = torch.zeros((), device=loss1.device)
        loss2_silhouette = torch.zeros((), device=loss1.device)
        loss2_negative_space = torch.zeros((), device=loss1.device)
        if self.target2 is not None:
            assert self.target2_mask is not None
            loss2_rgb = (
                torch.zeros((), device=render2.device)
                if self.target2_is_mask else masked_rgb_loss(render2, self.target2, self.target2_mask)
            )
            loss2_silhouette = silhouette_loss(render2, self.target2_mask)
            loss2_negative_space = negative_space_loss(render2, self.target2_mask)
            loss2 = (
                loss2_rgb
                + self.silhouette_weight * loss2_silhouette
                + self.negative_space_weight * loss2_negative_space
            )

        if patches:
            overlap = patch_overlap_loss(patches, self.overlap_margin, self.overlap_mode)
        else:
            overlap = torch.zeros((), device=loss1.device)
        loss = (
            loss1
            + loss2
            + self.overlap_weight * overlap
        )
        return loss, {
            "loss1": loss1,
            "loss2": loss2,
            "loss1_silhouette": loss1_silhouette,
            "loss2_silhouette": loss2_silhouette,
            "loss1_negative_space": loss1_negative_space,
            "loss2_negative_space": loss2_negative_space,
            "overlap": overlap,
        }

    def _capture_patch_shape_states(self) -> dict[int, list[torch.Tensor]]:
        """Snapshot local outline parameters for hard-constraint rollback."""
        return {
            id(patch): [
                value.detach().clone()
                for cp in patch.control_points
                for value in (cp.x, cp.y, cp.handle_scale, cp.handle_rotation)
            ]
            for patch in self.patches
        }

    @staticmethod
    def _restore_patch_shape(
        patch: "Patch",
        state: list[torch.Tensor],
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        parameters = [
            value
            for cp in patch.control_points
            for value in (cp.x, cp.y, cp.handle_scale, cp.handle_rotation)
        ]
        for parameter, saved in zip(parameters, state):
            parameter.copy_(saved)
            if optimizer is not None:
                optimizer.state.pop(parameter, None)

    def _post_step_constraints(
        self,
        valid_shape_states: dict[int, list[torch.Tensor]] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        with torch.no_grad():
            half_plane = max(float(self.hanging_plane_size) * 0.5, 1e-4)
            for patch in self.patches:
                patch.center.data = torch.nan_to_num(patch.center.data, nan=0.0)
                patch.theta.data = torch.nan_to_num(patch.theta.data, nan=0.0)
                constrained_theta = constrain_theta_to_camera_band(
                    float(patch.theta.detach().cpu()),
                    self.theta_camera_angles,
                    self.theta_camera_margin,
                )
                patch.theta.copy_(patch.theta.new_tensor(constrained_theta))
                for cp in patch.control_points:
                    cp.x.data = torch.nan_to_num(cp.x.data, nan=0.0)
                    cp.y.data = torch.nan_to_num(cp.y.data, nan=0.0)
                    cp.z.data.zero_()
                    cp.handle_scale.data = torch.nan_to_num(cp.handle_scale.data, nan=0.01).clamp(0.01, 2.0)
                    cp.handle_rotation.data = torch.nan_to_num(cp.handle_rotation.data, nan=0.0)
                constrain_patch_to_square_xz_bounds(patch, half_plane)
                if patch.is_self_intersecting():
                    saved_shape = (
                        valid_shape_states.get(id(patch))
                        if valid_shape_states is not None else None
                    )
                    if saved_shape is None:
                        raise ValueError(
                            f"Patch {patch.label!r} has a self-intersecting Bezier outline."
                        )
                    self._restore_patch_shape(patch, saved_shape, optimizer)
                    patch.self_intersect_counter = (
                        int(getattr(patch, "self_intersect_counter", 0)) + 1
                    )
            self._repair_overlaps(half_plane)

    def _repair_overlaps(self, half_plane: float) -> None:
        """Separate patches the exact planar test finds interpenetrating.

        Runs on an interval rather than every step: the exact test costs one
        extra outline sampling pass over every patch, and a hard constraint
        applied a few steps late is harmless when the gradient term is already
        pushing the same pairs apart.
        """
        if not self.overlap_repair or len(self.patches) < 2:
            return
        self._repair_counter += 1
        if self._repair_counter % self.overlap_repair_interval != 0:
            return

        pairs, shift = planar_overlap_repair(self.patches, self.overlap_margin)
        self.last_repaired_pairs = pairs
        self.last_repair_shift = shift
        if pairs:
            for patch in self.patches:
                constrain_patch_to_square_xz_bounds(patch, half_plane)

    def mesh_snapshot(self, n_per_segment: int = 20) -> list["Mesh"]:
        return [p.to_mesh(n_per_segment=n_per_segment) for p in self.patches]

    def evaluate_snapshot(
        self,
    ) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        """Evaluate and return detached final renders without changing parameters."""
        with torch.no_grad():
            render1, render2 = self.renderer.render_both(
                self.patches,
                self.camera1,
                self.camera2,
                self.resolution,
            )
            loss, components = self._loss_from_renders(
                render1,
                render2,
                self.patches,
            )
            metrics = self._metrics_from_components(loss, components)

            # Add IOU metrics, plus the coverage/precision/spill decomposition
            # that says whether a given IoU is losing area to unfilled target
            # or to render spilling into negative space.
            render1_cpu = render1.detach().cpu()
            render2_cpu = render2.detach().cpu()
            stats1 = self._silhouette_stats(render1_cpu, self.target1_mask)
            stats2 = (
                self._silhouette_stats(render2_cpu, self.target2_mask)
                if self.target2_mask is not None
                else {"iou": 0.0, "coverage": 0.0, "precision": 0.0, "spill": 0.0}
            )
            for key, value in stats1.items():
                metrics[f"view1_{key}"] = value
            if self.target2 is not None:
                for key, value in stats2.items():
                    metrics[f"view2_{key}"] = value
                for key in stats1:
                    metrics[f"mean_{key}"] = (stats1[key] + stats2[key]) / 2.0

        return (
            metrics,
            render1.detach().cpu(),
            render2.detach().cpu(),
        )

    def run(self, n_steps: int = 500) -> Iterator[tuple[int, dict[str, float]]]:
        for step_idx in range(1, n_steps + 1):
            yield step_idx, self.step(step_idx, n_steps)
