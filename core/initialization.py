"""Patch initialization.

initialize_patches scatters random patches within a 3D viewport-grid box
(or inside the swept volume when one is provided) and returns a list of
Patch objects ready to be passed to SceneOptimizer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from core.patch import ControlPoint, Patch

if TYPE_CHECKING:
    from scene.camera import Camera
    from core.swept_volume import SweptVolume


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_DEFAULT_RADIUS: float = 0.18   # spline radius in local patch units
_BOX_SIZE: float = 3.0
_HALF_EXTENT: float = _BOX_SIZE * 0.5
_BOX_MIN: np.ndarray = np.array(
    [-_HALF_EXTENT, -_HALF_EXTENT, -_HALF_EXTENT],
    dtype=np.float32,
)
_BOX_MAX: np.ndarray = np.array(
    [_HALF_EXTENT, _HALF_EXTENT, _HALF_EXTENT],
    dtype=np.float32,
)
_THETA_CAMERA_MARGIN: float = math.radians(15.0)


# ---------------------------------------------------------------------------
# Spline-patch factory
# ---------------------------------------------------------------------------


def _make_patch(
    center: list[float],
    theta: float,
    radius: float = _DEFAULT_RADIUS,
    albedo: list[float] | None = None,
    device: str = "cpu",
    label: str = "",
) -> Patch:
    """Create a Patch whose spline outline approximates a regular pentagon.

    Control points are placed evenly around a circle of ``radius`` in the
    local XY plane.  Handles are set for a smooth circular approximation
    so the initial shape is a rounded closed curve.

    Args:
        center: [x, y, z] world-space centre of the patch.
        theta:  Y-axis rotation in radians.
        radius: Radius of the initial circle in local units.
        albedo: RGB colour in [0, 1].  Defaults to white.
        device: PyTorch device string.
        label:  Human-readable name for the patch.
    """
    if albedo is None:
        albedo = [1.0, 1.0, 1.0]

    n = Patch.N_CONTROL_POINTS
    handle_scale = radius * (4.0 / 3.0) * math.tan(math.pi / n)

    control_points: list[ControlPoint] = []
    for i in range(n):
        # Start at the top (−π/2) and go counter-clockwise
        angle = 2.0 * math.pi * i / n - math.pi / 2.0
        x_local = radius * math.cos(angle)
        y_local = radius * math.sin(angle)
        # Tangent direction is perpendicular to the radius (CCW)
        handle_rot = angle + math.pi / 2.0

        control_points.append(ControlPoint(
            x=x_local,
            y=y_local,
            z=0.0,
            handle_scale=handle_scale,
            handle_rotation=handle_rot,
            device=device,
        ))

    return Patch(
        control_points=control_points,
        center=center,
        theta=theta,
        albedo=albedo,
        device=device,
        label=label,
    )


def _wrap_theta_half_turn(theta: float) -> float:
    """Wrap a Y rotation to [-pi/2, pi/2), treating theta and theta+pi as equivalent."""
    return ((theta + math.pi * 0.5) % math.pi) - math.pi * 0.5


def _theta_distance(a: float, b: float) -> float:
    return abs(_wrap_theta_half_turn(a - b))


def _camera_yaw_angles(cameras: list["Camera"] | None) -> list[float]:
    if not cameras:
        return [0.0, math.pi * 0.5]
    angles: list[float] = []
    for camera in cameras:
        offset = camera.position - camera.target
        angles.append(_wrap_theta_half_turn(float(math.atan2(offset[0], offset[2]))))
    return angles


def _theta_allowed(
    theta: float,
    camera_angles: list[float],
    margin: float = _THETA_CAMERA_MARGIN,
) -> bool:
    return all(_theta_distance(theta, angle) >= margin for angle in camera_angles)


def _sample_allowed_theta(
    rng: np.random.Generator,
    camera_angles: list[float],
    margin: float = _THETA_CAMERA_MARGIN,
) -> float:
    """Sample theta from the bands between camera edge-on exclusion zones."""
    for _ in range(128):
        theta = float(rng.uniform(-math.pi * 0.5, math.pi * 0.5))
        if _theta_allowed(theta, camera_angles, margin):
            return theta
    return math.radians(45.0)


# ---------------------------------------------------------------------------
# Random initialization
# ---------------------------------------------------------------------------


def initialize_patches(
    n_patches: int,
    cameras: list["Camera"] | None = None,
    radius: float = _DEFAULT_RADIUS,
    device: str = "cpu",
    seed: int | None = None,
    swept_volume: "SweptVolume | None" = None,
) -> list[Patch]:
    """Randomize patch centers within a 3x3x3 viewport-grid box.

    When a swept volume is provided, centers are sampled from it instead so
    every patch starts inside the region both cameras can see.
    """
    rng = np.random.default_rng(seed)
    extents = _BOX_MAX - _BOX_MIN
    patch_radius = max(radius, float(np.cbrt(np.prod(extents) / max(n_patches, 1)) * 0.16))
    camera_angles = _camera_yaw_angles(cameras)

    patches: list[Patch] = []
    for i in range(n_patches):
        if swept_volume is not None:
            point = swept_volume.sample_point(rng)
        else:
            point = rng.uniform(_BOX_MIN, _BOX_MAX).astype(np.float32)
            point = np.clip(point, _BOX_MIN, _BOX_MAX)
        patches.append(_make_patch(
            center=point.tolist(),
            theta=_sample_allowed_theta(rng, camera_angles),
            radius=patch_radius,
            device=device,
            label=f"patch_{i:04d}",
        ))

    return patches
