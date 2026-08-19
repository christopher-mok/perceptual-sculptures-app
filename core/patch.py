"""Spline-based patch primitive — the basic unit of the anamorphic sculpture.

Each Patch is a flat laser-cut piece whose outline is a closed cubic Bezier
spline with exactly 5 control points.  The piece stands upright and is
oriented by a single Y-axis rotation (theta).

Coordinate convention
---------------------
Local space  : control points live in the XY plane (z = 0).
               X = horizontal extent of the piece.
               Y = vertical extent (up).
After rot_y(theta) + translation to center → world space.

Bezier handles
--------------
Each control point has symmetric smooth handles (G1 continuity):
    handle_out = handle_scale * [cos(handle_rotation), sin(handle_rotation), 0]
    handle_in  = -handle_out

This matches Illustrator-style "smooth" nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from scene.scene import Mesh


# ---------------------------------------------------------------------------
# Outline triangulation
# ---------------------------------------------------------------------------


def _triangulate_outline_xy(xy: np.ndarray) -> np.ndarray:
    """Triangulate a closed 2-D outline into (F, 3) int32 vertex indices.

    Uses mapbox-earcut so non-convex outlines are filled correctly (a
    centroid/vertex fan over-covers concave shapes). A vertex-0 fan is used
    only as a last resort when earcut finds no triangles (degenerate
    polygon).
    """
    import mapbox_earcut

    n = len(xy)
    rings = np.array([n], dtype=np.uint32)
    indices = mapbox_earcut.triangulate_float64(
        np.ascontiguousarray(xy, dtype=np.float64),
        rings,
    )
    if len(indices) >= 3:
        return np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    return np.array(
        [[0, i, i + 1] for i in range(1, n - 1)],
        dtype=np.int32,
    )


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------


def rot_y(theta: torch.Tensor) -> torch.Tensor:
    """3×3 Y-axis rotation matrix. Fully differentiable."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack([
        torch.stack([ c, z, s]),
        torch.stack([ z, o, z]),
        torch.stack([-s, z, c]),
    ])  # (3, 3)


# ---------------------------------------------------------------------------
# Control point
# ---------------------------------------------------------------------------


class ControlPoint(nn.Module):
    """One node of the patch's closed Bezier spline.

    Parameters
    ----------
    x, y, z         : Position in local patch space.  z is initialised to 0
                       (flat piece) but is left learnable.
    handle_scale     : Length of both tangent handles (enforced positive).
    handle_rotation  : Direction angle of the outgoing handle in the XY plane.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        handle_scale: float = 0.15,
        handle_rotation: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        # Linked by Patch after construction. These are plain references, not
        # registered submodules, so the ModuleList below remains the owner.
        object.__setattr__(self, "next_control_point", None)
        object.__setattr__(self, "prev_control_point", None)
        self.x               = nn.Parameter(torch.tensor(x,               dtype=torch.float32, device=device))
        self.y               = nn.Parameter(torch.tensor(y,               dtype=torch.float32, device=device))
        self.z               = nn.Parameter(torch.tensor(z,               dtype=torch.float32, device=device))
        self.handle_scale    = nn.Parameter(torch.tensor(handle_scale,    dtype=torch.float32, device=device))
        self.handle_rotation = nn.Parameter(torch.tensor(handle_rotation, dtype=torch.float32, device=device))

    @property
    def pos(self) -> torch.Tensor:
        """(3,) position tensor."""
        return torch.stack([self.x, self.y, self.z])

    def handle_out(self) -> torch.Tensor:
        """(3,) outgoing tangent handle in local patch XY plane."""
        hs = self.handle_scale.abs()   # positive length
        return hs * torch.stack([
            torch.cos(self.handle_rotation),
            torch.sin(self.handle_rotation),
            torch.zeros_like(hs),
        ])

    def handle_in(self) -> torch.Tensor:
        """(3,) incoming tangent handle — symmetric opposite of handle_out."""
        return -self.handle_out()


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------


class Patch(nn.Module):
    """A laser-cut piece defined by a closed cubic Bezier spline.

    The spline has N_CONTROL_POINTS = 5 nodes in local XY space.
    The whole piece is oriented by theta (Y-axis rotation) and placed at center.

    Learnable parameters
    --------------------
    center          : (3,) world-space position.
    theta           : scalar Y-axis rotation in radians.
    albedo          : (3,) RGB surface colour in [0, 1].
    control_points  : 5 × ControlPoint  (each has x, y, z, handle_scale, handle_rotation)
    """

    N_CONTROL_POINTS: int = 5
    DEFAULT_THICKNESS: float = 0.035

    def __init__(
        self,
        control_points: list[ControlPoint],
        center: list | np.ndarray,
        theta: float,
        albedo: list | tuple | np.ndarray = (0.6, 0.7, 0.9),
        device: str = "cpu",
        label: str = "",
    ) -> None:
        super().__init__()
        if len(control_points) != self.N_CONTROL_POINTS:
            raise ValueError(f"Patch requires exactly {self.N_CONTROL_POINTS} control points, got {len(control_points)}")

        self.control_points = nn.ModuleList(control_points)
        self._link_control_points()
        self.center = nn.Parameter(torch.tensor(center, dtype=torch.float32, device=device))
        self.theta  = nn.Parameter(torch.tensor(theta,  dtype=torch.float32, device=device))
        self.albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float32, device=device))
        self.label  = label
        self.creation_step = 0
        self.self_intersect_counter = 0

    def _link_control_points(self) -> None:
        """Attach next/previous links for the closed control-point chain."""
        n = len(self.control_points)
        for i, cp in enumerate(self.control_points):
            object.__setattr__(cp, "prev_control_point", self.control_points[(i - 1) % n])
            object.__setattr__(cp, "next_control_point", self.control_points[(i + 1) % n])

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def rotation_matrix(self) -> torch.Tensor:
        """(3, 3) Y-axis rotation matrix."""
        return rot_y(self.theta)

    def local_to_world(self, local_pts: torch.Tensor) -> torch.Tensor:
        """Map (N, 3) local-space points to world space."""
        R = self.rotation_matrix()
        return (R @ local_pts.T).T + self.center.unsqueeze(0)

    # ------------------------------------------------------------------
    # Spline sampling (differentiable)
    # ------------------------------------------------------------------

    def sample_spline_local(self, n_per_segment: int = 20) -> torch.Tensor:
        """Sample the closed Bezier spline in local space.

        Returns (N_CONTROL_POINTS * n_per_segment, 3).
        The last sample of each segment is dropped to avoid duplicate
        points at segment joins (the closed loop reconnects automatically).
        """
        device = self.center.device
        t = torch.linspace(0.0, 1.0, n_per_segment + 1, device=device)[:-1]  # (T,)

        segments: list[torch.Tensor] = []
        n = self.N_CONTROL_POINTS
        for i in range(n):
            cp0 = cast(ControlPoint, self.control_points[i])
            cp1 = cast(ControlPoint, self.control_points[(i + 1) % n])

            P0 = cp0.pos
            P1 = P0 + cp0.handle_out()
            P2 = cp1.pos + cp1.handle_in()
            P3 = cp1.pos

            # Cubic Bezier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
            t_ = t.unsqueeze(1)          # (T, 1)
            pts = (
                (1 - t_) ** 3           * P0 +
                3 * (1 - t_) ** 2 * t_  * P1 +
                3 * (1 - t_)      * t_**2 * P2 +
                t_ ** 3                 * P3
            )  # (T, 3)
            segments.append(pts)

        return torch.cat(segments, dim=0)   # (N*T, 3)

    def sample_spline_world(self, n_per_segment: int = 20) -> torch.Tensor:
        """Spline points in world space. Returns (N*T, 3)."""
        return self.local_to_world(self.sample_spline_local(n_per_segment))

    def compute_area(self, n_per_segment: int = 20) -> torch.Tensor:
        """Differentiable signed-outline area magnitude in local patch space."""
        pts = self.sample_spline_local(n_per_segment)
        xy = pts[:, :2]
        nxt = torch.roll(xy, shifts=-1, dims=0)
        cross = xy[:, 0] * nxt[:, 1] - xy[:, 1] * nxt[:, 0]
        return 0.5 * cross.sum().abs()

    def is_self_intersecting(
        self,
        n_per_segment: int = 20,
        epsilon: float = 1e-7,
    ) -> bool:
        """Return whether the sampled closed outline contains crossing segments."""
        with torch.no_grad():
            pts = self.sample_spline_local(n_per_segment).detach().cpu().numpy()
            n = len(pts)
            if n < 4:
                return False

            def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
                return float(
                    (b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0])
                )

            def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
                return (
                    min(a[0], b[0]) - epsilon <= p[0] <= max(a[0], b[0]) + epsilon
                    and min(a[1], b[1]) - epsilon <= p[1] <= max(a[1], b[1]) + epsilon
                )

            def _segments_intersect(
                a0: np.ndarray,
                a1: np.ndarray,
                b0: np.ndarray,
                b1: np.ndarray,
            ) -> bool:
                o1 = _orientation(a0, a1, b0)
                o2 = _orientation(a0, a1, b1)
                o3 = _orientation(b0, b1, a0)
                o4 = _orientation(b0, b1, a1)

                if (
                    ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon))
                    and ((o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon))
                ):
                    return True
                if abs(o1) <= epsilon and _on_segment(a0, a1, b0):
                    return True
                if abs(o2) <= epsilon and _on_segment(a0, a1, b1):
                    return True
                if abs(o3) <= epsilon and _on_segment(b0, b1, a0):
                    return True
                if abs(o4) <= epsilon and _on_segment(b0, b1, a1):
                    return True
                return False

            for i in range(n):
                a0 = pts[i, :2]
                a1 = pts[(i + 1) % n, :2]
                for j in range(i + 1, n):
                    if j == i or j == (i + 1) % n or (j + 1) % n == i:
                        continue
                    b0 = pts[j, :2]
                    b1 = pts[(j + 1) % n, :2]
                    if _segments_intersect(a0, a1, b0, b1):
                        return True
        return False

    def split_down_middle(self, creation_step: int = 0) -> tuple["Patch", "Patch"]:
        """Split a five-point patch into two smaller five-point child patches.

        The split is built in local space, using the top-most control point and
        the midpoint between the two lowest control points as the split line.
        """
        with torch.no_grad():
            pts = torch.stack([cast(ControlPoint, cp).pos for cp in self.control_points], dim=0)
            xy = pts[:, :2].detach().cpu().numpy()
            top = xy[int(np.argmax(xy[:, 1]))]
            bottom_pair = xy[np.argsort(xy[:, 1])[:2]]
            bottom_mid = bottom_pair.mean(axis=0)
            center_mid = 0.5 * (top + bottom_mid)

            min_x = float(np.min(xy[:, 0]))
            max_x = float(np.max(xy[:, 0]))
            min_y = float(np.min(xy[:, 1]))
            max_y = float(np.max(xy[:, 1]))
            mid_x = float(bottom_mid[0])
            width = max(max_x - min_x, 1e-3)
            height = max(max_y - min_y, 1e-3)

            left_mid_x = 0.5 * (min_x + mid_x)
            right_mid_x = 0.5 * (max_x + mid_x)
            upper_y = min_y + 0.72 * height

            left_points = [
                top,
                np.array([left_mid_x, upper_y], dtype=np.float32),
                np.array([min_x, min_y + 0.2 * height], dtype=np.float32),
                bottom_mid,
                center_mid,
            ]
            right_points = [
                top,
                center_mid,
                bottom_mid,
                np.array([max_x, min_y + 0.2 * height], dtype=np.float32),
                np.array([right_mid_x, upper_y], dtype=np.float32),
            ]

            albedo = self.albedo.detach().cpu().numpy().tolist()
            center = self.center.detach().cpu().numpy().tolist()
            theta = float(self.theta.detach().cpu())

        def _make_child(points: list[np.ndarray], suffix: str) -> "Patch":
            scale = max(width, height) * 0.08
            control_points: list[ControlPoint] = []
            n = len(points)
            for idx, point in enumerate(points):
                prev_point = points[(idx - 1) % n]
                next_point = points[(idx + 1) % n]
                tangent = np.asarray(next_point, dtype=np.float32) - np.asarray(prev_point, dtype=np.float32)
                angle = float(np.arctan2(tangent[1], tangent[0]))
                control_points.append(ControlPoint(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=0.0,
                    handle_scale=float(max(scale, 0.01)),
                    handle_rotation=angle,
                    device=str(self.center.device),
                ))
            child = Patch(
                control_points=control_points,
                center=center,
                theta=theta,
                albedo=albedo,
                device=str(self.center.device),
                label=f"{self.label}_{suffix}" if self.label else suffix,
            )
            child.creation_step = creation_step
            return child

        return _make_child(left_points, "split_a"), _make_child(right_points, "split_b")

    # ------------------------------------------------------------------
    # Mesh for homogeneous coordinate assembly (renderer)
    # ------------------------------------------------------------------

    def world_vertices_homogeneous(self, n_per_segment: int = 20) -> torch.Tensor:
        """(V, 4) world-space spline points with homogeneous w=1, for nvdiffrast."""
        pts  = self.sample_spline_world(n_per_segment)  # (V, 3)
        ones = torch.ones(len(pts), 1, dtype=pts.dtype, device=pts.device)
        return torch.cat([pts, ones], dim=1)             # (V, 4)

    def extruded_mesh_world(
        self,
        n_per_segment: int = 20,
        thickness: float = DEFAULT_THICKNESS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a fixed-thickness extrusion of the spline as world vertices/faces.

        Vertex layout: [front outline (N), back outline (N)]. The caps are
        earcut-triangulated so the filled region is exactly the area enclosed
        by the Bezier outline, including non-convex shapes.
        """
        local = self.sample_spline_local(n_per_segment)
        N = len(local)
        half = thickness * 0.5
        offset = torch.tensor([0.0, 0.0, half], dtype=local.dtype, device=local.device)

        front = self.local_to_world(local + offset)
        back = self.local_to_world(local - offset)
        verts = torch.cat([front, back], dim=0)

        with torch.no_grad():
            cap = _triangulate_outline_xy(local[:, :2].detach().cpu().numpy())

        faces: list[list[int]] = []
        for a, b, c in cap.tolist():
            faces.append([a, b, c])                # front cap
            faces.append([a + N, c + N, b + N])    # back cap (reversed winding)
        for i in range(N):
            j = (i + 1) % N
            faces.append([i, N + i, N + j])
            faces.append([i, N + j, j])

        return verts, torch.tensor(faces, dtype=torch.int32, device=self.center.device)

    def triangle_faces(self, n_per_segment: int = 20) -> torch.Tensor:
        """(F, 3) int32 triangle indices filling the sampled outline.

        Indices reference the V = N_CONTROL_POINTS * n_per_segment spline
        sample points directly. Earcut-triangulated, so non-convex outlines
        are filled correctly.
        """
        with torch.no_grad():
            xy = self.sample_spline_local(n_per_segment)[:, :2].detach().cpu().numpy()
        faces = _triangulate_outline_xy(xy)
        return torch.from_numpy(faces).to(device=self.center.device)

    # ------------------------------------------------------------------
    # Viewport bridge — detached numpy, safe across threads
    # ------------------------------------------------------------------

    def to_mesh(self, n_per_segment: int = 20) -> "Mesh":
        """Extruded scene.Mesh for the 3D viewport."""
        from scene.scene import Mesh

        with torch.no_grad():
            verts, faces = self.extruded_mesh_world(n_per_segment)
            vertices = verts.cpu().numpy()
            faces_np = faces.cpu().numpy()

        rgb   = self.albedo.detach().cpu().clamp(0.0, 1.0).numpy()
        color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        return Mesh(vertices, faces_np, color=color, label=self.label)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def clamp_albedo(self) -> None:
        with torch.no_grad():
            self.albedo.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def _f(t: torch.Tensor) -> float:
            return float(t.detach().cpu())
        def _l(t: torch.Tensor) -> list:
            return t.detach().cpu().numpy().tolist()

        return {
            "label":  self.label,
            "center": _l(self.center),
            "theta":  _f(self.theta),
            "albedo": _l(self.albedo),
            "creation_step": int(getattr(self, "creation_step", 0)),
            "self_intersect_counter": int(getattr(self, "self_intersect_counter", 0)),
            "control_points": [
                {
                    "x":               _f(cast(ControlPoint, cp).x),
                    "y":               _f(cast(ControlPoint, cp).y),
                    "z":               _f(cast(ControlPoint, cp).z),
                    "handle_scale":    _f(cast(ControlPoint, cp).handle_scale),
                    "handle_rotation": _f(cast(ControlPoint, cp).handle_rotation),
                }
                for cp in self.control_points
            ],
        }

    @classmethod
    def from_dict(cls, d: dict, device: str = "cpu") -> "Patch":
        cps = [
            ControlPoint(
                x=cp["x"], y=cp["y"], z=cp["z"],
                handle_scale=cp["handle_scale"],
                handle_rotation=cp["handle_rotation"],
                device=device,
            )
            for cp in d["control_points"]
        ]
        patch = cls(
            control_points=cps,
            center=d["center"],
            theta=float(d["theta"]),
            albedo=d["albedo"],
            device=device,
            label=d.get("label", ""),
        )
        patch.creation_step = int(d.get("creation_step", 0))
        patch.self_intersect_counter = int(d.get("self_intersect_counter", 0))
        return patch

    def __repr__(self) -> str:
        c = self.center.detach().cpu().numpy().round(3).tolist()
        return f"Patch(label={self.label!r}, center={c}, theta={float(self.theta.detach()):.3f} rad)"
