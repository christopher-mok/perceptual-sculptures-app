"""Main application window: three-panel layout."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QSplitter

from scene.camera import Camera
from scene.scene import Mesh, Scene
from ui.controls_panel import ControlsPanel
from ui.image_panel import ImagePanel
from ui.viewport import Viewport

_TARGET_TRANSPARENT_BORDER_FRACTION = 0.10
_ORIGINAL_BOUNDS_SIZE = 5.0
_HANGING_PLANE_Y = (_ORIGINAL_BOUNDS_SIZE * 0.5) + 1.0
_HANGING_PLANE_FRAME_THICKNESS = 0.035
_SCENE_CAMERA_FOV_DEG = 30.22  # 50mm equivalent on a 36x27mm 4:3 sensor.
_STRING_DASH_LENGTH = 0.08
_STRING_GAP_LENGTH = 0.05
_MANUAL_PIECE_RADIUS = 0.25


def _make_scene_cameras() -> list[Camera]:
    """Create two scene cameras 90° apart, each aimed straight along its axis.

    Camera 1 sits on the +Z axis looking toward the origin (along −Z).
    Camera 2 sits on the +X axis looking toward the origin (along −X).
    """
    radius = 8.0
    origin = np.zeros(3, dtype=np.float32)

    cam1 = Camera(
        position=np.array([0.0, 0.0, radius], dtype=np.float32),
        target=origin,
        fov=_SCENE_CAMERA_FOV_DEG,
        aspect=4.0 / 3.0,
        near=0.35,
        far=18.0,
        color=(1.0, 0.85, 0.0),   # gold — View 1 along Z
        label="View 1",
    )
    cam2 = Camera(
        position=np.array([radius, 0.0, 0.0], dtype=np.float32),
        target=origin,
        fov=_SCENE_CAMERA_FOV_DEG,
        aspect=4.0 / 3.0,
        near=0.35,
        far=18.0,
        color=(0.2, 0.8, 1.0),   # cyan — View 2 along X
        label="View 2",
    )
    return [cam1, cam2]


def _load_target_image_with_border(path: str) -> np.ndarray:
    """Load an image as RGBA and add a transparent border around it."""
    from PIL import Image, ImageOps

    image = Image.open(path).convert("RGBA")
    border = max(1, int(round(max(image.size) * _TARGET_TRANSPARENT_BORDER_FRACTION)))
    return np.array(ImageOps.expand(image, border=border, fill=(0, 0, 0, 0)))


def _make_hanging_plane_mesh(size: float) -> Mesh:
    """Create a thin square frame marking the hanging plane footprint."""
    half = max(size * 0.5, _HANGING_PLANE_FRAME_THICKNESS)
    t = min(_HANGING_PLANE_FRAME_THICKNESS, half)
    y = _HANGING_PLANE_Y

    bars = [
        (-half, half, -half, -half + t),
        (-half, half, half - t, half),
        (-half, -half + t, -half, half),
        (half - t, half, -half, half),
    ]
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for x0, x1, z0, z1 in bars:
        start = len(vertices)
        vertices.extend([
            [x0, y, z0],
            [x1, y, z0],
            [x1, y, z1],
            [x0, y, z1],
        ])
        faces.extend([
            [start, start + 1, start + 2],
            [start, start + 2, start + 3],
        ])

    return Mesh(
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int32),
        color=(0.55, 0.62, 0.68),
        label="hanging_plane",
    )


def _dotted_line_segments(start: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-6:
        return []
    direction = delta / length
    period = _STRING_DASH_LENGTH + _STRING_GAP_LENGTH
    segments: list[np.ndarray] = []
    cursor = 0.0
    while cursor < length:
        dash_end = min(cursor + _STRING_DASH_LENGTH, length)
        segments.append(start + direction * cursor)
        segments.append(start + direction * dash_end)
        cursor += period
    return segments


class MainWindow(QMainWindow):
    """Top-level application window.

    Layout
    ------
    [ImagePanel | Viewport | ControlsPanel]
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Perspective Sculptor")
        self.resize(1400, 820)
        self.setMinimumSize(900, 600)

        # Dark palette for the whole window
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1a1a1b;
                color: #ddd;
            }
            QSplitter::handle {
                background: #333;
                width: 3px;
            }
            QScrollBar:vertical {
                background: #1a1a1b;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 4px;
                min-height: 20px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3a3a3a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -5px 0;
                background: #4a9eff;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #2a6496;
                border-radius: 2px;
            }
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ddd;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                selection-background-color: #2a6496;
                color: #ddd;
                border: 1px solid #444;
            }
            """
        )

        # Build scene
        self._scene = Scene()
        for cam in _make_scene_cameras():
            self._scene.add_camera(cam)

        # Patch state
        self._patches: list = []
        self._target1_img: np.ndarray | None = None  # uint8 (H,W,4), padded RGBA
        self._target2_img: np.ndarray | None = None
        self._worker = None
        self._worker_paused = False
        self._swept_volume = None
        self._show_swept_volume = False
        self._optimization_run_until_convergence = False
        self._reset_after_worker_stops = False

        # Build panels
        self._image_panel   = ImagePanel(self)
        self._viewport      = Viewport(self._scene, self)
        self._controls      = ControlsPanel(self)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._image_panel)
        splitter.addWidget(self._viewport)
        splitter.addWidget(self._controls)
        splitter.setStretchFactor(0, 0)   # image panel: fixed
        splitter.setStretchFactor(1, 1)   # viewport: expands
        splitter.setStretchFactor(2, 0)   # controls: fixed
        splitter.setSizes([270, 860, 270])
        splitter.setChildrenCollapsible(False)

        self.setCentralWidget(splitter)

        # Wire up signals
        self._image_panel.view1_loaded.connect(self._on_view1_loaded)
        self._image_panel.view2_loaded.connect(self._on_view2_loaded)
        self._controls.patches.initialize_requested.connect(self._on_initialize)
        self._controls.optimization.run_requested.connect(self._on_run_optimization)
        self._controls.optimization.pause_toggled.connect(self._on_pause_optimization)
        self._controls.optimization.palette_changed.connect(self._on_palette_changed)
        self._controls.optimization.reset_requested.connect(self._on_reset)
        self._viewport.piece_clicked.connect(self._on_viewport_piece_clicked)
        self._controls.edit.edit_mode_toggled.connect(self._on_edit_mode_toggled)
        self._controls.edit.piece_selected.connect(self._on_edit_piece_selected)
        self._controls.edit.nudge_requested.connect(self._on_edit_nudge)
        self._controls.edit.rotate_requested.connect(self._on_edit_rotate)
        self._controls.edit.scale_requested.connect(self._on_edit_scale)
        self._controls.edit.add_piece_requested.connect(self._on_edit_add_piece)
        self._controls.edit.delete_requested.connect(self._on_edit_delete)
        self._controls.export.export_requested.connect(self._on_export_json)
        self._controls.export.import_requested.connect(self._on_import_json)
        self._controls.export.strings_requested.connect(self._on_add_strings)
        self._controls.export.swept_volume_requested.connect(self._on_toggle_swept_volume)
        self._controls.export.split_test_requested.connect(self._on_visualize_split_test)
        self._controls.patches.hanging_plane_size_changed.connect(
            self._on_hanging_plane_size_changed
        )
        self._controls.patches.swept_volume_resolution_changed.connect(
            self._on_swept_volume_settings_changed
        )
        self._update_hanging_plane_mesh()
        self._sync_export_enabled()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_view1_loaded(self, path: str) -> None:
        new_image = _load_target_image_with_border(path)
        # Keep the cached swept volume when the target content is unchanged
        # (e.g. re-selecting the same image while iterating on a design).
        if self._target1_img is None or not np.array_equal(new_image, self._target1_img):
            self._clear_swept_volume_cache()
        self._target1_img = new_image
        self._update_hanging_plane_mesh()
        print(f"[View 1 target] loaded: {path}")

    def _on_view2_loaded(self, path: str) -> None:
        new_image = _load_target_image_with_border(path)
        if self._target2_img is None or not np.array_equal(new_image, self._target2_img):
            self._clear_swept_volume_cache()
        self._target2_img = new_image
        self._update_hanging_plane_mesh()
        print(f"[View 2 target] loaded: {path}")

    def _on_initialize(self, n_patches: int) -> None:
        from core.initialization import initialize_patches

        device = self._controls.patches.device

        try:
            swept_volume = self._ensure_swept_volume()
            self._patches = initialize_patches(
                n_patches=n_patches,
                cameras=self._scene.cameras,
                device=device,
                swept_volume=swept_volume,
            )
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette(self._patches, self._controls.optimization.palette)
            self._constrain_patches_to_hanging_plane()
        except (ValueError, FileNotFoundError, ImportError) as exc:
            QMessageBox.warning(self, "Initialize patches", str(exc))
            return
        except RuntimeError as exc:
            # Catches e.g. MPS/CUDA not available on this machine
            QMessageBox.warning(self, "Device error", str(exc))
            return

        self._viewport.set_patches(self._patches)
        self._update_string_lines()
        self._update_camera_previews_from_patches()
        self._controls.srd.set_stats({"patches": len(self._patches)})
        self._sync_export_enabled()
        print(f"[Initialize patches] {len(self._patches)} patches ({device})")

    def _on_run_optimization(self) -> None:
        if not self._patches:
            QMessageBox.warning(self, "Run optimization", "Initialize patches first.")
            return
        if self._target1_img is None:
            QMessageBox.warning(self, "Run optimization", "Load a View 1 target image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        from ui.worker import OptimizationWorker

        opt = self._controls.optimization

        try:
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette(self._patches, opt.palette)
        except ValueError as exc:
            QMessageBox.warning(self, "Run optimization", str(exc))
            return
        self._constrain_patches_to_hanging_plane()
        self._clear_string_connections()
        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()

        if self._swept_volume is None and self._target2_img is not None:
            try:
                self._ensure_swept_volume()
            except ValueError as exc:
                print(f"[Swept volume] unavailable, continuing without it: {exc}")

        self._optimization_run_until_convergence = opt.run_until_convergence
        self._worker = OptimizationWorker(
            patches=self._patches,
            cameras=self._scene.cameras,
            target1=self._target1_img,
            target2=self._target2_img,
            palette=opt.palette,
            lr=opt.learning_rate,
            n_steps=opt.n_steps,
            run_until_convergence=opt.run_until_convergence,
            convergence_threshold=opt.convergence_threshold,
            device=self._controls.patches.device,
            hanging_plane_size=self._controls.patches.hanging_plane_size,
            hanging_plane_y=_HANGING_PLANE_Y,
            srd_config=self._controls.srd.config,
            swept_volume=self._swept_volume,
            parent=self,
        )
        self._worker.step_completed.connect(self._on_optimization_step)
        self._worker.failed.connect(self._on_optimization_failed)
        self._worker.optimization_finished.connect(self._on_optimization_finished)
        self._worker.paused_state_changed.connect(self._on_worker_paused_changed)

        self._controls.optimization.set_running(True)
        self._controls.patches.set_running(True)
        self._controls.srd.set_running(True)
        self._controls.optimization.reset_progress()
        self._sync_export_enabled()
        self._worker.start()
        if opt.run_until_convergence:
            print(
                f"[Optimization] started: until loss <= {opt.convergence_threshold:.3e}, "
                f"lr={opt.learning_rate:.3e}, palette={opt.palette!r}, targets=mask"
            )
        else:
            print(
                f"[Optimization] started: steps={opt.n_steps}, lr={opt.learning_rate:.3e}, "
                f"palette={opt.palette!r}, targets=mask"
            )

    def _on_optimization_step(self, step_idx: int, metrics: object, meshes: object) -> None:
        self._viewport.set_meshes(meshes)
        self._image_panel.set_camera_previews(meshes, self._scene.cameras)
        if isinstance(metrics, dict):
            self._controls.srd.set_stats(metrics)
        if not self._optimization_run_until_convergence:
            self._controls.optimization.set_progress(step_idx)
        loss = metrics.get("loss", 0.0) if isinstance(metrics, dict) else 0.0
        if isinstance(metrics, dict):
            print(
                f"[Optimization] step={step_idx}, loss={loss:.6f}, "
                f"terms(avg): view1={metrics.get('view1_mse', 0.0):.6f}, "
                f"view2={metrics.get('view2_loss', 0.0):.6f}, "
                f"view1_sil={metrics.get('view1_silhouette', 0.0):.6f}, "
                f"view2_sil={metrics.get('view2_silhouette', 0.0):.6f}, "
                f"view1_neg={metrics.get('view1_negative_space', 0.0):.6f}, "
                f"view2_neg={metrics.get('view2_negative_space', 0.0):.6f}, "
                f"overlap={metrics.get('overlap', 0.0):.6f}; "
                f"smallest_area={metrics.get('smallest_patch_area', 0.0):.6f}, "
                f"tiny_deleted={metrics.get('tiny_patches_deleted', 0.0):.0f}; "
                f"weighted: negative_space={metrics.get('negative_space_weighted', 0.0):.6f}, "
                f"overlap={metrics.get('overlap_weighted', 0.0):.6f}; "
                f"srd: patches={metrics.get('srd_active_patches', metrics.get('patches', 0.0)):.0f}, "
                f"accepted={metrics.get('srd_accepted', 0.0):.0f}"
            )
        else:
            print(f"[Optimization] step={step_idx}, loss={loss:.6f}")

    def _on_pause_optimization(self, paused: bool) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.set_paused(paused)
        print("[Optimization] paused" if paused else "[Optimization] resumed")

    def _on_worker_paused_changed(self, paused: bool) -> None:
        """Worker confirmed it entered/left the pause wait loop.

        Piece editing unlocks only once the step loop is genuinely idle, and
        locks again the moment optimization resumes.
        """
        self._worker_paused = paused
        self._sync_edit_controls()
        if paused:
            print("[Edit] optimization idle — piece editing unlocked")

    def _piece_editing_blocked(self) -> bool:
        """True while pieces must not be mutated from the UI."""
        if self._worker is None or not self._worker.isRunning():
            return False
        return not self._worker_paused

    def _notify_worker_patches_modified(self) -> None:
        """Tell a paused worker that the piece list changed shape."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.notify_patches_modified()

    def _on_palette_changed(self) -> None:
        if not self._patches:
            return
        try:
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette(self._patches, self._controls.optimization.palette)
        except ValueError as exc:
            QMessageBox.warning(self, "Palette", str(exc))
            return
        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()

    def _on_edit_mode_toggled(self, enabled: bool) -> None:
        if enabled and (self._piece_editing_blocked() or not self._patches):
            self._controls.edit.set_edit_mode(False)
            return
        self._viewport.set_edit_selection(
            enabled,
            self._controls.edit.selected_piece_index,
        )
        print("[Edit] mode enabled" if enabled else "[Edit] mode disabled")

    def _on_edit_piece_selected(self, index: int) -> None:
        self._viewport.set_edit_selection(
            self._controls.edit.edit_mode_enabled,
            index,
        )

    def _on_viewport_piece_clicked(self, index: int) -> None:
        if not self._controls.edit.edit_mode_enabled:
            return
        self._controls.edit.set_selected_piece(index)
        print(f"[Edit] selected patch={index + 1} from viewport")

    def _on_edit_nudge(self, dx: float, dy: float, dz: float) -> None:
        patch = self._selected_patch_for_edit()
        if patch is None:
            return
        import torch

        with torch.no_grad():
            delta = torch.tensor(
                [dx, dy, dz],
                dtype=patch.center.dtype,
                device=patch.center.device,
            )
            patch.center.add_(delta)
        self._constrain_patches_to_hanging_plane()
        self._clear_string_connections()
        self._viewport.set_patches(self._patches)
        self._viewport.set_edit_selection(True, self._controls.edit.selected_piece_index)
        self._update_camera_previews_from_patches()
        print(
            f"[Edit] nudged patch={self._controls.edit.selected_piece_index + 1}, "
            f"delta=({dx:.3f}, {dy:.3f}, {dz:.3f})"
        )

    def _on_edit_rotate(self, delta_degrees: float) -> None:
        patch = self._selected_patch_for_edit()
        if patch is None:
            return
        import torch

        with torch.no_grad():
            patch.theta.add_(patch.theta.new_tensor(np.deg2rad(delta_degrees)))
        self._clear_string_connections()
        self._viewport.set_patches(self._patches)
        self._viewport.set_edit_selection(True, self._controls.edit.selected_piece_index)
        self._update_camera_previews_from_patches()
        print(
            f"[Edit] rotated patch={self._controls.edit.selected_piece_index + 1}, "
            f"delta={delta_degrees:.1f}°"
        )

    def _on_edit_scale(self, factor: float) -> None:
        patch = self._selected_patch_for_edit()
        if patch is None or factor <= 0.0:
            return
        import torch

        with torch.no_grad():
            for cp in patch.control_points:
                cp.x.mul_(factor)
                cp.y.mul_(factor)
                cp.handle_scale.mul_(factor)
                cp.handle_scale.clamp_(0.01, 2.0)
        self._constrain_patches_to_hanging_plane()
        self._clear_string_connections()
        self._viewport.set_patches(self._patches)
        self._viewport.set_edit_selection(True, self._controls.edit.selected_piece_index)
        self._update_camera_previews_from_patches()
        print(
            f"[Edit] scaled patch={self._controls.edit.selected_piece_index + 1}, "
            f"factor={factor:.3f}"
        )

    def _on_edit_add_piece(self) -> None:
        if not self._controls.edit.edit_mode_enabled:
            return
        if self._piece_editing_blocked():
            return
        from optimizer.srd import _small_default_patch

        device = (
            str(self._patches[0].center.device)
            if self._patches else self._controls.patches.device
        )
        patch = _small_default_patch(
            np.zeros(3, dtype=np.float32),
            device,
            [1.0, 1.0, 1.0],
            creation_step=0,
            label=f"manual_{len(self._patches) + 1:04d}",
            radius=_MANUAL_PIECE_RADIUS,
        )
        try:
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette([patch], self._controls.optimization.palette)
        except ValueError:
            pass
        self._patches.append(patch)
        self._notify_worker_patches_modified()
        self._clear_string_connections()
        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()
        self._controls.srd.set_stats({"patches": len(self._patches)})
        self._sync_export_enabled()
        self._controls.edit.set_selected_piece(len(self._patches) - 1)
        print(
            f"[Edit] added piece at origin, label={patch.label!r}; "
            f"total={len(self._patches)}"
        )

    def _on_edit_delete(self) -> None:
        if not self._controls.edit.edit_mode_enabled:
            return
        if self._piece_editing_blocked():
            return
        index = self._controls.edit.selected_piece_index
        if index < 0 or index >= len(self._patches):
            return
        if (
            self._worker is not None
            and self._worker.isRunning()
            and len(self._patches) <= 1
        ):
            QMessageBox.warning(
                self,
                "Delete piece",
                "Keep at least one piece while optimization is paused.",
            )
            return
        removed = self._patches.pop(index)
        self._notify_worker_patches_modified()
        self._clear_string_connections()
        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()
        self._controls.srd.set_stats({"patches": len(self._patches)})
        self._sync_export_enabled()
        print(
            f"[Edit] deleted patch={index + 1}, label={removed.label!r}; "
            f"remaining={len(self._patches)}"
        )

    def _on_hanging_plane_size_changed(self, size: float) -> None:
        self._clear_swept_volume_cache()
        self._update_hanging_plane_mesh()
        self._clear_string_connections()
        self._constrain_patches_to_hanging_plane()
        if self._patches:
            self._viewport.set_patches(self._patches)
            self._update_string_lines()
            self._update_camera_previews_from_patches()
        print(f"[Hanging plane] size={size:.1f}, y={_HANGING_PLANE_Y:.1f}")

    def _on_swept_volume_settings_changed(self) -> None:
        self._clear_swept_volume_cache()
        self._update_hanging_plane_mesh()
        print(
            "[Swept volume] resolution="
            f"{self._controls.patches.swept_volume_resolution}"
        )

    def _on_reset(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._reset_after_worker_stops = True
            self._worker.request_stop()
            print("[Reset] stopping optimization before clearing state")
            return

        self._reset_state()

    def _on_export_json(self) -> None:
        if not self._patches:
            QMessageBox.warning(self, "Export", "No patches to export.")
            return

        try:
            from core.export import export_patches_to_json, send_export_json_file

            output_path = export_patches_to_json(
                self._patches,
                hanging_plane_size=self._controls.patches.hanging_plane_size,
            )
            post_result = send_export_json_file()
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            print(f"[Export] failed: {exc}")
            return

        if post_result.get("ok", False):
            QMessageBox.information(
                self,
                "Export complete",
                (
                    f"Saved piece data to {output_path}\n"
                    f"Posted JSON to API successfully (status={post_result.get('status')})."
                ),
            )
            print(
                f"[Export] wrote JSON to {output_path}; "
                f"posted to API status={post_result.get('status')}"
            )
        else:
            QMessageBox.warning(
                self,
                "Export complete (API post failed)",
                (
                    f"Saved piece data to {output_path}\n"
                    f"API post failed: status={post_result.get('status')}, "
                    f"body={post_result.get('body')}"
                ),
            )
            print(
                f"[Export] wrote JSON to {output_path}; "
                f"API post failed status={post_result.get('status')}, "
                f"body={post_result.get('body')}"
            )

    def _on_add_strings(self) -> None:
        if not self._patches:
            QMessageBox.warning(self, "Add strings", "Initialize or import patches first.")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Add strings", "Stop optimization before adding strings.")
            return

        try:
            from core.export import add_strings_to_patches

            count = add_strings_to_patches(
                self._patches,
                hanging_plane_y=_HANGING_PLANE_Y,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Add strings failed", str(exc))
            print(f"[Strings] failed: {exc}")
            return

        QMessageBox.information(
            self,
            "Strings added",
            f"Added {count} string connections for {len(self._patches)} pieces.",
        )
        self._update_string_lines()
        print(
            f"[Strings] added {count} connections for {len(self._patches)} pieces; "
            f"hanging_plane_y={_HANGING_PLANE_Y:.3f}"
        )

    def _on_import_json(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Import", "Stop optimization before importing a design.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import pieces JSON",
            "exports",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return

        try:
            from core.export import import_patches_from_json

            self._patches = import_patches_from_json(
                path,
                device=self._controls.patches.device,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            print(f"[Import] failed: {exc}")
            return

        # Importing only replaces the pieces; the cached swept volume stays
        # valid (it depends on the target images, not the pieces) so users
        # can tweak a JSON, re-import, and continue without a rebuild.
        self._viewport.set_patches(self._patches)
        self._update_string_lines()
        self._update_camera_previews_from_patches()
        self._controls.srd.set_stats({"patches": len(self._patches)})
        self._sync_export_enabled()
        QMessageBox.information(
            self,
            "Import complete",
            f"Loaded {len(self._patches)} patches from {path}",
        )
        cache_note = (
            "swept volume cache preserved"
            if self._swept_volume is not None else "no swept volume cached"
        )
        print(f"[Import] loaded {len(self._patches)} patches from {path} ({cache_note})")

    def _reset_state(self) -> None:
        self._worker = None
        self._worker_paused = False
        self._optimization_run_until_convergence = False
        self._reset_after_worker_stops = False
        self._patches = []
        self._target1_img = None
        self._target2_img = None
        self._swept_volume = None
        self._show_swept_volume = False
        self._image_panel.reset()
        self._viewport.reset()
        self._update_hanging_plane_mesh()
        self._controls.optimization.reset_controls()
        self._controls.patches.set_running(False)
        self._controls.srd.set_running(False)
        self._controls.srd.set_stats({})
        self._sync_export_enabled()
        print("[Reset] cleared targets, patches, optimization state, and viewport")

    def _update_camera_previews_from_patches(self) -> None:
        meshes = [patch.to_mesh() for patch in self._patches]
        self._image_panel.set_camera_previews(meshes, self._scene.cameras)

    def _clear_swept_volume_cache(self) -> None:
        self._swept_volume = None
        self._show_swept_volume = False

    def _ensure_swept_volume(self):
        if self._swept_volume is not None:
            return self._swept_volume
        if self._target1_img is None or self._target2_img is None:
            raise ValueError("Load both target images before initializing swept-volume pieces.")

        from core.swept_volume import SweptVolume

        def _progress(done: int, total: int) -> None:
            if done == total or done % 12 == 0:
                print(f"[Swept volume] building {done}/{total} slices")

        self._swept_volume = SweptVolume.from_images(
            self._target1_img,
            self._target2_img,
            self._scene.cameras,
            hanging_plane_size=self._controls.patches.hanging_plane_size,
            resolution=self._controls.patches.swept_volume_resolution,
            progress_callback=_progress,
        )
        print(f"[Swept volume] precomputed {len(self._swept_volume.points)} occupied samples")
        self._update_hanging_plane_mesh()
        return self._swept_volume

    def _on_toggle_swept_volume(self) -> None:
        try:
            self._ensure_swept_volume()
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Swept volume", str(exc))
            return
        self._show_swept_volume = not self._show_swept_volume
        self._update_hanging_plane_mesh()
        state = "shown" if self._show_swept_volume else "hidden"
        print(f"[Swept volume] debug mesh {state}")

    def _on_visualize_split_test(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(
                self,
                "Split test",
                "Stop optimization before changing the scene.",
            )
            return

        try:
            parent = self._make_split_test_patch()
            child_a, child_b = parent.split_down_middle(creation_step=0)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Split test", str(exc))
            return
        self._patches = [child_a, child_b]
        self._clear_string_connections()
        self._constrain_patches_to_hanging_plane()
        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()
        self._controls.srd.set_stats({"patches": len(self._patches)})
        self._sync_export_enabled()
        print(
            "[Split test] spawned one large centered patch and visualized "
            f"split result as {len(self._patches)} child pieces"
        )

    def _make_split_test_patch(self):
        from core.patch import ControlPoint, Patch

        device = self._controls.patches.device
        radius = max(0.35, self._controls.patches.hanging_plane_size * 0.18)
        handle_scale = radius * (4.0 / 3.0) * math.tan(math.pi / Patch.N_CONTROL_POINTS)
        control_points: list[ControlPoint] = []
        for idx in range(Patch.N_CONTROL_POINTS):
            angle = 2.0 * math.pi * idx / Patch.N_CONTROL_POINTS - math.pi / 2.0
            control_points.append(ControlPoint(
                x=radius * math.cos(angle),
                y=radius * math.sin(angle),
                z=0.0,
                handle_scale=handle_scale,
                handle_rotation=angle + math.pi / 2.0,
                device=device,
            ))
        return Patch(
            control_points=control_points,
            center=[0.0, 0.0, 0.0],
            theta=0.0,
            albedo=[1.0, 1.0, 1.0],
            device=device,
            label="split_test_parent",
        )

    def _update_hanging_plane_mesh(self) -> None:
        meshes = [_make_hanging_plane_mesh(self._controls.patches.hanging_plane_size)]
        if self._show_swept_volume and self._swept_volume is not None:
            meshes.append(self._swept_volume.to_mesh())
        self._viewport.set_static_meshes(meshes)

    def _clear_string_connections(self) -> None:
        for patch in self._patches:
            if hasattr(patch, "string_connections"):
                delattr(patch, "string_connections")
        self._update_string_lines()

    def _update_string_lines(self) -> None:
        segments: list[np.ndarray] = []
        for patch in self._patches:
            for connection in getattr(patch, "string_connections", []):
                piece_point = connection.get("pieceWorldPoint")
                board_point = connection.get("boardPoint")
                if not isinstance(piece_point, dict) or not isinstance(board_point, dict):
                    continue
                start = np.array(
                    [piece_point["x"], piece_point["y"], piece_point["z"]],
                    dtype=np.float32,
                )
                end = np.array(
                    [board_point["x"], board_point["y"], board_point["z"]],
                    dtype=np.float32,
                )
                segments.extend(_dotted_line_segments(start, end))
        if segments:
            vertices = np.array(segments, dtype=np.float32)
        else:
            vertices = np.empty((0, 3), dtype=np.float32)
        self._viewport.set_string_segments(vertices)

    def _constrain_patches_to_hanging_plane(self) -> None:
        if not self._patches:
            return
        import torch
        from core.optimizer import constrain_patch_to_square_xz_bounds

        half = max(self._controls.patches.hanging_plane_size * 0.5, 1e-4)
        with torch.no_grad():
            for patch in self._patches:
                constrain_patch_to_square_xz_bounds(patch, half)

    def _on_optimization_failed(self, message: str) -> None:
        self._worker_paused = False
        if self._reset_after_worker_stops:
            self._reset_state()
            return
        self._controls.optimization.set_running(False)
        self._controls.patches.set_running(False)
        self._controls.srd.set_running(False)
        self._sync_export_enabled()
        QMessageBox.warning(self, "Optimization failed", message)
        print(f"[Optimization] failed: {message}")

    def _on_optimization_finished(self, metrics: object) -> None:
        self._worker_paused = False
        if self._reset_after_worker_stops:
            self._reset_state()
            return
        self._controls.optimization.set_running(False)
        self._controls.patches.set_running(False)
        self._controls.srd.set_running(False)
        self._sync_export_enabled()
        loss = metrics.get("loss", None) if isinstance(metrics, dict) else None
        if loss is None:
            print("[Optimization] finished")
        else:
            print(f"[Optimization] finished: loss={loss:.6f}")

    def _sync_export_enabled(self) -> None:
        optimizing = self._worker is not None and self._worker.isRunning()
        self._controls.export.set_enabled(
            bool(self._patches) and not optimizing,
            debug_enabled=not optimizing,
        )
        self._sync_edit_controls()

    def _sync_edit_controls(self) -> None:
        labels = self._build_piece_labels()
        self._controls.edit.set_running(self._piece_editing_blocked())
        self._controls.edit.set_piece_labels(labels)
        self._viewport.set_edit_selection(
            self._controls.edit.edit_mode_enabled,
            self._controls.edit.selected_piece_index,
        )

    def _build_piece_labels(self) -> list[str]:
        labels: list[str] = []
        for idx, patch in enumerate(self._patches, start=1):
            suffix = f" ({patch.label})" if getattr(patch, "label", "").strip() else ""
            labels.append(f"Piece {idx}{suffix}")
        return labels

    def _selected_patch_for_edit(self):
        if not self._controls.edit.edit_mode_enabled:
            return None
        if self._piece_editing_blocked():
            return None
        index = self._controls.edit.selected_piece_index
        if index < 0 or index >= len(self._patches):
            return None
        return self._patches[index]

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(1500)
        super().closeEvent(event)
