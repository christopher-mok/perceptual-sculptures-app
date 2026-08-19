"""Qt worker for running differentiable optimization off the main thread."""

from __future__ import annotations

import traceback
import time

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.optimizer import SceneOptimizer


class OptimizationWorker(QThread):
    """Run SceneOptimizer in a thread and emit viewport-safe mesh snapshots."""

    step_completed = pyqtSignal(int, object, object)  # step, metrics, meshes
    failed = pyqtSignal(str)
    optimization_finished = pyqtSignal(object)
    paused_state_changed = pyqtSignal(bool)  # True once the step loop is idle

    def __init__(
        self,
        *,
        patches: list,
        cameras: list,
        target1: object,
        target2: object | None,
        palette: object,
        lr: float,
        n_steps: int,
        run_until_convergence: bool,
        convergence_threshold: float,
        device: str,
        hanging_plane_size: float,
        hanging_plane_y: float,
        srd_config: dict[str, object] | None,
        swept_volume: object | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._patches = patches
        self._cameras = cameras
        self._target1 = target1
        self._target2 = target2
        self._palette = palette
        self._lr = lr
        self._n_steps = n_steps
        self._run_until_convergence = run_until_convergence
        self._convergence_threshold = convergence_threshold
        self._device = device
        self._hanging_plane_size = hanging_plane_size
        self._hanging_plane_y = hanging_plane_y
        self._srd_config = srd_config
        self._swept_volume = swept_volume
        self._stop_requested = False
        self._pause_requested = False
        self._patches_modified = False

    def request_stop(self) -> None:
        self._stop_requested = True
        self._pause_requested = False

    def set_paused(self, paused: bool) -> None:
        self._pause_requested = paused

    def notify_patches_modified(self) -> None:
        """Mark that pieces were added/removed while paused.

        The optimizer's parameter groups are rebuilt before the next step so
        edits made in the UI carry through to the running optimization.
        """
        self._patches_modified = True

    def _wait_if_paused(self) -> None:
        if not self._pause_requested or self._stop_requested:
            return
        # Signal only once the current step has finished, so the UI enables
        # piece editing only while the optimizer is genuinely idle.
        self.paused_state_changed.emit(True)
        while self._pause_requested and not self._stop_requested:
            time.sleep(0.05)
        self.paused_state_changed.emit(False)

    def _sync_external_edits(self, optimizer: SceneOptimizer) -> None:
        """Fold in piece list edits made from the UI while paused."""
        if not self._patches_modified:
            return
        self._patches_modified = False
        optimizer.rebuild_optim()

    def run(self) -> None:
        try:
            optimizer = SceneOptimizer(
                self._patches,
                self._cameras[0],
                self._cameras[1],
                self._target1,
                self._target2,
                palette=self._palette,
                lr=self._lr,
                device=self._device,
                hanging_plane_size=self._hanging_plane_size,
                hanging_plane_y=self._hanging_plane_y,
                srd_config=self._srd_config,
                swept_volume=self._swept_volume,
            )

            last_metrics: dict[str, float] = {}
            if self._run_until_convergence:
                step_idx = 0
                while not self._stop_requested:
                    self._wait_if_paused()
                    if self._stop_requested:
                        break
                    self._sync_external_edits(optimizer)
                    step_idx += 1
                    last_metrics = optimizer.step(step_idx, self._n_steps)
                    self.step_completed.emit(
                        step_idx,
                        last_metrics,
                        optimizer.mesh_snapshot(),
                    )
                    loss = last_metrics.get("loss", float("inf"))
                    if loss <= self._convergence_threshold:
                        break
            else:
                for step_idx in range(1, self._n_steps + 1):
                    self._wait_if_paused()
                    if self._stop_requested:
                        break
                    self._sync_external_edits(optimizer)
                    last_metrics = optimizer.step(step_idx, self._n_steps)
                    self.step_completed.emit(
                        step_idx,
                        last_metrics,
                        optimizer.mesh_snapshot(),
                    )

            self.optimization_finished.emit(last_metrics)
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(details)
