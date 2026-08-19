"""Right panel: Patches, Optimization, and Export controls."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECTION_STYLE = """
QGroupBox {
    color: #ccc;
    font-weight: bold;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
}
"""

_LABEL_STYLE = "color: #aaa; font-size: 12px;"
_VALUE_STYLE = "color: #ddd; font-size: 12px; min-width: 52px;"


def _row(label_text: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setStyleSheet(_LABEL_STYLE)
    row.addWidget(lbl)
    row.addWidget(widget)
    return row


def _labeled_slider(
    minimum: int,
    maximum: int,
    default: int,
    fmt: str = "{}",
) -> tuple[QSlider, QLabel]:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(default)

    value_lbl = QLabel(fmt.format(default))
    value_lbl.setStyleSheet(_VALUE_STYLE)
    value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    return slider, value_lbl


# ---------------------------------------------------------------------------
# Patches section
# ---------------------------------------------------------------------------


class PatchesSection(QGroupBox):
    initialize_requested = pyqtSignal(int)   # n_patches
    hanging_plane_size_changed = pyqtSignal(float)
    swept_volume_resolution_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Patches", parent)
        self.setStyleSheet(_SECTION_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        # Number of patches slider
        self._n_slider, self._n_lbl = _labeled_slider(4, 200, 20, "{}")
        self._n_slider.valueChanged.connect(
            lambda v: self._n_lbl.setText(str(v))
        )

        n_row = QHBoxLayout()
        n_lbl = QLabel("Number of patches")
        n_lbl.setStyleSheet(_LABEL_STYLE)
        n_row.addWidget(n_lbl)
        n_row.addStretch()
        n_row.addWidget(self._n_lbl)
        layout.addLayout(n_row)
        layout.addWidget(self._n_slider)

        # Hanging plane footprint
        self._plane_slider, self._plane_lbl = _labeled_slider(10, 100, 50, "{:.1f}")
        self._plane_lbl.setText(f"{self.hanging_plane_size:.1f}")
        self._plane_slider.valueChanged.connect(self._on_plane_size_changed)
        plane_row = QHBoxLayout()
        plane_lbl = QLabel("Hanging plane size")
        plane_lbl.setStyleSheet(_LABEL_STYLE)
        plane_row.addWidget(plane_lbl)
        plane_row.addStretch()
        plane_row.addWidget(self._plane_lbl)
        layout.addLayout(plane_row)
        layout.addWidget(self._plane_slider)

        # Swept volume resolution
        self._volume_resolution_slider, self._volume_resolution_lbl = _labeled_slider(
            24,
            512,
            108,
            "{}",
        )
        self._volume_resolution_slider.valueChanged.connect(
            lambda v: self._volume_resolution_lbl.setText(str(v))
        )
        self._volume_resolution_slider.valueChanged.connect(
            self.swept_volume_resolution_changed.emit
        )
        volume_row = QHBoxLayout()
        volume_lbl = QLabel("Swept volume resolution")
        volume_lbl.setStyleSheet(_LABEL_STYLE)
        volume_row.addWidget(volume_lbl)
        volume_row.addStretch()
        volume_row.addWidget(self._volume_resolution_lbl)
        layout.addLayout(volume_row)
        layout.addWidget(self._volume_resolution_slider)

        # Device toggle
        self._device_combo = QComboBox()
        self._device_combo.addItems(["Mac (CPU)", "Mac (MPS)", "CUDA (NVIDIA)"])
        self._device_combo.setCurrentText("CUDA (NVIDIA)")
        self._device_combo.setStyleSheet("color: #ddd; background: #2a2a2a;")
        layout.addLayout(_row("Device", self._device_combo))

        # Initialize button
        self._init_btn = QPushButton("Initialize patches")
        self._init_btn.setStyleSheet(
            "QPushButton { background: #2a6496; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #3276b1; }"
            "QPushButton:pressed { background: #204d74; }"
        )
        self._init_btn.clicked.connect(self._on_initialize)
        layout.addWidget(self._init_btn)

    def _on_initialize(self) -> None:
        n = self._n_slider.value()
        print(f"[Initialize patches] n={n}")
        self.initialize_requested.emit(n)

    @property
    def n_patches(self) -> int:
        return self._n_slider.value()

    @property
    def hanging_plane_size(self) -> float:
        """Returns the square hanging plane side length in scene units."""
        return self._plane_slider.value() / 10.0

    @property
    def swept_volume_resolution(self) -> int:
        return self._volume_resolution_slider.value()

    def _on_plane_size_changed(self, value: int) -> None:
        size = value / 10.0
        self._plane_lbl.setText(f"{size:.1f}")
        self.hanging_plane_size_changed.emit(size)

    def set_running(self, running: bool) -> None:
        for widget in (
            self._n_slider,
            self._device_combo,
            self._plane_slider,
            self._volume_resolution_slider,
            self._init_btn,
        ):
            widget.setEnabled(not running)

    @property
    def device(self) -> str:
        """Returns the torch device string for the selected option."""
        _map = {
            "Mac (CPU)":      "cpu",
            "Mac (MPS)":      "mps",
            "CUDA (NVIDIA)":  "cuda",
        }
        return _map.get(self._device_combo.currentText(), "cpu")


# ---------------------------------------------------------------------------
# Optimization section
# ---------------------------------------------------------------------------

_LR_MIN_LOG = -4.0   # 1e-4
_LR_MAX_LOG = -1.0   # 1e-1
_LR_STEPS = 1000
_THRESHOLD_MIN_LOG = -6.0   # 1e-6
_THRESHOLD_MAX_LOG = -1.0   # 0.1
_THRESHOLD_STEPS = 1000


def _slider_to_lr(pos: int) -> float:
    t = pos / _LR_STEPS
    return 10.0 ** (_LR_MIN_LOG + (_LR_MAX_LOG - _LR_MIN_LOG) * t)


def _lr_to_slider(lr: float) -> int:
    log = math.log10(max(lr, 1e-8))
    t = (log - _LR_MIN_LOG) / (_LR_MAX_LOG - _LR_MIN_LOG)
    return int(round(t * _LR_STEPS))


def _slider_to_threshold(pos: int) -> float:
    t = pos / _THRESHOLD_STEPS
    return 10.0 ** (_THRESHOLD_MIN_LOG + (_THRESHOLD_MAX_LOG - _THRESHOLD_MIN_LOG) * t)


def _threshold_to_slider(threshold: float) -> int:
    log = math.log10(max(threshold, 1e-12))
    t = (log - _THRESHOLD_MIN_LOG) / (_THRESHOLD_MAX_LOG - _THRESHOLD_MIN_LOG)
    return int(round(t * _THRESHOLD_STEPS))


class OptimizationSection(QGroupBox):
    run_requested = pyqtSignal()
    pause_toggled = pyqtSignal(bool)
    palette_changed = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Optimization", parent)
        self.setStyleSheet(_SECTION_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        # Learning rate slider (log scale)
        default_lr = 2.75e-3
        self._lr_slider = QSlider(Qt.Orientation.Horizontal)
        self._lr_slider.setRange(0, _LR_STEPS)
        self._lr_slider.setValue(_lr_to_slider(default_lr))

        self._lr_lbl = QLabel(f"{default_lr:.2e}")
        self._lr_lbl.setStyleSheet(_VALUE_STYLE)
        self._lr_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._lr_slider.valueChanged.connect(
            lambda v: self._lr_lbl.setText(f"{_slider_to_lr(v):.2e}")
        )

        lr_row = QHBoxLayout()
        lr_row_lbl = QLabel("Learning rate")
        lr_row_lbl.setStyleSheet(_LABEL_STYLE)
        lr_row.addWidget(lr_row_lbl)
        lr_row.addStretch()
        lr_row.addWidget(self._lr_lbl)
        layout.addLayout(lr_row)
        layout.addWidget(self._lr_slider)

        # Run mode
        self._run_mode_combo = QComboBox()
        self._run_mode_combo.addItems(["Fixed steps", "Until convergence"])
        self._run_mode_combo.setStyleSheet("color: #ddd; background: #2a2a2a;")
        self._run_mode_combo.currentTextChanged.connect(self._on_run_mode_changed)
        layout.addLayout(_row("Run mode", self._run_mode_combo))

        # Step count
        self._steps_slider, self._steps_lbl = _labeled_slider(1, 10000, 400, "{}")
        self._steps_slider.valueChanged.connect(
            lambda v: self._steps_lbl.setText(str(v))
        )
        self._steps_row = QHBoxLayout()
        steps_lbl = QLabel("Steps")
        steps_lbl.setStyleSheet(_LABEL_STYLE)
        self._steps_row.addWidget(steps_lbl)
        self._steps_row.addStretch()
        self._steps_row.addWidget(self._steps_lbl)
        layout.addLayout(self._steps_row)
        layout.addWidget(self._steps_slider)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, self._steps_slider.value())
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("0 / %m")
        self._progress_bar.setStyleSheet(
            "QProgressBar { color: #ddd; background: #252525; border: 1px solid #444; border-radius: 3px; text-align: center; height: 14px; }"
            "QProgressBar::chunk { background: #2a6496; border-radius: 2px; }"
        )
        self._steps_slider.valueChanged.connect(self._on_steps_changed)
        layout.addWidget(self._progress_bar)

        # Convergence threshold
        default_threshold = 1e-3
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, _THRESHOLD_STEPS)
        self._threshold_slider.setValue(_threshold_to_slider(default_threshold))
        self._threshold_lbl = QLabel(f"{default_threshold:.2e}")
        self._threshold_lbl.setStyleSheet(_VALUE_STYLE)
        self._threshold_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._threshold_slider.valueChanged.connect(
            lambda v: self._threshold_lbl.setText(f"{_slider_to_threshold(v):.2e}")
        )
        self._threshold_row = QHBoxLayout()
        threshold_lbl = QLabel("Threshold")
        threshold_lbl.setStyleSheet(_LABEL_STYLE)
        self._threshold_row.addWidget(threshold_lbl)
        self._threshold_row.addStretch()
        self._threshold_row.addWidget(self._threshold_lbl)
        layout.addLayout(self._threshold_row)
        layout.addWidget(self._threshold_slider)

        # Discrete colour palette
        palette_lbl = QLabel("Palette")
        palette_lbl.setStyleSheet(_LABEL_STYLE)
        self._palette_input = QLineEdit()
        self._palette_input.setText("#FFFFFF")
        self._palette_input.setPlaceholderText("#111111, #f4d35e, #2f6690")
        self._palette_input.setStyleSheet(
            "color: #ddd; background: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 4px;"
        )
        self._palette_input.editingFinished.connect(self.palette_changed.emit)
        layout.addWidget(palette_lbl)
        layout.addWidget(self._palette_input)

        # Run button
        self._run_btn = QPushButton("Run optimization")
        self._run_btn.setStyleSheet(
            "QPushButton { background: #3d6b2e; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #4e8a3a; }"
            "QPushButton:pressed { background: #2c5020; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        run_actions = QHBoxLayout()
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._pause_btn.setCheckable(True)
        self._pause_btn.setStyleSheet(
            "QPushButton { background: #5f5424; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #76682c; }"
            "QPushButton:checked { background: #8a5a22; }"
        )
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        run_actions.addWidget(self._pause_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setStyleSheet(
            "QPushButton { background: #5a2d2d; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #713838; }"
            "QPushButton:pressed { background: #462323; }"
        )
        self._reset_btn.clicked.connect(self.reset_requested.emit)
        run_actions.addWidget(self._reset_btn)
        layout.addLayout(run_actions)

        self._on_run_mode_changed(self._run_mode_combo.currentText())

    def _on_run_mode_changed(self, text: str) -> None:
        fixed_steps = text == "Fixed steps"
        for i in range(self._steps_row.count()):
            item = self._steps_row.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(fixed_steps)
        self._steps_slider.setVisible(fixed_steps)
        self._progress_bar.setVisible(fixed_steps)

        for i in range(self._threshold_row.count()):
            item = self._threshold_row.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(not fixed_steps)
        self._threshold_slider.setVisible(not fixed_steps)

    def _on_steps_changed(self, value: int) -> None:
        self._steps_lbl.setText(str(value))
        self._progress_bar.setMaximum(value)
        self._progress_bar.setFormat(f"%v / {value}")

    def _on_run(self) -> None:
        lr = _slider_to_lr(self._lr_slider.value())
        print(f"[Run optimization] lr={lr:.4e}")
        self.run_requested.emit()

    def _on_pause_toggled(self, paused: bool) -> None:
        self._pause_btn.setText("Resume" if paused else "Pause")
        self.pause_toggled.emit(paused)

    @property
    def learning_rate(self) -> float:
        return _slider_to_lr(self._lr_slider.value())

    @property
    def n_steps(self) -> int:
        return self._steps_slider.value()

    @property
    def run_until_convergence(self) -> bool:
        return self._run_mode_combo.currentText() == "Until convergence"

    @property
    def convergence_threshold(self) -> float:
        return _slider_to_threshold(self._threshold_slider.value())

    @property
    def palette(self) -> str:
        return self._palette_input.text()

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Optimizing..." if running else "Run optimization")
        self._pause_btn.blockSignals(True)
        self._pause_btn.setChecked(False)
        self._pause_btn.setText("Pause")
        self._pause_btn.blockSignals(False)
        self._pause_btn.setEnabled(running)
        for widget in (
            self._lr_slider,
            self._run_mode_combo,
            self._steps_slider,
            self._threshold_slider,
            self._palette_input,
        ):
            widget.setEnabled(not running)

    def reset_controls(self) -> None:
        self.set_running(False)
        self._reset_btn.setEnabled(True)
        self.reset_progress()

    def reset_progress(self) -> None:
        self._progress_bar.setRange(0, self._steps_slider.value())
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(f"0 / {self._steps_slider.value()}")

    def set_progress(self, step: int) -> None:
        total = self._steps_slider.value()
        self._progress_bar.setValue(min(step, total))
        self._progress_bar.setFormat(f"{min(step, total)} / {total}")


# ---------------------------------------------------------------------------
# Stochastic Rewrite Descent section
# ---------------------------------------------------------------------------


class SRDSection(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Stochastic Rewrite Descent", parent)
        self.setStyleSheet(_SECTION_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        self._enabled = QCheckBox("Enable SRD")
        self._enabled.setChecked(True)
        self._enabled.setStyleSheet(_LABEL_STYLE)
        layout.addWidget(self._enabled)

        self._propose_slider, self._propose_lbl = _labeled_slider(20, 200, 50, "{}")
        self._propose_slider.valueChanged.connect(lambda v: self._propose_lbl.setText(str(v)))
        layout.addLayout(self._value_row("Propose every N steps", self._propose_lbl))
        layout.addWidget(self._propose_slider)

        self._proposal_steps_slider, self._proposal_steps_lbl = _labeled_slider(1, 20, 5, "{}")
        self._proposal_steps_slider.valueChanged.connect(lambda v: self._proposal_steps_lbl.setText(str(v)))
        layout.addLayout(self._value_row("Proposal steps", self._proposal_steps_lbl))
        layout.addWidget(self._proposal_steps_slider)

        self._candidates_slider, self._candidates_lbl = _labeled_slider(16, 128, 64, "{}")
        self._candidates_slider.valueChanged.connect(lambda v: self._candidates_lbl.setText(str(v)))
        layout.addLayout(self._value_row("Num candidates K", self._candidates_lbl))
        layout.addWidget(self._candidates_slider)

        self._swept_spawn_slider, self._swept_spawn_lbl = _labeled_slider(0, 100, 75, "{}%")
        self._swept_spawn_slider.valueChanged.connect(
            lambda v: self._swept_spawn_lbl.setText(f"{v}%")
        )
        layout.addLayout(self._value_row("Swept-volume add %", self._swept_spawn_lbl))
        layout.addWidget(self._swept_spawn_slider)

        self._max_patches_slider, self._max_patches_lbl = _labeled_slider(10, 500, 200, "{}")
        self._max_patches_slider.valueChanged.connect(lambda v: self._max_patches_lbl.setText(str(v)))
        layout.addLayout(self._value_row("Max patches", self._max_patches_lbl))
        layout.addWidget(self._max_patches_slider)

        self._min_patches_slider, self._min_patches_lbl = _labeled_slider(1, 20, 4, "{}")
        self._min_patches_slider.valueChanged.connect(lambda v: self._min_patches_lbl.setText(str(v)))
        layout.addLayout(self._value_row("Min patches", self._min_patches_lbl))
        layout.addWidget(self._min_patches_slider)

        self._stats_lbl = QLabel("Patches: 0 | Added: 0 | Deleted: 0 | Mandatory deleted: 0")
        self._stats_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self._stats_lbl.setWordWrap(True)
        layout.addWidget(self._stats_lbl)

    def _value_row(self, label_text: str, value_lbl: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet(_LABEL_STYLE)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(value_lbl)
        return row

    @property
    def config(self) -> dict[str, object]:
        return {
            "enabled": self._enabled.isChecked(),
            "interval": self._propose_slider.value(),
            "rewrite_eval_steps": self._proposal_steps_slider.value(),
            "candidate_count": self._candidates_slider.value(),
            "swept_volume_spawn_fraction": self._swept_spawn_slider.value() / 100.0,
            "max_patches": self._max_patches_slider.value(),
            "min_patches": self._min_patches_slider.value(),
        }

    def set_stats(self, metrics: dict) -> None:
        patches = int(metrics.get("srd_active_patches", metrics.get("patches", 0)))
        added = int(metrics.get("srd_total_adds", 0))
        deleted = int(metrics.get("srd_total_deletes", 0))
        mandatory = int(metrics.get("srd_total_mandatory_deletes", 0))
        self._stats_lbl.setText(
            f"Patches: {patches} | Added: {added} | Deleted: {deleted} | Mandatory deleted: {mandatory}"
        )

    def set_running(self, running: bool) -> None:
        for widget in (
            self._enabled,
            self._propose_slider,
            self._proposal_steps_slider,
            self._candidates_slider,
            self._swept_spawn_slider,
            self._max_patches_slider,
            self._min_patches_slider,
        ):
            widget.setEnabled(not running)


# ---------------------------------------------------------------------------
# Edit section
# ---------------------------------------------------------------------------


class EditSection(QGroupBox):
    edit_mode_toggled = pyqtSignal(bool)
    piece_selected = pyqtSignal(int)
    nudge_requested = pyqtSignal(float, float, float)
    rotate_requested = pyqtSignal(float)
    scale_requested = pyqtSignal(float)
    add_piece_requested = pyqtSignal()
    delete_requested = pyqtSignal()

    SCALE_STEP_FACTOR = 1.1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Edit", parent)
        self.setStyleSheet(_SECTION_STYLE)

        self._running = False
        self._has_pieces = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        self._edit_btn = QPushButton("Enter edit mode")
        self._edit_btn.setCheckable(True)
        self._edit_btn.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #ddd; border-radius: 4px; padding: 6px; }"
            "QPushButton:checked { background: #2a6496; color: #fff; }"
            "QPushButton:hover:!checked { background: #4a4a4a; }"
        )
        self._edit_btn.toggled.connect(self._on_edit_toggled)
        layout.addWidget(self._edit_btn)

        self._add_piece_btn = self._make_action_button(
            "Add piece at origin",
            self.add_piece_requested.emit,
        )
        layout.addWidget(self._add_piece_btn)

        piece_lbl = QLabel("Piece selection")
        piece_lbl.setStyleSheet(_LABEL_STYLE)
        layout.addWidget(piece_lbl)

        self._piece_list = QListWidget()
        self._piece_list.setStyleSheet(
            "QListWidget { background: #232324; color: #ddd; border: 1px solid #444; border-radius: 3px; }"
            "QListWidget::item { padding: 4px 6px; }"
            "QListWidget::item:selected { background: #2a6496; color: #fff; }"
        )
        self._piece_list.setMinimumHeight(110)
        self._piece_list.itemSelectionChanged.connect(self._on_piece_changed)
        layout.addWidget(self._piece_list)

        self._move_slider, self._move_lbl = _labeled_slider(1, 20, 5, "{:.2f}")
        self._move_lbl.setText(f"{self.move_step:.2f}")
        self._move_slider.valueChanged.connect(self._on_move_step_changed)
        move_row = QHBoxLayout()
        move_lbl = QLabel("Move step")
        move_lbl.setStyleSheet(_LABEL_STYLE)
        move_row.addWidget(move_lbl)
        move_row.addStretch()
        move_row.addWidget(self._move_lbl)
        layout.addLayout(move_row)
        layout.addWidget(self._move_slider)

        self._rotate_slider, self._rotate_lbl = _labeled_slider(1, 30, 5, "{}°")
        self._rotate_lbl.setText(f"{self.rotate_step_degrees}°")
        self._rotate_slider.valueChanged.connect(self._on_rotate_step_changed)
        rotate_row = QHBoxLayout()
        rotate_lbl = QLabel("Rotate step")
        rotate_lbl.setStyleSheet(_LABEL_STYLE)
        rotate_row.addWidget(rotate_lbl)
        rotate_row.addStretch()
        rotate_row.addWidget(self._rotate_lbl)
        layout.addLayout(rotate_row)
        layout.addWidget(self._rotate_slider)

        move_grid = QGridLayout()
        move_grid.setHorizontalSpacing(6)
        move_grid.setVerticalSpacing(6)
        self._btn_x_neg = self._make_action_button("X-", lambda: self._emit_nudge(-self.move_step, 0.0, 0.0))
        self._btn_x_pos = self._make_action_button("X+", lambda: self._emit_nudge(self.move_step, 0.0, 0.0))
        self._btn_y_neg = self._make_action_button("Y-", lambda: self._emit_nudge(0.0, -self.move_step, 0.0))
        self._btn_y_pos = self._make_action_button("Y+", lambda: self._emit_nudge(0.0, self.move_step, 0.0))
        self._btn_z_neg = self._make_action_button("Z-", lambda: self._emit_nudge(0.0, 0.0, -self.move_step))
        self._btn_z_pos = self._make_action_button("Z+", lambda: self._emit_nudge(0.0, 0.0, self.move_step))
        move_grid.addWidget(self._btn_x_neg, 0, 0)
        move_grid.addWidget(self._btn_x_pos, 0, 1)
        move_grid.addWidget(self._btn_y_neg, 1, 0)
        move_grid.addWidget(self._btn_y_pos, 1, 1)
        move_grid.addWidget(self._btn_z_neg, 2, 0)
        move_grid.addWidget(self._btn_z_pos, 2, 1)
        layout.addLayout(move_grid)

        rotate_row = QHBoxLayout()
        self._btn_rot_neg = self._make_action_button("Rotate -", lambda: self.rotate_requested.emit(-self.rotate_step_degrees))
        self._btn_rot_pos = self._make_action_button("Rotate +", lambda: self.rotate_requested.emit(self.rotate_step_degrees))
        rotate_row.addWidget(self._btn_rot_neg)
        rotate_row.addWidget(self._btn_rot_pos)
        layout.addLayout(rotate_row)

        scale_row = QHBoxLayout()
        self._btn_scale_down = self._make_action_button(
            "Scale -",
            lambda: self.scale_requested.emit(1.0 / self.SCALE_STEP_FACTOR),
        )
        self._btn_scale_up = self._make_action_button(
            "Scale +",
            lambda: self.scale_requested.emit(self.SCALE_STEP_FACTOR),
        )
        scale_row.addWidget(self._btn_scale_down)
        scale_row.addWidget(self._btn_scale_up)
        layout.addLayout(scale_row)

        self._delete_btn = QPushButton("Delete selected piece")
        self._delete_btn.setStyleSheet(
            "QPushButton { background: #5a2d2d; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #713838; }"
            "QPushButton:pressed { background: #462323; }"
        )
        self._delete_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(self._delete_btn)

        hint = QLabel(
            "Use edit mode after initialization/import/optimization — or while "
            "optimization is paused — to add, nudge, rotate, scale, or delete "
            "pieces. Changes made while paused carry into the optimizer."
        )
        hint.setStyleSheet("color: #777; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.set_piece_labels([])

    def _make_action_button(self, label: str, on_click) -> QPushButton:
        btn = QPushButton(label)
        btn.setStyleSheet(
            "QPushButton { background: #2f4d5f; color: #fff; border-radius: 4px; padding: 5px; }"
            "QPushButton:hover { background: #3b6178; }"
            "QPushButton:pressed { background: #254052; }"
        )
        btn.clicked.connect(on_click)
        return btn

    def _on_edit_toggled(self, enabled: bool) -> None:
        self._edit_btn.setText("Exit edit mode" if enabled else "Enter edit mode")
        self._refresh_enabled_state()
        self.edit_mode_toggled.emit(enabled)

    def _on_piece_changed(self) -> None:
        self._refresh_enabled_state()
        self.piece_selected.emit(self.selected_piece_index)

    def _on_move_step_changed(self, _value: int) -> None:
        self._move_lbl.setText(f"{self.move_step:.2f}")

    def _on_rotate_step_changed(self, _value: int) -> None:
        self._rotate_lbl.setText(f"{self.rotate_step_degrees}°")

    def _emit_nudge(self, dx: float, dy: float, dz: float) -> None:
        self.nudge_requested.emit(dx, dy, dz)

    @property
    def move_step(self) -> float:
        return self._move_slider.value() / 100.0

    @property
    def rotate_step_degrees(self) -> float:
        return float(self._rotate_slider.value())

    @property
    def selected_piece_index(self) -> int:
        current = self._piece_list.currentItem()
        if current is None:
            return -1
        data = current.data(Qt.ItemDataRole.UserRole)
        if isinstance(data, int):
            return data
        return -1

    @property
    def edit_mode_enabled(self) -> bool:
        return self._edit_btn.isChecked()

    def set_piece_labels(self, labels: list[str]) -> None:
        previous = self.selected_piece_index
        self._piece_list.blockSignals(True)
        self._piece_list.clear()
        if labels:
            for index, label in enumerate(labels):
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, index)
                self._piece_list.addItem(item)
            self._piece_list.setCurrentRow(previous if 0 <= previous < len(labels) else 0)
        else:
            item = QListWidgetItem("No pieces available")
            item.setData(Qt.ItemDataRole.UserRole, -1)
            self._piece_list.addItem(item)
            self._piece_list.setCurrentRow(0)
        self._piece_list.blockSignals(False)
        self._has_pieces = bool(labels)
        if not self._has_pieces:
            self.set_edit_mode(False)
        self._refresh_enabled_state()
        self.piece_selected.emit(self.selected_piece_index)

    def set_selected_piece(self, index: int) -> None:
        if not self._has_pieces:
            return
        if index < 0 or index >= self._piece_list.count():
            return
        current = self.selected_piece_index
        if current == index:
            return
        self._piece_list.setCurrentRow(index)

    def set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self.set_edit_mode(False)
        self._refresh_enabled_state()

    def set_edit_mode(self, enabled: bool) -> None:
        checked = bool(enabled) and self._has_pieces and not self._running
        self._edit_btn.blockSignals(True)
        self._edit_btn.setChecked(checked)
        self._edit_btn.setText("Exit edit mode" if checked else "Enter edit mode")
        self._edit_btn.blockSignals(False)
        self._refresh_enabled_state()

    def _refresh_enabled_state(self) -> None:
        can_toggle = self._has_pieces and not self._running
        self._edit_btn.setEnabled(can_toggle)
        self._piece_list.setEnabled(can_toggle)
        self._add_piece_btn.setEnabled(can_toggle and self._edit_btn.isChecked())

        active = can_toggle and self._edit_btn.isChecked() and self.selected_piece_index >= 0
        for widget in (
            self._move_slider,
            self._rotate_slider,
            self._btn_x_neg,
            self._btn_x_pos,
            self._btn_y_neg,
            self._btn_y_pos,
            self._btn_z_neg,
            self._btn_z_pos,
            self._btn_rot_neg,
            self._btn_rot_pos,
            self._btn_scale_down,
            self._btn_scale_up,
            self._delete_btn,
        ):
            widget.setEnabled(active)


# ---------------------------------------------------------------------------
# Export section
# ---------------------------------------------------------------------------


class ExportSection(QGroupBox):
    export_requested = pyqtSignal()
    import_requested = pyqtSignal()
    strings_requested = pyqtSignal()
    swept_volume_requested = pyqtSignal()
    split_test_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Export", parent)
        self.setStyleSheet(_SECTION_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        self._export_btn = QPushButton("Export pieces JSON")
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #777; border-radius: 4px; padding: 6px; }"
            "QPushButton:enabled { background: #5a3e7a; color: #fff; }"
            "QPushButton:enabled:hover { background: #6f4e96; }"
        )
        self._export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(self._export_btn)

        self._import_btn = QPushButton("Import pieces JSON")
        self._import_btn.setStyleSheet(
            "QPushButton { background: #2a6496; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #3276b1; }"
            "QPushButton:pressed { background: #204d74; }"
        )
        self._import_btn.clicked.connect(self.import_requested.emit)
        layout.addWidget(self._import_btn)

        self._strings_btn = QPushButton("Add strings")
        self._strings_btn.setEnabled(False)
        self._strings_btn.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #777; border-radius: 4px; padding: 6px; }"
            "QPushButton:enabled { background: #49633a; color: #fff; }"
            "QPushButton:enabled:hover { background: #5b7b48; }"
        )
        self._strings_btn.clicked.connect(self.strings_requested.emit)
        layout.addWidget(self._strings_btn)

        self._swept_volume_btn = QPushButton("Toggle swept volume")
        self._swept_volume_btn.setStyleSheet(
            "QPushButton { background: #2f4d5f; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #3b6178; }"
            "QPushButton:pressed { background: #254052; }"
        )
        self._swept_volume_btn.clicked.connect(self.swept_volume_requested.emit)
        layout.addWidget(self._swept_volume_btn)

        self._split_test_btn = QPushButton("Visualize split test")
        self._split_test_btn.setStyleSheet(
            "QPushButton { background: #2f4d5f; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #3b6178; }"
            "QPushButton:pressed { background: #254052; }"
        )
        self._split_test_btn.clicked.connect(self.split_test_requested.emit)
        layout.addWidget(self._split_test_btn)

        note = QLabel("Import previous designs or export current patches to exports/pieces.json.")
        note.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        note.setWordWrap(True)
        layout.addWidget(note)

    def set_enabled(self, enabled: bool, debug_enabled: bool = True) -> None:
        self._export_btn.setEnabled(enabled)
        self._strings_btn.setEnabled(enabled)
        self._split_test_btn.setEnabled(debug_enabled)


# ---------------------------------------------------------------------------
# Main controls panel
# ---------------------------------------------------------------------------


class ControlsPanel(QWidget):
    """Right-side panel containing all controls in a scroll area."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Controls", self)
        header.setStyleSheet(
            "color: #ddd; font-size: 14px; font-weight: bold; padding: 10px 10px 4px 10px;"
        )
        outer.addWidget(header)

        # Scroll area so controls don't get clipped on small windows
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 6, 10, 10)
        layout.setSpacing(12)

        self.patches = PatchesSection(container)
        layout.addWidget(self.patches)

        self.optimization = OptimizationSection(container)
        layout.addWidget(self.optimization)

        self.srd = SRDSection(container)
        layout.addWidget(self.srd)

        self.edit = EditSection(container)
        layout.addWidget(self.edit)

        self.export = ExportSection(container)
        layout.addWidget(self.export)

        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)
