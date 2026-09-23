"""Stochastic Rewrite Descent for adaptive patch structure."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from core.patch import ControlPoint, Patch
from core.swept_volume import _project_points

if TYPE_CHECKING:
    from core.swept_volume import SweptVolume
    from scene.camera import Camera


RewriteKind = Literal["add", "delete", "split", "restart"]


@dataclass
class SRDStats:
    added: int = 0
    deleted: int = 0
    # Lifetime counts per operation. ``total_added`` is the total number of
    # growth rewrites (adds + splits) and is kept as the headline growth
    # figure; the two components are recorded separately because they grow the
    # sculpture in different ways -- an add drops a new small piece into
    # uncovered space, a split subdivides a piece that is already carrying
    # geometry.
    total_adds: int = 0
    total_splits: int = 0
    total_deleted: int = 0
    # Conflict-gated restarts (delete a conflicting piece and respawn one into
    # the residual it vacated). Counted on their own so the plain-delete and
    # add totals stay comparable across batches with the feature off.
    restarts: int = 0
    total_restarts: int = 0
    active: int = 0
    evaluated: int = 0
    promising: int = 0
    accepted: int = 0

    @property
    def total_added(self) -> int:
        return self.total_adds + self.total_splits


@dataclass
class RewriteCandidate:
    kind: RewriteKind
    position: np.ndarray | None = None
    patch_index: int | None = None
    improvement: float = 0.0
    applied_index: int | None = None
    reason: str = ""

    @property
    def label(self) -> str:
        if self.kind == "delete":
            return f"DeletePatch({self.patch_index})"
        if self.kind == "split":
            return f"SplitPatch({self.patch_index})"
        if self.kind == "restart":
            return f"RestartPatch({self.patch_index})"
        return "AddPatch"

    @property
    def net_count_change(self) -> int:
        """Change in the scene's patch count if this rewrite is applied.

        A delete removes one piece; an add or a split each net one more (a
        split turns one patch into two); a restart deletes one and respawns
        one, so it nets zero. Used to charge the piece-count penalty against a
        rewrite's loss improvement.
        """
        if self.kind == "delete":
            return -1
        if self.kind == "restart":
            return 0
        return 1


def _patch_parameters(patches: Sequence[Patch]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for patch in patches:
        params.extend([patch.center, patch.theta])
        for cp in patch.control_points:
            cp.z.requires_grad_(False)
            cp.z.grad = None
            params.extend([cp.x, cp.y, cp.handle_scale, cp.handle_rotation])
    return params


def _small_default_patch(
    position: np.ndarray,
    device: str,
    albedo: Sequence[float],
    creation_step: int,
    label: str,
    radius: float = 0.05,
) -> Patch:
    """Create a small regular-pentagon patch for SRD additions."""
    control_points: list[ControlPoint] = []
    handle_scale = radius * (4.0 / 3.0) * math.tan(math.pi / Patch.N_CONTROL_POINTS)
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
    patch = Patch(
        control_points=control_points,
        center=position.tolist(),
        theta=0.0,
        albedo=list(albedo),
        device=device,
        label=label,
    )
    patch.creation_step = creation_step
    patch.self_intersect_counter = 0
    return patch


class StochasticRewriteDescent:
    """Sample candidate add/delete rewrites and accept useful compatible ones."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        interval: int = 50,
        lambda_count: float = 0.05,
        lambda_mode: str = "fixed",
        lambda_target_iou: float = 0.90,
        lambda_eta: float = 0.5,
        lambda_min: float = 1e-5,
        lambda_max: float = 1.0,
        count_objective: bool = False,
        count_target_iou: float = 0.90,
        count_lambda: float = 2000.0,
        count_power: float = 2.0,
        count_softplus_beta: float = 0.0,
        lambda_area: float = 0.05,
        min_patch_area: float = 0.01,
        max_patches: int = 200,
        min_patches: int = 4,
        max_additions: int = 3,
        max_deletions: int = 3,
        cooldown_steps: int = 30,
        candidate_count: int = 64,
        add_weight: float = 0.35,
        delete_weight: float = 0.15,
        split_weight: float = 0.50,
        scene_box_size: float = 5.0,
        rewrite_eval_steps: int = 4,
        no_contribution_alpha: float = 1e-5,
        no_effect_image_delta: float = 1e-6,
        rule_violation_tol: float = 1e-4,
        swept_volume: "SweptVolume | None" = None,
        swept_volume_spawn_fraction: float = 1.0,
        disable_swept_volume_adds: bool = False,
        loss_only_deletion: bool = False,
        disable_splitting: bool = False,
        deletion_importance: bool = False,
        deletion_temperature: float = 1.0,
        deletion_proxy: str = "spill",
        conflict_restart: bool = False,
        conflict_eps: float = 1e-3,
    ) -> None:
        self.enabled = enabled
        self.interval = interval
        self.lambda_count = lambda_count
        # Piece-count penalty schedule. "fixed" holds lambda_count constant; in
        # "dual" mode it is a Lagrange multiplier on the constraint
        # mean_iou >= lambda_target_iou, adapted by update_lambda_count().
        self.lambda_mode = lambda_mode
        self.lambda_target_iou = float(lambda_target_iou)
        self.lambda_eta = float(lambda_eta)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        # Piece-count objective (see _count_objective_value). When on, rewrites
        # are scored by the change in
        #     J = n_patches + count_lambda * hinge(target_iou - mean_iou)^power
        # instead of by the change in image loss, so lambda_count is unused.
        self.count_objective = bool(count_objective)
        self.count_target_iou = float(count_target_iou)
        self.count_lambda = float(count_lambda)
        self.count_power = float(count_power)
        self.count_softplus_beta = float(count_softplus_beta)
        self._baseline_objective: float | None = None
        self.lambda_area = lambda_area
        self.min_patch_area = min_patch_area
        self.max_patches = max_patches
        self.min_patches = min_patches
        self.max_additions = max_additions
        self.max_deletions = max_deletions
        self.cooldown_steps = cooldown_steps
        self.candidate_count = candidate_count
        # How the per-step candidate budget is divided between the three
        # rewrite kinds. These are proportions, not probabilities of
        # acceptance: they say how many candidates of each kind SRD gets to
        # *look at* each rewrite step, and every candidate still has to beat
        # the loss before it is applied. Normalized so any positive triple
        # works; the defaults are the historical 0.35/0.15/0.50 split.
        weights = np.array(
            [max(0.0, float(add_weight)),
             max(0.0, float(delete_weight)),
             max(0.0, float(split_weight))],
            dtype=np.float64,
        )
        if weights.sum() <= 0.0:
            weights = np.array([0.35, 0.15, 0.50], dtype=np.float64)
        weights /= weights.sum()
        self.add_weight, self.delete_weight, self.split_weight = (float(w) for w in weights)
        self.scene_box_size = scene_box_size
        self.rewrite_eval_steps = rewrite_eval_steps
        self.no_contribution_alpha = no_contribution_alpha
        self.no_effect_image_delta = no_effect_image_delta
        self.rule_violation_tol = rule_violation_tol
        self.swept_volume = swept_volume
        self.swept_volume_spawn_fraction = float(np.clip(swept_volume_spawn_fraction, 0.0, 1.0))
        # Ablation switches: disable swept-volume-guided additions, restrict
        # deletion to loss-improving rewrites only, and disable splitting.
        self.disable_swept_volume_adds = bool(disable_swept_volume_adds)
        self.loss_only_deletion = bool(loss_only_deletion)
        self.disable_splitting = bool(disable_splitting)
        self._swept_point_order = np.empty(0, dtype=np.int64)
        self._swept_point_cursor = 0
        # Damage-based importance sampling of deletion candidates. When off, the
        # delete loop draws eligible pieces uniformly (the original behaviour).
        # When on, it draws them softmax-weighted by a cheap per-piece proxy for
        # how little their deletion would cost -- see _deletion_proxy_weights.
        self.deletion_importance = bool(deletion_importance)
        self.deletion_temperature = max(1e-6, float(deletion_temperature))
        if deletion_proxy not in ("spill", "net"):
            raise ValueError(
                f"deletion_proxy must be 'spill' or 'net', got {deletion_proxy!r}"
            )
        self.deletion_proxy = deletion_proxy
        # Conflict-gated restart. When on, a deletion candidate that helps one
        # view but hurts the other (opposite-signed per-view loss deltas) is
        # turned into an atomic delete+respawn -- the replacement seeded from
        # the swept volume over the hole the piece vacated -- instead of a plain
        # delete the lookahead would reject. Pieces bad in both views stay plain
        # deletes. conflict_eps is the per-view loss delta (in loss units) that
        # a view must clear to count as helped/hurt, filtering render noise.
        self.conflict_restart = bool(conflict_restart)
        self.conflict_eps = max(0.0, float(conflict_eps))
        self.stats = SRDStats()

    def step(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        current_loss: float,
        cameras: Sequence["Camera"],
        targets: tuple[torch.Tensor, torch.Tensor | None],
        current_step: int,
    ) -> SRDStats:
        """Run one SRD rewrite pass when the interval says it is time."""
        self.stats.added = 0
        self.stats.deleted = 0
        self.stats.evaluated = 0
        self.stats.promising = 0
        self.stats.accepted = 0
        self.stats.active = len(model.patches)

        if not self.enabled or self.interval <= 0 or current_step % self.interval != 0:
            return self.stats

        tiny_deletes = (
            [] if self.loss_only_deletion else self._tiny_area_delete_rewrites(model)
        )
        if tiny_deletes:
            self._apply_rewrites(model, optimizer, tiny_deletes, current_step)
            self.stats.accepted += len(tiny_deletes)
            self.stats.active = len(model.patches)

        mandatory_deletes = (
            [] if self.loss_only_deletion
            else self._mandatory_delete_rewrites(model, current_step)
        )
        if mandatory_deletes:
            self._apply_rewrites(model, optimizer, mandatory_deletes, current_step)
            self.stats.accepted += len(mandatory_deletes)
            self.stats.active = len(model.patches)
            with torch.no_grad():
                render1, render2 = model.renderer.render_both(
                    model.patches,
                    model.camera1,
                    model.camera2,
                    model.render_resolutions,
                )
                current_loss_tensor, _ = model._loss_from_renders(render1, render2, model.patches)
                current_loss = float(current_loss_tensor.detach().cpu())

        # Baseline for the piece-count objective, measured after the mandatory
        # deletions above so candidates are scored against the configuration
        # they will actually be applied to.
        self._baseline_objective = (
            self._current_count_objective(model) if self.count_objective else None
        )

        uncovered_masks = self._uncovered_target_masks(model)
        candidates = self._sample_rewrites(model, current_step, uncovered_masks)
        scored: list[RewriteCandidate] = []
        for candidate in candidates:
            improvement = self.evaluate_rewrite(model, optimizer, candidate, current_loss)
            self.stats.evaluated += 1
            if improvement > 0.0:
                candidate.improvement = improvement
                scored.append(candidate)

        self.stats.promising = len(scored)
        accepted = self._select_compatible(scored)
        self._apply_rewrites(model, optimizer, accepted, current_step)
        self.stats.accepted += len(accepted)
        self.stats.active = len(model.patches)

        print(
            f"SRD step {current_step}: evaluated {self.stats.evaluated} candidates, "
            f"{self.stats.promising} promising, accepted {self.stats.accepted}"
        )
        for candidate in accepted:
            patch_ref = candidate.applied_index if candidate.applied_index is not None else candidate.patch_index
            print(f"  Accepted {candidate.label} at patch {patch_ref}, improvement={candidate.improvement:.6f}")
        for candidate in [*tiny_deletes, *mandatory_deletes]:
            patch_ref = candidate.applied_index if candidate.applied_index is not None else candidate.patch_index
            print(f"  Mandatory {candidate.label} at patch {patch_ref}, reason={candidate.reason}")
        print(
            f"  Total patches: {len(model.patches)}, total adds: {self.stats.total_adds}, "
            f"total splits: {self.stats.total_splits}, "
            f"total deletes: {self.stats.total_deleted}, "
            f"total restarts: {self.stats.total_restarts}"
        )
        return self.stats

    def _tiny_area_delete_rewrites(self, model) -> list[RewriteCandidate]:
        """Delete all below-threshold pieces at the start of a rewrite step."""
        if len(model.patches) <= 1:
            return []

        areas = [float(patch.compute_area().detach().cpu()) for patch in model.patches]
        indices = [idx for idx, area in enumerate(areas) if area < self.min_patch_area]
        if len(indices) >= len(model.patches):
            keep_idx = min(range(len(areas)), key=lambda idx: areas[idx])
            indices = [idx for idx in indices if idx != keep_idx]

        return [
            RewriteCandidate(
                kind="delete",
                patch_index=idx,
                reason=f"Area {areas[idx]:.6f} below minimum {self.min_patch_area}",
            )
            for idx in indices
        ]

    def final_deletion_pass(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        current_step: int,
    ) -> dict[str, float]:
        """Greedily delete tiny or loss-worsening pieces at the end of a run."""
        stats = {
            "tiny_deleted": 0.0,
            "loss_improving_deleted": 0.0,
            "evaluated": 0.0,
        }
        if not self.enabled or len(model.patches) <= 1:
            return stats

        tiny_deletes = (
            [] if self.loss_only_deletion else self._tiny_area_delete_rewrites(model)
        )
        if tiny_deletes:
            self._apply_rewrites(model, optimizer, tiny_deletes, current_step)
            optimizer = model.optim
            stats["tiny_deleted"] = float(len(tiny_deletes))

        current_loss = self._current_model_loss(model)
        patch_index = 0
        while patch_index < len(model.patches) and len(model.patches) > 1:
            stats["evaluated"] += 1.0
            loss_without_patch = self._loss_without_patch(model, patch_index)
            # Keep a piece only if removing it worsens the loss by more than
            # lambda_count, i.e. delete whenever the piece-count-penalized
            # objective (loss + lambda_count * n_patches) improves.
            if loss_without_patch < current_loss + self.lambda_count:
                rewrite = RewriteCandidate(
                    kind="delete",
                    patch_index=patch_index,
                    improvement=current_loss - loss_without_patch + self.lambda_count,
                    reason="final deletion pass improved piece-count-penalized loss",
                )
                self._apply_single(
                    model,
                    rewrite,
                    current_step=current_step,
                    tentative=False,
                )
                optimizer = self._rebuild_optimizer(model, optimizer)
                model.optim = optimizer
                model._post_step_constraints()
                current_loss = loss_without_patch
                stats["loss_improving_deleted"] += 1.0
                continue
            patch_index += 1

        model._post_step_constraints()
        return stats

    def update_lambda_count(self, current_iou: float) -> float:
        """Dual-ascent update of the piece-count penalty (``lambda_mode='dual'``).

        A fixed ``lambda_count`` is an *absolute* threshold in loss units: a
        rewrite must beat the loss by more than lambda per piece it adds. The
        marginal loss a single piece can buy is shape-dependent, though -- a
        complicated silhouette is covered by many pieces that each win a little,
        a simple one by a few that each win a lot. So one global lambda rations
        pieces hardest exactly where each piece is worth least, i.e. on the
        shapes that need the most pieces. That is backwards.

        Dual ascent states the goal directly instead: minimise piece count
        subject to ``mean_iou >= lambda_target_iou``, with lambda_count as the
        multiplier on that constraint,

            lambda <- clip(lambda * exp(eta * (iou - target)), min, max)

        Above target, lambda rises and additions stop paying for themselves;
        below it, lambda decays and growth unblocks. Each shape therefore
        settles at its own lambda, and that converged value is a difficulty
        score: a shape that needs 30 pieces to hit the target drives lambda
        down and keeps them, one that needs 15 drives it up and sheds the rest.

        The update is multiplicative, so lambda_min must stay above zero -- a
        lambda that reaches 0 can never grow again.
        """
        if self.lambda_mode != "dual":
            return self.lambda_count
        error = float(current_iou) - self.lambda_target_iou
        scaled = self.lambda_count * float(np.exp(self.lambda_eta * error))
        self.lambda_count = float(np.clip(scaled, self.lambda_min, self.lambda_max))
        return self.lambda_count

    def pruning_path(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        *,
        min_pieces: int = 1,
        refit_steps: int = 4,
        iou_tolerance: float = 0.01,
        target_iou: float | None = None,
    ) -> tuple[list[dict], dict]:
        """Delete the least-damaging piece repeatedly, recording IoU at every count.

        This is the measurement counterpart to ``lambda_count``: rather than
        pick a penalty and see what piece count falls out, walk the entire
        IoU-vs-pieces curve for *this* shape in a single run, then choose the
        count from the curve. One run yields the whole tradeoff, so there is
        nothing left for a lambda sweep to discover.

        Two things differ from ``final_deletion_pass``, and both matter:

        * It deletes the **argmin-damage** piece each round. The deletion pass
          walks ``patch_index`` in order and removes the first piece under
          threshold, which is an arbitrary order, not a cheapest-first one.
        * It **re-fits** the survivors for ``refit_steps`` gradient steps after
          each commit. Scoring a deletion on frozen geometry overstates its
          cost, because the neighbouring pieces would have grown to cover the
          hole. Without this the curve reads far more pessimistic than the
          configuration a real run would reach at that piece count.

        Candidate *scoring* stays frozen (no_grad, no re-fit) -- that is the
        O(n^2) part, and re-fitting every candidate would multiply the whole
        pass by ``refit_steps``. Only the committed deletion is re-fit.

        Returns ``(path, selected)``: one row per piece count from the starting
        configuration down to ``min_pieces``, and the row the model was left
        at. Selection keeps the *fewest* pieces that still reach ``target_iou``
        if one is given, else the fewest within ``iou_tolerance`` of the best
        mean IoU seen anywhere on the path. The model and optimizer are
        restored to that configuration, so the run ends at the knee rather than
        at one piece.
        """
        path: list[dict] = []
        if not self.enabled or len(model.patches) <= 1:
            return path, {}

        floor = max(1, int(min_pieces))
        # Snapshot every configuration so the chosen one can be restored after
        # the path has walked past it.
        snapshots: dict[int, tuple[list[dict], dict]] = {}

        def record(deleted_index: int | None) -> None:
            metrics, _, _ = model.evaluate_snapshot()
            n = len(model.patches)
            snapshots[n] = self._save_state(model, model.optim)
            path.append({
                "patches": n,
                "deleted_patch_index": -1 if deleted_index is None else deleted_index,
                "loss": float(metrics.get("loss", 0.0)),
                "mean_iou": float(metrics.get("mean_iou", 0.0)),
                "view1_iou": float(metrics.get("view1_iou", 0.0)),
                "view2_iou": float(metrics.get("view2_iou", 0.0)),
                "mean_coverage": float(metrics.get("mean_coverage", 0.0)),
                "mean_precision": float(metrics.get("mean_precision", 0.0)),
                "mean_spill": float(metrics.get("mean_spill", 0.0)),
            })

        record(None)

        while len(model.patches) > floor:
            # Score every remaining piece frozen, cheapest deletion first.
            losses = [
                self._loss_without_patch(model, idx)
                for idx in range(len(model.patches))
            ]
            victim = int(np.argmin(losses))
            self._apply_single(
                model,
                RewriteCandidate(
                    kind="delete",
                    patch_index=victim,
                    reason="pruning path: least-damaging remaining piece",
                ),
                current_step=0,
                tentative=False,
            )
            model.optim = self._rebuild_optimizer(model, model.optim)
            model._post_step_constraints()
            self._refit(model, refit_steps)
            record(victim)

        best_iou = max(row["mean_iou"] for row in path)
        threshold = target_iou if target_iou is not None else best_iou - iou_tolerance
        eligible = [row for row in path if row["mean_iou"] >= threshold]
        # If the target is unreachable for this shape, fall back to the best
        # point on the path rather than to the smallest -- an unmeetable
        # constraint should not silently collapse the sculpture to one piece.
        selected = (
            min(eligible, key=lambda row: row["patches"])
            if eligible
            else max(path, key=lambda row: row["mean_iou"])
        )
        selected = dict(selected)
        selected["target_reached"] = bool(eligible)
        selected["threshold"] = float(threshold)

        patch_states, optimizer_state = snapshots[selected["patches"]]
        self._restore_state(model, model.optim, patch_states, optimizer_state)
        return path, selected

    def _refit(self, model, n_steps: int) -> None:
        """Run ``n_steps`` gradient steps on the current patch set."""
        for _ in range(max(0, int(n_steps))):
            valid_shape_states = model._capture_patch_shape_states()
            model.optim.zero_grad(set_to_none=True)
            render1, render2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            loss, _ = model._loss_from_renders(render1, render2, model.patches)
            loss.backward()
            model.optim.step()
            model.optim.zero_grad(set_to_none=True)
            model._post_step_constraints(valid_shape_states, model.optim)

    def _count_objective_value(self, n_patches: int, mean_iou: float) -> float:
        """Pieces, penalized only while mean IoU is below the threshold.

            J = n_patches + count_lambda * hinge(target_iou - mean_iou)^power

        The point of the hinge is that the IoU term switches *off* once the
        sculpture is good enough: above ``count_target_iou`` every rewrite is
        scored purely on how many pieces it costs, so SRD spends the rest of
        the run shedding them, and it only pays for IoU when it has fallen
        back below the threshold. Contrast ``lambda_count``, which charges for
        pieces at all times and so keeps trading IoU away however good the fit
        already is.

        ``count_lambda`` is an exchange rate: it says how many pieces one unit
        of squared IoU deficit is worth. Because the penalty is quadratic its
        slope vanishes at the threshold, so a run settles slightly below the
        target; ``count_softplus_beta`` > 0 replaces the kink with a softplus
        of that sharpness, which is smooth but also leaks a little penalty
        above the threshold. ``count_power`` 1 keeps a constant slope right up
        to the threshold instead.
        """
        deficit = self.count_target_iou - float(mean_iou)
        beta = self.count_softplus_beta
        if beta > 0.0:
            # softplus(beta * deficit) / beta, in the overflow-safe form.
            scaled = beta * deficit
            hinge = max(scaled, 0.0) + float(np.log1p(np.exp(-abs(scaled))))
            hinge /= beta
        else:
            hinge = max(0.0, deficit)
        return float(n_patches) + self.count_lambda * hinge ** self.count_power

    def _mean_iou_from_renders(
        self,
        model,
        render1: torch.Tensor,
        render2: torch.Tensor,
    ) -> float:
        """Mean silhouette IoU of renders already in hand (no extra render)."""
        stats1 = model._silhouette_stats(render1.detach().cpu(), model.target1_mask)
        if model.target2_mask is None:
            return float(stats1["iou"])
        stats2 = model._silhouette_stats(render2.detach().cpu(), model.target2_mask)
        return float(stats1["iou"] + stats2["iou"]) / 2.0

    def _current_count_objective(self, model) -> float:
        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
        return self._count_objective_value(
            len(model.patches),
            self._mean_iou_from_renders(model, render1, render2),
        )

    def _current_model_loss(self, model) -> float:
        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            loss, _ = model._loss_from_renders(render1, render2, model.patches)
        return float(loss.detach().cpu())

    def _loss_without_patch(self, model, patch_index: int) -> float:
        remaining = [
            patch
            for idx, patch in enumerate(model.patches)
            if idx != patch_index
        ]
        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                remaining,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            loss, _ = model._loss_from_renders(render1, render2, remaining)
        return float(loss.detach().cpu())

    def _mandatory_delete_rewrites(self, model, current_step: int) -> list[RewriteCandidate]:
        """Find patches that must be deleted before stochastic SRD scoring."""
        if len(model.patches) <= self.min_patches:
            return []

        deletes: list[RewriteCandidate] = []
        available_deletions = max(0, len(model.patches) - self.min_patches)
        for idx, patch in enumerate(model.patches):
            if len(deletes) >= min(self.max_deletions, available_deletions):
                break
            if current_step - int(getattr(patch, "creation_step", 0)) < self.cooldown_steps:
                continue

            reason = self._mandatory_delete_reason(model, idx)
            if reason:
                deletes.append(RewriteCandidate(
                    kind="delete",
                    patch_index=idx,
                    improvement=0.0,
                    reason=reason,
                ))
        return deletes

    def _mandatory_delete_reason(self, model, patch_index: int) -> str:
        reasons: list[str] = []
        if self._patch_is_entirely_above_hanging_plane(model, patch_index):
            reasons.append("entirely above hanging plane")
        if self._patch_violates_rules(model, patch_index):
            reasons.append("rule violation")
        if not self._patch_contributes_to_either_image(model, patch_index):
            reasons.append("no image contribution")
        if not self._deleting_patch_changes_image(model, patch_index):
            reasons.append("delete has no image effect")
        return ", ".join(reasons)

    def _patch_is_entirely_above_hanging_plane(self, model, patch_index: int) -> bool:
        patch = model.patches[patch_index]
        plane_y = float(getattr(model, "hanging_plane_y", 3.5))
        n_per_segment = int(getattr(model.renderer, "n_per_segment", 20))
        with torch.no_grad():
            verts, _ = patch.extruded_mesh_world(n_per_segment=n_per_segment)
            if verts.numel() == 0:
                return False
            return bool(torch.all(verts[:, 1] > plane_y).item())

    def _patch_violates_rules(self, model, patch_index: int) -> bool:
        patch = model.patches[patch_index]
        params = [patch.center, patch.theta]
        for cp in patch.control_points:
            params.extend([cp.x, cp.y, cp.z, cp.handle_scale, cp.handle_rotation])
        if any(not torch.isfinite(param.detach()).all().item() for param in params):
            return True
        if any(abs(float(cp.z.detach().cpu())) > self.rule_violation_tol for cp in patch.control_points):
            return True
        if any(float(cp.handle_scale.detach().cpu()) <= 0.0 for cp in patch.control_points):
            return True
        return patch.is_self_intersecting()

    def _patch_contributes_to_either_image(self, model, patch_index: int) -> bool:
        patch = model.patches[patch_index]
        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                [patch],
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            alpha1 = render1[..., 3].amax() if render1.shape[-1] >= 4 else render1[..., :3].amax()
            alpha2 = render2[..., 3].amax() if render2.shape[-1] >= 4 else render2[..., :3].amax()
        return (
            float(alpha1.detach().cpu()) > self.no_contribution_alpha
            or float(alpha2.detach().cpu()) > self.no_contribution_alpha
        )

    def _deleting_patch_changes_image(self, model, patch_index: int) -> bool:
        with torch.no_grad():
            full1, full2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            remaining = [patch for idx, patch in enumerate(model.patches) if idx != patch_index]
            without1, without2 = model.renderer.render_both(
                remaining,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )
            delta = torch.maximum(
                (full1 - without1).abs().amax(),
                (full2 - without2).abs().amax(),
            )
        return float(delta.detach().cpu()) > self.no_effect_image_delta

    def evaluate_rewrite(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        rewrite: RewriteCandidate,
        current_loss: float,
    ) -> float:
        """Tentatively apply a rewrite, run local lookahead steps, then score it."""
        saved_patches, saved_optimizer_state = self._save_state(model, optimizer)
        try:
            self._apply_single(model, rewrite, current_step=0, tentative=True)
            model.optim = self._rebuild_optimizer(model, optimizer)

            n_steps = self._rewrite_eval_steps(rewrite)
            for _ in range(n_steps):
                valid_shape_states = model._capture_patch_shape_states()
                model.optim.zero_grad(set_to_none=True)
                render1, render2 = model.renderer.render_both(
                    model.patches,
                    model.camera1,
                    model.camera2,
                    model.render_resolutions,
                )
                loss, _ = model._loss_from_renders(render1, render2, model.patches)
                loss.backward()
                model.optim.step()
                model.optim.zero_grad(set_to_none=True)
                model._post_step_constraints(valid_shape_states, model.optim)

            with torch.no_grad():
                render1_new, render2_new = model.renderer.render_both(
                    model.patches,
                    model.camera1,
                    model.camera2,
                    model.render_resolutions,
                )
                new_loss, _ = model._loss_from_renders(render1_new, render2_new, model.patches)

            if self.count_objective:
                # Minimise pieces subject to the IoU threshold: score the
                # rewrite by how much it lowers J, not the image loss. A
                # deletion is worth a full piece whenever the survivors still
                # clear the threshold; an addition has to buy back more than
                # one piece worth of IoU deficit to be accepted.
                baseline = (
                    self._baseline_objective
                    if self._baseline_objective is not None
                    else self._count_objective_value(
                        len(model.patches) - rewrite.net_count_change,
                        self.count_target_iou,
                    )
                )
                new_objective = self._count_objective_value(
                    len(model.patches),
                    self._mean_iou_from_renders(model, render1_new, render2_new),
                )
                return baseline - new_objective

            raw_improvement = current_loss - float(new_loss.detach().cpu())
            # Piece-count regularization: SRD optimizes loss + lambda_count *
            # n_patches, so a rewrite must beat the loss by more than
            # lambda_count for every piece it adds, and is rebated lambda_count
            # for every piece it removes. At lambda_count = 0 this is the raw
            # loss reduction, and the acceptance gate (improvement > 0) is
            # unchanged.
            return raw_improvement - self.lambda_count * rewrite.net_count_change
        finally:
            self._restore_state(model, optimizer, saved_patches, saved_optimizer_state)

    def _rewrite_eval_steps(self, rewrite: RewriteCandidate) -> int:
        """Use a slightly longer local lookahead for growth rewrites."""
        if rewrite.kind in ("add", "split", "restart"):
            return max(1, self.rewrite_eval_steps)
        return 1

    def _sample_rewrites(
        self,
        model,
        current_step: int,
        uncovered_masks: tuple[np.ndarray, np.ndarray | None] | None = None,
    ) -> list[RewriteCandidate]:
        candidates: list[RewriteCandidate] = []
        add_budget = int(round(self.candidate_count * self.add_weight))
        delete_budget = int(round(self.candidate_count * self.delete_weight))
        split_budget = max(0, self.candidate_count - add_budget - delete_budget)
        if self.disable_splitting:
            add_budget += split_budget
            split_budget = 0

        if len(model.patches) >= self.max_patches:
            add_budget = 0
            split_budget = 0
            delete_budget = self.candidate_count
        if len(model.patches) <= self.min_patches:
            delete_budget = 0
            add_budget = self.candidate_count - split_budget

        for _ in range(add_budget):
            if (
                self.swept_volume is not None
                and not self.disable_swept_volume_adds
                and np.random.random() < self.swept_volume_spawn_fraction
            ):
                position = self._sample_guided_swept_volume_position(
                    model,
                    uncovered_masks,
                )
                if position is None:
                    continue
            else:
                position = np.random.uniform(
                    -self.scene_box_size * 0.5,
                    self.scene_box_size * 0.5,
                    size=3,
                ).astype(np.float32)
            candidates.append(RewriteCandidate(kind="add", position=position))

        eligible_delete_indices = [
            idx for idx, patch in enumerate(model.patches)
            if current_step - int(getattr(patch, "creation_step", 0)) >= self.cooldown_steps
            and (
                self.loss_only_deletion
                # Under the piece-count objective every piece has to be a
                # deletion candidate: the whole point is to shed pieces the
                # sculpture can spare, and those are usually full-size ones.
                or self.count_objective
                or float(patch.compute_area().detach().cpu()) <= self.min_patch_area
            )
        ]
        # Conflict-gated restart. Classify pieces by their per-view loss delta:
        # one that helps one view but hurts the other is offered as an atomic
        # "restart" (delete + respawn into the residual it vacated) rather than
        # a plain delete, which the lookahead would reject because the hole
        # outweighs the spill relief. This also lets full-size conflicting
        # pieces be reconsidered, which the size-gated eligibility above never
        # would; pieces bad in both views fall through to the plain-delete pool.
        restart_positions: dict[int, np.ndarray] = {}
        if (
            self.conflict_restart
            and delete_budget > 0
            and self.swept_volume is not None
            and not self.disable_swept_volume_adds
        ):
            classify_indices = [
                idx for idx, patch in enumerate(model.patches)
                if current_step - int(getattr(patch, "creation_step", 0)) >= self.cooldown_steps
            ]
            restart_positions = self._classify_conflict_deletes(model, classify_indices)
            # A conflicting piece is offered as a restart, not both a restart and
            # a plain delete.
            eligible_delete_indices = [
                idx for idx in eligible_delete_indices if idx not in restart_positions
            ]

        # Weight deletion candidates by the damage proxy when importance
        # sampling is on; computed once per call, then drawn with replacement
        # like the uniform path it replaces. Falls back to uniform if the proxy
        # has no spread.
        delete_weights = None
        if self.deletion_importance and delete_budget > 0 and eligible_delete_indices:
            delete_weights = self._deletion_proxy_weights(model, eligible_delete_indices)

        # Offer each detected conflict once, up to the delete budget, so the
        # interesting cases are always evaluated rather than crowded out by the
        # uniform/proxy draw; fill the rest of the budget with plain deletes.
        remaining_delete_budget = delete_budget
        for idx in list(restart_positions)[:delete_budget]:
            candidates.append(RewriteCandidate(
                kind="restart", patch_index=idx, position=restart_positions[idx],
            ))
            remaining_delete_budget -= 1
        for _ in range(remaining_delete_budget):
            if not eligible_delete_indices:
                break
            if delete_weights is not None:
                idx = int(np.random.choice(eligible_delete_indices, p=delete_weights))
            else:
                idx = int(np.random.choice(eligible_delete_indices))
            candidates.append(RewriteCandidate(kind="delete", patch_index=idx))

        eligible_split_indices = [
            idx for idx, patch in enumerate(model.patches)
            if current_step - int(getattr(patch, "creation_step", 0)) >= self.cooldown_steps
        ]
        for _ in range(split_budget):
            if not eligible_split_indices or len(model.patches) + 1 > self.max_patches:
                break
            idx = self._sample_split_index(model, eligible_split_indices)
            candidates.append(RewriteCandidate(kind="split", patch_index=idx))

        np.random.shuffle(candidates)
        return candidates[:self.candidate_count]

    def _uncovered_target_masks(self, model) -> tuple[np.ndarray, np.ndarray | None] | None:
        if self.swept_volume is None or self.disable_swept_volume_adds:
            return None

        with torch.no_grad():
            render1, render2 = model.renderer.render_both(
                model.patches,
                model.camera1,
                model.camera2,
                model.render_resolutions,
            )

        target1 = model.target1_mask.detach().cpu().numpy().squeeze() > 0.5
        alpha1 = self._render_alpha_mask(render1)
        uncovered1 = target1 & ~alpha1

        uncovered2 = None
        if model.target2_mask is not None:
            target2 = model.target2_mask.detach().cpu().numpy().squeeze() > 0.5
            alpha2 = self._render_alpha_mask(render2)
            uncovered2 = target2 & ~alpha2

        return uncovered1, uncovered2

    @staticmethod
    def _render_alpha_mask(render: torch.Tensor, threshold: float = 0.05) -> np.ndarray:
        image = render.detach().cpu()
        if image.shape[-1] >= 4:
            alpha = image[..., 3]
        else:
            alpha = image[..., :3].amax(dim=-1)
        return alpha.numpy() > threshold

    def _sample_guided_swept_volume_position(
        self,
        model,
        uncovered_masks: tuple[np.ndarray, np.ndarray | None] | None,
        max_attempts: int = 128,
    ) -> np.ndarray | None:
        if self.swept_volume is None:
            return None
        if uncovered_masks is None or not self._has_uncovered_cells(uncovered_masks):
            return self._sample_swept_volume_position()

        best_position: np.ndarray | None = None
        best_score = -np.inf
        for _ in range(max(1, int(max_attempts))):
            position = self._sample_swept_volume_position()
            if not self._point_hits_uncovered(model, position, uncovered_masks):
                continue
            score = float(self.swept_volume.score_points(position[None, :])[0])
            if score > best_score:
                best_score = score
                best_position = position

        return best_position

    @staticmethod
    def _has_uncovered_cells(uncovered_masks: tuple[np.ndarray, np.ndarray | None]) -> bool:
        uncovered1, uncovered2 = uncovered_masks
        return bool(np.any(uncovered1) or (uncovered2 is not None and np.any(uncovered2)))

    def _point_hits_uncovered(
        self,
        model,
        point: np.ndarray,
        uncovered_masks: tuple[np.ndarray, np.ndarray | None],
    ) -> bool:
        uncovered1, uncovered2 = uncovered_masks
        pts = np.asarray(point, dtype=np.float32).reshape((1, 3))

        x1, y1, valid1 = _project_points(pts, model.camera1, uncovered1.shape)
        hit1 = bool(valid1[0] and uncovered1[y1[0], x1[0]])

        hit2 = False
        if uncovered2 is not None:
            x2, y2, valid2 = _project_points(pts, model.camera2, uncovered2.shape)
            hit2 = bool(valid2[0] and uncovered2[y2[0], x2[0]])

        return hit1 or hit2

    def _sample_swept_volume_position(self) -> np.ndarray:
        if self.swept_volume is None:
            raise ValueError("Swept-volume sampling requires a swept volume.")
        n_points = len(self.swept_volume.points)
        if n_points <= 0:
            raise ValueError("Cannot sample from an empty swept volume.")
        if (
            len(self._swept_point_order) != n_points
            or self._swept_point_cursor >= n_points
        ):
            self._swept_point_order = np.random.permutation(n_points)
            self._swept_point_cursor = 0
        point_index = int(self._swept_point_order[self._swept_point_cursor])
        self._swept_point_cursor += 1
        return self.swept_volume.sample_point_at_index(point_index)

    def _sample_split_index(self, model, eligible_indices: Sequence[int]) -> int:
        """Sample split candidates with larger patches more likely."""
        areas = np.array([
            max(1e-8, float(model.patches[idx].compute_area().detach().cpu()))
            for idx in eligible_indices
        ], dtype=np.float64)
        weights = areas / areas.sum()
        return int(np.random.choice(list(eligible_indices), p=weights))

    def _deletion_proxy_weights(
        self, model, eligible_indices: Sequence[int]
    ) -> np.ndarray | None:
        """Softmax weights over delete candidates, high for cheap-to-delete pieces.

        The proxy is ``area * score``, where ``score`` is ``spill_fraction``
        under ``deletion_proxy='spill'`` and ``spill_fraction -
        coverage_fraction`` under ``'net'``. Each eligible piece is rendered in
        isolation; ``spill_fraction`` is the share of its own silhouette that
        lands in negative space (outside the target, summed over the two views),
        ``coverage_fraction`` the share that lands on it, and ``area`` its
        world-space size. A large piece that mostly paints
        where nothing should be scores high and is offered for deletion far more
        often; a piece sitting squarely on target scores ~0 and is rarely
        offered. Weights are a softmax at ``deletion_temperature`` -- never an
        argmax -- so the sampler stays "mostly-good, occasionally-exploratory"
        rather than collapsing into greedy pruning, which would kill the
        stochastic diversity SRD needs to reconsider structure.

        The proxy is per-piece and ignores that damage is coupled: two
        overlapping pieces can each look low-damage alone yet jointly cover a
        region. That is the same blind spot the greedy compatibility + lookahead
        pass already carries, so this does not make it worse.

        The proxy is standardized by its own spread before the softmax, so
        ``deletion_temperature`` is dimensionless (one unit = one std of proxy)
        and independent of the world-space area scale. Returns ``None`` when the
        proxy carries no spread (e.g. nothing spills), which the caller reads as
        "fall back to uniform".
        """
        threshold = 0.05
        target1 = model.target1_mask.detach().cpu().numpy().squeeze() > 0.5
        target2 = (
            model.target2_mask.detach().cpu().numpy().squeeze() > 0.5
            if model.target2_mask is not None else None
        )

        proxies = np.empty(len(eligible_indices), dtype=np.float64)
        with torch.no_grad():
            for i, idx in enumerate(eligible_indices):
                patch = model.patches[idx]
                render1, render2 = model.renderer.render_both(
                    [patch], model.camera1, model.camera2, model.render_resolutions
                )
                alpha1 = self._render_alpha_mask(render1, threshold)
                painted = float(alpha1.sum())
                spill = float((alpha1 & ~target1).sum())
                coverage = float((alpha1 & target1).sum())
                if target2 is not None:
                    alpha2 = self._render_alpha_mask(render2, threshold)
                    painted += float(alpha2.sum())
                    spill += float((alpha2 & ~target2).sum())
                    coverage += float((alpha2 & target2).sum())
                spill_fraction = spill / painted if painted > 0 else 0.0
                # 'spill' scores a piece by wasted paint alone; 'net' subtracts
                # the on-target share, so a piece is cheap to delete only when it
                # spills *more* than it covers. Both are fractions of the piece's
                # own footprint, so spill_fraction + coverage_fraction == 1 and
                # net == 2*spill_fraction - 1 -- but multiplied by area they rank
                # pieces differently, which is the point.
                coverage_fraction = coverage / painted if painted > 0 else 0.0
                score = (
                    spill_fraction - coverage_fraction
                    if self.deletion_proxy == "net" else spill_fraction
                )
                area = max(0.0, float(patch.compute_area().detach().cpu()))
                proxies[i] = area * score

        spread = float(proxies.std())
        if not np.isfinite(spread) or spread <= 0.0:
            return None
        scaled = (proxies - proxies.max()) / (self.deletion_temperature * spread)
        weights = np.exp(scaled)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None
        return weights / total

    def _classify_conflict_deletes(
        self, model, indices: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Map each conflicting piece to a swept-volume respawn position.

        A piece *conflicts* when deleting it moves the two views' losses in
        opposite directions -- one view improves (the piece was spilling there)
        while the other worsens (the piece was covering there, so a hole opens).
        With the per-view delta

            d_v = L_v(without piece) - L_v(with piece),

        the piece conflicts iff ``max_v d_v > eps and min_v d_v < -eps``. For
        each such piece the respawn position is drawn from the swept volume over
        the target cells it *newly* uncovers, so the replacement starts in the
        residual it vacated. Pieces bad in both views (both deltas <= -eps) fall
        through to a plain delete; pieces good in both (both deltas >= eps) are
        not deletion candidates at all. A conflicting piece with no swept-volume
        point over its hole is omitted (it falls back to a plain delete).

        Returns ``{patch_index: respawn_position}`` for the conflicting pieces
        that have a valid respawn; per-view losses come from the model's own
        ``_loss_from_renders`` components, so the gate is in the same units and
        with the same weights as the acceptance test.
        """
        respawns: dict[int, np.ndarray] = {}
        if len(indices) == 0 or len(model.patches) <= 1:
            return respawns

        with torch.no_grad():
            full1, full2 = model.renderer.render_both(
                model.patches, model.camera1, model.camera2, model.render_resolutions
            )
            _, comp_full = model._loss_from_renders(full1, full2, model.patches)
        loss1_full = float(comp_full["loss1"].detach().cpu())
        loss2_full = float(comp_full["loss2"].detach().cpu())

        target1 = model.target1_mask.detach().cpu().numpy().squeeze() > 0.5
        uncovered1_before = target1 & ~self._render_alpha_mask(full1)
        target2 = None
        uncovered2_before = None
        if model.target2_mask is not None:
            target2 = model.target2_mask.detach().cpu().numpy().squeeze() > 0.5
            uncovered2_before = target2 & ~self._render_alpha_mask(full2)

        eps = self.conflict_eps
        for idx in indices:
            remaining = [p for i, p in enumerate(model.patches) if i != idx]
            if not remaining:
                continue
            with torch.no_grad():
                without1, without2 = model.renderer.render_both(
                    remaining, model.camera1, model.camera2, model.render_resolutions
                )
                _, comp_without = model._loss_from_renders(without1, without2, remaining)
            d1 = float(comp_without["loss1"].detach().cpu()) - loss1_full
            d2 = float(comp_without["loss2"].detach().cpu()) - loss2_full
            if not (max(d1, d2) > eps and min(d1, d2) < -eps):
                continue  # bad-in-both -> plain delete; good-in-both -> keep

            # The residual this piece vacated: target cells uncovered only after
            # its removal, in whichever view(s) it was carrying.
            hole1 = (target1 & ~self._render_alpha_mask(without1)) & ~uncovered1_before
            hole2 = None
            if target2 is not None:
                hole2 = (target2 & ~self._render_alpha_mask(without2)) & ~uncovered2_before
            if not (np.any(hole1) or (hole2 is not None and np.any(hole2))):
                continue
            position = self._sample_guided_swept_volume_position(model, (hole1, hole2))
            if position is not None:
                respawns[idx] = position
        return respawns

    def _select_compatible(self, candidates: Sequence[RewriteCandidate]) -> list[RewriteCandidate]:
        accepted: list[RewriteCandidate] = []
        touched_indices: set[int] = set()
        additions = 0
        deletions = 0

        for candidate in sorted(candidates, key=lambda c: c.improvement, reverse=True):
            if candidate.kind == "restart":
                # A restart both deletes and adds, so it must fit under both
                # budgets and, like a delete, claims the piece index it removes.
                if candidate.patch_index is None or candidate.patch_index in touched_indices:
                    continue
                if deletions >= self.max_deletions or additions >= self.max_additions:
                    continue
                touched_indices.add(candidate.patch_index)
                deletions += 1
                additions += 1
                accepted.append(candidate)
                continue

            if candidate.kind == "delete":
                if candidate.patch_index is None or candidate.patch_index in touched_indices:
                    continue
                if deletions >= self.max_deletions:
                    continue
                touched_indices.add(candidate.patch_index)
                deletions += 1
                accepted.append(candidate)
                continue

            if candidate.kind == "split":
                if candidate.patch_index is None or candidate.patch_index in touched_indices:
                    continue
                if additions >= self.max_additions:
                    continue
                touched_indices.add(candidate.patch_index)
                additions += 1
                accepted.append(candidate)
                continue

            if additions >= self.max_additions:
                continue
            additions += 1
            accepted.append(candidate)

        return accepted

    def _apply_rewrites(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        rewrites: Sequence[RewriteCandidate],
        current_step: int,
    ) -> None:
        if not rewrites:
            return

        # Restart, like delete and split, is keyed by a patch index and must be
        # applied high-index-first so earlier pops do not shift later indices;
        # its respawn appends to the end, which never disturbs a lower index.
        indexed_rewrites = [r for r in rewrites if r.kind in ("delete", "split", "restart")]
        for rewrite in sorted(indexed_rewrites, key=lambda r: r.patch_index or 0, reverse=True):
            self._apply_single(model, rewrite, current_step=current_step, tentative=False)
        for rewrite in [r for r in rewrites if r.kind not in ("delete", "split", "restart")]:
            self._apply_single(model, rewrite, current_step=current_step, tentative=False)

        model.optim = self._rebuild_optimizer(model, optimizer)
        model._post_step_constraints()

    def _apply_single(
        self,
        model,
        rewrite: RewriteCandidate,
        *,
        current_step: int,
        tentative: bool,
    ) -> None:
        if rewrite.kind == "delete":
            if rewrite.patch_index is None or rewrite.patch_index >= len(model.patches):
                return
            patch = model.patches.pop(rewrite.patch_index)
            rewrite.applied_index = rewrite.patch_index
            if not tentative:
                self.stats.deleted += 1
                self.stats.total_deleted += 1
                pos = patch.center.detach().cpu().numpy()
                print(
                    f"[SRD rewrite] deleted patch={rewrite.patch_index}, "
                    f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                    f"reason={rewrite.reason or 'accepted rewrite'}"
                )
            return

        if rewrite.kind == "restart":
            # Atomic delete-and-respawn: remove the conflicting piece, then drop
            # a fresh piece into the residual it vacated (position sampled from
            # the swept volume over its hole). Net piece count is unchanged.
            if rewrite.patch_index is None or rewrite.patch_index >= len(model.patches):
                return
            old_patch = model.patches.pop(rewrite.patch_index)
            rewrite.applied_index = rewrite.patch_index
            respawned = False
            if rewrite.position is not None and len(model.patches) < self.max_patches:
                new_patch = _small_default_patch(
                    rewrite.position,
                    model.device,
                    [1.0, 1.0, 1.0],
                    current_step,
                    label=f"patch_{len(model.patches):04d}",
                )
                model.patches.append(new_patch)
                respawned = True
            if not tentative:
                self.stats.restarts += 1
                self.stats.total_restarts += 1
                pos = old_patch.center.detach().cpu().numpy()
                print(
                    f"[SRD rewrite] restart patch={rewrite.patch_index} "
                    f"(conflicting views) at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                    f"respawned={respawned}, improvement={rewrite.improvement:.6f}"
                )
            return

        if rewrite.kind == "split":
            if rewrite.patch_index is None or rewrite.patch_index >= len(model.patches):
                return
            if len(model.patches) + 1 > self.max_patches:
                return
            patch = model.patches.pop(rewrite.patch_index)
            child_a, child_b = patch.split_down_middle(creation_step=current_step)
            if child_a.is_self_intersecting() or child_b.is_self_intersecting():
                model.patches.insert(rewrite.patch_index, patch)
                return
            model.patches.extend([child_a, child_b])
            rewrite.applied_index = rewrite.patch_index
            if not tentative:
                self.stats.added += 1
                self.stats.total_splits += 1
                print(
                    f"[SRD rewrite] split patch={rewrite.patch_index}, "
                    f"children={len(model.patches) - 2},{len(model.patches) - 1}, "
                    f"improvement={rewrite.improvement:.6f}"
                )
            return

        if rewrite.position is None:
            return
        if len(model.patches) >= self.max_patches:
            return
        palette_color = [1.0, 1.0, 1.0]
        patch = _small_default_patch(
            rewrite.position,
            model.device,
            palette_color,
            current_step,
            label=f"patch_{len(model.patches):04d}",
        )
        model.patches.append(patch)
        rewrite.applied_index = len(model.patches) - 1
        if not tentative:
            self.stats.added += 1
            self.stats.total_adds += 1
            pos = patch.center.detach().cpu().numpy()
            print(
                f"[SRD rewrite] added patch={rewrite.applied_index}, "
                f"position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                f"improvement={rewrite.improvement:.6f}"
            )

    def _save_state(self, model, optimizer: torch.optim.Optimizer) -> tuple[list[dict], dict]:
        return [patch.to_dict() for patch in model.patches], copy.deepcopy(optimizer.state_dict())

    def _restore_state(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        patch_states: Sequence[dict],
        optimizer_state: dict,
    ) -> None:
        model.patches[:] = [
            Patch.from_dict(copy.deepcopy(state), device=model.device)
            for state in patch_states
        ]
        model.optim = self._rebuild_optimizer(model, optimizer)
        try:
            model.optim.load_state_dict(optimizer_state)
        except ValueError:
            pass
        model._post_step_constraints()

    def _rebuild_optimizer(self, model, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        defaults = optimizer.defaults.copy()
        return optimizer.__class__(_patch_parameters(model.patches), **defaults)
