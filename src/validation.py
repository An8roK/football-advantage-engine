"""Strict 5-Fold Stratified Out-of-Fold (OOF) Quantitative Validation Audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES: list[str] = [
    "iso_c_t0",
    "iso_c_t1",
    "iso_delta_c",
    "nearest_def_dist",
    "ball_x_start",
    "ball_y_start",
]

BIOMECH_EXTRA_FEATURES: list[str] = [
    "aniso_c_t0",
    "aniso_c_t1",
    "aniso_delta_c",
    "aniso_minus_iso_delta",
    "delta_c_pos",
    "delta_c_orient",
    "delta_c_matrix",
    "nearest_def_pivot_deg",
    "nearest_def_back_turned",
    "nearest_def_weak_stance",
    "carrier_abs_twist_deg",
    "a_pass",
    "a_recv",
    "net_matchup_edge",
    "weak_foot_comfort",
    "lambda2_t0",
    "lambda2_delta",
    "hole_score_t0",
    "hole_score_delta",
]

FULL_FEATURES: list[str] = BASELINE_FEATURES + BIOMECH_EXTRA_FEATURES


def _build_classifier_pipeline() -> Pipeline:
    """Fold-local preprocessing + L2-regularized logistic classifier (zero leakage)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.15, max_iter=1000, solver="lbfgs")),
        ]
    )


def run_oof_cv(
    df: pd.DataFrame,
    target_series: pd.Series,
    feature_cols: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run strict Stratified K-Fold OOF evaluation with fold-local imputation and scaling."""
    y = target_series.astype(int).to_numpy()
    n_samples = int(len(y))
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    min_class = min(n_pos, n_neg)
    if n_samples < 10 or min_class < 2:
        return {
            "status": "insufficient_samples",
            "n_samples": n_samples,
            "n_positives": n_pos,
            "n_splits_used": 0,
            "roc_auc": None,
            "brier": None,
            "fold_splits": [],
        }

    eff_splits = min(n_splits, min_class)
    avail_cols = [c for c in feature_cols if c in df.columns]
    x_df = df[avail_cols].apply(pd.to_numeric, errors="coerce")

    # Drop all-NaN columns before fitting so SimpleImputer has valid statistics per fold
    valid_cols = [c for c in avail_cols if x_df[c].notna().any()]
    x_mat = x_df[valid_cols].to_numpy(dtype=float)

    skf = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(n_samples, dtype=float)
    fold_splits: list[tuple[list[int], list[int]]] = []

    for train_idx, test_idx in skf.split(x_mat, y):
        pipe = _build_classifier_pipeline()
        pipe.fit(x_mat[train_idx], y[train_idx])
        oof_probs[test_idx] = pipe.predict_proba(x_mat[test_idx])[:, 1]
        fold_splits.append((train_idx.tolist(), test_idx.tolist()))

    auc = float(roc_auc_score(y, oof_probs))
    brier = float(brier_score_loss(y, oof_probs))

    return {
        "status": "ok",
        "n_samples": n_samples,
        "n_positives": n_pos,
        "base_rate": round(n_pos / n_samples, 4),
        "n_splits_used": eff_splits,
        "roc_auc": round(auc, 4),
        "brier": round(brier, 4),
        "oof_probs": oof_probs,
        "fold_splits": fold_splits,
    }


def _evaluate_task(
    sub_df: pd.DataFrame,
    target: pd.Series,
    task_key: str,
    task_label: str,
) -> dict[str, Any]:
    """Compare Baseline (Isotropic) vs. Full (+ Biomechanics) on All Events and 3d_pose Only."""
    results: dict[str, Any] = {"task_key": task_key, "task_label": task_label, "slices": {}}

    for slice_key, mask in [
        ("all_events", pd.Series(True, index=sub_df.index)),
        ("3d_pose_only", sub_df["pose_source"] == "3d_pose"),
    ]:
        df_slice = sub_df[mask].reset_index(drop=True)
        y_slice = target[mask].reset_index(drop=True)

        base = run_oof_cv(df_slice, y_slice, BASELINE_FEATURES)
        full = run_oof_cv(df_slice, y_slice, FULL_FEATURES)

        # Strip raw probability arrays/indices for compact JSON artifact output
        def _clean(res: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in res.items() if k not in ("oof_probs", "fold_splits")}

        b_clean = _clean(base)
        f_clean = _clean(full)

        delta_auc = (
            round(f_clean["roc_auc"] - b_clean["roc_auc"], 4)
            if (f_clean["roc_auc"] is not None and b_clean["roc_auc"] is not None)
            else None
        )
        delta_brier = (
            round(f_clean["brier"] - b_clean["brier"], 4)
            if (f_clean["brier"] is not None and b_clean["brier"] is not None)
            else None
        )

        results["slices"][slice_key] = {
            "baseline": b_clean,
            "biomechanics": f_clean,
            "delta_auc": delta_auc,
            "delta_brier": delta_brier,
        }

    return results


def run_validation_audit(events_df: pd.DataFrame) -> dict[str, Any]:
    """Run the complete Tab 4 OOF validation suite across all real SkillCorner targets."""
    tasks: list[dict[str, Any]] = []

    # (i) Defender beaten_by_possession / beaten_by_movement
    eng = events_df[events_df["event_type"] == "on_ball_engagement"].copy()
    if not eng.empty:
        y_beaten = eng["target_defender_beaten"].fillna(0).astype(int)
        tasks.append(
            _evaluate_task(
                eng,
                y_beaten,
                task_key="defender_beaten",
                task_label="Defender Beaten (beaten_by_possession / movement)",
            )
        )

    # (ii-a) Pass Completion & (ii-b) Line-Break Success
    passes = events_df[
        (events_df["event_type"] == "player_possession")
        & (events_df["end_type"] == "pass")
    ].copy()
    if not passes.empty:
        y_pass = passes["target_pass_completed"].fillna(0).astype(int)
        tasks.append(
            _evaluate_task(
                passes,
                y_pass,
                task_key="pass_completion",
                task_label="Pass Completion (pass_outcome == successful)",
            )
        )
        y_lb = passes["target_line_break_success"].fillna(0).astype(int)
        tasks.append(
            _evaluate_task(
                passes,
                y_lb,
                task_key="line_break_success",
                task_label="Line-Break Pass Success (line_break & completed)",
            )
        )

    # (iii) High-EPV / Shot-Creation possessions
    poss = events_df[events_df["event_type"] == "player_possession"].copy()
    if not poss.empty:
        y_epv = poss["target_high_epv_or_shot"].fillna(0).astype(int)
        tasks.append(
            _evaluate_task(
                poss,
                y_epv,
                task_key="high_epv_or_shot",
                task_label="High-EPV / Shot-Creation Possession",
            )
        )

    n_total = int(len(events_df))
    n_3d = int((events_df["pose_source"] == "3d_pose").sum()) if n_total else 0
    return {
        "cv_protocol": "5-Fold Stratified OOF (fold-local MedianImputer + StandardScaler + LogisticRegression C=0.15)",
        "baseline_features": BASELINE_FEATURES,
        "biomech_extra_features": BIOMECH_EXTRA_FEATURES,
        "total_scored_events": n_total,
        "pose_3d_events": n_3d,
        "kinematic_fallback_events": n_total - n_3d,
        "pose_3d_pct": round(100.0 * n_3d / n_total, 2) if n_total else 0.0,
        "tasks": tasks,
    }
