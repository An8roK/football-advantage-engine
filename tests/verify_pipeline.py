"""Automated invariant verification suite for Project 4 (football-advantage-engine).

Verifies all 6 core architectural invariants:
  1. 3D forward-vector sign (lHip -> rHip non-inverted) and stance classification
  2. Frame-sync formula (pose_frame = round(2.5 * tracking_frame))
  3. Back-turned & weak-stance defender latency strictly reduces defender control
  4. Delaunay Fiedler connectivity λ₂ drops and Hole Score H_Δ rises when a defender is dragged out of line
  5. Exact additivity of the 3-way ΔC decomposition (|pos + orient + matrix - total| < 1e-9)
  6. OOF validation leakage isolation (zero train/test overlap & clean 3d_pose slicing)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matrix import analyze_defensive_matrix
from src.physics import (
    compute_pitch_control,
    compute_player_arrival_time,
    decompose_action_control_delta,
    get_pitch_grid,
    get_threat_surface,
)
from src.pose import (
    classify_stance,
    compute_forward_from_lr,
    compute_signed_twist,
    tracking_to_pose_frame,
)
from src.simulator import get_preset_scenarios, run_simulation
from src.validation import BASELINE_FEATURES, FULL_FEATURES, run_oof_cv, run_validation_audit


def test_1_forward_vector_sign_and_stance() -> None:
    """(1) Verify lHip -> rHip produces non-inverted forward vector and accurate stance labels."""
    # Player facing +y (North): lHip at (-0.15, 0), rHip at (+0.15, 0), nose at (0, +0.20)
    fwd_north = compute_forward_from_lr((-0.15, 0.0), (0.15, 0.0))
    assert fwd_north is not None
    assert np.allclose(fwd_north, [0.0, 1.0], atol=1e-6), f"Expected +y, got {fwd_north}"
    assert float(np.dot(fwd_north, [0.0, 0.20])) > 0.0, "Facing inverted vs nose!"

    # Player facing +x (East): lHip at (0, +0.15), rHip at (0, -0.15), nose at (+0.20, 0)
    fwd_east = compute_forward_from_lr((0.0, 0.15), (0.0, -0.15))
    assert fwd_east is not None
    assert np.allclose(fwd_east, [1.0, 0.0], atol=1e-6), f"Expected +x, got {fwd_east}"

    # Signed torso twist: shoulders rotated +30 deg CCW relative to hips
    sh_30 = np.array([math.cos(math.radians(30.0)), math.sin(math.radians(30.0))])
    twist = compute_signed_twist(fwd_east, sh_30)
    assert abs(math.degrees(twist) - 30.0) < 1e-6

    # Stance classification around +/- 0.10m threshold
    st_left, s_left = classify_stance((0.30, 0.1), (0.05, -0.1), fwd_east)
    assert st_left == "Left-Forward" and s_left > 0.10

    st_right, s_right = classify_stance((0.0, 0.1), (0.28, -0.1), fwd_east)
    assert st_right == "Right-Forward" and s_right < -0.10

    st_sq, s_sq = classify_stance((0.04, 0.1), (0.0, -0.1), fwd_east)
    assert st_sq == "Square" and abs(s_sq) <= 0.10


def test_2_frame_sync_formula() -> None:
    """(2) Verify exact 25 FPS <-> 10 FPS frame-sync mapping."""
    for tf in (0, 10, 100, 30240, 49918):
        pf = tracking_to_pose_frame(tf)
        assert pf == int(2.5 * tf), f"Even tracking frame {tf} mismatch: {pf}"

    # Phase 406 endpoints verified against MANIFEST.json (49795..49918 -> 124487..124795)
    assert abs(tracking_to_pose_frame(49795) - 124487) <= 1
    assert tracking_to_pose_frame(49918) == 124795


def test_3_biomechanical_penalty_monotonicity() -> None:
    """(3) Back-turned (>110 deg) and weak-stance defenders strictly lose control vs square forward-facing."""
    attacker = {
        "player_id": 1, "is_attacking": True, "is_gk": False,
        "x": 34.0, "y": 4.0, "vx": 1.0, "vy": 0.0,
        "hip_fwd": np.array([1.0, 0.0]), "torso_twist": 0.0,
        "stance": "Square", "stance_lead_m": 0.0,
    }
    # Defender at (30, 4) facing +x toward attacker & high-threat box
    base_def = {
        "player_id": 2, "is_attacking": False, "is_gk": False,
        "x": 30.0, "y": 4.0, "vx": 0.0, "vy": 0.0,
        "hip_fwd": np.array([1.0, 0.0]), "torso_twist": 0.0,
        "stance": "Square", "stance_lead_m": 0.0,
    }
    # Back-turned (~160 deg away from attacker & high-threat box)
    back_turned_def = {**base_def, "hip_fwd": np.array([-0.94, 0.34])}
    # Facing +x: anatomical RIGHT is -y -> Left-Forward locks right hip behind on -y reaches
    weak_stance_def = {**base_def, "stance": "Left-Forward", "stance_lead_m": 0.35}

    t_square = compute_player_arrival_time(base_def, anisotropic=True)
    t_back = compute_player_arrival_time(back_turned_def, anisotropic=True)
    t_weak = compute_player_arrival_time(weak_stance_def, anisotropic=True)

    _, _, xx, yy, _ = get_pitch_grid()
    idx_att = int(np.argmin((xx - 34.0) ** 2 + (yy - 4.0) ** 2))
    idx_right = int(np.argmin((xx - 32.0) ** 2 + (yy - (-2.0)) ** 2))

    assert t_back.ravel()[idx_att] > t_square.ravel()[idx_att] + 0.5, "Back-turned defender must pay turn + penalty latency"
    assert t_weak.ravel()[idx_right] > t_square.ravel()[idx_right] + 0.08, "Weak-stance lateral reach must pay stance latency"

    c_grid_sq, c_att_vs_sq = compute_pitch_control([attacker, base_def], anisotropic=True)
    c_grid_back, c_att_vs_back = compute_pitch_control([attacker, back_turned_def], anisotropic=True)
    c_grid_weak, c_att_vs_weak = compute_pitch_control([attacker, weak_stance_def], anisotropic=True)

    assert c_grid_back.ravel()[idx_att] > c_grid_sq.ravel()[idx_att], "Back-turned defender must lose local control"
    assert c_grid_weak.ravel()[idx_right] > c_grid_sq.ravel()[idx_right], "Weak-stance defender must lose weak-side control"
    assert c_att_vs_back > c_att_vs_sq, f"Expected higher attacking threat control vs back-turned: {c_att_vs_back} <= {c_att_vs_sq}"
    assert c_att_vs_weak > c_att_vs_sq, f"Expected higher attacking threat control vs weak-stance: {c_att_vs_weak} <= {c_att_vs_sq}"


def test_4_delaunay_fiedler_and_hole_sensitivity() -> None:
    """(4) Dragging a defender out of line & turning hips strictly drops λ₂ and raises Hole Score H_Δ."""
    _, _, _, _, grid_xy = get_pitch_grid()
    threat = get_threat_surface()
    att_ctrl = np.full(threat.shape, 0.65, dtype=float)

    # Compact 6-defender mid-block aligned toward ball (-x)
    compact_xy = np.array([
        [34.0, 14.0], [35.0, 5.0], [35.0, -5.0], [34.0, -14.0],
        [25.0, 6.0], [25.0, -6.0],
    ], dtype=float)
    compact_fwd = np.tile(np.array([[-1.0, 0.0]]), (6, 1))

    # Feint drags RDM (#4) far out toward wing and twists hips 80 deg away
    dragged_xy = compact_xy.copy()
    dragged_xy[4] = [14.0, 22.0]
    dragged_fwd = compact_fwd.copy()
    dragged_fwd[4] = [0.17, 0.98]

    m0 = analyze_defensive_matrix(compact_xy, compact_fwd, att_ctrl, threat, grid_xy)
    m1 = analyze_defensive_matrix(dragged_xy, dragged_fwd, att_ctrl, threat, grid_xy)

    assert m1["lambda2"] < m0["lambda2"], f"Expected λ₂ drop: {m1['lambda2']} vs {m0['lambda2']}"
    assert m1["hole_score"] > m0["hole_score"], f"Expected Hole Score rise: {m1['hole_score']} vs {m0['hole_score']}"


def test_5_exact_3way_delta_c_additivity() -> None:
    """(5) Exact additivity of 3-way counterfactual ΔC decomposition across all presets & actions."""
    scenarios = get_preset_scenarios()
    for name, scen in scenarios.items():
        for action in ("Body Feint", "Directional First Touch", "Driven / Disguised Pass"):
            rollout = run_simulation(scen, action=action, n_steps=12)
            fr_end = rollout["frames"][-1]
            d = fr_end["decomp"]
            recon = d["delta_c_pos"] + d["delta_c_orient"] + d["delta_c_matrix"]
            err = abs(recon - d["delta_c_total"])
            assert err < 1e-9, f"Non-additive ΔC in {name}/{action}: err={err}"

            # Also test standalone decompose_action_control_delta
            d2 = decompose_action_control_delta(rollout["frames"][0]["players"], fr_end["players"])
            assert abs((d2["delta_c_pos"] + d2["delta_c_orient"] + d2["delta_c_matrix"]) - d2["delta_c_total"]) < 1e-9


def test_6_oof_validation_leakage_isolation() -> None:
    """(6) Verify zero train/test fold overlap, OOF completeness, and strict 3d_pose slice isolation."""
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({col: rng.normal(0.0, 1.0, size=n) for col in FULL_FEATURES})
    # Inject NaNs and extreme scale differences to verify fold-internal imputation/scaling
    df.loc[rng.choice(n, size=10, replace=False), "a_pass"] = np.nan
    df["nearest_def_dist"] = df["nearest_def_dist"] * 500.0 + 1000.0

    # Construct synthetic signal correlated with biomechanics
    logits = 1.2 * df["delta_c_orient"] + 0.9 * df["net_matchup_edge"] - 0.8 * df["lambda2_delta"]
    y = pd.Series((logits + rng.normal(0.0, 0.8, size=n) > 0.0).astype(int))

    res = run_oof_cv(df, y, FULL_FEATURES, n_splits=5)
    assert res["status"] == "ok"
    assert res["n_splits_used"] == 5
    assert res["roc_auc"] > 0.70

    seen_test: list[int] = []
    for train_idx, test_idx in res["fold_splits"]:
        assert set(train_idx).isdisjoint(set(test_idx)), "Train/test leakage detected!"
        seen_test.extend(test_idx)
    assert sorted(seen_test) == list(range(n)), "OOF predictions must cover every row exactly once!"

    # Verify run_validation_audit isolates pose_source == '3d_pose' cleanly
    df["event_type"] = np.where(np.arange(n) < 60, "player_possession", "on_ball_engagement")
    df["end_type"] = "pass"
    df["pose_source"] = np.where(np.arange(n) % 2 == 0, "3d_pose", "kinematic_fallback")
    df["target_defender_beaten"] = y
    df["target_pass_completed"] = y
    df["target_line_break_success"] = y
    df["target_high_epv_or_shot"] = y

    audit = run_validation_audit(df)
    assert audit["total_scored_events"] == n
    assert audit["pose_3d_events"] == n // 2
    for t in audit["tasks"]:
        assert t["slices"]["3d_pose_only"]["baseline"]["n_samples"] == t["slices"]["all_events"]["baseline"]["n_samples"] // 2


def main() -> int:
    tests = [
        ("1. 3D Forward-Vector Sign & Stance", test_1_forward_vector_sign_and_stance),
        ("2. Exact 25<->10 FPS Frame Sync", test_2_frame_sync_formula),
        ("3. Back-Turned & Weak-Stance Monotonicity", test_3_biomechanical_penalty_monotonicity),
        ("4. Delaunay Fiedler λ₂ & Hole Score Sensitivity", test_4_delaunay_fiedler_and_hole_sensitivity),
        ("5. Exact 3-Way ΔC Additivity", test_5_exact_3way_delta_c_additivity),
        ("6. OOF Validation Leakage Isolation", test_6_oof_validation_leakage_isolation),
    ]
    for label, fn in tests:
        fn()
        print(f"[PASS] {label}")
    print("\nAll 6 invariant verification suites PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
