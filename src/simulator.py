"""Deterministic 10 Hz (dt = 0.1s) 2D Biomechanical Match Engine for Tab 1 interactive testing."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from src.matrix import compute_seam_expansion
from src.physics import (
    OMEGA_MAX,
    V_MAX_OUTFIELD,
    compute_local_matchup_tensor,
    decompose_action_control_delta,
    evaluate_frame_state,
    pass_release_favorability,
    reception_favorability,
)

SIM_DT = 0.1
DEFAULT_STEPS = 18  # 1.7s rollout (t = 0.0 .. 1.7s)
DEF_A_MAX = 4.6     # m/s^2 linear acceleration limit
DEF_REACT_STEPS = 3  # 0.3s reaction latency buffer


def _unit_from_deg(angle_deg: float) -> np.ndarray:
    rad = math.radians(angle_deg)
    return np.array([math.cos(rad), math.sin(rad)], dtype=float)


def _rotate_toward(current: np.ndarray, target: np.ndarray, max_rad: float) -> np.ndarray:
    """Rotate 2D unit vector toward target bounded by finite angular turn rate max_rad."""
    cn = float(np.hypot(current[0], current[1]))
    tn = float(np.hypot(target[0], target[1]))
    u_c = current / cn if cn > 1e-9 else np.array([1.0, 0.0])
    if tn < 1e-9:
        return u_c
    u_t = target / tn
    delta = math.atan2(u_c[0] * u_t[1] - u_c[1] * u_t[0], u_c[0] * u_t[0] + u_c[1] * u_t[1])
    step = float(np.clip(delta, -max_rad, max_rad))
    ca, sa = math.cos(step), math.sin(step)
    out = np.array([u_c[0] * ca - u_c[1] * sa, u_c[0] * sa + u_c[1] * ca], dtype=float)
    on = float(np.hypot(out[0], out[1]))
    return out / on if on > 1e-9 else u_c


def _make_player(
    pid: int,
    name: str,
    role: str,
    is_attacking: bool,
    x: float,
    y: float,
    hip_deg: float,
    vx: float = 0.0,
    vy: float = 0.0,
    stance: str = "Square",
    dominant_foot: str = "R",
    weak_foot_comfort: float = 0.45,
    is_gk: bool = False,
) -> dict[str, Any]:
    fwd = _unit_from_deg(hip_deg)
    lead_m = 0.28 if stance == "Left-Forward" else (-0.28 if stance == "Right-Forward" else 0.0)
    return {
        "player_id": pid,
        "short_name": name,
        "role": role,
        "is_attacking": is_attacking,
        "is_gk": is_gk,
        "x": float(x),
        "y": float(y),
        "vx": float(vx),
        "vy": float(vy),
        "hip_fwd": fwd,
        "sh_fwd": fwd.copy(),
        "torso_twist": 0.0,
        "stance": stance,
        "stance_lead_m": lead_m,
        "pose_source": "3d_pose",
        "dominant_foot": dominant_foot,
        "weak_foot_comfort": weak_foot_comfort,
    }


def get_preset_scenarios() -> dict[str, dict[str, Any]]:
    """Return the 4 canonical tactical scenarios for Tab 1."""
    return {
        "Inverted Winger Feint vs. Low Block": {
            "description": (
                "Right-footed LW attacks a Right-Forward RB on the outside fake before cutting inside "
                "onto the strong right foot, dragging the RCB and opening a central half-space seam."
            ),
            "default_action": "Body Feint",
            "carrier_id": 101,
            "primary_def_id": 201,
            "target_id": 103,
            "ball_xy": [24.0, 19.0],
            "players": [
                _make_player(101, "LW (Carrier)", "LW", True, 24.0, 19.0, -18.0, 3.2, -0.5, "Left-Forward", "R", 0.38),
                _make_player(102, "CF #9", "CF", True, 35.0, 2.0, -5.0, 1.0, 0.2, "Square", "R", 0.55),
                _make_player(103, "CAM #10", "AM", True, 26.0, 3.5, 10.0, 0.8, 0.0, "Square", "R", 0.62),
                _make_player(104, "RW", "RW", True, 28.0, -14.0, 25.0, 1.2, 0.4, "Right-Forward", "L", 0.42),
                _make_player(105, "CM #8", "CM", True, 14.0, 8.0, 0.0, 0.6, 0.0, "Square", "R", 0.58),
                _make_player(201, "RB (Primary)", "RB", False, 29.0, 17.5, 160.0, -0.5, 0.3, "Right-Forward", "R", 0.45),
                _make_player(202, "RCB", "RCB", False, 35.0, 7.5, 172.0, 0.0, 0.2, "Square", "R", 0.48),
                _make_player(203, "LCB", "LCB", False, 36.0, -6.0, 175.0, 0.0, 0.0, "Square", "L", 0.44),
                _make_player(204, "LB", "LB", False, 33.0, -17.0, 165.0, 0.0, 0.0, "Left-Forward", "L", 0.42),
                _make_player(205, "RDM", "DM", False, 23.0, 6.5, 155.0, 0.0, 0.3, "Square", "R", 0.52),
                _make_player(206, "LDM", "DM", False, 24.0, -6.5, 170.0, 0.0, 0.0, "Square", "R", 0.50),
                _make_player(200, "GK", "GK", False, 49.0, 0.5, 180.0, 0.0, 0.0, "Square", "R", 0.45, is_gk=True),
            ],
        },
        "Back-Foot Half-Space Reception": {
            "description": (
                "Central midfielder feeds a diagonal pass into the #10 in the left half-space. "
                "Compare an open-hip strong-foot back-foot touch toward goal vs. a closed weak-foot touch."
            ),
            "default_action": "Directional First Touch",
            "carrier_id": 103,
            "primary_def_id": 205,
            "target_id": 102,
            "ball_xy": [16.0, -1.0],
            "pass_origin_xy": [15.0, -2.5],
            "players": [
                _make_player(101, "CM (Passer)", "CM", True, 15.0, -2.5, 35.0, 0.5, 0.4, "Square", "R", 0.55),
                _make_player(103, "CAM (Receiver)", "AM", True, 27.0, 9.5, -15.0, 0.8, 0.3, "Left-Forward", "R", 0.40),
                _make_player(102, "CF #9", "CF", True, 37.0, 0.5, 0.0, 1.2, -0.2, "Square", "R", 0.52),
                _make_player(104, "LW", "LW", True, 31.0, 21.0, -25.0, 1.5, 0.0, "Square", "R", 0.45),
                _make_player(105, "RW", "RW", True, 30.0, -15.0, 20.0, 1.0, 0.3, "Square", "L", 0.42),
                _make_player(205, "RDM (Primary)", "DM", False, 24.5, 5.5, 125.0, 0.4, 1.0, "Right-Forward", "R", 0.48),
                _make_player(201, "RB", "RB", False, 34.0, 18.0, 185.0, 0.0, 0.0, "Square", "R", 0.45),
                _make_player(202, "RCB", "RCB", False, 36.0, 6.5, 175.0, 0.0, 0.0, "Square", "R", 0.46),
                _make_player(203, "LCB", "LCB", False, 36.5, -6.5, 178.0, 0.0, 0.0, "Square", "L", 0.44),
                _make_player(204, "LB", "LB", False, 34.0, -17.5, 170.0, 0.0, 0.0, "Square", "L", 0.44),
                _make_player(206, "LDM", "DM", False, 23.0, -7.5, 165.0, 0.0, 0.0, "Square", "R", 0.50),
                _make_player(200, "GK", "GK", False, 49.0, 0.0, 180.0, 0.0, 0.0, "Square", "R", 0.45, is_gk=True),
            ],
        },
        "Wrong-Footed Fullback 1v1": {
            "description": (
                "Left-footed RW attacks a Left-Forward LB whose trailing right hip is locked behind, "
                "maximizing lateral weak-stance latency on the inside cut."
            ),
            "default_action": "Body Feint",
            "carrier_id": 104,
            "primary_def_id": 204,
            "target_id": 102,
            "ball_xy": [26.0, -18.5],
            "players": [
                _make_player(104, "RW (Carrier)", "RW", True, 26.0, -18.5, 20.0, 3.4, 0.6, "Right-Forward", "L", 0.36),
                _make_player(102, "CF #9", "CF", True, 35.5, -2.5, 5.0, 1.0, -0.3, "Square", "R", 0.54),
                _make_player(103, "CAM #10", "AM", True, 26.5, -4.5, 10.0, 0.8, 0.0, "Square", "R", 0.58),
                _make_player(101, "LW", "LW", True, 29.0, 15.0, -25.0, 1.0, -0.2, "Square", "R", 0.46),
                _make_player(105, "CM #8", "CM", True, 16.0, -9.0, 5.0, 0.6, 0.0, "Square", "R", 0.52),
                _make_player(204, "LB (Primary)", "LB", False, 30.5, -16.5, -155.0, -0.4, -0.3, "Left-Forward", "L", 0.42),
                _make_player(203, "LCB", "LCB", False, 35.5, -6.5, -172.0, 0.0, -0.2, "Square", "L", 0.46),
                _make_player(202, "RCB", "RCB", False, 36.0, 6.5, 178.0, 0.0, 0.0, "Square", "R", 0.48),
                _make_player(201, "RB", "RB", False, 33.5, 17.5, 165.0, 0.0, 0.0, "Right-Forward", "R", 0.46),
                _make_player(206, "LDM", "DM", False, 24.0, -6.0, -155.0, 0.0, -0.3, "Square", "R", 0.50),
                _make_player(205, "RDM", "DM", False, 24.5, 7.0, 170.0, 0.0, 0.0, "Square", "R", 0.50),
                _make_player(200, "GK", "GK", False, 49.0, -0.5, 180.0, 0.0, 0.0, "Square", "R", 0.45, is_gk=True),
            ],
        },
        "Central Line-Break Pass": {
            "description": (
                "Deep #6 uses an open-body disguised pass between two shifting midfield pivots "
                "into an open-hip #10 operating in the central pocket ahead of the back line."
            ),
            "default_action": "Driven / Disguised Pass",
            "carrier_id": 105,
            "primary_def_id": 205,
            "target_id": 103,
            "ball_xy": [10.0, -1.0],
            "players": [
                _make_player(105, "DM #6 (Passer)", "DM", True, 10.0, -1.0, 8.0, 0.8, 0.2, "Left-Forward", "R", 0.55),
                _make_player(103, "CAM (Target)", "AM", True, 29.0, 4.5, -12.0, 0.6, 0.4, "Left-Forward", "R", 0.50),
                _make_player(102, "CF #9", "CF", True, 38.0, -1.5, 0.0, 1.1, 0.2, "Square", "R", 0.54),
                _make_player(101, "LW", "LW", True, 28.0, 19.0, -20.0, 1.4, 0.0, "Square", "R", 0.45),
                _make_player(104, "RW", "RW", True, 28.0, -18.0, 20.0, 1.4, 0.0, "Square", "L", 0.45),
                _make_player(205, "RDM (Primary)", "DM", False, 20.0, 6.0, 160.0, 0.0, -0.5, "Right-Forward", "R", 0.48),
                _make_player(206, "LDM", "DM", False, 20.5, -6.0, -162.0, 0.0, 0.3, "Left-Forward", "R", 0.48),
                _make_player(202, "RCB", "RCB", False, 35.5, 7.0, 175.0, 0.0, 0.0, "Square", "R", 0.48),
                _make_player(203, "LCB", "LCB", False, 36.0, -6.5, 178.0, 0.0, 0.0, "Square", "L", 0.45),
                _make_player(201, "RB", "RB", False, 33.0, 18.5, 172.0, 0.0, 0.0, "Square", "R", 0.46),
                _make_player(204, "LB", "LB", False, 33.5, -18.0, 170.0, 0.0, 0.0, "Square", "L", 0.44),
                _make_player(200, "GK", "GK", False, 49.0, 0.0, 180.0, 0.0, 0.0, "Square", "R", 0.45, is_gk=True),
            ],
        },
    }


def run_simulation(
    scenario: dict[str, Any],
    action: str = "Body Feint",
    n_steps: int = DEFAULT_STEPS,
    carrier_overrides: dict[str, Any] | None = None,
    defender_overrides: dict[str, Any] | None = None,
    touch_mode: str = "Open Strong-Foot",
) -> dict[str, Any]:
    """Run deterministic 10 Hz biomechanical rollout and evaluate control, matrix, and 3-way ΔC."""
    players = copy.deepcopy(scenario["players"])
    carrier_id = int(scenario["carrier_id"])
    primary_def_id = int(scenario["primary_def_id"])
    target_id = int(scenario["target_id"])

    by_id = {p["player_id"]: p for p in players}
    carrier = by_id[carrier_id]
    primary_def = by_id[primary_def_id]
    target = by_id[target_id]

    if carrier_overrides:
        if "x" in carrier_overrides:
            carrier["x"] = float(carrier_overrides["x"])
        if "y" in carrier_overrides:
            carrier["y"] = float(carrier_overrides["y"])
        if "dominant_foot" in carrier_overrides:
            carrier["dominant_foot"] = str(carrier_overrides["dominant_foot"])
        if "hip_deg" in carrier_overrides:
            fwd = _unit_from_deg(float(carrier_overrides["hip_deg"]))
            carrier["hip_fwd"] = fwd
            carrier["sh_fwd"] = fwd.copy()

    if defender_overrides:
        if "x" in defender_overrides:
            primary_def["x"] = float(defender_overrides["x"])
        if "y" in defender_overrides:
            primary_def["y"] = float(defender_overrides["y"])
        if "hip_deg" in defender_overrides:
            fwd = _unit_from_deg(float(defender_overrides["hip_deg"]))
            primary_def["hip_fwd"] = fwd
            primary_def["sh_fwd"] = fwd.copy()
        if "stance" in defender_overrides:
            st = str(defender_overrides["stance"])
            primary_def["stance"] = st
            primary_def["stance_lead_m"] = 0.30 if st == "Left-Forward" else (-0.30 if st == "Right-Forward" else 0.0)

    ball_xy = np.array(
        scenario.get("pass_origin_xy", [carrier["x"], carrier["y"]])
        if action == "Directional First Touch"
        else [carrier["x"], carrier["y"]],
        dtype=float,
    )

    # Record initial defender anchor positions for inter-defender matrix spring coupling
    def_anchors = {
        p["player_id"]: np.array([p["x"], p["y"]], dtype=float)
        for p in players
        if not p["is_attacking"] and not p["is_gk"]
    }

    # Precompute strong-foot cut direction for feints / touches
    dom_r = carrier["dominant_foot"] == "R"
    # Right-footed carrier on left flank (y > 0) cuts inside (-y); on right flank cuts inside (+y).
    inside_sign = -1.0 if carrier["y"] >= 0.0 else 1.0
    fake_sign = -inside_sign

    # Reaction buffer stores carrier/ball targets from 3 steps ago (tau_react = 0.3s)
    perceived_history: list[np.ndarray] = [
        np.array([carrier["x"], carrier["y"]], dtype=float) for _ in range(DEF_REACT_STEPS + 1)
    ]

    frames: list[dict[str, Any]] = []
    eval_t0: dict[str, Any] | None = None
    players_t0: list[dict[str, Any]] | None = None

    a_pass_val = 0.0
    a_recv_val = 0.0
    if action == "Driven / Disguised Pass":
        pvec = np.array([target["x"] - carrier["x"], target["y"] - carrier["y"]], dtype=float)
        a_pass_val = pass_release_favorability(
            carrier["hip_fwd"], pvec, carrier["dominant_foot"], carrier["weak_foot_comfort"]
        )
        t_goal = np.array([52.5 - target["x"], -target["y"]], dtype=float)
        a_recv_val = reception_favorability(
            target["hip_fwd"], pvec, t_goal, target["dominant_foot"], target["weak_foot_comfort"]
        )
    elif action == "Directional First Touch":
        in_vec = np.array([carrier["x"] - ball_xy[0], carrier["y"] - ball_xy[1]], dtype=float)
        if np.hypot(in_vec[0], in_vec[1]) < 0.5:
            in_vec = np.array([1.0, 0.3], dtype=float)
        c_goal = np.array([52.5 - carrier["x"], -carrier["y"]], dtype=float)
        eff_fwd = carrier["hip_fwd"] if touch_mode == "Open Strong-Foot" else -carrier["hip_fwd"]
        a_recv_val = reception_favorability(
            eff_fwd, in_vec, c_goal, carrier["dominant_foot"], carrier["weak_foot_comfort"]
        )

    for step in range(n_steps):
        t_sec = round(step * SIM_DT, 2)
        snap_players = copy.deepcopy(players)
        ev = evaluate_frame_state(snap_players)

        if step == 0:
            eval_t0 = ev
            players_t0 = snap_players
            seam = compute_seam_expansion(ev["matrix"], ev["matrix"])
            decomp = decompose_action_control_delta(players_t0, snap_players, eval_t0, ev)
        else:
            assert eval_t0 is not None and players_t0 is not None
            seam = compute_seam_expansion(eval_t0["matrix"], ev["matrix"])
            decomp = decompose_action_control_delta(players_t0, snap_players, eval_t0, ev)

        frames.append(
            {
                "step": step,
                "t": t_sec,
                "players": snap_players,
                "ball_xy": ball_xy.tolist(),
                "eval": ev,
                "seam": seam,
                "decomp": decomp,
            }
        )

        # ---- Advance deterministic 10 Hz physics to next step ----
        # 1. Carrier & Ball dynamics per chosen action
        if action == "Body Feint":
            if step < 5:
                # Shoulder/hip fake toward outside wing to draw defender momentum & hip pivot
                des_vel = np.array([2.2, 3.8 * fake_sign], dtype=float)
                carrier["torso_twist"] = math.radians(34.0 * fake_sign)
                carrier["stance"] = "Left-Forward" if fake_sign > 0 else "Right-Forward"
                carrier["stance_lead_m"] = 0.25 * fake_sign
                perceived_target = np.array([carrier["x"] + 2.5, carrier["y"] + 4.5 * fake_sign])
            else:
                # Explosive cut onto strong-foot inside corridor
                strong_boost = 1.12 if ((dom_r and inside_sign < 0) or ((not dom_r) and inside_sign > 0)) else 0.94
                des_vel = np.array([5.8 * strong_boost, 4.4 * inside_sign * strong_boost], dtype=float)
                carrier["torso_twist"] = math.radians(-12.0 * inside_sign)
                carrier["stance"] = "Left-Forward" if dom_r else "Right-Forward"
                carrier["stance_lead_m"] = 0.22 if dom_r else -0.22
                perceived_target = np.array([carrier["x"], carrier["y"]], dtype=float)

            c_vel = np.array([carrier["vx"], carrier["vy"]], dtype=float)
            dv = des_vel - c_vel
            dvn = float(np.hypot(dv[0], dv[1]))
            if dvn > 6.5 * SIM_DT:
                dv = dv * (6.5 * SIM_DT / dvn)
            c_vel = c_vel + dv
            carrier["vx"], carrier["vy"] = float(c_vel[0]), float(c_vel[1])
            carrier["x"] = float(np.clip(carrier["x"] + carrier["vx"] * SIM_DT, -50.0, 50.0))
            carrier["y"] = float(np.clip(carrier["y"] + carrier["vy"] * SIM_DT, -32.0, 32.0))
            carrier["hip_fwd"] = _rotate_toward(carrier["hip_fwd"], c_vel, 4.5 * SIM_DT)
            ca, sa = math.cos(carrier["torso_twist"]), math.sin(carrier["torso_twist"])
            hf = carrier["hip_fwd"]
            carrier["sh_fwd"] = np.array([hf[0] * ca - hf[1] * sa, hf[0] * sa + hf[1] * ca], dtype=float)
            ball_xy = np.array([carrier["x"] + 0.6 * hf[0], carrier["y"] + 0.6 * hf[1]], dtype=float)

        elif action == "Directional First Touch":
            if step < 4:
                # Incoming pass traveling toward receiver
                frac = (step + 1) / 4.0
                origin = np.array(scenario.get("pass_origin_xy", [carrier["x"] - 8.0, carrier["y"] - 5.0]), dtype=float)
                ball_xy = origin * (1.0 - frac) + np.array([carrier["x"], carrier["y"]]) * frac
                if touch_mode == "Open Strong-Foot":
                    carrier["hip_fwd"] = _rotate_toward(
                        carrier["hip_fwd"], np.array([52.5 - carrier["x"], -carrier["y"]]), 4.2 * SIM_DT
                    )
                else:
                    carrier["hip_fwd"] = _rotate_toward(
                        carrier["hip_fwd"], origin - np.array([carrier["x"], carrier["y"]]), 4.2 * SIM_DT
                    )
                perceived_target = np.array([carrier["x"], carrier["y"]], dtype=float)
            else:
                if touch_mode == "Open Strong-Foot":
                    goal_dir = np.array([52.5 - carrier["x"], -carrier["y"] * 0.65], dtype=float)
                    gn = float(np.hypot(goal_dir[0], goal_dir[1]))
                    u_g = goal_dir / gn if gn > 1e-6 else np.array([1.0, 0.0])
                    des_vel = u_g * 5.9
                    carrier["stance"] = "Left-Forward" if dom_r else "Right-Forward"
                    carrier["stance_lead_m"] = 0.24 if dom_r else -0.24
                else:
                    # Closed weak-foot touch stalls carrier backward/laterally into pressure
                    des_vel = np.array([-1.6, 1.2 * inside_sign], dtype=float)
                    carrier["stance"] = "Square"
                    carrier["stance_lead_m"] = 0.0

                c_vel = np.array([carrier["vx"], carrier["vy"]], dtype=float)
                dv = des_vel - c_vel
                dvn = float(np.hypot(dv[0], dv[1]))
                if dvn > 5.8 * SIM_DT:
                    dv = dv * (5.8 * SIM_DT / dvn)
                c_vel = c_vel + dv
                carrier["vx"], carrier["vy"] = float(c_vel[0]), float(c_vel[1])
                carrier["x"] = float(np.clip(carrier["x"] + carrier["vx"] * SIM_DT, -50.0, 50.0))
                carrier["y"] = float(np.clip(carrier["y"] + carrier["vy"] * SIM_DT, -32.0, 32.0))
                carrier["hip_fwd"] = _rotate_toward(carrier["hip_fwd"], c_vel, 4.0 * SIM_DT)
                carrier["sh_fwd"] = carrier["hip_fwd"].copy()
                ball_xy = np.array(
                    [carrier["x"] + 0.7 * carrier["hip_fwd"][0], carrier["y"] + 0.7 * carrier["hip_fwd"][1]],
                    dtype=float,
                )
                perceived_target = ball_xy.copy()

        else:  # Driven / Disguised Pass
            pass_start_step = 2
            pass_flight_steps = 8
            if step < pass_start_step:
                # Shoulder disguise away from pass lane
                carrier["torso_twist"] = math.radians(-26.0)
                ball_xy = np.array([carrier["x"] + 0.5, carrier["y"]], dtype=float)
                perceived_target = np.array([carrier["x"] + 4.0, carrier["y"] - 8.0], dtype=float)
            elif step <= pass_start_step + pass_flight_steps:
                frac = (step - pass_start_step) / float(pass_flight_steps)
                src_xy = np.array([carrier["x"], carrier["y"]], dtype=float)
                dst_xy = np.array([target["x"], target["y"]], dtype=float)
                ball_xy = src_xy * (1.0 - frac) + dst_xy * frac
                carrier["torso_twist"] = 0.0
                # Target checks into pocket and opens hips toward goal
                target["x"] = float(np.clip(target["x"] + 1.4 * SIM_DT, -50.0, 50.0))
                target["hip_fwd"] = _rotate_toward(
                    target["hip_fwd"], np.array([52.5 - target["x"], -target["y"]]), 4.0 * SIM_DT
                )
                target["sh_fwd"] = target["hip_fwd"].copy()
                perceived_target = ball_xy.copy()
            else:
                # Target bursts onto goal-facing first touch
                g_dir = np.array([52.5 - target["x"], -target["y"] * 0.5], dtype=float)
                gn = float(np.hypot(g_dir[0], g_dir[1]))
                u_g = g_dir / gn if gn > 1e-6 else np.array([1.0, 0.0])
                target["vx"], target["vy"] = float(u_g[0] * 5.2), float(u_g[1] * 5.2)
                target["x"] = float(np.clip(target["x"] + target["vx"] * SIM_DT, -50.0, 50.0))
                target["y"] = float(np.clip(target["y"] + target["vy"] * SIM_DT, -32.0, 32.0))
                target["hip_fwd"] = _rotate_toward(target["hip_fwd"], u_g, 4.2 * SIM_DT)
                target["sh_fwd"] = target["hip_fwd"].copy()
                ball_xy = np.array([target["x"] + 0.6 * u_g[0], target["y"] + 0.6 * u_g[1]], dtype=float)
                perceived_target = ball_xy.copy()

        perceived_history.append(perceived_target)
        delayed_target = perceived_history[-1 - DEF_REACT_STEPS]

        # 2. Supporting attackers drift smoothly into space
        for p in players:
            if p["is_attacking"] and p["player_id"] not in (carrier_id, target_id):
                p["x"] = float(np.clip(p["x"] + p["vx"] * SIM_DT, -50.0, 49.0))
                p["y"] = float(np.clip(p["y"] + p["vy"] * SIM_DT, -32.0, 32.0))

        # 3. Defender biomechanical tracking with reaction latency, turn-rate limit, and matrix spring coupling
        for p in players:
            if p["is_attacking"] or p["is_gk"]:
                continue
            pos = np.array([p["x"], p["y"]], dtype=float)
            vel = np.array([p["vx"], p["vy"]], dtype=float)
            anchor = def_anchors[p["player_id"]]

            if p["player_id"] == primary_def_id:
                # Primary defender closes down the delayed perceived target -> overshoots on feint!
                to_targ = delayed_target - pos
                dist_t = float(np.hypot(to_targ[0], to_targ[1]))
                u_t = to_targ / dist_t if dist_t > 1e-6 else np.array([-1.0, 0.0])
                desired_v = u_t * min(V_MAX_OUTFIELD * 0.85, dist_t * 1.45)
                if action == "Body Feint" and step in (4, 5, 6):
                    # Committed lunge opens hip and locks tackling stance toward fake side
                    p["stance"] = "Right-Forward" if fake_sign > 0 else "Left-Forward"
                    p["stance_lead_m"] = -0.34 if fake_sign > 0 else 0.34
            else:
                # Secondary defenders balance ball-shift tracking with inter-defender matrix spring coupling
                to_ball = delayed_target - pos
                db = float(np.hypot(to_ball[0], to_ball[1]))
                u_b = to_ball / db if db > 1e-6 else np.array([-1.0, 0.0])
                spring_pull = (anchor - pos) * 0.45
                ball_pull = u_b * (2.6 if db < 15.0 else 1.2)
                desired_v = spring_pull + ball_pull

            # Enforce linear acceleration and speed limits
            dv = desired_v - vel
            dvn = float(np.hypot(dv[0], dv[1]))
            if dvn > DEF_A_MAX * SIM_DT:
                dv = dv * (DEF_A_MAX * SIM_DT / dvn)
            vel = vel + dv
            spd = float(np.hypot(vel[0], vel[1]))
            if spd > V_MAX_OUTFIELD:
                vel = vel * (V_MAX_OUTFIELD / spd)

            pos = pos + vel * SIM_DT
            p["x"] = float(np.clip(pos[0], -50.0, 50.0))
            p["y"] = float(np.clip(pos[1], -32.0, 32.0))
            p["vx"], p["vy"] = float(vel[0]), float(vel[1])

            # Enforce finite angular hip turn rate omega_max
            look_vec = delayed_target - pos
            p["hip_fwd"] = _rotate_toward(p["hip_fwd"], look_vec, OMEGA_MAX * SIM_DT)
            p["sh_fwd"] = p["hip_fwd"].copy()

    # Local matchup tensors at t0 and final step
    def _matchup_for(frame_players: list[dict[str, Any]]) -> dict[str, Any]:
        c_p = next(p for p in frame_players if p["player_id"] == carrier_id)
        defs = [p for p in frame_players if not p["is_attacking"]]
        tms = [p for p in frame_players if p["is_attacking"]]
        return compute_local_matchup_tensor(c_p, defs, tms)

    return {
        "scenario_name": scenario.get("name", "Custom"),
        "action": action,
        "carrier_id": carrier_id,
        "primary_def_id": primary_def_id,
        "target_id": target_id,
        "a_pass": round(a_pass_val, 4),
        "a_recv": round(a_recv_val, 4),
        "matchup_t0": _matchup_for(frames[0]["players"]),
        "matchup_end": _matchup_for(frames[-1]["players"]),
        "frames": frames,
    }
