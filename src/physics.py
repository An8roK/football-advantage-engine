"""Vectorized Anisotropic vs. Isotropic Pitch Control, Strong/Weak-Side Coupling, and exact ΔC ablation."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from src.matrix import analyze_defensive_matrix

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GRID_NX = 35
GRID_NY = 22

# Calibrated football biomechanical physics parameters
TAU_REACT = 0.30          # s: cognitive/neuromuscular reaction latency
V_MAX_OUTFIELD = 7.2      # m/s: sustained sprint speed
V_MAX_GK = 5.5            # m/s: goalkeeper movement speed
OMEGA_MAX = 3.2           # rad/s (~183 deg/s): full-hip pivot rate
ALPHA_STANCE = 0.42       # s: max stance + back-turned penalty
BETA_MATRIX = 0.45        # s: structural mutual-cover deficit latency on defenders
CONTROL_TEMP = 0.45       # s: logistic soft-min competition scale
BACK_TURNED_RAD = math.radians(110.0)


@lru_cache(maxsize=1)
def get_pitch_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cached 35 x 22 (3m resolution) pitch grid coordinates."""
    xs = np.linspace(-51.0, 51.0, GRID_NX, dtype=float)
    ys = np.linspace(-31.5, 31.5, GRID_NY, dtype=float)
    xx, yy = np.meshgrid(xs, ys)  # shape (22, 35)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    return xs, ys, xx, yy, grid_xy


@lru_cache(maxsize=1)
def get_threat_surface() -> np.ndarray:
    """Return normalized spatial goal-threat surface V(x, y) peaking toward attacking goal (+52.5, 0)."""
    _, _, xx, yy, _ = get_pitch_grid()
    dx = np.maximum(0.5, 52.5 - xx)
    dist_goal = np.hypot(dx, yy)
    angle_to_goal = np.arctan2(np.abs(yy), dx)
    angle_weight = 0.55 + 0.45 * np.cos(0.5 * angle_to_goal)
    raw_v = np.exp(-dist_goal / 28.0) * angle_weight
    return raw_v / float(np.sum(raw_v))


def compute_stance_penalty(
    signed_angle: np.ndarray,
    stance: str,
    stance_lead_m: float = 0.0,
) -> np.ndarray:
    """Compute σ_i(r) in [0, 1] penalizing weak/trailing tackling-foot reaches and back-turned pivots (>110 deg)."""
    lat = np.sin(signed_angle)  # >0 = target is to player's LEFT, <0 = RIGHT
    lead_scale = min(1.0, max(0.35, abs(stance_lead_m) / 0.35)) if stance != "Square" else 0.0

    if stance == "Left-Forward":
        # Left foot forward locks right hip behind -> lateral reach to RIGHT (lat < 0) is weak side.
        weak_reach = np.maximum(0.0, -lat) * lead_scale
    elif stance == "Right-Forward":
        # Right foot forward locks left hip behind -> lateral reach to LEFT (lat > 0) is weak side.
        weak_reach = np.maximum(0.0, lat) * lead_scale
    else:
        weak_reach = np.zeros_like(signed_angle)

    abs_ang = np.abs(signed_angle)
    back_vuln = np.clip((abs_ang - BACK_TURNED_RAD) / (math.pi - BACK_TURNED_RAD), 0.0, 1.0)
    return np.clip(0.55 * weak_reach + 0.65 * back_vuln, 0.0, 1.0)


def compute_player_arrival_time(
    player: dict[str, Any],
    anisotropic: bool = True,
    deficit_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 2D arrival time grid T_i(r) of shape (22, 35) for a single player."""
    _, _, xx, yy, _ = get_pitch_grid()
    px, py = float(player["x"]), float(player["y"])
    vx, vy = float(player.get("vx", 0.0)), float(player.get("vy", 0.0))
    v_max = V_MAX_GK if player.get("is_gk", False) else V_MAX_OUTFIELD

    rx = px + vx * TAU_REACT
    ry = py + vy * TAU_REACT
    dist = np.hypot(xx - rx, yy - ry)
    t_grid = TAU_REACT + dist / v_max

    if anisotropic:
        hip_fwd = np.asarray(player.get("hip_fwd", [1.0, 0.0]), dtype=float)
        fn = float(np.hypot(hip_fwd[0], hip_fwd[1]))
        fx, fy = (hip_fwd[0] / fn, hip_fwd[1] / fn) if fn > 1e-9 else (1.0, 0.0)

        dx = xx - px
        dy = yy - py
        dn = np.maximum(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / dn, dy / dn

        cross = fx * uy - fy * ux
        dot = fx * ux + fy * uy
        signed_ang = np.arctan2(cross, dot)

        # Mild shoulder pre-turn credit when upper body is already rotating toward target
        twist = float(player.get("torso_twist", 0.0))
        eff_abs_ang = np.maximum(0.0, np.abs(signed_ang) - 0.20 * np.clip(twist * np.sign(signed_ang), -0.35, 0.35))

        sigma = compute_stance_penalty(
            signed_ang,
            stance=str(player.get("stance", "Square")),
            stance_lead_m=float(player.get("stance_lead_m", 0.0)),
        )
        t_grid = t_grid + (eff_abs_ang / OMEGA_MAX) + (ALPHA_STANCE * sigma)

        # Couple defensive structural cover deficit into defending outfield players
        if deficit_grid is not None and not player.get("is_attacking", False) and not player.get("is_gk", False):
            t_grid = t_grid + BETA_MATRIX * deficit_grid

    return t_grid


def compute_pitch_control(
    players: list[dict[str, Any]],
    anisotropic: bool = True,
    deficit_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Compute vectorized attacking pitch control grid (22, 35) and threat-weighted scalar C_threat."""
    threat = get_threat_surface()
    att_players = [p for p in players if p.get("is_attacking", False)]
    def_players = [p for p in players if not p.get("is_attacking", False)]

    if not att_players and not def_players:
        neutral = np.full((GRID_NY, GRID_NX), 0.5, dtype=float)
        return neutral, float(np.sum(neutral * threat))
    if not att_players:
        zeros = np.zeros((GRID_NY, GRID_NX), dtype=float)
        return zeros, 0.0
    if not def_players:
        ones = np.ones((GRID_NY, GRID_NX), dtype=float)
        return ones, 1.0

    t_att = np.stack(
        [compute_player_arrival_time(p, anisotropic=anisotropic, deficit_grid=None) for p in att_players],
        axis=0,
    )
    t_def = np.stack(
        [compute_player_arrival_time(p, anisotropic=anisotropic, deficit_grid=deficit_grid) for p in def_players],
        axis=0,
    )

    # Log-sum-exp stabilization prevents numerical underflow at distant cells
    t_min = np.minimum(np.min(t_att, axis=0), np.min(t_def, axis=0))
    att_score = np.sum(np.exp(-(t_att - t_min[None, :, :]) / CONTROL_TEMP), axis=0)
    def_score = np.sum(np.exp(-(t_def - t_min[None, :, :]) / CONTROL_TEMP), axis=0)
    control = att_score / (att_score + def_score + 1e-12)
    return control, float(np.sum(control * threat))


def extract_defending_outfield(players: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Extract defending outfield (non-GK) coordinates and hip-facing vectors."""
    defs = [p for p in players if not p.get("is_attacking", False) and not p.get("is_gk", False)]
    if not defs:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
    xy = np.array([[float(p["x"]), float(p["y"])] for p in defs], dtype=float)
    fwd = np.array([np.asarray(p.get("hip_fwd", [-1.0, 0.0]), dtype=float) for p in defs], dtype=float)
    return xy, fwd


def evaluate_frame_state(players: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute isotropic control, base anisotropic control, defensive matrix M(t), and coupled control."""
    _, _, _, _, grid_xy = get_pitch_grid()
    threat = get_threat_surface()

    iso_grid, iso_threat = compute_pitch_control(players, anisotropic=False, deficit_grid=None)
    base_aniso_grid, _ = compute_pitch_control(players, anisotropic=True, deficit_grid=None)

    def_xy, def_fwd = extract_defending_outfield(players)
    matrix_info = analyze_defensive_matrix(
        def_xy=def_xy,
        def_fwd=def_fwd,
        att_control_grid=base_aniso_grid,
        threat_grid=threat,
        grid_xy=grid_xy,
    )
    aniso_grid, aniso_threat = compute_pitch_control(
        players,
        anisotropic=True,
        deficit_grid=matrix_info["deficit_grid"],
    )

    return {
        "iso_grid": iso_grid,
        "iso_threat": iso_threat,
        "aniso_grid": aniso_grid,
        "aniso_threat": aniso_threat,
        "matrix": matrix_info,
    }


def decompose_action_control_delta(
    players_t0: list[dict[str, Any]],
    players_t1: list[dict[str, Any]],
    eval_t0: dict[str, Any] | None = None,
    eval_t1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exact 3-way counterfactual telescoping decomposition of ΔC_threat over [t0, t1].

    Guarantees: delta_c_pos + delta_c_orient + delta_c_matrix == delta_c_total identically.
    """
    e0 = eval_t0 or evaluate_frame_state(players_t0)
    e1 = eval_t1 or evaluate_frame_state(players_t1)

    deficit_0 = e0["matrix"]["deficit_grid"]
    deficit_1 = e1["matrix"]["deficit_grid"]

    # Build counterfactual state (P1, Theta0): t1 positions/velocities with t0 orientations/stances
    t0_by_id = {p["player_id"]: p for p in players_t0 if "player_id" in p}
    players_p1_theta0: list[dict[str, Any]] = []
    for p1 in players_t1:
        p0 = t0_by_id.get(p1.get("player_id"))
        merged = dict(p1)
        if p0 is not None:
            merged["hip_fwd"] = np.asarray(p0["hip_fwd"], dtype=float).copy()
            merged["sh_fwd"] = np.asarray(p0.get("sh_fwd", p0["hip_fwd"]), dtype=float).copy()
            merged["torso_twist"] = float(p0.get("torso_twist", 0.0))
            merged["stance"] = str(p0.get("stance", "Square"))
            merged["stance_lead_m"] = float(p0.get("stance_lead_m", 0.0))
        players_p1_theta0.append(merged)

    c_0 = float(e0["aniso_threat"])
    _, c_p1_t0_m0 = compute_pitch_control(players_p1_theta0, anisotropic=True, deficit_grid=deficit_0)
    _, c_p1_t1_m0 = compute_pitch_control(players_t1, anisotropic=True, deficit_grid=deficit_0)
    c_1 = float(e1["aniso_threat"])

    dc_pos = c_p1_t0_m0 - c_0
    dc_orient = c_p1_t1_m0 - c_p1_t0_m0
    dc_matrix = c_1 - c_p1_t1_m0
    dc_total = c_1 - c_0

    iso_dc = float(e1["iso_threat"] - e0["iso_threat"])
    drivers = {
        "Positional Shift": abs(dc_pos),
        "Orientation & Wrong-Footing": abs(dc_orient),
        "Matrix Distortion": abs(dc_matrix),
    }
    dominant_driver = max(drivers, key=drivers.get)

    return {
        "c_t0": c_0,
        "c_t1": c_1,
        "delta_c_total": dc_total,
        "delta_c_pos": dc_pos,
        "delta_c_orient": dc_orient,
        "delta_c_matrix": dc_matrix,
        "iso_c_t0": float(e0["iso_threat"]),
        "iso_c_t1": float(e1["iso_threat"]),
        "iso_delta_c": iso_dc,
        "aniso_minus_iso_delta": dc_total - iso_dc,
        "dominant_driver": dominant_driver,
    }


def pass_release_favorability(
    passer_fwd: np.ndarray,
    pass_vec: np.ndarray,
    dominant_foot: str = "R",
    weak_foot_comfort: float = 0.51,
) -> float:
    """Score outgoing pass alignment with passer's open strong-foot striking arc in [-1, 1]."""
    fn = float(np.hypot(passer_fwd[0], passer_fwd[1]))
    pn = float(np.hypot(pass_vec[0], pass_vec[1]))
    if fn < 1e-9 or pn < 1e-9:
        return 0.0
    u_f = passer_fwd / fn
    u_p = pass_vec / pn

    theta = math.atan2(u_f[0] * u_p[1] - u_f[1] * u_p[0], u_f[0] * u_p[0] + u_f[1] * u_p[1])
    # Right-footed natural wrap-across arc peaks ~+20 deg (left of hip facing); left-footed at -20 deg.
    theta_opt = math.radians(20.0) if dominant_foot == "R" else math.radians(-20.0)
    align_strong = math.cos(theta - theta_opt)
    align_weak = math.cos(theta + theta_opt) * float(np.clip(weak_foot_comfort, 0.25, 0.90))
    closed_hip_pen = 0.35 * max(0.0, -math.cos(theta))
    return float(np.clip(max(align_strong, align_weak) - closed_hip_pen, -1.0, 1.0))


def reception_favorability(
    recv_fwd: np.ndarray,
    incoming_pass_vec: np.ndarray,
    goal_vec: np.ndarray,
    dominant_foot: str = "R",
    weak_foot_comfort: float = 0.51,
) -> float:
    """Score whether pass arrives onto receiver's open back/strong foot toward goal in [-1, 1]."""
    def _u(v: np.ndarray) -> np.ndarray | None:
        n = float(np.hypot(v[0], v[1]))
        return v / n if n > 1e-9 else None

    u_f = _u(recv_fwd)
    u_in = _u(incoming_pass_vec)
    u_g = _u(goal_vec)
    if u_f is None or u_in is None or u_g is None:
        return 0.0

    # Positive when receiver's hips are already open toward the opponent goal
    goal_openness = float(u_f[0] * u_g[0] + u_f[1] * u_g[1])

    # Ball arrives from direction -u_in; cross(u_f, -u_in) > 0 means arriving from receiver's LEFT
    approach = -u_in
    approach_side = float(u_f[0] * approach[1] - u_f[1] * approach[0])
    dom_sign = 1.0 if dominant_foot == "R" else -1.0
    foot_side_score = dom_sign * approach_side  # >0 = strong back-foot cushion, <0 = weak side

    w = float(np.clip(weak_foot_comfort, 0.25, 0.90))
    foot_term = foot_side_score if foot_side_score >= 0.0 else foot_side_score * (1.25 - w)
    return float(np.clip(0.55 * goal_openness + 0.45 * foot_term, -1.0, 1.0))


def compute_local_matchup_tensor(
    carrier: dict[str, Any],
    defenders: list[dict[str, Any]],
    teammates: list[dict[str, Any]],
    radius_m: float = 14.0,
) -> dict[str, Any]:
    """Quantify net strong-vs-weak edge between carrier driving arc, nearest defenders, and teammates."""
    cx, cy = float(carrier["x"]), float(carrier["y"])
    dom = str(carrier.get("dominant_foot", "R"))
    w_comfort = float(carrier.get("weak_foot_comfort", 0.51))
    c_fwd = np.asarray(carrier.get("hip_fwd", [1.0, 0.0]), dtype=float)
    fn = float(np.hypot(c_fwd[0], c_fwd[1]))
    c_fwd = c_fwd / fn if fn > 1e-9 else np.array([1.0, 0.0], dtype=float)

    # Strong-foot driving corridor: biased toward goal & dominant cut-in angle
    goal_vec = np.array([52.5 - cx, -cy], dtype=float)
    gn = float(np.hypot(goal_vec[0], goal_vec[1]))
    u_goal = goal_vec / gn if gn > 1e-9 else np.array([1.0, 0.0], dtype=float)

    cut_angle = math.radians(22.0) if dom == "R" else math.radians(-22.0)
    ca, sa = math.cos(cut_angle), math.sin(cut_angle)
    drive_vec = np.array(
        [u_goal[0] * ca - u_goal[1] * sa, u_goal[0] * sa + u_goal[1] * ca],
        dtype=float,
    )

    # Evaluate nearest defending outfield players within radius_m
    def_exposures: list[float] = []
    nearest_def_dist = 99.0
    nearest_pivot_deg = 0.0
    nearest_back_turned = 0.0
    nearest_weak_stance = 0.0

    def_candidates = sorted(
        (d for d in defenders if not d.get("is_gk", False)),
        key=lambda d: math.hypot(float(d["x"]) - cx, float(d["y"]) - cy),
    )
    for rank, d in enumerate(def_candidates[:3]):
        dx = float(d["x"]) - cx
        dy = float(d["y"]) - cy
        dist = math.hypot(dx, dy)
        if rank == 0:
            nearest_def_dist = dist
        if dist > radius_m and rank > 0:
            continue

        # Target point ahead of carrier along strong-foot driving corridor
        target_xy = np.array([cx + 3.0 * drive_vec[0], cy + 3.0 * drive_vec[1]], dtype=float)
        to_target = target_xy - np.array([float(d["x"]), float(d["y"])], dtype=float)
        tn = float(np.hypot(to_target[0], to_target[1]))
        u_t = to_target / tn if tn > 1e-6 else np.array([-1.0, 0.0], dtype=float)

        d_fwd = np.asarray(d.get("hip_fwd", [-1.0, 0.0]), dtype=float)
        dn = float(np.hypot(d_fwd[0], d_fwd[1]))
        u_d = d_fwd / dn if dn > 1e-6 else np.array([-1.0, 0.0], dtype=float)

        signed_ang = float(math.atan2(u_d[0] * u_t[1] - u_d[1] * u_t[0], u_d[0] * u_t[0] + u_d[1] * u_t[1]))
        sigma = float(
            compute_stance_penalty(
                np.array([signed_ang]),
                stance=str(d.get("stance", "Square")),
                stance_lead_m=float(d.get("stance_lead_m", 0.0)),
            )[0]
        )
        prox_w = math.exp(-dist / 8.0)
        def_exposures.append(sigma * prox_w)

        if rank == 0:
            nearest_pivot_deg = abs(math.degrees(signed_ang))
            nearest_back_turned = 1.0 if abs(signed_ang) > BACK_TURNED_RAD else 0.0
            nearest_weak_stance = sigma

    def_weak_exposure = float(np.mean(def_exposures)) if def_exposures else 0.0

    # Evaluate adjacent teammates' strong-foot passing lanes
    lane_scores: list[float] = []
    for tm in teammates:
        if tm.get("player_id") == carrier.get("player_id") or tm.get("is_gk", False):
            continue
        pvec = np.array([float(tm["x"]) - cx, float(tm["y"]) - cy], dtype=float)
        pdist = float(np.hypot(pvec[0], pvec[1]))
        if pdist < 3.0 or pdist > 30.0:
            continue
        ap = pass_release_favorability(c_fwd, pvec, dom, w_comfort)
        tm_goal = np.array([52.5 - float(tm["x"]), -float(tm["y"])], dtype=float)
        ar = reception_favorability(
            np.asarray(tm.get("hip_fwd", [1.0, 0.0]), dtype=float),
            pvec,
            tm_goal,
            str(tm.get("dominant_foot", "R")),
            float(tm.get("weak_foot_comfort", 0.51)),
        )
        lane_scores.append(0.5 * (ap + ar))

    lane_scores.sort(reverse=True)
    passing_lane_edge = float(np.mean(lane_scores[:2])) if lane_scores else 0.0
    carrier_drive_align = pass_release_favorability(c_fwd, drive_vec, dom, w_comfort)

    net_edge = float(np.clip(0.45 * def_weak_exposure + 0.30 * carrier_drive_align + 0.25 * passing_lane_edge, -1.0, 1.0))

    return {
        "net_matchup_edge": round(net_edge, 4),
        "defender_weak_side_exposure": round(def_weak_exposure, 4),
        "passing_lane_edge": round(passing_lane_edge, 4),
        "nearest_def_dist": round(nearest_def_dist, 3),
        "nearest_def_pivot_deg": round(nearest_pivot_deg, 2),
        "nearest_def_back_turned": nearest_back_turned,
        "nearest_def_weak_stance": round(nearest_weak_stance, 4),
    }
