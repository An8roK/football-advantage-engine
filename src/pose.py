"""3D Body Pose biomechanical feature extraction, off-camera fallback, and footedness inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

POSE_FPS = 25.0
TRACKING_FPS = 10.0
STANCE_THRESH_M = 0.10
VELOCITY_FALLBACK_SPEED_MS = 1.2

# Empirical foot-contact thresholds & Beta-Binomial prior (right-footed baseline)
FOOT_CONTACT_MAX_DIST_M = 0.85
FOOT_CONTACT_MIN_MARGIN_M = 0.10
PRIOR_ALPHA_R = 4.0
PRIOR_BETA_L = 1.0


def tracking_to_pose_frame(tracking_frame: int) -> int:
    """Map 10 FPS tracking frame index to nearest 25 FPS 3D-pose frame index."""
    return int(round(2.5 * int(tracking_frame)))


def _joint_xy(joints: dict[str, Any], name: str) -> tuple[float, float] | None:
    """Extract horizontal (x, y) from verified SkillCorner joint schema {'xyz': [x, y, z], 'p90_mae_cm': ...}."""
    entry = joints.get(name)
    if not isinstance(entry, dict):
        return None
    xyz = entry.get("xyz")
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
        return None
    x, y = xyz[0], xyz[1]
    if x is None or y is None:
        return None
    return (float(x), float(y))


def _joint_mae(joints: dict[str, Any], name: str) -> float | None:
    entry = joints.get(name)
    if not isinstance(entry, dict):
        return None
    val = entry.get("p90_mae_cm")
    if isinstance(val, (int, float)) and not math.isnan(val):
        return float(val)
    return None


def compute_forward_from_lr(
    left_xy: tuple[float, float] | np.ndarray,
    right_xy: tuple[float, float] | np.ndarray,
) -> np.ndarray | None:
    """Compute horizontal forward unit vector from anatomical Left -> Right joint pair.

    In SkillCorner's right-handed (+z up) coordinate frame, rotating (r - l) by +90 deg CCW
    yields (-dy, dx) / ||delta||, verified 99.5%-100% aligned with nose and BigToe vectors.
    """
    dx = float(right_xy[0]) - float(left_xy[0])
    dy = float(right_xy[1]) - float(left_xy[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return np.array([-dy / norm, dx / norm], dtype=float)


def compute_signed_twist(hip_fwd: np.ndarray, sh_fwd: np.ndarray) -> float:
    """Compute signed torso twist angle in radians: atan2(hip x sh, hip . sh)."""
    cross = float(hip_fwd[0] * sh_fwd[1] - hip_fwd[1] * sh_fwd[0])
    dot = float(hip_fwd[0] * sh_fwd[0] + hip_fwd[1] * sh_fwd[1])
    return float(math.atan2(cross, dot))


def classify_stance(
    l_ankle_xy: tuple[float, float] | np.ndarray,
    r_ankle_xy: tuple[float, float] | np.ndarray,
    hip_fwd: np.ndarray,
    threshold_m: float = STANCE_THRESH_M,
) -> tuple[str, float]:
    """Classify tackling/stride stance from (lAnkle - rAnkle) . hip_fwd."""
    s = float(
        (float(l_ankle_xy[0]) - float(r_ankle_xy[0])) * float(hip_fwd[0])
        + (float(l_ankle_xy[1]) - float(r_ankle_xy[1])) * float(hip_fwd[1])
    )
    if s > threshold_m:
        return "Left-Forward", s
    if s < -threshold_m:
        return "Right-Forward", s
    return "Square", s


def kinematic_fallback_facing(
    pos_xy: tuple[float, float] | np.ndarray,
    vel_xy: tuple[float, float] | np.ndarray,
    ball_xy: tuple[float, float] | np.ndarray | None,
) -> np.ndarray:
    """Smooth off-camera facing estimate blending velocity (when speed >= 1.2 m/s) with ball-facing."""
    u_ball = np.array([1.0, 0.0], dtype=float)
    if ball_xy is not None:
        bx = float(ball_xy[0]) - float(pos_xy[0])
        by = float(ball_xy[1]) - float(pos_xy[1])
        bnorm = math.hypot(bx, by)
        if bnorm > 1e-6:
            u_ball = np.array([bx / bnorm, by / bnorm], dtype=float)

    vx, vy = float(vel_xy[0]), float(vel_xy[1])
    spd = math.hypot(vx, vy)
    if spd >= VELOCITY_FALLBACK_SPEED_MS:
        u_vel = np.array([vx / spd, vy / spd], dtype=float)
        # Smooth ramp from 0.55 weight at 1.2 m/s up to 0.85 weight at high sprint speeds.
        w_vel = min(0.85, 0.55 + 0.08 * (spd - VELOCITY_FALLBACK_SPEED_MS))
        blended = w_vel * u_vel + (1.0 - w_vel) * u_ball
        bn = float(np.hypot(blended[0], blended[1]))
        if bn > 1e-6:
            return blended / bn
        return u_vel

    return u_ball


def extract_player_biomechanics(
    pose_player: dict[str, Any] | None,
    pos_xy: tuple[float, float],
    vel_xy: tuple[float, float],
    ball_xy: tuple[float, float] | None,
    attack_sign: float = 1.0,
) -> dict[str, Any]:
    """Extract normalized hip/shoulder facing, torso twist, stance, and provenance tag."""
    sign = 1.0 if attack_sign >= 0 else -1.0
    joints = pose_player.get("joints") if isinstance(pose_player, dict) else None

    if isinstance(joints, dict) and len(joints) > 0:
        l_hip = _joint_xy(joints, "lHip")
        r_hip = _joint_xy(joints, "rHip")
        if l_hip is not None and r_hip is not None:
            hip_fwd_raw = compute_forward_from_lr(l_hip, r_hip)
            if hip_fwd_raw is not None:
                l_sh = _joint_xy(joints, "lShoulder")
                r_sh = _joint_xy(joints, "rShoulder")
                sh_fwd_raw = (
                    compute_forward_from_lr(l_sh, r_sh)
                    if (l_sh is not None and r_sh is not None)
                    else None
                )
                if sh_fwd_raw is None:
                    sh_fwd_raw = hip_fwd_raw

                twist = compute_signed_twist(hip_fwd_raw, sh_fwd_raw)

                l_ank = _joint_xy(joints, "lAnkle")
                r_ank = _joint_xy(joints, "rAnkle")
                if l_ank is not None and r_ank is not None:
                    stance, stance_lead_m = classify_stance(l_ank, r_ank, hip_fwd_raw)
                else:
                    stance, stance_lead_m = "Square", 0.0

                l_mae = _joint_mae(joints, "lHip")
                r_mae = _joint_mae(joints, "rHip")
                hip_mae = (
                    0.5 * (l_mae + r_mae)
                    if (l_mae is not None and r_mae is not None)
                    else float("nan")
                )

                # Multiplying by attack_sign rotates 180 deg around (0, 0), preserving handedness & twist sign.
                return {
                    "hip_fwd": hip_fwd_raw * sign,
                    "sh_fwd": sh_fwd_raw * sign,
                    "torso_twist": twist,
                    "stance": stance,
                    "stance_lead_m": stance_lead_m,
                    "pose_source": "3d_pose",
                    "hip_mae_cm": hip_mae,
                }

    # Off-camera / null joints fallback
    norm_pos = (pos_xy[0] * sign, pos_xy[1] * sign)
    norm_vel = (vel_xy[0] * sign, vel_xy[1] * sign)
    norm_ball = (ball_xy[0] * sign, ball_xy[1] * sign) if ball_xy is not None else None
    fb_fwd = kinematic_fallback_facing(norm_pos, norm_vel, norm_ball)

    return {
        "hip_fwd": fb_fwd,
        "sh_fwd": fb_fwd.copy(),
        "torso_twist": 0.0,
        "stance": "Square",
        "stance_lead_m": 0.0,
        "pose_source": "kinematic_fallback",
        "hip_mae_cm": float("nan"),
    }


def build_frame_player_states(
    tracking_frame: dict[str, Any] | None,
    pose_frame: dict[str, Any] | None,
    velocities: dict[int, tuple[float, float]],
    match_meta: dict[str, Any],
    attacking_team_id: int,
    attack_sign: float,
    footedness_map: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Combine 10 FPS tracking, 25 FPS 3D pose, metadata, and footedness into normalized player states."""
    sign = 1.0 if attack_sign >= 0 else -1.0
    players_by_id = match_meta.get("_players_by_id", {})
    foot_map = footedness_map or {}

    # Index 3D pose entries by player_id
    pose_by_pid: dict[int, dict[str, Any]] = {}
    if isinstance(pose_frame, dict):
        for pp in pose_frame.get("player_data", []):
            pid = pp.get("player_id")
            if pid is not None:
                pose_by_pid[int(pid)] = pp

    # Prefer ball from pose_frame when detected, else tracking_frame
    ball_raw: tuple[float, float] | None = None
    for src_frame in (pose_frame, tracking_frame):
        if isinstance(src_frame, dict):
            bd = src_frame.get("ball_data")
            if isinstance(bd, dict) and bd.get("x") is not None and bd.get("y") is not None:
                ball_raw = (float(bd["x"]), float(bd["y"]))
                break
    ball_xy = np.array(
        [ball_raw[0] * sign, ball_raw[1] * sign] if ball_raw else [0.0, 0.0],
        dtype=float,
    )

    # Prefer tracking_frame player list for full-pitch 22-player coverage; fall back to pose_frame
    base_players = (
        tracking_frame.get("player_data", [])
        if isinstance(tracking_frame, dict) and tracking_frame.get("player_data")
        else (pose_frame.get("player_data", []) if isinstance(pose_frame, dict) else [])
    )

    states: list[dict[str, Any]] = []
    for tr in base_players:
        pid_raw = tr.get("player_id")
        x_raw, y_raw = tr.get("x"), tr.get("y")
        if pid_raw is None or x_raw is None or y_raw is None:
            continue
        pid = int(pid_raw)
        meta_p = players_by_id.get(pid)
        if not meta_p:
            continue

        team_id = int(meta_p["team_id"])
        role_info = meta_p.get("player_role") or {}
        role_acr = str(role_info.get("acronym", ""))
        is_gk = role_acr == "GK" or str(role_info.get("position_group", "")).lower() == "goalkeeper"

        vx_raw, vy_raw = velocities.get(pid, (0.0, 0.0))
        bio = extract_player_biomechanics(
            pose_player=pose_by_pid.get(pid),
            pos_xy=(float(x_raw), float(y_raw)),
            vel_xy=(vx_raw, vy_raw),
            ball_xy=ball_raw,
            attack_sign=sign,
        )
        f_info = foot_map.get(pid, {})

        states.append(
            {
                "player_id": pid,
                "short_name": str(meta_p.get("short_name", pid)),
                "number": int(meta_p.get("number", 0) or 0),
                "team_id": team_id,
                "is_attacking": team_id == int(attacking_team_id),
                "is_gk": is_gk,
                "role": role_acr,
                "x": float(x_raw) * sign,
                "y": float(y_raw) * sign,
                "vx": float(vx_raw) * sign,
                "vy": float(vy_raw) * sign,
                "hip_fwd": bio["hip_fwd"],
                "sh_fwd": bio["sh_fwd"],
                "torso_twist": bio["torso_twist"],
                "stance": bio["stance"],
                "stance_lead_m": bio["stance_lead_m"],
                "pose_source": bio["pose_source"],
                "hip_mae_cm": bio["hip_mae_cm"],
                "dominant_foot": str(f_info.get("dominant_foot", "R")),
                "weak_foot_comfort": float(f_info.get("weak_foot_comfort", 0.51)),
            }
        )

    return states, ball_xy


def infer_player_footedness(
    pose_frames: dict[int, dict[str, Any]],
    events_df: pd.DataFrame,
    match_meta: dict[str, Any],
    match_id: str,
) -> pd.DataFrame:
    """Infer per-player dominant_foot (R/L) and weak_foot_comfort [0.25, 0.90] via 3D foot-ball contacts.

    Uses Beta-Binomial shrinkage toward a right-footed population baseline when n < 5.
    """
    counts: dict[int, list[int]] = {}  # pid -> [n_right, n_left]

    poss_df = events_df[events_df["event_type"] == "player_possession"]
    for row in poss_df.itertuples(index=False):
        pid = getattr(row, "player_id", None)
        f_start = getattr(row, "frame_start", None)
        f_end = getattr(row, "frame_end", None)
        if pd.isna(pid) or pd.isna(f_start) or pd.isna(f_end):
            continue
        pid_int = int(pid)
        t_start, t_end = int(f_start), int(f_end)

        # Sample reception and release frames (plus immediate neighbors if available in cache)
        sample_tracking = {t_start, t_end}
        if t_end - t_start >= 2:
            sample_tracking.add(t_start + 1)
            sample_tracking.add(t_end - 1)

        for tf in sample_tracking:
            pf = pose_frames.get(tracking_to_pose_frame(tf))
            if not pf:
                continue
            bd = pf.get("ball_data")
            if not isinstance(bd, dict) or not bd.get("is_detected"):
                continue
            bx, by = bd.get("x"), bd.get("y")
            if bx is None or by is None:
                continue

            for pp in pf.get("player_data", []):
                if pp.get("player_id") != pid_int:
                    continue
                joints = pp.get("joints")
                if not isinstance(joints, dict):
                    break
                l_toe = _joint_xy(joints, "lBigToe")
                r_toe = _joint_xy(joints, "rBigToe")
                if l_toe is None or r_toe is None:
                    break
                d_l = math.hypot(l_toe[0] - float(bx), l_toe[1] - float(by))
                d_r = math.hypot(r_toe[0] - float(bx), r_toe[1] - float(by))
                if min(d_l, d_r) <= FOOT_CONTACT_MAX_DIST_M and abs(d_l - d_r) >= FOOT_CONTACT_MIN_MARGIN_M:
                    c = counts.setdefault(pid_int, [0, 0])
                    if d_r < d_l:
                        c[0] += 1
                    else:
                        c[1] += 1
                break

    rows: list[dict[str, Any]] = []
    home_id = int(match_meta["home_team"]["id"])
    home_name = str(match_meta["home_team"].get("short_name", "Home"))
    away_name = str(match_meta["away_team"].get("short_name", "Away"))

    for p in match_meta.get("players", []):
        pid = int(p["id"])
        n_r, n_l = counts.get(pid, [0, 0])
        n_tot = n_r + n_l
        if n_tot < 5:
            # Beta-Binomial shrinkage toward 80% right-footed baseline for low-sample players
            p_r = (n_r + PRIOR_ALPHA_R) / (n_tot + PRIOR_ALPHA_R + PRIOR_BETA_L)
            prior_shrunk = True
        else:
            # Mild Laplace smoothing for well-observed players
            p_r = (n_r + 1.0) / (n_tot + 2.0)
            prior_shrunk = False

        dom = "R" if p_r >= 0.5 else "L"
        weak_share = 1.0 - max(p_r, 1.0 - p_r)  # [0.0, 0.5]
        weak_comfort = float(np.clip(0.25 + 1.30 * weak_share, 0.25, 0.90))

        role = (p.get("player_role") or {}).get("acronym", "")
        team_id = int(p["team_id"])
        rows.append(
            {
                "match_id": str(match_id),
                "player_id": pid,
                "short_name": str(p.get("short_name", pid)),
                "team_id": team_id,
                "team_name": home_name if team_id == home_id else away_name,
                "role": str(role),
                "n_right_contacts": n_r,
                "n_left_contacts": n_l,
                "n_contacts": n_tot,
                "p_right_posterior": round(float(p_r), 4),
                "dominant_foot": dom,
                "weak_foot_comfort": round(weak_comfort, 4),
                "prior_shrunk": prior_shrunk,
            }
        )

    return pd.DataFrame(rows)
