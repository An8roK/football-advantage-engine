"""Offline streaming pipeline building compact Parquet/JSON artifacts for Cloud Run.

Usage:
    python -m src.build_artifacts --sample-only   # Fast smoke build (~seconds)
    python -m src.build_artifacts                 # Full 2-match build (1925299 & 1996435)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingest import (
    ARTIFACTS_DIR,
    FULL_MATCH_IDS,
    SAMPLE_MATCH_ID,
    compute_frame_velocities,
    ensure_match_files,
    get_attacking_sign,
    load_dynamic_events,
    load_match_metadata,
    load_phases_of_play,
    stream_pose_frames,
    stream_tracking_frames,
)
from src.matrix import compute_seam_expansion
from src.physics import (
    compute_local_matchup_tensor,
    decompose_action_control_delta,
    evaluate_frame_state,
    pass_release_favorability,
    reception_favorability,
)
from src.pose import (
    build_frame_player_states,
    infer_player_footedness,
    tracking_to_pose_frame,
)
from src.validation import run_validation_audit


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    return str(val).strip().lower() in ("true", "1", "yes")


def _select_curated_phases(
    phases_df: pd.DataFrame,
    match_id: str,
    sample_only: bool = False,
) -> list[dict[str, Any]]:
    """Select curated high-impact 10 Hz replay sequences (10-14s max each) per match."""
    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()

    def _add_phase(row: pd.Series, tag: str, max_frames: int = 125) -> None:
        pidx = int(row["index"])
        if pidx in seen_indices:
            return
        f_start = int(row["frame_start"])
        f_end = int(row["frame_end"])
        # Keep replay window tight around the climax (last max_frames) so JSON stays compact
        if f_end - f_start > max_frames:
            f_start = f_end - max_frames
        if f_end <= f_start + 5:
            return
        seen_indices.add(pidx)
        selected.append(
            {
                "match_id": str(match_id),
                "phase_index": pidx,
                "tag": tag,
                "period": int(row["period"]),
                "team_in_possession_id": int(row["team_in_possession_id"]),
                "team_in_possession_name": str(row.get("team_in_possession_shortname", "")),
                "tip_phase": str(row.get("team_in_possession_phase_type", "")),
                "otip_phase": str(row.get("team_out_of_possession_phase_type", "")),
                "time_start": str(row.get("time_start", "")),
                "frame_start": f_start,
                "frame_end": f_end,
            }
        )

    # Always prioritize Phase 406 for match 1925299 (matches official 3D sample)
    if str(match_id) == "1925299":
        p406 = phases_df[phases_df["index"] == 406]
        if not p406.empty:
            _add_phase(p406.iloc[0], "Featured Phase 406 (3D Sample)", max_frames=125)

    if sample_only:
        return selected

    # Include goal-producing phases
    goals = phases_df[phases_df["team_possession_lead_to_goal"].apply(_as_bool)]
    for _, r in goals.head(3).iterrows():
        _add_phase(r, "Goal Sequence", max_frames=110)

    # Include top shot-creating finish/quick_break/create phases
    shots = phases_df[phases_df["team_possession_lead_to_shot"].apply(_as_bool)]
    for _, r in shots.head(4).iterrows():
        if len(selected) >= 5:
            break
        _add_phase(r, f"Shot Creation ({r['team_in_possession_phase_type']})", max_frames=95)

    # Include a high-tempo build-up / create phase if still under cap
    create_phases = phases_df[
        phases_df["team_in_possession_phase_type"].isin(["quick_break", "finish", "create"])
        & (phases_df["n_player_possessions_in_phase"] >= 3)
    ]
    for _, r in create_phases.head(4).iterrows():
        if len(selected) >= 6:
            break
        _add_phase(r, f"High-Tempo {str(r['team_in_possession_phase_type']).title()}", max_frames=90)

    return selected


def _compact_player_for_replay(p: dict[str, Any]) -> dict[str, Any]:
    """Serialize player state compactly (3 decimals) for fast Cloud Run JSON loading."""
    return {
        "id": int(p["player_id"]),
        "n": p["short_name"],
        "r": p["role"],
        "a": bool(p["is_attacking"]),
        "g": bool(p["is_gk"]),
        "x": round(float(p["x"]), 2),
        "y": round(float(p["y"]), 2),
        "vx": round(float(p["vx"]), 2),
        "vy": round(float(p["vy"]), 2),
        "fx": round(float(p["hip_fwd"][0]), 3),
        "fy": round(float(p["hip_fwd"][1]), 3),
        "tw": round(float(math.degrees(p["torso_twist"])), 1),
        "st": p["stance"],
        "sl": round(float(p["stance_lead_m"]), 3),
        "ps": "3d" if p["pose_source"] == "3d_pose" else "fb",
        "df": p["dominant_foot"],
        "wf": round(float(p["weak_foot_comfort"]), 2),
    }


def _process_single_match(
    match_id: str,
    sample_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Run streaming ingestion, 3D pose sync, event scoring, and replay building for one match."""
    t_match_start = time.perf_counter()
    files = ensure_match_files(match_id, sample_only=sample_only)
    match_meta = load_match_metadata(files["match_json"])
    events_df = load_dynamic_events(files["dynamic_events"])
    phases_df = load_phases_of_play(files["phases"])

    home_name = str(match_meta["home_team"].get("short_name", "Home"))
    away_name = str(match_meta["away_team"].get("short_name", "Away"))
    fixture_label = (
        f"{home_name} {match_meta.get('home_team_score', '')}–"
        f"{match_meta.get('away_team_score', '')} {away_name}"
    )

    # Filter to scored tactical events: player_possession and on_ball_engagement
    scored_src = events_df[events_df["event_type"].isin(["player_possession", "on_ball_engagement"])].copy()

    if sample_only:
        # Keep all events overlapping sample Phase 406 (49795..49918) + deterministic 260-event sample for CV
        in_sample = (scored_src["frame_start"] <= 49918) & (scored_src["frame_end"] >= 49795)
        outside = scored_src[~in_sample]
        extra = outside.iloc[:: max(1, len(outside) // 240)].head(240)
        scored_src = pd.concat([scored_src[in_sample], extra], ignore_index=True)

    curated_specs = _select_curated_phases(phases_df, match_id, sample_only=sample_only)

    # 1. Precompute exact set of needed 10 FPS tracking frames & 25 FPS pose frames
    target_tracking: set[int] = set()
    target_pose: set[int] = set()

    for row in scored_src.itertuples(index=False):
        fs, fe = int(row.frame_start), int(row.frame_end)
        target_tracking.add(fs)
        target_tracking.add(fe)
        target_pose.add(tracking_to_pose_frame(fs))
        target_pose.add(tracking_to_pose_frame(fe))
        if getattr(row, "event_type", "") == "player_possession" and fe - fs >= 2:
            # Extra touch frames for empirical dominant-foot contact inference
            target_pose.add(tracking_to_pose_frame(fs + 1))
            target_pose.add(tracking_to_pose_frame(fe - 1))

    for spec in curated_specs:
        for f in range(spec["frame_start"], spec["frame_end"] + 1):
            target_tracking.add(f)
            target_pose.add(tracking_to_pose_frame(f))

    # 2. Stream tracking JSONL & 3D pose archive in constant memory
    tracking_frames = stream_tracking_frames(
        files["tracking"], target_tracking, include_velocity_neighbors=True
    )
    pose_frames = stream_pose_frames(files["pose_archive"], target_pose_frames=target_pose)

    # 3. Empirical dominant-foot inference from 3D toe-ball contacts
    footedness_df = infer_player_footedness(pose_frames, scored_src, match_meta, match_id)
    footedness_map = {
        int(r["player_id"]): {
            "dominant_foot": r["dominant_foot"],
            "weak_foot_comfort": float(r["weak_foot_comfort"]),
        }
        for _, r in footedness_df.iterrows()
    }

    # Cache frame evaluations keyed by (tracking_frame, attacking_team_id, period) to avoid redundant grid work
    frame_cache: dict[tuple[int, int, int], tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]] = {}

    def _get_eval_frame(tf: int, att_team_id: int, period: int) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]] | None:
        key = (tf, att_team_id, period)
        if key in frame_cache:
            return frame_cache[key]
        tr = tracking_frames.get(tf)
        pf = pose_frames.get(tracking_to_pose_frame(tf))
        if tr is None and pf is None:
            return None
        sign = get_attacking_sign(match_meta, att_team_id, period)
        vels = compute_frame_velocities(tracking_frames, tf)
        p_states, b_xy = build_frame_player_states(
            tr, pf, vels, match_meta, att_team_id, sign, footedness_map
        )
        if len(p_states) < 4:
            return None
        ev = evaluate_frame_state(p_states)
        frame_cache[key] = (p_states, b_xy, ev)
        return frame_cache[key]

    # 4. Score all dynamic events
    scored_rows: list[dict[str, Any]] = []
    home_id = int(match_meta["home_team"]["id"])
    away_id = int(match_meta["away_team"]["id"])

    for row in scored_src.itertuples(index=False):
        etype = str(row.event_type)
        fs, fe = int(row.frame_start), int(row.frame_end)
        period = int(row.period)
        event_team = int(row.team_id)
        # For on_ball_engagement, the defending player's team_id is the out-of-possession team
        att_team = event_team if etype == "player_possession" else (away_id if event_team == home_id else home_id)

        f0 = _get_eval_frame(fs, att_team, period)
        f1 = _get_eval_frame(fe, att_team, period)
        if f0 is None or f1 is None:
            continue

        p0, ball0, e0 = f0
        p1, ball1, e1 = f1
        decomp = decompose_action_control_delta(p0, p1, e0, e1)
        seam = compute_seam_expansion(e0["matrix"], e1["matrix"])

        focal_pid = int(row.player_id) if not pd.isna(row.player_id) else -1
        carrier_pid = (
            focal_pid
            if etype == "player_possession"
            else (int(row.player_in_possession_id) if not pd.isna(getattr(row, "player_in_possession_id", np.nan)) else -1)
        )

        p0_by_id = {p["player_id"]: p for p in p0}
        p1_by_id = {p["player_id"]: p for p in p1}

        focal_p0 = p0_by_id.get(focal_pid)
        focal_p1 = p1_by_id.get(focal_pid)
        carrier_p0 = p0_by_id.get(carrier_pid)
        carrier_p1 = p1_by_id.get(carrier_pid)

        # Provenance: pure 3d_pose if focal player has 3D pose at start or end of event
        has_3d = (
            (focal_p0 is not None and focal_p0["pose_source"] == "3d_pose")
            or (focal_p1 is not None and focal_p1["pose_source"] == "3d_pose")
        )
        pose_source = "3d_pose" if has_3d else "kinematic_fallback"
        n_3d_pitch = sum(1 for p in p1 if p["pose_source"] == "3d_pose")
        pose_3d_frac = n_3d_pitch / max(1, len(p1))

        # Local matchup tensor for ball carrier
        att_list = [p for p in p1 if p["is_attacking"]]
        def_list = [p for p in p1 if not p["is_attacking"]]
        if carrier_p1 is not None:
            matchup = compute_local_matchup_tensor(carrier_p1, def_list, att_list)
            carrier_twist_deg = abs(math.degrees(carrier_p1["torso_twist"]))
            dom_foot = carrier_p1["dominant_foot"]
            weak_comfort = float(carrier_p1["weak_foot_comfort"])
        else:
            matchup = {
                "net_matchup_edge": 0.0,
                "defender_weak_side_exposure": 0.0,
                "passing_lane_edge": 0.0,
                "nearest_def_dist": 15.0,
                "nearest_def_pivot_deg": 0.0,
                "nearest_def_back_turned": 0.0,
                "nearest_def_weak_stance": 0.0,
            }
            carrier_twist_deg = 0.0
            dom_foot = "R"
            weak_comfort = 0.51

        # Strong/weak-side pass release & reception favorability
        a_pass = 0.0
        a_recv = 0.0
        target_pid = getattr(row, "player_targeted_id", np.nan)
        if etype == "player_possession" and carrier_p1 is not None:
            if not pd.isna(target_pid):
                tp = p1_by_id.get(int(target_pid))
                if tp is not None:
                    pvec = np.array([tp["x"] - carrier_p1["x"], tp["y"] - carrier_p1["y"]], dtype=float)
                    a_pass = pass_release_favorability(carrier_p1["hip_fwd"], pvec, dom_foot, weak_comfort)
                    t_goal = np.array([52.5 - tp["x"], -tp["y"]], dtype=float)
                    a_recv = reception_favorability(
                        tp["hip_fwd"], pvec, t_goal, tp["dominant_foot"], tp["weak_foot_comfort"]
                    )
            # Reception favorability at t0 from incoming pass direction
            if carrier_p0 is not None and not pd.isna(getattr(row, "pass_angle_received", np.nan)):
                ang_rad = math.radians(float(row.pass_angle_received))
                in_vec = np.array([math.cos(ang_rad), math.sin(ang_rad)], dtype=float)
                c_goal = np.array([52.5 - carrier_p0["x"], -carrier_p0["y"]], dtype=float)
                a_recv_start = reception_favorability(
                    carrier_p0["hip_fwd"], in_vec, c_goal, carrier_p0["dominant_foot"], carrier_p0["weak_foot_comfort"]
                )
                if a_recv == 0.0:
                    a_recv = a_recv_start

        # If on_ball_engagement, measure the engaging defender's own pivot/stance exposure against carrier
        if etype == "on_ball_engagement" and focal_p1 is not None and carrier_p1 is not None:
            single_m = compute_local_matchup_tensor(carrier_p1, [focal_p1], att_list)
            matchup["nearest_def_dist"] = single_m["nearest_def_dist"]
            matchup["nearest_def_pivot_deg"] = single_m["nearest_def_pivot_deg"]
            matchup["nearest_def_back_turned"] = single_m["nearest_def_back_turned"]
            matchup["nearest_def_weak_stance"] = single_m["nearest_def_weak_stance"]

        # Extract SkillCorner ground-truth outcome targets
        end_type = str(getattr(row, "end_type", "") or "")
        start_type = str(getattr(row, "start_type", "") or "")
        pass_outcome = str(getattr(row, "pass_outcome", "") or "")
        beaten = _as_bool(getattr(row, "beaten_by_possession", False)) or _as_bool(
            getattr(row, "beaten_by_movement", False)
        )
        lb_any = (
            _as_bool(getattr(row, "first_line_break", False))
            or _as_bool(getattr(row, "second_last_line_break", False))
            or _as_bool(getattr(row, "last_line_break", False))
        )
        pass_ok = pass_outcome == "successful"
        lead_shot = _as_bool(getattr(row, "lead_to_shot", False))
        lead_goal = _as_bool(getattr(row, "lead_to_goal", False))
        poss_epv = float(row.possession_epv_total) if not pd.isna(getattr(row, "possession_epv_total", np.nan)) else np.nan
        pass_epv = float(row.pass_epv_total) if not pd.isna(getattr(row, "pass_epv_total", np.nan)) else np.nan
        xshot_max = float(row.xshot_player_possession_max) if not pd.isna(getattr(row, "xshot_player_possession_max", np.nan)) else np.nan

        scored_rows.append(
            {
                "match_id": str(match_id),
                "fixture": fixture_label,
                "event_id": str(row.event_id),
                "event_type": etype,
                "event_subtype": str(getattr(row, "event_subtype", "") or ""),
                "phase_index": int(row.phase_index) if not pd.isna(getattr(row, "phase_index", np.nan)) else -1,
                "tip_phase": str(getattr(row, "team_in_possession_phase_type", "") or ""),
                "period": period,
                "minute": int(row.minute_start) if not pd.isna(getattr(row, "minute_start", np.nan)) else 0,
                "time_start": str(getattr(row, "time_start", "")),
                "frame_start": fs,
                "frame_end": fe,
                "player_id": focal_pid,
                "player_name": str(getattr(row, "player_name", "")),
                "player_position": str(getattr(row, "player_position", "")),
                "team_shortname": str(getattr(row, "team_shortname", "")),
                "start_type": start_type,
                "end_type": end_type,
                "one_touch": _as_bool(getattr(row, "one_touch", False)),
                "carry": _as_bool(getattr(row, "carry", False)),
                "pass_outcome": pass_outcome,
                "pose_source": pose_source,
                "pose_3d_pitch_frac": round(pose_3d_frac, 3),
                "dominant_foot": dom_foot,
                "weak_foot_comfort": round(weak_comfort, 4),
                "ball_x_start": round(float(ball0[0]), 2),
                "ball_y_start": round(float(ball0[1]), 2),
                "iso_c_t0": round(decomp["iso_c_t0"], 5),
                "iso_c_t1": round(decomp["iso_c_t1"], 5),
                "iso_delta_c": round(decomp["iso_delta_c"], 5),
                "aniso_c_t0": round(decomp["c_t0"], 5),
                "aniso_c_t1": round(decomp["c_t1"], 5),
                "aniso_delta_c": round(decomp["delta_c_total"], 5),
                "aniso_minus_iso_delta": round(decomp["aniso_minus_iso_delta"], 5),
                "delta_c_pos": round(decomp["delta_c_pos"], 5),
                "delta_c_orient": round(decomp["delta_c_orient"], 5),
                "delta_c_matrix": round(decomp["delta_c_matrix"], 5),
                "dominant_driver": decomp["dominant_driver"],
                "lambda2_t0": e0["matrix"]["lambda2"],
                "lambda2_t1": e1["matrix"]["lambda2"],
                "lambda2_delta": seam["lambda2_delta"],
                "hole_score_t0": e0["matrix"]["hole_score"],
                "hole_score_t1": e1["matrix"]["hole_score"],
                "hole_score_delta": seam["h_delta"],
                "expanding_seam_score": seam["expanding_seam_score"],
                "a_pass": round(a_pass, 4),
                "a_recv": round(a_recv, 4),
                "net_matchup_edge": matchup["net_matchup_edge"],
                "defender_weak_side_exposure": matchup["defender_weak_side_exposure"],
                "passing_lane_edge": matchup["passing_lane_edge"],
                "nearest_def_dist": matchup["nearest_def_dist"],
                "nearest_def_pivot_deg": matchup["nearest_def_pivot_deg"],
                "nearest_def_back_turned": matchup["nearest_def_back_turned"],
                "nearest_def_weak_stance": matchup["nearest_def_weak_stance"],
                "carrier_abs_twist_deg": round(carrier_twist_deg, 2),
                "possession_epv_total": poss_epv,
                "pass_epv_total": pass_epv,
                "xshot_max": xshot_max,
                "lead_to_shot": lead_shot,
                "lead_to_goal": lead_goal,
                "target_defender_beaten": int(beaten),
                "target_pass_completed": int(pass_ok),
                "target_line_break_success": int(lb_any and pass_ok),
            }
        )

    scored_df = pd.DataFrame(scored_rows)

    # Compute High-EPV / Shot-Creation binary target (top quartile EPV or shot creation)
    if not scored_df.empty:
        poss_mask = scored_df["event_type"] == "player_possession"
        epv_comb = scored_df.loc[poss_mask, ["possession_epv_total", "pass_epv_total"]].max(axis=1)
        epv_q75 = float(epv_comb.quantile(0.75)) if epv_comb.notna().any() else 0.01
        scored_df["target_high_epv_or_shot"] = (
            (scored_df[["possession_epv_total", "pass_epv_total"]].max(axis=1) >= epv_q75)
            | scored_df["lead_to_shot"]
        ).astype(int)
    else:
        scored_df["target_high_epv_or_shot"] = 0

    # 5. Build compact curated 10 Hz replay sequences for Tab 2
    replays: list[dict[str, Any]] = []
    for spec in curated_specs:
        r_frames: list[dict[str, Any]] = []
        p_t0: list[dict[str, Any]] | None = None
        e_t0: dict[str, Any] | None = None

        for tf in range(spec["frame_start"], spec["frame_end"] + 1):
            ev_bundle = _get_eval_frame(tf, spec["team_in_possession_id"], spec["period"])
            if ev_bundle is None:
                continue
            p_st, b_xy, ev = ev_bundle
            if p_t0 is None:
                p_t0, e_t0 = p_st, ev
            assert e_t0 is not None
            decomp = decompose_action_control_delta(p_t0, p_st, e_t0, ev)
            seam = compute_seam_expansion(e_t0["matrix"], ev["matrix"])
            r_frames.append(
                {
                    "f": tf,
                    "pf": tracking_to_pose_frame(tf),
                    "bx": round(float(b_xy[0]), 2),
                    "by": round(float(b_xy[1]), 2),
                    "c_aniso": round(ev["aniso_threat"], 5),
                    "c_iso": round(ev["iso_threat"], 5),
                    "dc": round(decomp["delta_c_total"], 5),
                    "dc_pos": round(decomp["delta_c_pos"], 5),
                    "dc_ori": round(decomp["delta_c_orient"], 5),
                    "dc_mat": round(decomp["delta_c_matrix"], 5),
                    "lam2": ev["matrix"]["lambda2"],
                    "hole": ev["matrix"]["hole_score"],
                    "comp": ev["matrix"]["mean_compactness"],
                    "n3d": sum(1 for p in p_st if p["pose_source"] == "3d_pose"),
                    "seam_tri": (
                        seam["highlighted_triangle"]["coords"]
                        if seam.get("highlighted_triangle")
                        else None
                    ),
                    "players": [_compact_player_for_replay(p) for p in p_st],
                }
            )

        if len(r_frames) >= 5:
            max_dc = max(fr["dc"] for fr in r_frames)
            min_lam2 = min(fr["lam2"] for fr in r_frames)
            replays.append(
                {
                    "id": f"{match_id}_phase_{spec['phase_index']}",
                    "match_id": str(match_id),
                    "fixture": fixture_label,
                    "label": (
                        f"{fixture_label} | Phase {spec['phase_index']} — {spec['tag']} "
                        f"({spec['team_in_possession_name']} · {spec['time_start']})"
                    ),
                    "tag": spec["tag"],
                    "team_in_possession": spec["team_in_possession_name"],
                    "tip_phase": spec["tip_phase"],
                    "otip_phase": spec["otip_phase"],
                    "max_delta_c": round(max_dc, 5),
                    "min_lambda2": round(min_lam2, 4),
                    "frames": r_frames,
                }
            )

    match_summary = {
        "match_id": str(match_id),
        "fixture": fixture_label,
        "pose_frames_loaded": len(pose_frames),
        "tracking_frames_loaded": len(tracking_frames),
        "scored_events": int(len(scored_df)),
        "pose_3d_events": int((scored_df["pose_source"] == "3d_pose").sum()) if not scored_df.empty else 0,
        "replays_built": len(replays),
        "elapsed_sec": round(time.perf_counter() - t_match_start, 2),
    }
    return scored_df, footedness_df, replays, match_summary


def build_all_artifacts(
    sample_only: bool = False,
    output_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Run offline build and write compact Parquet/JSON artifacts."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    match_ids = (SAMPLE_MATCH_ID,) if sample_only else FULL_MATCH_IDS
    all_events: list[pd.DataFrame] = []
    all_foot: list[pd.DataFrame] = []
    all_replays: list[dict[str, Any]] = []
    match_summaries: list[dict[str, Any]] = []

    for mid in match_ids:
        ev_df, foot_df, replays, summary = _process_single_match(mid, sample_only=sample_only)
        all_events.append(ev_df)
        all_foot.append(foot_df)
        all_replays.extend(replays)
        match_summaries.append(summary)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    foot_df = pd.concat(all_foot, ignore_index=True) if all_foot else pd.DataFrame()

    # Recompute pooled top-quartile EPV target across all matches for consistent multi-match OOF audit
    if not events_df.empty:
        poss_mask = events_df["event_type"] == "player_possession"
        epv_comb = events_df.loc[poss_mask, ["possession_epv_total", "pass_epv_total"]].max(axis=1)
        epv_q75 = float(epv_comb.quantile(0.75)) if epv_comb.notna().any() else 0.01
        events_df["target_high_epv_or_shot"] = (
            (events_df[["possession_epv_total", "pass_epv_total"]].max(axis=1) >= epv_q75)
            | events_df["lead_to_shot"]
        ).astype(int)

    audit = run_validation_audit(events_df)

    # Top 5 Orientation / Matrix ΔC moments (prioritizing 3d_pose player possessions where orient+matrix dominate)
    top_moments: list[dict[str, Any]] = []
    if not events_df.empty:
        poss_only = events_df[events_df["event_type"] == "player_possession"].copy()
        poss_only["orient_matrix_gain"] = poss_only["delta_c_orient"] + poss_only["delta_c_matrix"]
        preferred = poss_only[
            (poss_only["pose_source"] == "3d_pose")
            & (poss_only["dominant_driver"].isin(["Orientation & Wrong-Footing", "Matrix Distortion"]))
        ]
        pool = preferred if len(preferred) >= 5 else poss_only
        top5 = pool.sort_values("orient_matrix_gain", ascending=False).head(5)
        for _, r in top5.iterrows():
            action_desc = r["end_type"] or r["start_type"] or "possession"
            if r["carry"]:
                action_desc = f"carry -> {action_desc}"
            elif r["one_touch"]:
                action_desc = f"one_touch -> {action_desc}"
            top_moments.append(
                {
                    "match_id": str(r["match_id"]),
                    "fixture": str(r["fixture"]),
                    "minute": int(r["minute"]),
                    "time_start": str(r["time_start"]),
                    "player_name": str(r["player_name"]),
                    "player_position": str(r["player_position"]),
                    "team_shortname": str(r["team_shortname"]),
                    "action_type": action_desc,
                    "pose_source": str(r["pose_source"]),
                    "iso_delta_c": round(float(r["iso_delta_c"]), 5),
                    "aniso_delta_c": round(float(r["aniso_delta_c"]), 5),
                    "delta_c_orient": round(float(r["delta_c_orient"]), 5),
                    "delta_c_matrix": round(float(r["delta_c_matrix"]), 5),
                    "dominant_driver": str(r["dominant_driver"]),
                }
            )

    # Write compact artifacts
    events_path = output_dir / "dynamic_events_enriched.parquet"
    foot_path = output_dir / "player_footedness.parquet"
    replays_path = output_dir / "curated_replays.json"
    audit_path = output_dir / "validation_audit.json"
    manifest_path = output_dir / "artifact_manifest.json"

    events_df.to_parquet(events_path, index=False, compression="snappy")
    foot_df.to_parquet(foot_path, index=False, compression="snappy")
    with open(replays_path, "w", encoding="utf-8") as f:
        json.dump(all_replays, f, separators=(",", ":"))
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    elapsed_total = round(time.perf_counter() - t0, 2)
    artifact_files = [events_path, foot_path, replays_path, audit_path]
    total_bytes = sum(p.stat().st_size for p in artifact_files if p.exists())

    manifest = {
        "mode": "sample_only" if sample_only else "full_2_match",
        "matches": match_summaries,
        "total_runtime_sec": elapsed_total,
        "artifacts_total_mb": round(total_bytes / (1024 * 1024), 3),
        "file_sizes_kb": {p.name: round(p.stat().st_size / 1024, 1) for p in artifact_files},
        "total_scored_events": audit["total_scored_events"],
        "pose_3d_events": audit["pose_3d_events"],
        "kinematic_fallback_events": audit["kinematic_fallback_events"],
        "pose_3d_pct": audit["pose_3d_pct"],
        "top_5_moments": top_moments,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Include manifest file size in final total
    manifest["artifacts_total_mb"] = round(
        (total_bytes + manifest_path.stat().st_size) / (1024 * 1024), 3
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _print_summary(manifest: dict[str, Any], audit: dict[str, Any]) -> None:
    print(f"\n=== Build Complete ({manifest['mode']}) ===")
    print(f"Total Runtime: {manifest['total_runtime_sec']:.2f} s")
    print(f"Artifacts Size: {manifest['artifacts_total_mb']:.3f} MB ({manifest['file_sizes_kb']})")
    print(
        f"Scored Events: {manifest['total_scored_events']} | "
        f"3D Pose: {manifest['pose_3d_events']} ({manifest['pose_3d_pct']:.1f}%) | "
        f"Kinematic Fallback: {manifest['kinematic_fallback_events']}"
    )
    print("\n--- OOF Validation Comparison ---")
    for t in audit["tasks"]:
        print(f"\nTask: {t['task_label']}")
        for sk in ("all_events", "3d_pose_only"):
            sl = t["slices"][sk]
            b, f = sl["baseline"], sl["biomechanics"]
            print(
                f"  [{sk}] n={b['n_samples']} (pos={b['n_positives']}) | "
                f"Base AUC={b['roc_auc']} Brier={b['brier']} -> "
                f"Full AUC={f['roc_auc']} (d={sl['delta_auc']}) Brier={f['brier']} (d={sl['delta_brier']})"
            )

    print("\n--- Top 5 Orientation / Matrix ΔC Moments ---")
    for i, m in enumerate(manifest["top_5_moments"], 1):
        print(
            f"  {i}. {m['fixture']} ({m['minute']}') | {m['player_name']} ({m['player_position']}) | "
            f"{m['action_type']} | iso ΔC={m['iso_delta_c']:+.4f} vs aniso ΔC={m['aniso_delta_c']:+.4f} | "
            f"driver={m['dominant_driver']} [{m['pose_source']}]"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build SkillCorner biomechanical artifacts.")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Fast smoke build using sample_1925299_phase406.jsonl.gz instead of full HF zips.",
    )
    args = parser.parse_args()
    manifest = build_all_artifacts(sample_only=args.sample_only)
    with open(ARTIFACTS_DIR / "validation_audit.json", "r", encoding="utf-8") as f:
        audit = json.load(f)
    _print_summary(manifest, audit)


if __name__ == "__main__":
    main()
