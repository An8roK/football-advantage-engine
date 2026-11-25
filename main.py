"""Project 4: Biomechanical Football Tactical Advantage & Defensive-Matrix Engine.

4-Tab Streamlit + Plotly Workbench loading exclusively from compact artifacts/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.charts import (
    build_audit_comparison_chart,
    build_delta_c_waterfall,
    build_matchup_scatter,
    build_pitch_figure,
    build_replay_timeline,
)
from src.ingest import ARTIFACTS_DIR
from src.physics import evaluate_frame_state
from src.simulator import get_preset_scenarios, run_simulation


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    """Load precomputed Parquet/JSON artifacts into memory (fast <1s cold start)."""
    events_path = artifacts_dir / "dynamic_events_enriched.parquet"
    foot_path = artifacts_dir / "player_footedness.parquet"
    replays_path = artifacts_dir / "curated_replays.json"
    audit_path = artifacts_dir / "validation_audit.json"
    manifest_path = artifacts_dir / "artifact_manifest.json"

    events_df = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    foot_df = pd.read_parquet(foot_path) if foot_path.exists() else pd.DataFrame()
    replays = json.loads(replays_path.read_text(encoding="utf-8")) if replays_path.exists() else []
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    return {
        "events": events_df,
        "footedness": foot_df,
        "replays": replays,
        "audit": audit,
        "manifest": manifest,
    }


@st.cache_data(show_spinner=False)
def get_cached_artifacts() -> dict[str, Any]:
    """Streamlit-cached loader for compact runtime artifacts."""
    return load_artifacts(ARTIFACTS_DIR)


def _expand_replay_players(compact_players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand compact JSON player dicts back into full physics-ready player states."""
    out: list[dict[str, Any]] = []
    for p in compact_players:
        fwd = np.array([float(p["fx"]), float(p["fy"])], dtype=float)
        out.append(
            {
                "player_id": int(p["id"]),
                "short_name": str(p["n"]),
                "role": str(p["r"]),
                "is_attacking": bool(p["a"]),
                "is_gk": bool(p["g"]),
                "x": float(p["x"]),
                "y": float(p["y"]),
                "vx": float(p["vx"]),
                "vy": float(p["vy"]),
                "hip_fwd": fwd,
                "sh_fwd": fwd.copy(),
                "torso_twist": math.radians(float(p["tw"])),
                "stance": str(p["st"]),
                "stance_lead_m": float(p["sl"]),
                "pose_source": "3d_pose" if p["ps"] == "3d" else "kinematic_fallback",
                "dominant_foot": str(p["df"]),
                "weak_foot_comfort": float(p["wf"]),
            }
        )
    return out


def _render_tab1_match_engine() -> None:
    st.subheader("Interactive Biomechanical Match Engine")
    st.caption(
        "Deterministic 10 Hz biomechanical simulator (reaction latency τ=0.30s, turn rate ω_max=3.2 rad/s, "
        "stance vulnerability, and inter-defender matrix spring coupling)."
    )

    scenarios = get_preset_scenarios()
    c_top1, c_top2, c_top3 = st.columns([2.2, 1.4, 1.4])
    with c_top1:
        scen_name = st.selectbox("Tactical Preset", list(scenarios.keys()), key="t1_scen")
    scen = scenarios[scen_name]
    scen["name"] = scen_name

    actions = ["Body Feint", "Directional First Touch", "Driven / Disguised Pass"]
    def_idx = actions.index(scen["default_action"]) if scen["default_action"] in actions else 0
    with c_top2:
        action = st.selectbox("Action Type", actions, index=def_idx, key="t1_action")
    with c_top3:
        control_mode = st.radio(
            "Pitch Control Surface",
            ["Anisotropic (3D Biomech)", "Isotropic (Baseline)"],
            horizontal=True,
            key="t1_ctrl_mode",
        )

    st.info(scen["description"])

    # Editable carrier & primary defender controls
    carrier_init = next(p for p in scen["players"] if p["player_id"] == scen["carrier_id"])
    def_init = next(p for p in scen["players"] if p["player_id"] == scen["primary_def_id"])
    c_ang0 = int(round(math.degrees(math.atan2(carrier_init["hip_fwd"][1], carrier_init["hip_fwd"][0]))))
    d_ang0 = int(round(math.degrees(math.atan2(def_init["hip_fwd"][1], def_init["hip_fwd"][0]))))

    with st.expander("Edit Carrier & Primary Defender Biomechanics", expanded=True):
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown(f"**Ball Carrier — `{carrier_init['short_name']}`**")
            cc1, cc2 = st.columns(2)
            cx = cc1.slider("Carrier X (m)", -10.0, 46.0, float(carrier_init["x"]), 0.5, key=f"cx_{scen_name}")
            cy = cc2.slider("Carrier Y (m)", -28.0, 28.0, float(carrier_init["y"]), 0.5, key=f"cy_{scen_name}")
            cc3, cc4 = st.columns(2)
            c_foot = cc3.radio(
                "Strong Foot",
                ["R", "L"],
                index=0 if carrier_init["dominant_foot"] == "R" else 1,
                horizontal=True,
                key=f"cf_{scen_name}",
            )
            c_hip = cc4.slider("Carrier Hip Facing (°)", -180, 180, c_ang0, 5, key=f"ch_{scen_name}")
            touch_mode = "Open Strong-Foot"
            if action == "Directional First Touch":
                touch_mode = st.radio(
                    "First-Touch Execution",
                    ["Open Strong-Foot", "Closed Weak-Foot"],
                    horizontal=True,
                    key="t1_touch_mode",
                )

        with col_d:
            st.markdown(f"**Primary Defender — `{def_init['short_name']}`**")
            dc1, dc2 = st.columns(2)
            dx = dc1.slider("Defender X (m)", 0.0, 48.0, float(def_init["x"]), 0.5, key=f"dx_{scen_name}")
            dy = dc2.slider("Defender Y (m)", -28.0, 28.0, float(def_init["y"]), 0.5, key=f"dy_{scen_name}")
            dc3, dc4 = st.columns(2)
            stances = ["Left-Forward", "Right-Forward", "Square"]
            d_stance = dc3.selectbox(
                "Tackling Stance",
                stances,
                index=stances.index(def_init["stance"]),
                key=f"ds_{scen_name}",
            )
            d_hip = dc4.slider("Defender Hip Facing (°)", -180, 180, d_ang0, 5, key=f"dh_{scen_name}")

    rollout = run_simulation(
        scen,
        action=action,
        carrier_overrides={"x": cx, "y": cy, "dominant_foot": c_foot, "hip_deg": c_hip},
        defender_overrides={"x": dx, "y": dy, "hip_deg": d_hip, "stance": d_stance},
        touch_mode=touch_mode,
    )
    frames = rollout["frames"]
    step_idx = st.slider(
        "Simulation Step (0.1s resolution)",
        0,
        len(frames) - 1,
        len(frames) - 1,
        format="Step %d",
        key="t1_step",
    )
    fr = frames[step_idx]
    ev = fr["eval"]
    decomp = fr["decomp"]
    seam = fr["seam"]

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Anisotropic ΔC", f"{decomp['delta_c_total']*100:+.2f} pp", f"vs Iso {decomp['iso_delta_c']*100:+.2f} pp")
    k2.metric("Fiedler λ₂ Integrity", f"{ev['matrix']['lambda2']:.3f}", f"{seam['lambda2_delta']:+.3f}")
    k3.metric("Matrix Hole Score H_Δ", f"{ev['matrix']['hole_score']:.3f}", f"{seam['h_delta']:+.3f}")
    k4.metric("Matchup Edge", f"{rollout['matchup_end']['net_matchup_edge']:+.2f}", f"Def Weak {rollout['matchup_end']['defender_weak_side_exposure']:.2f}")
    k5.metric("A_pass / A_recv", f"{rollout['a_pass']:+.2f} / {rollout['a_recv']:+.2f}", decomp["dominant_driver"])

    use_aniso = control_mode.startswith("Anisotropic")
    grid = ev["aniso_grid"] if use_aniso else ev["iso_grid"]
    hl_tri = seam["highlighted_triangle"]["coords"] if seam.get("highlighted_triangle") else None

    col_pitch, col_wf = st.columns([1.65, 1.0])
    with col_pitch:
        fig_pitch = build_pitch_figure(
            players=fr["players"],
            ball_xy=fr["ball_xy"],
            control_grid=grid,
            control_label="Anisotropic Control" if use_aniso else "Isotropic Control",
            triangles=ev["matrix"]["triangles"],
            highlighted_triangle_coords=hl_tri,
            title=f"{scen_name} — {action} (t = {fr['t']:.1f}s)",
            show_control=True,
            show_delaunay=True,
        )
        st.plotly_chart(fig_pitch, width="stretch")

    with col_wf:
        fig_wf = build_delta_c_waterfall(decomp, title=f"3-Way ΔC Decomposition (t=0.0s → {fr['t']:.1f}s)")
        st.plotly_chart(fig_wf, width="stretch")
        st.markdown(
            f"**Additivity check:** `{decomp['delta_c_pos']*100:+.3f}` (Pos) + "
            f"`{decomp['delta_c_orient']*100:+.3f}` (Orient) + "
            f"`{decomp['delta_c_matrix']*100:+.3f}` (Matrix) = "
            f"**`{decomp['delta_c_total']*100:+.3f} pp`**"
        )


def _render_tab2_replays(replays: list[dict[str, Any]]) -> None:
    st.subheader("Real-Match 3D Pose & Defensive Matrix Replay")
    if not replays:
        st.warning("No replay artifacts found. Run `python -m src.build_artifacts` first.")
        return

    labels = [r["label"] for r in replays]
    c1, c2, c3 = st.columns([2.6, 1.2, 1.0])
    with c1:
        sel_lbl = st.selectbox("Curated 10 Hz Sequence", labels, key="t2_seq")
    seq = next(r for r in replays if r["label"] == sel_lbl)
    frames = seq["frames"]

    with c2:
        overlay_mode = st.radio(
            "Control Surface",
            ["Anisotropic (3D Pose)", "Isotropic (Baseline)"],
            horizontal=True,
            key="t2_overlay",
        )
    with c3:
        show_mesh = st.checkbox("Delaunay Defensive Mesh", value=True, key="t2_mesh")

    f_idx = st.slider(
        "10 Hz Tracking Frame Scrubber",
        0,
        len(frames) - 1,
        min(len(frames) - 1, len(frames) // 2),
        key="t2_frame",
    )
    fr = frames[f_idx]
    expanded_players = _expand_replay_players(fr["players"])
    ev = evaluate_frame_state(expanded_players)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tracking / 3D Frame", f"{fr['f']} / {fr['pf']}", f"{fr['n3d']}/{len(expanded_players)} 3D pose")
    m2.metric("Anisotropic ΔC", f"{fr['dc']*100:+.2f} pp", f"Orient {fr['dc_ori']*100:+.2f} / Mat {fr['dc_mat']*100:+.2f}")
    m3.metric("Fiedler λ₂ Integrity", f"{fr['lam2']:.3f}", f"Compactness {fr['comp']:.3f}")
    m4.metric("Matrix Hole Score H_Δ", f"{fr['hole']:.3f}")
    m5.metric("Phase Structure", seq["tip_phase"].upper(), f"vs {seq['otip_phase']}")

    use_aniso = overlay_mode.startswith("Anisotropic")
    fig_pitch = build_pitch_figure(
        players=expanded_players,
        ball_xy=[fr["bx"], fr["by"]],
        control_grid=ev["aniso_grid"] if use_aniso else ev["iso_grid"],
        control_label="Anisotropic Control" if use_aniso else "Isotropic Control",
        triangles=ev["matrix"]["triangles"],
        highlighted_triangle_coords=fr.get("seam_tri"),
        title=f"{seq['fixture']} · {seq['tag']} (t = +{f_idx * 0.1:.1f}s)",
        show_control=True,
        show_delaunay=show_mesh,
    )
    st.plotly_chart(fig_pitch, width="stretch")

    fig_tl = build_replay_timeline(frames, f_idx)
    st.plotly_chart(fig_tl, width="stretch")


def _render_tab3_strong_weak_lab(events_df: pd.DataFrame, foot_df: pd.DataFrame) -> None:
    st.subheader("Strong/Weak-Side & First-Touch Lab")
    if events_df.empty:
        st.warning("No scored event artifacts found.")
        return

    poss = events_df[events_df["event_type"] == "player_possession"].copy()

    f1, f2, f3 = st.columns(3)
    fixtures = ["All Matches"] + sorted(poss["fixture"].dropna().unique().tolist())
    sel_fix = f1.selectbox("Match Filter", fixtures, key="t3_fix")
    pose_filt = f2.radio("Pose Provenance", ["All Events", "3d_pose Only"], horizontal=True, key="t3_pose")
    action_filt = f3.selectbox("Action Subset", ["All Possessions", "Passes", "One-Touch", "Carries"], key="t3_act")

    sub = poss.copy()
    if sel_fix != "All Matches":
        sub = sub[sub["fixture"] == sel_fix]
    if pose_filt == "3d_pose Only":
        sub = sub[sub["pose_source"] == "3d_pose"]
    if action_filt == "Passes":
        sub = sub[sub["end_type"] == "pass"]
    elif action_filt == "One-Touch":
        sub = sub[sub["one_touch"]]
    elif action_filt == "Carries":
        sub = sub[sub["carry"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filtered Possessions", f"{len(sub):,}")
    k2.metric("Mean Aniso − Iso ΔC", f"{(sub['aniso_minus_iso_delta'].mean() * 100):+.3f} pp" if len(sub) else "N/A")
    k3.metric("Mean Pass Favorability A_pass", f"{sub['a_pass'].mean():+.3f}" if len(sub) else "N/A")
    k4.metric("Mean Matchup Edge", f"{sub['net_matchup_edge'].mean():+.3f}" if len(sub) else "N/A")

    st.plotly_chart(build_matchup_scatter(sub), width="stretch")

    st.markdown("#### Top Biomechanical Advantage Moments (Where Orientation & Matrix Outperformed Naive Isotropic)")
    display_cols = [
        "fixture", "minute", "player_name", "player_position", "end_type",
        "pose_source", "dominant_foot", "iso_delta_c", "aniso_delta_c",
        "delta_c_orient", "delta_c_matrix", "a_pass", "a_recv", "net_matchup_edge",
        "xshot_max", "dominant_driver",
    ]
    ranked = sub.sort_values("aniso_minus_iso_delta", ascending=False).head(25)
    st.dataframe(ranked[[c for c in display_cols if c in ranked.columns]], width="stretch")

    if not foot_df.empty:
        st.markdown("#### Empirical 3D Dominant-Foot & Weak-Foot Comfort Profile")
        st.dataframe(
            foot_df.sort_values(["n_contacts", "player_id"], ascending=[False, True]),
            width="stretch",
        )


def _render_tab4_validation(audit: dict[str, Any], manifest: dict[str, Any]) -> None:
    st.subheader("Quantitative Validation Audit (Strict 5-Fold Stratified OOF)")
    if not audit or not audit.get("tasks"):
        st.warning("No validation audit artifact found.")
        return

    st.caption(audit.get("cv_protocol", ""))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scored Events", f"{audit.get('total_scored_events', 0):,}")
    k2.metric("3D Pose Coverage", f"{audit.get('pose_3d_pct', 0.0):.1f}%", f"{audit.get('pose_3d_events', 0):,} events")
    k3.metric("Kinematic Fallback", f"{audit.get('kinematic_fallback_events', 0):,}")
    k4.metric("Artifact Bundle Size", f"{manifest.get('artifacts_total_mb', 0.0):.2f} MB", f"Build: {manifest.get('mode', '')}")

    slice_choice = st.radio(
        "Evaluation Slice",
        ["all_events", "3d_pose_only"],
        format_func=lambda s: "All Scored Events" if s == "all_events" else "Pure 3D Pose Only (pose_source == '3d_pose')",
        horizontal=True,
        key="t4_slice",
    )
    st.plotly_chart(build_audit_comparison_chart(audit, slice_key=slice_choice), width="stretch")

    rows: list[dict[str, Any]] = []
    for t in audit["tasks"]:
        for sk_lbl, sk in [("All Events", "all_events"), ("3D Pose Only", "3d_pose_only")]:
            sl = t["slices"][sk]
            b, f = sl["baseline"], sl["biomechanics"]
            rows.append(
                {
                    "Target": t["task_label"],
                    "Slice": sk_lbl,
                    "N": b["n_samples"],
                    "Positives": b["n_positives"],
                    "Baseline AUC": b["roc_auc"],
                    "+ Biomech AUC": f["roc_auc"],
                    "Δ AUC": sl["delta_auc"],
                    "Baseline Brier": b["brier"],
                    "+ Biomech Brier": f["brier"],
                    "Δ Brier": sl["delta_brier"],
                }
            )
    st.dataframe(pd.DataFrame(rows), width="stretch")


def render_app() -> None:
    """Render the 4-Tab Streamlit Workbench."""
    st.set_page_config(
        page_title="Biomechanical Football Advantage Engine",
        page_icon="⚽",
        layout="wide",
    )

    st.title("Biomechanical Football Tactical Advantage & Defensive-Matrix Engine")
    st.caption(
        "Anisotropic 3D Body Pose Pitch Control · Delaunay Defensive Matrix (λ₂ & H_Δ) · "
        "Exact 3-Way ΔC Decomposition · SkillCorner 1925299 & 1996435 Validation"
    )

    data = get_cached_artifacts()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Interactive Biomechanical Match Engine",
            "2. Real-Match 3D Pose & Matrix Replay",
            "3. Strong/Weak-Side & First-Touch Lab",
            "4. Quantitative Validation Audit",
        ]
    )
    with tab1:
        _render_tab1_match_engine()
    with tab2:
        _render_tab2_replays(data["replays"])
    with tab3:
        _render_tab3_strong_weak_lab(data["events"], data["footedness"])
    with tab4:
        _render_tab4_validation(data["audit"], data["manifest"])


if __name__ == "__main__":
    render_app()
