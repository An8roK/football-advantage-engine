"""Plotly tactical pitch maps, Delaunay overlays, ΔC waterfalls, and OOF audit charts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.physics import PITCH_LENGTH, PITCH_WIDTH, get_pitch_grid

BG_DARK = "#0b1120"
PITCH_BG = "#0f1e1a"
LINE_COLOR = "rgba(226, 232, 240, 0.35)"
ATT_COLOR = "#38bdf8"   # Sky cyan for attacking team
DEF_COLOR = "#f43f5e"   # Rose crimson for defending team
SEAM_COLOR = "#f59e0b"  # Amber gold for highlighted matrix seam


def _pitch_shapes() -> list[dict[str, Any]]:
    """Standard 105m x 68m centered football pitch markings."""
    hl = PITCH_LENGTH / 2.0  # 52.5
    hw = PITCH_WIDTH / 2.0   # 34.0
    line = {"color": LINE_COLOR, "width": 1.4}
    shapes = [
        # Outer boundary & halfway line
        {"type": "rect", "x0": -hl, "y0": -hw, "x1": hl, "y1": hw, "line": line},
        {"type": "line", "x0": 0, "y0": -hw, "x1": 0, "y1": hw, "line": line},
        # Center circle (r = 9.15m)
        {"type": "circle", "x0": -9.15, "y0": -9.15, "x1": 9.15, "y1": 9.15, "line": line},
        # Left penalty & 6-yard boxes
        {"type": "rect", "x0": -hl, "y0": -20.16, "x1": -hl + 16.5, "y1": 20.16, "line": line},
        {"type": "rect", "x0": -hl, "y0": -9.16, "x1": -hl + 5.5, "y1": 9.16, "line": line},
        # Right (attacking) penalty & 6-yard boxes
        {"type": "rect", "x0": hl - 16.5, "y0": -20.16, "x1": hl, "y1": 20.16, "line": line},
        {"type": "rect", "x0": hl - 5.5, "y0": -9.16, "x1": hl, "y1": 9.16, "line": line},
        # Goals
        {"type": "rect", "x0": -hl - 2.0, "y0": -3.66, "x1": -hl, "y1": 3.66, "line": {"color": "#94a3b8", "width": 2}},
        {"type": "rect", "x0": hl, "y0": -3.66, "x1": hl + 2.0, "y1": 3.66, "line": {"color": "#38bdf8", "width": 2}},
    ]
    return shapes


def _stance_short(stance: str) -> str:
    if stance == "Left-Forward":
        return "L-Fwd"
    if stance == "Right-Forward":
        return "R-Fwd"
    return "Sq"


def build_pitch_figure(
    players: list[dict[str, Any]],
    ball_xy: list[float] | np.ndarray,
    control_grid: np.ndarray | None = None,
    control_label: str = "Anisotropic Control",
    triangles: list[dict[str, Any]] | None = None,
    highlighted_triangle_coords: list[list[float]] | None = None,
    title: str = "",
    show_control: bool = True,
    show_delaunay: bool = True,
    height: int = 520,
) -> go.Figure:
    """Render an interactive 105x68 pitch with control heatmap, Delaunay mesh, hip arrows, and stance badges."""
    fig = go.Figure()

    # 1. Pitch Control contour surface
    if show_control and control_grid is not None:
        xs, ys, _, _, _ = get_pitch_grid()
        fig.add_trace(
            go.Heatmap(
                x=xs,
                y=ys,
                z=control_grid,
                zmin=0.0,
                zmax=1.0,
                zsmooth="best",
                opacity=0.62,
                colorscale=[
                    [0.0, "rgba(244, 63, 94, 0.78)"],
                    [0.5, "rgba(15, 23, 42, 0.15)"],
                    [1.0, "rgba(56, 189, 248, 0.78)"],
                ],
                colorbar={
                    "title": {"text": control_label, "font": {"size": 11, "color": "#cbd5e1"}},
                    "thickness": 12,
                    "len": 0.75,
                    "tickfont": {"color": "#cbd5e1", "size": 10},
                },
                hovertemplate="x=%{x:.1f}m, y=%{y:.1f}m<br>Att Control=%{z:.2f}<extra></extra>",
                name=control_label,
            )
        )

    # 2. Delaunay defensive structure triangulation mesh
    if show_delaunay and triangles:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for tri in triangles:
            c = tri["coords"]
            edge_x.extend([c[0][0], c[1][0], c[2][0], c[0][0], None])
            edge_y.extend([c[0][1], c[1][1], c[2][1], c[0][1], None])
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "rgba(251, 191, 36, 0.45)", "width": 1.3, "dash": "dot"},
                hoverinfo="skip",
                name="Defensive Matrix Mesh",
            )
        )

    # 3. Highlighted seam / hole triangle
    if show_delaunay and highlighted_triangle_coords and len(highlighted_triangle_coords) == 3:
        hc = highlighted_triangle_coords
        fig.add_trace(
            go.Scatter(
                x=[hc[0][0], hc[1][0], hc[2][0], hc[0][0]],
                y=[hc[0][1], hc[1][1], hc[2][1], hc[0][1]],
                mode="lines",
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.24)",
                line={"color": SEAM_COLOR, "width": 2.4},
                name="Exposed Matrix Seam",
                hovertemplate="Highlighted Matrix Seam / Hole<extra></extra>",
            )
        )

    # 4. Hip-facing directional vectors (arrow shafts)
    arrow_x_att: list[float | None] = []
    arrow_y_att: list[float | None] = []
    arrow_x_def: list[float | None] = []
    arrow_y_def: list[float | None] = []

    for p in players:
        px, py = float(p["x"]), float(p["y"])
        fwd = p["hip_fwd"]
        fx, fy = float(fwd[0]), float(fwd[1])
        length = 3.0
        if p.get("is_attacking", False):
            arrow_x_att.extend([px, px + length * fx, None])
            arrow_y_att.extend([py, py + length * fy, None])
        else:
            arrow_x_def.extend([px, px + length * fx, None])
            arrow_y_def.extend([py, py + length * fy, None])

    if arrow_x_att:
        fig.add_trace(
            go.Scatter(
                x=arrow_x_att,
                y=arrow_y_att,
                mode="lines",
                line={"color": "#bae6fd", "width": 2.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if arrow_x_def:
        fig.add_trace(
            go.Scatter(
                x=arrow_x_def,
                y=arrow_y_def,
                mode="lines",
                line={"color": "#fecdd3", "width": 2.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 5. Attacking & Defending player markers with stance + foot badges
    for is_att, name_lbl, color in [
        (True, "Attacking Unit (→)", ATT_COLOR),
        (False, "Defending Unit", DEF_COLOR),
    ]:
        grp = [p for p in players if bool(p.get("is_attacking", False)) == is_att]
        if not grp:
            continue
        xs_p = [float(p["x"]) for p in grp]
        ys_p = [float(p["y"]) for p in grp]
        symbols = ["diamond" if p.get("is_gk", False) else "circle" for p in grp]
        labels = [
            f"{p.get('role', '')} [{_stance_short(str(p.get('stance', 'Square')))}]"
            for p in grp
        ]
        hovers = [
            (
                f"<b>{p.get('short_name', p.get('player_id', ''))}</b> ({p.get('role', '')})<br>"
                f"Pos: ({p['x']:.1f}, {p['y']:.1f}) | Speed: {math.hypot(p.get('vx', 0), p.get('vy', 0)):.1f} m/s<br>"
                f"Stance: {p.get('stance', 'Square')} ({p.get('stance_lead_m', 0):+.2f} m)<br>"
                f"Torso Twist: {math.degrees(p.get('torso_twist', 0)):+.1f}°<br>"
                f"Foot: {p.get('dominant_foot', 'R')} (weak comfort {p.get('weak_foot_comfort', 0.5):.2f})<br>"
                f"Pose Source: {p.get('pose_source', '3d_pose')}"
            )
            for p in grp
        ]
        fig.add_trace(
            go.Scatter(
                x=xs_p,
                y=ys_p,
                mode="markers+text",
                marker={
                    "size": 14,
                    "color": color,
                    "symbol": symbols,
                    "line": {"color": "#f8fafc", "width": 1.6},
                },
                text=labels,
                textposition="top center",
                textfont={"size": 9, "color": "#f1f5f9"},
                hovertext=hovers,
                hoverinfo="text",
                name=name_lbl,
            )
        )

    # 6. Ball marker
    fig.add_trace(
        go.Scatter(
            x=[float(ball_xy[0])],
            y=[float(ball_xy[1])],
            mode="markers",
            marker={"size": 10, "color": "#fef08a", "line": {"color": "#0f172a", "width": 1.8}},
            name="Ball",
            hovertemplate="Ball (%{x:.1f}m, %{y:.1f}m)<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#f8fafc"}, "x": 0.01},
        shapes=_pitch_shapes(),
        xaxis={
            "range": [-56, 56],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "fixedrange": True,
        },
        yaxis={
            "range": [-36, 36],
            "scaleanchor": "x",
            "scaleratio": 1,
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "fixedrange": True,
        },
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=BG_DARK,
        font={"color": "#e2e8f0"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11},
        },
        margin={"l": 12, "r": 12, "t": 44, "b": 12},
        height=height,
    )
    return fig


def build_delta_c_waterfall(decomp: dict[str, Any], title: str = "Exact 3-Way ΔC Decomposition") -> go.Figure:
    """Render Plotly waterfall of Positional Shift + Orientation/Wrong-Footing + Matrix Distortion = Total ΔC."""
    # Display in threat-control percentage points (x100) for intuitive tactical readability
    vals = [
        decomp["delta_c_pos"] * 100.0,
        decomp["delta_c_orient"] * 100.0,
        decomp["delta_c_matrix"] * 100.0,
        decomp["delta_c_total"] * 100.0,
    ]
    iso_pp = decomp["iso_delta_c"] * 100.0

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Positional Shift", "Orientation &<br>Wrong-Footing", "Matrix Distortion<br>& Seam Opening", "Total Aniso ΔC"],
            y=vals,
            text=[f"{v:+.2f} pp" for v in vals],
            textposition="outside",
            connector={"line": {"color": "rgba(148, 163, 184, 0.5)"}},
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#f43f5e"}},
            totals={"marker": {"color": "#38bdf8"}},
        )
    )
    fig.add_hline(
        y=iso_pp,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Naive Isotropic ΔC: {iso_pp:+.2f} pp",
        annotation_font_color="#fbbf24",
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": "#f8fafc"}},
        yaxis_title="Threat-Weighted Control Shift (pp)",
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font={"color": "#e2e8f0", "size": 11},
        margin={"l": 45, "r": 20, "t": 44, "b": 40},
        height=340,
    )
    return fig


def build_replay_timeline(frames: list[dict[str, Any]], current_idx: int) -> go.Figure:
    """Dual-panel frame-by-frame timeline for ΔC (Aniso vs Iso) and Defensive Matrix Integrity λ₂."""
    t_vals = [round(i * 0.1, 1) for i in range(len(frames))]
    dc_aniso = [f["dc"] * 100.0 for f in frames]
    c_iso0 = frames[0]["c_iso"] if frames else 0.0
    dc_iso = [(f["c_iso"] - c_iso0) * 100.0 for f in frames]
    lam2 = [f["lam2"] for f in frames]
    hole = [f["hole"] for f in frames]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Action Control Shift ΔC (pp): Anisotropic vs. Isotropic", "Defensive Matrix Integrity λ₂ & Hole Score H_Δ"),
    )
    fig.add_trace(
        go.Scatter(x=t_vals, y=dc_aniso, name="Anisotropic ΔC (pp)", line={"color": "#38bdf8", "width": 2.5}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_vals, y=dc_iso, name="Isotropic ΔC (pp)", line={"color": "#94a3b8", "width": 1.8, "dash": "dash"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_vals, y=lam2, name="Fiedler λ₂ Integrity", line={"color": "#22c55e", "width": 2.2}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_vals, y=hole, name="Matrix Hole Score H_Δ", line={"color": "#f59e0b", "width": 2.0, "dash": "dot"}),
        row=2,
        col=1,
    )

    cur_t = t_vals[min(current_idx, len(t_vals) - 1)] if t_vals else 0.0
    fig.add_vline(x=cur_t, line_color="#f43f5e", line_width=2, line_dash="solid")

    fig.update_layout(
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font={"color": "#e2e8f0", "size": 11},
        legend={"orientation": "h", "y": 1.12, "x": 0.0},
        margin={"l": 45, "r": 20, "t": 50, "b": 35},
        height=360,
    )
    fig.update_xaxes(title_text="Elapsed Sequence Time (s)", row=2, col=1, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def build_matchup_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter comparing Naive Isotropic ΔC vs. Anisotropic ΔC colored by dominant driver."""
    color_map = {
        "Orientation & Wrong-Footing": "#f59e0b",
        "Matrix Distortion": "#a855f7",
        "Positional Shift": "#38bdf8",
    }
    fig = go.Figure()
    for driver, col in color_map.items():
        sub = df[df["dominant_driver"] == driver]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["iso_delta_c"] * 100.0,
                y=sub["aniso_delta_c"] * 100.0,
                mode="markers",
                name=driver,
                marker={"color": col, "size": 7, "opacity": 0.78},
                text=[
                    f"{r.player_name} ({r.fixture}, {r.minute}')<br>"
                    f"A_pass={r.a_pass:+.2f} | A_recv={r.a_recv:+.2f} | Matchup={r.net_matchup_edge:+.2f}<br>"
                    f"Iso ΔC={r.iso_delta_c*100:+.2f} pp vs Aniso ΔC={r.aniso_delta_c*100:+.2f} pp"
                    for r in sub.itertuples(index=False)
                ],
                hoverinfo="text",
            )
        )

    # Diagonal y = x line showing where Isotropic == Anisotropic
    lim = max(
        2.0,
        float(np.nanpercentile(np.abs(df["aniso_delta_c"] * 100.0), 99)) if not df.empty else 5.0,
    )
    fig.add_trace(
        go.Scatter(
            x=[-lim, lim],
            y=[-lim, lim],
            mode="lines",
            line={"color": "rgba(148,163,184,0.5)", "dash": "dash"},
            name="Isotropic = Anisotropic",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title={"text": "Isotropic vs. Anisotropic ΔC (Points Above Diagonal = Missed Orientation/Matrix Gain)", "font": {"size": 13}},
        xaxis_title="Naive Isotropic ΔC (pp)",
        yaxis_title="Anisotropic Biomechanical ΔC (pp)",
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font={"color": "#e2e8f0", "size": 11},
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 50, "r": 20, "t": 55, "b": 45},
        height=420,
    )
    return fig


def build_audit_comparison_chart(audit: dict[str, Any], slice_key: str = "all_events") -> go.Figure:
    """Grouped bar chart comparing Baseline vs. +Biomechanics ROC-AUC and Brier Score."""
    labels: list[str] = []
    base_auc: list[float] = []
    full_auc: list[float] = []
    base_brier: list[float] = []
    full_brier: list[float] = []

    for task in audit.get("tasks", []):
        sl = task["slices"].get(slice_key, {})
        b, f = sl.get("baseline", {}), sl.get("biomechanics", {})
        if b.get("roc_auc") is None or f.get("roc_auc") is None:
            continue
        labels.append(task["task_label"].split("(")[0].strip())
        base_auc.append(b["roc_auc"])
        full_auc.append(f["roc_auc"])
        base_brier.append(b["brier"])
        full_brier.append(f["brier"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("ROC-AUC (Higher is Better)", "Brier Score (Lower is Better)"),
    )
    fig.add_trace(
        go.Bar(name="Baseline (Isotropic + Dist)", x=labels, y=base_auc, marker_color="#64748b", text=[f"{v:.3f}" for v in base_auc], textposition="outside"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="+ Biomechanics & Matrix", x=labels, y=full_auc, marker_color="#38bdf8", text=[f"{v:.3f}" for v in full_auc], textposition="outside"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="Baseline Brier", x=labels, y=base_brier, marker_color="#64748b", showlegend=False, text=[f"{v:.3f}" for v in base_brier], textposition="outside"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(name="+ Biomechanics Brier", x=labels, y=full_brier, marker_color="#22c55e", showlegend=False, text=[f"{v:.3f}" for v in full_brier], textposition="outside"),
        row=1,
        col=2,
    )
    fig.update_layout(
        barmode="group",
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font={"color": "#e2e8f0", "size": 11},
        legend={"orientation": "h", "y": 1.14},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        height=360,
    )
    fig.update_yaxes(range=[0.4, 1.05], row=1, col=1, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", row=1, col=2)
    return fig
