"""Defensive Structure Matrix M(t), Delaunay Triangulation, Fiedler λ₂, and Seam/Hole Score H_Δ."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError

DEFAULT_COVER_LAMBDA = 14.0
GRID_SHAPE = (22, 35)


def compute_mutual_cover_matrix(
    def_xy: np.ndarray,
    def_fwd: np.ndarray,
    cover_lambda: float = DEFAULT_COVER_LAMBDA,
) -> np.ndarray:
    """Compute defender mutual cover matrix M_ij = exp(-d_ij / lambda) * max(0, cos(Δθ_ij / 2))."""
    n = len(def_xy)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    # Pairwise Euclidean distances
    diff = def_xy[:, None, :] - def_xy[None, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    spatial = np.exp(-dist / max(cover_lambda, 1e-6))

    # Normalize facing vectors defensively
    norms = np.hypot(def_fwd[:, 0], def_fwd[:, 1])
    norms = np.where(norms < 1e-9, 1.0, norms)
    u = def_fwd / norms[:, None]

    # Exact half-angle identity: cos(Δθ / 2) = sqrt((1 + cos(Δθ)) / 2) >= 0
    cos_theta = np.clip(u @ u.T, -1.0, 1.0)
    coherence = np.sqrt(0.5 * (1.0 + cos_theta))

    mat = spatial * coherence
    np.fill_diagonal(mat, 0.0)
    return mat


def compute_fiedler_connectivity(cover_matrix: np.ndarray) -> float:
    """Compute normalized Fiedler algebraic connectivity λ₂(L) / N in [0, 1].

    Dividing the second-smallest Laplacian eigenvalue by N bounds connectivity to [0, 1]
    while remaining strictly sensitive to both single-defender isolation and global block stretching.
    """
    n = cover_matrix.shape[0]
    if n < 2:
        return 0.0

    deg = np.sum(cover_matrix, axis=1)
    laplacian = np.diag(deg) - cover_matrix
    eigvals = np.linalg.eigvalsh(laplacian)
    fiedler = float(max(0.0, eigvals[1]))
    return float(np.clip(fiedler / float(n), 0.0, 1.0))


def _safe_delaunay(points: np.ndarray) -> Delaunay | None:
    """Run Delaunay triangulation, adding deterministic micro-jitter if points are collinear."""
    if len(points) < 3:
        return None
    try:
        return Delaunay(points)
    except QhullError:
        jitter = np.array(
            [[1e-3 * ((-1) ** i), 1e-3 * ((i % 3) - 1)] for i in range(len(points))],
            dtype=float,
        )
        try:
            return Delaunay(points + jitter)
        except QhullError:
            return None


def _triangle_area(coords: np.ndarray) -> float:
    """Area of a 2D triangle given (3, 2) vertex coordinates."""
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    return float(0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0]))


def analyze_defensive_matrix(
    def_xy: np.ndarray,
    def_fwd: np.ndarray,
    att_control_grid: np.ndarray | None = None,
    threat_grid: np.ndarray | None = None,
    grid_xy: np.ndarray | None = None,
    cover_lambda: float = DEFAULT_COVER_LAMBDA,
) -> dict[str, Any]:
    """Compute M(t), Delaunay triangles, Fiedler λ₂, compactness, static hole score, and deficit grid."""
    n = len(def_xy)
    cover_mat = compute_mutual_cover_matrix(def_xy, def_fwd, cover_lambda=cover_lambda)
    lambda2 = compute_fiedler_connectivity(cover_mat)

    grid_shape = att_control_grid.shape if att_control_grid is not None else GRID_SHAPE
    deficit_grid = np.full(grid_shape, 0.25 * (1.0 - lambda2), dtype=float)

    tri = _safe_delaunay(def_xy)
    triangles: list[dict[str, Any]] = []
    edge_covers: list[float] = []

    if tri is not None:
        simplices = tri.simplices
        simplex_ids = None
        if grid_xy is not None:
            simplex_ids = tri.find_simplex(grid_xy).reshape(grid_shape)

        for k, simp in enumerate(simplices):
            i, j, l = int(simp[0]), int(simp[1]), int(simp[2])
            coords = def_xy[[i, j, l]]
            area = _triangle_area(coords)
            c_ij, c_jl, c_li = float(cover_mat[i, j]), float(cover_mat[j, l]), float(cover_mat[l, i])
            cover_mean = (c_ij + c_jl + c_li) / 3.0
            edge_covers.extend([c_ij, c_jl, c_li])
            centroid = np.mean(coords, axis=0)

            att_ctrl = 0.5
            threat_val = 0.002
            if simplex_ids is not None and att_control_grid is not None and threat_grid is not None:
                mask = simplex_ids == k
                if np.any(mask):
                    att_ctrl = float(np.mean(att_control_grid[mask]))
                    threat_val = float(np.mean(threat_grid[mask]))
                    # Structural cover deficit scales with broken mutual cover and expanded triangle area.
                    tri_deficit = float(
                        np.clip(1.0 - cover_mean, 0.0, 1.0) * np.clip(area / 90.0, 0.15, 1.50)
                    )
                    deficit_grid[mask] = tri_deficit
                else:
                    # Fallback to nearest grid cell to centroid for compact triangles
                    d2 = np.sum((grid_xy - centroid[None, :]) ** 2, axis=1)
                    nearest_idx = int(np.argmin(d2))
                    att_ctrl = float(att_control_grid.ravel()[nearest_idx])
                    threat_val = float(threat_grid.ravel()[nearest_idx])

            # Threat-weighted hole exposure for triangle k (scaled by 1000 for readable tactical units)
            hole_exp = float((area / 100.0) * att_ctrl * (threat_val * 1000.0) * (1.0 - cover_mean))
            triangles.append(
                {
                    "vertices": [i, j, l],
                    "coords": coords.tolist(),
                    "centroid": [float(centroid[0]), float(centroid[1])],
                    "area_m2": round(area, 2),
                    "cover_mean": round(cover_mean, 4),
                    "att_control": round(att_ctrl, 4),
                    "threat": round(threat_val, 6),
                    "hole_exposure": round(hole_exp, 5),
                }
            )

    mean_compactness = float(np.mean(edge_covers)) if edge_covers else (
        float(np.mean(cover_mat[np.triu_indices(n, k=1)])) if n >= 2 else 0.0
    )
    hole_score = float(sum(t["hole_exposure"] for t in triangles))
    max_seam = max(triangles, key=lambda t: t["hole_exposure"]) if triangles else None

    return {
        "n_defenders": n,
        "cover_matrix": cover_mat,
        "lambda2": round(lambda2, 5),
        "mean_compactness": round(mean_compactness, 5),
        "hole_score": round(hole_score, 5),
        "triangles": triangles,
        "max_seam_triangle": max_seam,
        "deficit_grid": deficit_grid,
    }


def compute_seam_expansion(
    matrix_t0: dict[str, Any],
    matrix_t1: dict[str, Any],
) -> dict[str, Any]:
    """Identify Delaunay triangles expanding in area (ΔA > 0) while attacking control increases."""
    tris0 = matrix_t0.get("triangles", [])
    tris1 = matrix_t1.get("triangles", [])

    by_verts0 = {tuple(sorted(t["vertices"])): t for t in tris0}
    expanding_score = 0.0
    best_seam: dict[str, Any] | None = None
    best_gain = -1e9

    for t1 in tris1:
        key = tuple(sorted(t1["vertices"]))
        t0 = by_verts0.get(key)
        if t0 is None and tris0:
            # Nearest-centroid fallback if triangulation topology flipped between t0 and t1
            c1 = np.array(t1["centroid"])
            t0 = min(tris0, key=lambda cand: float(np.hypot(c1[0] - cand["centroid"][0], c1[1] - cand["centroid"][1])))

        if t0 is not None:
            d_area = t1["area_m2"] - t0["area_m2"]
            d_ctrl = t1["att_control"] - t0["att_control"]
            d_hole = t1["hole_exposure"] - t0["hole_exposure"]
        else:
            d_area = t1["area_m2"]
            d_ctrl = t1["att_control"]
            d_hole = t1["hole_exposure"]

        if d_area > 0.0 and d_ctrl > 0.0:
            gain = float((d_area / 100.0) * d_ctrl * (t1["threat"] * 1000.0))
            expanding_score += gain
        else:
            gain = max(0.0, d_hole)

        if d_hole > best_gain:
            best_gain = d_hole
            best_seam = {
                **t1,
                "delta_area_m2": round(d_area, 2),
                "delta_att_control": round(d_ctrl, 4),
                "delta_hole_exposure": round(d_hole, 5),
            }

    return {
        "h_delta": round(float(matrix_t1["hole_score"] - matrix_t0["hole_score"]), 5),
        "expanding_seam_score": round(expanding_score, 5),
        "lambda2_delta": round(float(matrix_t1["lambda2"] - matrix_t0["lambda2"]), 5),
        "highlighted_triangle": best_seam or matrix_t1.get("max_seam_triangle"),
    }
