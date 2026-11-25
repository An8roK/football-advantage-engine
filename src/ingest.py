"""SkillCorner Open Data & Hugging Face 3D Body Pose ingestion and constant-memory streaming."""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master"
MEDIA_BASE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master"
HF_BASE = "https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/main"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

FULL_MATCH_IDS: tuple[str, ...] = ("1925299", "1996435")
SAMPLE_MATCH_ID = "1925299"
SAMPLE_POSE_REL = "data/bodypose/sample_1925299_phase406.jsonl.gz"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TIMEOUT_SEC = 60
CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunks


def download_if_missing(
    url: str,
    dest: Path,
    fallback_url: str | None = None,
    session: requests.Session | None = None,
) -> Path:
    """Download a remote file to dest atomically if not already cached locally."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    own_session = session is None
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", "football-advantage-engine/1.0")

    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        resp = sess.get(url, stream=True, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        it = resp.iter_content(chunk_size=CHUNK_SIZE)
        first_chunk = next(it, b"")

        # Automatically follow Git LFS pointer to media.githubusercontent.com when provided.
        if fallback_url and first_chunk.startswith(LFS_POINTER_PREFIX):
            resp.close()
            resp = sess.get(fallback_url, stream=True, timeout=TIMEOUT_SEC * 2)
            resp.raise_for_status()
            it = resp.iter_content(chunk_size=CHUNK_SIZE)
            first_chunk = next(it, b"")

        with open(tmp_path, "wb") as f:
            if first_chunk:
                f.write(first_chunk)
            for chunk in it:
                if chunk:
                    f.write(chunk)
        resp.close()
        tmp_path.replace(dest)
        return dest
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if own_session:
            sess.close()


def ensure_match_files(
    match_id: str,
    sample_only: bool = False,
    raw_dir: Path = RAW_DIR,
) -> dict[str, Path]:
    """Ensure all required SkillCorner files for a match are cached under data/raw/."""
    match_dir = raw_dir / "matches" / str(match_id)
    pose_dir = raw_dir / "bodypose"

    with requests.Session() as sess:
        sess.headers.update({"User-Agent": "football-advantage-engine/1.0"})

        match_json = download_if_missing(
            f"{RAW_BASE}/data/matches/{match_id}/{match_id}_match.json",
            match_dir / f"{match_id}_match.json",
            session=sess,
        )
        dynamic_events = download_if_missing(
            f"{RAW_BASE}/data/matches/{match_id}/{match_id}_dynamic_events.csv",
            match_dir / f"{match_id}_dynamic_events.csv",
            session=sess,
        )
        phases_csv = download_if_missing(
            f"{RAW_BASE}/data/matches/{match_id}/{match_id}_phases_of_play.csv",
            match_dir / f"{match_id}_phases_of_play.csv",
            session=sess,
        )
        tracking_jsonl = download_if_missing(
            f"{MEDIA_BASE}/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl",
            match_dir / f"{match_id}_tracking_extrapolated.jsonl",
            session=sess,
        )
        if sample_only:
            pose_archive = download_if_missing(
                f"{RAW_BASE}/{SAMPLE_POSE_REL}",
                pose_dir / "sample_1925299_phase406.jsonl.gz",
                fallback_url=f"{MEDIA_BASE}/{SAMPLE_POSE_REL}",
                session=sess,
            )
        else:
            pose_archive = download_if_missing(
                f"{HF_BASE}/raw/{match_id}.jsonl.zip",
                pose_dir / f"{match_id}.jsonl.zip",
                session=sess,
            )

    return {
        "match_json": match_json,
        "dynamic_events": dynamic_events,
        "phases": phases_csv,
        "tracking": tracking_jsonl,
        "pose_archive": pose_archive,
    }


def load_match_metadata(match_json_path: Path) -> dict[str, Any]:
    """Load match.json and attach convenient player lookup tables."""
    with open(match_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    players_by_id: dict[int, dict[str, Any]] = {}
    players_by_trackable: dict[int, dict[str, Any]] = {}
    for p in meta.get("players", []):
        pid = int(p["id"])
        players_by_id[pid] = p
        if p.get("trackable_object") is not None:
            players_by_trackable[int(p["trackable_object"])] = p

    meta["_players_by_id"] = players_by_id
    meta["_players_by_trackable"] = players_by_trackable
    return meta


def load_dynamic_events(csv_path: Path) -> pd.DataFrame:
    """Load dynamic_events.csv into a DataFrame."""
    return pd.read_csv(csv_path, low_memory=False)


def load_phases_of_play(csv_path: Path) -> pd.DataFrame:
    """Load phases_of_play.csv into a DataFrame."""
    return pd.read_csv(csv_path, low_memory=False)


def get_attacking_sign(match_meta: dict[str, Any], team_id: int, period: int) -> float:
    """Return +1.0 if team attacks left-to-right (+x) in period, else -1.0."""
    home_id = int(match_meta["home_team"]["id"])
    sides = match_meta.get("home_team_side", ["left_to_right", "right_to_left"])
    idx = max(0, min(int(period) - 1, len(sides) - 1))
    home_ltr = sides[idx] == "left_to_right"
    if int(team_id) == home_id:
        return 1.0 if home_ltr else -1.0
    return -1.0 if home_ltr else 1.0


def extract_frame_id_fast(raw_line: bytes) -> int | None:
    """Fast O(1) byte-prefix extraction of `"frame": <int>` before json.loads.

    SkillCorner pose and tracking JSONL records place `"frame"` near the start of each line.
    Inspecting raw bytes avoids UTF-8 decoding and JSON parsing on >95% of skipped frames.
    """
    idx = raw_line.find(b'"frame":', 0, 96)
    if idx < 0:
        idx = raw_line.find(b'"frame":')
        if idx < 0:
            return None
    pos = idx + 8
    n = len(raw_line)
    while pos < n and raw_line[pos] in (32, 9):  # space or tab
        pos += 1
    start = pos
    while pos < n and 48 <= raw_line[pos] <= 57:  # ASCII '0'..'9'
        pos += 1
    if pos == start:
        return None
    return int(raw_line[start:pos])


def _iter_raw_archive_lines(archive_path: Path) -> Iterator[bytes]:
    """Yield raw lines from a .jsonl.zip, .jsonl.gz, or plain .jsonl in constant memory."""
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            jsonl_members = [
                name for name in zf.namelist()
                if name.endswith(".jsonl") or name.endswith(".json")
            ]
            if not jsonl_members:
                raise FileNotFoundError(f"No JSONL member found inside {archive_path}")
            with zf.open(jsonl_members[0], "r") as member:
                for raw_line in member:
                    yield raw_line
    elif suffix == ".gz":
        with gzip.open(archive_path, "rb") as gz:
            for raw_line in gz:
                yield raw_line
    else:
        with open(archive_path, "rb") as f:
            for raw_line in f:
                yield raw_line


def stream_pose_frames(
    archive_path: Path,
    target_pose_frames: set[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Stream 25 FPS 3D body-pose frames from zip/gz in constant memory.

    When target_pose_frames is provided, skips non-matching lines in O(1) before json.loads
    and exits early once all target frames have been collected.
    """
    frames: dict[int, dict[str, Any]] = {}
    remaining = len(target_pose_frames) if target_pose_frames is not None else -1

    for raw_line in _iter_raw_archive_lines(archive_path):
        if not raw_line.strip():
            continue
        if target_pose_frames is not None:
            fid = extract_frame_id_fast(raw_line)
            if fid is None or fid not in target_pose_frames:
                continue
            rec = json.loads(raw_line)
            frames[fid] = rec
            remaining -= 1
            if remaining <= 0:
                break
        else:
            rec = json.loads(raw_line)
            fid = rec.get("frame")
            if fid is not None:
                frames[int(fid)] = rec

    return frames


def stream_tracking_frames(
    tracking_path: Path,
    target_tracking_frames: set[int],
    include_velocity_neighbors: bool = True,
) -> dict[int, dict[str, Any]]:
    """Stream 10 FPS tracking frames, optionally retaining t-1/t+1 neighbors for velocity."""
    needed = set(target_tracking_frames)
    if include_velocity_neighbors:
        for f in target_tracking_frames:
            needed.add(f - 1)
            needed.add(f + 1)

    frames: dict[int, dict[str, Any]] = {}
    max_needed = max(needed) if needed else -1

    with open(tracking_path, "rb") as fp:
        for raw_line in fp:
            if not raw_line.strip():
                continue
            fid = extract_frame_id_fast(raw_line)
            if fid is None:
                continue
            if fid in needed:
                frames[fid] = json.loads(raw_line)
            if max_needed >= 0 and fid > max_needed:
                break

    return frames


def compute_frame_velocities(
    tracking_frames: dict[int, dict[str, Any]],
    frame_id: int,
    dt: float = 0.1,
    max_speed: float = 11.0,
) -> dict[int, tuple[float, float]]:
    """Estimate 10 FPS player velocities via central finite difference across [t-1, t+1]."""
    curr = tracking_frames.get(frame_id)
    if not curr:
        return {}

    prev_rec = tracking_frames.get(frame_id - 1)
    next_rec = tracking_frames.get(frame_id + 1)

    def _index_positions(rec: dict[str, Any] | None) -> dict[int, tuple[float, float]]:
        if not rec:
            return {}
        out: dict[int, tuple[float, float]] = {}
        for p in rec.get("player_data", []):
            pid = p.get("player_id")
            x, y = p.get("x"), p.get("y")
            if pid is not None and x is not None and y is not None:
                out[int(pid)] = (float(x), float(y))
        return out

    pos_c = _index_positions(curr)
    pos_p = _index_positions(prev_rec)
    pos_n = _index_positions(next_rec)

    velocities: dict[int, tuple[float, float]] = {}
    for pid, (xc, yc) in pos_c.items():
        if pid in pos_p and pid in pos_n:
            vx = (pos_n[pid][0] - pos_p[pid][0]) / (2.0 * dt)
            vy = (pos_n[pid][1] - pos_p[pid][1]) / (2.0 * dt)
        elif pid in pos_n:
            vx = (pos_n[pid][0] - xc) / dt
            vy = (pos_n[pid][1] - yc) / dt
        elif pid in pos_p:
            vx = (xc - pos_p[pid][0]) / dt
            vy = (yc - pos_p[pid][1]) / dt
        else:
            vx, vy = 0.0, 0.0

        spd = float(np.hypot(vx, vy))
        if spd > max_speed and spd > 1e-6:
            scale = max_speed / spd
            vx *= scale
            vy *= scale
        velocities[pid] = (vx, vy)

    return velocities
