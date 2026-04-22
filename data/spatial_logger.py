"""
spatial_logger.py — Thread-safe SQLite backend for spatial position tracking.

Logs one row per tracked person per N frames. Supports filterable heatmap
and common-path queries.
"""

import sqlite3
from system import config
import threading
import time
from system.logger_setup import setup_logger
from typing import Optional

_logger = setup_logger(__name__)

_DB_PATH = "spatial_log.db"
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def init_db(db_path: str = _DB_PATH) -> None:
    """Create the table and indexes if they do not exist. Call once at startup."""
    global _DB_PATH, _conn
    _DB_PATH = db_path
    with _lock:
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL;")          # concurrent reads
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         INTEGER NOT NULL,
                camera_id  TEXT NOT NULL,
                track_id   INTEGER NOT NULL,
                cx         REAL NOT NULL,
                cy         REAL NOT NULL,
                gender     TEXT NOT NULL DEFAULT 'unknown',
                age_group  TEXT NOT NULL DEFAULT 'unknown'
            )
        """)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_ts    ON positions(ts);")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_cam   ON positions(camera_id, ts);")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_track ON positions(track_id, ts);")
        _conn.commit()


def _ensure_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        init_db()
    return _conn  # type: ignore[return-value]


def log_position(
    camera_id: str,
    track_id: int,
    cx: float,
    cy: float,
    gender: str = "unknown",
    age_group: str = "unknown",
) -> None:
    """Insert one spatial position row. Non-blocking (WAL mode)."""
    ts = int(time.time() * 1000)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    try:
        with _lock:
            conn = _ensure_conn()
            conn.execute(
                "INSERT INTO positions(ts,camera_id,track_id,cx,cy,gender,age_group) VALUES(?,?,?,?,?,?,?)",
                (ts, camera_id, track_id, cx, cy, gender, age_group),
            )
            conn.commit()
    except Exception as e:
        _logger.error(f"SpatialLogger DB error: {e}", exc_info=True)
        # Don't re-raise — logging failure should not crash tracking


def query_heatmap(
    camera_id: str,
    gender: Optional[str] = None,
    age_group: Optional[str] = None,
    from_ts: Optional[int] = None,
    to_ts: Optional[int] = None,
) -> list:
    """
    Return a list of (cx, cy) floats matching the filters.
    All filter args are optional.
    """
    clauses = ["camera_id = ?"]
    params: list = [camera_id]
    if gender and gender != "all":
        clauses.append("gender = ?")
        params.append(gender)
    if age_group and age_group != "all":
        clauses.append("age_group = ?")
        params.append(age_group)
    if from_ts is not None:
        clauses.append("ts >= ?")
        params.append(from_ts)
    if to_ts is not None:
        clauses.append("ts <= ?")
        params.append(to_ts)

    sql = f"SELECT cx, cy FROM positions WHERE {' AND '.join(clauses)}"
    try:
        with _lock:
            conn = _ensure_conn()
            rows = conn.execute(sql, params).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception as e:
        _logger.error(f"SpatialLogger DB error: {e}", exc_info=True)
        # Don't re-raise — logging failure should not crash tracking
        return []


def query_paths(
    camera_id: str,
    gender: Optional[str] = None,
    age_group: Optional[str] = None,
    from_ts: Optional[int] = None,
    to_ts: Optional[int] = None,
    max_paths: int = 5,
    min_points: int = 6,
) -> list:
    """
    Return up to *max_paths* common track paths as lists of (cx, cy) tuples.

    Strategy:
    1. Fetch all positions ordered by (track_id, ts).
    2. Build a polyline per track (only those with >= min_points samples).
    3. Downsample each to 10 waypoints.
    4. Cluster by bucketing start+end cell and return the top max_paths.
    """
    clauses = ["camera_id = ?"]
    params: list = [camera_id]
    if gender and gender != "all":
        clauses.append("gender = ?")
        params.append(gender)
    if age_group and age_group != "all":
        clauses.append("age_group = ?")
        params.append(age_group)
    if from_ts is not None:
        clauses.append("ts >= ?")
        params.append(from_ts)
    if to_ts is not None:
        clauses.append("ts <= ?")
        params.append(to_ts)

    sql = f"SELECT track_id, cx, cy FROM positions WHERE {' AND '.join(clauses)} ORDER BY track_id, ts"
    try:
        with _lock:
            conn = _ensure_conn()
            rows = conn.execute(sql, params).fetchall()
    except Exception as e:
        _logger.error(f"SpatialLogger DB error: {e}", exc_info=True)
        # Don't re-raise — logging failure should not crash tracking
        return []

    # Group into per-track lists
    tracks: dict = {}
    for tid, cx, cy in rows:
        tracks.setdefault(tid, []).append((cx, cy))

    # Downsample to 10 waypoints
    def downsample(pts, n=10):
        if len(pts) <= n:
            return pts
        step = len(pts) / n
        return [pts[int(i * step)] for i in range(n)]

    sampled = [downsample(pts) for pts in tracks.values() if len(pts) >= min_points]

    if not sampled:
        return []

    # Cluster: bucket start-cell + end-cell (grid 10x10)
    def cell(pt):
        return (int(pt[0] * 10), int(pt[1] * 10))

    clusters: dict = {}
    for path in sampled:
        key = (cell(path[0]), cell(path[-1]))
        clusters.setdefault(key, []).append(path)

    # Sort clusters by size, return representative path of top clusters
    top = sorted(clusters.values(), key=len, reverse=True)[:max_paths]
    return [group[len(group) // 2] for group in top]  # median-index member
