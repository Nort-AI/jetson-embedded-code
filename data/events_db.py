"""
data/events_db.py — Flagged event store for the Nort operator escalation workflow.

Operators flag anomalous events (suspicious loitering, queue overflow, anything
worth a second look) from the UI. Events are stored in a lightweight SQLite
database alongside spatial_logger.db, with an optional JPEG crop snapshot.

Design
------
  • Separate DB (events.db) — never blocks the spatial_logger writer
  • Snapshots stored as base64 JPEG in a TEXT column, compressed to ≤ 100×150 px
  • Fully synchronous — called from Flask request thread, no background threads
  • All public functions are safe to call before the DB file exists
  • Thread-safe: a single module-level lock serialises all writes
"""

import base64
import os
import sqlite3
import threading
import time
import uuid
from typing import Optional

try:
    from system import config as _cfg
    _BASE_DIR = _cfg.BASE_DIR
except Exception:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from system.logger_setup import setup_logger
    logger = setup_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(_BASE_DIR, "events.db")
_write_lock = threading.Lock()

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flagged_events (
    id           TEXT    PRIMARY KEY,       -- uuid4 hex (32 chars)
    ts           INTEGER NOT NULL,          -- milliseconds since epoch
    camera_id    TEXT    NOT NULL DEFAULT '',
    global_id    TEXT    NOT NULL DEFAULT '',
    event_type   TEXT    NOT NULL DEFAULT 'manual',
    severity     TEXT    NOT NULL DEFAULT 'info',
    note         TEXT    NOT NULL DEFAULT '',
    snapshot_b64 TEXT,                      -- base64 JPEG crop (nullable)
    reviewed     INTEGER NOT NULL DEFAULT 0,
    created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ev_ts  ON flagged_events(ts);
CREATE INDEX IF NOT EXISTS idx_ev_cam ON flagged_events(camera_id, ts);
"""


def _get_conn() -> sqlite3.Connection:
    """Open (and initialise) the events DB. Caller must hold _write_lock for writes."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ── Snapshot helper ───────────────────────────────────────────────────────────

def _encode_snapshot(crop_bgr) -> Optional[str]:
    """
    Compress a BGR numpy array to a ≤100×150 JPEG and return as base64.
    Returns None on any error (missing crop, import failure, etc.).
    """
    if crop_bgr is None:
        return None
    try:
        import cv2
        import numpy as np
        h, w = crop_bgr.shape[:2]
        max_w, max_h = 100, 150
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:
        logger.debug(f"[EventsDB] snapshot encode error: {e}")
    return None


def _fetch_crop_for_gid(global_id: str):
    """Try to get the best crop for a global_id from the live track store."""
    if not global_id:
        return None
    try:
        from core import vlm_analyst as _vlm
        with _vlm._track_crops_lock:
            entry = _vlm._track_crops.get(str(global_id))
            if not entry:
                return None
            # Prefer the sharpest crop from the multi-pool, else the primary crop
            pool = entry.get("all_crops", [])
            if pool:
                return pool[0]["crop"].copy()
            crop = entry.get("crop")
            return crop.copy() if crop is not None else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def flag_event(
    camera_id:  str = "",
    global_id:  str = "",
    event_type: str = "manual",
    note:       str = "",
    severity:   str = "info",
    snapshot_b64: Optional[str] = None,
) -> str:
    """
    Save a flagged event to the DB. Returns the new event id.

    If snapshot_b64 is None and global_id is provided, the function will
    attempt to fetch the best current crop from the live track store.
    """
    event_id  = uuid.uuid4().hex
    now_ms    = int(time.time() * 1000)

    # Auto-fetch snapshot if not supplied
    if snapshot_b64 is None and global_id:
        crop = _fetch_crop_for_gid(global_id)
        snapshot_b64 = _encode_snapshot(crop)

    with _write_lock:
        try:
            conn = _get_conn()
            conn.execute(
                """INSERT INTO flagged_events
                   (id, ts, camera_id, global_id, event_type, severity, note,
                    snapshot_b64, reviewed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (event_id, now_ms, camera_id, global_id, event_type,
                 severity, note, snapshot_b64, now_ms)
            )
            conn.commit()
            conn.close()
            logger.info(f"[EventsDB] Flagged event {event_id} type={event_type} cam={camera_id} gid={global_id}")
        except Exception as e:
            logger.error(f"[EventsDB] flag_event error: {e}")
            return ""

    return event_id


def list_events(
    camera_id: Optional[str] = None,
    reviewed:  Optional[bool] = None,
    limit:     int = 100,
) -> list[dict]:
    """
    Return flagged events as a list of dicts, newest first.

    Args:
        camera_id: Filter to a specific camera (None = all cameras).
        reviewed:  True = only reviewed, False = only unreviewed, None = all.
        limit:     Maximum number of events to return (max 500).
    """
    limit = min(max(1, limit), 500)
    clauses, params = [], []

    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if reviewed is not None:
        clauses.append("reviewed = ?")
        params.append(1 if reviewed else 0)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    try:
        conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM flagged_events {where} ORDER BY ts DESC LIMIT ?", params
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"[EventsDB] list_events error: {e}")
        return []


def mark_reviewed(event_id: str) -> bool:
    """Mark a flagged event as reviewed. Returns True on success."""
    with _write_lock:
        try:
            conn = _get_conn()
            conn.execute(
                "UPDATE flagged_events SET reviewed = 1 WHERE id = ?", (event_id,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[EventsDB] mark_reviewed error: {e}")
            return False


def delete_event(event_id: str) -> bool:
    """Permanently delete a flagged event. Returns True on success."""
    with _write_lock:
        try:
            conn = _get_conn()
            conn.execute("DELETE FROM flagged_events WHERE id = ?", (event_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"[EventsDB] delete_event error: {e}")
            return False


def get_event(event_id: str) -> Optional[dict]:
    """Return a single event dict by id, or None."""
    try:
        conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM flagged_events WHERE id = ?", (event_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.debug(f"[EventsDB] get_event error: {e}")
        return None
