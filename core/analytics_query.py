"""
core/analytics_query.py — Historical analytics for the Nort NL interface.

Reads from spatial_log.db (fast, indexed SQLite) for count/time queries.
Falls back to tracking_log.csv for zone-distribution queries (CSV has zone col).

All functions are read-only and cache results for 5 minutes so repeated NL
queries (e.g. "how many today?" asked twice) cost a single SQLite read.

Zero Claude API calls — every answer here is free.
"""

import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

try:
    from system import config as _cfg
    _BASE_DIR = _cfg.BASE_DIR
    _CSV_FILE = os.path.join(_BASE_DIR, _cfg.CSV_FILENAME)
    _TZ_NAME  = None  # use local system timezone
except Exception:
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _CSV_FILE = os.path.join(_BASE_DIR, "tracking_log.csv")
    _TZ_NAME  = None

try:
    from system.logger_setup import setup_logger
    logger = setup_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(_BASE_DIR, "spatial_log.db")

# ── Simple 5-minute result cache ─────────────────────────────────────────────
_cache: dict = {}          # key → (result, expires_ts)
_cache_lock = threading.Lock()
_CACHE_TTL = 300           # 5 minutes


def _cached(key: str, fn, *args, **kwargs):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() < entry[1]:
            return entry[0]
    result = fn(*args, **kwargs)
    with _cache_lock:
        _cache[key] = (result, time.time() + _CACHE_TTL)
    return result


# ── DB connection (read-only, separate from spatial_logger's writer) ─────────
_rconn: Optional[sqlite3.Connection] = None
_rconn_lock = threading.Lock()


def _get_conn() -> Optional[sqlite3.Connection]:
    global _rconn
    with _rconn_lock:
        if _rconn is None:
            if not os.path.exists(_DB_PATH):
                return None
            try:
                _rconn = sqlite3.connect(
                    f"file:{_DB_PATH}?mode=ro", uri=True,
                    check_same_thread=False
                )
                _rconn.row_factory = sqlite3.Row
            except Exception as e:
                logger.warning(f"[Analytics] Cannot open {_DB_PATH}: {e}")
                return None
        return _rconn


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _midnight_ts(date_str: Optional[str] = None, offset_days: int = 0) -> tuple:
    """Return (start_ms, end_ms) for a calendar day in local time."""
    if date_str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if offset_days:
        from datetime import timedelta
        dt = dt + timedelta(days=offset_days)
    start_ms = int(dt.timestamp() * 1000)
    end_ms   = start_ms + 86_400_000  # +24 hours
    return start_ms, end_ms


# ═══════════════════════════════════════════════════════════════════════════════
# CORE QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def query_today_total(camera_id: Optional[str] = None) -> int:
    """Count distinct track_ids seen today (since midnight local time)."""
    start_ms, end_ms = _midnight_ts()
    key = f"today_total:{camera_id}:{start_ms}"
    def _run():
        conn = _get_conn()
        if conn is None:
            return 0
        try:
            if camera_id:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT track_id) FROM positions "
                    "WHERE ts >= ? AND ts < ? AND camera_id = ?",
                    (start_ms, end_ms, camera_id)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT track_id) FROM positions "
                    "WHERE ts >= ? AND ts < ?",
                    (start_ms, end_ms)
                ).fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.debug(f"[Analytics] query_today_total error: {e}")
            return 0
    return _cached(key, _run)


def query_yesterday_total(camera_id: Optional[str] = None) -> int:
    """Count distinct track_ids seen yesterday."""
    start_ms, end_ms = _midnight_ts(offset_days=-1)
    key = f"yesterday_total:{camera_id}:{start_ms}"
    def _run():
        conn = _get_conn()
        if conn is None:
            return 0
        try:
            if camera_id:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT track_id) FROM positions "
                    "WHERE ts >= ? AND ts < ? AND camera_id = ?",
                    (start_ms, end_ms, camera_id)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT track_id) FROM positions "
                    "WHERE ts >= ? AND ts < ?",
                    (start_ms, end_ms)
                ).fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.debug(f"[Analytics] query_yesterday_total error: {e}")
            return 0
    return _cached(key, _run)


def query_hourly_traffic(camera_id: Optional[str] = None,
                         date_str: Optional[str] = None) -> dict:
    """
    Returns {hour_int: count} for a given calendar day (default today).
    hour_int is 0–23 in local time.
    """
    start_ms, end_ms = _midnight_ts(date_str)
    key = f"hourly:{camera_id}:{start_ms}"
    def _run():
        conn = _get_conn()
        if conn is None:
            return {}
        try:
            # SQLite stores ts in ms; convert to seconds for strftime
            if camera_id:
                rows = conn.execute(
                    "SELECT CAST(strftime('%H', datetime(ts/1000, 'unixepoch', 'localtime')) AS INTEGER) "
                    "  AS hr, COUNT(DISTINCT track_id) AS cnt "
                    "FROM positions WHERE ts >= ? AND ts < ? AND camera_id = ? "
                    "GROUP BY hr ORDER BY hr",
                    (start_ms, end_ms, camera_id)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT CAST(strftime('%H', datetime(ts/1000, 'unixepoch', 'localtime')) AS INTEGER) "
                    "  AS hr, COUNT(DISTINCT track_id) AS cnt "
                    "FROM positions WHERE ts >= ? AND ts < ? "
                    "GROUP BY hr ORDER BY hr",
                    (start_ms, end_ms)
                ).fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception as e:
            logger.debug(f"[Analytics] query_hourly_traffic error: {e}")
            return {}
    return _cached(key, _run)


def query_peak_hour(camera_id: Optional[str] = None,
                    date_str: Optional[str] = None) -> tuple:
    """Returns (peak_hour: int, count: int). (None, 0) if no data."""
    hourly = query_hourly_traffic(camera_id, date_str)
    if not hourly:
        return (None, 0)
    peak_h = max(hourly, key=lambda h: hourly[h])
    return (peak_h, hourly[peak_h])


def query_busiest_period(camera_id: Optional[str] = None,
                         window_min: int = 30) -> dict:
    """
    Find the busiest consecutive window_min-minute window today.
    Returns {"start": "HH:MM", "end": "HH:MM", "count": N} or {}.
    """
    start_ms, end_ms = _midnight_ts()
    key = f"busiest_period:{camera_id}:{window_min}:{start_ms}"
    def _run():
        conn = _get_conn()
        if conn is None:
            return {}
        try:
            window_ms = window_min * 60 * 1000
            if camera_id:
                rows = conn.execute(
                    "SELECT ts FROM positions WHERE ts >= ? AND ts < ? AND camera_id = ? "
                    "ORDER BY ts",
                    (start_ms, end_ms, camera_id)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ts FROM positions WHERE ts >= ? AND ts < ? ORDER BY ts",
                    (start_ms, end_ms)
                ).fetchall()
            if not rows:
                return {}
            timestamps = [r[0] for r in rows]
            best_start, best_count = timestamps[0], 0
            left = 0
            for right in range(len(timestamps)):
                while timestamps[right] - timestamps[left] > window_ms:
                    left += 1
                if right - left + 1 > best_count:
                    best_count = right - left + 1
                    best_start = timestamps[left]
            s = datetime.fromtimestamp(best_start / 1000)
            e = datetime.fromtimestamp((best_start + window_ms) / 1000)
            return {"start": s.strftime("%H:%M"), "end": e.strftime("%H:%M"),
                    "count": best_count}
        except Exception as ex:
            logger.debug(f"[Analytics] query_busiest_period error: {ex}")
            return {}
    return _cached(key, _run)


def query_last_hour_total(camera_id: Optional[str] = None) -> int:
    """Count distinct track_ids seen in the last 60 minutes."""
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - 3_600_000
    key = f"last_hour:{camera_id}:{now_ms // 60000}"  # 1-min resolution cache key
    def _run():
        conn = _get_conn()
        if conn is None:
            return 0
        try:
            if camera_id:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT track_id) FROM positions "
                    "WHERE ts >= ? AND camera_id = ?",
                    (since_ms, camera_id)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT track_id) FROM positions WHERE ts >= ?",
                    (since_ms,)
                ).fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.debug(f"[Analytics] query_last_hour_total error: {e}")
            return 0
    return _cached(key, _run)


# ═══════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE ANSWER BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_hour(h: int) -> str:
    """Format hour int as '2pm' / '14h'."""
    if h is None:
        return "unknown"
    return f"{h}:00–{h+1}:00"


def answer_natural_history_query(query: str, cls: dict,
                                 lang: str = "en") -> Optional[str]:
    """
    Map a classified historical query to the right analytics function and
    format a natural-language response. Returns None if no data is available.
    """
    low = query.lower()
    pt  = (lang == "pt")
    cam = cls.get("camera_id")

    try:
        # "how many people came in today?"
        if re.search(r'\b(today|hoje)\b', low):
            n = query_today_total(cam)
            if n == 0:
                return ("Ainda não há dados para hoje." if pt
                        else "No tracking data for today yet.")
            y = query_yesterday_total(cam)
            cam_part = ""
            if cam:
                try:
                    from core import store_context as _sc
                    cam_part = f" em {_sc.get_camera_label(cam)}"
                except Exception:
                    cam_part = f" em {cam}"
            if y > 0:
                diff = n - y
                diff_str = f"+{diff}" if diff >= 0 else str(diff)
                return (
                    f"{n} pessoas visitaram{cam_part} hoje "
                    f"({diff_str} em relação a ontem)."
                    if pt else
                    f"{n} people tracked{cam_part} today "
                    f"({diff_str} vs yesterday)."
                )
            return (
                f"{n} pessoas visitaram{cam_part} hoje."
                if pt else
                f"{n} people tracked{cam_part} today."
            )

        # "how many yesterday?"
        if re.search(r'\b(yesterday|ontem)\b', low):
            n = query_yesterday_total(cam)
            if n == 0:
                return ("Sem dados para ontem." if pt
                        else "No tracking data for yesterday.")
            cam_part = ""
            if cam:
                try:
                    from core import store_context as _sc
                    cam_part = f" em {_sc.get_camera_label(cam)}"
                except Exception:
                    cam_part = f" em {cam}"
            return (
                f"{n} pessoas foram registradas{cam_part} ontem."
                if pt else
                f"{n} people were tracked{cam_part} yesterday."
            )

        # "when was it busiest?" / "peak hour?"
        if re.search(r'\b(peak|pico|busiest|rush|hor[aá]rio)\b', low):
            h, cnt = query_peak_hour(cam)
            if h is None:
                return ("Ainda não há dados suficientes para hoje." if pt
                        else "Not enough data for today yet.")
            bp = query_busiest_period(cam)
            cam_part = ""
            if cam:
                try:
                    from core import store_context as _sc
                    cam_part = f" em {_sc.get_camera_label(cam)}"
                except Exception:
                    cam_part = f" em {cam}"
            if bp:
                return (
                    f"Horário de pico{cam_part}: {bp['start']}–{bp['end']} "
                    f"({bp['count']} detecções). Hora mais movimentada: {_fmt_hour(h)}."
                    if pt else
                    f"Busiest period{cam_part}: {bp['start']}–{bp['end']} "
                    f"({bp['count']} detections). Peak hour: {_fmt_hour(h)}."
                )
            return (
                f"Hora de pico{cam_part}: {_fmt_hour(h)} ({cnt} detecções)."
                if pt else
                f"Peak hour{cam_part}: {_fmt_hour(h)} ({cnt} detections)."
            )

        # "last hour"
        if re.search(r'\b(last hour|[uú]ltima hora)\b', low):
            n = query_last_hour_total(cam)
            cam_part = ""
            if cam:
                try:
                    from core import store_context as _sc
                    cam_part = f" em {_sc.get_camera_label(cam)}"
                except Exception:
                    cam_part = f" em {cam}"
            return (
                f"{n} {'pessoa' if n == 1 else 'pessoas'} na última hora{cam_part}."
                if pt else
                f"{n} {'person' if n == 1 else 'people'} in the last hour{cam_part}."
            )

        # "how does today compare?" / trend
        if re.search(r'\b(compared|comparado|trend|tend[eê]ncia|vs\.?)\b', low):
            t = query_today_total(cam)
            y = query_yesterday_total(cam)
            if t == 0 and y == 0:
                return ("Sem dados suficientes para comparar." if pt
                        else "Not enough data to compare.")
            diff = t - y
            pct  = int(abs(diff) / max(y, 1) * 100)
            direction = ("acima" if diff >= 0 else "abaixo") if pt else ("up" if diff >= 0 else "down")
            return (
                f"Hoje: {t} pessoas, ontem: {y}. "
                f"Movimento {direction} {pct}%."
                if pt else
                f"Today: {t} people, yesterday: {y}. "
                f"Traffic {direction} {pct}%."
            )

    except Exception as e:
        logger.debug(f"[Analytics] answer_natural_history_query error: {e}")

    return None  # caller falls through to VLM
