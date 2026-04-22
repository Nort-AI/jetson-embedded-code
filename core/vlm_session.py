"""
vlm_session.py — Per-camera VLM interaction layer for NORT.

Sits between camera_processor.py and vlm_analyst.py.
Adds throttling, auto-scan, result callbacks, and a per-camera
request lifecycle so VLM jobs are never accidentally spammed.

Usage (in CameraProcessor.__init__):
    from core import vlm_session
    self.vlm_session = vlm_session.get_session(self.camera_id)

Usage (in tracking loop, once per detected track):
    self.vlm_session.request(str(global_id), mode="describe")
    vlm_result = self.vlm_session.get(str(global_id))
    # pass vlm_result to bbox_renderer.draw_hud_box(...)

Usage (once per frame, for auto-scan):
    self.vlm_session.tick(active_global_ids)
"""

import time
import logging
import threading
from typing import Callable, Dict, List, Optional

from core import vlm_analyst

logger = logging.getLogger(__name__)


# ── Configuration defaults ─────────────────────────────────────────────────────
_DEFAULT_COOLDOWN_S   = 20.0   # seconds between re-analyses of the same track
_AUTO_SCAN_INTERVAL_S = 2.0    # seconds between auto-scan ticks (wall-clock)
_RESULT_TTL_S         = 300    # how long to keep a completed result locally


# ── VLMSession ─────────────────────────────────────────────────────────────────

class VLMSession:
    """
    Per-camera VLM interaction manager.

    Responsibilities:
    - Throttle: each track is blocked from re-submission for `cooldown_s` seconds.
    - Auto-scan: `tick()` selects the most-stale active track and fires a job.
    - Callback: notify caller when a pending result becomes "done".
    - Result cache: exposes `get()` wrapping vlm_analyst.get_result().
    """

    def __init__(
        self,
        camera_id: str,
        cooldown_s: float = _DEFAULT_COOLDOWN_S,
        auto_scan: bool = True,
        auto_scan_mode: str = "describe",
    ) -> None:
        self.camera_id      = camera_id
        self.cooldown_s     = cooldown_s
        self.auto_scan      = auto_scan
        self.auto_scan_mode = auto_scan_mode

        # ts of last submission per track_id
        self._last_submitted: Dict[str, float] = {}
        # ts of last known "done" status per track_id (for callback dedup)
        self._last_completed: Dict[str, float] = {}
        # optional result callback
        self._callback: Optional[Callable[[str, dict], None]] = None
        # last wall-clock time auto-scan ran
        self._last_auto_scan_ts: float = 0.0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def request(
        self,
        track_id: str,
        mode: str = "describe",
        question: str = "",
    ) -> bool:
        """
        Submit a VLM analysis job for *track_id* if not on cooldown.

        Returns True if a job was queued, False if throttled or VLM disabled.
        """
        if not vlm_analyst.is_enabled():
            return False

        track_id = str(track_id)
        now = time.time()

        with self._lock:
            last = self._last_submitted.get(track_id, 0.0)
            if now - last < self.cooldown_s:
                return False          # still on cooldown

            # Require at least one crop to be saved before submitting
            if not vlm_analyst.has_crop(track_id):
                return False

            queued = vlm_analyst.submit_analysis(track_id, question=question, mode=mode)
            if queued:
                self._last_submitted[track_id] = now
                logger.debug(f"[VLMSession:{self.camera_id}] Submitted {mode} for track {track_id}")
            return queued

    def get(self, track_id: str) -> dict:
        """
        Return the latest VLM result for *track_id*.

        Fires the result callback (once) when status transitions to 'done'.
        Format: {"status": "pending"|"done"|"error"|"not_found", "text": str, ...}
        """
        track_id = str(track_id)
        result   = vlm_analyst.get_result(track_id)

        # Fire callback exactly once per completed result
        if result.get("status") == "done" and self._callback is not None:
            result_ts = result.get("ts", 0.0)
            with self._lock:
                if self._last_completed.get(track_id, 0.0) < result_ts:
                    self._last_completed[track_id] = result_ts
                    try:
                        self._callback(track_id, result)
                    except Exception as e:
                        logger.warning(f"[VLMSession:{self.camera_id}] Callback error: {e}")

        return result

    def tick(self, active_track_ids: List[str]) -> None:
        """
        Call **once per frame** with the list of currently visible global IDs.

        Runs auto-scan at most every _AUTO_SCAN_INTERVAL_S seconds:
        picks the track whose last submission is oldest (or never submitted)
        and fires a background analysis job.
        """
        if not self.auto_scan or not vlm_analyst.is_enabled():
            return

        now = time.time()
        if now - self._last_auto_scan_ts < _AUTO_SCAN_INTERVAL_S:
            return
        self._last_auto_scan_ts = now

        if not active_track_ids:
            return

        # Pick the track most overdue for analysis
        with self._lock:
            submitted = self._last_submitted
            # Filter to those that have a crop saved
            candidates = [t for t in active_track_ids if vlm_analyst.has_crop(str(t))]
            if not candidates:
                return
            best = min(candidates, key=lambda t: submitted.get(str(t), 0.0))

        # Submit outside the lock to avoid potential re-entry
        self.request(str(best), mode=self.auto_scan_mode)

    def set_result_callback(self, fn: Callable[[str, dict], None]) -> None:
        """
        Register a callback invoked (once) when a job result becomes 'done'.

        fn(track_id: str, result: dict) — called from the thread that calls get().
        """
        self._callback = fn

    def clear_track(self, track_id: str) -> None:
        """Remove cooldown state for a track that has been cleaned up."""
        track_id = str(track_id)
        with self._lock:
            self._last_submitted.pop(track_id, None)
            self._last_completed.pop(track_id, None)

    def get_stats(self) -> dict:
        """Return a snapshot of session state (useful for admin/debug)."""
        with self._lock:
            return {
                "camera_id":        self.camera_id,
                "cooldown_s":       self.cooldown_s,
                "auto_scan":        self.auto_scan,
                "total_submitted":  len(self._last_submitted),
                "total_completed":  len(self._last_completed),
                "last_auto_scan_age_s": round(time.time() - self._last_auto_scan_ts, 1),
            }


# ── Global registry ────────────────────────────────────────────────────────────

_sessions: Dict[str, VLMSession] = {}
_registry_lock = threading.Lock()


def get_session(
    camera_id: str,
    cooldown_s: float = _DEFAULT_COOLDOWN_S,
    auto_scan: bool = True,
    auto_scan_mode: str = "describe",
) -> VLMSession:
    """
    Return an existing VLMSession for *camera_id* or create one.

    All kwargs are only applied on first creation; subsequent calls return
    the cached session unchanged.
    """
    with _registry_lock:
        if camera_id not in _sessions:
            _sessions[camera_id] = VLMSession(
                camera_id=camera_id,
                cooldown_s=cooldown_s,
                auto_scan=auto_scan,
                auto_scan_mode=auto_scan_mode,
            )
            logger.info(f"[vlm_session] Created session for camera '{camera_id}' "
                        f"(cooldown={cooldown_s}s, auto_scan={auto_scan})")
        return _sessions[camera_id]


def all_stats() -> dict:
    """Return stats dict for all registered sessions (useful for admin endpoint)."""
    with _registry_lock:
        return {cid: s.get_stats() for cid, s in _sessions.items()}
