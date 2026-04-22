"""
frame_buffer.py — 5-minute per-camera DVR ring buffer.

Saves 1 annotated JPEG per second per camera into a temp directory.
Old frames beyond the retention window are pruned automatically.
"""
import os
import time
import threading
import tempfile
import cv2

BUFFER_SECONDS  = 300          # 5-minute window
_SAVE_INTERVAL  = 1.0          # save at most once per second
_JPEG_QUALITY   = 60           # lower = smaller files (~20-30 KB each)
_BUFFER_DIR     = os.path.join(tempfile.gettempdir(), "nort_dvr")


class FrameRingBuffer:
    """
    Per-camera ring buffer that saves one JPEG/sec to disk and prunes
    frames older than BUFFER_SECONDS.

    Usage:
        buf = FrameRingBuffer("cam1")
        buf.push(annotated_bgr_frame)   # call every frame; internally rate-limited
        entries = buf.list_entries()    # [(epoch_int, path), ...]
        path = buf.seek(epoch_int)      # nearest frame path (or None)
    """

    def __init__(self, camera_id: str):
        self.camera_id    = str(camera_id)
        self._dir         = os.path.join(_BUFFER_DIR, self.camera_id)
        os.makedirs(self._dir, exist_ok=True)
        self._last_save   = 0.0
        self._lock        = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, frame) -> None:
        """Rate-limited: saves at most one frame per second."""
        now = time.time()
        if now - self._last_save < _SAVE_INTERVAL:
            return
        self._last_save = now
        ts = int(now)
        path = os.path.join(self._dir, f"{ts}.jpg")
        try:
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        except Exception:
            pass
        self._prune(now)

    def list_entries(self):
        """Return sorted list of (timestamp_int, filepath) tuples within window."""
        cutoff = int(time.time()) - BUFFER_SECONDS
        entries = []
        try:
            for fname in os.listdir(self._dir):
                if not fname.endswith(".jpg"):
                    continue
                try:
                    ts = int(fname[:-4])
                except ValueError:
                    continue
                if ts >= cutoff:
                    entries.append((ts, os.path.join(self._dir, fname)))
        except FileNotFoundError:
            pass
        entries.sort(key=lambda x: x[0])
        return entries

    def seek(self, target_ts: int):
        """Return the path of the frame whose timestamp is closest to target_ts."""
        entries = self.list_entries()
        if not entries:
            return None
        # Binary-search for nearest
        best = min(entries, key=lambda e: abs(e[0] - target_ts))
        return best[1]

    def min_ts(self) -> int:
        entries = self.list_entries()
        return entries[0][0] if entries else int(time.time())

    def max_ts(self) -> int:
        return int(time.time())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prune(self, now: float) -> None:
        """Delete frames older than the retention window."""
        cutoff = int(now) - BUFFER_SECONDS - 2   # 2-second grace
        with self._lock:
            try:
                for fname in os.listdir(self._dir):
                    if not fname.endswith(".jpg"):
                        continue
                    try:
                        ts = int(fname[:-4])
                    except ValueError:
                        continue
                    if ts < cutoff:
                        try:
                            os.remove(os.path.join(self._dir, fname))
                        except OSError:
                            pass
            except FileNotFoundError:
                pass


# ── Global registry ───────────────────────────────────────────────────────────

_registry: dict[str, FrameRingBuffer] = {}
_reg_lock = threading.Lock()


def get_buffer(camera_id: str) -> FrameRingBuffer:
    """Return (creating if needed) the ring buffer for the given camera."""
    with _reg_lock:
        if camera_id not in _registry:
            _registry[camera_id] = FrameRingBuffer(camera_id)
        return _registry[camera_id]


def list_cameras() -> list[str]:
    with _reg_lock:
        return list(_registry.keys())
