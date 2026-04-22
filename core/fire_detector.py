"""
fire_detector.py — Tiered fire & smoke detection for NORT.

Cost model (ultra-cheap):
  Layer 1 — OpenCV HSV color filter  → runs every N frames,  FREE (CPU, <1ms)
  Layer 2 — VLM confirmation         → called ONLY when L1 triggers, ~$0.0008/query

In a normal retail environment L1 should fire at most a handful of times per day
(bright clothing, direct sunlight) so Layer 2 API cost is essentially $0.

Configuration (device.json  →  "fire_detection" section):
    {
        "fire_detection": {
            "enabled": true,
            "check_every_n_frames": 30,     // how often L1 runs per camera (default: 30)
            "fire_pixel_ratio": 0.003,      // fraction of frame needed to trigger L1 (default: 0.3%)
            "smoke_pixel_ratio": 0.04,       // smoke is subtler; higher area needed
            "alert_cooldown_seconds": 300,   // min gap between repeated alerts per camera
            "alert_webhook_url": "",         // optional: POST JSON alert here
            "telegram_token": "",            // optional: Telegram bot token
            "telegram_chat_id": ""           // optional: Telegram chat id
        }
    }
"""

import cv2
import time
import logging
import threading
import json
import os
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load fire_detection section from device.json."""
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "device.json"),
        "device.json",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f).get("fire_detection", {})
            except Exception:
                pass
    return {}

_cfg = _load_config()

ENABLED                 = _cfg.get("enabled", True)
CHECK_EVERY_N_FRAMES    = int(_cfg.get("check_every_n_frames", 30))
FIRE_PIXEL_RATIO        = float(_cfg.get("fire_pixel_ratio", 0.003))   # 0.3% of frame
SMOKE_PIXEL_RATIO       = float(_cfg.get("smoke_pixel_ratio", 0.04))   # 4% of frame
ALERT_COOLDOWN          = float(_cfg.get("alert_cooldown_seconds", 300))
ALERT_WEBHOOK_URL       = _cfg.get("alert_webhook_url", "")
TELEGRAM_TOKEN          = _cfg.get("telegram_token", "")
TELEGRAM_CHAT_ID        = _cfg.get("telegram_chat_id", "")

# ── Layer 1: OpenCV HSV color pre-filter ─────────────────────────────────────
#
# Fire  → warm hues (red/orange/yellow), high saturation, high brightness.
#          Two HSV ranges because red wraps around 0°/180° in OpenCV.
# Smoke → near-neutral, low saturation, mid-brightness gray cloud.
#
# All constants are empirically tuned for indoor retail lighting.

_FIRE_LOWER_1  = np.array([0,   120,  150], dtype=np.uint8)  # red-orange (low boundary)
_FIRE_UPPER_1  = np.array([25,  255,  255], dtype=np.uint8)
_FIRE_LOWER_2  = np.array([160, 120,  150], dtype=np.uint8)  # red (hue wrap-around)
_FIRE_UPPER_2  = np.array([180, 255,  255], dtype=np.uint8)

_SMOKE_LOWER   = np.array([0,   0,   100], dtype=np.uint8)   # low-sat gray
_SMOKE_UPPER   = np.array([180, 50,  200], dtype=np.uint8)


def _hsv_fire_score(frame_bgr: np.ndarray) -> dict:
    """
    Run HSV analysis on a frame.

    Returns:
        {
            "fire":  bool,   # Layer 1 triggered for fire
            "smoke": bool,   # Layer 1 triggered for smoke
            "fire_ratio":  float,
            "smoke_ratio": float,
        }
    """
    # Downsample to 320px wide for speed (still reliable for color detection)
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 320 / max(w, 1))
    if scale < 1.0:
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    total_px = hsv.shape[0] * hsv.shape[1]

    # Fire mask (two ranges for red hue wrap-around)
    fire_mask = cv2.bitwise_or(
        cv2.inRange(hsv, _FIRE_LOWER_1, _FIRE_UPPER_1),
        cv2.inRange(hsv, _FIRE_LOWER_2, _FIRE_UPPER_2),
    )
    # Optional morphological cleanup: remove single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_px = int(cv2.countNonZero(fire_mask))
    fire_ratio = fire_px / total_px

    # Smoke mask
    smoke_mask = cv2.inRange(hsv, _SMOKE_LOWER, _SMOKE_UPPER)
    # Smoke excludes fire pixels (avoid double-counting orange glow)
    smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(fire_mask))
    smoke_px = int(cv2.countNonZero(smoke_mask))
    smoke_ratio = smoke_px / total_px

    return {
        "fire":        fire_ratio  >= FIRE_PIXEL_RATIO,
        "smoke":       smoke_ratio >= SMOKE_PIXEL_RATIO,
        "fire_ratio":  round(fire_ratio, 5),
        "smoke_ratio": round(smoke_ratio, 5),
    }


# ── Layer 2: VLM confirmation ─────────────────────────────────────────────────

_VLM_FIRE_PROMPT = (
    "Is there any visible fire, flames, or smoke in this image? "
    "Answer YES or NO, then in one sentence explain what you observe."
)

def _vlm_confirm(frame_bgr: np.ndarray) -> Optional[str]:
    """Run VLM query to confirm the HSV alert. Returns the answer text or None."""
    try:
        from core import vlm_analyst
        if not vlm_analyst.is_enabled():
            return None
        job_id = vlm_analyst.submit_scene_query("__fire_check__", frame_bgr, _VLM_FIRE_PROMPT)
        # Poll up to 15 s
        deadline = time.time() + 15
        while time.time() < deadline:
            result = vlm_analyst.get_search_result(job_id)
            if result.get("status") == "done":
                return result.get("result", "")
            time.sleep(0.5)
        return None
    except Exception as e:
        logger.warning(f"[FireDetector] VLM confirmation error: {e}")
        return None


# ── Alert dispatcher ──────────────────────────────────────────────────────────

def _send_alert(camera_id: str, kind: str, vlm_text: str, frame_bgr: np.ndarray):
    """Dispatch fire/smoke alert through all configured channels."""
    msg = (
        f"🔥 FIRE ALERT — Camera: {camera_id}\n"
        f"Type: {kind.upper()}\n"
        f"VLM analysis: {vlm_text}"
    )
    logger.critical(f"[FireDetector] {msg}")

    # ── Webhook ──────────────────────────────────────────────────────────────
    if ALERT_WEBHOOK_URL:
        try:
            import requests
            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            import base64
            img_b64 = base64.b64encode(buf.tobytes()).decode()
            requests.post(
                ALERT_WEBHOOK_URL,
                json={
                    "event": "fire_alert",
                    "camera_id": camera_id,
                    "type": kind,
                    "vlm_text": vlm_text,
                    "image_base64": img_b64,
                    "ts": time.time(),
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"[FireDetector] Webhook failed: {e}")

    # ── Telegram ─────────────────────────────────────────────────────────────
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            import requests
            import io
            # Send image with caption
            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": msg},
                files={"photo": ("alert.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
                timeout=8,
            )
        except Exception as e:
            logger.warning(f"[FireDetector] Telegram alert failed: {e}")


# ── Background watcher thread ─────────────────────────────────────────────────

# Per-camera frame counters and cooldown timestamps — no mutex needed (one writer)
_frame_counters: Dict[str, int] = {}
_last_alert_ts: Dict[str, float] = {}
_pending_vlm: Dict[str, str] = {}   # camera_id → job_id (in-flight VLM jobs)
_lock = threading.Lock()


def _fire_watch_loop(get_frame_fn, get_camera_ids_fn, stop_event: threading.Event):
    """
    Daemon loop called from run.py.

    Args:
        get_frame_fn:      callable(camera_id) → np.ndarray | None
        get_camera_ids_fn: callable() → list[str]  — returns active camera ids
        stop_event:        threading.Event shared with the main app.
    """
    if not ENABLED:
        logger.info("[FireDetector] Disabled in configuration.")
        return

    logger.info(
        f"[FireDetector] Started — L1 every {CHECK_EVERY_N_FRAMES} frames, "
        f"fire_ratio≥{FIRE_PIXEL_RATIO}, smoke_ratio≥{SMOKE_PIXEL_RATIO}, "
        f"cooldown={ALERT_COOLDOWN}s"
    )

    try:
        from core import vlm_analyst
        _has_vlm = vlm_analyst.is_enabled()
    except Exception:
        _has_vlm = False

    if not _has_vlm:
        logger.warning("[FireDetector] VLM unavailable — L1-only mode (no VLM confirmation).")

    while not stop_event.is_set():
        camera_ids = get_camera_ids_fn()

        for cam_id in camera_ids:
            # Throttle: only run L1 every N frames
            with _lock:
                cnt = _frame_counters.get(cam_id, 0) + 1
                _frame_counters[cam_id] = cnt
            if cnt % CHECK_EVERY_N_FRAMES != 0:
                continue

            frame = get_frame_fn(cam_id)
            if frame is None:
                continue

            # ── Layer 1: fast HSV check ───────────────────────────────────
            scores = _hsv_fire_score(frame)
            triggered = scores["fire"] or scores["smoke"]

            if not triggered:
                continue

            kind = "fire" if scores["fire"] else "smoke"
            logger.info(
                f"[FireDetector] L1 TRIGGER on {cam_id} — "
                f"fire_ratio={scores['fire_ratio']}, smoke_ratio={scores['smoke_ratio']}"
            )

            # Cooldown guard
            with _lock:
                last = _last_alert_ts.get(cam_id, 0)
            if time.time() - last < ALERT_COOLDOWN:
                logger.debug(f"[FireDetector] {cam_id} still in cooldown, skipping.")
                continue

            if not _has_vlm:
                # No VLM → alert on L1 alone (higher false-positive risk)
                _send_alert(cam_id, kind, "VLM unavailable — HSV pre-filter triggered.", frame)
                with _lock:
                    _last_alert_ts[cam_id] = time.time()
                continue

            # ── Layer 2: VLM confirmation (runs in-line — already async inside vlm_analyst) ──
            logger.info(f"[FireDetector] L1 passed — asking VLM to confirm on {cam_id}…")
            vlm_text = _vlm_confirm(frame)
            if vlm_text is None:
                logger.warning(f"[FireDetector] VLM confirmation timed out for {cam_id}.")
                continue

            import re
            is_confirmed = (
                bool(re.search(r'\byes\b', vlm_text.lower()))
                and not vlm_text.lower().strip().startswith("no")
            )
            logger.info(f"[FireDetector] VLM result for {cam_id}: confirmed={is_confirmed} — {vlm_text[:120]}")

            if is_confirmed:
                _send_alert(cam_id, kind, vlm_text, frame)
                with _lock:
                    _last_alert_ts[cam_id] = time.time()

        # Sleep a bit between full sweeps (keeps CPU minimal)
        stop_event.wait(0.5)


def start(get_frame_fn, stop_event: threading.Event, get_camera_ids_fn=None) -> threading.Thread:
    """
    Spawn and return the fire-watcher daemon thread.

    Args:
        get_frame_fn:      callable(camera_id) → np.ndarray | None
        stop_event:        shared stop signal from run.py
        get_camera_ids_fn: callable() → list[str] (optional, defaults to returning [])

    Example (in run.py)::

        from core.fire_detector import start as start_fire_detector

        def _get_raw_frame(cam_id):
            with lock:
                for p in processors:
                    if p.camera_id == cam_id:
                        raw = getattr(p, 'latest_raw_frame', None)
                        return raw.copy() if raw is not None else None
            return None

        def _get_cam_ids():
            return [p.camera_id for p in processors]

        start_fire_detector(_get_raw_frame, stop_event, _get_cam_ids)
    """
    if get_camera_ids_fn is None:
        get_camera_ids_fn = lambda: []

    t = threading.Thread(
        target=_fire_watch_loop,
        args=(get_frame_fn, get_camera_ids_fn, stop_event),
        daemon=True,
        name="FireDetector",
    )
    t.start()
    logger.info("[FireDetector] Watcher thread started.")
    return t
