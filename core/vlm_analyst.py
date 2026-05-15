"""
vlm_analyst.py — Visual Language Model analyst for NORT V1.

Runs moondream2 (1.9B INT4, ~1.4GB VRAM) with configuration-driven settings.
Designed for Jetson edge deployment with cloud API fallback.

Key Features:
- Non-blocking inference via background worker thread
- Configurable via device.json (enable/disable, model settings, rate limits)
- Automatic local/cloud mode selection based on kestrel-native availability
- Metrics tracking for performance monitoring
- LRU crop cache with configurable limits

Usage:
    from core.vlm_analyst import submit_analysis, get_result, get_crop_jpeg, is_enabled

    if is_enabled():
        submit_analysis(track_id, question="What is this person carrying?")
        result = get_result(track_id)
"""

import threading
import time
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

from system.logger_setup import setup_logger
logger = setup_logger(__name__)

# ── Circuit breaker for cloud API ─────────────────────────────────────────────
_cloud_consecutive_failures = 0
_CLOUD_MAX_FAILURES = 5  # Disable cloud after this many consecutive errors
_cloud_disabled_logged = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VLMConfig:
    """VLM configuration loaded from device.json with defaults."""
    enabled: bool = True
    warmup_on_start: bool = False
    prefer_local: bool = True          # Prefer kestrel-native local inference
    allow_cloud_fallback: bool = False  # Allow cloud API if local fails
    cloud_api_key: str = ""            # Moondream cloud API key
    
    # Resource limits
    max_crops: int = 200               # Max cached crops (LRU eviction)
    max_queue_size: int = 32           # Max pending analysis jobs
    job_timeout: float = 5.0             # Seconds to wait for queue
    
    # Rate limiting for crop capture
    capture_interval_frames: int = 10    # Capture crop every N frames
    min_sharpness: float = 10.0        # Minimum Laplacian variance
    min_crop_area: int = 5000          # Minimum pixels in crop
    
    # Model settings
    model_version: str = "moondream-2"  # Model identifier
    max_tokens: int = 512              # Max response tokens
    temperature: float = 0.2           # Lower = more consistent

    # Claude Haiku provider (cheaper + faster than Moondream cloud, multi-image support)
    # Set "anthropic_api_key" in device.json vlm section to enable.
    anthropic_api_key: str = ""

    # CLIP settings
    clip_enabled: bool = True
    clip_model: str = "ViT-B/32"
    clip_labels: List[str] = None
    
    def __post_init__(self):
        if self.clip_labels is None:
            self.clip_labels = ["store employee in uniform", "customer shopping"]


def _load_vlm_config() -> VLMConfig:
    """Load VLM configuration from device.json with environment overrides."""
    cfg = VLMConfig()
    
    # Try to load from device.json
    device_json_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "device.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "Jetson-Embedded-code-3", "device.json"),
        os.path.join(os.path.dirname(__file__), "..", "device.json"),
        "device.json",
    ]
    
    device_data = {}
    for path in device_json_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    device_data = json.load(f)
                    logger.debug(f"[VLM] Loaded device.json from {path}")
                    break
            except Exception as e:
                logger.debug(f"[VLM] Could not load {path}: {e}")
    
    vlm_section = device_data.get("vlm", {})
    
    # Override defaults with device.json settings
    cfg.enabled = vlm_section.get("enabled", cfg.enabled)
    cfg.warmup_on_start = vlm_section.get("warmup_on_start", cfg.warmup_on_start)
    cfg.prefer_local = vlm_section.get("prefer_local", cfg.prefer_local)
    cfg.allow_cloud_fallback = vlm_section.get("allow_cloud_fallback", cfg.allow_cloud_fallback)
    cfg.cloud_api_key = vlm_section.get("cloud_api_key", device_data.get("moondream_api_key", ""))
    cfg.anthropic_api_key = vlm_section.get("anthropic_api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
    cfg.max_crops = vlm_section.get("max_crops", cfg.max_crops)
    cfg.max_queue_size = vlm_section.get("max_queue_size", cfg.max_queue_size)
    cfg.capture_interval_frames = vlm_section.get("capture_interval_frames", cfg.capture_interval_frames)
    cfg.min_sharpness = vlm_section.get("min_sharpness", cfg.min_sharpness)
    cfg.min_crop_area = vlm_section.get("min_crop_area", cfg.min_crop_area)
    cfg.clip_enabled = vlm_section.get("clip_enabled", cfg.clip_enabled)
    
    # Environment variable overrides (highest priority)
    if os.environ.get("VLM_DISABLED", "").lower() in ("1", "true", "yes"):
        cfg.enabled = False
    if os.environ.get("MOONDREAM_API_KEY"):
        logger.info("[VLM] Using API key from MOONDREAM_API_KEY environment variable")
        cfg.cloud_api_key = os.environ["MOONDREAM_API_KEY"]
    if os.environ.get("VLM_LOCAL_ONLY", "").lower() in ("1", "true", "yes"):
        cfg.allow_cloud_fallback = False
        cfg.prefer_local = True
    
    # Debug: log config state
    key_src = "env" if os.environ.get("MOONDREAM_API_KEY") else ("vlm.cloud_api_key" if vlm_section.get("cloud_api_key") else ("moondream_api_key" if device_data.get("moondream_api_key") else "none"))
    key_preview = cfg.cloud_api_key[:8] + "..." if len(cfg.cloud_api_key) > 8 else ("[empty]" if not cfg.cloud_api_key else cfg.cloud_api_key)
    logger.info(f"[VLM] Config loaded: enabled={cfg.enabled}, prefer_local={cfg.prefer_local}, cloud_fallback={cfg.allow_cloud_fallback}, key_source={key_src}, key_preview={key_preview}")
    
    return cfg


# Global configuration instance
_CONFIG = _load_vlm_config()

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Shared crop store ─────────────────────────────────────────────────────────
# Dict[global_id_str → {"crop": ndarray, "ts": float, "cam": str,
#                        "pose_jpeg": bytes|None, "pose_data": dict|None,
#                        "pose_ts": float}]
_track_crops: "OrderedDict[str, dict]" = OrderedDict()
_track_crops_lock = threading.RLock()
MAX_CROPS = _CONFIG.max_crops

# ── Background pose estimation worker ─────────────────────────────────────────
# Crops are enqueued here (deduped: only the latest per track_id is kept).
# A single daemon thread drains the queue and writes results back into
# _track_crops so /api/pose/* can serve them without any inference latency.
_pose_pending: dict          = {}
_pose_pending_lock           = threading.Lock()
_pose_wake_event             = threading.Event()
_pose_worker_started         = False
_pose_worker_start_lock      = threading.Lock()
_POSE_MIN_INTERVAL           = 30.0   # seconds between re-estimates (normal mode)
# With --live-pose, interval drops to 0.15 s so the video-feed overlay stays fresh.


_active_target_id = None
_active_target_lock = threading.Lock()

def set_active_target(global_id: str):
    global _active_target_id
    with _active_target_lock:
        if not global_id or str(global_id).lower() == "none":
            _active_target_id = None
        else:
            _active_target_id = str(global_id)

def get_active_target():
    with _active_target_lock:
        return _active_target_id



def save_crop(global_id: str, crop_bgr: np.ndarray, camera_id: str,
              crop_bounds: "tuple | None" = None) -> None:
    """Called by camera_processor on each accepted frame for a track.

    crop_bounds — optional (cx1, cy1, cx2, cy2) pixel coords of the padded
                  crop in the full frame.  Stored so camera_processor can
                  back-project normalised keypoints onto the video feed.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return
    key = str(global_id)
    with _track_crops_lock:
        prev = _track_crops.get(key) or {}
        if key in _track_crops:
            _track_crops.move_to_end(key)
        _track_crops[key] = {
            "crop":        crop_bgr.copy(),
            "ts":          time.time(),
            "cam":         camera_id,
            "crop_bounds": crop_bounds,          # (cx1,cy1,cx2,cy2) in full frame
            # preserve pre-computed pose so it survives crop updates
            "pose_jpeg": prev.get("pose_jpeg"),
            "pose_data": prev.get("pose_data"),
            "pose_ts":   prev.get("pose_ts", 0.0),
        }
        while len(_track_crops) > MAX_CROPS:
            _track_crops.popitem(last=False)
    # Enqueue background pose estimation (respects _POSE_MIN_INTERVAL)
    _enqueue_pose(key, crop_bgr)


def _enqueue_pose(key: str, crop_bgr: np.ndarray, force: bool = False) -> None:
    """Put crop in the pose queue; replaces any pending crop for the same key.
    Skips if a fresh pose was already computed recently (_POSE_MIN_INTERVAL).
    Pass force=True to bypass the interval (e.g. when user explicitly requests
    pose on the crop they're currently looking at).
    """
    global _pose_worker_started
    if not force:
        # Rate-limit: skip if pose is still fresh.
        # With --live-pose the interval shrinks to 0.15 s so the video overlay stays fresh.
        try:
            from system import config as _cfg
            min_interval = 0.15 if getattr(_cfg, "LIVE_POSE_ENABLED", False) else _POSE_MIN_INTERVAL
        except Exception:
            min_interval = _POSE_MIN_INTERVAL
        with _track_crops_lock:
            pose_age = time.time() - (_track_crops.get(key) or {}).get("pose_ts", 0.0)
        if pose_age < min_interval:
            return
    # Dedup: replace any existing pending crop for this track
    with _pose_pending_lock:
        _pose_pending[key] = crop_bgr.copy()
    _pose_wake_event.set()
    # Lazy-start the background worker (double-checked lock)
    if not _pose_worker_started:
        with _pose_worker_start_lock:
            if not _pose_worker_started:
                _pose_worker_started = True
                t = threading.Thread(
                    target=_pose_worker_loop,
                    name="PoseWorker",
                    daemon=True,
                )
                t.start()
                logger.info("[Pose] Background pose worker started.")


def _pose_worker_loop() -> None:
    """Daemon thread: drain _pose_pending, run pose estimation, store results.
    Import is done per-crop so the thread survives import failures and recovers
    automatically once ultralytics is installed (after a server restart).
    """
    print("[PoseWorker] Thread started.", flush=True)
    while True:
        _pose_wake_event.wait(timeout=10.0)
        _pose_wake_event.clear()

        # Snapshot + clear so new crops can queue while we work
        with _pose_pending_lock:
            if not _pose_pending:
                continue
            items = list(_pose_pending.items())
            _pose_pending.clear()

        for tid, crop in items:
            try:
                from core.pose_estimator import estimate_pose_both
                jpeg, data = estimate_pose_both(crop)
                now = time.time()
                with _track_crops_lock:
                    entry = _track_crops.get(tid)
                    if entry is not None:
                        entry["pose_jpeg"] = jpeg
                        entry["pose_data"] = data
                        entry["pose_ts"]   = now
                logger.debug("[Pose] worker OK track=%s posture=%s detected=%s",
                             tid, (data or {}).get("posture", "?"),
                             (data or {}).get("detected", False))
            except Exception as e:
                logger.warning("[Pose] Worker error for %s: %s", tid, e)


def is_enabled() -> bool:
    """Check if VLM is enabled in configuration."""
    return _CONFIG.enabled


def get_config() -> VLMConfig:
    """Get current VLM configuration (read-only reference)."""
    return _CONFIG


def get_crop_jpeg(global_id: str, quality: int = 85) -> Optional[bytes]:
    """Return the latest stored crop for a track as JPEG bytes, or None."""
    if not _CONFIG.enabled:
        return None
    key = str(global_id)
    with _track_crops_lock:
        entry = _track_crops.get(key)
        if entry is None:
            return None
        crop = entry["crop"].copy()
    try:
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
    except Exception:
        return None


# ── Analysis cache ─────────────────────────────────────────────────────────────
# Dict[global_id_str → {"status": str, "text": str, "ts": float, "mode": str}]
_analysis_cache: Dict[str, dict] = {}
_analysis_lock = threading.Lock()
ANALYSIS_TTL_SECONDS = 1800  # Auto-expire old analyses (30 min — appearance doesn't change)
_CLIP_FAST_THRESHOLD = 0.22  # Return top CLIP match immediately without VLM verification

# ── Job queue ──────────────────────────────────────────────────────────────────
import queue as _queue_mod
_job_queue: "_queue_mod.Queue" = _queue_mod.Queue(maxsize=_CONFIG.max_queue_size)

# ── Metrics tracking ───────────────────────────────────────────────────────────
_metrics = {
    "jobs_submitted": 0,
    "jobs_completed": 0,
    "jobs_failed": 0,
    "jobs_dropped": 0,
    "inference_time_ms": [],
    "last_inference_at": 0.0,
}
_metrics_lock = threading.Lock()

# ── Model state ───────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_model_loaded = False

# ── CLIP (zero-shot classification, V1 only — already in requirements) ─────────
_clip_model = None
_clip_preprocess = None
_clip_loaded = False

# Default prompts per use-case
DEFAULT_QUESTION = (
    "Describe this person's appearance: clothing color and style, "
    "any items they are carrying, and what they appear to be doing."
)

PROMPTS = {
    "describe":  DEFAULT_QUESTION,
    "behavior":  "What is this person doing? Do they appear engaged with any product on the shelf?",
    "anomaly":   "Does this person's behavior appear unusual for a retail environment? Explain briefly.",
    "staff":     "Is this person likely a store employee or a customer? Describe the clues.",
    "carrying":  "What items, bags, or products is this person carrying or holding?",
    "counting":  "How many items is this person holding? List them if visible.",
    "attention": "What is this person looking at? Where is their attention focused?",
    "demographics": "Estimate this person's approximate age range and gender based on visible cues.",
    
    # Temporal analysis prompts (for video clips)
    "behavior_timeline": "Analyze this person's behavior sequence. What actions did they perform over time? Describe their movement, interactions with products/shelves, and any notable behavioral patterns.",
    "movement_pattern": "Trace this person's movement path. Where did they enter from, where did they go, and what areas or products caught their attention? Describe their trajectory and stops.",
    "suspicious_activity": "Analyze this clip for suspicious behavior. Look for: shoplifting indicators (concealing items, unusual bag behavior), loitering, vandalism, or any activity that warrants security attention. Report confidence level.",
}

CLIP_STAFF_LABELS = _CONFIG.clip_labels if _CONFIG.clip_labels else ["store employee in uniform", "customer shopping"]


def _load_model() -> bool:
    """
    Lazy-load moondream using configuration from device.json.

    Priority:
    1. Cloud API (preferred on Jetson — zero local GPU usage, no freezes)
    2. Local inference via kestrel-native (only if prefer_local=True AND cloud disabled)
    """
    global _model, _model_loaded

    if not _CONFIG.enabled:
        _model_loaded = True
        logger.info("[VLM] VLM is disabled in configuration.")
        return False

    with _model_lock:
        if _model_loaded:
            return _model is not None

        try:
            import moondream as md

            # ── Cloud API (preferred: pure requests, no SDK GIL lock) ────────────
            if _CONFIG.allow_cloud_fallback and _CONFIG.cloud_api_key:
                key_preview = _CONFIG.cloud_api_key[:8] + "..." if len(_CONFIG.cloud_api_key) > 8 else "[empty]"
                logger.info(f"[VLM] Using Moondream CLOUD API explicitly (key: {key_preview})")
                _model = "CLOUD_API"
                _model_loaded = True
                logger.info("[VLM] Moondream cloud API ready — direct HTTPS, zero Python GIL blocking.")
                return True

            # ── Local inference fallback (Jetson / kestrel-native) ────────────
            if _CONFIG.prefer_local:
                try:
                    logger.info("[VLM] Attempting local Moondream inference (kestrel-native)...")
                    _model = md.vl(local=True)
                    _model_loaded = True
                    logger.info("[VLM] Moondream loaded in LOCAL mode.")
                    return True
                except Exception as local_err:
                    logger.warning(f"[VLM] Local inference failed: {local_err}")

            logger.error(
                "[VLM] No inference method available. "
                "Set cloud_api_key and allow_cloud_fallback=true in device.json vlm section."
            )
            _model_loaded = True
            return False

        except Exception as e:
            _model_loaded = True
            logger.error(f"[VLM] Failed to load moondream: {e}. VLM features disabled.")
            return False


def warmup_model() -> bool:
    """Pre-load and warm up the model to avoid first-inference latency."""
    if not _CONFIG.enabled:
        return False
    if _model_loaded and _model is not None:
        return True
    return _load_model()


def _load_clip() -> bool:
    """Lazy-load CLIP. Respects clip_enabled config — disabled by default on Jetson."""
    global _clip_model, _clip_preprocess, _clip_loaded
    if _clip_loaded:
        return _clip_model is not None
    # Hard gate: if clip is disabled in device.json, never load it
    if not _CONFIG.clip_enabled:
        logger.info("[VLM] CLIP disabled in config — skipping load.")
        _clip_loaded = True
        return False
    try:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load(_CONFIG.clip_model, device=device)
        _clip_loaded = True
        logger.info(f"[VLM] CLIP loaded on {device}.")
        return True
    except Exception as e:
        _clip_loaded = True
        logger.error(f"[VLM] Failed to load CLIP: {e}.")
        return False


def _crop_to_pil(crop_bgr: np.ndarray):
    """Convert BGR numpy array to PIL Image, downscaling to Moondream's native size."""
    from PIL import Image
    import cv2

    # Moondream's native input resolution is 378×378 (cloud) / 448×448 (local).
    # Sending anything larger wastes bandwidth on cloud API and CPU time on resize.
    # Cap at 448px on longest side; this also prevents any accidental local GPU usage.
    h, w = crop_bgr.shape[:2]
    MAX_DIM = 448
    if w > MAX_DIM or h > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _compress_for_api(crop_bgr: np.ndarray, max_dim: int = 224, quality: int = 65) -> bytes:
    """Resize + JPEG-compress a crop for API transmission. ~10 KB vs ~50 KB at 448/Q85."""
    h, w = crop_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        crop_bgr = cv2.resize(crop_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else b""


def _crop_to_b64(crop_bgr: np.ndarray, max_dim: int = 224, quality: int = 65) -> str:
    import base64
    return base64.b64encode(_compress_for_api(crop_bgr, max_dim, quality)).decode()


def _extract_keyframes(frames: list, n: int = 3) -> list:
    """Sample n frames evenly from a clip (first, middle, last pattern)."""
    if not frames:
        return []
    if len(frames) <= n:
        return frames
    step = (len(frames) - 1) / (n - 1)
    return [frames[round(i * step)] for i in range(n)]


def _has_claude() -> bool:
    return bool(_CONFIG.anthropic_api_key)


def _run_claude_haiku(crop_bgr: np.ndarray, question: str, max_tokens: int = 200) -> str:
    """Single-image analysis via Claude Haiku. ~5x cheaper than Moondream cloud, ~2x faster."""
    try:
        import anthropic
    except ImportError:
        return "anthropic package not installed (pip install anthropic)."
    if not _CONFIG.anthropic_api_key:
        return "Anthropic API key not configured (set vlm.anthropic_api_key in device.json)."

    img_b64 = _crop_to_b64(crop_bgr, max_dim=224, quality=65)
    client = anthropic.Anthropic(api_key=_CONFIG.anthropic_api_key)
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": question},
                ]
            }]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.error(f"[VLM] Claude Haiku error: {e}")
        return f"Analysis failed: {type(e).__name__}"


def _run_claude_haiku_multi(crops_bgr: list, question: str, max_tokens: int = 300) -> str:
    """
    Multi-image analysis in ONE API call — for clip keyframes or person comparison.
    Sends up to len(crops_bgr) images + question as a single Haiku message.
    """
    try:
        import anthropic
    except ImportError:
        return "anthropic package not installed."
    if not _CONFIG.anthropic_api_key:
        return "Anthropic API key not configured."

    content = []
    for i, crop in enumerate(crops_bgr[:5]):  # hard cap at 5 images
        img_b64 = _crop_to_b64(crop, max_dim=224, quality=65)
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
        content.append({"type": "text", "text": f"[Frame {i + 1}]"})
    content.append({"type": "text", "text": question})

    client = anthropic.Anthropic(api_key=_CONFIG.anthropic_api_key)
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.error(f"[VLM] Claude Haiku multi error: {e}")
        return f"Analysis failed: {type(e).__name__}"


def _run_vlm(crop_bgr: np.ndarray, question: str, max_tokens: int = 200) -> str:
    """Provider-agnostic single-image inference. Prefers Claude Haiku, falls back to Moondream."""
    if _has_claude():
        return _run_claude_haiku(crop_bgr, question, max_tokens=max_tokens)
    return _run_moondream(crop_bgr, question)


def _run_moondream(crop_bgr: np.ndarray, question: str) -> str:
    """Run Moondream inference, avoiding SDK GIL locks for cloud API."""
    if not _load_model() or _model is None:
        return "VLM not available (see startup log for reason)."

    global _cloud_consecutive_failures, _cloud_disabled_logged

    import base64
    import io
    import requests
    start_time = time.time()
    
    try:
        pil_img = _crop_to_pil(crop_bgr)
        
        # pure requests HTTPS for the cloud API (releases Python GIL flawlessly)
        if _model == "CLOUD_API":
            # Circuit breaker: stop hammering a broken API
            if _cloud_consecutive_failures >= _CLOUD_MAX_FAILURES:
                if not _cloud_disabled_logged:
                    logger.warning(
                        f"[VLM] Cloud API disabled after {_CLOUD_MAX_FAILURES} consecutive failures. "
                        "Check your moondream_api_key in device.json. Restart to retry."
                    )
                    _cloud_disabled_logged = True
                return "VLM cloud API temporarily disabled (auth error)."

            logger.debug(f"[VLM] Sending request to CLOUD API. Question: {question}")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            payload = {
                "image_url": f"data:image/jpeg;base64,{img_b64}",
                "question": question
            }
            logger.debug(f"[VLM] API payload size: {len(img_b64)} bytes of image")
            
            res = requests.post(
                "https://api.moondream.ai/v1/query",
                json=payload,
                headers={"X-Moondream-Auth": _CONFIG.cloud_api_key},
                timeout=12.0
            )
            
            logger.debug(f"[VLM] Cloud API HTTP Status: {res.status_code}")
            if res.status_code != 200:
                _cloud_consecutive_failures += 1
                if _cloud_consecutive_failures <= 3:
                    logger.error(f"[VLM] Cloud API error ({res.status_code}): {res.text}")
                return "Analysis failed: API error."
            
            # Success — reset circuit breaker
            _cloud_consecutive_failures = 0
            _cloud_disabled_logged = False

            data = res.json()
            logger.debug(f"[VLM] Cloud API response: {data}")
            result_text = data.get("answer", str(data)).strip()
        else:
            # fallback to local kestrel-native SDK inference
            logger.debug(f"[VLM] Sending request to LOCAL MODEL. Question: {question}")
            encoded = _model.encode_image(pil_img)
            result  = _model.query(encoded, question)
            logger.debug(f"[VLM] Local model raw result: {result}")
            
            if hasattr(result, "answer"):
                ans = result.answer
                result_text = "".join(ans).strip() if hasattr(ans, "__iter__") and not isinstance(ans, str) else str(ans).strip()
            elif isinstance(result, dict):
                result_text = result.get("answer", str(result)).strip()
            else:
                result_text = str(result).strip()

        inference_ms = (time.time() - start_time) * 1000
        with _metrics_lock:
            _metrics["inference_time_ms"].append(inference_ms)
            if len(_metrics["inference_time_ms"]) > 100:
                _metrics["inference_time_ms"] = _metrics["inference_time_ms"][-100:]
            _metrics["last_inference_at"] = time.time()

        return result_text

    except Exception as e:
        logger.error(f"[VLM] moondream inference error: {e}", exc_info=True)
        return f"Analysis failed: {type(e).__name__}"


def _run_moondream_clip(representative_frame: np.ndarray, temporal_context: str) -> str:
    """Run moondream on a representative frame with temporal context."""
    return _run_moondream(representative_frame, temporal_context.strip())




def classify_staff(crop_bgr: np.ndarray) -> dict:
    """
    Zero-shot CLIP classification: employee vs customer.
    Returns {"employee": 0.82, "customer": 0.18} (probabilities).
    """
    if not _load_clip() or _clip_model is None:
        return {"error": "CLIP not available"}
    try:
        import clip
        import torch
        device = next(_clip_model.parameters()).device
        pil_img = _crop_to_pil(crop_bgr)
        image_input = _clip_preprocess(pil_img).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(CLIP_STAFF_LABELS).to(device)
        with torch.no_grad():
            logits_per_image, _ = _clip_model(image_input, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        return {
            "employee": round(float(probs[0]), 3),
            "customer": round(float(probs[1]), 3),
        }
    except Exception as e:
        logger.error(f"[VLM] CLIP classify_staff error: {e}", exc_info=True)
        return {"error": str(type(e).__name__)}


# ── Background worker ──────────────────────────────────────────────────────────

def _worker_loop():
    """Main worker loop for processing VLM jobs."""
    logger.info("[VLM] Worker thread started.")
    
    # Warm up model if configured to do so
    if _CONFIG.warmup_on_start:
        logger.info("[VLM] Warming up model on startup...")
        warmup_model()
    
    while True:
        try:
            job = _job_queue.get(timeout=5)
        except _queue_mod.Empty:
            # Cleanup expired analyses periodically
            _cleanup_expired_analyses()
            continue
        
        track_id = job["track_id"]
        mode = job.get("mode", "describe")
        
        # Ensure VLM replies in the language of the prompt (for Portuguese/Spanish support).
        # Prepend the instruction so the model sees it BEFORE the question — models are more
        # likely to follow language constraints when stated first.
        q_raw = job.get("question") or PROMPTS.get(mode, DEFAULT_QUESTION)
        q_raw = str(q_raw).strip()
        user_lang = _detect_language(q_raw)  # detect BEFORE adding the bilingual prefix
        lang_prefix = "[INSTRUÇÃO / INSTRUCTION] Responda OBRIGATORIAMENTE no mesmo idioma da pergunta abaixo. You MUST reply in the EXACT SAME LANGUAGE as the question below."
        question = f"{lang_prefix}\n{q_raw}"
            
        is_clip = job.get("is_clip", False)

        _store_result(track_id, "pending", "", mode=mode)
        
        try:
            if is_clip:
                frames = job.get("frames", [])
                if not frames:
                    _store_result(track_id, "error", "No video frames available for analysis.", mode=mode)
                    with _metrics_lock:
                        _metrics["jobs_failed"] += 1
                    continue

                keyframes = _extract_keyframes(frames, n=3)
                if _has_claude():
                    # One API call with 3 keyframes — true temporal analysis
                    text = _run_claude_haiku_multi(keyframes, q_raw, max_tokens=300)
                else:
                    # Moondream fallback: first frame only (its limitation, not ours)
                    text = _run_moondream_clip(keyframes[0], question)
                    text = _translate_to_target(text, user_lang)
                _store_result(track_id, "done", text, mode=mode)
                with _metrics_lock:
                    _metrics["jobs_completed"] += 1
                    
            else:
                # Regular single-frame analysis - use crop from cache
                with _track_crops_lock:
                    entry = _track_crops.get(str(track_id))
                    crop = entry["crop"].copy() if entry else None

                if crop is None:
                    _store_result(track_id, "error", "No crop available for this track. Try again when the person is visible.", mode=mode)
                    with _metrics_lock:
                        _metrics["jobs_failed"] += 1
                    continue

                if _has_claude():
                    # Claude handles multilingual natively — no translate round-trips
                    text = _run_claude_haiku(crop, q_raw, max_tokens=200)
                else:
                    text = _run_moondream(crop, question)
                    text = _translate_to_target(text, user_lang)
                _store_result(track_id, "done", text, mode=mode)
                with _metrics_lock:
                    _metrics["jobs_completed"] += 1
                    
        except Exception as e:
            logger.error(f"[VLM] Worker job failed for {track_id}: {e}", exc_info=True)
            _store_result(track_id, "error", "Analysis failed unexpectedly.", mode=mode)
            with _metrics_lock:
                _metrics["jobs_failed"] += 1


def _store_result(track_id: str, status: str, text: str, mode: str = "describe"):
    """Store analysis result in cache."""
    with _analysis_lock:
        _analysis_cache[str(track_id)] = {
            "status": status,
            "text": text,
            "ts": time.time(),
            "mode": mode,
        }


def _cleanup_expired_analyses():
    """Remove expired analysis results from cache."""
    now = time.time()
    with _analysis_lock:
        expired = [tid for tid, entry in _analysis_cache.items() 
                   if now - entry.get("ts", 0) > ANALYSIS_TTL_SECONDS]
        for tid in expired:
            del _analysis_cache[tid]


# Start the worker thread
_worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="vlm-worker")
_worker_thread.start()


# ── Metrics API ──────────────────────────────────────────────────────────────

def get_metrics() -> dict:
    """Get current VLM metrics for monitoring."""
    with _metrics_lock:
        metrics = dict(_metrics)
        # Calculate average inference time
        times = _metrics["inference_time_ms"]
        if times:
            metrics["avg_inference_ms"] = sum(times) / len(times)
            metrics["max_inference_ms"] = max(times)
            metrics["min_inference_ms"] = min(times)
        else:
            metrics["avg_inference_ms"] = 0.0
            metrics["max_inference_ms"] = 0.0
            metrics["min_inference_ms"] = 0.0
        return metrics


def reset_metrics():
    """Reset metrics counters (useful for testing/debugging)."""
    with _metrics_lock:
        _metrics["jobs_submitted"] = 0
        _metrics["jobs_completed"] = 0
        _metrics["jobs_failed"] = 0
        _metrics["jobs_dropped"] = 0
        _metrics["inference_time_ms"] = []


# ── Public API ─────────────────────────────────────────────────────────────────

def submit_analysis(track_id: str, question: str = "", mode: str = "describe") -> bool:
    """Submit an analysis job. Returns True if queued, False if queue full."""
    if not _CONFIG.enabled:
        return False
        
    try:
        _job_queue.put_nowait({
            "track_id": str(track_id),
            "question": question,
            "mode":     mode,
            "is_clip":  False,
        })
        _store_result(track_id, "pending", "", mode=mode)
        with _metrics_lock:
            _metrics["jobs_submitted"] += 1
        return True
    except _queue_mod.Full:
        logger.warning(f"[VLM] Job queue full — dropping request for {track_id}")
        with _metrics_lock:
            _metrics["jobs_dropped"] += 1
        return False


def submit_clip_analysis(track_id: str, camera_id: str, frames: list, question: str = "", mode: str = "behavior_timeline") -> bool:
    """
    Submit a video clip analysis job for temporal behavior analysis.
    
    Args:
        track_id: The track ID to analyze
        camera_id: Camera ID for the track
        frames: List of BGR frames from video buffer
        question: Custom question (or use default based on mode)
        mode: Analysis mode (behavior_timeline, movement_pattern, suspicious_activity)
    
    Returns:
        True if queued, False if queue full
    """
    if not _CONFIG.enabled:
        return False
    
    if not frames:
        logger.warning(f"[VLM] No frames provided for clip analysis of {track_id}")
        return False
        
    try:
        _job_queue.put_nowait({
            "track_id": str(track_id),
            "camera_id": camera_id,
            "frames": frames,
            "question": question,
            "mode": mode,
            "is_clip": True,
        })
        _store_result(track_id, "pending", "", mode=mode)
        with _metrics_lock:
            _metrics["jobs_submitted"] += 1
        return True
    except _queue_mod.Full:
        logger.warning(f"[VLM] Job queue full — dropping clip request for {track_id}")
        with _metrics_lock:
            _metrics["jobs_dropped"] += 1
        return False


def get_result(track_id: str) -> dict:
    """
    Returns {"status": "pending"|"done"|"error"|"not_found", "text": str, "ts": float, "mode": str}
    """
    with _analysis_lock:
        entry = _analysis_cache.get(str(track_id))
    if entry is None:
        return {"status": "not_found", "text": "", "ts": 0.0, "mode": ""}
    return dict(entry)


def has_crop(track_id: str) -> bool:
    with _track_crops_lock:
        return str(track_id) in _track_crops




def list_tracked_ids() -> list:
    """Return list of global_ids that currently have a saved crop."""
    with _track_crops_lock:
        return list(_track_crops.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND SEARCH WORKER
# ═══════════════════════════════════════════════════════════════════════════════
# VLM searches are submitted as jobs and processed by a single daemon thread.
# This keeps Flask request threads free (streams never freeze) while still
# serializing GPU inference safely through _model_lock.

import uuid as _uuid

# Job store: job_id → {status, found, camera_id, global_id, result, ts}
_search_jobs: Dict[str, dict] = {}
_search_jobs_lock = threading.Lock()
_SEARCH_JOB_TTL = 300  # seconds — auto-expire old jobs

# Single unbounded queue (searches are rare, no need to drop them)
_search_queue: "_queue_mod.Queue" = _queue_mod.Queue()


def _expire_old_jobs() -> None:
    """Remove jobs older than TTL. Call inside _search_jobs_lock."""
    cutoff = time.time() - _SEARCH_JOB_TTL
    expired = [jid for jid, j in _search_jobs.items() if j.get("ts", 0) < cutoff]
    for jid in expired:
        del _search_jobs[jid]


def _translate_to_english(text: str) -> str:
    """Fast, free translation using Google Translate API (gtx client) for CLIP compatibility."""
    import urllib.parse
    import requests
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text)}"
        r = requests.get(url, timeout=2.0)
        return r.json()[0][0][0]
    except Exception as e:
        logger.warning(f"[VLM] Auto-translation failed: {e}. Passing raw query to CLIP.")
        return text


def _detect_language(text: str) -> str:
    """
    Detect the language of `text` using the Google Translate gtx API.
    Returns an ISO 639-1 code (e.g. 'pt', 'es', 'en') or 'en' on failure.
    """
    import urllib.parse
    import requests
    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text[:200])}"
        )
        r = requests.get(url, timeout=2.0)
        data = r.json()
        # The detected source language is at index [2] of the response
        detected = data[2] if len(data) > 2 and isinstance(data[2], str) else "en"
        return detected
    except Exception as e:
        logger.warning(f"[VLM] Language detection failed: {e}. Assuming 'en'.")
        return "en"


def _translate_to_target(text: str, target_lang: str) -> str:
    """
    Translate `text` into `target_lang` (ISO 639-1 code).
    Returns the original text unchanged if translation fails or target is English.
    """
    if not text or target_lang == "en":
        return text
    import urllib.parse
    import requests
    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=en&tl={target_lang}&dt=t&q={urllib.parse.quote(text)}"
        )
        r = requests.get(url, timeout=4.0)
        parts = r.json()[0]
        translated = "".join(seg[0] for seg in parts if seg[0])
        logger.debug(f"[VLM] Translated answer to '{target_lang}': {translated[:80]}")
        return translated
    except Exception as e:
        logger.warning(f"[VLM] Output translation to '{target_lang}' failed: {e}. Returning original.")
        return text


def _sort_candidates_by_clip(candidates: list, query: str) -> tuple:
    """
    Sort candidates by CLIP text-image similarity.
    Returns (sorted_list, top_score) so caller can fast-path on confident matches.
    """
    if not candidates or not _load_clip() or _clip_model is None:
        return candidates[::-1], 0.0

    try:
        import clip
        import torch
        device = next(_clip_model.parameters()).device
        text_tokens = clip.tokenize([query], truncate=True).to(device)

        with torch.no_grad():
            text_features = _clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Batch all image embeddings in one pass for speed
            pil_imgs = [_crop_to_pil(crop) for _, crop, _ in candidates]
            image_tensors = torch.stack([_clip_preprocess(p) for p in pil_imgs]).to(device)
            image_features = _clip_model.encode_image(image_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sims = (text_features @ image_features.T).squeeze(0).cpu().tolist()

        scored = sorted(zip(sims, candidates), key=lambda x: x[0], reverse=True)
        top_score = scored[0][0] if scored else 0.0
        logger.info(f"[VLM Search] CLIP scores (top-5): {[(round(s,3), c[2]+'_'+str(c[0])) for s,c in scored[:5]]}")
        return [c for _, c in scored], top_score
    except Exception as e:
        logger.error(f"[VLM Search] CLIP sorting failed: {e}")
        return candidates[::-1], 0.0


def _search_worker() -> None:
    """
    Daemon thread: pops search jobs and uses a Two-Stage Hybrid pipeline:
    1. CLIP (Fast): Score and sort all candidates by similarity to the query
    2. Moondream (Heavy): Verify only the top 1-2 candidates to prevent stream freezes
    """
    while True:
        try:
            job = _search_queue.get(timeout=1.0)
        except _queue_mod.Empty:
            continue

        job_id = job["job_id"]
        query = job["query"]
        
        # ── SCENE QUERY (Single frame, Non-blocking) ──
        if job.get("is_scene"):
            logger.info(f"[VLM Scene Query] Processing job {job_id}")
            camera_id = job["camera_id"]
            
            # Force behavior-centric descriptions for scene queries.
            # Prepend language instruction so the model replies in the user's language.
            lang_prefix = "[INSTRUÇÃO / INSTRUCTION] Responda OBRIGATORIAMENTE no mesmo idioma da pergunta abaixo. You MUST reply in the EXACT SAME LANGUAGE as the question below."
            scene_prompt = (
                f"{lang_prefix}\n"
                f"{query} "
                "IMPORTANT: Focus STRICTLY on people, their behavior, actions, and interactions. "
                "Do NOT describe the room, static scenery, or background elements."
            )
            scene_lang = _detect_language(query)
            try:
                if _has_claude():
                    # Claude: no translation needed, compress scene frame to 480px
                    res = _run_claude_haiku(job["frame_bgr"], scene_prompt, max_tokens=250)
                else:
                    res = _run_moondream(job["frame_bgr"], scene_prompt)
                    res = _translate_to_target(res, scene_lang)
                with _search_jobs_lock:
                    _search_jobs[job_id].update({
                        "status": "done",
                        "found": True,
                        "result": res,
                        "ts": time.time()
                    })
            except Exception as e:
                logger.error(f"[VLM Scene Query] Error: {e}", exc_info=True)
                with _search_jobs_lock:
                    _search_jobs[job_id].update({
                        "status": "error",
                        "found": False,
                        "result": "Análise falhou." if scene_lang == "pt" else "Analysis failed.",
                        "ts": time.time()
                    })
            finally:
                _search_queue.task_done()
                with _search_jobs_lock:
                    _expire_old_jobs()
            continue

        candidates = job["candidates"]  # list of (gid, crop_bgr, camera_id)

        english_query = _translate_to_english(query)
        logger.info(f"[VLM Search] Job {job_id}: {len(candidates)} candidates, query='{english_query}'")

        # ── STAGE 1: CLIP — rank all candidates in one batched forward pass ──
        sorted_candidates, top_clip_score = _sort_candidates_by_clip(candidates, english_query)

        found = False

        # Fast path: CLIP is confident enough — no VLM call needed
        # Guard: don't short-circuit when pool is tiny (scores are unreliable on small sets)
        if top_clip_score >= _CLIP_FAST_THRESHOLD and len(sorted_candidates) >= 5 and sorted_candidates:
            best_gid, best_crop, best_cam = sorted_candidates[0]
            logger.info(f"[VLM Search] CLIP fast-path: {best_cam}_{best_gid} score={top_clip_score:.3f}")
            with _search_jobs_lock:
                _search_jobs[job_id].update({
                    "status": "done", "found": True,
                    "camera_id": best_cam, "global_id": str(best_gid),
                    "result": f"Best match (CLIP score {top_clip_score:.2f}): camera {best_cam}",
                    "ts": time.time(),
                })
            found = True

        # ── STAGE 2: VLM verification — top 3 only ──
        if not found:
            top3 = sorted_candidates[:3]
            import re

            if _has_claude() and top3:
                # ONE batched Claude Haiku call for all top candidates
                crops = [c[1] for c in top3]
                batch_q = (
                    f'I am looking for a person matching: "{english_query}". '
                    f'I show you {len(crops)} images labeled Frame 1 to Frame {len(crops)}. '
                    'Which frame number best matches? Reply with just the number (e.g. "2") '
                    'or "none" if no match. Then one sentence of reasoning.'
                )
                try:
                    res = _run_claude_haiku_multi(crops, batch_q, max_tokens=80)
                    logger.info(f"[VLM Search] Claude batch result: {res}")
                    # Parse "1", "2", "3" or "none"
                    match = re.search(r'\b([123])\b', res)
                    if match:
                        idx = int(match.group(1)) - 1
                        if 0 <= idx < len(top3):
                            gid, _, cam = top3[idx]
                            with _search_jobs_lock:
                                _search_jobs[job_id].update({
                                    "status": "done", "found": True,
                                    "camera_id": cam, "global_id": str(gid),
                                    "result": res, "ts": time.time(),
                                })
                            found = True
                except Exception as e:
                    logger.warning(f"[VLM Search] Claude batch error: {e}")

            if not found:
                # Moondream fallback: sequential, top 3 only (was 15)
                search_prompt = (
                    f'Does the person in this image match: "{english_query}"? '
                    "Answer YES or NO, then one sentence."
                )
                for gid, crop, cam in top3:
                    try:
                        res = _run_moondream(crop, search_prompt)
                        logger.info(f"[VLM Search] Moondream {cam}_{gid}: {res}")
                        if re.search(r'\byes\b', res.lower()) and not res.lower().startswith("no"):
                            with _search_jobs_lock:
                                _search_jobs[job_id].update({
                                    "status": "done", "found": True,
                                    "camera_id": cam, "global_id": str(gid),
                                    "result": res, "ts": time.time(),
                                })
                            found = True
                            break
                    except Exception as e:
                        logger.warning(f"[VLM Search] Moondream error {cam}_{gid}: {e}")

        if not found:
            with _search_jobs_lock:
                if job_id in _search_jobs:
                    _search_jobs[job_id].update({
                        "status": "done", "found": False,
                        "result": "Target not found in any visible camera streams.",
                        "ts": time.time(),
                    })

        # Housekeeping
        with _search_jobs_lock:
            _expire_old_jobs()

        logger.info(f"[VLM Search] Job {job_id} done — found={found}")
        _search_queue.task_done()


# Start the worker daemon on import
_search_worker_thread = threading.Thread(target=_search_worker, daemon=True, name="VLM-SearchWorker")
_search_worker_thread.start()


def submit_scene_query(camera_id: str, frame_bgr: "np.ndarray", query: str) -> str:
    """Submit a single-frame scene query to the background search worker."""
    job_id = _uuid.uuid4().hex[:8]
    with _search_jobs_lock:
        _expire_old_jobs()
        _search_jobs[job_id] = {
            "status": "pending",
            "found": False,
            "camera_id": camera_id,
            "global_id": None,
            "result": None,
            "ts": time.time(),
            "query": query
        }

    _search_queue.put({
        "job_id": job_id,
        "query": query,
        "is_scene": True,
        "camera_id": camera_id,
        "frame_bgr": frame_bgr
    })
    return job_id


def submit_search(query: str, candidates: list) -> str:
    """
    Enqueue a background search job.

    Args:
        query: Natural language description to search for.
        candidates: list of (global_id, crop_bgr_numpy, camera_id)

    Returns:
        job_id string — poll get_search_result(job_id) for the result.
    """
    job_id = str(_uuid.uuid4())
    with _search_jobs_lock:
        _search_jobs[job_id] = {"status": "pending", "found": False, "result": "", "ts": time.time()}
        _expire_old_jobs()
    _search_queue.put({"job_id": job_id, "query": query, "candidates": candidates})
    logger.info(f"[VLM Search] Queued job {job_id} ({len(candidates)} candidates)")
    return job_id


def get_search_result(job_id: str) -> dict:
    """
    Returns the result of a submitted search job.
    Possible statuses: 'pending', 'done', 'not_found'
    """
    with _search_jobs_lock:
        job = _search_jobs.get(job_id)
    if job is None:
        return {"status": "not_found", "found": False, "result": ""}
    return dict(job)

