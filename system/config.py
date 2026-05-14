# config.py
import json
import os
import sys

APP_VERSION = "0.1.3"

# ── Device identity — loaded from device.json (never hardcoded) ───────────────
# Copy device.json.example → device.json and fill in your real values.
# device.json is git-ignored to prevent client identity from leaking.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEVICE_JSON_PATH = os.path.join(BASE_DIR, "device.json")

if not os.path.exists(_DEVICE_JSON_PATH):
    print(
        "\n[FATAL] device.json not found.\n"
        "  Copy device.json.example → device.json and fill in your values.\n"
        "  Cannot start without client/store identity — data integrity risk.\n"
    )
    sys.exit(1)

with open(_DEVICE_JSON_PATH) as _f:
    _d = json.load(_f)

# Auto-hash the admin PIN if it's cleartext; detect and replace default/empty PIN
_dirty = False
_DEFAULT_PIN = "1234"
if "admin_pin" in _d:
    _pin = str(_d["admin_pin"])
    _is_cleartext = not _pin.startswith(("scrypt:", "pbkdf2:", "argon2:", "$argon2", "sha256:"))

    # Replace default or empty cleartext PIN with a secure random one
    if _is_cleartext and (_pin == _DEFAULT_PIN or _pin == ""):
        import secrets as _secrets
        _new_pin = str(_secrets.randbelow(900000) + 100000)  # 6-digit random
        print(
            f"[WARNING] Default or empty admin PIN detected. "
            f"Generated new PIN: {_new_pin} — please note this down. "
            f"Saving hashed PIN to device.json."
        )
        _pin = _new_pin
        _is_cleartext = True  # ensure it gets hashed below

    if _is_cleartext:
        try:
            from werkzeug.security import generate_password_hash
            _d["admin_pin"] = generate_password_hash(_pin)
            _dirty = True
        except ImportError:
            import hashlib as _hashlib
            _d["admin_pin"] = "sha256:" + _hashlib.sha256(_pin.encode()).hexdigest()
            _dirty = True

if _dirty:
    try:
        with open(_DEVICE_JSON_PATH, "w") as _f:
            json.dump(_d, _f, indent=4)
        print("[INFO] device.json admin_pin was plain-text and has been securely hashed.")
    except Exception as e:
        print(f"[WARNING] Could not securely hash admin_pin in device.json: {e}")

# Required identity fields
CLIENT_ID = _d.get("client_id", "unknown")
STORE_ID  = _d.get("store_id", "unknown")
DEVICE_ID = _d.get("device_id", "unknown")

# Cloud API connectivity
API_URL    = _d.get("api_url", "")        # e.g. https://your-api.run.app
DEVICE_KEY = _d.get("device_key", "")     # shared secret for heartbeat auth
GCS_BUCKET = _d.get("gcs_bucket", "nort-data-landing")

# OTA update settings
OTA_INSTALL_DIR = BASE_DIR

# Local admin page
# L7-fix: empty default forces operator to set a real PIN on first boot.
# local_admin.py generates a random 6-digit PIN when ADMIN_PIN is empty,
# logs it once to the console, and persists it to device.json.
ADMIN_PIN  = _d.get("admin_pin", "")
ADMIN_PORT = _d.get("admin_port", 8080)
# ENABLE_REID is now unified as REID_ENABLED below (line ~89)

# ── General Settings ──────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# H2-fix: lazy torch import so the system boots even when torch is not yet
# installed (common on a fresh Jetson before the pip wheel is deployed).
def _get_torch_device():
    try:
        import torch as _torch
        return _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')
    except ImportError:
        return 'cpu'

DEVICE = _get_torch_device()

# ── YOLO & RE-ID SETTINGS ──────────────────────────────────────────────────
# All model paths are absolute so the process works from any working directory.
_BASE_DIR = BASE_DIR  # alias for clarity (BASE_DIR already set above)
YOLO_MODEL_PATH     = os.path.join(_BASE_DIR, "assets", "models", "yolox_m.onnx")
YOLO_IMGSZ          = 640
YOLO_CONF_THRESHOLD = 0.15  # Low baseline so ByteTrack can keep occluded tracks alive
TRACK_CLASS_ID     = 0

# ── Quality Control Parameters ────────────────────────────────────────────────
MIN_LAPLACIAN_SHARPNESS = 10.0
MIN_DETECTION_AREA      = 50 * 100
MIN_ASPECT_RATIO        = 1.2
MAX_ASPECT_RATIO        = 5.0

# ── Tracker Settings ─────────────────────────────────────────────
TRACKER_CONFIG_PATH   = os.path.join(_BASE_DIR, "tracker_configs", "custom_bytetrack.yaml")
ATTRIBUTE_MODEL_PATH  = os.path.join(_BASE_DIR, "assets", "models", "net_last.pth")

# EMA smoothing for zone assignment — prevents zone flickering near boundaries.
# 0.35 is a good default at 30 FPS. Raise to 0.5 if tracking feels laggy.
TRACK_EMA_ALPHA = 0.35

# ── Live pose estimation (admin panel) ──────────────────────────
# Off by default. Enable with:  python run.py --live-pose
# Runs YOLOv8n-pose continuously on the selected person at ~6-7 fps.
# Adds ~10-15 % extra GPU load while a person is selected in the admin panel.
LIVE_POSE_ENABLED = False

# ── Re-ID Manager Settings ──────────────────────────────────────
# Set "enable_reid": false in device.json to disable Re-ID and run as a plain multi-camera tracker.
REID_ENABLED             = _d.get("enable_reid", True)

# 0.42: tuned for CPU-inferred OSNet on Jetson where same-person embeddings from
# different angles score ~0.38–0.65.  The +0.08 context bonus for recently-seen IDs
# and the clothing-color / gender veto keep false positives under control.
REID_SIMILARITY_THRESHOLD = 0.42

# How long a gallery entry stays alive without being seen (seconds).
# 300s = 5 minutes. A person who leaves and returns within this window keeps their ID.
REID_EXPIRY_SECONDS       = 300.0

# Max appearance embeddings stored per person (max-pooled for robustness against posture changes).
# Increased to 45 so we can store 15 seconds of history at 3 updates/second.
REID_MAX_EMBEDDINGS       = 45

# Margin from edge of frame to extract the *first* global ID embedding (to ensure full visibility).
# 0.05 = 5% margin on all 4 sides. 
REID_EDGE_MARGIN_PCT      = 0.05

# Minimum bounding box dimension (px) for a crop to be sent to the Re-ID model.
# Dropped to 32 so partial waist-up bodies and distant people are still embedded.
REID_MIN_CROP_SIZE        = 32

# Re-ID model device. Leave empty to auto-detect (CUDA if available, else CPU).
# Set to "cpu" to force CPU inference on Jetson NX if VRAM is tight.
REID_DEVICE               = ""   # "" = auto

# How often (in frames) to refresh an existing track's embedding in the gallery.
# 45 frames @ 15 fps ≈ every 3 s. Keeps the gallery fresh without flooding the
# GPU — was 30 which hammered the OSNet session every ~1 s per active track.
REID_UPDATE_INTERVAL_FRAMES = _d.get("reid_update_interval_frames", 45)

# ── Inference Throttle ────────────────────────────────────────────────────────
# Run YOLOX detection every N frames; ByteTrack coasts via Kalman between them.
# 1  = run every frame (original behaviour, highest accuracy)
# 2  = detect every other frame (~2× throughput, undetectable for retail analytics)
# 3  = good for 4+ cameras on Orin Nano; ByteTrack handles occlusion gaps well
# Default changed to 2 — halves YOLOX load with no visible tracking regression.
# Override per deployment: device.json "detect_every_n_frames": 1|2|3
DETECT_EVERY_N_FRAMES = _d.get("detect_every_n_frames", 2)

# ── Data Handling ─────────────────────────────────────────────────────────────
CSV_FILENAME = "tracking_log.csv"

# YOLO26 Performance Optimizations
# NOTE: 1280 was a dev/RTX setting. On Jetson Orin Nano, 640 is the
# sweet spot — TRT FP16 at 640 gives 2-3x the FPS vs 1280 with no
# meaningful accuracy loss at typical retail camera distances.
# Use 416 for 4+ camera configs or Orin Nano 8 GB constrained deployments.
YOLO_IMGSZ = 640   # Jetson-tuned: 640 matches the ONNX model's native input
YOLO_HALF = True   # FP16 via TRT (handled by TensorrtExecutionProvider)

# Additional GPU optimizations
YOLO_MAX_DET = 300            # Allow tracking up to 300 people per frame
YOLO_IOU_THRESHOLD = 0.50     # Slightly relaxed NMS to preserve nearby partially-overlapping people

# CSV_FIELDNAMES = ["client_id", "store_id", "camera_id", "timestamp", "track_id", "x1", "y1", "x2", "y2", "zone", "gender", 
#                   "store_occupancy", "has_entered", "first_zone_after_entry", "crossing_status"]

CSV_FIELDNAMES = ["client_id", "store_id", "camera_id", "timestamp", "track_id", "global_id",
                  "x1", "y1", "x2", "y2", "zone", "gender", "age_category",
                  "store_occupancy", "has_entered", "first_zone_after_entry", "crossing_status"]

# ── Camera configuration (loaded from cameras.json) ──────────────────────────
_CAMERAS_JSON_PATH = os.path.join(BASE_DIR, "cameras.json")

_FALLBACK_VIDEO_SOURCES = {
    "camera_1": "videos_1080p/CAM00.mp4",
    "camera_2": "videos_1080p/CAM01.mp4",
    "camera_3": "videos_1080p/CAM02.mp4",
    "camera_4": "videos_1080p/CAM03.mp4",
    "camera_5": "videos_1080p/CAM04.mp4",
    "camera_6": "videos_1080p/CAM05.mp4",
}


def load_cameras() -> dict:
    """Load camera config from cameras.json. Returns full camera objects."""
    if os.path.exists(_CAMERAS_JSON_PATH):
        try:
            with open(_CAMERAS_JSON_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback: create cameras.json from hardcoded defaults
    cameras = {}
    for cam_id, src in _FALLBACK_VIDEO_SOURCES.items():
        cameras[cam_id] = {
            "name": cam_id.replace("_", " ").title(),
            "source": src,
            "type": "file",
            "enabled": True,
        }
    save_cameras(cameras)
    return cameras


def save_cameras(cameras: dict) -> None:
    """Persist camera config to cameras.json."""
    with open(_CAMERAS_JSON_PATH, "w") as f:
        json.dump(cameras, f, indent=2)


def _build_video_sources() -> dict:
    """Build the VIDEO_SOURCES dict from cameras.json (enabled cameras only)."""
    cameras = load_cameras()
    return {
        cam_id: cam["source"]
        for cam_id, cam in cameras.items()
        if cam.get("enabled", True)
    }


VIDEO_SOURCES = _build_video_sources()
GRID_WINDOW_TITLE = "Multi-Camera Tracking"
ENABLE_PATH_DRAWING = True
MAX_PATH_LENGTH = 30
UNKNOWN_ZONE = "Unknown"
POLYGON_POINTS_FILE = "zones_per_camera.json"
DRAW_ZONES = True

# --- Shoplifting Detection Settings ---
SHOPLIFTING_MODEL_PATH = os.path.join(_BASE_DIR, "models", "shoplifting_wights.pt")
ENABLE_SHOPLIFTING_DETECTION = False
SHOPLIFTING_DETECTION_INTERVAL = 5  # Run shoplifting detection every N frames

DRAW_ENTRANCE_DEBUG = os.getenv("DRAW_ENTRANCE_DEBUG", "false").lower() == "true"
ENTRANCE_DETECTION_SENSITIVITY = 1e-6  # For parallel line detection

# --- Age Detection Settings ---
AGE_DETECTION_ENABLED = True
AGE_CATEGORIES = ["child", "teenager", "adult", "senior"]  # Available age categories
DEFAULT_AGE_CATEGORY = "adult"  # Default when age cannot be determined