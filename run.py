import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Silence C++ library noise BEFORE any imports that trigger it ──────────────
# MediaPipe / absl / glog — suppress all C++ INFO/WARNING/ERROR output
os.environ.setdefault("GLOG_minloglevel",      "3")   # 0=INFO 1=WARN 2=ERROR 3=FATAL
os.environ.setdefault("GLOG_logtostderr",      "0")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU_INFERENCE", "0")
# TensorFlow Lite C++ runtime
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")   # 0=ALL 1=INFO 2=WARN 3=ERROR
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
# ONNX Runtime / cpuinfo noise ("Error in cpuinfo: prctl(PR_SVE_GET_VL) failed")
os.environ.setdefault("ORT_LOGGING_LEVEL",     "3")   # 0=VERBOSE … 3=WARNING 4=ERROR

import sys
import warnings

# ── Python-level warning filters ─────────────────────────────────────────────
# Silence spammy PyTorch/Vision/ReID deprecation and environment warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchreid|torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch|torchreid")
# ByteTrack deprecation notice from supervision (v0.28+ removes it in v0.30)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*ByteTrack.*deprecated.*")
# Google api_core Python version deprecation (shows on every run on 3.10)
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

import threading
import cv2
import numpy as np
import math
import time
from core.yolox_detector import YOLOXDetector

from system import config
from admin import local_admin
from system import ota_agent
from data import uploader
from data.sync_manager import SyncManager
from system.logger_setup import setup_logger
from data.data_handler import DataHandler
from core.camera_processor import CameraProcessor
from core.polygon_zone import find_zone
from core.reid_manager import ReIDManager
from reid_gallery_window import ReidGalleryWindow
from data import spatial_logger

logger = setup_logger(__name__)

# ── Heartbeat sender (Gap 4A) ─────────────────────────────────────────────────

def _heartbeat_loop(processors, lock, last_upload_ref, stop_event, interval=300):
    """
    Posts a heartbeat to the Nort-API every `interval` seconds.
    Runs as a daemon thread — never crashes the main tracking loop.
    """
    if not config.API_URL or not config.DEVICE_KEY:
        logger.warning(
            "Heartbeat disabled: API_URL or DEVICE_KEY not set in device.json"
        )
        return

    import requests, shutil

    while not stop_event.is_set():
        try:
            with lock:
                active_procs = [p for p in processors if p.fps > 0]
                cameras_active = len(active_procs)
                fps_avg = (
                    sum(p.fps for p in active_procs) / cameras_active
                    if cameras_active else 0.0
                )

            disk = shutil.disk_usage("/")
            disk_pct = round(disk.used / disk.total * 100, 1)

            last_upload_at = last_upload_ref.get("ts")  # datetime or None

            payload = {
                "client_id":      config.CLIENT_ID,
                "store_id":       config.STORE_ID,
                "device_id":      config.DEVICE_ID,
                "cameras_active": cameras_active,
                "fps_avg":        round(fps_avg, 1),
                "disk_usage_pct": disk_pct,
                "uptime_seconds": int(time.monotonic()),
                "last_upload_at": (
                    last_upload_at.isoformat() if last_upload_at else None
                ),
                "current_version": getattr(config, "APP_VERSION", "0.0.0"),
            }

            try:
                import subprocess
                telemetry = subprocess.check_output(["tegrastats", "--interval", "1", "--count", "1"], timeout=2).decode().strip()
                payload["recent_logs"] = telemetry
            except Exception:
                pass

            resp = requests.post(
                f"{config.API_URL}/api/v1/devices/heartbeat",
                json=payload,
                headers={"Authorization": f"Bearer {config.DEVICE_KEY}"},
                timeout=5,
                verify=True,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                # Support both multi-command list and single-command (backwards compat)
                commands = data.get("commands", [])
                if not commands and data.get("command"):
                    commands = [data["command"]]
                for cmd in commands:
                    ota_agent.handle_remote_command(cmd)
            logger.debug("Heartbeat sent")

        except Exception as exc:
            # Never let heartbeat crash the tracking loop
            logger.debug(f"Heartbeat error (non-fatal): {exc}")

        stop_event.wait(interval)


def _sync_loop(sm, stop_event, interval=60):
    """
    Periodically processes the upload queue.
    """
    while not stop_event.is_set():
        try:
            sm.process_queue()
        except Exception as e:
            logger.error(f"Sync loop error: {e}")
        stop_event.wait(interval)


def _systemd_watchdog_loop(stop_event, interval=10):
    """
    L5-fix: Ping the systemd watchdog every `interval` seconds so the service
    manager knows the process is still alive and responsive.
    Requires WatchdogSec= in the .service unit (set to 30 s).
    No-ops silently on non-Linux or when WATCHDOG_USEC env var is not set.
    """
    try:
        import ctypes
        import ctypes.util
        _libsystemd_path = ctypes.util.find_library("systemd")
        if not _libsystemd_path:
            return  # systemd not available (dev machine, Docker without systemd)
        _libsystemd = ctypes.CDLL(_libsystemd_path)
        _sd_notify = _libsystemd.sd_notify
        _sd_notify.argtypes = [ctypes.c_int, ctypes.c_char_p]
        _sd_notify.restype = ctypes.c_int
        # Also signal READY=1 once on startup so Type=simple becomes Type=notify-compatible
        _sd_notify(0, b"READY=1")
        logger.info("L5: systemd watchdog active (pinging every %ds)", interval)
        while not stop_event.is_set():
            _sd_notify(0, b"WATCHDOG=1")
            stop_event.wait(interval)
    except Exception:
        pass  # Never crash the main process over a watchdog failure


def _log_rotation_loop(stop_event, sync_manager=None, interval=300):
    """
    Periodically rotates local logs and queues them for the shared SyncManager.
    Passing sync_manager avoids creating a new SyncManager (and its 5-second
    GCS init timeout) on every rotation tick.
    """
    while not stop_event.is_set():
        try:
            uploader.queue_for_upload(sync_manager=sync_manager)
        except Exception as e:
            logger.error(f"Log rotation error: {e}")
        stop_event.wait(interval)


# ── Local admin helper (Gap 4B) ───────────────────────────────────────────────

def _make_camera_status(processors):
    """Build the camera dict that local_admin.py expects."""
    return {
        p.camera_id: {"active": p.fps > 0, "fps": f"{p.fps:.1f}"}
        for p in processors
    }

_group_history = {}  # {frozenset([tid1, tid2]): first_seen_timestamp}
_last_group_prune = 0.0          # epoch seconds of last _group_history prune

def _make_retail_data(processors):
    """Extract real-time retail intelligence (occupancy and live tracks)."""
    import time
    global _last_group_prune
    tracks = {}
    now = time.time()
    for p in processors:
        # Snapshot track_attributes under the camera lock to avoid a race
        # where the camera thread mutates the dict while we iterate it.
        with p.lock:
            attrs_snapshot = dict(p.track_attributes)

        for tid, attrs in attrs_snapshot.items():
            if now - attrs.get('last_seen', 0) < 5:  # Only count people seen in last 5 secs
                gid = attrs.get('global_id')
                if gid is not None:
                    gid = str(gid)
                else:
                    gid = f"{p.camera_id}_{tid}"
                
                zone = "Unknown"
                smoothed_pos = attrs.get('smoothed_position')
                if smoothed_pos is not None:
                    z = find_zone(smoothed_pos, p.zones)
                    if z: zone = z
                
                det_count = attrs.get('detection_count', 0)
                if gid not in tracks or det_count > tracks[gid].get('detection_count', 0):
                    tracks[gid] = {
                        'zone': zone,
                        'gender': attrs.get('gender', 'Unknown'),
                        'age': attrs.get('age_category', 'adult'),
                        'detection_count': det_count,
                        'world_x': attrs.get('world_x'),
                        'world_y': attrs.get('world_y'),
                        # Bounding box + camera context for VLM click detection
                        # Cast to plain Python int — numpy int64 is not JSON-serializable.
                        'camera_id': p.camera_id,
                        'x1': int(attrs['x1']) if attrs.get('x1') is not None else None,
                        'y1': int(attrs['y1']) if attrs.get('y1') is not None else None,
                        'x2': int(attrs['x2']) if attrs.get('x2') is not None else None,
                        'y2': int(attrs['y2']) if attrs.get('y2') is not None else None,
                        'frame_w': int(attrs['frame_w']) if attrs.get('frame_w') is not None else None,
                        'frame_h': int(attrs['frame_h']) if attrs.get('frame_h') is not None else None,
                    }
    # Simple spatial-temporal Group Detection MVP
    groups = []
    track_ids = list(tracks.keys())
    current_pairs = set()
    
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            t1 = tracks[track_ids[i]]
            t2 = tracks[track_ids[j]]
            
            if t1.get('world_x') is not None and t2.get('world_x') is not None:
                dx = t1['world_x'] - t2['world_x']
                dy = t1['world_y'] - t2['world_y']
                dist = math.hypot(dx, dy)
                
                # Assume 100 units distance threshold
                if dist < 100.0:
                    current_pairs.add(frozenset([track_ids[i], track_ids[j]]))
                    
    for p in current_pairs:
        if p not in _group_history:
            _group_history[p] = now
            
    stale_pairs = [p for p in list(_group_history.keys()) if p not in current_pairs]
    for p in stale_pairs:
        del _group_history[p]

    # Cap history to prevent unbounded growth on very busy scenes
    if len(_group_history) > 500:
        # Keep only the 400 oldest (most mature) entries
        sorted_pairs = sorted(_group_history.items(), key=lambda kv: kv[1])
        _group_history.clear()
        _group_history.update(dict(sorted_pairs[:400]))
    _last_group_prune = now
        
    # Pairs that have been close for more than 5 seconds are considered a group
    mature_pairs = [p for p, first_seen in _group_history.items() if (now - first_seen) > 5.0]
    
    adj = {tid: set() for tid in track_ids}
    for p in mature_pairs:
        l = list(p)
        adj[l[0]].add(l[1])
        adj[l[1]].add(l[0])
        
    visited = set()
    group_idx = 1
    for tid in track_ids:
        if tid not in visited and adj[tid]:
            q = [tid]
            comp = set()
            while q:
                curr = q.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    q.extend(list(adj[curr] - visited))
                    
            if len(comp) > 1:
                group_id = f"Group-{group_idx}"
                groups.append({"id": group_id, "members": list(comp)})
                for member in comp:
                    tracks[member]['group_id'] = group_id
                group_idx += 1

    # Occupancy = unique global_ids active in the last 5 s (same window as tracks).
    # This is self-correcting and immune to ReID fragmentation: each person's latest
    # global_id overwrites older ones in the tracks dict (line above), so one physical
    # person always counts as exactly 1 regardless of how many IDs ReID has minted.
    occupancy = len(tracks)
    return {"occupancy": occupancy, "tracks": tracks, "groups": groups}

def get_screen_resolution():
    """Gets the primary monitor's resolution with multiple fallback methods."""
    try:
        import screeninfo
        screen = screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except ImportError:
        logger.warning("screeninfo module not found. Using default resolution.")
    except Exception as e:
        logger.warning(f"Could not detect screen resolution: {e}")
    
    try:
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except ImportError:
        logger.warning("tkinter module not found. Using hardcoded resolution.")
    except Exception as e:
        logger.warning(f"Could not get screen resolution via tkinter: {e}")
    
    return 1920, 1080

def create_grid_view(frames, screen_width, screen_height):
    """
    Creates a dynamic grid and draws the camera name on each cell.
    """
    num_frames = len(config.VIDEO_SOURCES)
    if num_frames == 0:
        return np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    num_cols = math.ceil(math.sqrt(num_frames))
    num_rows = math.ceil(num_frames / num_cols)
    
    cell_w = screen_width // num_cols
    cell_h = screen_height // num_rows

    grid = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    placeholder = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'Connecting...', (50, cell_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    camera_ids = sorted(config.VIDEO_SOURCES.keys())

    for i, camera_id in enumerate(camera_ids):
        frame = frames.get(camera_id)
        
        if frame is not None:
            resized_frame = cv2.resize(frame, (cell_w, cell_h))
        else:
            resized_frame = placeholder.copy()

        # --- DRAW CAMERA NAME LABEL (BOTTOM-RIGHT) ---
        cam_text = camera_id.replace('_', ' ').title()
        (text_w, text_h), _ = cv2.getTextSize(cam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        box_x1 = cell_w - text_w - 20
        box_y1 = cell_h - text_h - 20
        box_x2 = cell_w - 10
        box_y2 = cell_h - 10
        
        cv2.rectangle(resized_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.putText(resized_frame, cam_text, (box_x1 + 5, box_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        row, col = divmod(i, num_cols)
        y_start, x_start = row * cell_h, col * cell_w
        grid[y_start:y_start + cell_h, x_start:x_start + cell_w] = resized_frame

    return grid

def draw_info_overlay(frame, processors, threads, reid_manager=None, debug=False):
    """Draws performance and system status information."""
    y_offset = 30

    alive_threads = sum(1 for t in threads if t.is_alive())
    cv2.putText(frame, f"Active Threads: {alive_threads}/{len(threads)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 40

    # --- ALWAYS DRAW GLOBAL FPS ---
    if processors:
        active_processors = [p for p in processors if p.fps > 0]
        if active_processors:
            avg_fps = sum(p.fps for p in active_processors) / len(active_processors)
            cv2.putText(frame, f"System FPS (Avg): {avg_fps:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 25

    # --- Re-ID gallery stats ---
    if reid_manager and reid_manager.is_available:
        gallery_size = reid_manager.gallery_size()
        cv2.putText(frame, f"ReID Gallery: {gallery_size} identities",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
        y_offset += 25

    # --- DEBUG OVERLAY: extra diagnostics ---
    if debug and processors:
        y_offset += 10
        cv2.putText(frame, "-- DEBUG MODE --", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25

        for p in processors:
            tracks = getattr(p, 'active_track_count', 0)
            det_count = getattr(p, 'frame_detection_count', 0)
            info = f"{p.camera_id}: {p.fps:.1f}fps | {det_count} det | {tracks} trk"
            cv2.putText(frame, info, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

        # ONNX provider info
        y_offset += 5
        try:
            providers = processors[0].yolo.session.get_providers()
            provider_str = providers[0].replace('ExecutionProvider', '')
            cv2.putText(frame, f"ONNX Backend: {provider_str}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        except Exception:
            pass

    return frame

def validate_config():
    """Validate required config fields and log warnings for missing ones. Returns True if all OK."""
    warnings_list = []
    if not config.API_URL:
        warnings_list.append("API_URL is empty — heartbeat and cloud sync DISABLED")
    elif not config.API_URL.startswith("https://"):
        # L7-fix: enforce HTTPS for all cloud API calls to prevent credential leakage
        warnings_list.append(
            f"API_URL does not start with https:// — cleartext HTTP is insecure. "
            f"Current value: {config.API_URL!r}"
        )
    if not config.DEVICE_KEY:
        warnings_list.append("DEVICE_KEY is empty — API authentication will fail")
    if not os.path.exists(config.YOLO_MODEL_PATH):
        warnings_list.append(f"YOLO model not found at: {config.YOLO_MODEL_PATH}")
    if not os.path.exists(config.ATTRIBUTE_MODEL_PATH):
        warnings_list.append(f"Attribute model not found at: {config.ATTRIBUTE_MODEL_PATH}")

    for w in warnings_list:
        logger.warning(f"CONFIG WARNING: {w}")

    return len(warnings_list) == 0  # True if all OK


def validate_camera_sources(sources):
    """Check each camera source is accessible via cv2.VideoCapture. Returns valid sources dict."""
    valid = {}
    for src_id, src in sources.items():
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            valid[src_id] = src
            cap.release()
        else:
            logger.warning(f"Camera source not accessible: {src_id} → {src}")
    if not valid:
        logger.error("No valid camera sources found — check cameras.json or device config")
        # Don't sys.exit here; warn loudly so local testing still works
    return valid


def _print_banner():
    """Prints a colored NORT startup banner to stdout."""
    C  = "\033[38;2;36;182;252m"  # brand #24B6FC
    B  = "\033[94m"   # bright blue
    DIM= "\033[90m"   # dark grey
    W  = "\033[97m"   # white
    G  = "\033[92m"   # green
    Y  = "\033[93m"   # yellow
    R  = "\033[0m"    # reset
    BD = "\033[1m"    # bold

    logo_raw = [
        "               ██",
        "             ▄████",
        "            ▄██████▄",
        "           ▄████████▄",
        "          ███████████▄",
        "         █████████████▄",
        "        ████████████████",
        "      ▄██████████████████",
        "     ▄████████████████████▄",
        "    ▄██████████████████████▄",
        "   ▄████████████████████████▄",
        "  ██████████▀▀    ▀▀█████████▄",
        " ██████▀▀             ▀▀▀██████",
        "▀█▀▀▀                      ▀▀▀█▀"
    ]

    right_col = [
        "",
        "",
        "",
        "",
        "",
        f"{BD}{W}N O R T{R}",
        f"{DIM}AI-Powered Retail Intelligence{R}",
        f"{DIM}https://ainort.com  │  v{config.APP_VERSION}{R}",
        "",
        "",
        "",
        "",
        "",
        ""
    ]

    print()
    for left, right in zip(logo_raw, right_col):
        # 4 indent + 34-width padded logo + side text
        print(f"    {C}{left.ljust(34)}{R}{right}")

    print()
    print(f"  {DIM}{'─' * 60}{R}")
    print()
    print(f"  {DIM}Device{R}  {G}{config.DEVICE_ID}{R}  {DIM}│ Client{R}  {Y}{config.CLIENT_ID}{R}  {DIM}│ Store{R}  {Y}{config.STORE_ID}{R}")
    print(f"  {DIM}Cameras{R}  {W}{len(config.VIDEO_SOURCES)} configured{R}  {DIM}│ Admin{R}  {W}http://localhost:{config.ADMIN_PORT}{R}")
    print()
    print(f"  {DIM}{'─' * 60}{R}")
    print()




def build_arg_parser():
    """Build the CLI argument parser for NORT."""
    import argparse
    p = argparse.ArgumentParser(
        prog="NORT",
        description="NORT — Edge AI People Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py                          # default: all cameras, display on, ReID on
  python run.py --headless               # production: no GUI windows
  python run.py --cameras camera_1,camera_3  # only enable two cameras
  python run.py --no-reid --headless     # lightweight mode, no Re-ID, no GUI
  python run.py --conf 0.25 --debug      # higher threshold + debug overlays
  python run.py --log-level DEBUG        # verbose logging
  python run.py --live-pose --headless   # enable live pose stream in admin panel
""",
    )

    # ── Display ──────────────────────────────────────────────────────────────
    disp = p.add_argument_group("Display")
    disp.add_argument(
        "--headless", action="store_true",
        help="Run without any GUI windows (OpenCV). Use for production / Jetson.",
    )
    disp.add_argument(
        "--no-gallery", action="store_true",
        help="Disable the Re-ID gallery window (saves GPU memory).",
    )
    disp.add_argument(
        "--no-zones", action="store_true",
        help="Hide zone polygon overlays on the video feed.",
    )
    disp.add_argument(
        "--no-paths", action="store_true",
        help="Hide person movement paths on the video feed.",
    )
    disp.add_argument(
        "--no-entrance-debug", action="store_true",
        help="Hide entrance line debug drawing.",
    )

    # ── Detection ────────────────────────────────────────────────────────────
    det = p.add_argument_group("Detection")
    det.add_argument(
        "--model", type=str, default=None,
        help="Path to the YOLOX ONNX model (default: assets/models/yolox_m.onnx).",
    )
    det.add_argument(
        "--conf", type=float, default=None,
        help="Detection confidence threshold (default: 0.15).",
    )
    det.add_argument(
        "--imgsz", type=int, default=None,
        help="YOLOX input image size (default: 640).",
    )

    # ── Re-ID ────────────────────────────────────────────────────────────────
    reid = p.add_argument_group("Re-ID")
    reid.add_argument(
        "--no-reid", action="store_true",
        help="Disable cross-camera Re-ID entirely.",
    )
    reid.add_argument(
        "--reid-threshold", type=float, default=None,
        help="Re-ID similarity threshold (default: 0.52).",
    )

    # ── Cameras ──────────────────────────────────────────────────────────────
    cam = p.add_argument_group("Camera Selection")
    cam.add_argument(
        "--cameras", type=str, default=None,
        help="Comma-separated list of camera IDs to enable (e.g. camera_1,camera_3).",
    )

    # ── Admin & Networking ───────────────────────────────────────────────────
    net = p.add_argument_group("Admin & Networking")
    net.add_argument(
        "--admin-port", type=int, default=None,
        help="Port for the local admin web UI (default: from device.json).",
    )
    net.add_argument(
        "--no-heartbeat", action="store_true",
        help="Disable the heartbeat thread (offline / dev mode).",
    )
    net.add_argument(
        "--no-sync", action="store_true",
        help="Disable the cloud sync thread (offline / dev mode).",
    )
    net.add_argument(
        "--setup", action="store_true",
        help="Run in SETUP MODE (ai-off): no YOLOX/Re-ID loading, camera feeds only for configuration.",
    )

    # ── Pose Estimation ──────────────────────────────────────────────────────
    pose = p.add_argument_group("Pose Estimation")
    pose.add_argument(
        "--live-pose", action="store_true",
        help=(
            "Enable continuous live pose estimation on the selected person in the "
            "admin panel (~6-7 fps MJPEG stream). Adds ~10-15%% GPU load while "
            "a person is selected. Off by default. "
            "Requires: pip install ultralytics"
        ),
    )

    # ── Debug & Logging ──────────────────────────────────────────────────────
    dbg = p.add_argument_group("Debug & Logging")
    dbg.add_argument(
        "--debug", action="store_true",
        help="Enable debug-level logging and all debug overlays.",
    )
    dbg.add_argument(
        "--log-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO).",
    )
    dbg.add_argument(
        "--force-trt", action="store_true",
        help="Force TensorRT provider on Windows (requires TensorRT DLLs on PATH). No-op on Jetson.",
    )

    return p


def apply_args_to_config(args):
    """Override config.py values with any CLI flags that were explicitly set."""
    if args.model:
        config.YOLO_MODEL_PATH = args.model
    if args.conf is not None:
        config.YOLO_CONF_THRESHOLD = args.conf
    if args.imgsz is not None:
        config.YOLO_IMGSZ = args.imgsz
    if args.no_reid:
        config.REID_ENABLED = False
    if args.live_pose:
        config.LIVE_POSE_ENABLED = True
    if args.reid_threshold is not None:
        config.REID_SIMILARITY_THRESHOLD = args.reid_threshold
    if args.no_zones:
        config.DRAW_ZONES = False
    if args.no_paths:
        config.ENABLE_PATH_DRAWING = False
    if args.no_entrance_debug:
        config.DRAW_ENTRANCE_DEBUG = False
    if args.admin_port is not None:
        config.ADMIN_PORT = args.admin_port
    if args.debug:
        config.LOG_LEVEL = "DEBUG"
        config.DRAW_ENTRANCE_DEBUG = True
        config.DRAW_ZONES = True
    if args.log_level:
        config.LOG_LEVEL = args.log_level

    # Reconfigure all existing loggers to the new level
    import logging as _logging
    level = getattr(_logging, config.LOG_LEVEL, _logging.INFO)
    for _name, _logger in _logging.Logger.manager.loggerDict.items():
        if isinstance(_logger, _logging.Logger):
            _logger.setLevel(level)
    _logging.getLogger().setLevel(level)
    if args.cameras:
        selected = [c.strip() for c in args.cameras.split(",")]
        config.VIDEO_SOURCES = {
            k: v for k, v in config.VIDEO_SOURCES.items() if k in selected
        }
        if not config.VIDEO_SOURCES:
            logger.critical(f"No matching cameras for: {args.cameras}")
            logger.critical(f"Available: {list(config._build_video_sources().keys())}")
            sys.exit(1)


def main():
    """Main function to initialize and run the application."""
    # ── Parse CLI arguments ────────────────────────────────────────────────
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_args_to_config(args)

    # ── Validate config and environment early ─────────────────────────────────
    validate_config()

    _print_banner()

    # Print active flags
    active_flags = []
    if args.headless:       active_flags.append("headless")
    if args.no_reid:        active_flags.append("no-reid")
    if args.no_gallery:     active_flags.append("no-gallery")
    if args.no_heartbeat:   active_flags.append("no-heartbeat")
    if args.no_sync:        active_flags.append("no-sync")
    if args.debug:          active_flags.append("debug")
    if args.live_pose:      active_flags.append("live-pose")
    if args.cameras:        active_flags.append(f"cameras={args.cameras}")
    if args.conf is not None: active_flags.append(f"conf={args.conf}")
    if active_flags:
        logger.info(f"CLI flags: {', '.join(active_flags)}")

    logger.info(
        f"Starting NORT tracking — "
        f"device={config.DEVICE_ID}, client={config.CLIENT_ID}, store={config.STORE_ID}"
    )

    # --- Initialize shared resources FIRST ---
    if not args.headless:
        screen_width, screen_height = get_screen_resolution()
        logger.info(f"Screen resolution: {screen_width}x{screen_height}")
    else:
        screen_width, screen_height = 1920, 1080  # dummy for headless
        logger.info("Running in HEADLESS mode — no GUI windows will be shown.")

    # ── Validate camera sources before loading any heavy models ──────────────
    if not args.setup:
        config.VIDEO_SOURCES = validate_camera_sources(config.VIDEO_SOURCES)
        if not config.VIDEO_SOURCES:
            logger.critical(
                "No accessible camera sources found after validation. "
                "Check cameras.json and ensure all video files/streams are reachable. "
                "Aborting startup."
            )
            sys.exit(1)

    data_handler = DataHandler()
    output_frames = {}
    lock = threading.Lock()
    stop_event = threading.Event()
    barrier = threading.Barrier(len(config.VIDEO_SOURCES) + 1)  # +1 for main thread

    # ── Cross-Camera Re-ID Setup ───────────────────────────────────────────
    reid_manager = None
    reid_gallery = None
    
    if not args.setup and getattr(config, 'REID_ENABLED', True):
        reid_manager = ReIDManager(
            similarity_threshold=getattr(config, 'REID_SIMILARITY_THRESHOLD', 0.70),
            device=config.DEVICE
        )
        if reid_manager.is_available:
            logger.info("[ReID] Cross-camera Re-ID is ENABLED (OSNet-x0_25)")
        else:
            logger.warning("[ReID] torchreid not found — Re-ID disabled gracefully. "
                           "Install with: pip install torchreid")
        # Gallery window
        if not args.headless and not args.no_gallery:
            reid_gallery = ReidGalleryWindow(reid_manager, max_rows=5)
            logger.info("[ReID] Gallery window opened")
    elif args.setup:
        logger.info("🛠 SETUP MODE: Skipping Re-ID manager initialization.")
    else:
        logger.info("[ReID] Cross-camera Re-ID is DISABLED explicitly via configuration")

    # ── VLM (Moondream) Warmup ───────────────────────────────────────────────
    if not args.setup:
        try:
            from core import vlm_analyst
            vlm_cfg = vlm_analyst.get_config()
            if vlm_cfg.enabled and vlm_cfg.warmup_on_start:
                logger.info("[VLM] Warming up Moondream model...")
                success = vlm_analyst.warmup_model()
                if success:
                    logger.info("✓ [VLM] Moondream model warmed up successfully!")
                else:
                    logger.warning("[VLM] Moondream warmup returned False — VLM may be unavailable.")
            elif not vlm_cfg.enabled:
                logger.info("[VLM] Visual Language Model is DISABLED in configuration.")
            else:
                logger.info("[VLM] Visual Language Model enabled (lazy-load on first use).")
        except Exception as e:
            logger.warning(f"[VLM] Could not initialize VLM module: {e}")

    # Fire detector import — deferred start happens after processors are created
    _fire_detector_mod = None
    if not args.setup:
        try:
            from core import fire_detector as _fire_detector_mod
        except Exception as e:
            logger.warning(f"[FireDetector] Could not import fire_detector module: {e}")

    shared_yolo_model = None
    if not args.setup:
        try:
            logger.info(f"Loading YOLOX ONNX model from: {config.YOLO_MODEL_PATH}")
            
            # Load YOLOX model via ONNXRuntime (auto-selects TensorRT > CUDA > CPU)
            shared_yolo_model = YOLOXDetector(
                model_path=config.YOLO_MODEL_PATH,
                conf_threshold=config.YOLO_CONF_THRESHOLD,
                force_trt=getattr(args, 'force_trt', False),
            )
            
            # Display model info
            logger.info(f"✓ YOLOX Model Information:")
            logger.info(f"  - Path: {config.YOLO_MODEL_PATH}")
            logger.info(f"  - Input Size: {shared_yolo_model.input_size}")
            logger.info(f"  - Active Providers: {shared_yolo_model.session.get_providers()}")
            
            # Warm up the model with dummy inference
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            warmup_results = shared_yolo_model.detect(dummy_img)
            
            logger.info("✓ YOLOX model loaded and warmed up successfully!")
            
        except Exception as e:
            logger.critical(f"Failed to load the YOLOX ONNX model. Error: {e}")
            logger.critical(
                "On Jetson: run  python3 scripts/build_engines.py  first to build "
                "the TRT .engine files.  Falling back to ORT requires onnxruntime."
            )
            return
    else:
        logger.info("🛠 SETUP MODE: Skipping YOLOX model loading.")

    # ── Shared Attribute Session ───────────────────────────────────────────────
    # Load ONE session shared across all cameras — avoids N GPU model copies.
    # Priority: pre-built .engine (native TRT) → ORT TRT EP → ORT CUDA → CPU.
    shared_attr_session = None
    if not args.setup and shared_yolo_model is not None:
        _attr_onnx_path   = os.path.join("assets", "models", "attribute_model.onnx")
        _attr_engine_path = os.path.splitext(_attr_onnx_path)[0] + ".engine"

        # ── 1. Try native TRTSession (.engine file) ───────────────────────────
        if os.path.exists(_attr_engine_path):
            try:
                from core.trt_session import TRTSession, is_available as _trt_ok
                if _trt_ok():
                    shared_attr_session = TRTSession(_attr_engine_path)
                    logger.info(
                        f"✓ Shared attribute model loaded via TRTSession: "
                        f"{os.path.basename(_attr_engine_path)}"
                    )
            except Exception as _e:
                logger.warning(
                    f"TRTSession for attribute model failed ({_e}), trying ORT..."
                )
                shared_attr_session = None

        # ── 2. Fall back to ONNXRuntime ───────────────────────────────────────
        if shared_attr_session is None and os.path.exists(_attr_onnx_path):
            try:
                import onnxruntime as _ort
                _so = _ort.SessionOptions()
                _so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Re-use the same provider chain as the YOLO model
                _providers = shared_yolo_model.session.get_providers()
                shared_attr_session = _ort.InferenceSession(
                    _attr_onnx_path, sess_options=_so, providers=_providers
                )
                logger.info(
                    f"✓ Shared attribute model loaded via ORT: "
                    f"{shared_attr_session.get_providers()}"
                )
            except Exception as _e:
                logger.warning(
                    f"Could not load shared attribute model: {_e} "
                    f"(each camera will load its own)"
                )

        if shared_attr_session is None:
            logger.warning(
                f"Shared attribute model not found at: "
                f"{_attr_engine_path} or {_attr_onnx_path}"
            )

    processors = []
    threads = []

    for camera_id, video_source in config.VIDEO_SOURCES.items():
        processor = CameraProcessor(
            camera_id,
            video_source,
            data_handler,
            output_frames,
            lock,
            stop_event,
            barrier,
            shared_yolo_model,
            reid_manager=reid_manager,
            setup_mode=args.setup,
            shared_attr_session=shared_attr_session,  # single GPU session shared across all cameras
        )
        processors.append(processor)
        thread = threading.Thread(target=processor.run, name=f"Thread-{camera_id}")
        threads.append(thread)
        thread.start()

    # ── Fire & Smoke Detection ────────────────────────────────────────────────
    # Started HERE (after processors are populated) so _get_cam_ids closure works.
    if not args.setup and _fire_detector_mod is not None:
        try:
            def _get_raw_frame(cam_id: str):
                with lock:
                    for p in processors:
                        if p.camera_id == cam_id:
                            raw = getattr(p, 'latest_raw_frame', None)
                            return raw.copy() if raw is not None else None
                return None

            def _get_cam_ids():
                return [p.camera_id for p in processors]

            _fire_detector_mod.start(_get_raw_frame, stop_event, _get_cam_ids)
            logger.info("[FireDetector] Fire & smoke detection ENABLED.")
        except Exception as e:
            logger.warning(f"[FireDetector] Could not start fire detector: {e}")

    # ── Register processors for remote commands (Gap 4C) ─────────────────────
    ota_agent.set_processors(processors)

    # ── Start local admin page (Gap 4B) ─────────────────────────────────────
    local_admin.DEVICE_ID = config.DEVICE_ID
    local_admin.CLIENT_ID  = config.CLIENT_ID
    local_admin.STORE_ID   = config.STORE_ID
    local_admin.ADMIN_PIN  = config.ADMIN_PIN
    local_admin.API_URL    = config.API_URL
    threading.Thread(
        target=lambda: local_admin.start_local_admin(port=config.ADMIN_PORT),
        daemon=True,
        name="LocalAdmin",
    ).start()

    # ── Start proactive alert engine ─────────────────────────────────────────
    from core import alert_engine as _alert_engine_mod
    _alert_engine_mod.start_alert_engine()
    logger.info("[Run] Alert engine started.")

    def _get_local_ip():
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    local_ip = _get_local_ip()
    _admin_cert = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "admin", "admin_cert.pem"
    )
    _admin_proto = "https" if os.path.exists(_admin_cert) else "http"
    logger.info(
        f"Local admin page started on {_admin_proto}://{local_ip}:{config.ADMIN_PORT}/ "
        f"({'self-signed TLS — accept browser warning' if _admin_proto == 'https' else 'HTTP — no TLS'})"
    )

    # ── L5-fix: systemd watchdog pinger ──────────────────────────────────────
    threading.Thread(
        target=_systemd_watchdog_loop,
        args=(stop_event,),
        daemon=True,
        name="SystemdWatchdog",
    ).start()

    # ── Start heartbeat thread (Gap 4A) ───────────────────────────────────────
    last_upload_ref = {"ts": None}   # shared mutable ref updated by uploader
    if not args.no_heartbeat and not args.setup:
        threading.Thread(
            target=_heartbeat_loop,
            args=(processors, lock, last_upload_ref, stop_event),
            daemon=True,
            name="Heartbeat",
        ).start()
    elif args.setup:
        logger.info("🛠 SETUP MODE: Heartbeat thread DISABLED.")
    else:
        logger.info("Heartbeat thread DISABLED via --no-heartbeat flag.")

    if not args.no_sync and not args.setup:
        sync_manager = SyncManager(last_upload_ref=last_upload_ref)
        threading.Thread(
            target=_sync_loop,
            args=(sync_manager, stop_event),
            daemon=True,
            name="SyncManager",
        ).start()
        
        threading.Thread(
            target=_log_rotation_loop,
            args=(stop_event, sync_manager),
            daemon=True,
            name="LogRotation",
        ).start()
    elif args.setup:
        logger.info("🛠 SETUP MODE: Sync manager and log rotation DISABLED.")
    else:
        logger.info("Cloud sync and log rotation DISABLED via --no-sync flag.")

    # ── Graceful shutdown on SIGINT / SIGTERM ────────────────────────────────
    import signal

    _shutdown_pending = threading.Event()   # Ctrl+C received, waiting for y/n
    _shutdown_confirmed = threading.Event()  # User typed 'y'

    def _graceful_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        if signum == signal.SIGTERM:
            # SIGTERM (systemd stop) — no prompt, shut down immediately
            logger.info(f"Received {sig_name} — shutting down immediately...")
            _shutdown_confirmed.set()
            stop_event.set()
            return

        # SIGINT (Ctrl+C)
        if _shutdown_confirmed.is_set() or stop_event.is_set():
            return  # already shutting down

        if _shutdown_pending.is_set():
            # Second Ctrl+C — force shutdown without waiting
            print("\n  ⚠  Force shutdown...")
            _shutdown_confirmed.set()
            stop_event.set()
            return

        # First Ctrl+C — ask for confirmation
        _shutdown_pending.set()
        print(f"\n  ⚠  Received {sig_name}. Shut down NORT? [y/N] ", end="", flush=True)

    def _read_shutdown_confirmation():
        """Tiny thread that blocks on stdin waiting for the y/n answer."""
        while not stop_event.is_set():
            _shutdown_pending.wait(timeout=1.0)
            if not _shutdown_pending.is_set():
                continue
            try:
                answer = input().strip().lower()
                if answer in ("y", "yes"):
                    logger.info("Shutdown confirmed by user.")
                    _shutdown_confirmed.set()
                    stop_event.set()
                else:
                    print("  ↳ Shutdown cancelled. Resuming...", flush=True)
                    _shutdown_pending.clear()
            except (EOFError, OSError):
                _shutdown_confirmed.set()
                stop_event.set()
            break

    threading.Thread(target=_read_shutdown_confirmation, daemon=True, name="ShutdownPrompt").start()

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        logger.info("Waiting for all camera processors to start...")
        barrier.wait()
        logger.info("All processors started. Entering main display loop.")

        gallery_frame_idx = 0
        while not stop_event.is_set():
            if barrier.broken:
                logger.error("A thread failed to start. Shutting down.")
                break

            with lock:
                current_frames = output_frames.copy()

            if not args.headless:
                grid_view = create_grid_view(current_frames, screen_width, screen_height)
                grid_view_with_info = draw_info_overlay(grid_view, processors, threads, reid_manager=reid_manager, debug=args.debug)

                cv2.imshow(config.GRID_WINDOW_TITLE, grid_view_with_info)

                # ── Re-ID Gallery window ────────────────────────────────────
                if reid_gallery is not None:
                    gallery_frame_idx += 1
                    reid_gallery.update(gallery_frame_idx)
                    reid_gallery.show()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  ⚠  Quit requested. Shut down NORT? [y/N] ", end="", flush=True)
                    _shutdown_pending.set()
                    # Wait briefly for the answer
                    _shutdown_confirmed.wait(timeout=10.0)
                    if _shutdown_confirmed.is_set():
                        break
                    else:
                        _shutdown_pending.clear()
                        continue
                if not any(t.is_alive() for t in threads):
                    break
                if key == ord('f'):
                    is_fullscreen = cv2.getWindowProperty(config.GRID_WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(config.GRID_WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, 1 - is_fullscreen)
            else:
                # Headless loop — check threads are alive, sleep efficiently
                if not any(t.is_alive() for t in threads):
                    logger.info("All threads exited. Shutting down.")
                    break

            # Keep local admin camera status up to date
            local_admin.set_camera_status(_make_camera_status(processors))
            local_admin.set_retail_data(_make_retail_data(processors))

            # Push latest annotated frames to admin panel for MJPEG streaming
            # CRITICAL FIX: Extract frames quickly inside the lock, then encode
            # outside the lock. Doing cv2.imencode inside the global lock starves
            # the CameraProcessor threads and permanently freezes the video stream!
            frames_to_encode = []
            with lock:
                for p in processors:
                    if p.camera_id in output_frames and output_frames[p.camera_id] is not None:
                        frames_to_encode.append((p.camera_id, output_frames[p.camera_id].copy()))
                        if getattr(p, 'latest_raw_frame', None) is not None:
                            frames_to_encode.append((p.camera_id + "_raw", p.latest_raw_frame.copy()))

            # Now perform the heavy JPEG encoding outside the global lock
            for cam_id, frame_bgr in frames_to_encode:
                if cam_id.endswith("_raw"):
                    local_admin.set_latest_raw_frame(cam_id.replace("_raw", ""), frame_bgr)
                else:
                    local_admin.set_latest_frame(cam_id, frame_bgr)

            time.sleep(0.01)
    
    except threading.BrokenBarrierError:
        logger.critical("Could not synchronize all threads. One or more cameras may have failed.")
    
    finally:
        # ── Phase 1: Signal all threads to stop ────────────────────────────
        stop_event.set()
        logger.info("Stop event set. Waiting for threads to join...")

        # C5-fix: give camera threads 30 s to finish their current frame +
        # flush the final CSV rows (DataHandler.write_data calls fsync).
        for thread in threads:
            thread.join(timeout=30.0)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} did not exit within 30 s timeout.")

        # C3/C5-fix: stop ReID background threads cleanly so the gallery is
        # not corrupted mid-merge when the process exits.
        if reid_manager is not None:
            try:
                reid_manager.shutdown(timeout=5.0)
                logger.info("ReID manager shut down cleanly.")
            except Exception as _e:
                logger.warning(f"ReID manager shutdown error: {_e}")

        # ── Phase 2: Flush pending data ────────────────────────────────────
        try:
            # DataHandler writes are synchronous + fsync on every call, so
            # no explicit flush is needed — just log that we're past the join.
            logger.info("Data handler: all camera threads joined, CSV data is durable.")
        except Exception as e:
            logger.warning(f"Error during data handler cleanup: {e}")

        try:
            if hasattr(spatial_logger, '_conn') and spatial_logger._conn is not None:
                spatial_logger._conn.close()
                logger.info("Spatial logger DB closed.")
        except Exception as e:
            logger.warning(f"Error closing spatial logger: {e}")

        # ── Phase 3: Release camera captures ──────────────────────────────
        for p in processors:
            try:
                if hasattr(p, 'cap') and p.cap is not None:
                    p.cap.release()
            except Exception:
                pass

        # ── Phase 4: Destroy GUI resources ─────────────────────────────────
        if not args.headless:
            cv2.destroyAllWindows()
        if reid_gallery is not None:
            reid_gallery.destroy()

        logger.info("✓ Application shut down gracefully.")

if __name__ == "__main__":
    main()
