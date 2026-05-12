"""
local_admin.py — Nort Jetson Store Manager Panel (v2).

Runs as a background daemon thread inside main.py.
A store employee (or Nort support) can open:

    http://<jetson-ip>:8080/

in any browser on the store's local network (no SSH, no app, no code).

Features:
  • Smart actionable notification center (camera offline, sync overdue, high CPU/disk)
  • Today's store insights: visitors, active cameras, last sync, uptime
  • System health: CPU, RAM, temp, disk, internet, API
  • Camera pipeline table with FPS
  • Device controls: Sync Now, Restart Pipeline
  • Live JSON polling via /api/status (no full-page reload)
  • GET /healthz for cloud checker
"""
import csv
import subprocess
import os
import time
import shutil
from threading import Event
from datetime import datetime, date

import cv2
import queue
import logging
from system.logger_setup import setup_logger

# Module-level logger — used by functions that don't have their own local logger
_log = logging.getLogger("nort.admin")

from flask import Flask, render_template_string, jsonify, request, redirect, Response, send_file
from urllib.parse import urlparse
from functools import wraps
import threading
import json
from collections import deque

try:
    import psutil
    # psutil.cpu_percent() returns 0.0 on the very first call because it needs a
    # baseline interval.  Discard one reading now so that _poll_performance gets
    # accurate data from its very first real sample (~2 s later).
    psutil.cpu_percent()
    print(f"[perf-init] psutil OK  ram={psutil.virtual_memory().percent:.1f}%", flush=True)
except ImportError:
    psutil = None
    print("[perf-init] psutil NOT installed — cpu/ram will read 0", flush=True)

# ── Globals set by main.py ────────────────────────────────────────────────────
DEVICE_ID    = "unknown"
CLIENT_ID    = "unknown"
STORE_ID     = "unknown"
ADMIN_PIN    = "1234"
API_URL      = ""

_start_time  = time.monotonic()

# Mutable state injected by main.py via set_*() helpers below.
_camera_status: dict = {}   # camera_id → {"active": bool, "fps": float}
_last_upload_ts = None       # datetime or None
_force_upload_fn = None      # callable or None
_restart_fn = None           # callable or None

# Dismissed notification IDs (resets on restart)
_dismissed_notifications: set = set()

OCCUPANCY_LOG_PATH = os.path.join(os.path.dirname(__file__), "occupancy_log.csv")

app = Flask(__name__)

# Derive a stable secret key from the device identity so sessions survive
# process restarts (OTA updates, systemd restarts) without forcing re-login.
# Falls back to a random key if device.json is not yet available.
try:
    import hashlib as _hashlib, json as _json
    _dj = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "device.json")
    with open(_dj) as _f:
        _did = _json.load(_f).get("device_id", "")
    app.secret_key = _hashlib.sha256(f"nort-admin-{_did}".encode()).digest()
except Exception:
    app.secret_key = os.urandom(16)  # fallback: sessions reset on restart

# ── M7-fix: CSRF / same-origin protection ────────────────────────────────────
# Reject state-changing requests (POST/PUT/DELETE/PATCH) whose Origin or Referer
# header does not match the server's own host.  This defeats CSRF attacks from
# malicious web pages on the same LAN without requiring flask-wtf or any new
# dependency.  Requests from curl/scripts that omit both headers are allowed
# (same as a browser on the same machine); browsers always send at least one.
_CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

@app.before_request
def _csrf_origin_check():
    if request.method in _CSRF_SAFE_METHODS:
        return  # read-only — no state to protect
    # Determine the server's own authority (host:port)
    server_host = request.host  # e.g. "192.168.1.10:8080"
    for header in ("Origin", "Referer"):
        value = request.headers.get(header)
        if not value:
            continue
        try:
            parsed = urlparse(value)
            request_host = parsed.netloc or parsed.path  # Origin has netloc; bare Referer may not
            if request_host and request_host != server_host:
                _log.warning(
                    f"CSRF check blocked {request.method} {request.path} — "
                    f"{header}: {value!r} does not match server host {server_host!r}"
                )
                return jsonify({"error": "CSRF check failed"}), 403
        except Exception:
            pass  # malformed header — let it through (conservative)
    return  # all headers present matched, or no headers present (CLI/curl)


# ── NumPy-aware JSON encoder ──────────────────────────────────────────────────
# NumPy scalar types (int64, float32, …) are not serializable by stdlib json.
# This provider handles them globally so any jsonify() call in any route is safe.
import json as _json

class _NumpyJSONProvider(app.json_provider_class):
    def dumps(self, obj, **kwargs):
        return _json.dumps(obj, default=_numpy_default, **kwargs)
    def loads(self, s, **kwargs):
        return _json.loads(s, **kwargs)

def _numpy_default(o):
    try:
        import numpy as _np
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

app.json_provider_class = _NumpyJSONProvider
app.json = _NumpyJSONProvider(app)

# ── Translation System ────────────────────────────────────────────────────────

_translations = {}
def _load_translations():
    global _translations
    locales_dir = os.path.join(os.path.dirname(__file__), "locales")
    for lang in ["en", "pt", "es"]:
        path = os.path.join(locales_dir, f"{lang}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _translations[lang] = json.load(f)
            except Exception as e:
                print(f"Error loading translation {lang}: {e}")

_load_translations()

def t(key, fallback=None, **kwargs):
    """
    Translation helper used in both Python and Jinja2 templates.
    Usage in Python:  t('sidebar.dashboard')
    Usage in Jinja2:  {{ t('sidebar.dashboard') }}
                      {{ t('nav.help', 'Help') }}   ← optional positional fallback
    """
    # Try to get lang from cookie, fallback to 'pt'
    try:
        lang = request.cookies.get("lang", "pt")
    except RuntimeError:
        # Outside of request context
        lang = "pt"

    if lang not in _translations:
        lang = "pt"

    # Nested key lookup (e.g., "sidebar.dashboard")
    val = _translations.get(lang, {})
    for part in key.split("."):
        if isinstance(val, dict):
            val = val.get(part)
        else:
            val = None
            break

    if val is None:
        # Use explicit fallback if provided, otherwise return the key itself
        val = fallback if fallback is not None else key

    if not isinstance(val, str):
        val = fallback if fallback is not None else key

    # Simple variable replacement for {{ var }}
    for k, v in kwargs.items():
        val = val.replace(f"{{{{ {k} }}}}", str(v))
    return val

@app.context_processor
def inject_t():
    return dict(t=t, current_lang=request.cookies.get("lang", "pt"))

@app.route("/set_lang/<lang>")
def set_lang(lang):
    if lang not in ["en", "pt", "es"]:
        lang = "pt"
    # Redirect back to where the user was
    ref = request.referrer
    if not ref or 'set_lang' in ref:
        ref = "/"
    response = redirect(ref)
    response.set_cookie("lang", lang, max_age=60*60*24*30) # 30 days
    return response


# ── Rate Limiter (PIN brute-force protection) ─────────────────────────────────

class _RateLimiter:
    """
    L2-fix: Lightweight IP-based rate limiter with SQLite-backed persistence.
    Lockouts survive process restarts (important on a device that auto-restarts
    after OTA updates — an attacker can't bypass lockout by triggering a restart).

    Tracks failed auth attempts per IP. After `max_attempts` failures
    within `window_seconds`, the IP is blocked for `lockout_seconds`.
    """
    _DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_limit.db")

    def __init__(self, max_attempts=5, window_seconds=300, lockout_seconds=900):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self.lockout = lockout_seconds
        self._failures = {}   # ip → deque of wall-clock timestamps (epoch)
        self._init_db()

    def _init_db(self):
        import sqlite3 as _sqlite3
        try:
            with _sqlite3.connect(self._DB_PATH) as _conn:
                _conn.execute("""
                    CREATE TABLE IF NOT EXISTS lockouts (
                        ip TEXT PRIMARY KEY,
                        until_epoch REAL NOT NULL
                    )
                """)
        except Exception:
            pass  # non-fatal — fall back to in-memory only

    def _now_epoch(self):
        return time.time()

    def is_locked(self, ip):
        import sqlite3 as _sqlite3
        now = self._now_epoch()
        try:
            with _sqlite3.connect(self._DB_PATH) as _conn:
                row = _conn.execute(
                    "SELECT until_epoch FROM lockouts WHERE ip = ?", (ip,)
                ).fetchone()
                if row:
                    if now < row[0]:
                        return True
                    # Expired — remove it
                    _conn.execute("DELETE FROM lockouts WHERE ip = ?", (ip,))
        except Exception:
            pass
        return False

    def record_failure(self, ip):
        import sqlite3 as _sqlite3
        now = self._now_epoch()
        if ip not in self._failures:
            self._failures[ip] = deque()
        q = self._failures[ip]
        q.append(now)
        # Purge old entries outside the window
        while q and q[0] < now - self.window:
            q.popleft()
        if len(q) >= self.max_attempts:
            # Persist lockout so it survives restarts
            until = now + self.lockout
            try:
                with _sqlite3.connect(self._DB_PATH) as _conn:
                    _conn.execute(
                        "INSERT OR REPLACE INTO lockouts (ip, until_epoch) VALUES (?, ?)",
                        (ip, until)
                    )
            except Exception:
                pass
            self._failures.pop(ip, None)

    def record_success(self, ip):
        import sqlite3 as _sqlite3
        self._failures.pop(ip, None)
        try:
            with _sqlite3.connect(self._DB_PATH) as _conn:
                _conn.execute("DELETE FROM lockouts WHERE ip = ?", (ip,))
        except Exception:
            pass


_rate_limiter = _RateLimiter(max_attempts=5, window_seconds=300, lockout_seconds=900)

# ── Auth ───────────────────────────────────────────────────────────────────────

def check_auth(username, password):
    if username != 'admin':
        return False
    from werkzeug.security import check_password_hash
    try:
        # If it looks like a werkzeug/argon2 hash
        if ADMIN_PIN.startswith(("scrypt:", "pbkdf2:", "argon2:", "$argon2")):
            return check_password_hash(ADMIN_PIN, password)
        else:
            return password == ADMIN_PIN
    except Exception:
        return password == ADMIN_PIN

def authenticate():
    from flask import Response
    return Response(
        t('auth.failed'), 401,
        {'WWW-Authenticate': 'Basic realm="Nort Store Panel"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        client_ip = request.remote_addr or "unknown"

        # Check if this IP is locked out
        if _rate_limiter.is_locked(client_ip):
            return Response(
                t('auth.lockout'), 429,
                {"Retry-After": "900"})

        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            _rate_limiter.record_failure(client_ip)
            return authenticate()

        _rate_limiter.record_success(client_ip)
        return f(*args, **kwargs)
    return decorated


# ── Public setters (called by main.py) ───────────────────────────────────────

_camera_status: dict = {}
_retail_data: dict = {"occupancy": 0, "tracks": {}}
_latest_frames = {}   # Dict[camera_id, compressed JPEG payload]
_last_frame_encode_ts = {}

_latest_raw_frames = {}   # Dict[camera_id, compressed JPEG payload]
_last_raw_encode_ts = {}

_last_upload_ts = None
_force_upload_fn = None
_restart_fn = None
_start_time = time.monotonic()
_dismissed_notifications = set()

# Selected track for VLM analysis (sync between frontend and backend drawing)
_selected_track: dict = {"track_id": None, "camera_id": None, "ts": 0}
_selected_track_lock = threading.Lock()

_FRAME_ENCODE_INTERVAL = 0.05   # encode at most 20 fps for the stream
_RAW_FRAME_ENCODE_INTERVAL = 0.05 # encode at most 20 fps for raw frames

def set_camera_status(status: dict):
    global _camera_status
    _camera_status = status

def set_retail_data(data: dict):
    global _retail_data
    _retail_data = data

def set_selected_track(track_id: str, camera_id: str):
    """Set the currently selected track for backend-drawn highlight."""
    global _selected_track
    with _selected_track_lock:
        _selected_track = {
            "track_id": track_id,
            "camera_id": camera_id,
            "ts": time.time()
        }

def get_selected_track() -> dict:
    """Get the currently selected track."""
    with _selected_track_lock:
        return dict(_selected_track)

def clear_selected_track():
    """Clear the selected track."""
    global _selected_track
    with _selected_track_lock:
        _selected_track = {"track_id": None, "camera_id": None, "ts": 0}

def set_latest_frame(camera_id: str, bgr_frame) -> None:
    """Called from main.py to push the latest annotated BGR frame.
    Throttled to ~10 fps to avoid re-encoding every tracking frame.
    """
    import cv2 as _cv2
    now = time.monotonic()
    if now - _last_frame_encode_ts.get(camera_id, 0) < _FRAME_ENCODE_INTERVAL:
        return
    try:
        small = _cv2.resize(bgr_frame, (960, 540))
        ok, buf = _cv2.imencode('.jpg', small, [_cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            _latest_frames[camera_id] = buf.tobytes()
            _last_frame_encode_ts[camera_id] = now
    except Exception as e:
        print(f"FAILED TO SET LATEST FRAME: {e}")

def set_latest_raw_frame(camera_id: str, bgr_frame) -> None:
    """Called from main.py to push the raw BGR frame (no drawings).
    Throttled for snapshot / zone editor usage.
    """
    import cv2 as _cv2
    now = time.monotonic()
    if now - _last_raw_encode_ts.get(camera_id, 0) < _RAW_FRAME_ENCODE_INTERVAL:
        return
    try:
        ok, buf = _cv2.imencode('.jpg', bgr_frame, [_cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            _latest_raw_frames[camera_id] = buf.tobytes()
            _last_raw_encode_ts[camera_id] = now
    except Exception:
        pass

def set_last_upload(ts):
    global _last_upload_ts
    _last_upload_ts = ts

def set_force_upload_fn(fn):
    global _force_upload_fn
    _force_upload_fn = fn

def set_restart_fn(fn):
    global _restart_fn
    _restart_fn = fn


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.1.1"

def _scan_port(ip: str, port: int, timeout: float = 0.3) -> bool:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((ip, port))
        s.close()
        return (result == 0)
    except Exception:
        return False

def _format_uptime(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m:02d}m {s:02d}s"


def _read_tail(lines: int = 60) -> str:
    path = "tracking.log" if os.path.exists("tracking.log") else "/var/log/nort/tracking.log"
    if not os.path.exists(path):
        return "(log file not found)"
    try:
        from collections import deque
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            last_lines = deque(f, maxlen=lines)
        return "".join(last_lines) or "(log file empty)"
    except Exception as e:
        return f"(error reading log: {e})"


def _disk_info():
    try:
        d = shutil.disk_usage("/")
    except Exception:
        d = shutil.disk_usage("C:\\")
    pct = round(d.used / d.total * 100, 1)
    used_gb = round(d.used / 1e9, 1)
    total_gb = round(d.total / 1e9, 1)
    css = "ok" if pct < 70 else ("warn" if pct < 90 else "err")
    return pct, used_gb, total_gb, css


def _ping_host(host: str) -> bool:
    if not host:
        return False
    param = '-n' if os.name == 'nt' else '-c'
    try:
        subprocess.check_call(
            ["ping", param, "1", "-w", "2", host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def _read_temp() -> str:
    """Read CPU temperature from sysfs. Safe to call from any context."""
    if os.name != 'posix':
        return "N/A"
    _thermal_paths = [
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone0/temp",
    ]
    for _tp in _thermal_paths:
        try:
            with open(_tp) as f:
                temp_c = int(f.read().strip()) / 1000.0
                if temp_c > 0:
                    return f"{temp_c:.1f} °C"
        except Exception:
            continue
    return "N/A"


def _perf_info():
    """Return (cpu_pct, ram_pct, temp_str).
    Only called from _poll_performance (background OS thread).
    Route handlers must NOT call this — it calls psutil.cpu_percent() which
    resets the baseline and corrupts the background thread's readings.
    Route handlers should read _cpu_hist[-1] / _ram_hist[-1] directly and
    call _read_temp() for temperature."""
    if psutil:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
    else:
        cpu = 0
        ram = 0
    return cpu, ram, _read_temp()


# Cache the discovered Jetson GPU sysfs path so we don't glob on every poll.
_jetson_gpu_sysfs_path: str = ""   # "" = not yet discovered; None = not found


def _discover_jetson_gpu_path() -> str:
    """
    Walk every plausible sysfs path to find the Jetson GPU load file.
    Returns the path string on success, None if not found.
    Called once from _poll_performance (real OS thread, blocking is fine).
    """
    import glob as _glob
    # 1. Explicit known paths (fastest check)
    _explicit = [
        "/sys/devices/17000000.ga10b/load",
        "/sys/devices/platform/17000000.ga10b/load",
        "/sys/bus/platform/devices/17000000.ga10b/load",
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/gpu.0/load",
    ]
    # 2. Dynamic discovery for any board variant
    _dynamic = (
        _glob.glob("/sys/devices/*/load")
        + _glob.glob("/sys/devices/platform/*/load")
        + _glob.glob("/sys/bus/platform/devices/*/load")
    )
    for _p in _explicit + _dynamic:
        try:
            _val = int(open(_p).read().strip())
            if 0 <= _val <= 1000:    # valid range: 0 = idle, 1000 = 100%
                _log.info("[GPU] Jetson GPU sysfs path: %s  (current load: %d%%)", _p, _val // 10)
                return _p
        except (OSError, ValueError):
            continue
    _log.warning("[GPU] No Jetson sysfs GPU load path found — falling back to tegrastats")
    return None


def _gpu_info():
    """
    Returns (gpu_usage_pct, gpu_ram_pct). Called from _poll_performance (real OS
    thread) so blocking subprocess calls are safe here.

    Jetson aarch64 (JetPack 6.x):
      1. Read discovered sysfs 'load' node (0-1000, divide by 10 for %).
         Path is found once at startup via _discover_jetson_gpu_path().
      2. If sysfs unavailable: parse one line of tegrastats (GR3D_FREQ).
    x86: pynvml → nvidia-smi.
    """
    global _jetson_gpu_sysfs_path
    import platform as _platform

    # ── Jetson / aarch64 ─────────────────────────────────────────────────────
    if _platform.machine() == "aarch64":
        # Discover path on first call
        if _jetson_gpu_sysfs_path == "":
            _jetson_gpu_sysfs_path = _discover_jetson_gpu_path()  # may return None

        if _jetson_gpu_sysfs_path:
            try:
                _val = int(open(_jetson_gpu_sysfs_path).read().strip())
                _gpu_pct    = min(100, max(0, _val // 10))
                _gpu_mem_pct = int(psutil.virtual_memory().percent) if psutil else 0
                return _gpu_pct, _gpu_mem_pct
            except (OSError, ValueError):
                _jetson_gpu_sysfs_path = ""   # force rediscovery next poll

        # ── tegrastats fallback ────────────────────────────────────────────
        # Runs in _poll_performance (real OS thread) — blocking is fine.
        try:
            import re as _re
            _proc = subprocess.Popen(
                ["tegrastats"], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True
            )
            _line = _proc.stdout.readline()
            _proc.terminate()
            _m = _re.search(r'GR3D_FREQ\s+(\d+)%', _line)
            if _m:
                _gpu_pct     = int(_m.group(1))
                _gpu_mem_pct = int(psutil.virtual_memory().percent) if psutil else 0
                return _gpu_pct, _gpu_mem_pct
        except Exception:
            pass

        return 0, 0

    # ── x86 / discrete NVIDIA GPU ─────────────────────────────────────────────
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(util.gpu), int((mem.used / mem.total) * 100)
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip().split('\n')[0].split(',')
        if len(out) >= 3:
            return int(out[0]), int((float(out[1]) / float(out[2])) * 100)
    except Exception:
        pass

    return 0, 0


def _scan_occupancy_log() -> tuple:
    """
    Single pass over occupancy_log.csv → returns (visitor_count, peak_hour_str).

    Called by the background cache thread every 30 s.  Never called directly
    from a request handler — that would block the gevent event loop on slow
    Jetson eMMC I/O (the file grows by ~hundreds of rows per hour).
    """
    today = date.today().isoformat()
    count = 0
    hour_counts: dict = {}
    try:
        with open(OCCUPANCY_LOG_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0].startswith(today):
                    if row[1] == 'entry':
                        count += 1
                        try:
                            hour = int(row[0][11:13])
                            hour_counts[hour] = hour_counts.get(hour, 0) + 1
                        except Exception:
                            pass
                    elif row[1] == 'exit':
                        count -= 1
    except Exception:
        pass

    visitor_count = max(0, count)
    if hour_counts:
        peak = max(hour_counts, key=hour_counts.get)
        peak_hour = f"{peak:02d}:00–{peak+1:02d}:00"
    else:
        peak_hour = "N/A"
    return visitor_count, peak_hour


# ── Thread-pool helper for blocking I/O in gevent greenlets ──────────────────
# gevent greenlets run on a SINGLE OS thread (the hub).  Any blocking call
# (sqlite3, file I/O, subprocess) inside a greenlet freezes the entire event
# loop until it returns.  The fix: run blocking work in a real OS thread via
# gevent's thread pool and *await* the result cooperatively.  The current
# greenlet suspends (yielding to other greenlets) while the thread works.
#
# Usage:  result = _run_in_thread(blocking_fn, arg1, arg2)
_gevent_pool = None

def _run_in_thread(fn, *args, **kwargs):
    """Run *fn* in a gevent thread-pool worker; yield event loop while waiting.

    IMPORTANT: Never fall back to fn(*args, **kwargs) on failure — that would
    call the blocking function directly inside a greenlet, freezing the entire
    gevent hub OS-thread and making every page navigation time out.
    Return [] instead; callers already handle empty results gracefully.
    """
    global _gevent_pool
    try:
        if _gevent_pool is None:
            from gevent.threadpool import ThreadPool as _TP
            _gevent_pool = _TP(8)   # up to 8 parallel blocking ops
        _async = _gevent_pool.spawn(fn, *args, **kwargs)
        return _async.get(timeout=20)
    except Exception as _e:
        _log.warning("_run_in_thread: %s timed-out or failed (%r) — returning []",
                     getattr(fn, "__name__", fn), _e)
        return []  # safe empty result; NEVER call fn() here


# ── KPI cache: visitor count + peak hour ─────────────────────────────────────
# Updated every 30 s by a background thread so request handlers never block
# on file I/O.  Reads from the cache dict are instantaneous.
_kpi_cache: dict = {"visitors_today": -1, "peak_hour": "N/A", "updated": 0.0}


def _refresh_kpi_cache():
    """Background thread: refresh visitor count and peak hour every 30 s."""
    while True:
        try:
            visitors, peak = _scan_occupancy_log()
            _kpi_cache["visitors_today"] = visitors
            _kpi_cache["peak_hour"]      = peak
            _kpi_cache["updated"]        = time.time()
        except Exception:
            pass
        time.sleep(30)


threading.Thread(target=_refresh_kpi_cache, daemon=True, name="KPI-Cache").start()


def _today_visitor_count() -> int:
    """Return cached visitor count (updated every 30 s by background thread)."""
    return _kpi_cache["visitors_today"]


def _peak_hour_today() -> str:
    """Return cached peak hour (updated every 30 s by background thread)."""
    return _kpi_cache["peak_hour"]


def _recent_entries(minutes=2) -> list:
    """Returns a list of track_ids that entered in the last `minutes`."""
    try:
        out = subprocess.check_output(
            ["tail", "-50", OCCUPANCY_LOG_PATH],
            stderr=subprocess.DEVNULL
        ).decode(errors="replace")
        lines = out.strip().split("\n")
        now = datetime.now()
        entries = []
        for line in reversed(lines):
            parts = line.split(",")
            if len(parts) >= 4 and parts[1] == "entry":
                try:
                    ts = datetime.fromisoformat(parts[0])
                    if (now - ts).total_seconds() <= minutes * 60:
                        entries.append(parts[2])
                    else:
                        break
                except Exception:
                    pass
        return entries
    except Exception:
        return []


# ── Performance History (Background Thread) ───────────────────────────────────

_perf_history_len = 30
_cpu_hist = deque([0] * _perf_history_len, maxlen=_perf_history_len)
_ram_hist = deque([0] * _perf_history_len, maxlen=_perf_history_len)
_gpu_hist = deque([0] * _perf_history_len, maxlen=_perf_history_len)
_time_hist = deque([""] * _perf_history_len, maxlen=_perf_history_len)

# ── Connectivity cache ────────────────────────────────────────────────────────
# _ping_host() is slow (subprocess + potential DNS timeout up to 10+ seconds).
# Calling it synchronously on every page load made the dashboard 12+ seconds.
# Instead a background thread refreshes the cache every 30 s; the route reads
# the cached values instantly.
_connectivity = {"internet": False, "api": False}

def _refresh_connectivity():
    """Background thread: update _connectivity cache every 30 s."""
    while True:
        _connectivity["internet"] = _ping_host("8.8.8.8")
        if API_URL:
            try:
                _hostname = urlparse(API_URL).hostname or API_URL
                _connectivity["api"] = _ping_host(_hostname)
            except Exception:
                _connectivity["api"] = False
        time.sleep(30)

threading.Thread(target=_refresh_connectivity, daemon=True).start()


def _poll_performance():
    _iter = 0
    while True:
        try:
            cpu, ram, _ = _perf_info()
            gpu, _ = _gpu_info()
            now_str = time.strftime("%H:%M:%S")
            _cpu_hist.append(cpu if isinstance(cpu, (int, float)) else 0)
            _ram_hist.append(ram if isinstance(ram, (int, float)) else 0)
            _gpu_hist.append(gpu if isinstance(gpu, (int, float)) else 0)
            _time_hist.append(now_str)
            _iter += 1
            # Print first few readings so startup issues are always visible
            if _iter <= 3:
                print(f"[perf] iter={_iter}  cpu={cpu:.1f}%  ram={ram:.1f}%  gpu={gpu:.1f}%", flush=True)
        except Exception as _e:
            _log.error("[perf] poll error (thread stays alive): %s", _e, exc_info=True)
        time.sleep(2)

threading.Thread(target=_poll_performance, daemon=True, name="perf-poll").start()


# ── Smart Notification Engine ─────────────────────────────────────────────────

def _generate_notifications() -> list:
    """Evaluate current state and return a list of notification dicts."""
    notifs = []
    cpu, ram, _ = _perf_info()
    disk_pct, _, _, _ = _disk_info()

    # Camera offline
    offline_cams = [cid for cid, s in _camera_status.items() if not s.get('active')]
    if offline_cams:
        cam_list = ", ".join(offline_cams)
        notifs.append({
            "id": "cam_offline",
            "level": "error",
            "icon": "📷",
            "title": t("notifications.cam_offline.title"),
            "message": t("notifications.cam_offline.message", cam_list=cam_list),
            "action_label": t("notifications.cam_offline.action"),
            "action_url": "/action/restart",
            "action_method": "POST",
        })

    # Sync overdue (>1 hour)
    if _last_upload_ts:
        delta = datetime.now() - _last_upload_ts
        if delta.total_seconds() > 3600:
            mins = int(delta.total_seconds() // 60)
            notifs.append({
                "id": "sync_overdue",
                "level": "warning",
                "icon": "📤",
                "title": t("notifications.sync_overdue.title"),
                "message": t("notifications.sync_overdue.message", mins=mins),
                "action_label": t("notifications.sync_overdue.action"),
                "action_url": "/action/upload",
                "action_method": "POST",
            })
    elif _last_upload_ts is None and _force_upload_fn:
        notifs.append({
            "id": "sync_never",
            "level": "warning",
            "icon": "📤",
            "title": t("notifications.sync_never.title"),
            "message": t("notifications.sync_never.message"),
            "action_label": t("notifications.sync_never.action"),
            "action_url": "/action/upload",
            "action_method": "POST",
        })

    # High CPU
    if isinstance(cpu, (int, float)) and cpu > 85:
        notifs.append({
            "id": "high_cpu",
            "level": "warning",
            "icon": "🔥",
            "title": t("notifications.high_cpu.title"),
            "message": t("notifications.high_cpu.message", cpu=f"{cpu:.0f}"),
            "action_label": None,
            "action_url": None,
            "action_method": None,
        })

    # Disk nearly full
    if disk_pct > 85:
        notifs.append({
            "id": "disk_full",
            "level": "error" if disk_pct > 92 else "warning",
            "icon": "💾",
            "title": t("notifications.disk_full.title"),
            "message": t("notifications.disk_full.message", pct=disk_pct),
            "action_label": None,
            "action_url": None,
            "action_method": None,
        })

    # Filter dismissed
    notifs = [n for n in notifs if n["id"] not in _dismissed_notifications]

    # Recent entries (info notification) — generates a unique ID tied to the 2-min window
    recent = _recent_entries(2)
    if recent:
        count = len(recent)
        window_id = int(time.time() // 120)
        nid = f"recent_entry_{window_id}"
        if nid not in _dismissed_notifications:
            notifs.append({
                "id": nid,
                "level": "info",
                "icon": "👋",
                "title": t("notifications.recent_visitors.title"),
                "message": t("notifications.recent_visitors.message", count=count),
                "action_label": None,
                "action_url": None,
                "action_method": None,
            })

    # Retail Operations Alerts
    tracks = _retail_data.get("tracks", {})
    if tracks:
        # Checkout Queue Alert
        checkout_count = sum(1 for t in tracks.values() if t.get('zone') and ('checkout' in t['zone'].lower() or 'caixa' in t['zone'].lower()))
        if checkout_count >= 3:
            notifs.append({
                "id": "queue_alert",
                "level": "warning",
                "icon": "⏳",
                "title": t("notifications.queue_alert.title"),
                "message": t("notifications.queue_alert.message", count=checkout_count),
                "action_label": None,
                "action_url": None,
                "action_method": None,
            })
            
        # High Dwell Time Warning (>1500 frames is roughly 1 minute at ~25fps)
        long_dwellers = sum(1 for t in tracks.values() if t.get('detection_count', 0) > 1500)
        if long_dwellers > 0:
            notifs.append({
                "id": "dwell_alert",
                "level": "info",
                "icon": "⏱️",
                "title": t("notifications.dwell_alert.title"),
                "message": t("notifications.dwell_alert.message", count=long_dwellers),
                "action_label": None,
                "action_url": None,
                "action_method": None,
            })
            
        # Demographic Insights
        teens = sum(1 for tx in tracks.values() if tx.get('age') == 'teen')
        if teens >= 3 and teens / len(tracks) >= 0.4:
            nid = f"demo_teens_{int(time.time() // 300)}"  # Unique every 5 mins
            if nid not in _dismissed_notifications:
                notifs.append({
                    "id": nid,
                    "level": "info",
                    "icon": "👥",
                    "title": t("notifications.demo_teens.title"),
                    "message": t("notifications.demo_teens.message"),
                    "action_label": None,
                    "action_url": None,
                    "action_method": None,
                })

    # All-clear
    if not notifs:
        notifs.append({
            "id": "all_clear",
            "level": "ok",
            "icon": "✅",
            "title": t("notifications.all_clear.title"),
            "message": t("notifications.all_clear.message"),
            "action_label": None,
            "action_url": None,
            "action_method": None,
        })

    return notifs


# ── HTML Page ──────────────────────────────────────────────────────────────────

_PAGE = ""


# ── Flask Routes ───────────────────────────────────────────────────────────────

@app.route("/")
@requires_auth
def index():
    disk_pct, disk_used, disk_total, disk_class = _disk_info()
    cpu_pct = _cpu_hist[-1]
    ram_pct = _ram_hist[-1]
    gpu_pct = _gpu_hist[-1]
    temp_c = _read_temp()

    # Classify CPU/RAM/GPU for progress bar colour
    cpu_class = "ok" if cpu_pct < 70 else ("warn" if cpu_pct < 85 else "err")
    ram_class = "ok" if ram_pct < 70 else ("warn" if ram_pct < 85 else "err")
    gpu_class = "ok" if gpu_pct < 70 else ("warn" if gpu_pct < 85 else "err")

    flash_msg = request.args.get("ok", "")
    error_msg = request.args.get("err", "")

    last_upload = (
        _last_upload_ts.strftime("%H:%M:%S")
        if _last_upload_ts else t("header.never")
    )

    # Read from cache — updated every 30 s by _refresh_connectivity() thread.
    # Never call _ping_host() synchronously here; it blocks for seconds.
    ping_internet = _connectivity["internet"]
    ping_api      = _connectivity["api"]

    # Dynamically determine log path
    log_path = "tracking.log" if os.path.exists("tracking.log") else "/var/log/nort/tracking.log"

    active_cameras = sum(1 for s in _camera_status.values() if s.get('active'))
    total_cameras  = len(_camera_status)

    from flask import render_template
    return render_template(
        "dashboard.html",
        device_id=DEVICE_ID,
        client_id=CLIENT_ID,
        store_id=STORE_ID,
        api_url=API_URL,
        uptime=_format_uptime(int(time.monotonic() - _start_time)),
        last_upload=last_upload,
        cpu_hist=json.dumps(list(_cpu_hist)),
        ram_hist=json.dumps(list(_ram_hist)),
        gpu_hist=json.dumps(list(_gpu_hist)),
        time_hist=json.dumps(list(_time_hist)),
        temp_c=temp_c,
        cpu_pct=round(cpu_pct, 1),
        ram_pct=round(ram_pct, 1),
        gpu_pct=round(gpu_pct, 1),
        cpu_class=cpu_class,
        ram_class=ram_class,
        gpu_class=gpu_class,
        disk_pct=disk_pct,
        disk_used=disk_used,
        disk_total=disk_total,
        disk_class=disk_class,
        cameras=_camera_status,
        logs=_read_tail(),
        flash_msg=flash_msg,
        error_msg=error_msg,
        ping_internet=ping_internet,
        ping_api=ping_api,
        active_cameras=active_cameras,
        total_cameras=total_cameras,
        current_occupancy=_retail_data.get("occupancy", 0),
        visitors_today=_today_visitor_count(),
        peak_hour=_peak_hour_today(),
        notifications=_generate_notifications(),
    )


@app.route("/home")
@requires_auth
def home_page():
    disk_pct, disk_used, disk_total, disk_class = _disk_info()
    cpu_pct = _cpu_hist[-1]
    ram_pct = _ram_hist[-1]
    temp_c = _read_temp()
    active_cameras = sum(1 for s in _camera_status.values() if s.get('active'))
    total_cameras  = len(_camera_status)
    from flask import render_template
    return render_template(
        "home.html",
        device_id=DEVICE_ID,
        client_id=CLIENT_ID,
        store_id=STORE_ID,
        uptime=_format_uptime(int(time.monotonic() - _start_time)),
        cpu_pct=round(cpu_pct, 1),
        ram_pct=round(_ram_hist[-1], 1),
        gpu_pct=round(_gpu_hist[-1], 1),
        disk_pct=disk_pct,
        active_cameras=active_cameras,
        total_cameras=total_cameras,
        current_occupancy=_retail_data.get("occupancy", 0),
        cloud_sync_ok=True,
    )


@app.route("/profile")
@requires_auth
def profile_page():
    device_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "device.json")
    config_data = {}
    if os.path.exists(device_json_path):
        try:
            with open(device_json_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            logging.error(f"Failed to read device.json for profile route: {e}")
            
    from flask import render_template
    return render_template(
        "profile.html",
        device_id=DEVICE_ID,
        store_id=STORE_ID,
        config=config_data
    )



@app.route("/api/status")
@requires_auth
def api_status():
    """Live JSON endpoint — polled every 10s by the frontend."""
    cpu_pct = _cpu_hist[-1]
    ram_pct = _ram_hist[-1]
    gpu_pct = _gpu_hist[-1]
    last_upload = (
        _last_upload_ts.strftime("%H:%M:%S")
        if _last_upload_ts else t("header.never")
    )
    active_cameras = sum(1 for s in _camera_status.values() if s.get('active'))
    total_cameras  = len(_camera_status)

    return jsonify({
        "visitors_today": _today_visitor_count(),
        "peak_hour": _peak_hour_today(),
        "active_cameras": active_cameras,
        "total_cameras": total_cameras,
        "last_upload": last_upload,
        "current_occupancy": _retail_data.get("occupancy", 0),
        "retail_data": _retail_data,
        "cpu_pct": round(cpu_pct, 1),
        "ram_pct": round(ram_pct, 1),
        "gpu_pct": round(gpu_pct, 1),
        "cpu_hist": list(_cpu_hist),
        "ram_hist": list(_ram_hist),
        "gpu_hist": list(_gpu_hist),
        "time_hist": list(_time_hist),
        "uptime_seconds": int(time.monotonic() - _start_time),
    })


@app.route("/api/notifications/<notif_id>/dismiss", methods=["POST"])
@requires_auth
def dismiss_notification(notif_id: str):
    """Mark a notification as dismissed (persists until restart)."""
    _dismissed_notifications.add(notif_id)
    return jsonify({"ok": True})


@app.route("/healthz")
def healthz():
    """Used by the cloud heartbeat checker."""
    return jsonify({
        "status": "ok",
        "device_id": DEVICE_ID,
        "client_id": CLIENT_ID,
        "store_id": STORE_ID,
        "uptime_seconds": int(time.monotonic() - _start_time),
    })


@app.route("/action/upload", methods=["POST"])
@requires_auth
def manual_upload():
    if _force_upload_fn:
        try:
            _force_upload_fn()
            return redirect("/?ok=Force+sync+triggered.")
        except Exception as e:
            return redirect(f"/?err=Sync+failed:+{e}")
    else:
        return redirect("/?err=Sync+not+available+yet.")


@app.route("/action/restart", methods=["POST"])
@requires_auth
def restart_pipeline():
    if _restart_fn:
        try:
            _restart_fn()
            return redirect("/?ok=Restart+command+issued.+Please+wait+~30s.")
        except Exception as e:
            return redirect(f"/?err=Restart+failed:+{e}")
    else:
        return redirect("/?err=Restart+not+available+yet.")


# ── MJPEG Live Stream ─────────────────────────────────────────────────────────

def _enabled_camera_ids() -> list:
    """Return list of camera IDs where enabled=True (or key missing) from cameras.json."""
    import json as _json, os as _os
    _path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "cameras.json")
    try:
        with open(_path, 'r', encoding='utf-8') as _f:
            _cams = _json.load(_f)
        return [cid for cid, cam in _cams.items() if cam.get("enabled", True)]
    except Exception:
        return list(_camera_status.keys())


def _mjpeg_gen(camera_id: str):
    """Generator that yields MJPEG frames for a given camera."""
    # gevent.sleep() is a cooperative yield — other greenlets (page requests)
    # run during the sleep.  stdlib time.sleep() without monkey-patching blocks
    # the entire gevent hub OS-thread, making every page navigation wait behind
    # all active MJPEG streams.  Always use gevent.sleep here.
    try:
        from gevent import sleep as _gsleep
    except ImportError:
        import time as _t; _gsleep = _t.sleep

    _BLANK = None
    while True:
        frame_bytes = _latest_frames.get(camera_id)
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            # Send a small grey placeholder if no frame available yet
            if _BLANK is None:
                try:
                    import cv2 as _cv2, numpy as _np
                    grey = _np.full((60, 120, 3), 40, dtype=_np.uint8)
                    _cv2.putText(grey, "No feed", (4, 38), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                    _, buf = _cv2.imencode('.jpg', grey)
                    _BLANK = buf.tobytes()
                except Exception:
                    _BLANK = b""
            if _BLANK:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + _BLANK + b"\r\n"
                )
        _gsleep(0.1)    # ~10 fps — cooperative yield so other greenlets can run


@app.route("/stream/<camera_id>")
@requires_auth
def live_stream(camera_id: str):
    if camera_id not in _enabled_camera_ids():
        from flask import abort
        abort(404)
    return Response(
        _mjpeg_gen(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_raw_gen(camera_id: str):
    """Generator that yields MJPEG RAW frames for a given camera."""
    try:
        from gevent import sleep as _gsleep
    except ImportError:
        import time as _t; _gsleep = _t.sleep

    _BLANK = None
    while True:
        frame_bytes = _latest_raw_frames.get(camera_id)
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            if _BLANK is None:
                try:
                    import cv2 as _cv2, numpy as _np
                    grey = _np.full((60, 120, 3), 40, dtype=_np.uint8)
                    _cv2.putText(grey, "No feed", (4, 38), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                    _, buf = _cv2.imencode('.jpg', grey)
                    _BLANK = buf.tobytes()
                except Exception:
                    _BLANK = b""
            if _BLANK:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + _BLANK + b"\r\n"
                )
        _gsleep(0.1)    # ~10 fps — cooperative yield so other greenlets can run


@app.route("/stream_raw/<camera_id>")
@requires_auth
def live_stream_raw(camera_id: str):
    return Response(
        _mjpeg_raw_gen(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Zones API ─────────────────────────────────────────────────────────────────

@app.route("/api/zones")
@requires_auth
def api_zones():
    """Return zone polygons for a camera, normalized to [0,1] for canvas rendering."""
    import json as _json, os as _os
    camera_id = request.args.get("camera", "")
    # Determine resolution for normalization (default 1920x1080)
    W, H = 1920, 1080

    # Load zones config - try both locations
    zone_paths = [
        _os.path.join(_os.path.dirname(__file__), "..", "zones_per_camera.json"),
        _os.path.join(_os.path.dirname(__file__), "..", "data_store", "zones_per_camera.json"),
    ]
    zones_raw = []
    for zp in zone_paths:
        zp = _os.path.normpath(zp)
        if _os.path.exists(zp):
            try:
                with open(zp, "r") as f:
                    data = _json.load(f)
                # Navigate: any client -> any store -> camera_id
                for client_data in data.values():
                    for store_data in client_data.values():
                        cam_data = store_data.get(camera_id, {})
                        zones_raw = cam_data.get("zones", [])
                        break
                    break
            except Exception:
                pass
            break

    # Normalize polygon_vertices to [0,1]
    zones_out = []
    for z in zones_raw:
        verts = z.get("polygon_vertices", [])
        name = z.get("sector_name", "")
        pts = [[round(x / W, 4), round(y / H, 4)] for x, y in verts]
        if len(pts) >= 3:
            zones_out.append({"name": name, "points": pts})

    return jsonify({"zones": zones_out, "camera": camera_id})


# ── Heatmap / Paths API ───────────────────────────────────────────────────────

@app.route("/api/streams")
@requires_auth
def api_heatmap():
    """
    Returns a transparent PNG image for rendering a professional heatmap.
    Uses OpenCV COLORMAP_TURBO and applies a large Gaussian blur for smooth density.
    """
    from data import spatial_logger as _sl
    from datetime import datetime as _dt
    import numpy as _np
    import cv2 as _cv2

    camera_id = request.args.get("camera", "")
    gender    = request.args.get("gender", "all")
    age_group = request.args.get("age",    "all")

    from_ts = to_ts = None
    try:
        if request.args.get("from"):
            from_ts = int(_dt.fromisoformat(request.args["from"]).timestamp() * 1000)
        if request.args.get("to"):
            to_ts = int(_dt.fromisoformat(request.args["to"]).timestamp() * 1000)
    except Exception:
        pass

    # Run SQLite query in a thread — sqlite3 is a blocking C extension and
    # calling it from a gevent greenlet freezes the entire event loop.
    points = _run_in_thread(_sl.query_heatmap, camera_id, gender, age_group, from_ts, to_ts)

    # 16:9 density grid for professional rendering
    bins_w, bins_h = 320, 180
    grid = _np.zeros((bins_h, bins_w), dtype=_np.float32)

    for cx, cy in points:
        bx = min(int(cx * bins_w), bins_w - 1)
        by = min(int(cy * bins_h), bins_h - 1)
        grid[by, bx] += 1.0

    if grid.max() > 0:
        # High radius blur for smooth splatting
        grid = _cv2.GaussianBlur(grid, (25, 25), 0)
        grid = (grid / grid.max() * 255).astype(_np.uint8)

        # Apply professional TURBO colormap
        heatmap_c = _cv2.applyColorMap(grid, _cv2.COLORMAP_TURBO)

        # Build alpha transparency map based on density
        alpha = grid.copy()
        alpha[alpha < 3] = 0  # Filter lowest noise
        alpha = (alpha * 0.85).astype(_np.uint8)  # Max 85% opacity

        rgba = _cv2.cvtColor(heatmap_c, _cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        _, buf = _cv2.imencode('.png', rgba)
    else:
        # 1x1 transparent PNG if empty
        empty = _np.zeros((1, 1, 4), dtype=_np.uint8)
        _, buf = _cv2.imencode('.png', empty)

    resp = Response(buf.tobytes(), mimetype='image/png')
    resp.headers['X-Total-Points'] = str(len(points))
    return resp


@app.route("/api/paths")
@requires_auth
def api_paths():
    """Returns up to 5 common track paths as lists of [cx, cy] waypoints."""
    from data import spatial_logger as _sl
    from datetime import datetime as _dt

    camera_id = request.args.get("camera", "")
    gender    = request.args.get("gender", "all")
    age_group = request.args.get("age",    "all")
    
    from_ts = to_ts = None
    try:
        if request.args.get("from"):
            from_ts = int(_dt.fromisoformat(request.args["from"]).timestamp() * 1000)
        if request.args.get("to"):
            to_ts = int(_dt.fromisoformat(request.args["to"]).timestamp() * 1000)
    except Exception:
        pass

    # Run in thread — sqlite3 C extension blocks gevent event loop if called from greenlet
    paths = _run_in_thread(_sl.query_paths, camera_id, gender, age_group, from_ts, to_ts)
    return jsonify({"paths": [[[round(x, 3), round(y, 3)] for x, y in p] for p in paths]})


@app.route("/api/camera_snapshot/<camera_id>")
@requires_auth
def api_camera_snapshot(camera_id):
    """
    Returns a JPEG of the latest camera frame.
    Support parameters:
      - rendered=1: include bounding boxes and zones.
      - heatmap=1: overlay real-time heatmap.
      - paths=1: overlay common trajectories.
    """
    from flask import request as _req
    import cv2 as _cv2
    import numpy as _np
    
    do_render  = _req.args.get("rendered") == "1"
    do_heatmap = _req.args.get("heatmap")  == "1"
    do_paths   = _req.args.get("paths")    == "1"

    # 1. Get base frame
    raw_bytes = _latest_raw_frames.get(camera_id)
    ann_bytes = _latest_frames.get(camera_id)
    
    # Use annotated bytes if rendered=1 is requested and available, otherwise raw
    source_bytes = (ann_bytes if (do_render and ann_bytes) else raw_bytes) or raw_bytes or ann_bytes
    
    if not source_bytes:
        return "Camera offline or not found", 404
        
    # If no advanced baking (heatmap/paths) is requested, just return the existing bytes
    if not (do_heatmap or do_paths):
        return Response(source_bytes, mimetype='image/jpeg')

    # 2. Advanced Baking (Requires decoding)
    nparr = _np.frombuffer(source_bytes, _np.uint8)
    img = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
    if img is None:
        return Response(source_bytes, mimetype='image/jpeg') # fallback
    
    h, w = img.shape[:2]

    # 2a. Add Heatmap
    if do_heatmap:
        from data import spatial_logger as _sl
        points = _run_in_thread(_sl.query_heatmap, camera_id, "all", "all")  # Latest data as fallback
        if points:
            # Re-use logic from api_heatmap but bake it directly
            grid = _np.zeros((180, 320), dtype=_np.float32)
            for cx, cy in points:
                bx = min(int(cx * 320), 319)
                by = min(int(cy * 180), 179)
                grid[by, bx] += 1.0
            if grid.max() > 0:
                grid = _cv2.GaussianBlur(grid, (15, 15), 0)
                grid = (grid / grid.max() * 255).astype(_np.uint8)
                hm_c = _cv2.applyColorMap(grid, _cv2.COLORMAP_TURBO)
                hm_large = _cv2.resize(hm_c, (w, h))
                # Alpha blend
                mask = _cv2.resize(grid, (w, h))
                mask = (mask.astype(_np.float32) / 255.0 * 0.6) # 60% max opacity
                for c in range(3):
                    img[:,:,c] = (img[:,:,c] * (1-mask) + hm_large[:,:,c] * mask).astype(_np.uint8)

    # 2b. Add Trajectories (Generic placeholder for now as api_paths is complex)
    if do_paths:
        # Drawing a simplified version of active trajectories if available
        # Implementation depends on spatial_logger having a 'get_active_tracks' or similar
        pass

    _, buf = _cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route("/api/config/zones/<camera_id>", methods=["GET", "POST"])
@requires_auth
def api_config_zones(camera_id):
    """GET or POST the polygon zones config for a given camera."""
    from system import config
    from core.polygon_zone import load_camera_config, save_camera_zones

    if request.method == "GET":
        camera_config = load_camera_config(config.POLYGON_POINTS_FILE, config.CLIENT_ID, config.STORE_ID, camera_id)
        return jsonify(camera_config.get("zones", []))
        
    elif request.method == "POST":
        try:
            data = request.get_json()
            if data is None:
                return jsonify({"error": "Invalid JSON"}), 400
            if not isinstance(data, list):
                return jsonify({"error": "Expected a list of zones"}), 400

            # Validate polygon structure for each zone
            for zone in data:
                if "points" not in zone and "polygon" not in zone and "coordinates" not in zone and "polygon_vertices" not in zone:
                    return jsonify({"error": f"Zone missing polygon points: {zone.get('name', zone.get('sector_name', 'unknown'))}"}), 400
                points = zone.get("points") or zone.get("polygon") or zone.get("coordinates") or zone.get("polygon_vertices", [])
                if not isinstance(points, list) or len(points) < 3:
                    return jsonify({"error": "Zone polygon must have at least 3 points"}), 400

            success = save_camera_zones(
                filename=config.POLYGON_POINTS_FILE,
                client_id=config.CLIENT_ID,
                store_id=config.STORE_ID,
                camera_id=camera_id,
                zones=data,
                camera_type="standard_camera"  # Keep standard by default unless explicitly changed
            )
            # H6-fix: missing return caused Flask to return None → HTTP 500
            if success:
                return jsonify({"status": "ok", "camera_id": camera_id, "zones_saved": len(data)}), 200
            else:
                return jsonify({"error": "Failed to save zones to disk"}), 500

        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route("/api/config/homography/<camera_id>", methods=["GET", "POST"])
@requires_auth
def api_config_homography(camera_id):
    """GET or POST the homography calibration points for a given camera."""
    from core.homography_manager import load_all_homographies, save_camera_homography
    
    if request.method == "GET":
        data = load_all_homographies()
        return jsonify(data.get(camera_id, {}))
        
    elif request.method == "POST":
        data = request.json
        if not data or "camera_pts" not in data or "world_pts" not in data:
            return jsonify({"error": "Missing camera_pts or world_pts"}), 400
            
        H, ok = save_camera_homography(camera_id, data["camera_pts"], data["world_pts"])
        if ok:
            return jsonify({"status": "success", "H": H})
        return jsonify({"error": "Failed to compute or save homography matrix"}), 400

@app.route("/api/camera_snapshot_warped/<camera_id>")
@requires_auth
def api_camera_snapshot_warped(camera_id):
    """Returns a JPEG of the latest camera frame warped to top-down."""
    import cv2 as _cv2, numpy as _np, os
    from core.homography_manager import load_camera_homography
    
    H = load_camera_homography(camera_id)
    if H is None:
        return "Homography not configured for this camera", 404
        
    use_clean = request.args.get('use_clean', '0') == '1'
    frame = None
    
    if use_clean:
        raw_clean_path = os.path.join(os.path.dirname(__file__), "static", f"clean_bg_raw_{camera_id}.jpg")
        if os.path.exists(raw_clean_path):
            frame = _cv2.imread(raw_clean_path)
            
    if frame is None:
        frame_bytes = _latest_raw_frames.get(camera_id) or _latest_frames.get(camera_id)
        if not frame_bytes:
            return "Camera offline or no frame", 404
        np_arr = _np.frombuffer(frame_bytes, _np.uint8)
        frame = _cv2.imdecode(np_arr, _cv2.IMREAD_COLOR)
    
    from core.homography_manager import load_all_homographies
    h_data = load_all_homographies().get(camera_id, {})
    world_pts = h_data.get("world_pts", [])
    if len(world_pts) < 4:
        return "Invalid world points", 400
        
    pts = _np.array(world_pts, dtype=_np.float32)
    min_x, max_x = _np.min(pts[:,0]), _np.max(pts[:,0])
    min_y, max_y = _np.min(pts[:,1]), _np.max(pts[:,1])
    
    # Add 20% padding
    pad_x = (max_x - min_x) * 0.2
    pad_y = (max_y - min_y) * 0.2
    w_min_x, w_max_x = min_x - pad_x, max_x + pad_x
    w_min_y, w_max_y = min_y - pad_y, max_y + pad_y
    
    target_w = 800
    scale = target_w / max(1, w_max_x - w_min_x)
    target_h = int(max(1, w_max_y - w_min_y) * scale)
    
    M = _np.array([
        [scale, 0, -w_min_x * scale],
        [0, scale, -w_min_y * scale],
        [0, 0, 1]
    ], dtype=_np.float32)
    
    H_target = M @ H
    warped = _cv2.warpPerspective(frame, H_target, (target_w, target_h))
    
    ok, buf = _cv2.imencode('.jpg', warped, [_cv2.IMWRITE_JPEG_QUALITY, 80])
    resp = Response(buf.tobytes(), mimetype='image/jpeg')
    resp.headers['X-World-Bounds'] = f"{w_min_x},{w_min_y},{w_max_x},{w_max_y}"
    return resp

def _cam_has_detections(camera_id):
    """Check if any active tracks belong to this camera."""
    tracks = _retail_data.get("tracks", {})
    for tid, attrs in tracks.items():
        cam = attrs.get("camera_id", "")
        if cam == camera_id:
            return True
    return False

def _compute_median_bg_thread(camera_id, H, world_pts, bg_path):
    """
    Capture clean background frames for a camera by ONLY grabbing frames
    when zero people are detected in that camera's feed.
    Falls back to standard median if not enough clean frames are found.
    """
    import cv2 as _cv2, numpy as _np, time as _time
    logger = logging.getLogger("nort.admin")
    
    target_clean = 15       # ideal number of detection-free frames
    min_frames = 5          # minimum to produce a usable background
    timeout_seconds = 120   # wait up to 2 minutes for empty moments
    poll_interval = 0.5     # check every 500ms
    
    clean_frames = []
    fallback_frames = []    # all frames regardless of detections
    t_start = _time.monotonic()
    
    logger.info(f"[CleanBG] Starting capture for {camera_id} (waiting for empty frames, timeout={timeout_seconds}s)")
    
    while _time.monotonic() - t_start < timeout_seconds:
        raw = _latest_raw_frames.get(camera_id)
        if not raw:
            _time.sleep(poll_interval)
            continue
            
        arr = _np.frombuffer(raw, _np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if img is None:
            _time.sleep(poll_interval)
            continue
        
        # Skip duplicate frames
        is_dup = False
        if len(clean_frames) > 0 and _np.array_equal(img, clean_frames[-1]):
            is_dup = True
        elif len(fallback_frames) > 0 and _np.array_equal(img, fallback_frames[-1]):
            is_dup = True
            
        if not is_dup:
            # Always collect for fallback median
            if len(fallback_frames) < 60:
                fallback_frames.append(img.copy())
            
            # Only collect for clean stack if zero detections on this camera
            if not _cam_has_detections(camera_id):
                clean_frames.append(img.copy())
                logger.debug(f"[CleanBG] {camera_id}: captured clean frame {len(clean_frames)}/{target_clean}")
                if len(clean_frames) >= target_clean:
                    break
        
        _time.sleep(poll_interval)
    
    # Decide which stack to use
    if len(clean_frames) >= min_frames:
        stack_frames = clean_frames
        logger.info(f"[CleanBG] {camera_id}: using {len(clean_frames)} detection-free frames ✓")
    elif len(fallback_frames) >= 3:
        stack_frames = fallback_frames
        logger.warning(f"[CleanBG] {camera_id}: only {len(clean_frames)} clean frames found, falling back to median of {len(fallback_frames)} frames")
    else:
        logger.error(f"[CleanBG] {camera_id}: not enough frames captured ({len(fallback_frames)}), aborting")
        return
    
    stack = _np.stack(stack_frames, axis=0)
    median_frame = _np.median(stack, axis=0).astype(_np.uint8)
    
    # Save the RAW unwarped median background for dynamic re-warping later
    import os
    raw_path = os.path.join(os.path.dirname(bg_path), f"clean_bg_raw_{camera_id}.jpg")
    _cv2.imwrite(raw_path, median_frame, [_cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Warp to top-down
    pts = _np.array(world_pts, dtype=_np.float32)
    min_x, max_x = _np.min(pts[:,0]), _np.max(pts[:,0])
    min_y, max_y = _np.min(pts[:,1]), _np.max(pts[:,1])
    pad_x, pad_y = (max_x - min_x) * 0.2, (max_y - min_y) * 0.2
    w_min_x, w_max_x = min_x - pad_x, max_x + pad_x
    w_min_y, w_max_y = min_y - pad_y, max_y + pad_y
    target_w = 800
    scale = target_w / max(1, w_max_x - w_min_x)
    target_h = int(max(1, w_max_y - w_min_y) * scale)
    M = _np.array([[scale, 0, -w_min_x * scale], [0, scale, -w_min_y * scale], [0, 0, 1]], dtype=_np.float32)
    warped = _cv2.warpPerspective(median_frame, M @ H, (target_w, target_h))
    
    _cv2.imwrite(bg_path, warped, [_cv2.IMWRITE_JPEG_QUALITY, 85])
    logger.info(f"[CleanBG] {camera_id}: background saved successfully ✓")

@app.route("/api/camera_clean_bg/<camera_id>", methods=["POST"])
@requires_auth
def api_camera_clean_bg(camera_id):
    import threading as _threading, os
    from core.homography_manager import load_camera_homography, load_all_homographies
    
    H = load_camera_homography(camera_id)
    if H is None:
        return jsonify({"error": "Homography not configured"}), 404

    h_data = load_all_homographies().get(camera_id, {})
    world_pts = h_data.get("world_pts", [])
    if len(world_pts) < 4:
        return jsonify({"error": "Need at least 4 world points"}), 400

    bg_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(bg_dir, exist_ok=True)
    bg_path = os.path.join(bg_dir, f"clean_bg_{camera_id}.jpg")

    # Start the median extraction in the background so it doesn't freeze Flask UI
    t = _threading.Thread(target=_compute_median_bg_thread, args=(camera_id, H, world_pts, bg_path))
    t.daemon = True
    t.start()

    return jsonify({
        "ok": True,
        "message": "Median extraction started in background",
        "eta": 15
    })

@app.route("/api/floorplan/bg", methods=["GET", "POST", "DELETE"])
@requires_auth
def manage_floorplan_bg():
    import json, time as _time
    bg_path = os.path.join(os.path.dirname(__file__), "static", "floorplan_bg.jpg")
    cfg_path = os.path.join(os.path.dirname(__file__), "static", "floorplan_bg.json")
    
    if request.method == "GET":
        if os.path.exists(bg_path):
            cfg = {"scale":1,"angle":0,"tx":0,"ty":0}
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, 'r') as f: cfg = json.load(f)
                except: pass
            return jsonify({"exists": True, "url": "/static/floorplan_bg.jpg?t=" + str(os.path.getmtime(bg_path)), "config": cfg})
        return jsonify({"exists": False})
        
    if request.method == "DELETE":
        if os.path.exists(bg_path):
            try: os.remove(bg_path)
            except: pass
        if os.path.exists(cfg_path):
            try: os.remove(cfg_path)
            except: pass
        return jsonify({"ok": True})
        
    if request.method == "POST":
        if request.is_json:
            data = request.json
            with open(cfg_path, 'w') as f: json.dump(data, f)
            return jsonify({"ok": True})
            
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        os.makedirs(os.path.dirname(bg_path), exist_ok=True)
        try:
            from PIL import Image
            img = Image.open(file.stream).convert('RGB')
            img.save(bg_path, "JPEG", quality=85)
            with open(cfg_path, 'w') as f: json.dump({"scale":1,"angle":0,"tx":0,"ty":0}, f)
            return jsonify({"ok": True, "url": "/static/floorplan_bg.jpg?t=" + str(_time.time())})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

BLINDSPOTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "blindspots.json")

@app.route("/api/floorplan/blindspots", methods=["GET", "POST", "DELETE"])
@requires_auth
def manage_blindspots():
    import json
    if request.method == "GET":
        if os.path.exists(BLINDSPOTS_FILE):
            try:
                with open(BLINDSPOTS_FILE, "r") as f:
                    return jsonify(json.load(f))
            except:
                return jsonify([])
        return jsonify([])
        
    if request.method == "POST":
        data = request.json
        os.makedirs(os.path.dirname(BLINDSPOTS_FILE), exist_ok=True)
        try:
            with open(BLINDSPOTS_FILE, "w") as f:
                json.dump(data, f)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    if request.method == "DELETE":
        if os.path.exists(BLINDSPOTS_FILE):
            try:
                os.remove(BLINDSPOTS_FILE)
            except:
                pass
        return jsonify({"ok": True})

# ── Camera Management ──────────────────────────────────────────────────────────

_needs_restart = False  # Set when camera config changes (UI shows restart banner)

@app.route("/cameras")
@requires_auth
def cameras_page():
    from flask import render_template
    from system import config
    cameras = config.load_cameras()
    # Merge live status
    for cam_id in cameras:
        status = _camera_status.get(cam_id, {})
        cameras[cam_id]["active"] = status.get("active", False)
        cameras[cam_id]["fps"] = status.get("fps", 0)
    return render_template("cameras.html",
        cameras=cameras,
        needs_restart=_needs_restart,
        device_id=DEVICE_ID,
        store_id=STORE_ID,
    )


@app.route("/api/cameras", methods=["GET"])
@requires_auth
def api_cameras_list():
    from system import config
    cameras = config.load_cameras()
    for cam_id in cameras:
        status = _camera_status.get(cam_id, {})
        cameras[cam_id]["active"] = status.get("active", False)
        cameras[cam_id]["fps"] = status.get("fps", 0)
    return jsonify({"cameras": cameras, "needs_restart": _needs_restart})


@app.route("/api/cameras", methods=["POST"])
@requires_auth
def api_cameras_add():
    global _needs_restart
    from system import config
    data = request.json
    if not data or not data.get("source"):
        return jsonify({"error": "source is required"}), 400

    cameras = config.load_cameras()
    # Auto-generate camera_id
    existing_nums = []
    for k in cameras:
        try:
            existing_nums.append(int(k.split("_")[-1]))
        except ValueError:
            pass
    next_num = max(existing_nums, default=0) + 1
    cam_id = f"camera_{next_num}"

    cameras[cam_id] = {
        "name": data.get("name", f"Camera {next_num}"),
        "source": data["source"],
        "type": data.get("type", "rtsp"),
        "enabled": data.get("enabled", True),
    }
    config.save_cameras(cameras)
    _needs_restart = True
    return jsonify({"status": "ok", "camera_id": cam_id}), 201


@app.route("/api/cameras/<camera_id>", methods=["PUT"])
@requires_auth
def api_cameras_edit(camera_id):
    global _needs_restart
    from system import config
    data = request.json
    cameras = config.load_cameras()
    if camera_id not in cameras:
        return jsonify({"error": "Camera not found"}), 404

    if "name" in data:
        cameras[camera_id]["name"] = data["name"]
    if "source" in data:
        cameras[camera_id]["source"] = data["source"]
    if "type" in data:
        cameras[camera_id]["type"] = data["type"]
    if "enabled" in data:
        cameras[camera_id]["enabled"] = data["enabled"]

    config.save_cameras(cameras)
    _needs_restart = True
    return jsonify({"status": "ok"})


@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
@requires_auth
def api_cameras_delete(camera_id):
    global _needs_restart
    from system import config
    cameras = config.load_cameras()
    if camera_id not in cameras:
        return jsonify({"error": "Camera not found"}), 404

    del cameras[camera_id]
    config.save_cameras(cameras)
    _needs_restart = True
    return jsonify({"status": "ok"})


@app.route("/api/cameras/scan", methods=["POST"])
@requires_auth
def api_cameras_scan():
    """Scan the local subnet for IP cameras."""
    import concurrent.futures
    import subprocess
    import platform

    local_ip = _get_local_ip()
    base_ip = ".".join(local_ip.split(".")[:3]) + "."
    target_ports = [80, 554, 8000, 8080, 37777, 34567, 8899]
    discovered = []

    print(f"\n[Scanner] Starting network scan. Local IP: {local_ip}")
    print(f"[Scanner] Running ping sweep on {base_ip}1-254 to awake devices...")

    def ping_ip(i: int):
        ip = f"{base_ip}{i}"
        if ip == local_ip: return ip
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        subprocess.run(["ping", param, "1", "-w", "200", ip], capture_output=True)
        return ip

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        list(executor.map(ping_ip, range(1, 255)))

    arp_ips = set()
    try:
        out = subprocess.check_output(["arp", "-a"], text=True)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith(base_ip):
                ip_found = parts[0]
                if ip_found != local_ip and not ip_found.endswith(".255"):
                    arp_ips.add(ip_found)
    except Exception:
        pass

    if not arp_ips:
        arp_ips = {f"{base_ip}{i}" for i in range(1, 255) if f"{base_ip}{i}" != local_ip}

    def check_ip(ip: str):
        open_ports = []
        for port in target_ports:
            # 1.0s timeout ensures embedded devices don't get missed
            if _scan_port(ip, port, timeout=1.0):
                open_ports.append(port)
        
        if open_ports:
            device_type = "IP Camera / Web Device"
            if 37777 in open_ports:
                device_type = "Dahua NVR/DVR"
            elif 8000 in open_ports:
                device_type = "Hikvision NVR/IPC"
            elif 34567 in open_ports:
                device_type = "XMEye DVR/IPC"
            elif 8899 in open_ports:
                device_type = "ONVIF Protocol Camera"
            elif 554 in open_ports:
                device_type = "RTSP Camera / NVR"
            
            print(f"[Scanner] Found {device_type} at {ip}:{open_ports}")
            return {"ip": ip, "ports": open_ports, "device_type": device_type}
        return None

    print(f"[Scanner] Starting fast port-scan on {len(arp_ips)} active ARP IPs...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(check_ip, arp_ips)
        for res in results:
            if res:
                discovered.append(res)
                
    try:
        with open(r"C:\tmp\scan_debug.txt", "a") as f:
            import datetime
            f.write(f"\n[{datetime.datetime.now()}] ARP IPs ({len(arp_ips)}): {arp_ips}\n")
            f.write(f"Scanned Local IP: {local_ip}\n")
            f.write(f"Discovered ({len(discovered)}): {discovered}\n")
    except Exception:
        pass
        
    print(f"[Scanner] Finished! Discovered {len(discovered)} camera devices.")
    return jsonify({
        "success": True,
        "local_ip": local_ip,
        "devices": discovered
    })


@app.route("/api/cameras/test", methods=["POST"])
@requires_auth
def api_cameras_test():
    """Test a camera URL — try to grab one frame and return success + thumbnail."""
    import cv2 as _cv2
    data = request.json
    source = data.get("source", "")
    if not source:
        return jsonify({"success": False, "error": "No source URL provided"}), 400

    try:
        cap = _cv2.VideoCapture(source)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Cannot open stream. Check URL/credentials."})

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return jsonify({"success": False, "error": "Stream opened but no frame received."})

        # Generate thumbnail
        import base64
        small = _cv2.resize(frame, (320, 180))
        ok, buf = _cv2.imencode('.jpg', small, [_cv2.IMWRITE_JPEG_QUALITY, 70])
        thumb = base64.b64encode(buf.tobytes()).decode() if ok else None

        h, w = frame.shape[:2]
        return jsonify({
            "success": True,
            "resolution": f"{w}x{h}",
            "thumbnail": f"data:image/jpeg;base64,{thumb}" if thumb else None,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/restart", methods=["POST"])
@requires_auth
def api_restart_pipeline():
    """Restart the entire Python process to apply camera configs."""
    import threading
    import sys
    import os
    import time

    def do_restart():
        time.sleep(1.0)
        print("--- RESTARTING PIPELINE VIA LOCAL ADMIN API ---")
        os.execv(sys.executable, [sys.executable] + sys.argv)
        
    threading.Thread(target=do_restart, daemon=True).start()
    return """
    <html><body>
    <h3>Restarting Pipeline...</h3>
    <p>Please wait a few seconds and refresh the page.</p>
    <script>setTimeout(() => { window.location.href = '/cameras'; }, 5000);</script>
    </body></html>
    """

# ── Heatmap UI Page ───────────────────────────────────────────────────────────

_HEATMAP_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cameras — NORT Analytics</title>
<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
      /* Sidebar Dark Tokens */
      --side-bg: hsl(222, 28%, 8%);
      --side-border: hsl(222, 22%, 14%);
      --side-fg: hsl(213, 28%, 90%);
      --side-fg-dim: hsl(213, 18%, 52%);
      --side-hover-bg: hsl(222, 24%, 14%);
      --side-active-bg: rgba(59,130,246,0.1);
      --side-primary: hsl(217, 91%, 60%);
      --side-primary-dim: rgba(59,130,246,0.2);
      --side-logo: #ffffff;


      --bg: #0f1117;
      --card: #161b27;
      --card-hover: #1c2333;
      --border: #212c3f;
      --primary: #3b82f6;
      --primary-dim: rgba(59, 130, 246, 0.15);
      --primary-glow: rgba(59, 130, 246, 0.3);
      --fg: #f8fafc;
      --fg-dim: #94a3b8;
      --fg-muted: #64748b;
      --ok: #10b981;
      --warn: #f59e0b;
      --err: #ef4444;
      --sidebar-bg: #10151f;
      --radius: 8px;
      --radius-sm: 6px;
  }
  html[data-theme="light"] {
    --bg: #f0f3f7;
    --card: #ffffff;
    --card-hover: #e8edf4;
    --border: #cbd5e1;
    --fg: #1e2d40;
    --fg-dim: #475569;
    --fg-muted: #6b7fa0;
    --sidebar-bg: #e8edf4;
    --console-bg: #f1f5f9;
    /* Sidebar Light Tokens */
      --side-bg: hsl(215, 20%, 93%);
      --side-border: hsl(215, 18%, 84%);
      --side-fg: hsl(215, 30%, 16%);
      --side-fg-dim: hsl(215, 16%, 48%);
      --side-hover-bg: hsl(215, 16%, 88%);
      --side-active-bg: rgba(59,130,246,0.15);
      --side-primary: hsl(217, 91%, 60%);
      --side-primary-dim: rgba(59,130,246,0.25);
      --side-logo: #0f172a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Inter', system-ui, sans-serif;
    height: 100vh;
    display: flex;
    overflow: hidden;
  }
  .sidebar {
    transition: width 0.3s ease;
    position: relative;
  }
  .sidebar.collapsed {
    width: 80px;
  }
  .sidebar.collapsed .logo-wrapper,
  .sidebar.collapsed .nav-section-label,
  .sidebar.collapsed .status-badge,
  .sidebar.collapsed .live-clock,
  .sidebar.collapsed .nav-link > span {
    display: none !important;
  }
  .sidebar.collapsed .nav-link {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 12px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 500;
      font-size: 0.875rem;
      transition: all 0.15s;
      margin-bottom: 2px;
      border: 1px solid transparent;
      white-space: nowrap;
      position: relative;
    }
    
  
  /* ── Sidebar ── */
  .sidebar {
    width: 224px;
    background: var(--side-bg);
    border-right: 1px solid var(--side-border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    position: relative;
    transition: width 0.3s ease;
  }
  .sidebar.collapsed {
    width: 80px;
  }
  .sidebar-header {
    padding: 0 16px;
    height: 56px; /* h-14 */
    border-bottom: 1px solid var(--side-border);
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }
  .sidebar.collapsed .sidebar-header {
    justify-content: center;
    padding: 0 8px;
  }
  .logo-wrapper {
    display: flex; align-items: center; gap: 8px; min-width: 0;
  }
  .sidebar.collapsed .logo-wrapper {
    display: none !important;
  }
  .logo-text {
    font-family: 'Manrope', sans-serif;
    font-size: 0.875rem;
    letter-spacing: 3px;
    font-weight: 700;
    color: var(--side-logo);
  }
  .sidebar nav { flex: 1; padding: 8px; overflow-y: auto; }
  .sidebar.collapsed nav { padding: 8px 6px; }
  
  .nav-section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--side-fg-dim);
    padding: 16px 12px 4px;
  }
  .sidebar.collapsed .nav-section-label {
    display: none;
  }
  
  .nav-link {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    color: var(--side-fg-dim);
    text-decoration: none;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all 0.15s;
    margin-bottom: 2px;
    border: 1px solid transparent;
    white-space: nowrap;
    position: relative;
  }
  .sidebar.collapsed .nav-link {
    justify-content: center;
    padding-left: 0;
    padding-right: 0;
    width: 48px;
    height: 40px;
    margin-left: auto;
    margin-right: auto;
  }
  .sidebar.collapsed .nav-link span {
    display: none;
  }
  
  .nav-link:hover {
    background: var(--side-hover-bg);
    color: var(--side-fg);
  }
  .nav-link.active {
    background: var(--side-active-bg);
    color: var(--side-primary);
    border-color: var(--side-primary-dim);
  }
  .nav-link.active::before {
    content: '';
    position: absolute;
    left: -1px;
    top: 6px;
    bottom: 6px;
    width: 2px;
    border-radius: 0 2px 2px 0;
    background: var(--side-primary);
  }
  .nav-icon { flex-shrink: 0; width: 16px; height: 16px; }
  
  .sidebar-footer { padding: 12px; border-top: 1px solid var(--side-border); }
  .sidebar.collapsed .sidebar-footer { padding: 12px 6px; }
  .sidebar.collapsed .sidebar-footer .nav-link { width: 40px; height: 40px; }
  
  .beta-tag {
    font-size: 9px;
    font-weight: 600;
    background: rgba(59,130,246,0.1);
    color: hsl(217, 91%, 60%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 4px;
    padding: 2px 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    flex-shrink: 0;
  }
  .logout-link { margin-top: 4px; }
  .logout-link:hover {
    background: rgba(239, 68, 68, 0.1) !important;
    color: #ef4444 !important;
  }
  .logout-link:hover .nav-icon { color: #ef4444; }

  /* ── Status Badge override if any ── */
  .status-badge {
    display: flex; align-items: center; justify-content: center; gap: 6px;
    font-size: 0.75rem; font-weight: 500; color: var(--side-fg-dim); padding: 8px;
    border-radius: 8px; background: rgba(63,185,80,0.06); border: 1px solid rgba(63,185,80,0.15);
  }
  .status-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: #3fb950; animation: pulse 2s infinite alternate; }
  @keyframes pulse { to { opacity: 0.4; } }
  
  


    /* ── Main ── */
  .main-wrapper { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .top-header {
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 0 24px;
            height: 56px; display: flex; align-items: center; justify-content: space-between;
  }
  .top-header h1 { font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 700; color: var(--fg); }
  .top-header .sub { font-size: 0.78rem; color: var(--fg-dim); font-weight: 400; }

  .content { flex: 1; overflow-y: auto; padding: 24px; }

  .layout { display: grid; grid-template-columns: 300px 1fr; gap: 20px; align-items: start; height: 100%; }

  /* ── Control Panel ── */
  .ctrl-panel {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 24px; display: flex; flex-direction: column; gap: 16px;
    max-height: calc(100vh - 140px); overflow-y: auto;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
  }
  .cam-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .cam-tab {
    padding: 6px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border);
    background: transparent; color: var(--fg-dim); cursor: pointer;
    font-size: 0.8rem; font-weight: 600; transition: all 0.2s; font-family: inherit;
  }
  .cam-tab:hover { background: rgba(255,255,255,0.03); color: var(--fg); }
  .cam-tab.active { background: var(--primary-dim); color: var(--primary); border-color: rgba(36,182,252,0.3); }

  .input-group { display: flex; flex-direction: column; gap: 5px; }
  label { font-size: 0.7rem; color: var(--fg-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  select, input[type="datetime-local"] {
    width: 100%; padding: 8px 12px; background: rgba(0,0,0,0.2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--fg); font-size: 0.85rem; transition: all 0.2s; font-family: inherit;
  }
  select:focus, input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px rgba(36,182,252,0.15); }
  option { background: #0d1117; color: #fff; }

  .btn-primary {
    padding: 10px; border-radius: var(--radius-sm); border: none; background: var(--primary);
    color: #0A0D0F; font-weight: 700; cursor: pointer; font-size: 0.88rem;
    transition: all 0.15s; font-family: inherit;
  }
  .btn-primary:hover { filter: brightness(1.1); }

  .toggles { display: flex; gap: 8px; }
  .toggle-btn {
    flex: 1; padding: 8px; border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: transparent; color: var(--fg-muted); cursor: pointer; font-size: 0.78rem;
    font-weight: 600; transition: all 0.2s; font-family: inherit;
  }
  .toggle-btn.on { background: var(--primary-dim); color: var(--primary); border-color: rgba(36,182,252,0.3); }
  .toggle-btn:hover:not(.on) { background: rgba(255,255,255,0.03); color: var(--fg); }

  .stats {
    display: flex; gap: 10px; margin-top: auto; padding-top: 14px; border-top: 1px solid var(--border);
  }
  .stat-card {
    flex: 1; background: rgba(0,0,0,0.15); border-radius: var(--radius-sm);
    padding: 10px 12px; border: 1px solid var(--border);
  }
  .stat-val { font-size: 1.3rem; font-weight: 700; color: var(--fg); margin-top: 2px; }
  .stat-lbl { font-size: 0.68rem; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }

  /* ── Viewport area wrapper (holds feed + VLM panel) ── */
  #viewAreaWrapper {
    display: flex; flex-direction: row; gap: 0; min-width: 0; align-items: stretch;
    position: relative; /* Needed for VLM HUD overlay positioning */
    border: 1px solid var(--border); border-radius: var(--radius);
    overflow: hidden;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px -1px rgba(0,0,0,0.1);
    align-self: start;
  }

  /* ── Viewport ── */
  .view-container {
    position: relative; overflow: hidden; background: #000;
    flex: 1; min-width: 0;
  }
  img#feed { display: block; width: 100%; height: auto; cursor: crosshair; }
  img#overlayImg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; transition: opacity 0.3s ease; mix-blend-mode: screen; opacity: 0.9; }
  canvas#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  .loading-glow { position: absolute; top: 16px; right: 16px; width: 10px; height: 10px; border-radius: 50%; background: var(--primary); box-shadow: 0 0 10px var(--primary); animation: glow 1s infinite alternate; display: none; z-index: 10; }
  @keyframes glow { from { opacity: 0.3; transform: scale(0.8); } to { opacity: 1; transform: scale(1.2); } }
  .empty-state { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--fg-muted); font-weight: 500; font-size: 1rem; }

  /* Click-to-analyze hint badge */
  #clickHint {
    position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,0.65); backdrop-filter: blur(6px);
    color: rgba(255,255,255,0.8); font-size: 0.72rem; font-weight: 500;
    padding: 5px 12px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.12);
    pointer-events: none; z-index: 8; white-space: nowrap;
    transition: opacity 0.4s;
  }

  /* ── VLM Floating HUD Overlay ── */
  #vlmPanel {
    display: none; /* shown via JS as flex */
    position: absolute;
    top: 16px;
    right: 16px;
    bottom: 64px; /* Stop above the bottom DVR scrubber */
    max-height: calc(100% - 80px);
    width: 320px; flex-shrink: 0;
    flex-direction: column;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: color-mix(in srgb, var(--card) 65%, transparent);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25), 0 0 0 1px var(--border);
    overflow: hidden;
    z-index: 100;
  }
  #vlmPanel.open { 
    display: flex; 
    animation: arHUDIn 0.35s cubic-bezier(0.1, 1, 0.2, 1) forwards; 
  }
  @keyframes arHUDIn { 
    from { transform: scale(0.95) translateX(20px); opacity: 0; filter: blur(5px); } 
    to { transform: scale(1) translateX(0); opacity: 1; filter: blur(0); } 
  }
  .vlm-header {
    background: var(--card-hover);
    border-bottom: 1px solid var(--border);
    padding: 16px 18px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
  }
  .vlm-header-title { 
    font-weight: 700; font-size: 0.9rem; color: var(--fg); 
    display: flex; align-items: center; gap: 8px; 
    letter-spacing: 0.05em; text-transform: uppercase; 
  }
  .vlm-header-title .ai-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--primary);
    box-shadow: 0 0 8px var(--primary); animation: glow 1.5s infinite alternate;
    flex-shrink: 0;
  }
  .vlm-close { 
    background: transparent; border: 1px solid transparent; 
    cursor: pointer; color: var(--fg-muted); font-size: 1.1rem; line-height: 1; 
    padding: 4px 8px; border-radius: var(--radius-sm); transition: all 0.2s;
  }
  .vlm-close:hover { background: rgba(239, 68, 68, 0.1); color: var(--err); border-color: rgba(239, 68, 68, 0.2); }
  .vlm-body { padding: 16px 18px; flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; background: transparent; }
  .vlm-crop-wrap { text-align: center; display: none; position: relative; }
  .vlm-crop-wrap img { 
    max-width: 100%; max-height: 280px; border-radius: var(--radius); 
    border: 1px solid var(--border); 
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); 
    object-fit: contain; 
  }
  .vlm-track-info { 
    font-size: 0.75rem; color: var(--fg); line-height: 1.6; 
    background: var(--card-hover); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 8px 12px; font-weight: 500;
  }
  .vlm-label { 
    font-size: 0.68rem; color: var(--fg-muted); font-weight: 700; 
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; 
  }
  .vlm-select, .vlm-textarea {
    width: 100%; background: var(--card); border: 1px solid var(--border);
    color: var(--fg); border-radius: var(--radius); padding: 9px 12px; font-size: 0.85rem;
    font-family: inherit; transition: all 0.2s; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
  }
  .vlm-select:focus, .vlm-textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px var(--primary); }
  .vlm-textarea { resize: vertical; }
  .vlm-btn {
    width: 100%; padding: 12px; 
    background: linear-gradient(135deg, var(--primary) 0%, rgba(37,99,235,1) 100%); 
    color: #fff; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    border: none; border-radius: var(--radius); cursor: pointer; font-weight: 700;
    font-size: 0.9rem; letter-spacing: 0.5px; transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
  }
  .vlm-btn:hover:not(:disabled) { filter: brightness(1.15); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35); }
  .vlm-btn:active:not(:disabled) { transform: translateY(1px); }
  .vlm-btn:disabled { opacity: 0.6; cursor: not-allowed; filter: grayscale(50%); box-shadow: none; border-color: transparent; }
  .vlm-time-btn {
    padding: 6px 12px; background: var(--card); border: 1px solid var(--border);
    color: var(--fg-dim); border-radius: var(--radius-sm); cursor: pointer; font-size: 0.75rem;
    font-weight: 500; transition: all 0.15s;
  }
  .vlm-time-btn:hover { background: var(--border); color: var(--fg); }
  .vlm-time-btn.active { background: var(--primary); color: #fff; border-color: var(--primary); box-shadow: 0 0 10px rgba(59,130,246, 0.2); }
  .vlm-status { font-size: 0.78rem; color: var(--fg-dim); min-height: 1.2em; font-weight: 500; }
  .vlm-result {
    font-size: 0.85rem; color: var(--fg); line-height: 1.7;
    white-space: pre-wrap; background: var(--card-hover);
    border-radius: var(--radius); padding: 12px 14px; display: none;
    border: 1px solid var(--border);
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
  }

  /* ── Footer ── */
  .status-footer {
    background: var(--card); border-top: 1px solid var(--border);
    padding: 8px 24px; display: flex; align-items: center; justify-content: space-between;
    font-size: 0.7rem; color: var(--fg-muted);
  }
  .status-footer .left { display: flex; align-items: center; gap: 12px; }
  .status-footer .connected { display: flex; align-items: center; gap: 5px; }
  .status-footer .connected .dot { width: 5px; height: 5px; background: var(--ok); border-radius: 50%; animation: pulse 2s infinite alternate; }

  /* ── Scene Query Bar ── */
  .sq-chip {
    padding: 4px 11px; border-radius: 20px;
    border: 1px solid var(--border); background: transparent;
    color: var(--fg-muted); cursor: pointer; font-size: 0.75rem;
    font-weight: 600; transition: all 0.18s; font-family: inherit;
    white-space: nowrap;
  }
  .sq-chip:hover {
    background: var(--primary-dim); color: var(--primary);
    border-color: rgba(59,130,246,.35);
  }
  .sq-chip.active {
    background: var(--primary-dim); color: var(--primary);
    border-color: rgba(59,130,246,.35);
  }
  #sqBtn.loading { opacity: 0.7; pointer-events: none; }

    html[data-theme="light"] {
    --bg: #f0f3f7;
    --card: #ffffff;
    --card-hover: #e8edf4;
    --border: #cbd5e1;
    --fg: #1e2d40;
    --fg-dim: #475569;
    --fg-muted: #6b7fa0;
    --sidebar-bg: #e8edf4;
    --console-bg: #f1f5f9;
    /* Sidebar Light Tokens */
      --side-bg: hsl(215, 20%, 93%);
      --side-border: hsl(215, 18%, 84%);
      --side-fg: hsl(215, 30%, 16%);
      --side-fg-dim: hsl(215, 16%, 48%);
      --side-hover-bg: hsl(215, 16%, 88%);
      --side-active-bg: rgba(59,130,246,0.15);
      --side-primary: hsl(217, 91%, 60%);
      --side-primary-dim: rgba(59,130,246,0.25);
      --side-logo: #0f172a;
  }

  </style>
</head>
<body>

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-header">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <path d="M11.2966 4.12501C11.5853 3.62501 12.307 3.62501 12.5957 4.12501L21.2787 19.1645C21.638 19.7867 21.0013 20.5092 20.3388 20.231L12.2365 16.8289C12.0508 16.751 11.8415 16.751 11.6558 16.8289L3.55346 20.231C2.89106 20.5092 2.25435 19.7867 2.61357 19.1645L11.2966 4.12501Z" fill="var(--primary)" />
      </svg>
            <div class="logo-wrapper">
        <span class="logo-text">NORT</span>
        <span class="beta-tag">Beta</span>
      </div>
    </div>
    <nav>
      <div class="nav-section-label">Platform</div>
      <a href="/home" class="nav-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
        </svg>
        <span>Início</span>
      </a>
      <a href="/" class="nav-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /></svg>
        <span>Dashboard</span>
      </a>
      <a href="/streams" class="nav-link active">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
        <span>Live Streams</span>
      </a>
      <a href="/cameras" class="nav-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
        <span>Camera Setup</span>
      </a>
      <a href="/zones" class="nav-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
        <span>Zone Editor</span>
      </a>
      <a href="/floorplan" class="nav-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line><line x1="3" y1="12" x2="21" y2="12"></line>
        </svg>
        <span>Floor Plan</span>
      </a>
    </nav>
        <div class="sidebar-footer">
      <a href="#" class="nav-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        <span>Perfil</span>
      </a>
      <a href="/cameras" class="nav-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        <span>Configurações</span>
      </a>
      <a href="#" class="nav-link logout-link">
        <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
        <span>Sair</span>
      </a>
    </div>
    </div>
  </div>

  <div class="main-wrapper">
    <div class="top-header">
      <div>
        <h1>Spatial Analytics</h1>
        <div class="sub">Real-time heatmap, trajectories &amp; AI person analysis</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <div id="liveTrackBadge" style="display:none;align-items:center;gap:6px;font-size:0.75rem;font-weight:600;color:var(--primary);background:var(--primary-dim);border:1px solid rgba(59,130,246,0.25);border-radius:20px;padding:4px 11px;">
          <span style="width:6px;height:6px;border-radius:50%;background:var(--primary);display:inline-block;animation:pulse 1.5s infinite alternate;"></span>
          <span id="liveTrackCount">0</span> tracked
        </div>
        <div style="font-size:0.72rem;color:var(--fg-muted);background:var(--card-hover);border:1px solid var(--border);border-radius:6px;padding:4px 10px;">
          Click a person on the feed to analyze with AI
        </div>
      </div>
    </div>

    <div class="content">
      <div class="layout">
        <!-- Control Panel -->
        <div class="ctrl-panel">
          <div class="cam-tabs" id="camTabs">
            {% for cam_id in cameras %}
            <button id="btn_{{ cam_id }}" class="cam-tab{% if loop.first %} active{% endif %}" onclick="setCamera('{{ cam_id }}',this)">{{ cam_id }}</button>
            {% endfor %}
          </div>

          <div class="input-group">
            <label>Gender Filter</label>
            <select id="fGender">
              <option value="all">All Genders</option>
              <option value="male">Male Only</option>
              <option value="female">Female Only</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>

          <div class="input-group">
            <label>Age Demographic</label>
            <select id="fAge">
              <option value="all">All Ages</option>
              <option value="child">Child (0-12)</option>
              <option value="young_adult">Young Adult (13-35)</option>
              <option value="adult">Adult (36-60)</option>
              <option value="senior">Senior (60+)</option>
            </select>
          </div>

          <div style="display:flex; flex-direction:column; gap:10px;">
            <div class="input-group">
              <label>Time From</label>
              <input type="datetime-local" id="fFrom">
            </div>
            <div class="input-group">
              <label>Time To</label>
              <input type="datetime-local" id="fTo">
            </div>
          </div>

          <button class="btn-primary" onclick="applyFilters()">Query Database</button>

          <div>
            <label style="margin-bottom:6px; display:block;">Visualization Layers</label>
            <div class="toggles">
              <button class="toggle-btn" id="btnHeat" onclick="toggle('heatmap')">Heatmap</button>
              <button class="toggle-btn" id="btnPaths" onclick="toggle('paths')">Trajectories</button>
            </div>
            <div class="toggles" style="margin-top:8px;">
              <button class="toggle-btn on" id="btnFeed" onclick="toggle('feed')">Live Camera</button>
              <button class="toggle-btn on" id="btnOverlay" onclick="toggle('overlay')">Detections</button>
            </div>
            <div class="toggles" style="margin-top:8px;">
              <button class="toggle-btn" id="btnZones" onclick="toggle('zones')">Zones</button>
            </div>
          </div>

          <div class="stats">
            <div class="stat-card">
              <div class="stat-lbl">Data Points</div>
              <div class="stat-val" id="statPts">—</div>
            </div>
            <div class="stat-card">
              <div class="stat-lbl">Track Paths</div>
              <div class="stat-val" id="statPaths">—</div>
            </div>
          </div>
        </div>

        <!-- Viewport + VLM side panel in one flex row -->
        <div id="viewAreaWrapper">
          <div style="display:flex; flex-direction:column; flex:1; min-width:0;">
            <div class="view-container" id="viewContainer" style="overflow:hidden; position:relative;">
              <div class="loading-glow" id="loader"></div>
              <div id="zoomLayer" style="width:100%;height:100%;transform-origin:center center;transition:transform 0.15s ease-out, transform-origin 0.1s ease-out;">
                <img id="feed" src="" alt="">
                <img id="dvrFrame" src="" alt="" style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;z-index:5;">
                <img id="overlayImg" src="" alt="">
                <canvas id="overlay"></canvas>
                <canvas id="bboxOverlay" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:6;"></canvas>
              </div>
              <div class="empty-state" id="emptyState" style="display:none;">No cameras connected.</div>
              <div id="clickHint">Click a tracked person to analyze with AI</div>

              <!-- DVR Player Bar -->
              <div id="dvrBar">
                <style>
                  #dvrBar {
                    position: absolute; bottom: 0; left: 0; right: 0; z-index: 20;
                    background: linear-gradient(to top, rgba(0,0,0,0.72) 0%, transparent 100%);
                    padding: 14px 16px 10px;
                    display: flex; align-items: center; gap: 10px;
                    opacity: 0; pointer-events: none;
                    transition: opacity 0.25s ease;
                  }
                  #viewContainer:hover #dvrBar { opacity: 1; pointer-events: auto; }
                  #dvrPlayBtn, #dvrRevBtn {
                    flex-shrink: 0; background: rgba(255,255,255,0.15);
                    backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.25);
                    border-radius: 50%; width: 34px; height: 34px;
                    display: flex; align-items: center; justify-content: center;
                    cursor: pointer; color: #fff; font-size: 14px;
                    transition: background 0.15s;
                  }
                  #dvrPlayBtn:hover, #dvrRevBtn:hover { background: rgba(255,255,255,0.28); }
                  #dvrScrubber {
                    flex: 1; -webkit-appearance: none; appearance: none;
                    height: 4px; border-radius: 4px; cursor: pointer;
                    background: rgba(255,255,255,0.25);
                    outline: none; accent-color: var(--primary);
                  }
                  #dvrScrubber::-webkit-slider-thumb {
                    -webkit-appearance: none; width: 14px; height: 14px;
                    border-radius: 50%; background: #fff; box-shadow: 0 0 4px rgba(0,0,0,0.5);
                    cursor: pointer;
                  }
                  #dvrTimestamp {
                    flex-shrink: 0; font-size: 0.72rem; color: rgba(255,255,255,0.85);
                    font-family: 'SF Mono', 'Fira Mono', monospace; min-width: 90px;
                    text-align: center;
                  }
                  #dvrLiveBtn {
                    flex-shrink: 0; font-size: 0.7rem; font-weight: 700;
                    background: var(--primary); color: #fff; border: none;
                    border-radius: 12px; padding: 3px 10px; cursor: pointer;
                    letter-spacing: 0.05em; display: none;
                    transition: opacity 0.15s;
                  }
                  #dvrLiveBtn:hover { opacity: 0.85; }
                  #dvrLiveIndicator {
                    flex-shrink: 0; display: flex; align-items: center; gap: 5px;
                    font-size: 0.7rem; font-weight: 700; color: #f87171;
                    letter-spacing: 0.05em;
                  }
                  #dvrLiveIndicator span.dot {
                    width: 7px; height: 7px; border-radius: 50%;
                    background: #f87171; animation: pulse 1.2s ease-in-out infinite;
                  }
                </style>

                <button id="dvrRevBtn" title="Reverse Playback" onclick="dvrToggleReverse()">
                  <svg id="dvrIconRev" viewBox="0 0 24 24" width="14" height="14" fill="currentColor" style="transform: scaleX(-1);"><polygon points="5,3 19,12 5,21"/></svg>
                </button>
                <button id="dvrPlayBtn" title="Pause / Play" onclick="dvrTogglePlay()">
                  <svg id="dvrIconPause" viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="5" y="4" width="4" height="16" rx="1"/><rect x="15" y="4" width="4" height="16" rx="1"/></svg>
                  <svg id="dvrIconPlay" viewBox="0 0 24 24" width="14" height="14" fill="currentColor" style="display:none"><polygon points="5,3 19,12 5,21"/></svg>
                </button>
                <input type="range" id="dvrScrubber" min="0" max="300" value="300"
                       title="Scrub timeline"
                       oninput="dvrOnScrub(this.value)" onchange="dvrOnScrubEnd(this.value)">
                <span id="dvrTimestamp">LIVE</span>
                <button id="dvrLiveBtn" onclick="dvrGoLive()">⏩ LIVE</button>
                <div id="dvrLiveIndicator"><span class="dot"></span>LIVE</div>
              </div>
            </div>

            <!-- ── AI Scene Query Bar ────────────────────────────────────── -->
            <div id="sceneQueryBar" style="
              margin-top: 12px;
              background: var(--card);
              border: 1px solid var(--border);
              border-radius: 10px;
              padding: 14px 16px;
              display: flex;
              flex-direction: column;
              gap: 10px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            ">
              <!-- Preset chips -->
              <div style="display:flex; flex-wrap:wrap; gap:6px;">
                <span style="font-size:0.68rem;font-weight:600;color:var(--fg-muted);text-transform:uppercase;letter-spacing:.05em;align-self:center;margin-right:4px;">Quick:</span>
                <button class="sq-chip" onclick="sqPreset('Describe the behavior, actions, and interactions of the people in this scene. Ignore the room and static background.')">🎥 Describe behavior</button>
                <button class="sq-chip" onclick="sqPreset('How many people are in the scene and what are they doing?')">👥 Count people</button>
                <button class="sq-chip" onclick="sqPreset('Are there any groups of people standing together or congregating?')">🧑‍🤝‍🧑 Find groups</button>
                <button class="sq-chip" onclick="sqPreset('Find anyone who looks suspicious, loitering, or behaving unusually')">⚠️ Security issues</button>
                <button class="sq-chip" onclick="sqPreset('Describe the people in the scene — clothing, appearance, and what they are carrying')">🔍 Find person</button>
                <button class="sq-chip" onclick="sqPreset('Which areas of the store are busiest right now?')">📍 Busy areas</button>
                <button class="sq-chip" onclick="sqPreset('Is there anything unusual, unexpected, or potentially dangerous in this scene?')">🚨 Anomalies</button>
              </div>
              <!-- Text input row -->
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <label style="display:flex; align-items:center; gap:6px; font-size:.78rem; color:var(--fg-muted); cursor:pointer;">
                  <input type="checkbox" id="sqGlobal" style="cursor:pointer;" title="Find a person across all active cameras" onchange="document.getElementById('sqInput').placeholder = this.checked ? 'Find a person globally... e.g. &quot;man in red shirt&quot;' : 'Ask about this scene... e.g. &quot;what is happening here?&quot;'">
                  Find a person across all cameras (Global Search)
                </label>
              </div>
              <div style="display:flex; gap:8px; align-items:center;">
                <div style="position:relative;flex:1;">
                  <svg style="position:absolute;left:11px;top:50%;transform:translateY(-50%);opacity:.45;pointer-events:none;" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
                  <input id="sqInput" type="text"
                    placeholder="Ask about this scene... e.g. &quot;what is happening here?&quot;"
                    style="width:100%;padding:9px 12px 9px 32px;background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:7px;color:var(--fg);font-size:.85rem;font-family:inherit;transition:border-color .2s;"
                    onkeydown="if(event.key==='Enter')sqSend()"
                    onfocus="this.style.borderColor='var(--primary)'"
                    onblur="this.style.borderColor='var(--border)'"
                  >
                </div>
                <button id="sqBtn" onclick="sqSend()" style="
                  padding:9px 18px;background:var(--primary);color:#fff;border:none;
                  border-radius:7px;font-weight:700;font-size:.85rem;cursor:pointer;
                  font-family:inherit;white-space:nowrap;transition:filter .15s;flex-shrink:0;
                " onmouseover="this.style.filter='brightness(1.1)'" onmouseout="this.style.filter=''">Ask AI ✦</button>
              </div>
                <!-- Result area -->
              <div id="sqResult" style="display:none;">
                <div style="display:flex;align-items:center;gap:7px;margin-bottom:6px;">
                  <span style="width:7px;height:7px;border-radius:50%;background:var(--primary);display:inline-block;animation:pulse 1.5s infinite alternate;"></span>
                  <span style="font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--fg-muted);">AI Response</span>
                  <span id="sqCam" style="font-size:.68rem;color:var(--primary);margin-left:auto;"></span>
                </div>

                <div id="sqText" style="
                  font-size:.875rem;color:var(--fg);line-height:1.65;
                  white-space:pre-wrap;background:rgba(0,0,0,.15);
                  border-radius:7px;padding:12px 14px;
                  border:1px solid var(--border);
                "></div>
              </div>
            </div>
          </div>


          <div id="vlmPanel">
            <div class="vlm-header">
              <div class="vlm-header-title">
                <span class="ai-dot"></span>
                AI Analysis
              </div>
              <button class="vlm-close" onclick="closeVlmPanel()" title="Close">✕</button>
            </div>
            <div class="vlm-body">
              <div class="vlm-crop-wrap" id="vlmCropWrap">
                <img id="vlmCrop" src="" alt="Person crop">
              </div>
              <div class="vlm-track-info" id="vlmTrackInfo"></div>

              <div>
                <div class="vlm-label">Analysis mode</div>
                <select id="vlmMode" class="vlm-select">
                  <option value="describe">Describe appearance (single frame)</option>
                  <option value="behavior">Behavior / engagement (single frame)</option>
                  <option value="carrying">Items being carried (single frame)</option>
                  <option value="staff">Employee vs customer (single frame)</option>
                  <option value="staff_clip">Quick staff check (CLIP)</option>
                  <option value="anomaly">Anomaly detection (single frame)</option>
                  <option value="behavior_timeline">Behavior over time (video clip)</option>
                  <option value="movement_pattern">Movement pattern (video clip)</option>
                  <option value="suspicious_activity">Detect suspicious activity (video clip)</option>
                </select>
              </div>
              
              <!-- Time range selector for video clip analysis -->
              <div id="timeRangeSelector" style="display:none;">
                <div class="vlm-label">Time range</div>
                <div style="display:flex; gap:8px; flex-wrap:wrap;">
                  <button class="vlm-time-btn" data-seconds="1" onclick="setTimeRange(1)">1s</button>
                  <button class="vlm-time-btn active" data-seconds="3" onclick="setTimeRange(3)">3s</button>
                  <button class="vlm-time-btn" data-seconds="5" onclick="setTimeRange(5)">5s</button>
                  <button class="vlm-time-btn" data-seconds="10" onclick="setTimeRange(10)">10s</button>
                </div>
              </div>
              
              <div>
                <div class="vlm-label">Custom question <span style="font-weight:400;text-transform:none;letter-spacing:0;">(optional)</span></div>
                <textarea id="vlmQuestion" class="vlm-textarea" rows="2" placeholder="e.g. Is this person wearing a hat?"></textarea>
              </div>
              
              <!-- Video clip preview area -->
              <div id="videoClipPreview" style="display:none;">
                <div class="vlm-label">Video clip preview</div>
                <div id="clipFrames" style="display:flex; gap:4px; overflow-x:auto; padding:4px 0;"></div>
              </div>
              
              <button onclick="runVlmAnalysis()" id="vlmRunBtn" class="vlm-btn">Analyze</button>
              <div class="vlm-status" id="vlmStatus">Select a mode and click Analyze.</div>
              <div class="vlm-result" id="vlmResult"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="status-footer">
      <div class="left">
        <span>System Online</span>
        <div class="connected"><span class="dot"></span><span>Connected</span></div>
      </div>
      <div>© 2026 NORT Analytics</div>
    </div>
  </div>

<script>
if (localStorage.getItem('nort-theme') === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
}
if (localStorage.getItem('nort-sidebar-collapsed') === 'true') {
    const sb = document.querySelector('.sidebar');
    if (sb) sb.classList.add('collapsed');
}
document.addEventListener('DOMContentLoaded', () => {
    // Relocate VLM panel into viewContainer so it stays bounded within the video player
    const viewContainer = document.getElementById('viewContainer');
    const vlmPanel = document.getElementById('vlmPanel');
    if (viewContainer && vlmPanel) {
        viewContainer.appendChild(vlmPanel);
    }

    const sb = document.querySelector('.sidebar');
    if (sb && !sb.querySelector('.sidebar-collapse-btn')) {
        const btn = document.createElement('button');
        btn.className = 'sidebar-collapse-btn';
        btn.title = 'Toggle Sidebar';
        const chevronLeft  = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>`;
        const chevronRight = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>`;
        btn.innerHTML = sb.classList.contains('collapsed') ? chevronRight : chevronLeft;
        btn.onclick = () => {
            sb.classList.toggle('collapsed');
            const isCollapsed = sb.classList.contains('collapsed');
            localStorage.setItem('nort-sidebar-collapsed', isCollapsed);
            btn.innerHTML = isCollapsed ? chevronRight : chevronLeft;
        };
        sb.appendChild(btn);
    }
});

let currentCamera = "{{ cameras|first if cameras else '' }}";
let showHeat  = false;
let showPaths = false;
let showFeed  = true;
let showOverlay = true;
let showZones = false;
let zoneData = null;

let pathData  = null;
let heatImgUrl = null;

const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const feedEl  = document.getElementById('feed');
const overlayImg = document.getElementById('overlayImg');
const loader  = document.getElementById('loader');

function setCamera(cam, btn) {
  currentCamera = cam;
  zoneData = null;
  document.querySelectorAll('.cam-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  feedEl.src = showFeed ? (showOverlay ? '/stream/' + cam : '/stream_raw/' + cam) : 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
  if (showZones) fetchZones();
  applyFilters();
}

function toggle(what) {
  if (what === 'heatmap') { showHeat  = !showHeat;  document.getElementById('btnHeat').classList.toggle('on',  showHeat); }
  if (what === 'paths')   { showPaths = !showPaths; document.getElementById('btnPaths').classList.toggle('on', showPaths);}
  if (what === 'feed')    { showFeed  = !showFeed;  document.getElementById('btnFeed').classList.toggle('on',  showFeed);
                            feedEl.src = showFeed ? (showOverlay ? '/stream/' + currentCamera : '/stream_raw/' + currentCamera) : 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
                            feedEl.style.opacity = showFeed ? '1' : '0.1'; }
  if (what === 'overlay') { showOverlay = !showOverlay; document.getElementById('btnOverlay').classList.toggle('on', showOverlay);
                            if (showFeed) { feedEl.src = showOverlay ? '/stream/' + currentCamera : '/stream_raw/' + currentCamera; } }
  if (what === 'zones')   { showZones = !showZones; document.getElementById('btnZones').classList.toggle('on', showZones);
                            if (showZones && !zoneData) fetchZones(); }
  redraw();
}

async function fetchZones() {
  if (!currentCamera) return;
  try {
      const res = await fetch('/api/zones?camera=' + encodeURIComponent(currentCamera));
      if (res.ok) { zoneData = await res.json(); redraw(); }
  } catch(e) {}
}

function buildParams() {
  const p = new URLSearchParams({ camera: currentCamera });
  const g = document.getElementById('fGender').value; if (g !== 'all') p.set('gender', g);
  const a = document.getElementById('fAge').value;    if (a !== 'all') p.set('age',    a);
  const from = document.getElementById('fFrom').value; if (from) p.set('from', from);
  const to   = document.getElementById('fTo').value;   if (to)   p.set('to',   to);
  return p;
}

async function applyFilters() {
  if (!currentCamera) return;
  const p = buildParams();
  loader.style.display = 'block';
  
  try {
      const pRes = await fetch('/api/paths?' + p);
      if (pRes.ok) {
          pathData = await pRes.json();
          document.getElementById('statPaths').textContent = pathData.paths ? pathData.paths.length : '0';
      }
  } catch(e) {}

  try {
      const hRes = await fetch('/api/streams?' + p);
      if (hRes.ok) {
          const pts = hRes.headers.get('X-Total-Points');
          document.getElementById('statPts').textContent = pts || '0';
          
          const blob = await hRes.blob();
          const url = URL.createObjectURL(blob);
          if (heatImgUrl) URL.revokeObjectURL(heatImgUrl);
          heatImgUrl = url;
          overlayImg.src = url;
      }
  } catch(e) {}

  loader.style.display = 'none';
  redraw();
}

function redraw() {
  canvas.width  = feedEl.offsetWidth  || 960;
  canvas.height = feedEl.offsetHeight || 540;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  overlayImg.style.display = (showHeat && heatImgUrl) ? 'block' : 'none';

  if (showPaths && pathData && pathData.paths && pathData.paths.length > 0) {
    drawPaths(pathData.paths);
  }
  if (showZones && zoneData && zoneData.zones) {
    drawZones(zoneData.zones);
  }
}

const ZONE_COLORS = ['#24B6FC','#3fb950','#F0A500','#f85149','#bc8cff'];
function drawZones(zones) {
  zones.forEach((zone, idx) => {
    const pts = zone.points || zone.polygon || [];
    if (pts.length < 3) return;
    const col = ZONE_COLORS[idx % ZONE_COLORS.length];
    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.shadowColor = col;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    pts.forEach(([nx, ny], i) => {
        const px = nx * canvas.width;
        const py = ny * canvas.height;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = col;
    ctx.fill();
    ctx.globalAlpha = 0.9;
    ctx.stroke();
    // Label
    if (zone.name) {
        const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length * canvas.width;
        const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length * canvas.height;
        ctx.globalAlpha = 1;
        ctx.fillStyle = col;
        ctx.font = 'bold 13px Inter, sans-serif';
        ctx.shadowBlur = 4;
        ctx.fillText(zone.name, cx, cy);
    }
    ctx.restore();
  });
}

const PATH_COLORS = ['#24B6FC','#3fb950','#F0A500','#f85149','#bc8cff', '#0384ff'];
function drawPaths(paths) {
  paths.forEach((path, idx) => {
    if (path.length < 2) return;
    const col = PATH_COLORS[idx % PATH_COLORS.length];
    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth   = 4;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.shadowColor = col;
    ctx.shadowBlur  = 12;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    path.forEach(([cx, cy], i) => {
      const px = cx * canvas.width;
      const py = cy * canvas.height;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
    const last = path[path.length - 1];
    const prev = path[path.length - 2];
    const angle = Math.atan2((last[1] - prev[1]) * canvas.height, (last[0] - prev[0]) * canvas.width);
    const tip = [last[0] * canvas.width, last[1] * canvas.height];
    ctx.fillStyle = col;
    ctx.shadowBlur  = 16;
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(tip[0] - 18 * Math.cos(angle - 0.45), tip[1] - 18 * Math.sin(angle - 0.45));
    ctx.lineTo(tip[0] - 18 * Math.cos(angle + 0.45), tip[1] - 18 * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  });
}

if (currentCamera) {
  feedEl.src = showOverlay ? '/stream/' + currentCamera : '/stream_raw/' + currentCamera;
  feedEl.onload = () => { applyFilters(); };
  setInterval(() => { if (showHeat || showPaths) applyFilters(); }, 15000);
} else {
  document.getElementById('emptyState').style.display = 'flex';
}
window.addEventListener('resize', redraw);

// ── AI Scene Query Bar ─────────────────────────────────────────────────────
function sqPreset(text) {
  document.getElementById('sqInput').value = text;
  // Highlight the active chip
  document.querySelectorAll('.sq-chip').forEach(c => c.classList.remove('active'));
  event.currentTarget.classList.add('active');
  
  // Presets are always for Scene Understanding on the current camera
  const globalCb = document.getElementById('sqGlobal');
  globalCb.checked = false;
  // Trigger placeholder update
  globalCb.dispatchEvent(new Event('change'));

  sqSend();
}

async function sqSend() {
  const query = (document.getElementById('sqInput').value || '').trim();
  const isGlobal = document.getElementById('sqGlobal').checked;
  if (!query) return;
  if (!isGlobal && !currentCamera) return;

  const btn = document.getElementById('sqBtn');
  const resultEl = document.getElementById('sqResult');
  const textEl = document.getElementById('sqText');
  const camLbl = document.getElementById('sqCam');

  btn.textContent = 'Thinking…';
  btn.classList.add('loading');
  resultEl.style.display = 'block';
  textEl.textContent = '';
  camLbl.textContent = isGlobal ? 'ALL CAMERAS' : currentCamera;

  // Simple blinking cursor while waiting
  let dotCount = 0;
  const dotInterval = setInterval(() => {
    textEl.textContent = '●'.repeat((dotCount++ % 3) + 1);
  }, 350);

  _scanningActive = true;
  _activeCropImages = [];
  if (isGlobal || currentCamera) {
      fetch('/api/vlm/active_crops').then(r=>r.json()).then(data => {
          if (data.crops && data.crops.length > 0) {
              const images = [];
              let loadedCount = 0;
              data.crops.forEach(c => {
                  const img = new Image();
                  img.onload = () => loadedCount++;
                  img.src = c.image;
                  images.push({img, id: c.global_id});
              });
              _activeCropImages = images;
          }
      }).catch(e => console.error(e));
  }

  async function _pollSearchResult(jobId, textEl, camLbl, btn, dotInterval) {
    const MAX_POLLS = 300; // 300 × 0.4s = 2 min max
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise(r => setTimeout(r, 400));
      try {
        const res = await fetch(`/api/vlm/search_status/${jobId}`);
        const data = await res.json();
        if (data.status !== 'pending') {
          // Done — stop scanning animation
          clearInterval(dotInterval);
          _scanningActive = false;
          btn.textContent = 'Ask AI ✦';
          btn.classList.remove('loading');

          if (data.error || data.found === false) {
            textEl.textContent = '\u26a0\ufe0f ' + (data.error || data.result || 'Not found.');
            if (!data.found && !data.error) camLbl.textContent = 'NONE';
          } else {
            const answer = data.result || '(no answer)';
            textEl.textContent = '';
            let ci = 0;
            const typeInterval = setInterval(() => {
              if (ci >= answer.length) { clearInterval(typeInterval); return; }
              textEl.textContent += answer[ci++];
            }, 12);
            camLbl.textContent = data.camera_id || currentCamera;
            if (data.found && data.camera_id) {
              if (data.camera_id !== currentCamera) {
                const tabBox = document.getElementById('btn_' + data.camera_id);
                if (tabBox) setCamera(data.camera_id, tabBox);
              }
              if (data.global_id) {
                setTimeout(() => {
                  openVlmPanel(data.global_id);
                }, 100);
              }
            }
          }
          return;
        }
      } catch(e) {
        console.warn('[VLM poll]', e);
      }
    }
    // Timeout
    clearInterval(dotInterval);
    _scanningActive = false;
    btn.textContent = 'Ask AI ✦';
    btn.classList.remove('loading');
    textEl.textContent = '\u26a0\ufe0f Search timed out.';
  }

  try {
    const endpoint = isGlobal ? '/api/vlm/global_query' : '/api/vlm/scene_query';
    const payload = isGlobal ? { query } : { camera_id: currentCamera, query };

    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (data.error) {
      clearInterval(dotInterval);
      _scanningActive = false;
      btn.textContent = 'Ask AI ✦';
      btn.classList.remove('loading');
      textEl.textContent = '\u26a0\ufe0f ' + data.error;
      return;
    }

    if (data.job_id) {
      // Non-blocking: poll for result in background — stream stays live
      _pollSearchResult(data.job_id, textEl, camLbl, btn, dotInterval);
      // NOTE: we intentionally do NOT await here — the try/finally below
      // must NOT stop the scanning animation prematurely.
      return;
    }

    // Fallback if no job_id for some reason
    clearInterval(dotInterval);
    _scanningActive = false;
    btn.textContent = 'Ask AI ✦';
    btn.classList.remove('loading');
    textEl.textContent = '\u26a0\ufe0f Unexpected response from VLM.';
  } catch(e) {
    clearInterval(dotInterval);
    _scanningActive = false;
    textEl.textContent = '\u26a0\ufe0f Network error: ' + e.message;
  } finally {
    // Only reset button if we didn't hand off to the async poller
    if (!isGlobal) {
      btn.textContent = 'Ask AI ✦';
      btn.classList.remove('loading');
    }
  }
}


// ── VLM: click-to-analyze ─────────────────────────────────────────────────────
// The MJPEG stream already draws all bounding boxes at full pipeline FPS.
// We do NOT re-draw them here — that would create a lagged duplicate layer.
//
// Instead:
//   • Poll /api/status at 500ms to keep _vlmTracks fresh for click hit-testing
//   • bboxOverlay canvas draws ONLY the selection highlight for the clicked track
//   • Click anywhere on the feed → find which track bbox contains that point

let _vlmTracks = {};        // { global_id → {x1,y1,x2,y2,frame_w,frame_h,...} }
let _selectedTrackId = null;
let _scanningActive = false;
let _arFrame = 0;
let _activeCropImages = [];
let _activeCropIdx = 0;
let _lastCropTime = 0;
const bboxCanvas = document.getElementById('bboxOverlay');
const bboxCtx = bboxCanvas.getContext('2d');

function _normBbox(t) {
  let nx1 = t.x1, ny1 = t.y1, nx2 = t.x2, ny2 = t.y2;
  if (nx1 > 2) {
    const fw = t.frame_w || 1920, fh = t.frame_h || 1080;
    nx1 /= fw; ny1 /= fh; nx2 /= fw; ny2 /= fh;
  }
  return [nx1, ny1, nx2, ny2];
}

function _drawAROverlay() {
  requestAnimationFrame(_drawAROverlay);

  // Resize canvas to match the rendered feed image only if changed
  const fw = feedEl.offsetWidth  || 960;
  const fh = feedEl.offsetHeight || 540;
  if (bboxCanvas.width !== fw || bboxCanvas.height !== fh) {
      bboxCanvas.width = fw;
      bboxCanvas.height = fh;
  }
  bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
  
  _arFrame += 0.05;

  if (_scanningActive && _activeCropImages.length > 0) {
      if (performance.now() - _lastCropTime > 400) {
          _activeCropIdx = (_activeCropIdx + 1) % _activeCropImages.length;
          _lastCropTime = performance.now();
      }
      const activeCrop = _activeCropImages[_activeCropIdx];
      
      const boxW = 86;
      const boxH = 86;
      const padding = 20;
      const px = bboxCanvas.width - boxW - padding;
      const py = bboxCanvas.height - boxH - padding;
      
      bboxCtx.save();
      bboxCtx.shadowColor = '#06b6d4';
      bboxCtx.shadowBlur = 10;
      bboxCtx.strokeStyle = '#06b6d4';
      bboxCtx.lineWidth = 1.5;
      
      bboxCtx.fillStyle = 'rgba(0,0,0,0.5)';
      bboxCtx.fillRect(px, py, boxW, boxH);
      
      if (activeCrop.img.complete) {
          bboxCtx.shadowBlur = 0;
          bboxCtx.drawImage(activeCrop.img, px, py, boxW, boxH);
      }
      
      bboxCtx.strokeRect(px, py, boxW, boxH);
      
      // Corner brackets
      const cs = 10;
      bboxCtx.shadowBlur = 8;
      [[px,py,1,1],[px+boxW,py,-1,1],[px,py+boxH,1,-1],[px+boxW,py+boxH,-1,-1]].forEach(([x,y,dx,dy]) => {
        bboxCtx.beginPath();
        bboxCtx.moveTo(x + dx*cs, y); bboxCtx.lineTo(x, y); bboxCtx.lineTo(x, y + dy*cs);
        bboxCtx.stroke();
      });
      
      // Scanning line
      const lineY = py + ((Math.sin(_arFrame * 3) + 1) / 2) * boxH;
      bboxCtx.beginPath();
      bboxCtx.moveTo(px, lineY);
      bboxCtx.lineTo(px + boxW, lineY);
      bboxCtx.lineWidth = 2;
      bboxCtx.stroke();
      
      // Text
      bboxCtx.shadowBlur = 0;
      bboxCtx.fillStyle = 'rgba(6,182,212,0.9)';
      bboxCtx.font = 'bold 10px monospace';
      bboxCtx.fillText('MATCHING DB', px, py - 16);
      bboxCtx.fillText('ID: ' + activeCrop.id, px, py - 4);
      
      bboxCtx.restore();
  }

  // Frontend UI no longer draws the red tracking box; 
  // It is now rendered directly on the camera stream by bbox_renderer 
  // for perfect, zero-latency sync.

}
// Kick off the AR loop
requestAnimationFrame(_drawAROverlay);


// ── DVR Player ────────────────────────────────────────────────────────────────
// DESIGN: The live MJPEG (feedEl) is NEVER cleared — it keeps streaming in
// background.  A separate #dvrFrame img overlays it during DVR mode so the
// container never collapses.  Playback advances FORWARD in time when "playing"
// in DVR mode (offset decreases until it reaches 0 → go live).
//
//  Scrubber: left (0) = oldest available, right = live edge
//  _dvrWindow is queried from /api/dvr/info on each camera switch.

let DVR_WINDOW     = 300;   // defaults to 5 min; updated from server on each camera switch
let _dvrMode       = false; // true = DVR overlay visible
let _dvrPlaying    = false; // true = auto-advancing toward live
let _dvrPlayDir    = 1;     // 1 = forward, -1 = reverse
let _dvrOffset     = 0;     // seconds behind NOW (0 = live edge)
let _dvrPlayTimer  = null;
let _dvrScrubbing  = false;
let _dvrDebTimer   = null;  // debounce timer for scrub renders

const _dvrScrubEl = document.getElementById('dvrScrubber');
const _dvrTsEl    = document.getElementById('dvrTimestamp');
const _dvrPlayBtn = document.getElementById('dvrPlayBtn');
const _dvrRevBtn  = document.getElementById('dvrRevBtn');
const _dvrLiveBtn = document.getElementById('dvrLiveBtn');
const _dvrLiveInd = document.getElementById('dvrLiveIndicator');
const _dvrFrameEl = document.getElementById('dvrFrame');
const _dvrIconPause = document.getElementById('dvrIconPause');
const _dvrIconPlay  = document.getElementById('dvrIconPlay');

function _dvrSetIcon(isPlaying) {
  if (_dvrIconPause) _dvrIconPause.style.display = (isPlaying && _dvrPlayDir === 1) ? '' : 'none';
  if (_dvrIconPlay)  _dvrIconPlay.style.display  = (isPlaying && _dvrPlayDir === 1) ? 'none' : '';
  if (_dvrRevBtn) {
    _dvrRevBtn.style.color = (isPlaying && _dvrPlayDir === -1) ? 'var(--primary)' : '#fff';
  }
}

function _dvrFmt(epochSec) {
  return new Date(epochSec * 1000)
    .toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function _dvrFill(sliderVal, win) {
  win = win || DVR_WINDOW || 300;
  const pct = (sliderVal / win) * 100;
  _dvrScrubEl.style.background =
    'linear-gradient(to right, var(--primary) 0%, var(--primary) ' + pct +
    '%, rgba(255,255,255,0.25) ' + pct + '%, rgba(255,255,255,0.25) 100%)';
}

// Render a DVR frame — debounced during scrubbing to avoid flooding the server
function _dvrRenderFrame(epochSec) {
  const cam = currentCamera || '';
  if (!cam) return;
  if (_dvrDebTimer) clearTimeout(_dvrDebTimer);
  _dvrDebTimer = setTimeout(() => {
    _dvrFrameEl.src = '/api/dvr/seek/' + cam + '/' + Math.floor(epochSec) +
                      '?_=' + Math.floor(epochSec);
  }, _dvrScrubbing ? 120 : 0);  // 120ms debounce while dragging, instant otherwise
}

// Update the scrubber + timestamp from current _dvrOffset
function _dvrSyncUI() {
  const sliderVal = DVR_WINDOW - _dvrOffset;
  _dvrScrubEl.value = sliderVal;
  _dvrFill(sliderVal);
  const epoch = Date.now() / 1000 - _dvrOffset;
  _dvrTsEl.textContent = _dvrOffset <= 0 ? 'LIVE' : _dvrFmt(epoch);
}

// Query the server for actual available buffer length and update scrubber max
function dvrInitFromCamera(camId) {
  if (!camId) return;
  fetch('/api/dvr/info/' + camId)
    .then(r => r.json())
    .then(d => {
      // original API returns {min_ts, max_ts, count}
      const avail = (d.max_ts && d.min_ts) ? (d.max_ts - d.min_ts) : 0;
      // Use actual available buffer, clamped between 5s and 300s
      DVR_WINDOW = Math.max(5, Math.min(300, avail));
      _dvrScrubEl.max = DVR_WINDOW;
      // Reset to live edge
      _dvrScrubEl.value = DVR_WINDOW;
      _dvrFill(DVR_WINDOW);
    })
    .catch(() => { /* leave defaults */ });
}

// Enter DVR mode at a specific offset (seconds behind NOW)
function _dvrEnterAt(offsetSec) {
  _dvrMode    = true;
  _dvrPlaying = false;
  _dvrOffset  = Math.max(1, Math.min(DVR_WINDOW, offsetSec));
  _dvrStop();
  _dvrFrameEl.style.display = 'block';
  _dvrSetIcon(false);  // show play icon
  _dvrLiveBtn.style.display = 'inline-block';
  _dvrLiveInd.style.display = 'none';
  _dvrSyncUI();
  _dvrRenderFrame(Date.now() / 1000 - _dvrOffset);
}

// Start auto-playback (forward through time) while in DVR mode
function _dvrPlay() {
  if (!_dvrMode) return;
  _dvrPlaying = true;
  _dvrSetIcon(true);
  _dvrStop();
  _dvrPlayTimer = setInterval(() => {
    if (_dvrScrubbing) return;
    _dvrOffset = Math.max(0, Math.min(DVR_WINDOW, _dvrOffset - _dvrPlayDir));
    _dvrSyncUI();
    if (_dvrOffset <= 0 && _dvrPlayDir === 1) {
      dvrGoLive();
    } else if (_dvrOffset >= DVR_WINDOW && _dvrPlayDir === -1) {
      dvrPause(); // paused at start of buffer
    } else {
      _dvrRenderFrame(Date.now() / 1000 - _dvrOffset);
    }
  }, 1000);
}

function _dvrStop() {
  if (_dvrPlayTimer) { clearInterval(_dvrPlayTimer); _dvrPlayTimer = null; }
}

// Public: pause (freeze on current frame)
function dvrPause() {
  _dvrPlaying = false;
  _dvrStop();
  _dvrSetIcon(false);
}

// Public: toggle reverse
function dvrToggleReverse() {
  if (!_dvrMode) {
    _dvrEnterAt(Math.min(5, DVR_WINDOW));
    _dvrPlayDir = -1;
    _dvrPlay();
    return;
  }
  if (_dvrPlaying && _dvrPlayDir === -1) {
    dvrPause();
  } else {
    _dvrPlayDir = -1;
    _dvrPlay();
  }
}

// Public: toggle pause/play
function dvrTogglePlay() {
  if (!_dvrMode) {
    _dvrEnterAt(Math.min(5, DVR_WINDOW));
    _dvrPlayDir = 1;
    _dvrPlay();
    return;
  }
  if (_dvrPlaying && _dvrPlayDir === 1) dvrPause(); 
  else {
    _dvrPlayDir = 1;
    _dvrPlay();
  }
}

// Public: go back to live
function dvrGoLive() {
  _dvrStop();
  _dvrMode    = false;
  _dvrPlaying = false;
  _dvrPlayDir = 1;
  _dvrOffset  = 0;
  _dvrFrameEl.style.display = 'none';
  _dvrFrameEl.src = '';
  _dvrSetIcon(true);  // reset to play icon
  _dvrLiveBtn.style.display = 'none';
  _dvrLiveInd.style.display = 'flex';
  _dvrScrubEl.value = DVR_WINDOW;
  _dvrFill(DVR_WINDOW);
  _dvrTsEl.textContent = 'LIVE';
}

// Scrubber oninput — called continuously while dragging
function dvrOnScrub(val) {
  _dvrScrubbing = true;
  const sliderVal = parseInt(val, 10);
  _dvrOffset = DVR_WINDOW - sliderVal;
  _dvrFill(sliderVal);
  const epoch = Date.now() / 1000 - _dvrOffset;
  _dvrTsEl.textContent = _dvrOffset <= 0 ? 'LIVE' : _dvrFmt(epoch);
  if (!_dvrMode && _dvrOffset > 0) {
    _dvrMode = true;
    _dvrFrameEl.style.display = 'block';
    _dvrLiveBtn.style.display = 'inline-block';
    _dvrLiveInd.style.display = 'none';
    _dvrStop();
    _dvrPlaying = false;
    _dvrSetIcon(false);
  }
  // Debounced render happens inside _dvrRenderFrame
  _dvrRenderFrame(epoch);
}

// Scrubber onchange — called once when user releases the thumb
function dvrOnScrubEnd(val) {
  _dvrScrubbing = false;
  const sliderVal = parseInt(val, 10);
  _dvrOffset = DVR_WINDOW - sliderVal;
  if (_dvrOffset <= 0) {
    dvrGoLive();
  } else {
    _dvrMode = true;
    _dvrFrameEl.style.display = 'block';
    _dvrLiveBtn.style.display = 'inline-block';
    _dvrLiveInd.style.display = 'none';
    _dvrRenderFrame(Date.now() / 1000 - _dvrOffset);
    _dvrSyncUI();
  }
}

// Passively track live MJPEG url via MutationObserver (for camera-switch resilience)
let _dvrLiveSrc = '';
new MutationObserver(() => {
  const src = feedEl.getAttribute('src') || '';
  if (src && src.startsWith('/stream')) {
    _dvrLiveSrc = src;
    // Re-query buffer size whenever the camera changes
    dvrGoLive();
    dvrInitFromCamera(currentCamera);
  }
}).observe(feedEl, {attributes: true, attributeFilter: ['src']});

// Initialize scrubber on page load
_dvrFill(DVR_WINDOW);
if (currentCamera) dvrInitFromCamera(currentCamera);

// ── Mouse Wheel Zoom ──────────────────────────────────────────────────────────
let _zoomLvl = 1;
let _zoomX = 50; let _zoomY = 50;
const _zoomLayer = document.getElementById('zoomLayer');
const _viewCont  = document.getElementById('viewContainer');

_viewCont.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (e.deltaY < 0) {
    _zoomLvl = Math.min(5, _zoomLvl + 0.25);
  } else {
    _zoomLvl = Math.max(1, _zoomLvl - 0.25);
  }
  
  if (_zoomLvl === 1) {
    _zoomX = 50; _zoomY = 50;
    _zoomLayer.style.transform = '';
  } else {
    const rect = _viewCont.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    _zoomX = _zoomX * 0.7 + x * 0.3; // ease towards mouse
    _zoomY = _zoomY * 0.7 + y * 0.3;
    _zoomLayer.style.transformOrigin = `${_zoomX}% ${_zoomY}%`;
    _zoomLayer.style.transform = `scale(${_zoomLvl})`;
  }
});














// ── SSE Real-time Track Streaming ────────────────────────────────────────────
// Uses Server-Sent Events for smooth, low-latency track position updates (~15fps)
// Replaces the old 500ms polling which caused selection box lag

let _trackEventSource = null;

function connectTrackStream() {
  // Close existing connection
  if (_trackEventSource) {
    _trackEventSource.close();
    _trackEventSource = null;
  }
  
  // Connect to SSE endpoint
  _trackEventSource = new EventSource('/api/stream/tracks');
  
  _trackEventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      const allTracks = data.tracks || {};
      
      // Filter to current camera
      _vlmTracks = {};
      for (const [gid, t] of Object.entries(allTracks)) {
        if (t.camera_id === currentCamera || !t.camera_id) {
          _vlmTracks[gid] = t;
        }
      }
      
      // Note: Selection highlight is now drawn on backend for perfect sync
      // _drawSelectionHighlight() removed - highlight is part of video feed
      
      // Update live track count badge
      const n = Object.keys(_vlmTracks).length;
      const badge = document.getElementById('liveTrackBadge');
      if (badge) {
        badge.style.display = n > 0 ? 'flex' : 'none';
        document.getElementById('liveTrackCount').textContent = n;
      }
      
      // Show/hide click hint
      const hint = document.getElementById('clickHint');
      if (hint && hint.style.opacity !== '0') {
        hint.style.opacity = n > 0 ? '1' : '0';
      }
    } catch(e) {}
  };
  
  _trackEventSource.onerror = (e) => {
    console.log('[SSE] Connection error, will retry...');
    // Auto-reconnect after 2 seconds
    setTimeout(connectTrackStream, 2000);
  };
}

// Connect when camera changes
const _origSetCamera = setCamera;
setCamera = function(camId, btn) {
  _origSetCamera(camId, btn);
  connectTrackStream();
};

// Initial connection
connectTrackStream();

// Click the feed → find which track bbox contains the click point
feedEl.addEventListener('click', (e) => {
  const rect = feedEl.getBoundingClientRect();
  const cx = (e.clientX - rect.left) / rect.width;
  const cy = (e.clientY - rect.top)  / rect.height;
  let bestId = null, bestArea = Infinity;
  for (const [gid, t] of Object.entries(_vlmTracks)) {
    if (t.x1 == null) continue;
    const [nx1, ny1, nx2, ny2] = _normBbox(t);
    if (cx >= nx1 && cx <= nx2 && cy >= ny1 && cy <= ny2) {
      // If overlapping, prefer the smallest (frontmost) box
      const area = (nx2 - nx1) * (ny2 - ny1);
      if (area < bestArea) { bestArea = area; bestId = gid; }
    }
  }
  if (bestId) openVlmPanel(bestId);
});

async function openVlmPanel(trackId) {
  _selectedTrackId = trackId;
  document.getElementById('vlmPanel').classList.add('open');
  // Hide the hint once user has discovered the feature
  const hint = document.getElementById('clickHint');
  if (hint) hint.style.opacity = '0';
  
  // Notify backend to draw selection highlight (syncs with tracking boxes)
  const t = _vlmTracks[trackId] || {};
  try {
    await fetch('/api/vlm/select', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ track_id: trackId, camera_id: t.camera_id })
    });
  } catch(e) {}
  
  // Track info
  document.getElementById('vlmTrackInfo').textContent =
    'ID: ' + trackId + (t.gender ? '  ·  ' + t.gender : '') + (t.age ? '  ·  ' + t.age : '') + (t.zone ? '  ·  Zone: ' + t.zone : '');
  
  // Check if video mode is selected
  const mode = document.getElementById('vlmMode').value;
  const isVideo = isVideoMode(mode);
  
  // Show/hide crop vs video preview
  const cropWrap = document.getElementById('vlmCropWrap');
  const videoPreview = document.getElementById('videoClipPreview');
  
  if (isVideo) {
    cropWrap.style.display = 'none';
    videoPreview.style.display = 'block';
    loadClipPreview();
  } else {
    videoPreview.style.display = 'none';
    // Crop thumbnail
    const cropImg = document.getElementById('vlmCrop');
    cropImg.src = '/api/crop/' + encodeURIComponent(trackId) + '?t=' + Date.now();
    cropImg.onload = () => { cropWrap.style.display = 'block'; };
    cropImg.onerror = () => { cropWrap.style.display = 'none'; };
  }
  
  // Show/hide time range selector
  document.getElementById('timeRangeSelector').style.display = isVideo ? 'block' : 'none';
  
  // Reset result
  document.getElementById('vlmResult').style.display = 'none';
  document.getElementById('vlmResult').textContent = '';
  document.getElementById('vlmStatus').textContent = 'Select a mode and click Analyze.';
  document.getElementById('vlmRunBtn').disabled = false;
  document.getElementById('vlmRunBtn').textContent = 'Analyze';
}

async function closeVlmPanel() {
  document.getElementById('vlmPanel').classList.remove('open');
  _selectedTrackId = null;
  if (_vlmPollTimer) { clearInterval(_vlmPollTimer); _vlmPollTimer = null; }
  
  // Clear backend selection highlight
  try {
    await fetch('/api/vlm/select', { method: 'DELETE' });
  } catch(e) {}
}

// Time range selection for video clip analysis
let _selectedTimeRange = 3; // default 3 seconds

function setTimeRange(seconds) {
  _selectedTimeRange = seconds;
  // Update UI
  document.querySelectorAll('.vlm-time-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.seconds) === seconds);
  });
  // Refresh clip preview if video mode
  const mode = document.getElementById('vlmMode').value;
  if (isVideoMode(mode)) {
    loadClipPreview();
  }
}

function isVideoMode(mode) {
  return ['behavior_timeline', 'movement_pattern', 'suspicious_activity'].includes(mode);
}

// Show/hide time range selector based on mode
document.getElementById('vlmMode').addEventListener('change', (e) => {
  const mode = e.target.value;
  const isVideo = isVideoMode(mode);
  document.getElementById('timeRangeSelector').style.display = isVideo ? 'block' : 'none';
  document.getElementById('videoClipPreview').style.display = isVideo ? 'block' : 'none';
  if (isVideo && _selectedTrackId) {
    loadClipPreview();
  }
});

// Load and display video clip preview
async function loadClipPreview() {
  if (!_selectedTrackId) return;
  const container = document.getElementById('clipFrames');
  container.innerHTML = '<div style="color:var(--fg-muted)">Loading frames…</div>';
  
  try {
    const track = _vlmTracks[_selectedTrackId] || {};
    const cameraId = track.camera_id;
    if (!cameraId) throw new Error('No camera_id');
    
    const res = await fetch(`/api/clip/${encodeURIComponent(_selectedTrackId)}?camera_id=${cameraId}&last_n_seconds=${_selectedTimeRange}&max_frames=5`);
    if (!res.ok) {
      container.innerHTML = '<div style="color:var(--fg-muted)">No video clip available yet</div>';
      return;
    }
    
    const data = await res.json();
    container.innerHTML = '';
    
    data.frames.forEach((frame, i) => {
      const img = document.createElement('img');
      img.src = 'data:image/jpeg;base64,' + frame.image_jpeg;
      img.style.cssText = 'height:60px; border-radius:4px; border:1px solid var(--border);';
      img.title = `Frame ${i+1} @ ${frame.timestamp.toFixed(1)}s`;
      container.appendChild(img);
    });
  } catch(e) {
    container.innerHTML = '<div style="color:var(--fg-muted)">Error loading clip</div>';
  }
}

async function runVlmAnalysis() {
  if (!_selectedTrackId) return;
  const mode     = document.getElementById('vlmMode').value;
  const question = document.getElementById('vlmQuestion').value.trim();
  const btn      = document.getElementById('vlmRunBtn');
  const statusEl = document.getElementById('vlmStatus');
  const resultEl = document.getElementById('vlmResult');

  btn.disabled = true;
  btn.textContent = 'Analyzing…';
  statusEl.textContent = 'Submitting request…';
  resultEl.style.display = 'none';

  // Check if this is a video clip analysis
  const isVideo = isVideoMode(mode);
  
  try {
    let res;
    if (isVideo) {
      // Video clip analysis endpoint
      res = await fetch('/api/analyze_clip', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ 
          track_id: _selectedTrackId, 
          mode, 
          question,
          last_n_seconds: _selectedTimeRange,
          camera_id: (_vlmTracks[_selectedTrackId] || {}).camera_id
        }),
      });
    } else {
      // Single frame analysis
      res = await fetch('/api/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ track_id: _selectedTrackId, mode, question }),
      });
    }
    
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = '✗ ' + (data.error || 'Request failed');
      btn.disabled = false; btn.textContent = 'Analyze';
      return;
    }
  } catch(e) {
    statusEl.textContent = '✗ Network error';
    btn.disabled = false; btn.textContent = 'Analyze';
    return;
  }

  statusEl.textContent = 'Analyzing… (this may take 3–8 seconds for video clips)';
  // Poll for result
  if (_vlmPollTimer) clearInterval(_vlmPollTimer);
  const pollTrackId = _selectedTrackId;
  _vlmPollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/analysis/' + encodeURIComponent(pollTrackId));
      const d = await r.json();
      if (d.status === 'done') {
        clearInterval(_vlmPollTimer); _vlmPollTimer = null;
        statusEl.textContent = '✓ Analysis complete';
        resultEl.textContent = d.text;
        resultEl.style.display = 'block';
        btn.disabled = false; btn.textContent = 'Analyze again';
      } else if (d.status === 'error') {
        clearInterval(_vlmPollTimer); _vlmPollTimer = null;
        statusEl.textContent = '✗ ' + (d.text || 'Analysis failed');
        btn.disabled = false; btn.textContent = 'Retry';
      }
      // 'pending' → keep polling
    } catch(e) {}
  }, 800);
}
</script>
</body>
</html>"""


@app.route("/streams")
@requires_auth
def heatmap_page():
    from flask import render_template
    cameras = _enabled_camera_ids()
    return render_template('streams.html', cameras=cameras)

@app.route("/zones")
@requires_auth
def zones_page():
    from flask import render_template
    cameras = _enabled_camera_ids()
    return render_template("zones.html", cameras=cameras)

@app.route("/floorplan")
@requires_auth
def floorplan_page():
    from flask import render_template
    from core.homography_manager import load_all_homographies
    cameras = list(_camera_status.keys())
    homographies = load_all_homographies()
    # Build a simplified dict: camera_id -> world_pts (list of [x,y])
    world_zones = {
        cam_id: data.get("world_pts", [])
        for cam_id, data in homographies.items()
        if data.get("world_pts")
    }
    return render_template("floorplan.html", cameras=cameras, world_zones=world_zones)


@app.route("/settings")
@requires_auth
def settings_page():
    from flask import render_template
    import json, os
    device_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "device.json")
    try:
        with open(device_json_path, 'r', encoding='utf-8') as f:
            d_config = json.load(f)
    except Exception:
        d_config = {}
    return render_template("settings.html", settings=d_config.get("ui_settings", {}))


@app.route("/help")
@requires_auth
def help_page():
    from flask import render_template
    return render_template("help.html", device_id=DEVICE_ID, store_id=STORE_ID)


# ── VLM (Visual Language Model) endpoints ─────────────────────────────────────
# These routes are authenticated and allow users to click a tracked person in the
# live stream and receive an AI-generated description of what they see.

@app.route("/api/analyze", methods=["POST"])
@requires_auth
def api_vlm_analyze():
    """
    Submit an analysis request for a tracked person.

    Body (JSON): { "track_id": "str", "question": "optional free text", "mode": "describe|behavior|anomaly|staff|carrying|staff_clip" }
    Returns:     { "ok": true } on success, { "error": "..." } if queue full.
    """
    try:
        from core import vlm_analyst
    except ImportError:
        return jsonify({"error": "VLM module not available"}), 503

    body = request.get_json(silent=True) or {}
    track_id = str(body.get("track_id", "")).strip()
    if not track_id:
        return jsonify({"error": "track_id required"}), 400

    question = str(body.get("question", "")).strip()
    mode     = str(body.get("mode", "describe")).strip()

    if not vlm_analyst.has_crop(track_id):
        return jsonify({"error": "No crop available for this track yet. Wait until the person is clearly visible."}), 404

    queued = vlm_analyst.submit_analysis(track_id, question=question, mode=mode)
    if not queued:
        return jsonify({"error": "Analysis queue is full. Try again in a moment."}), 503
    return jsonify({"ok": True, "track_id": track_id})


@app.route("/api/analyze_clip", methods=["POST"])
@requires_auth
def api_vlm_analyze_clip():
    """
    Submit a video clip analysis request for temporal behavior analysis.

    Body (JSON): { 
        "track_id": "str", 
        "camera_id": "str",
        "question": "optional free text", 
        "mode": "behavior_timeline|movement_pattern|suspicious_activity",
        "last_n_seconds": 5 (default)
    }
    Returns: { "ok": true } on success, { "error": "..." } if queue full.
    """
    try:
        from core import vlm_analyst
        from core import video_buffer
    except ImportError as e:
        return jsonify({"error": f"Module not available: {e}"}), 503

    body = request.get_json(silent=True) or {}
    track_id = str(body.get("track_id", "")).strip()
    camera_id = str(body.get("camera_id", "")).strip()
    
    if not track_id:
        return jsonify({"error": "track_id required"}), 400
    if not camera_id:
        return jsonify({"error": "camera_id required"}), 400

    question = str(body.get("question", "")).strip()
    mode = str(body.get("mode", "behavior_timeline")).strip()
    last_n_seconds = float(body.get("last_n_seconds", 5.0))

    # Get video clip
    clip = video_buffer.get_clip(track_id, camera_id, last_n_seconds, max_frames=10)
    if not clip:
        return jsonify({"error": "No video clip available for this track. Wait until tracking has captured enough frames."}), 404

    # Build temporal analysis prompt
    if not question:
        temporal_prompts = {
            "behavior_timeline": f"Describe this person's behavior over the last {last_n_seconds:.0f} seconds. What actions did they take? Did they interact with any products or staff?",
            "movement_pattern": f"Analyze this person's movement pattern over the last {last_n_seconds:.0f} seconds. Where did they move and what caught their attention?",
            "suspicious_activity": f"Detect any suspicious or unusual behavior in this {last_n_seconds:.0f} second clip. Report anything that might indicate theft or vandalism."
        }
        question = temporal_prompts.get(mode, temporal_prompts["behavior_timeline"])

    # Extract frames from clip for analysis
    frames = [entry.image for entry in clip]
    
    # Submit clip analysis with actual video frames
    queued = vlm_analyst.submit_clip_analysis(track_id, camera_id, frames, question=question, mode=mode)
    if not queued:
        return jsonify({"error": "Analysis queue is full. Try again in a moment."}), 503
    
    return jsonify({
        "ok": True, 
        "track_id": track_id,
        "clip_frames": len(clip),
        "clip_duration": clip[-1].timestamp - clip[0].timestamp if len(clip) > 1 else 0
    })


@app.route("/api/analysis/<track_id>")
@requires_auth
def api_vlm_result(track_id: str):
    """
    Poll for analysis result.
    Returns: { "status": "pending"|"done"|"error"|"not_found", "text": "...", "ts": float }
    """
    try:
        from core import vlm_analyst
        result = vlm_analyst.get_result(track_id)
    except ImportError:
        result = {"status": "error", "text": "VLM module not available", "ts": 0.0}
    return jsonify(result)


@app.route("/api/crop/<track_id>")
@requires_auth
def api_vlm_crop(track_id: str):
    """Return the latest stored crop for a track as JPEG."""
    try:
        from core import vlm_analyst
        jpeg = vlm_analyst.get_crop_jpeg(track_id)
    except ImportError:
        jpeg = None
    if not jpeg:
        return "No crop available", 404
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/api/pose/<track_id>")
@requires_auth
def api_pose_jpeg(track_id: str):
    """
    Serve pre-computed pose skeleton JPEG from cache.
    Computed by the background PoseWorker in vlm_analyst as crops arrive.
    Returns 404 while the first computation is in flight (client should retry).
    """
    from core import vlm_analyst as _va
    with _va._track_crops_lock:
        entry = _va._track_crops.get(str(track_id))
        jpeg  = entry.get("pose_jpeg") if entry else None
    if not jpeg:
        return "Pose not ready yet", 404
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/api/pose/data/<track_id>")
@requires_auth
def api_pose_data(track_id: str):
    """Serve pre-computed pose keypoint + angle data from cache."""
    from core import vlm_analyst as _va
    with _va._track_crops_lock:
        entry = _va._track_crops.get(str(track_id))
        data  = entry.get("pose_data") if entry else None
    if not data:
        return jsonify({"detected": False, "keypoints": {}, "angles": {}, "posture": "unknown"})
    return jsonify(data)

@app.route("/api/vlm/tracks")
@requires_auth
def api_vlm_tracks():
    """Return list of global_ids that currently have a stored VLM crop."""
    try:
        from core import vlm_analyst
        ids = vlm_analyst.list_tracked_ids()
    except ImportError:
        ids = []
    return jsonify({"track_ids": ids})






@app.route("/api/vlm/metrics")
@requires_auth
def api_vlm_metrics():
    """
    Return VLM performance metrics for monitoring.
    Returns: {
        "enabled": bool,
        "jobs_submitted": int,
        "jobs_completed": int,
        "jobs_failed": int,
        "jobs_dropped": int,
        "avg_inference_ms": float,
        "max_inference_ms": float,
        "min_inference_ms": float,
        "last_inference_at": float,
        "cached_crops": int
    }
    """
    try:
        from core import vlm_analyst
        enabled = vlm_analyst.is_enabled()
        metrics = vlm_analyst.get_metrics() if enabled else {}
        cached_crops = len(vlm_analyst.list_tracked_ids()) if enabled else 0
    except ImportError:
        enabled = False
        metrics = {}
        cached_crops = 0
    
    return jsonify({
        "enabled": enabled,
        "cached_crops": cached_crops,
        **metrics
    })


# ── SSE Real-time Track Streaming ───────────────────────────────────────────
# Replaces the 500ms polling with push-based real-time updates

import json as _json
import time as _time

@app.route("/api/vlm/select", methods=["POST", "DELETE"])
@requires_auth
def api_vlm_select():
    """Tells the backend which track is currently selected in the UI so its box can be highlighted."""
    from flask import request as _req, jsonify as _jsonify
    try:
        from core import vlm_analyst
        if _req.method == "DELETE":
            vlm_analyst.set_active_target(None)
            return _jsonify({"success": True})
            
        data = _req.get_json(silent=True) or {}
        track_id = data.get("track_id")
        vlm_analyst.set_active_target(track_id)
        return _jsonify({"success": True})
    except Exception as e:
        return _jsonify({"error": str(e)}), 500

@app.route("/api/vlm/scene_query", methods=["POST"])
@requires_auth
def api_vlm_scene_query():
    """
    Run a free-form VLM query against the latest frame of a camera.
    Body: {"camera_id": "camera_1", "query": "Describe the scene"}
    Returns: {"result": "...", "camera_id": "..."}
    """
    from flask import request as _req, jsonify as _jsonify
    data = _req.get_json(silent=True) or {}
    camera_id = data.get("camera_id", "")
    query = (data.get("query") or "").strip()
    if not query:
        return _jsonify({"error": "query is required"}), 400

    # Get the latest annotated frame (JPEG bytes) for the requested camera
    frame_bytes = _latest_raw_frames.get(camera_id) or _latest_frames.get(camera_id)
    if not frame_bytes:
        return _jsonify({"error": f"No frame available for camera '{camera_id}' — is tracking running?"}), 404

    try:
        import cv2 as _cv2, numpy as _np
        nparr = _np.frombuffer(frame_bytes, _np.uint8)
        frame_bgr = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return _jsonify({"error": "Could not decode camera frame"}), 500
    except Exception as e:
        return _jsonify({"error": f"Frame decode error: {e}"}), 500

    try:
        from core import vlm_analyst
        if not vlm_analyst.is_enabled():
            return _jsonify({"error": 'VLM is disabled — set "vlm": {"enabled": true} in device.json'}), 503
        
        # Submit non-blocking job
        job_id = vlm_analyst.submit_scene_query(camera_id, frame_bgr, query)
    except Exception as e:
        return _jsonify({"error": str(e)}), 500

    return _jsonify({"job_id": job_id, "status": "queued", "camera_id": camera_id})


# ── Settings API ──────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET", "POST"])
@requires_auth
def api_settings():
    from flask import request, jsonify
    import json, os
    device_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "device.json")
    
    if request.method == "GET":
        try:
            with open(device_json_path, 'r', encoding='utf-8') as f:
                d_config = json.load(f)
            return jsonify(d_config.get("ui_settings", {}))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    if request.method == "POST":
        try:
            data = request.get_json(silent=True) or {}
            with open(device_json_path, 'r', encoding='utf-8') as f:
                d_config = json.load(f)
            
            # Merge with existing settings so we don't wipe out other ui stuff
            current_ui = d_config.get("ui_settings", {})
            current_ui.update(data)
            d_config["ui_settings"] = current_ui
            
            with open(device_json_path, 'w', encoding='utf-8') as f:
                json.dump(d_config, f, indent=4)
                
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/api/settings/pin", methods=["POST"])
@requires_auth
def api_settings_pin():
    from flask import request, jsonify
    import json, os
    from werkzeug.security import generate_password_hash
    
    device_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "device.json")
    data = request.get_json(silent=True) or {}
    new_pin = data.get("pin")
    
    if not new_pin:
         return jsonify({"error": "No PIN provided"}), 400
         
    try:
        with open(device_json_path, 'r', encoding='utf-8') as f:
            d_config = json.load(f)
            
        hashed_pin = generate_password_hash(str(new_pin))
        d_config["admin_pin"] = hashed_pin
        
        with open(device_json_path, 'w', encoding='utf-8') as f:
            json.dump(d_config, f, indent=4)
            
        # Update the running instance so it applies immediately
        global ADMIN_PIN
        ADMIN_PIN = hashed_pin
            
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vlm/global_query", methods=["POST"])
@requires_auth
def api_vlm_global_query():
    """
    Search across ALL active cameras — non-blocking.
    Body: {"query": "find the man in red"}
    Returns: {"job_id": "...", "status": "queued", "candidate_count": N}
    Poll /api/vlm/search_status/<job_id> for the result.
    """
    from flask import request as _req, jsonify as _jsonify
    data = _req.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return _jsonify({"error": "query is required"}), 400

    if not _latest_raw_frames:
        return _jsonify({"error": "No cameras available to search."}), 404

    try:
        from core import vlm_analyst
        if not vlm_analyst.is_enabled():
            return _jsonify({"error": "VLM is disabled in configuration."}), 503
    except ImportError:
        return _jsonify({"error": "VLM module not found"}), 500

    import time

    # Snapshot the crop store (hold lock briefly, copy crops to avoid holding lock during inference)
    search_candidates = []
    with vlm_analyst._track_crops_lock:
        for gid, entry in vlm_analyst._track_crops.items():
            if time.time() - entry.get("ts", 0) < 30.0:
                crop = entry.get("crop")
                if crop is not None and crop.size > 0:
                    search_candidates.append((gid, crop.copy(), entry.get("cam")))

    if not search_candidates:
        return _jsonify({"found": False, "status": "done", "result": "No one has been seen recently to search."})

    # Submit job to background worker — returns instantly
    job_id = vlm_analyst.submit_search(query, search_candidates)
    return _jsonify({"job_id": job_id, "status": "queued", "candidate_count": len(search_candidates)})


@app.route("/api/vlm/search_status/<job_id>")
@requires_auth
def api_vlm_search_status(job_id):
    """
    Poll for the result of a VLM global search job.
    Returns: {status: 'pending'|'done'|'not_found', found: bool, camera_id, global_id, result}
    """
    from flask import jsonify as _jsonify
    try:
        from core import vlm_analyst
    except ImportError:
        return _jsonify({"status": "error", "found": False, "result": "VLM module not found"}), 500
    result = vlm_analyst.get_search_result(job_id)
    return _jsonify(result)




@app.route("/api/vlm/active_crops")
@requires_auth
def api_vlm_active_crops():
    """Returns a list of base64 images of recently tracked people for UI scanning animation."""
    from flask import jsonify as _jsonify
    import base64
    import time
    try:
        from core import vlm_analyst
        if not vlm_analyst.is_enabled():
            return _jsonify({"crops": []})
    except ImportError:
        return _jsonify({"crops": []})
        
    crops_b64 = []
    with vlm_analyst._track_crops_lock:
        # take up to 20 recent crops
        recent_tracks = list(vlm_analyst._track_crops.items())[-20:]
        for gid, entry in recent_tracks:
            if time.time() - entry.get("ts", 0) < 60.0:
                jpeg_bytes = vlm_analyst.get_crop_jpeg(gid)
                if jpeg_bytes:
                    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                    crops_b64.append({"global_id": str(gid), "image": "data:image/jpeg;base64," + b64})
                    
    return _jsonify({"crops": crops_b64})


@app.route("/api/stream/tracks")
@requires_auth
def api_stream_tracks():
    """
    Server-Sent Events endpoint for real-time track positions.
    Streams track positions at ~15fps for smooth selection box sync.
    
    Client usage:
        const es = new EventSource('/api/stream/tracks');
        es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            // data.tracks = {global_id: {x1, y1, x2, y2, camera_id, ...}}
        };
    """
    def generate():
        # CRITICAL: must use gevent.sleep(), NOT time.sleep().
        # This generator runs as a gevent greenlet (not a real OS thread).
        # stdlib time.sleep() blocks the entire gevent hub OS thread, starving
        # every other request. At 15fps that is 76% hub starvation → total freeze.
        try:
            from gevent import sleep as _gsleep
        except ImportError:
            from time import sleep as _gsleep

        last_sent = 0
        min_interval = 0.066  # ~15fps max (66ms)

        while True:
            try:
                now = _time.time()

                # Rate limiting — yield hub while waiting
                if now - last_sent < min_interval:
                    _gsleep(0.01)
                    continue

                # Get current retail data from global variable
                global _retail_data
                tracks = _retail_data.get("tracks", {})

                last_sent = now

                # Format: SSE data frame
                yield f"data: {_json.dumps({'tracks': tracks, 'ts': now})}\n\n"

                _gsleep(0.05)  # 50ms cooperative yield — hub stays responsive

            except GeneratorExit:
                break
            except Exception as e:
                app.logger.error(f"[SSE] Error: {e}")
                _gsleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'X-Accel-Buffering': 'no'  # Disable nginx buffering if present
                   })


@app.route("/api/clip/<track_id>")
@requires_auth
def api_get_clip(track_id: str):
    """
    Get video clip frames for a track for temporal analysis.
    Returns list of frames with timestamps.
    
    Query params:
        - last_n_seconds: How many seconds of history (default: 5)
        - max_frames: Max frames to return (default: 10)
    """
    try:
        from core import video_buffer
        from flask import request
        
        last_n = request.args.get('last_n_seconds', 5, type=float)
        max_frames = request.args.get('max_frames', 10, type=int)
        
        # Need camera_id - try to find it
        camera_id = request.args.get('camera_id', None)
        if not camera_id:
            # Try to get from retail data
            global _retail_data
            track = _retail_data.get("tracks", {}).get(track_id, {})
            camera_id = track.get("camera_id")
        
        if not camera_id:
            return jsonify({"error": "camera_id required"}), 400
        
        clip = video_buffer.get_clip(track_id, camera_id, last_n, max_frames)
        
        if not clip:
            return jsonify({"error": "No video clip available for this track"}), 404
        
        # Convert frames to JPEG for transmission
        frames_data = []
        for entry in clip:
            ok, buf = cv2.imencode(".jpg", entry.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                import base64
                frames_data.append({
                    "timestamp": entry.timestamp,
                    "frame_number": entry.frame_number,
                    "bbox": entry.bbox,
                    "image_jpeg": base64.b64encode(buf.tobytes()).decode('utf-8')
                })
        
        return jsonify({
            "track_id": track_id,
            "camera_id": camera_id,
            "frames": frames_data,
            "duration_seconds": clip[-1].timestamp - clip[0].timestamp if len(clip) > 1 else 0,
            "frame_count": len(frames_data)
        })
        
    except ImportError:
        return jsonify({"error": "Video buffer not available"}), 503
    except Exception as e:
        app.logger.error(f"[Clip] Error getting clip: {e}")
        return jsonify({"error": str(e)}), 500



# ── DVR: ring-buffer seek ──────────────────────────────────────────────────────

@app.route('/api/dvr/info/<cam_id>')
@requires_auth
def api_dvr_info(cam_id: str):
    """Return the available time window for the DVR buffer of a camera."""
    try:
        from core import frame_buffer as _fb
        buf = _fb.get_buffer(cam_id)
        entries = buf.list_entries()
        if not entries:
            return jsonify({"min_ts": None, "max_ts": None, "count": 0})
        return jsonify({
            "min_ts": entries[0][0],
            "max_ts": entries[-1][0],
            "count":  len(entries),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dvr/seek/<cam_id>/<int:ts>')
@requires_auth
def api_dvr_seek(cam_id: str, ts: int):
    """Serve the buffered JPEG frame closest to the requested epoch timestamp."""
    try:
        from core import frame_buffer as _fb
        buf = _fb.get_buffer(cam_id)
        path = buf.seek(ts)
        if not path or not os.path.exists(path):
            return ('No frame available', 404)
        from flask import send_file
        return send_file(path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ── Logout (HTTP Basic Auth clear trick) ─────────────────────────────────────

@app.route("/logout", methods=["GET", "POST"])
def logout():
    """
    Log out by returning a 401 with an empty WWW-Authenticate header.
    This clears the stored Basic Auth credentials from the browser.
    After clearing, redirect to root which will prompt for credentials again.
    """
    # Most browsers will clear stored Basic Auth when they receive a 401
    # with a different realm or after the credentials were accepted once.
    # The JS in templates calls: fetch('/logout',{method:'POST'}).then(()=>location.href='/')
    # We return 401 so the browser forgets the credentials, then JS redirects to /
    from flask import make_response
    resp = make_response('', 401)
    resp.headers['WWW-Authenticate'] = 'Basic realm="Logged out"'
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


# ── Entry point ────────────────────────────────────────────────────────────────

def _collect_local_san_ips() -> list:
    """
    Return a list of ipaddress.IPv4Address objects for every local interface
    IP that should appear as a Subject Alternative Name in the TLS cert.
    Always includes 127.0.0.1; adds every non-loopback LAN IP so the cert
    is valid when the admin panel is accessed from another machine on the
    store's network (e.g. 10.x.x.x, 192.168.x.x).
    """
    import ipaddress as _ip
    import socket as _socket
    ips = set()
    ips.add(_ip.IPv4Address("127.0.0.1"))
    try:
        # socket.getaddrinfo returns all addresses for the local hostname
        for _res in _socket.getaddrinfo(_socket.gethostname(), None):
            _addr = _res[4][0]
            try:
                _v4 = _ip.IPv4Address(_addr)
                ips.add(_v4)
            except ValueError:
                pass   # skip IPv6
    except Exception:
        pass
    # Also try the UDP trick — connects but sends nothing; gets the outbound IP
    try:
        _s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        ips.add(_ip.IPv4Address(_s.getsockname()[0]))
        _s.close()
    except Exception:
        pass
    return sorted(ips, key=lambda a: (a != _ip.IPv4Address("127.0.0.1"), str(a)))


def _cert_needs_regen(cert_path: str) -> bool:
    """
    Return True if the existing cert should be replaced.  Triggers when:
    - A current local IP is not in the cert's SAN list (DHCP change), OR
    - The cert is missing the ExtendedKeyUsage=serverAuth extension that
      Chrome requires to accept self-signed certs on parallel connections.
    """
    try:
        from cryptography import x509 as _x509
        from cryptography.x509.oid import ExtensionOID as _EOID, ExtendedKeyUsageOID as _EKUOID
        import ipaddress as _ip

        with open(cert_path, "rb") as _f:
            _cert = _x509.load_pem_x509_certificate(_f.read())

        # 1. Check IP SANs
        _san_ext = _cert.extensions.get_extension_for_oid(_EOID.SUBJECT_ALTERNATIVE_NAME)
        _cert_ips = set(_san_ext.value.get_values_for_type(_x509.IPAddress))
        _current_ips = set(_collect_local_san_ips())
        if not _current_ips.issubset(_cert_ips):
            _log.info("H5: cert SANs outdated — will regenerate")
            return True

        # 2. Check ExtendedKeyUsage — Chrome requires serverAuth
        try:
            _eku = _cert.extensions.get_extension_for_oid(_EOID.EXTENDED_KEY_USAGE)
            if _EKUOID.SERVER_AUTH not in _eku.value:
                _log.info("H5: cert missing ExtendedKeyUsage=serverAuth — will regenerate")
                return True
        except Exception:
            # Extension missing entirely
            _log.info("H5: cert has no ExtendedKeyUsage extension — will regenerate")
            return True

        return False
    except Exception:
        return False   # if we can't parse the cert, keep it


def _ensure_self_signed_cert() -> tuple:
    """
    H5-fix: Generate a self-signed TLS certificate on first run so the admin
    panel is served over HTTPS instead of cleartext HTTP.

    The cert includes ALL current local interface IPs as SANs so it is valid
    whether the browser connects via 127.0.0.1, localhost, or the store LAN IP
    (e.g. 10.10.10.40).  The cert is regenerated automatically when new IPs
    are detected (e.g. after a DHCP change).

    Returns (cert_path, key_path).  Both files live next to local_admin.py
    so they survive process restarts but are never committed (add *.pem to
    .gitignore).
    """
    _admin_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(_admin_dir, "admin_cert.pem")
    key_path  = os.path.join(_admin_dir, "admin_key.pem")

    _exists = os.path.exists(cert_path) and os.path.exists(key_path)
    if _exists and not _cert_needs_regen(cert_path):
        return cert_path, key_path

    if _exists:
        _log.info("H5: Local IP changed — regenerating TLS cert with updated SANs.")

    # Try cryptography library first (no external process needed)
    try:
        import datetime as _dt
        import ipaddress as _ip
        from cryptography import x509
        from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        _key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        _name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"nort-jetson-admin")])
        _now = _dt.datetime.utcnow()

        # Build SANs: localhost + every current LAN IP
        _san_entries = [x509.DNSName(u"localhost")]
        for _lip in _collect_local_san_ips():
            _san_entries.append(x509.IPAddress(_lip))

        # Chrome requires BasicConstraints + KeyUsage + ExtendedKeyUsage(serverAuth).
        # Without ExtendedKeyUsage=serverAuth, Chrome sends certificate_unknown for every
        # new parallel connection (MJPEG streams, API calls) even after the user clicks
        # "proceed" for the main page.  All three extensions are mandatory for Chrome
        # to accept a self-signed cert on concurrent connections.
        _cert = (
            x509.CertificateBuilder()
            .subject_name(_name)
            .issuer_name(_name)
            .public_key(_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_now)
            .not_valid_after(_now + _dt.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(_san_entries), critical=False)
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, key_encipherment=True,
                    content_commitment=False, data_encipherment=False,
                    key_agreement=False, key_cert_sign=False,
                    crl_sign=False, encipher_only=False, decipher_only=False,
                ), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False)
            .sign(_key, hashes.SHA256())
        )
        with open(key_path, "wb") as _f:
            _f.write(_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))
        with open(cert_path, "wb") as _f:
            _f.write(_cert.public_bytes(serialization.Encoding.PEM))
        _lip_strs = [str(a) for a in _collect_local_san_ips()]
        _log.info(
            "H5: Generated self-signed TLS cert "
            "(SANs: localhost, %s; extensions: BasicConstraints, KeyUsage, serverAuth)",
            ", ".join(_lip_strs))
        return cert_path, key_path
    except Exception as _exc:
        # Catch both ImportError and any runtime failure (PermissionError, crypto error…)
        if not isinstance(_exc, ImportError):
            _log.warning(f"H5: cryptography cert generation failed: {_exc}")

    # Fallback: use openssl subprocess (no SAN support without a config file,
    # but better than no TLS at all)
    try:
        result = subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path,
            "-days", "3650", "-nodes",
            "-subj", "/CN=nort-jetson-admin",
        ], capture_output=True, timeout=30)
        if result.returncode == 0:
            _log.info("H5: Generated self-signed TLS cert via openssl (no SAN — install cryptography for LAN IP support).")
            return cert_path, key_path
        _log.warning(f"H5: openssl cert generation failed: {result.stderr.decode()}")
    except Exception as _exc:
        _log.warning(f"H5: Could not generate TLS cert: {_exc}")

    return None, None


def start_local_admin(port: int = 8080):
    """
    Start the Flask admin server as a daemon thread from main.py:

        import local_admin
        local_admin.DEVICE_ID = config.DEVICE_ID
        ...
        threading.Thread(target=local_admin.start_local_admin, daemon=True).start()

    The panel is served over HTTPS (self-signed cert auto-generated on first run).
    Accept the browser security warning once; subsequent visits are cached.
    """
    from data import spatial_logger as _sl
    _sl.init_db()

    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.WARNING)

    # H5-fix: use HTTPS so credentials and the live video stream are not sent
    # in cleartext over the store LAN.
    cert_path, key_path = _ensure_self_signed_cert()
    has_tls = bool(cert_path and key_path)
    if has_tls:
        _log.info(f"Admin panel starting on https://0.0.0.0:{port}/ (self-signed TLS)")
    else:
        _log.warning("Admin panel starting on HTTP (TLS cert unavailable — install cryptography or openssl)")

    # Prefer gevent WSGIServer — it serves MJPEG streams as async greenlets
    # instead of one blocking thread per stream, which eliminates the GIL
    # contention and thread exhaustion that makes Flask's dev server sluggish
    # when 5 cameras are streaming simultaneously.
    try:
        from gevent.pywsgi import WSGIServer as _WSGIServer
        # Do NOT call monkey.patch_all() here — ssl/urllib3 are already imported
        # by the time this thread starts and late-patching ssl causes RecursionError
        # in the heartbeat thread.  gevent WSGIServer handles concurrent greenlets
        # for MJPEG streams without any monkey-patching.
        _log.info("[Admin] Using gevent WSGIServer (async MJPEG streaming)")
        if has_tls:
            server = _WSGIServer(
                ("0.0.0.0", port), app,
                keyfile=key_path, certfile=cert_path,
            )
        else:
            server = _WSGIServer(("0.0.0.0", port), app)
        server.serve_forever()
        return
    except ImportError:
        _log.warning("[Admin] gevent not installed — falling back to Flask dev server "
                     "(install gevent for better performance: pip install gevent)")
    except Exception as _e:
        _log.warning(f"[Admin] gevent server failed ({_e}) — falling back to Flask dev server")

    # Fallback: Flask dev server (threaded)
    ssl_context = (cert_path, key_path) if has_tls else None
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False,
            threaded=True, ssl_context=ssl_context)

