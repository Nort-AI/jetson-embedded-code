"""
core/alert_engine.py — Rule-based proactive alert engine for NORT.

Evaluates alert conditions every EVAL_INTERVAL seconds against live tracker
state (_track_crops). Pushes AlertEvent objects into a thread-safe Queue
consumed by the SSE endpoint in local_admin.py.

Features
--------
  • queue_alert    — camera has ≥ N people visible
  • dwell_alert    — person in same zone > T minutes
  • capacity_alert — total store count ≥ max_capacity
  • loitering      — person hasn't changed zone in > 10 min (non-checkout)
  • visitor_surge  — N+ people entered in the last 2 minutes
  • camera_offline — configured camera has not sent a frame in 30 s

Design constraints
------------------
  • Zero VLM / Claude API calls — pure tracker-state logic
  • Daemon thread, respects stop event for clean shutdown
  • Per-alert cooldowns to avoid notification spam
  • Quiet-hours gate from device.json (ui_settings)
  • Reads _track_crops via lazy import inside functions (no circular imports)
"""

import json
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from system.logger_setup import setup_logger
    logger = setup_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ── Tunable thresholds (overridable via device.json vlm.alerts section) ──────
EVAL_INTERVAL        = 5      # seconds between rule evaluations
QUEUE_ALERT_THRESHOLD  = 6    # people in one camera to trigger queue alert
DWELL_ALERT_MINUTES    = 5    # minutes in same zone before dwell alert
LOITERING_MINUTES      = 10   # minutes without zone change = loitering
SURGE_WINDOW_SEC       = 120  # 2-minute window for visitor surge
SURGE_THRESHOLD        = 5    # new arrivals in that window = surge
CAMERA_OFFLINE_SEC     = 45   # seconds without frame = camera offline

# Alert type → cooldown seconds (per entity_id)
_COOLDOWNS = {
    "queue_alert":    180,   # 3 min
    "dwell_alert":    300,   # 5 min per person
    "capacity_alert": 300,   # 5 min
    "loitering":      600,   # 10 min per person
    "visitor_surge":  600,   # 10 min
    "camera_offline": 900,   # 15 min per camera
}

# Checkout-like zone names — loitering expected there, don't alert
_CHECKOUT_ZONES = frozenset(
    w.lower() for w in ["checkout", "caixa", "caixas", "caixa_rapido",
                         "caixa_normal", "register", "pagamento", "payment"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlertEvent:
    id:        str
    type:      str        # alert type key
    severity:  str        # "info" | "warning" | "critical"
    camera_id: str
    global_id: str        # "" for store/camera-level alerts
    message:   str        # human-readable text (language from device.json)
    ts:        float      # unix timestamp
    data:      dict = field(default_factory=dict)
    dismissed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AlertEngine:
    """Daemon thread that evaluates alert rules every EVAL_INTERVAL seconds."""

    def __init__(self):
        self._stop          = threading.Event()
        self._cooldowns:    dict  = {}          # (type, entity_id) → last_fired_ts
        self._alert_queue   = queue.Queue()     # consumed by SSE generator
        self._recent:       deque = deque(maxlen=50)
        self._cam_ping:     dict  = {}          # camera_id → last_seen_ts (from ping_camera)
        self._entry_times:  deque = deque(maxlen=200)  # ts of recent crossing_status=="entered"
        self._lock          = threading.Lock()

        # Load thresholds from device.json if available
        self._queue_thresh    = QUEUE_ALERT_THRESHOLD
        self._dwell_min       = DWELL_ALERT_MINUTES
        self._loiter_min      = LOITERING_MINUTES
        self._max_capacity    = 85
        self._lang            = "en"
        self._quiet_start     = None   # (hour, minute) or None
        self._quiet_end       = None
        self._load_config()

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="AlertEngine"
        )
        self._thread.start()
        logger.info("[AlertEngine] Started.")

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        try:
            from system import config as _cfg
            base = _cfg.BASE_DIR
            dpath = os.path.join(base, "device.json")
            if os.path.exists(dpath):
                with open(dpath, encoding="utf-8") as f:
                    dev = json.load(f)
                ui = dev.get("ui_settings", {})
                self._max_capacity = int(ui.get("max_capacity", 85))
                # Queue alert threshold: max_capacity / active_cameras, min 4
                self._queue_thresh = max(4, self._max_capacity // 4)
                # Language from app_timezone hint or default
                tz_name = ui.get("app_timezone", "")
                # Detect PT from timezone
                if "sao_paulo" in tz_name.lower() or "brazil" in tz_name.lower():
                    self._lang = "pt"
                # Quiet hours
                qs = ui.get("quiet_hours_start", "")
                qe = ui.get("quiet_hours_end", "")
                if qs and qe:
                    try:
                        sh, sm = map(int, qs.split(":"))
                        eh, em = map(int, qe.split(":"))
                        self._quiet_start = (sh, sm)
                        self._quiet_end   = (eh, em)
                    except Exception:
                        pass
                # Per-alert thresholds from vlm.alerts (optional)
                alerts_cfg = dev.get("vlm", {}).get("alerts", {})
                self._dwell_min  = int(alerts_cfg.get("dwell_minutes",  DWELL_ALERT_MINUTES))
                self._loiter_min = int(alerts_cfg.get("loiter_minutes", LOITERING_MINUTES))
        except Exception as e:
            logger.debug(f"[AlertEngine] config load error: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def ping_camera(self, camera_id: str) -> None:
        """Called by local_admin.set_latest_raw_frame to track liveness."""
        with self._lock:
            self._cam_ping[camera_id] = time.time()

    def notify_entry(self, camera_id: str) -> None:
        """Called by camera_processor when a person crosses the entrance line."""
        with self._lock:
            self._entry_times.append(time.time())

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def get_recent(self, n: int = 50) -> list:
        with self._lock:
            return list(self._recent)[-n:]

    def get_queue(self) -> queue.Queue:
        return self._alert_queue

    def dismiss(self, alert_id: str) -> None:
        with self._lock:
            for ev in self._recent:
                if ev.id == alert_id:
                    ev.dismissed = True
                    break

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._evaluate_rules()
            except Exception as e:
                logger.error(f"[AlertEngine] evaluation error: {e}", exc_info=True)
            self._stop.wait(EVAL_INTERVAL)

    # ── Rule evaluation ───────────────────────────────────────────────────────

    def _evaluate_rules(self) -> None:
        if self._in_quiet_hours():
            return

        now = time.time()

        # Snapshot _track_crops safely
        try:
            from core import vlm_analyst
            with vlm_analyst._track_crops_lock:
                tracks = {k: dict(v) for k, v in vlm_analyst._track_crops.items()
                          if now - v.get("ts", 0) < 12.0}
        except Exception:
            tracks = {}

        # Per-camera people counts
        cam_counts: dict = {}
        for entry in tracks.values():
            cam = entry.get("cam", "")
            if cam:
                cam_counts[cam] = cam_counts.get(cam, 0) + 1

        total = sum(cam_counts.values())

        # Rule: queue_alert — camera has many people
        for cam, count in cam_counts.items():
            if count >= self._queue_thresh:
                self._fire("queue_alert", "warning", cam, "",
                           count=count, cam=cam)

        # Rule: capacity_alert — whole store at/near max
        if total >= self._max_capacity:
            self._fire("capacity_alert", "critical", "", "store",
                       count=total)

        # Rule: dwell_alert & loitering — per-person zone time
        dwell_secs  = self._dwell_min  * 60
        loiter_secs = self._loiter_min * 60
        for gid, entry in tracks.items():
            zone         = entry.get("zone", "")
            zone_entry   = entry.get("zone_entry_ts", now)
            time_in_zone = now - zone_entry
            cam          = entry.get("cam", "")

            if zone and time_in_zone >= dwell_secs:
                self._fire("dwell_alert", "warning", cam, gid,
                           gid=gid, zone=zone,
                           duration_min=int(time_in_zone / 60))

            if (zone and time_in_zone >= loiter_secs
                    and zone.lower() not in _CHECKOUT_ZONES):
                self._fire("loitering", "warning", cam, f"loiter_{gid}",
                           gid=gid, zone=zone,
                           duration_min=int(time_in_zone / 60))

        # Rule: visitor_surge — many new entries in surge window
        surge_cutoff = now - SURGE_WINDOW_SEC
        with self._lock:
            recent_entries = sum(1 for t in self._entry_times if t >= surge_cutoff)
        surge_threshold = max(SURGE_THRESHOLD,
                              int(self._max_capacity * 0.15))
        if recent_entries >= surge_threshold:
            self._fire("visitor_surge", "info", "", "store",
                       count=recent_entries,
                       window_min=int(SURGE_WINDOW_SEC / 60))

        # Rule: camera_offline — configured camera not pinging
        try:
            from core import store_context as _sc
            configured_cams = _sc.get_all_camera_ids()
            with self._lock:
                pings = dict(self._cam_ping)
            for cam_id in configured_cams:
                last_ping = pings.get(cam_id, 0)
                if last_ping > 0 and now - last_ping > CAMERA_OFFLINE_SEC:
                    self._fire("camera_offline", "warning", cam_id,
                               f"offline_{cam_id}", cam=cam_id)
        except Exception:
            pass

    # ── Alert dispatch ────────────────────────────────────────────────────────

    def _fire(self, alert_type: str, severity: str,
              camera_id: str, entity_id: str, **data) -> None:
        """Emit an alert if not in cooldown for this (type, entity)."""
        cooldown_key = f"{alert_type}:{entity_id}"
        now = time.time()
        with self._lock:
            last = self._cooldowns.get(cooldown_key, 0)
            cd   = _COOLDOWNS.get(alert_type, 300)
            if now - last < cd:
                return
            self._cooldowns[cooldown_key] = now

        msg = self._format_message(alert_type, severity, camera_id, data)
        ev  = AlertEvent(
            id        = uuid.uuid4().hex[:8],
            type      = alert_type,
            severity  = severity,
            camera_id = camera_id,
            global_id = data.get("gid", ""),
            message   = msg,
            ts        = now,
            data      = data,
        )
        with self._lock:
            self._recent.append(ev)
        self._alert_queue.put(ev)
        logger.info(f"[AlertEngine] {severity.upper()} {alert_type} — {msg}")

    def _format_message(self, alert_type: str, severity: str,
                        camera_id: str, data: dict) -> str:
        """Format a human-readable alert message."""
        pt = (self._lang == "pt")

        try:
            from core import store_context as _sc
            cam_label = _sc.get_camera_label(camera_id) if camera_id else ""
        except Exception:
            cam_label = camera_id

        if alert_type == "queue_alert":
            n = data.get("count", "?")
            return (f"Fila se formando em {cam_label} — {n} pessoas."
                    if pt else
                    f"Queue forming at {cam_label} — {n} people.")

        if alert_type == "dwell_alert":
            gid = data.get("gid", "?")
            zone = data.get("zone", "?")
            dur  = data.get("duration_min", "?")
            return (f"Pessoa {gid} está em [{zone}] há {dur} minutos."
                    if pt else
                    f"Person {gid} has been in [{zone}] for {dur} min.")

        if alert_type == "capacity_alert":
            n = data.get("count", "?")
            return (f"Loja na capacidade máxima — {n} pessoas rastreadas."
                    if pt else
                    f"Store at max capacity — {n} people tracked.")

        if alert_type == "loitering":
            gid = data.get("gid", "?")
            zone = data.get("zone", "?")
            dur  = data.get("duration_min", "?")
            return (f"Possível permanência suspeita: pessoa {gid} em [{zone}] há {dur} min."
                    if pt else
                    f"Possible loitering: person {gid} in [{zone}] for {dur} min.")

        if alert_type == "visitor_surge":
            n   = data.get("count", "?")
            win = data.get("window_min", 2)
            return (f"{n} pessoas entraram nos últimos {win} minutos."
                    if pt else
                    f"{n} people entered in the last {win} minutes.")

        if alert_type == "camera_offline":
            return (f"{cam_label} parece offline — sem frames recebidos."
                    if pt else
                    f"{cam_label} appears offline — no frames received.")

        return alert_type

    # ── Quiet-hours gate ──────────────────────────────────────────────────────

    def _in_quiet_hours(self) -> bool:
        if self._quiet_start is None or self._quiet_end is None:
            return False
        now_dt = time.localtime()
        now_min = now_dt.tm_hour * 60 + now_dt.tm_min
        s_min   = self._quiet_start[0] * 60 + self._quiet_start[1]
        e_min   = self._quiet_end[0]   * 60 + self._quiet_end[1]
        if s_min <= e_min:
            return s_min <= now_min < e_min
        else:
            # Wraps midnight (e.g. 22:00 → 06:00)
            return now_min >= s_min or now_min < e_min


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[AlertEngine] = None
_engine_lock = threading.Lock()


def start_alert_engine() -> "AlertEngine":
    """Start (or return the existing) alert engine singleton."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = AlertEngine()
    return _engine


def get_engine() -> Optional["AlertEngine"]:
    return _engine


def get_recent_alerts(n: int = 50) -> list:
    """Return the most recent N alerts as AlertEvent objects."""
    with _engine_lock:
        if _engine is None:
            return []
        return _engine.get_recent(n)


def get_alert_queue() -> queue.Queue:
    """Return the live alert Queue for SSE streaming."""
    with _engine_lock:
        if _engine is None:
            return queue.Queue()
        return _engine.get_queue()


def dismiss_alert(alert_id: str) -> None:
    with _engine_lock:
        if _engine:
            _engine.dismiss(alert_id)


def ping_camera(camera_id: str) -> None:
    """Notify the engine that camera_id just delivered a frame."""
    with _engine_lock:
        if _engine:
            _engine.ping_camera(camera_id)


def notify_entry(camera_id: str) -> None:
    """Notify the engine that a person just entered (crossing_status='entered')."""
    with _engine_lock:
        if _engine:
            _engine.notify_entry(camera_id)
