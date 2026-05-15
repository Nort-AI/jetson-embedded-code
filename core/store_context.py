"""
core/store_context.py — Store intelligence layer for NORT.

Assembles grounded, structured context for natural language queries by combining:
  1. Static config   — camera positions, area descriptions  (cameras.json)
  2. Zone layout     — named zones per camera               (zones_per_camera.json)
  3. Live state      — current occupancy, person locations  (vlm_analyst._track_crops)
  4. Store identity  — name, type                           (device.json ui_settings)

Design goals
------------
  A. Answer simple data queries for FREE — no VLM call needed.
  B. Enrich VLM prompts with ground-truth context so Claude answers precisely
     and grounded (knows "camera 2 = Checkout", "Person 42 is near cold aisle").
  C. Resolve natural language references:
       "camera 1"  → configured name + area description
       "person 42" → last seen camera + zone + time
       "checkout"  → camera_id via area name alias

Product context
---------------
This module is the "brain" behind the Nort natural language interface.
The vision: store operators ask plain questions ("is there a queue at checkout?",
"where is person 42?", "is anyone acting suspicious at entrance?") and the system
answers instantly — using tracker data when possible, VLM only when a visual
judgment is genuinely needed. This keeps Claude API costs near zero for most
queries while delivering real intelligence about what is happening in the store.
"""

import json
import os
import re
import time
import threading
from typing import Optional

try:
    from system.logger_setup import setup_logger
    logger = setup_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION LOADING
# ═══════════════════════════════════════════════════════════════════════════════

_cfg_lock = threading.Lock()
_cameras_cfg: dict = {}     # camera_id → {name, source, area_description, ...}
_zones_cfg: dict = {}       # camera_id → [{sector_name, polygon_vertices}, ...]
_store_name: str = ""
_store_type: str = "retail store"
_client_id: str = ""
_store_id: str = ""
_loaded: bool = False


def _find_file(*relative_paths) -> Optional[str]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in relative_paths:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None


def _load_config() -> None:
    global _cameras_cfg, _zones_cfg, _store_name, _store_type
    global _client_id, _store_id, _loaded

    # ── cameras.json ──────────────────────────────────────────────────────────
    cameras_path = _find_file("cameras.json", "data_store/cameras.json",
                              "nort_data/config/cameras.json")
    if cameras_path:
        try:
            with open(cameras_path, encoding="utf-8") as f:
                _cameras_cfg = json.load(f)
        except Exception as e:
            logger.warning(f"[StoreCtx] Could not load cameras.json: {e}")

    # ── device.json — store name / client / store identity ───────────────────
    device_path = _find_file("device.json", "data_store/device.json",
                             "nort_data/config/device.json")
    if device_path:
        try:
            with open(device_path, encoding="utf-8") as f:
                dev = json.load(f)
            _client_id = dev.get("client_id", "")
            _store_id = dev.get("store_id", "")
            ui = dev.get("ui_settings", {})
            _store_name = ui.get("store_name", "") or dev.get("store_name", "")
            _store_type = dev.get("store_type", "retail store")
        except Exception as e:
            logger.warning(f"[StoreCtx] Could not load device.json: {e}")

    # ── zones_per_camera.json — named zone polygons ───────────────────────────
    zones_path = _find_file("zones_per_camera.json",
                            "data_store/zones_per_camera.json")
    if zones_path:
        try:
            with open(zones_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Navigate client_id → store_id → camera_id
            store_level = (raw.get(_client_id) or raw.get(next(iter(raw), ""), {}))
            if isinstance(store_level, dict):
                cam_level = store_level.get(_store_id) or store_level.get(
                    next(iter(store_level), ""), {})
                if isinstance(cam_level, dict):
                    for cam_id, cam_data in cam_level.items():
                        zones = cam_data.get("zones", [])
                        if zones:
                            _zones_cfg[cam_id] = zones
        except Exception as e:
            logger.warning(f"[StoreCtx] Could not load zones_per_camera.json: {e}")

    _loaded = True
    logger.info(
        f"[StoreCtx] Loaded: {len(_cameras_cfg)} cameras, "
        f"{sum(len(z) for z in _zones_cfg.values())} zones, "
        f"store='{_store_name}', type='{_store_type}'"
    )


def _ensure_loaded() -> None:
    with _cfg_lock:
        if not _loaded:
            _load_config()


def reload() -> None:
    """Force a config reload (e.g. after the operator edits cameras.json)."""
    global _loaded
    with _cfg_lock:
        _loaded = False
    _ensure_loaded()


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC ACCESSORS
# ═══════════════════════════════════════════════════════════════════════════════

def get_camera_config(camera_id: str) -> dict:
    _ensure_loaded()
    with _cfg_lock:
        return dict(_cameras_cfg.get(camera_id, {}))


def get_camera_label(camera_id: str) -> str:
    """Human-readable camera label, e.g. 'Entrada (camera_1)'."""
    cfg = get_camera_config(camera_id)
    name = cfg.get("name", "")
    if name and name.lower() != camera_id.lower():
        return f"{name} ({camera_id})"
    return camera_id


def get_camera_area_description(camera_id: str) -> str:
    """Return operator-configured description of what area this camera covers."""
    return get_camera_config(camera_id).get("area_description", "")


def get_zone_names(camera_id: str) -> list:
    """Return list of zone names configured for a camera."""
    _ensure_loaded()
    with _cfg_lock:
        zones = _zones_cfg.get(camera_id, [])
    return [z.get("sector_name", "") for z in zones if z.get("sector_name")]


def get_all_camera_ids() -> list:
    _ensure_loaded()
    with _cfg_lock:
        return [cid for cid, cfg in _cameras_cfg.items() if cfg.get("enabled", True)]


def get_store_name() -> str:
    _ensure_loaded()
    return _store_name


def get_store_type() -> str:
    _ensure_loaded()
    return _store_type


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE TRACKER STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _get_live_tracks(window_sec: float = 12.0) -> dict:
    """Return {gid_str: entry} for persons active within window_sec.
    Reads directly from vlm_analyst._track_crops to avoid circular imports.
    """
    try:
        from core import vlm_analyst
        now = time.time()
        with vlm_analyst._track_crops_lock:
            return {
                k: dict(v)
                for k, v in vlm_analyst._track_crops.items()
                if now - v.get("ts", 0) < window_sec
            }
    except Exception:
        return {}


def get_current_occupancy(window_sec: float = 12.0) -> dict:
    """Return {camera_id: count} of unique people seen in the last window_sec."""
    tracks = _get_live_tracks(window_sec)
    occ: dict = {}
    for entry in tracks.values():
        cam = entry.get("cam") or "unknown"
        occ[cam] = occ.get(cam, 0) + 1
    return occ


def get_total_people(window_sec: float = 12.0) -> int:
    return len(_get_live_tracks(window_sec))


def get_person_state(global_id: str) -> Optional[dict]:
    """Return last known state for a person or None if not tracked."""
    try:
        from core import vlm_analyst
        with vlm_analyst._track_crops_lock:
            entry = vlm_analyst._track_crops.get(str(global_id))
            return dict(entry) if entry else None
    except Exception:
        return None


def get_people_in_camera(camera_id: str, window_sec: float = 12.0) -> list:
    """Return list of (gid, entry) for people currently visible in a camera."""
    tracks = _get_live_tracks(window_sec)
    return [(gid, e) for gid, e in tracks.items() if e.get("cam") == camera_id]


def get_person_trail(global_id: str) -> list:
    """Return trail list for a person: [(cam, zone, entry_ts, exit_ts|None), ...].

    The list is ordered oldest → newest. The last entry has exit_ts=None if the
    person is still in that zone. Requires save_crop() to be called with zone= kwarg.
    """
    try:
        from core import vlm_analyst
        with vlm_analyst._track_crops_lock:
            entry = vlm_analyst._track_crops.get(str(global_id))
            if not entry:
                return []
            trail = list(entry.get("trail", []))
            # Append current open zone as last stop (no exit_ts yet)
            zone = entry.get("zone", "")
            if zone:
                trail.append((
                    entry.get("cam", ""),
                    zone,
                    entry.get("zone_entry_ts", entry.get("ts", 0)),
                    None,
                ))
        return trail
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BLOCK BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)}s"
    return f"{int(seconds / 60)}m"


def build_camera_context_block(camera_id: str) -> str:
    """
    Build a grounded context block for a camera-specific VLM query.

    Example output:
        Store: Supermercado Central (supermarket)
        Camera: Caixas (camera_2) — Checkout area, 3 POS registers visible
        Zones configured: caixa_rapido, caixa_normal
        Currently visible: 4 people
          Person 15 [caixa_rapido], in store ~3m
          Person 7, in store ~8m
          Person 22 [caixa_normal], in store ~1m
    """
    _ensure_loaded()
    lines = []

    # Store identity
    sname = get_store_name()
    stype = get_store_type()
    store_line = sname if sname else stype.capitalize()
    if sname and stype and stype != "retail store":
        store_line = f"{sname} ({stype})"
    lines.append(f"Store: {store_line}")

    # Camera identity + area description
    cfg = get_camera_config(camera_id)
    cam_name = cfg.get("name", camera_id)
    area_desc = cfg.get("area_description", "")
    cam_line = f"Camera: {cam_name} ({camera_id})"
    if area_desc:
        cam_line += f" — {area_desc}"
    lines.append(cam_line)

    # Zone names
    zone_names = get_zone_names(camera_id)
    if zone_names:
        lines.append(f"Zones in view: {', '.join(zone_names)}")

    # Live people in this camera
    now = time.time()
    people = get_people_in_camera(camera_id, window_sec=12.0)
    count = len(people)
    lines.append(f"People currently visible: {count}")

    if people:
        for gid, entry in people[:8]:   # cap to keep prompt tight
            age = now - entry.get("ts", now)
            zone = entry.get("zone", "")
            line = f"  Person {gid}"
            if zone:
                line += f" [{zone}]"
            line += f", visible ~{_format_duration(age)}"
            lines.append(line)

    return "\n".join(lines)


def build_person_context_block(global_id: str) -> str:
    """
    Build a grounded context block for a person-specific VLM query.

    Example output:
        Store: Supermercado Central (supermarket)
        Person 42:
          Last seen: Caixas (camera_2) — Checkout area
          Zone: caixa_rapido
          Last seen: 8s ago
    """
    _ensure_loaded()
    sname = get_store_name()
    stype = get_store_type()
    store_line = sname if sname else stype.capitalize()
    if sname and stype and stype != "retail store":
        store_line = f"{sname} ({stype})"

    entry = get_person_state(global_id)
    if not entry:
        return (
            f"Store: {store_line}\n"
            f"Person {global_id}: not currently tracked "
            f"(may have left the store or ID reassigned)."
        )

    now = time.time()
    cam = entry.get("cam", "?")
    cam_cfg = get_camera_config(cam)
    cam_name = cam_cfg.get("name", cam)
    area_desc = cam_cfg.get("area_description", "")
    zone = entry.get("zone", "")
    age_sec = now - entry.get("ts", now)

    lines = [
        f"Store: {store_line}",
        f"Person {global_id}:",
        f"  Last seen in: {cam_name} ({cam})" + (f" — {area_desc}" if area_desc else ""),
    ]
    if zone:
        lines.append(f"  Zone: {zone}")
    lines.append(f"  Last seen: {_format_duration(age_sec)} ago")

    # Append trail if available
    trail_block = build_person_trail_block(global_id)
    if trail_block:
        lines.append(trail_block)

    return "\n".join(lines)


def build_person_trail_block(global_id: str) -> str:
    """
    Format a person's cross-camera journey as a readable string for VLM context.

    Example:
        Journey (last stops):
          Entrada (camera_1, ~8m ago) → Corredor Central (camera_2, ~3m) → Caixas (camera_3, now ~1m)
    """
    trail = get_person_trail(global_id)
    if not trail:
        return ""

    now = time.time()
    parts = []
    for cam, zone, entry_ts, exit_ts in trail:
        cam_name = get_camera_config(cam).get("name", cam) if cam else cam
        dwell = _format_duration((exit_ts or now) - entry_ts)
        if exit_ts is None:
            parts.append(f"{cam_name} [{zone}] now ~{dwell}")
        else:
            ago = _format_duration(now - exit_ts)
            parts.append(f"{cam_name} [{zone}] ~{dwell} ({ago} ago)")

    if not parts:
        return ""
    return "Journey (last stops):\n  " + " → ".join(parts)


def build_store_overview_block() -> str:
    """
    Full store snapshot for general queries.

    Example:
        Store: Supermercado Central (supermarket)
        Total people tracked right now: 14
          Entrada (camera_1): 3 people
          Caixas (camera_2): 4 people
          Laticínios (camera_3): 2 people
          ...
    """
    _ensure_loaded()
    sname = get_store_name()
    stype = get_store_type()
    store_line = sname if sname else stype.capitalize()
    if sname and stype and stype != "retail store":
        store_line = f"{sname} ({stype})"

    occ = get_current_occupancy()
    total = sum(occ.values())
    lines = [
        f"Store: {store_line}",
        f"Total people tracked right now: {total}",
    ]

    cam_ids = get_all_camera_ids()
    for cam_id in cam_ids:
        cfg = get_camera_config(cam_id)
        name = cfg.get("name", cam_id)
        count = occ.get(cam_id, 0)
        lines.append(f"  {name} ({cam_id}): {count} {'person' if count == 1 else 'people'}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY CLASSIFICATION & ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

# Queries that can be answered from tracker data alone — no VLM needed.
# Patterns are accent-insensitive where possible; re.IGNORECASE is applied at match time.
_DATA_PATTERNS = [
    r'\b(how many|quantas?|quantos?)\b',
    r'\b(count\b|contar|contagem)\b',
    r'\b(where is|onde (est[aá]|fica|est[aã]o))\b',
    r'\b(who is|quem (est[aá]|\xe9|s[aã]o))\b',
    r'\b(how long|h[aá] quanto tempo|faz quanto tempo|desde quando)\b',
    r'\b(occupancy|ocupa[cç][aã]o|lota[cç][aã]o)\b',
    r'\b(total|number of|n[uú]mero de)\b',
    r'\b(list\b|listar)\b',
    r'\b(currently|agora|no momento|right now)\b',
    r'\b(pessoa|person)\s+\d+\b',   # "pessoa 42" / "person 7" always data-resolvable
    r'\b(where has|por onde (andou|passou|foi))\b',  # trail queries
    r'\b(journey|trajeto|percurso|caminho)\b',
    # Temporal / historical — answered by analytics_query
    r'\b(today|hoje|this morning|esta manh[ãa])\b',
    r'\b(yesterday|ontem)\b',
    r'\b(peak|pico|rush hour|hor[aá]rio de pico|busiest)\b',
    r'\b(average|m[eé]dia|typically|normalmente|usually)\b',
    r'\b(last hour|[uú]ltima hora|last \d+ min)\b',
    r'\b(trend|tend[eê]ncia|compared|comparado)\b',
    r'\b(how many.*came|quantas?.*entr)\b',  # "how many came in today"
]

# Queries that genuinely require looking at images.
_VISUAL_PATTERNS = [
    r'\b(wearing|vestindo|usando|roupa|outfit)\b',
    r'\b(doing|fazendo|comportamento|behavior)\b',
    r'\b(carrying|carregando|segurando|holding)\b',
    r'\b(describe|descreva?|como (está|parece|ficou))\b',
    r'\b(suspicious|suspeito|estranh|weird)\b',
    r'\b(interact|interagindo|conversando|talking)\b',
    r'\b(what.{0,25}(happen|going on|ocorrendo|acontecendo))\b',
    r'\b(o que (está acontecendo|acontece|está rolando))\b',
    r'\b(analyze|analisar|análise)\b',
    r'\b(look like|parece|aparência|appearance)\b',
    r'\b(show me|me mostre|me conta sobre)\b',
    r'\b(queue|fila|crowded|lotado|busy)\b',
]


def _resolve_camera_id(text: str) -> Optional[str]:
    """
    Resolve camera reference from natural language.
    Handles numeric IDs ("camera 1", "cam 3"), and area name aliases
    configured in cameras.json ("checkout", "entrada", "caixa", etc.).
    """
    low = text.lower()

    # Numeric: "camera 1", "cam_2", "câmera 3"
    m = re.search(r'\bcam(?:era|[aâ]ra|)[\s_-]?(\d+)\b', low)
    if m:
        return f"camera_{m.group(1)}"

    # Named alias from cameras.json
    _ensure_loaded()
    with _cfg_lock:
        cams = dict(_cameras_cfg)
    for cam_id, cfg in cams.items():
        name = cfg.get("name", "").lower()
        area = cfg.get("area_description", "").lower()
        aliases = [a.lower() for a in cfg.get("aliases", [])]
        candidates = [name] + aliases
        # Also try first word of area description
        if area:
            candidates.append(area.split()[0] if area.split() else "")
        if any(c and c in low for c in candidates):
            return cam_id

    return None


def _resolve_person_id(text: str) -> Optional[str]:
    """
    Resolve person ID reference from natural language.
    Handles: "person 42", "pessoa 7", "id 15", "gid 3", "track 99".
    """
    m = re.search(
        r'\b(?:person|pessoa|id|gid|track|individual)[\s#_\-]?(\d+)\b',
        text, re.IGNORECASE
    )
    return m.group(1) if m else None


_TEMPORAL_PATTERNS = [
    r'\b(today|hoje|this morning|esta manh[ãa])\b',
    r'\b(yesterday|ontem)\b',
    r'\b(peak|pico|rush hour|hor[aá]rio de pico|busiest)\b',
    r'\b(average|m[eé]dia|typically|normalmente|usually)\b',
    r'\b(last hour|[uú]ltima hora|last \d+ min)\b',
    r'\b(trend|tend[eê]ncia|compared|comparado)\b',
    r'\b(how many.*came|quantas?.*entr)\b',
]

_TRAIL_PATTERNS = [
    r'\b(where has|por onde (andou|passou|foi))\b',
    r'\b(journey|trajeto|percurso|caminho)\b',
    r'\b(been to|visitou|passou por)\b',
    r'\b(history|hist[oó]rico)\b',
]


def classify_query(query: str) -> dict:
    """
    Classify a natural language store query.

    Returns:
        {
          "needs_vlm":  bool    — False means answer from tracker data for free
          "type":       str     — "scene" | "person" | "store" | "historical" | "trail"
          "camera_id":  str|None
          "person_id":  str|None
          "is_data":    bool
          "is_visual":  bool
          "is_temporal": bool
          "is_trail":   bool
        }
    """
    low = query.lower()

    camera_id = _resolve_camera_id(query)
    person_id = _resolve_person_id(query)

    is_data     = any(re.search(p, low) for p in _DATA_PATTERNS)
    is_visual   = any(re.search(p, low) for p in _VISUAL_PATTERNS)
    is_temporal = any(re.search(p, low) for p in _TEMPORAL_PATTERNS)
    is_trail    = bool(person_id) and any(re.search(p, low) for p in _TRAIL_PATTERNS)

    # Pure data → free; pure visual → VLM; ambiguous → VLM (safer default)
    needs_vlm = is_visual or (not is_data)

    if is_trail:
        qtype = "trail"
    elif is_temporal:
        qtype = "historical"
    elif person_id:
        qtype = "person"
    elif camera_id:
        qtype = "scene"
    else:
        qtype = "store"

    return {
        "needs_vlm":   needs_vlm,
        "type":        qtype,
        "camera_id":   camera_id,
        "person_id":   person_id,
        "is_data":     is_data,
        "is_visual":   is_visual,
        "is_temporal": is_temporal,
        "is_trail":    is_trail,
    }


def answer_data_query(query: str, cls: dict, lang: str = "en") -> Optional[str]:
    """
    Try to answer a data query directly from tracker state or historical DB.
    Returns a natural language answer string, or None if VLM is needed.
    """
    if cls["needs_vlm"]:
        return None

    low = query.lower()
    pt = (lang == "pt")

    # ── Person trail / journey ─────────────────────────────────────────────────
    if cls.get("type") == "trail" and cls["person_id"]:
        gid = cls["person_id"]
        trail = get_person_trail(gid)
        if not trail:
            return (
                f"Sem trajeto registrado para a pessoa {gid} — pode ter acabado de entrar."
                if pt else
                f"No journey recorded for person {gid} — they may have just entered."
            )
        block = build_person_trail_block(gid)
        return block or (
            f"Pessoa {gid} ainda não mudou de zona."
            if pt else
            f"Person {gid} hasn't moved between zones yet."
        )

    # ── Historical / temporal queries ─────────────────────────────────────────
    if cls.get("type") == "historical" or cls.get("is_temporal"):
        try:
            from core import analytics_query as _aq
            return _aq.answer_natural_history_query(query, cls, lang)
        except Exception as _e:
            logger.debug(f"[StoreCtx] analytics_query failed: {_e}")
            return None   # fall through to VLM

    # ── Person location ────────────────────────────────────────────────────────
    if cls["type"] == "person" and cls["person_id"]:
        gid = cls["person_id"]
        entry = get_person_state(gid)
        if not entry:
            return (
                f"Pessoa {gid} não está sendo rastreada no momento."
                if pt else
                f"Person {gid} is not currently being tracked."
            )
        cam = entry.get("cam", "?")
        cam_label = get_camera_label(cam)
        zone = entry.get("zone", "")
        age = _format_duration(time.time() - entry.get("ts", time.time()))

        if re.search(r'\b(where|onde)\b', low):
            loc = cam_label + (f" [{zone}]" if zone else "")
            return (
                f"Pessoa {gid} foi vista em {loc} ({age} atrás)."
                if pt else
                f"Person {gid} was last seen at {loc} ({age} ago)."
            )

        if re.search(r'\b(how long|quanto tempo|since)\b', low):
            return (
                f"Pessoa {gid} está visível há aproximadamente {age}."
                if pt else
                f"Person {gid} has been visible for approximately {age}."
            )

    # ── Camera / scene occupancy count ────────────────────────────────────────
    if re.search(r'\b(how many|quantas?|quantos?|count|total)\b', low):
        if cls["camera_id"]:
            people = get_people_in_camera(cls["camera_id"])
            n = len(people)
            label = get_camera_label(cls["camera_id"])
            return (
                f"{n} {'pessoa' if n == 1 else 'pessoas'} visíveis em {label} agora."
                if pt else
                f"{n} {'person' if n == 1 else 'people'} currently visible in {label}."
            )
        else:
            total = get_total_people()
            return (
                f"{total} {'pessoa rastreada' if total == 1 else 'pessoas rastreadas'} na loja agora."
                if pt else
                f"{total} {'person' if total == 1 else 'people'} currently tracked in the store."
            )

    # ── Overall store status ───────────────────────────────────────────────────
    if cls["type"] == "store":
        occ = get_current_occupancy()
        total = sum(occ.values())
        cam_ids = get_all_camera_ids()
        if pt:
            parts = [f"{total} pessoas na loja agora:"]
            for cid in cam_ids:
                n = occ.get(cid, 0)
                parts.append(f"  {get_camera_label(cid)}: {n} pessoa{'s' if n != 1 else ''}")
            return "\n".join(parts)
        else:
            parts = [f"{total} people in store right now:"]
            for cid in cam_ids:
                n = occ.get(cid, 0)
                parts.append(f"  {get_camera_label(cid)}: {n} person{'s' if n != 1 else ''}")
            return "\n".join(parts)

    return None  # couldn't answer — caller should use VLM


# ═══════════════════════════════════════════════════════════════════════════════
# VLM-OPTIMISED FRAME RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_vlm_frame(frame_bgr, camera_id: str) -> "np.ndarray":
    """
    Render a minimal, VLM-friendly annotated frame.

    Unlike the full MJPEG stream (thick colored boxes, heatmaps, trajectory
    lines), this uses subtle visual cues designed to give Claude spatial
    context without introducing noise that confuses the model:

      • Thin gray zone outlines + zone name text
      • Small numbered white circles above each currently tracked person

    Args:
        frame_bgr:  Raw BGR numpy array from camera
        camera_id:  Used to look up zone polygons and current occupants

    Returns:
        BGR frame with minimal annotations (copy of input if no data available)
    """
    import cv2
    import numpy as np

    draw = frame_bgr.copy()

    # ── Zone outlines ──────────────────────────────────────────────────────────
    _ensure_loaded()
    with _cfg_lock:
        zones = list(_zones_cfg.get(camera_id, []))

    for zone in zones:
        name = zone.get("sector_name", "")
        verts = zone.get("polygon_vertices", [])
        if not verts or not name:
            continue
        try:
            pts = np.array(verts, dtype=np.int32).reshape(-1, 1, 2)
            # Very faint outline — just enough to show zone boundary
            cv2.polylines(draw, [pts], True, (180, 180, 180), 1, cv2.LINE_AA)
            # Zone name at centroid
            cx = int(np.mean(pts[:, 0, 0]))
            cy = int(np.mean(pts[:, 0, 1]))
            cv2.putText(draw, name, (cx - 4, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    # ── Person numbered markers ────────────────────────────────────────────────
    # Use crop_bounds stored at save_crop time to place person number circles
    try:
        from core import vlm_analyst
        now = time.time()
        person_num = 1
        with vlm_analyst._track_crops_lock:
            for gid, entry in vlm_analyst._track_crops.items():
                if entry.get("cam") != camera_id:
                    continue
                if now - entry.get("ts", 0) > 12.0:
                    continue
                bounds = entry.get("crop_bounds")
                if not bounds:
                    continue
                try:
                    x1, y1, x2, y2 = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
                    cx = (x1 + x2) // 2
                    cy_top = max(14, y1 - 10)
                    # White circle + dark number
                    cv2.circle(draw, (cx, cy_top), 13, (255, 255, 255), -1)
                    cv2.circle(draw, (cx, cy_top), 13, (120, 120, 120), 1)
                    label = str(gid)[-3:]  # last 3 digits of global_id (short enough)
                    fs = 0.35 if len(label) > 2 else 0.42
                    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
                    cv2.putText(draw, label,
                                (cx - tw // 2, cy_top + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 30, 30), 1, cv2.LINE_AA)
                    person_num += 1
                except Exception:
                    pass
    except Exception:
        pass

    return draw
