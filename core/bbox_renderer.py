"""
bbox_renderer.py — Cinematic HUD-grade bounding box renderer for NORT.

Drop-in replacement for the inline draw_person_box / _draw_selection_highlight
methods in camera_processor.py.  Pure OpenCV — no extra dependencies.

Visual language
───────────────
• Corner-only brackets : outer 1 px grey + inner 2 px neon — no solid rectangle.
• Subtle glow          : 2 px alpha-blended pad in the track colour.
• Scan-line bar        : animated progress bar between the top-left corners.
• VLM status dot       : 4 px dot (grey / amber / green / red) inside top-right.
• Top HUD chip         : compact dark pill above top-left: "G:42  CAM1".
• Bottom info strip    : slim 20 px translucent bar: ♂ Male | adult | Zone A.
• Selection overlay    : white corner brackets (3 px) + soft pulse glow +
                         compact LOCKED label *below* the box (never overlaps
                         the identity chip above).
"""

import cv2
import numpy as np
from typing import Optional, Dict

# ── Per-track animation state ──────────────────────────────────────────────────
# Keyed by str(global_id).  Stores scanline phase (0..1) updated each render call.
_scan_state: Dict[str, float] = {}
_SCAN_SPEED = 0.025          # phase units per frame_counter tick
_SCAN_BAR_W = 40             # max pixels of the scan-line bar

# ── Colour helpers ─────────────────────────────────────────────────────────────

def _neon_color(global_id) -> tuple:
    """Map global_id to a stable, vivid HSV→BGR colour.

    Accepts both integer and string global_ids (e.g. 'camera_1_3' when
    ReID is disabled and the fallback '{cam}_{track}' string is used).
    """
    try:
        h = int(global_id) * 47 % 180
    except (ValueError, TypeError):
        # String fallback: use Python's stable hash for a consistent colour
        h = abs(hash(str(global_id))) % 180
    hsv = np.uint8([[[h, 230, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, alpha: float) -> None:
    """Alpha-blend a filled rectangle onto frame (in-place). Zero heap allocation if done right."""
    h, w = frame.shape[:2]
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(w, x2), min(h, y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return
    roi = frame[ry1:ry2, rx1:rx2]
    original = roi.copy()
    cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), color, -1)
    cv2.addWeighted(roi, alpha, original, 1 - alpha, 0, roi)





# ── VLM status colours ────────────────────────────────────────────────────────
_VLM_STATUS_COLORS = {
    "not_found": (120, 120, 120),   # grey
    "pending":   (0, 165, 255),     # amber
    "done":      (50, 220, 50),     # green
    "error":     (0, 60, 220),      # red
}

# ── Public API ────────────────────────────────────────────────────────────────

def draw_hud_box(
    frame: np.ndarray,
    bbox: tuple,
    global_id,
    local_track_id: int,
    camera_id: str = "",
    gender: str = "?",
    age_category: str = "adult",
    zone: str = "",
    crossing_indicator: str = "",
    vlm_result: Optional[dict] = None,
    frame_counter: int = 0,
    reid_enabled: bool = True,
) -> None:
    """
    Draw a cinematic HUD bounding box onto *frame* in-place.

    Parameters
    ----------
    frame           : BGR numpy array (modified in place).
    bbox            : (x1, y1, x2, y2) in pixel coordinates.
    global_id       : Cross-camera identity (int or str).  Drives colour.
    local_track_id  : Per-camera tracker ID shown as debug text.
    camera_id       : Short camera identifier shown in the top chip.
    gender          : "Male" / "Female" / "?" etc.
    age_category    : "adult" / "young" / "senior" etc.
    zone            : Zone name from polygon hit-test.
    crossing_indicator : "" | " ->" | " <-"
    vlm_result      : dict from vlm_analyst.get_result() or None.
    frame_counter   : Global frame counter – drives scan-line animation.
    reid_enabled    : Whether ReID is active (changes chip prefix).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_frame, w_frame = frame.shape[:2]
    color = _neon_color(global_id)
    box_w = x2 - x1
    box_h = y2 - y1
    corner_len = max(10, min(18, int(box_w * 0.15)))

    # ── 1. GLOW (very subtle — 2 px pad, low alpha) ───────────────────────────
    _blend_rect(frame,
                x1 - 2, y1 - 2, x2 + 2, y2 + 2,
                color, alpha=0.12)

    # ── 2. CORNER-ONLY BRACKETS (thin double-layer) ───────────────────────────
    outer_color = (160, 160, 160)   # dim grey outer
    outer_thick = 1
    inner_thick = 2                 # thin neon inner
    _gap        = 3                 # tight inset

    def _draw_corner_pair(p, dx, dy):
        cv2.line(frame, p,
                 (p[0] + dx * (corner_len + _gap), p[1]),
                 outer_color, outer_thick, cv2.LINE_AA)
        cv2.line(frame, p,
                 (p[0], p[1] + dy * (corner_len + _gap)),
                 outer_color, outer_thick, cv2.LINE_AA)
        ip = (p[0] + dx * _gap, p[1] + dy * _gap)
        cv2.line(frame, ip,
                 (ip[0] + dx * corner_len, ip[1]),
                 color, inner_thick, cv2.LINE_AA)
        cv2.line(frame, ip,
                 (ip[0], ip[1] + dy * corner_len),
                 color, inner_thick, cv2.LINE_AA)

    _draw_corner_pair((x1, y1), +1, +1)
    _draw_corner_pair((x2, y1), -1, +1)
    _draw_corner_pair((x1, y2), +1, -1)
    _draw_corner_pair((x2, y2), -1, -1)

    # ── 3. SCAN-LINE ANIMATION BAR ────────────────────────────────────────────
    track_key = str(global_id)
    phase = _scan_state.get(track_key, 0.0)
    phase = (phase + _SCAN_SPEED) % 1.0
    _scan_state[track_key] = phase

    bar_x1 = x1 + corner_len + _gap + 3
    bar_x2 = min(x2 - corner_len - _gap - 3, bar_x1 + _SCAN_BAR_W)
    bar_y  = y1 + 4

    if bar_x2 > bar_x1 and 0 <= bar_y < h_frame:
        total_w = bar_x2 - bar_x1
        fill_w  = int(total_w * phase)
        cv2.line(frame, (bar_x1, bar_y), (bar_x2, bar_y), (50, 50, 50), 1, cv2.LINE_AA)
        if fill_w > 0:
            cv2.line(frame, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y), color, 1, cv2.LINE_AA)

    # ── 4. TOP HUD CHIP (above top-left) ──────────────────────────────────────
    prefix    = "G" if reid_enabled else "ID"
    cam_short = camera_id.split("_")[-1] if "_" in camera_id else camera_id
    cam_short = cam_short[:6].upper()
    chip_text = f"{prefix}:{global_id}  {cam_short}" if cam_short else f"{prefix}:{global_id}"

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.42
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(chip_text, font, font_scale, font_thick)

    chip_pad = 4
    chip_h   = th + chip_pad * 2
    chip_w   = tw + chip_pad * 2 + 5   # +5 for accent stripe
    chip_y1  = max(0, y1 - chip_h - 3)
    chip_y2  = chip_y1 + chip_h
    chip_x1  = x1
    chip_x2  = min(w_frame, x1 + chip_w)

    if chip_y2 > chip_y1 and chip_x2 > chip_x1:
        _blend_rect(frame, chip_x1, chip_y1, chip_x2, chip_y2, (10, 10, 10), alpha=0.78)
        cv2.line(frame,
                 (chip_x1 + 2, chip_y1 + 2),
                 (chip_x1 + 2, chip_y2 - 2),
                 color, 2)
        cv2.putText(frame, chip_text,
                    (chip_x1 + 7, chip_y2 - chip_pad),
                    font, font_scale, (210, 210, 210), font_thick, cv2.LINE_AA)

    # ── 5. VLM STATUS DOT (top-right inside box) ──────────────────────────────
    vlm_status = (vlm_result or {}).get("status", "not_found")
    dot_color  = _VLM_STATUS_COLORS.get(vlm_status, (120, 120, 120))
    dot_r      = 4
    dot_cx     = x2 - dot_r - 4
    dot_cy     = y1 + dot_r + 5

    if 0 <= dot_cx < w_frame and 0 <= dot_cy < h_frame:
        cv2.circle(frame, (dot_cx, dot_cy), dot_r + 1, (15, 15, 15), -1)
        cv2.circle(frame, (dot_cx, dot_cy), dot_r,     dot_color,    -1)

    # ── 6. BOTTOM INFO STRIP ──────────────────────────────────────────────────
    strip_h  = 20
    strip_y1 = max(0, y2 - strip_h)
    strip_y2 = min(h_frame, y2)
    strip_x2 = min(w_frame, x2)

    if strip_y2 > strip_y1 and strip_x2 > x1:
        _blend_rect(frame, x1, strip_y1, strip_x2, strip_y2, (5, 5, 5), alpha=0.70)
        g_icon   = "M" if gender.lower().startswith("m") else ("F" if gender.lower().startswith("f") else "?")
        zone_str = f"  {zone}" if zone and zone.lower() not in ("", "unknown") else ""
        info     = f"{g_icon} {age_category}{zone_str}{crossing_indicator}"
        cv2.putText(frame, info,
                    (x1 + 5, strip_y2 - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (190, 190, 190), 1, cv2.LINE_AA)


def draw_selection_hud(
    frame: np.ndarray,
    bbox: tuple,
    frame_counter: int = 0,
) -> None:
    """
    Draw a selection highlight on top of the normal HUD box.

    Design rules:
      • White corners (outer 1 px grey + inner 3 px white) — slightly longer
        than normal corners so they're clearly visible over any neon colour.
      • Very soft pulsing glow (2 px pad, low alpha).
      • "LOCKED" label is placed *below* the box, never above — so it can
        never collide with the G:ID chip that draw_hud_box puts above.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_frame, x2); y2 = min(h_frame, y2)
    box_w = x2 - x1

    accent = (255, 255, 255)   # white — distinct from every per-person neon hue

    # ── 1. Soft pulsing glow (2 px pad) ──────────────────────────────────────
    pulse_alpha = 0.08 + 0.07 * abs(np.sin(frame_counter * 0.08))
    _blend_rect(frame, x1 - 2, y1 - 2, x2 + 2, y2 + 2, accent, pulse_alpha)

    # ── 2. White corner brackets (slightly longer than normal) ────────────────
    corner_len  = max(12, min(22, int(box_w * 0.18)))
    outer_thick = 1
    inner_thick = 3   # one step thicker than the normal 2 px neon
    _gap        = 3

    def _sel_corner(p, dx, dy):
        cv2.line(frame, p,
                 (p[0] + dx * (corner_len + _gap), p[1]),
                 (150, 150, 150), outer_thick, cv2.LINE_AA)
        cv2.line(frame, p,
                 (p[0], p[1] + dy * (corner_len + _gap)),
                 (150, 150, 150), outer_thick, cv2.LINE_AA)
        ip = (p[0] + dx * _gap, p[1] + dy * _gap)
        cv2.line(frame, ip,
                 (ip[0] + dx * corner_len, ip[1]),
                 accent, inner_thick, cv2.LINE_AA)
        cv2.line(frame, ip,
                 (ip[0], ip[1] + dy * corner_len),
                 accent, inner_thick, cv2.LINE_AA)

    _sel_corner((x1, y1), +1, +1)
    _sel_corner((x2, y1), -1, +1)
    _sel_corner((x1, y2), +1, -1)
    _sel_corner((x2, y2), -1, -1)

    # ── 3. LOCKED label — placed *below* the box, no overlap possible ────────
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.42
    font_thick = 1
    badge_txt  = "LOCKED"
    (tw, th), _ = cv2.getTextSize(badge_txt, font, font_scale, font_thick)

    chip_pad = 3
    chip_h   = th + chip_pad * 2
    chip_w   = tw + chip_pad * 2 + 5
    chip_y1  = y2 + 3
    chip_y2  = chip_y1 + chip_h
    chip_x1  = x1
    chip_x2  = min(w_frame, x1 + chip_w)

    if chip_y2 < h_frame and chip_x2 > chip_x1:
        _blend_rect(frame, chip_x1, chip_y1, chip_x2, chip_y2, (10, 10, 10), alpha=0.78)
        cv2.line(frame,
                 (chip_x1 + 2, chip_y1 + 2),
                 (chip_x1 + 2, chip_y2 - 2),
                 accent, 2)
        cv2.putText(frame, badge_txt,
                    (chip_x1 + 7, chip_y2 - chip_pad),
                    font, font_scale, (210, 210, 210), font_thick, cv2.LINE_AA)


def clear_track_state(global_id) -> None:
    """Remove scan-line animation state for a track that has been cleaned up."""
    _scan_state.pop(str(global_id), None)
