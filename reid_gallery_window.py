"""
reid_gallery_window.py — Live Re-ID Identity Gallery
=====================================================
Renders a separate OpenCV window that displays all currently known
global identities in a scrollable grid. Each card shows:

    ┌───────────────────┐
    │  [crop 1][crop 2] │   ← up to MAX_THUMBS person crops
    │  G:14             │   ← global ID (large)
    │  cam=2  hits=7    │   ← last seen camera + match count
    │  adult · Male     │   ← latest attributes (if available)
    └───────────────────┘

Usage (from main.py)
--------------------
    from reid_gallery_window import ReidGalleryWindow

    gallery = ReidGalleryWindow(reid_manager)
    # inside the display loop:
    gallery.update(frame_index)   # only re-renders every N frames
    gallery.show()                # calls cv2.imshow internally
    # on shutdown:
    gallery.destroy()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from reid_manager import ReIDManager

# ── Layout constants (all in pixels) ──────────────────────────────────────────
THUMB_W        = 64    # thumbnail width
THUMB_H        = 128   # thumbnail height
MAX_THUMBS     = 3     # how many crops to display per card
CARD_PAD       = 8     # inner padding
CARD_GAP       = 10    # gap between cards
LABEL_H        = 52    # height of the text area below the thumbnails
COLS           = 6     # cards per row
BG_COLOR       = (30, 30, 30)
CARD_BG        = (50, 50, 50)
TEXT_COLOR     = (230, 230, 230)
DIM_COLOR      = (140, 140, 140)
WINDOW_NAME    = "Re-ID Identity Gallery"
UPDATE_EVERY_N = 15    # re-render every N frames (saves CPU)
STALE_SECS     = 60    # grey-out cards not seen for this many seconds


def _generate_color(gid: int) -> tuple:
    """Same deterministic colour as camera_processor.generate_color."""
    np.random.seed(int(gid))
    return tuple(int(v) for v in np.random.randint(50, 255, 3))


def _placeholder_thumb(color: tuple) -> np.ndarray:
    """Return a solid-colour thumbnail (used when no crop is available yet)."""
    img = np.full((THUMB_H, THUMB_W, 3), color, dtype=np.uint8)
    # Draw a question-mark in the centre
    (tw, th), _ = cv2.getTextSize("?", cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    cv2.putText(img, "?",
                (THUMB_W // 2 - tw // 2, THUMB_H // 2 + th // 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class ReidGalleryWindow:
    """
    Manages the gallery OpenCV window lifecycle.

    Parameters
    ----------
    reid_manager : ReIDManager
        The shared manager whose gallery is visualised.
    max_rows : int
        Maximum number of card rows to show (window height clamp).
    """

    def __init__(self, reid_manager: "ReIDManager", max_rows: int = 6):
        self._rm          = reid_manager
        self._max_rows    = max_rows
        self._frame_idx   = 0
        self._last_canvas = None   # cached rendered frame

        card_w = MAX_THUMBS * THUMB_W + (MAX_THUMBS - 1) * CARD_PAD + CARD_PAD * 2
        card_h = THUMB_H + LABEL_H + CARD_PAD * 2
        self._card_w = card_w
        self._card_h = card_h

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        win_w = COLS * card_w + (COLS + 1) * CARD_GAP
        win_h = max_rows * card_h + (max_rows + 1) * CARD_GAP + 36  # 36 for header
        cv2.resizeWindow(WINDOW_NAME, win_w, win_h)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame_index: int = 0) -> None:
        """Re-render the gallery canvas (throttled to every UPDATE_EVERY_N frames)."""
        self._frame_idx = frame_index
        if frame_index % UPDATE_EVERY_N != 0:
            return
        self._last_canvas = self._render()

    def show(self) -> None:
        """Push the latest canvas to the OpenCV window."""
        if self._last_canvas is not None:
            cv2.imshow(WINDOW_NAME, self._last_canvas)

    def destroy(self) -> None:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        with self._rm._lock:
            entries = sorted(self._rm._gallery.values(), key=lambda e: e.global_id)

        now   = time.monotonic()
        cols  = COLS
        rows  = max(1, -(-len(entries) // cols))  # ceiling division
        rows  = min(rows, self._max_rows)

        win_w = cols * self._card_w + (cols + 1) * CARD_GAP
        win_h = rows * self._card_h + (rows + 1) * CARD_GAP + 36

        canvas = np.full((win_h, win_w, 3), BG_COLOR, dtype=np.uint8)

        # ── Header bar ──────────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (win_w, 34), (20, 20, 20), -1)
        cv2.putText(canvas,
                    f"Re-ID Gallery — {len(entries)} identities",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Cards ────────────────────────────────────────────────────────────
        for idx, entry in enumerate(entries):
            if idx >= cols * rows:
                break  # overflow — increase max_rows if needed

            row, col = divmod(idx, cols)
            cx = CARD_GAP + col * (self._card_w + CARD_GAP)
            cy = 36 + CARD_GAP + row * (self._card_h + CARD_GAP)

            self._draw_card(canvas, cx, cy, entry, now)

        return canvas

    def _draw_card(self, canvas: np.ndarray, cx: int, cy: int,
                   entry, now: float) -> None:
        cw, ch = self._card_w, self._card_h
        color  = _generate_color(entry.global_id)
        stale  = (now - entry.last_seen) > STALE_SECS

        # Card background
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), CARD_BG, -1)

        # Coloured top border (3 px) — greyed out if stale
        border_col = (80, 80, 80) if stale else color
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + 3), border_col, -1)

        # ── Thumbnails ───────────────────────────────────────────────────────
        thumbs = entry.thumbnails[:MAX_THUMBS]
        n_show = max(1, len(thumbs))  # at least one placeholder

        total_thumb_w = n_show * THUMB_W + (n_show - 1) * CARD_PAD
        x_start = cx + (cw - total_thumb_w) // 2

        for t_idx in range(n_show):
            tx = x_start + t_idx * (THUMB_W + CARD_PAD)
            ty = cy + CARD_PAD + 3

            if t_idx < len(thumbs):
                thumb = cv2.resize(thumbs[t_idx], (THUMB_W, THUMB_H))
            else:
                thumb = _placeholder_thumb(color)

            # Grey-tint stale thumbnails
            if stale:
                thumb = cv2.addWeighted(thumb, 0.4,
                                        np.full_like(thumb, 80), 0.6, 0)

            # Paste into canvas (bounds check)
            y_end = min(ty + THUMB_H, canvas.shape[0])
            x_end = min(tx + THUMB_W, canvas.shape[1])
            if ty < canvas.shape[0] and tx < canvas.shape[1]:
                canvas[ty:y_end, tx:x_end] = thumb[:y_end - ty, :x_end - tx]

            # Thin border around each thumb
            cv2.rectangle(canvas, (tx, ty), (tx + THUMB_W - 1, ty + THUMB_H - 1),
                          (100, 100, 100), 1)

        # ── Text area ────────────────────────────────────────────────────────
        label_y = cy + CARD_PAD + 3 + THUMB_H + 6

        # Large global-ID number
        gid_text  = f"G:{entry.global_id}"
        gid_col   = (120, 120, 120) if stale else color
        cv2.putText(canvas, gid_text,
                    (cx + CARD_PAD, label_y + 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, gid_col, 1, cv2.LINE_AA)

        # Camera + hit count
        age_secs = int(now - entry.last_seen)
        age_str  = f"{age_secs}s ago" if stale else "active"
        sub_text = f"cam={entry.camera_id.split('_')[-1]}  hits={entry.hit_count}  {age_str}"
        cv2.putText(canvas, sub_text,
                    (cx + CARD_PAD, label_y + 34),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, DIM_COLOR, 1, cv2.LINE_AA)
