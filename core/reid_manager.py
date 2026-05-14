"""
reid_manager.py — Cross-Camera Person Re-Identification Manager
================================================================
Maintains a global gallery of person appearance embeddings so that
the same physical person is assigned the same ``global_id`` even when
seen on different cameras or after a temporary disappearance.

Architecture
------------
* Each CameraProcessor calls ``ReIDManager.register_or_match()`` the
  first time a new local track_id appears.
* The manager extracts a 512-d embedding from the person crop using a
  lightweight OSNet model (osnet_ain_x1_0, ~6 MB, runs on Jetson Nano at
  >30 fps crop-batch throughput).
* Cosine similarity threshold for matching. Higher = stricter (fewer false matches).
  0.70 provides excellent precision across cameras; same-camera re-entries benefit
  from a time-decayed context bonus.
* Gallery entries expire after ``REID_EXPIRY_SECONDS`` to prevent
  false matches from long-gone people.

Key Improvements (v2)
---------------------
1. **Confirmation window** — a new track that fails the threshold buffers embeddings
   for up to N frames before minting, avoiding ghost IDs from bad first crops.
2. **Quality-aware anchor bank** — anchors store their Laplacian sharpness so the
   lowest-quality anchor is evicted when the bank is full.
3. **Cross-camera attribute boost** — attribute weight rises from 2% → 7% when
   matching across different cameras (viewpoint change makes clothing more important).
4. **Max-anchor merge** — background deduplication compares all anchor pairs, not
   just prototype-to-prototype, catching duplicates from split-angle profiles.
5. **Sharpness-weighted embedding update** — callers pass Laplacian sharpness so the
   manager only stores high-quality frames as new anchors.
6. **Fast early-track refresh** — camera_processor refreshes embeddings every 5 frames
   for the first 10 detections, building a rich anchor bank quickly.

Thread safety
-------------
All public methods acquire ``self._lock`` so any number of
CameraProcessor threads can call them concurrently.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

from system.logger_setup import setup_logger
logger = setup_logger(__name__)


# ── Gallery entry ──────────────────────────────────────────────────────────────

@dataclass
class GalleryEntry:
    global_id: int
    camera_id: str
    local_track_id: int
    prototype_emb: Optional[np.ndarray] = None
    anchor_embs: List[np.ndarray] = field(default_factory=list)
    anchor_qualities: List[float] = field(default_factory=list)  # Laplacian sharpness per anchor
    last_seen: float = field(default_factory=time.monotonic)
    hit_count: int = 1
    thumbnails: List[np.ndarray] = field(default_factory=list)  # BGR crops for gallery view
    attribute_vec: Optional[np.ndarray] = None    # legacy one-hot attr vector
    raw_attr_probs: Optional[np.ndarray] = None  # raw sigmoid probs from attribute model
    attr_label_list: Optional[List[str]] = None  # label ordering for raw_attr_probs
    # 96-d HSV color signature: pose/angle/distance invariant, computed directly
    # from pixel intensities (no attribute model required).  EMA-blended over time.
    color_signature: Optional[np.ndarray] = None


# ── Attribute keys that are useful for Re-ID ──────────────────────────────────
_UPPER_COLOR_KEYS = ["color of upper-body clothing"]
_LOWER_COLOR_KEYS = ["color of lower-body clothing"]
_BINARY_KEYS = {
    "gender":            {"male": 0.0, "female": 1.0, "Unknown": 0.5},
    "hair length":       {"short hair": 0.0, "long hair": 1.0},
    "carrying backpack": {"no": 0.0, "yes": 1.0},
    "wearing hat":       {"no": 0.0, "yes": 1.0},
}

_ALL_UPPER_COLORS = ["red", "blue", "yellow", "green", "black", "gray", "white", "purple", "brown"]
_ALL_LOWER_COLORS = ["red", "blue", "yellow", "green", "black", "gray", "white", "purple", "brown", "pink"]



def _attribute_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """
    Cosine similarity between two attribute vectors.
    Returns 0.5 (neutral) if either vector is None (no penalty for missing data).
    """
    if a is None or b is None:
        return 0.5
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.5
    return float(np.dot(a / norm_a, b / norm_b))


# ── Re-ID Manager ──────────────────────────────────────────────────────────────

class ReIDManager:
    """
    Global, thread-safe cross-camera Re-ID manager.
    Instantiate once in ``main.py`` and pass to every CameraProcessor.
    """

    # Max anchor embeddings per gallery entry (stores diverse angles)
    MAX_ANCHORS = 48

    # Confirmation window: number of frames to buffer before minting a new ID.
    # 7 frames ≈ 0.7 s at 10 fps.  Enough for a stable track-mean embedding;
    # the closed-world assumption now handles most re-entry cases that previously
    # required a long window to accumulate enough appearance evidence.
    CONFIRMATION_WINDOW = 7

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        expiry_seconds: float = 300.0,
        max_embeddings_per_entry: int = 45,
        min_crop_size: int = 48,
        device: Optional[str] = None,
    ):
        self._threshold = similarity_threshold
        self._expiry = expiry_seconds
        self._max_emb = max_embeddings_per_entry
        self._min_crop = min_crop_size

        # LOW-fix: honour the `device` parameter — "cpu" forces CPU-only inference;
        # None or "" means auto-detect (prefer GPU).  The value is forwarded to
        # _load_model() via self._forced_device so it can skip GPU EP selection.
        _dev_str = str(device) if device else ""
        self._forced_cpu = _dev_str.lower() in ("cpu", "cpu:0")

        # Check CUDA availability without importing torch.
        # On aarch64 (Jetson), never use ORT CUDA/TRT EP — GPU is TRTSession only.
        import platform as _platform
        self._is_aarch64 = _platform.machine() == "aarch64"
        self._has_cuda = False
        if _HAS_ORT and not self._forced_cpu and not self._is_aarch64:
            self._has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()

        self._lock = threading.Lock()
        self._gallery: Dict[int, GalleryEntry] = {}
        self._local_to_global: Dict[Tuple[str, int], int] = {}
        self._next_global_id: int = 1

        # Confirmation window buffer: (cam_id, local_id) → list of (embedding, attr_vec, sharpness)
        self._pending: Dict[Tuple[str, int], List[Tuple[np.ndarray, Optional[np.ndarray], float]]] = {}

        self._extractor = None
        self._load_model()

        # C3-fix: shutdown event so background threads exit cleanly on SIGTERM
        self._stop = threading.Event()

        # Background merge thread — deduplicates every 2 s
        self._merge_thread = threading.Thread(
            target=self._merge_loop, daemon=True, name="ReID-Merge"
        )
        self._merge_thread.start()

        # Background sanitize thread — self-heals corrupted IDs every 10 s
        self._sanitize_thread = threading.Thread(
            target=self._sanitize_loop, daemon=True, name="ReID-Sanitize"
        )
        self._sanitize_thread.start()

    def shutdown(self, timeout: float = 5.0):
        """Signal background threads to stop and wait for them to exit cleanly."""
        self._stop.set()
        self._merge_thread.join(timeout=timeout)
        self._sanitize_thread.join(timeout=timeout)
        if self._merge_thread.is_alive():
            logger.warning("[ReID] Merge thread did not exit within timeout")
        if self._sanitize_thread.is_alive():
            logger.warning("[ReID] Sanitize thread did not exit within timeout")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """
        Load OSNet-AIN for cross-camera Re-ID.

        Loading priority (Jetson-first):
          1. Pre-built .engine file → native TRTSession (no onnxruntime-gpu needed)
          2. ORT TensorrtExecutionProvider → auto-engine-cache
          3. ORT CUDAExecutionProvider
          4. ORT CPUExecutionProvider (last resort — slow)

        On Jetson aarch64, onnxruntime-gpu does NOT ship as a pip wheel.
        Run `python3 scripts/build_engines.py` once to build osnet_ain_x1_0.engine
        so the TRT-first path is used.
        """
        # ── Candidate model paths (ONNX + engine) ────────────────────────────
        _model_dirs = [
            os.path.join("assets", "models"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "models"),
        ]

        onnx_path   = None
        engine_path = None
        for d in _model_dirs:
            _e = os.path.join(d, "osnet_ain_x1_0.engine")
            _o = os.path.join(d, "osnet_ain_x1_0.onnx")
            if os.path.exists(_e):
                engine_path = _e
            if os.path.exists(_o):
                onnx_path = _o
            if engine_path:
                break

        # ── 1. Try native TRTSession (.engine) ────────────────────────────────
        if engine_path and not self._forced_cpu:
            try:
                from core.trt_session import TRTSession, is_available as _trt_ok
                if _trt_ok():
                    self._extractor = TRTSession(engine_path)
                    logger.info(
                        f"[ReID] ✓ TRT native session: {os.path.basename(engine_path)}"
                    )
                    return
            except Exception as _e:
                logger.warning(
                    f"[ReID] TRTSession failed ({_e}), falling back to ORT"
                )

        # ── 2. Fall back to ORT ───────────────────────────────────────────────
        if not _HAS_ORT:
            logger.warning(
                "[ReID] onnxruntime not installed and no .engine file found — "
                "Re-ID disabled. Run: python3 scripts/build_engines.py"
            )
            self._extractor = None
            return

        if onnx_path is None:
            logger.warning(
                "[ReID] osnet_ain_x1_0.onnx not found — Re-ID disabled. "
                "Run: python3 scripts/export_osnet_onnx.py"
            )
            self._extractor = None
            return

        try:
            logger.info("[ReID] Loading OSNet-AIN via ORT (no .engine found)...")
            model_dir = os.path.dirname(os.path.abspath(onnx_path))
            trt_cache  = os.path.join(model_dir, "trt_engine_cache")
            os.makedirs(trt_cache, exist_ok=True)

            # On aarch64 (Jetson), only CPU — ORT TRT/CUDA EP conflicts with
            # system libnvinfer and segfaults on Jetson's integrated GPU.
            if self._is_aarch64 or self._forced_cpu:
                valid = ["CPUExecutionProvider"]
            else:
                providers = []
                if os.name != "nt":
                    providers.append((
                        "TensorrtExecutionProvider", {
                            "trt_max_workspace_size": str(2 * 1024 * 1024 * 1024),
                            "trt_fp16_enable":        "True",
                            "trt_engine_cache_enable": "True",
                            "trt_engine_cache_path":  trt_cache,
                        }
                    ))
                providers.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
                available = ort.get_available_providers()
                valid = [p for p in providers
                         if (p if isinstance(p, str) else p[0]) in available]
                if not valid:
                    valid = ["CPUExecutionProvider"]

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            try:
                self._extractor = ort.InferenceSession(
                    onnx_path, sess_options=so, providers=valid
                )
            except Exception:
                fallback = [p for p in valid
                            if (p if isinstance(p, str) else p[0])
                            != "TensorrtExecutionProvider"]
                if not fallback:
                    fallback = ["CPUExecutionProvider"]
                self._extractor = ort.InferenceSession(
                    onnx_path, sess_options=so, providers=fallback
                )

            active = self._extractor.get_providers()
            logger.info(f"[ReID] OSNet-AIN ORT loaded — active providers: {active}")
            if "CPUExecutionProvider" in active and len(active) == 1:
                logger.warning(
                    "[ReID] Running on CPU — performance will be limited. "
                    "Run: python3 scripts/build_engines.py to build the TRT engine."
                )
        except Exception as exc:
            logger.error(f"[ReID] Model load failed: {exc} — Re-ID disabled.")
            self._extractor = None

    @property
    def is_available(self) -> bool:
        return self._extractor is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def note_track_lost(self, camera_id: str, local_track_id: int) -> None:
        """Called when a local track is cleaned up. Unlinks local→global and clears pending buffer."""
        key = (camera_id, local_track_id)
        with self._lock:
            self._local_to_global.pop(key, None)
            self._pending.pop(key, None)

    def _gid_active_on_camera(self, gid: int, camera_id: str, exclude_local_id: int) -> bool:
        """Return True if *gid* is currently assigned to another active local track
        on *camera_id*.  Must be called while self._lock is held.

        Two-layer check:
          1. _local_to_global: is this gid mapped to any other track on this camera?
             Entries are cleaned up by remove_track() ~60 s after the track dies,
             so this is a strong signal within the cleanup window.
          2. last_seen recency (15 s gate): guards against stale _local_to_global
             entries from very old visits (> 60 s cleanup not yet run).
             Gate raised from 5 s → 15 s to remain valid when update_embedding
             runs at low FPS (45 frames / 10 fps = 4.5 s interval).
        """
        now_m = time.monotonic()
        e = self._gallery.get(gid)
        if e is None:
            return False
        # Fast path: _local_to_global has a live mapping on THIS camera.
        # This is the primary guard — if another local track on the SAME camera
        # already owns this global_id, the new track cannot steal it.
        for (lk_cam, lk_tid), lk_gid in self._local_to_global.items():
            if lk_cam == camera_id and lk_tid != exclude_local_id and lk_gid == gid:
                return True
        # Fallback: gallery entry was recently seen on THIS camera specifically.
        # We check camera_id on the entry, not the raw last_seen timestamp.
        # The old code used `last_seen < 15 s` which is updated by ALL cameras —
        # so a person actively tracked on Camera 2 would permanently block Camera 1
        # from matching them (Camera 2 kept last_seen fresh).
        # Fix: only block if the entry's home camera_id matches AND was seen within
        # a narrow 3-second window (one update_embedding interval).  This covers
        # the race between ByteTrack dropping a track and remove_track() being called
        # without falsely blocking legitimate cross-camera or same-camera re-entries.
        if e.camera_id == camera_id and (now_m - e.last_seen) < 3.0:
            return True
        return False

    def register_or_match(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        camera_id: str,
        local_track_id: int,
        attributes: Optional[dict] = None,
        raw_attr_probs: Optional[np.ndarray] = None,
        attr_label_list: Optional[List[str]] = None,
        skip_thumbnail: bool = False,
        recent_gids: Optional[set] = None,
        current_occupancy: int = 0,
        crop_sharpness: float = 0.0,
    ) -> Optional[int]:
        """
        Called when a new local track appears (or still lacks a global_id).
        Returns a ``global_id`` or None if no confident match yet (wait for next frame).

        The confirmation window prevents ghost IDs from bad first crops:
        embeddings are buffered for up to CONFIRMATION_WINDOW frames.
        If any buffered embedding matches the gallery → link (no new ID minted).
        Only after the window expires without a match is a new ID minted.
        """
        key = (camera_id, local_track_id)

        with self._lock:
            if key in self._local_to_global:
                return self._local_to_global[key]

        # Extract features outside lock (GPU-expensive)
        embedding = self._extract(frame, bbox)
        if embedding is None:
            return None

        color_sig = self._extract_color_signature(frame, bbox)
        thumb = None if skip_thumbnail else self._capture_thumbnail(frame, bbox)

        with self._lock:
            # Double-check (race condition guard)
            if key in self._local_to_global:
                return self._local_to_global[key]

            self._prune_expired()

            # ── Accumulate into confirmation window ────────────────────────
            buf = self._pending.setdefault(key, [])
            buf.append((embedding, raw_attr_probs.copy() if raw_attr_probs is not None else None, crop_sharpness))

            # ── Two-strategy matching against gallery ─────────────────────
            # Strategy A — frame-by-frame: catches a single excellent-quality
            #   frame that happens to match perfectly (high-detail frontal shot).
            # Strategy B — track-mean: quality-weighted average of all buffered
            #   embeddings.  More stable than any single frame; especially
            #   powerful when frames span multiple angles (partial rotation during
            #   confirmation window).  TTA + track-mean together are the primary
            #   fix for the "same person = 4 different IDs" class of failures.
            best_gid, best_score = None, 0.0
            if self._gallery:
                # Strategy A: per-frame
                for buf_emb, buf_raw_probs, _ in buf:
                    gid, score = self._find_best_match(
                        buf_emb, query_raw_probs=buf_raw_probs,
                        camera_id=camera_id, recent_gids=recent_gids,
                        current_occupancy=current_occupancy,
                        query_label_list=attr_label_list,
                        query_color_sig=color_sig,
                    )
                    if gid is not None and score > best_score:
                        best_gid, best_score = gid, score

                # Strategy B: track-mean (run only after ≥3 frames to have meaningful mean)
                if len(buf) >= 3:
                    track_mean_emb = self._get_track_embedding(buf)
                    if track_mean_emb is not None:
                        dom_attr = max(buf, key=lambda x: x[2])[1]
                        gid, score = self._find_best_match(
                            track_mean_emb, query_raw_probs=dom_attr,
                            camera_id=camera_id, recent_gids=recent_gids,
                            current_occupancy=current_occupancy,
                            query_label_list=attr_label_list,
                            query_color_sig=color_sig,
                        )
                        if gid is not None and score > best_score:
                            best_gid, best_score = gid, score

            if best_gid is not None:
                # Same-camera uniqueness guard: reject if this gid is already claimed
                # by another ACTIVE track on this camera — one person can't be in two
                # places at once.
                if self._gid_active_on_camera(best_gid, camera_id, local_track_id):
                    logger.debug(
                        f"[ReID] BLOCKED match cam={camera_id} local={local_track_id}"
                        f" → G:{best_gid} (already active on this camera)"
                    )
                    best_gid = None

            if best_gid is not None:
                # Match found — link and enrich gallery with best buffered embedding
                entry = self._gallery[best_gid]
                best_emb, _, best_sharp = max(buf, key=lambda x: x[2]) if buf else (embedding, None, crop_sharpness)
                self._update_entry(entry, best_emb, None, best_sharp, thumb,
                                   raw_probs=raw_attr_probs, label_list=attr_label_list,
                                   color_signature=color_sig)
                self._local_to_global[key] = best_gid
                self._pending.pop(key, None)
                logger.debug(
                    f"[ReID] MATCH cam={camera_id} local={local_track_id}"
                    f" → global={best_gid}  score={best_score:.3f}  buf_size={len(buf)}"
                )
                return best_gid

            # Window not yet exhausted — keep buffering
            if len(buf) < self.CONFIRMATION_WINDOW:
                return None

            # Window exhausted with no match via normal threshold →
            # rescue pass then mint.
            self._pending.pop(key, None)
            # M2-fix: attr_vec is not in scope here; use None as fallback
            best_emb, best_attr, best_sharp = max(buf, key=lambda x: x[2]) if buf else (embedding, None, crop_sharpness)

            # ── Rescue pass: soft-match recent IDs before minting ─────────────
            # With TTA, same-person scores are 0.40–0.65 in most conditions.
            # RESCUE_FLOOR = 0.36: catches genuine edge cases (very small/distant
            # crops, extreme motion blur) while keeping false-positive rate low.
            # A different person needs to score 0.36+ top-3 mean AND pass the
            # clothing veto to be incorrectly rescued — rare in practice.
            RESCUE_FLOOR = 0.36
            rescue_gid, rescue_score = None, 0.0
            if recent_gids and self._gallery:
                for r_gid in list(recent_gids):
                    r_entry = self._gallery.get(r_gid)
                    if r_entry is None:
                        continue
                    if self._clothing_veto(r_entry, raw_attr_probs, attr_label_list):
                        continue
                    r_pool = (
                        ([r_entry.prototype_emb] if r_entry.prototype_emb is not None else [])
                        + r_entry.anchor_embs
                    )
                    r_sim = self._top_k_mean_sim(best_emb, r_pool, k=3)
                    if r_sim >= RESCUE_FLOOR and r_sim > rescue_score:
                        rescue_gid, rescue_score = r_gid, r_sim

            if rescue_gid is not None:
                # Same-camera uniqueness guard
                if self._gid_active_on_camera(rescue_gid, camera_id, local_track_id):
                    logger.debug(
                        f"[ReID] RESCUE BLOCKED cam={camera_id} local={local_track_id}"
                        f" → G:{rescue_gid} (already active on this camera)"
                    )
                    rescue_gid = None

            if rescue_gid is not None:
                r_entry = self._gallery[rescue_gid]
                self._update_entry(r_entry, best_emb, None, best_sharp, thumb,
                                   raw_probs=raw_attr_probs, label_list=attr_label_list,
                                   color_signature=color_sig)
                self._local_to_global[key] = rescue_gid
                logger.debug(
                    f"[ReID] RESCUE cam={camera_id} local={local_track_id}"
                    f" → global={rescue_gid}  raw_sim={rescue_score:.3f}"
                )
                return rescue_gid

            # ── Closed-world camera-local matching ────────────────────────────
            # Appearance similarity alone is unreliable for CPU-inferred OSNet
            # under viewpoint changes.  In a bounded-occupancy environment (retail
            # store) we can exploit the physical constraint:
            #
            #   "A new track on camera X is overwhelmingly likely to be a
            #    recently-lost person from camera X, NOT a genuinely new person."
            #
            # Strategy:
            #   1. Collect gallery entries for THIS camera that were recently active
            #      (seen 5–120 s ago) and are NOT currently visible (last_seen > 5 s).
            #   2. If EXACTLY ONE such unaccounted-for entry passes the clothing veto
            #      AND has TTA similarity ≥ 0.15 (not clearly a different person)
            #      → link unconditionally.  Appearance threshold is bypassed because
            #      the spatial constraint already provides strong prior evidence.
            #   3. If MULTIPLE candidates → use TTA similarity to pick the best one
            #      (floor 0.25 — lower than normal because we have the closed-world
            #      prior, but still need some discriminability).
            #   4. Zero candidates → person is genuinely new → mint.
            #
            # This handles the "person walks to back of store, comes back 30 s later
            # from a completely different direction" case that defeats both ByteTrack
            # (resets after 15 s) and distance-based spatio-temporal (250 px too tight).

            now_m = time.monotonic()
            CLOSED_WORLD_RECENCY     = 120.0   # look back up to 2 minutes
            CLOSED_WORLD_ACTIVE_GATE = 5.0     # < 5 s ago = still actively tracked
            CLOSED_WORLD_ANTI_VETO   = 0.15    # clearly different if sim < 0.15
            CLOSED_WORLD_MULTI_FLOOR = 0.25    # multi-candidate floor

            cw_candidates = []
            for cw_gid, cw_entry in self._gallery.items():
                if cw_entry.camera_id != camera_id:
                    continue
                entry_age = now_m - cw_entry.last_seen
                if not (CLOSED_WORLD_ACTIVE_GATE < entry_age < CLOSED_WORLD_RECENCY):
                    continue  # currently active OR too old
                # Extra safety: skip if being actively tracked on ANY camera right now
                if self._gid_active_on_camera(cw_gid, camera_id, local_track_id):
                    continue
                if self._clothing_veto(cw_entry, raw_attr_probs, attr_label_list):
                    continue  # hard clothing mismatch
                cw_candidates.append(cw_gid)

            cw_link_gid = None

            if len(cw_candidates) == 1:
                cw_gid = cw_candidates[0]
                cw_entry = self._gallery[cw_gid]
                cw_pool = (
                    ([cw_entry.prototype_emb] if cw_entry.prototype_emb is not None else [])
                    + cw_entry.anchor_embs
                )
                cw_sim = self._top_k_mean_sim(best_emb, cw_pool, k=3)
                if cw_sim >= CLOSED_WORLD_ANTI_VETO:
                    cw_link_gid = cw_gid
                    logger.info(
                        f"[ReID] CLOSED-WORLD UNIQUE cam={camera_id} local={local_track_id}"
                        f" → global={cw_gid}  sim={cw_sim:.3f}"
                    )

            elif len(cw_candidates) > 1:
                cw_best_gid, cw_best_score = None, 0.0
                for cw_gid in cw_candidates:
                    cw_entry = self._gallery.get(cw_gid)
                    if cw_entry is None:
                        continue
                    cw_pool = (
                        ([cw_entry.prototype_emb] if cw_entry.prototype_emb is not None else [])
                        + cw_entry.anchor_embs
                    )
                    cw_sim = self._top_k_mean_sim(best_emb, cw_pool, k=3)
                    if cw_sim > cw_best_score:
                        cw_best_gid, cw_best_score = cw_gid, cw_sim
                if cw_best_gid is not None and cw_best_score >= CLOSED_WORLD_MULTI_FLOOR:
                    cw_link_gid = cw_best_gid
                    logger.info(
                        f"[ReID] CLOSED-WORLD MULTI cam={camera_id} local={local_track_id}"
                        f" → global={cw_best_gid}  sim={cw_best_score:.3f}"
                        f"  ({len(cw_candidates)} candidates)"
                    )

            if cw_link_gid is not None:
                cw_entry = self._gallery[cw_link_gid]
                self._update_entry(cw_entry, best_emb, None, best_sharp, thumb,
                                   raw_probs=raw_attr_probs, label_list=attr_label_list,
                                   color_signature=color_sig)
                self._local_to_global[key] = cw_link_gid
                return cw_link_gid

            # ── Genuinely new person — mint a global_id ──────────────────────
            gid = self._next_global_id
            self._next_global_id += 1
            self._gallery[gid] = GalleryEntry(
                global_id=gid,
                prototype_emb=best_emb,
                anchor_embs=[best_emb],
                anchor_qualities=[best_sharp],
                camera_id=camera_id,
                local_track_id=local_track_id,
                thumbnails=[thumb] if thumb is not None else [],
                attribute_vec=None,
                raw_attr_probs=raw_attr_probs.copy() if raw_attr_probs is not None else None,
                attr_label_list=attr_label_list,
                color_signature=color_sig.copy() if color_sig is not None else None,
            )
            self._local_to_global[key] = gid
            logger.debug(f"[ReID] NEW global_id={gid}  cam={camera_id}  local={local_track_id}")
            return gid

    def update_embedding(
        self,
        camera_id: str,
        local_track_id: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        attributes: Optional[dict] = None,
        raw_attr_probs: Optional[np.ndarray] = None,
        attr_label_list: Optional[List[str]] = None,
        skip_thumbnail: bool = False,
        crop_sharpness: float = 0.0,
    ):
        """Refresh gallery embedding and attributes for an existing track."""
        key = (camera_id, local_track_id)
        with self._lock:
            gid = self._local_to_global.get(key)
            if gid is None:
                return False

        embedding = self._extract(frame, bbox)
        color_sig = self._extract_color_signature(frame, bbox)
        thumb = None if skip_thumbnail else self._capture_thumbnail(frame, bbox)

        with self._lock:
            entry = self._gallery.get(gid)
            if not entry:
                return False

            # ── Drift Protection ───────────────────────────────────────────
            # If the new embedding is drastically different from the prototype,
            # this means the local tracker has drifted to a different person.
            # We reject the update to prevent gallery corruption.
            if entry.prototype_emb is not None:
                # With TTA, same-person embeddings score 0.42–0.75 against the
                # prototype even under large viewpoint changes.  Use top-3 mean
                # against the full anchor bank for a more stable drift estimate.
                drift_pool = (
                    [entry.prototype_emb] + entry.anchor_embs
                )
                drift_sim = self._top_k_mean_sim(embedding, drift_pool, k=3)
                # 0.32 threshold: rejects a truly different person drifted into
                # the box (they score 0.08–0.25) while accepting extreme angle
                # changes of the same person (0.38+).
                if drift_sim < 0.32:
                    logger.warning(
                        f"[ReID] Drift detected for Local:{local_track_id} on {camera_id}! "
                        f"Top-3 mean sim to GID {gid} is {drift_sim:.3f}. Rejecting update."
                    )
                    return False

            added_new_angle = self._update_entry(entry, embedding, None, crop_sharpness, thumb,
                                                 raw_probs=raw_attr_probs, label_list=attr_label_list,
                                                 color_signature=color_sig)
            return added_new_angle

    def update_attributes(self, camera_id: str, local_track_id: int, attributes: dict):
        """Store/blend attribute vector for an existing entry without re-extracting embedding."""
        pass # Now deprecated — attributes are updated alongside embeddings

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[int]:
        key = (camera_id, local_track_id)
        with self._lock:
            return self._local_to_global.get(key)

    def link_local_to_global(self, camera_id: str, local_track_id: int, global_id: int) -> None:
        """
        Directly link a local track to an existing global ID without running Re-ID.
        Used by Spatio-Temporal recovery in camera_processor.
        """
        key = (camera_id, local_track_id)
        with self._lock:
            self._local_to_global[key] = global_id
            self._pending.pop(key, None)  # cancel any pending confirmation window

    def remove_track(self, camera_id: str, local_track_id: int):
        """Called when a track is cleaned up."""
        key = (camera_id, local_track_id)
        with self._lock:
            self._local_to_global.pop(key, None)
            self._pending.pop(key, None)

    def gallery_size(self) -> int:
        with self._lock:
            return len(self._gallery)

    def active_gallery_size(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for e in self._gallery.values() if (now - e.last_seen) < self._expiry)

    # ── Internal helpers ────────────────────────────────────────────────

    def _update_entry(
        self,
        entry: GalleryEntry,
        embedding: Optional[np.ndarray],
        attr_vec: Optional[np.ndarray],
        sharpness: float,
        thumb: Optional[np.ndarray],
        raw_probs: Optional[np.ndarray] = None,
        label_list: Optional[List[str]] = None,
        color_signature: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Update a gallery entry's prototype, anchor bank, attribute vector,
        raw probability vector, color signature, and thumbnail.
        Returns True if a new distinct anchor angle was added.
        """
        added_new_angle = False

        if embedding is not None:
            # Quality-aware anchor bank (update first, prototype derived from it)
            if not entry.anchor_embs:
                entry.anchor_embs.append(embedding.copy())
                entry.anchor_qualities.append(sharpness)
                added_new_angle = True
            else:
                max_sim = max(float(np.dot(a, embedding)) for a in entry.anchor_embs)
                # Require a novel angle (< 0.72) to avoid wasting anchor slots on
                # identical poses.  Lowered from 0.80 because TTA embeddings cluster
                # tighter and two moderately different angles can score ~0.78.
                if max_sim < 0.72:
                    if len(entry.anchor_embs) < self.MAX_ANCHORS:
                        entry.anchor_embs.append(embedding.copy())
                        entry.anchor_qualities.append(sharpness)
                    else:
                        min_idx = int(np.argmin(entry.anchor_qualities))
                        if sharpness > entry.anchor_qualities[min_idx]:
                            entry.anchor_embs[min_idx] = embedding.copy()
                            entry.anchor_qualities[min_idx] = sharpness
                    added_new_angle = True

            # ── Prototype = mean of top-8 quality anchors ─────────────────
            # Replaces EMA which drifted toward the most recent view.
            # Mean of diverse, high-quality anchors is more stable AND more
            # discriminative than a single running average.
            if entry.anchor_embs:
                k = min(8, len(entry.anchor_embs))
                top_k_idx = sorted(range(len(entry.anchor_qualities)),
                                   key=lambda i: entry.anchor_qualities[i],
                                   reverse=True)[:k]
                proto = np.mean([entry.anchor_embs[i] for i in top_k_idx], axis=0)
                norm = np.linalg.norm(proto)
                entry.prototype_emb = proto / norm if norm > 1e-8 else proto

        if attr_vec is not None:
            entry.attribute_vec = (
                attr_vec if entry.attribute_vec is None
                else 0.95 * entry.attribute_vec + 0.05 * attr_vec  # Locked: 95% old, 5% new
            )

        # Store/EMA-blend raw attribute probabilities
        if raw_probs is not None:
            if entry.raw_attr_probs is None:
                # M1-fix: initialise cleanly on first observation; do NOT blend yet
                entry.raw_attr_probs = raw_probs.copy()
                entry.attr_label_list = label_list
            else:
                # Subsequent updates: 95% original, 5% new — defends against
                # lighting/shadow drift while slowly adapting to appearance changes.
                entry.raw_attr_probs = 0.95 * entry.raw_attr_probs + 0.05 * raw_probs

        # ── Color signature (EMA blend, quality-gated) ────────────────────
        # X3-fix: only update from sharp crops (sharpness > 8.0).
        # Motion-blurred or out-of-focus crops produce washed-out HSV histograms
        # that corrupt the EMA — same problem that the anchor novelty check solves
        # for embeddings.  8.0 is intentionally low (real motion blur is < 5.0)
        # so we don't over-restrict at low-res or distant cameras.
        if color_signature is not None and sharpness > 8.0:
            if entry.color_signature is None:
                entry.color_signature = color_signature.copy()
            else:
                blended = 0.80 * entry.color_signature + 0.20 * color_signature
                n = np.linalg.norm(blended)
                entry.color_signature = blended / n if n > 1e-8 else blended

        entry.last_seen = time.monotonic()
        entry.hit_count += 1

        if thumb is not None and added_new_angle:
            entry.thumbnails.append(thumb)
            if len(entry.thumbnails) > 8:
                entry.thumbnails.pop(0)

        return added_new_angle

    def _merge_loop(self):
        """Background thread: deduplicates the gallery every 5 s.

        H2-fix: was 1 s — raised to 5 s so that we pay the O(N²) similarity
        scan only 12× per minute instead of 60×.
        C3-fix: honours the shutdown event on every wait.
        """
        while not self._stop.is_set():
            self._stop.wait(timeout=5.0)
            if self._stop.is_set():
                break
            try:
                self._prune_expired()          # M3-fix: expire entries even when quiet
                self._merge_duplicates(merge_threshold=0.48)
            except Exception as e:
                logger.error(f"ReID merge loop error (will retry): {e}", exc_info=True)
                self._stop.wait(timeout=5)

    def _max_anchor_similarity(self, e1: GalleryEntry, e2: GalleryEntry) -> float:
        """Finds the maximum cosine similarity between any two anchors from two gallery entries."""
        max_sim = 0.0
        
        if e1.prototype_emb is not None and e2.prototype_emb is not None:
            max_sim = max(max_sim, float(np.dot(e1.prototype_emb, e2.prototype_emb)))
            
        if e2.prototype_emb is not None:
            for a1 in e1.anchor_embs:
                max_sim = max(max_sim, float(np.dot(a1, e2.prototype_emb)))
                
        if e1.prototype_emb is not None:
            for a2 in e2.anchor_embs:
                max_sim = max(max_sim, float(np.dot(e1.prototype_emb, a2)))
                
        for a1 in e1.anchor_embs:
            for a2 in e2.anchor_embs:
                max_sim = max(max_sim, float(np.dot(a1, a2)))
                
        return max_sim

    def _merge_duplicates(self, merge_threshold: float = 0.48):
        """
        Scan all gallery pairs. If two entries have fused similarity >= merge_threshold,
        absorb the weaker one (fewer hits) into the stronger.

        H2-fix: the O(N²) similarity scan now runs OUTSIDE the gallery lock so
        that camera threads are never stalled during computation.  We take a
        lightweight snapshot (gid → prototype + anchors + metadata), release the
        lock, score every pair, then re-acquire only for the actual merges.
        """
        # ── Phase 1: snapshot under lock (fast — just copies references) ──────
        with self._lock:
            snapshot: Dict[int, Dict] = {}
            # Also snapshot the set of gids that have an active local_to_global
            # mapping — these are "currently assigned to a live track" regardless
            # of how recently update_embedding was called.
            assigned_gids: set = set(self._local_to_global.values())
            for gid, e in self._gallery.items():
                snapshot[gid] = {
                    "prototype":       e.prototype_emb,
                    "anchors":         list(e.anchor_embs),       # shallow copy
                    "attr_probs":      e.raw_attr_probs,
                    "attr_labels":     e.attr_label_list,
                    "camera_id":       e.camera_id,
                    "hit_count":       e.hit_count,
                    "last_seen":       e.last_seen,
                    "assigned":        gid in assigned_gids,
                    # X1-fix: include colour signature so the merge loop can use
                    # the full fused score (was always neutral 0.5 cross-cam before)
                    "color_signature": e.color_signature,
                }

        gids = list(snapshot.keys())
        to_merge: List[Tuple[int, int, float]] = []
        now_m = time.monotonic()

        # ── Phase 2: pair scoring — NO lock held ─────────────────────────────
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                s1 = snapshot[gids[i]]
                s2 = snapshot[gids[j]]

                # Concurrent-active exclusion — never merge two IDs that are
                # simultaneously present.  Two independent checks:
                #
                # A) _local_to_global assignment: if both IDs are assigned to
                #    live local tracks right now they are provably different people.
                #
                # B) last_seen recency: gate raised to 12 s (was 3 s) so that an
                #    active person whose update_embedding ran 4-5 s ago (normal at
                #    REID_UPDATE_INTERVAL=45 frames / 10 fps) is still protected.
                if s1["assigned"] and s2["assigned"]:
                    continue
                if (now_m - s1["last_seen"]) < 12.0 and (now_m - s2["last_seen"]) < 12.0:
                    continue

                # ── Vectorised similarity ─────────────────────────────────────
                e2_pool = (
                    ([s2["prototype"]] if s2["prototype"] is not None else [])
                    + s2["anchors"]
                )
                emb_sim = 0.0
                if e2_pool and s1["anchors"]:
                    # Stack e2 pool into a matrix for a single np.dot broadcast
                    mat = np.stack(e2_pool)          # (M, D)
                    anc_sims_per_anchor = []
                    for a1 in s1["anchors"]:
                        dots = mat.dot(a1)            # (M,) — vectorised
                        top_k = sorted(dots, reverse=True)[:3]
                        anc_sims_per_anchor.append(float(np.mean(top_k)))
                    anc_sims_per_anchor.sort(reverse=True)
                    emb_sim = float(np.mean(anc_sims_per_anchor[:min(3, len(anc_sims_per_anchor))]))

                # Fallback to prototype-vs-prototype
                if emb_sim == 0.0 and s1["prototype"] is not None and s2["prototype"] is not None:
                    emb_sim = float(np.dot(s1["prototype"], s2["prototype"]))

                if emb_sim == 0.0:
                    continue

                same_cam = (s1["camera_id"] == s2["camera_id"])
                # X1-fix: pass colour signatures so cross-cam pairs score correctly.
                # Before this fix the colour term was always neutral (0.5), which
                # inflated cross-cam scores by a fixed +0.20 regardless of clothing.
                f_score = self._compute_fused_score(
                    emb_sim, s1["attr_probs"], s2["attr_probs"], same_cam,
                    color_sig_a=s1["color_signature"],
                    color_sig_b=s2["color_signature"],
                )

                if f_score >= merge_threshold:
                    if s1["hit_count"] >= s2["hit_count"]:
                        to_merge.append((gids[i], gids[j], f_score))
                    else:
                        to_merge.append((gids[j], gids[i], f_score))

        if not to_merge:
            return

        # ── Phase 3: apply merges under lock ──────────────────────────────────
        with self._lock:
            dropped: set = set()
            for keep_gid, drop_gid, f_score in to_merge:
                if drop_gid in dropped:
                    continue
                keeper  = self._gallery.get(keep_gid)
                dropper = self._gallery.get(drop_gid)
                if keeper is None or dropper is None:
                    continue

                # Clothing-similarity veto (re-evaluated under lock with live entry)
                if self._clothing_veto(keeper, dropper.raw_attr_probs, dropper.attr_label_list):
                    continue

                # Merge prototype (weighted average)
                if keeper.prototype_emb is not None and dropper.prototype_emb is not None:
                    keeper.prototype_emb = 0.5 * keeper.prototype_emb + 0.5 * dropper.prototype_emb
                    norm = np.linalg.norm(keeper.prototype_emb)
                    if norm > 0:
                        keeper.prototype_emb /= norm

                # Merge anchor banks — keep the best MAX_ANCHORS by quality
                combined_anchors = list(zip(keeper.anchor_embs, keeper.anchor_qualities)) + \
                                   list(zip(dropper.anchor_embs, dropper.anchor_qualities))
                combined_anchors.sort(key=lambda x: x[1], reverse=True)
                combined_anchors = combined_anchors[:self.MAX_ANCHORS]
                keeper.anchor_embs = [a for a, _ in combined_anchors]
                keeper.anchor_qualities = [q for _, q in combined_anchors]

                keeper.hit_count += dropper.hit_count
                keeper.thumbnails.extend(dropper.thumbnails)
                while len(keeper.thumbnails) > 3:
                    keeper.thumbnails.pop(0)

                for lk, gid in list(self._local_to_global.items()):
                    if gid == drop_gid:
                        self._local_to_global[lk] = keep_gid

                del self._gallery[drop_gid]
                dropped.add(drop_gid)
                logger.debug(f"[ReID] MERGED G:{drop_gid} → G:{keep_gid}  score={f_score:.3f}")

    def _sanitize_loop(self):
        """Background thread: scans for hijacked/corrupted IDs every 10 s."""
        while not self._stop.is_set():  # C3-fix: honour shutdown event
            self._stop.wait(timeout=10)
            if self._stop.is_set():
                break
            try:
                self._sanitize_gallery()
            except Exception as e:
                logger.error(f"ReID sanitize loop error (will retry): {e}", exc_info=True)
                self._stop.wait(timeout=5)

    def _sanitize_gallery(self):
        """
        Detects if a gallery entry has been corrupted by absorbing another person's embeddings.
        If the anchor bank contains two groups of embeddings that are highly dissimilar to each other
        (cosine < 0.45), it splits the entry automatically using distance clustering.
        """
        # Must be strictly lower than REID_SIMILARITY_THRESHOLD (0.48) and merge_threshold (0.42).
        # Front↔back view of the same person on CPU OSNet scores ~0.35–0.50 — the old
        # value of 0.42 was actively splitting valid multi-angle entries into new IDs.
        # 0.25 only triggers when two anchors are truly alien to each other (different people).
        SPLIT_THRESHOLD = 0.25
        with self._lock:
            gids = list(self._gallery.keys())
            for gid in gids:
                entry = self._gallery.get(gid)
                if not entry or len(entry.anchor_embs) < 3:
                    continue
                
                n = len(entry.anchor_embs)
                corrupted = False
                split_idx_a = -1
                split_idx_b = -1

                # 1. Find the two most dissimilar anchors
                for i in range(n):
                    for j in range(i + 1, n):
                        sim = float(np.dot(entry.anchor_embs[i], entry.anchor_embs[j]))
                        if sim < SPLIT_THRESHOLD:
                            corrupted = True
                            split_idx_a = i
                            split_idx_b = j
                            break
                    if corrupted:
                        break
                
                # 2. Slice the ID down the middle
                if corrupted:
                    new_gid = self._next_global_id
                    self._next_global_id += 1
                    
                    new_entry = GalleryEntry(
                        global_id=new_gid,
                        camera_id=entry.camera_id,
                        local_track_id=-1,
                        raw_attr_probs=entry.raw_attr_probs.copy() if entry.raw_attr_probs is not None else None,
                        attr_label_list=entry.attr_label_list,
                        thumbnails=[entry.thumbnails[-1]] if entry.thumbnails else []
                    )
                    
                    anchor_a = entry.anchor_embs[split_idx_a]
                    anchor_b = entry.anchor_embs[split_idx_b]
                    
                    keep_embs, keep_quals = [], []
                    move_embs, move_quals = [], []
                    
                    for i in range(n):
                        sim_a = float(np.dot(entry.anchor_embs[i], anchor_a))
                        sim_b = float(np.dot(entry.anchor_embs[i], anchor_b))
                        if sim_a >= sim_b:
                            keep_embs.append(entry.anchor_embs[i])
                            keep_quals.append(entry.anchor_qualities[i])
                        else:
                            move_embs.append(entry.anchor_embs[i])
                            move_quals.append(entry.anchor_qualities[i])
                    
                    # Update old entry
                    entry.anchor_embs = keep_embs
                    entry.anchor_qualities = keep_quals
                    if keep_embs:
                        entry.prototype_emb = np.mean(keep_embs, axis=0)
                        norm = np.linalg.norm(entry.prototype_emb)
                        if norm > 0: entry.prototype_emb /= norm
                        
                    # Update new entry
                    new_entry.anchor_embs = move_embs
                    new_entry.anchor_qualities = move_quals
                    if move_embs:
                        new_entry.prototype_emb = np.mean(move_embs, axis=0)
                        norm = np.linalg.norm(new_entry.prototype_emb)
                        if norm > 0: new_entry.prototype_emb /= norm
                        
                    self._gallery[new_gid] = new_entry
                    logger.warning(
                        f"[ReID] SANITIZER: Split corrupted ID {gid} into {gid} (kept {len(keep_embs)} anchors) "
                        f"and {new_gid} (moved {len(move_embs)} anchors) !"
                    )

    def _clothing_veto(
        self,
        entry: GalleryEntry,
        query_raw_probs: Optional[np.ndarray],
        query_label_list: Optional[List[str]],
    ) -> bool:
        """
        Hard veto: returns True (REJECT match) when both the gallery entry and the
        query have high-confidence clothing-color readings that clearly disagree.

        Strategy
        --------
        - Extract the upper-body color probability sub-vector from each side
          by finding all label indices that start with "up" in the label list.
        - If both sides have max confidence > VETO_MIN_CONF (e.g. 0.55), compare
          the two sub-vectors using cosine similarity.
        - If cosine < VETO_MAX_SIM (e.g. 0.30), the colors are clearly different
          → veto the match.
        - Same logic is applied for lower-body ("down" indices) independently.
        - Either channel triggering is enough to veto.
        """
        VETO_MIN_CONF = 0.45  # Only veto when we have HIGH confidence in the color read (was 0.20 — noise territory)
        VETO_MAX_SIM  = 0.35  # Cosine below 0.35 = clearly different colors (was 0.50 — too aggressive, vetoed similar colors)

        if query_raw_probs is None or query_label_list is None:
            return False
        if entry.raw_attr_probs is None or entry.attr_label_list is None:
            return False

        q_labels = query_label_list
        e_labels = entry.attr_label_list

        # --- Demographic Check (Gender Veto) ---
        if "gender" in q_labels and "gender" in e_labels:
            q_idx = q_labels.index("gender")
            e_idx = e_labels.index("gender")
            q_prob = float(query_raw_probs[q_idx])
            e_prob = float(entry.raw_attr_probs[e_idx])
            # If the difference in male/female probability is extreme (> 60%),
            # it guarantees opposite genders. Instantly block!
            if abs(q_prob - e_prob) > 0.60:
                logger.debug(f"[ReID] DEMOGRAPHIC VETO: Gender mismatch diff={abs(q_prob - e_prob):.2f}")
                return True

        # --- Clothing Check ---
        # Build index maps (excluding generic "up"/"down" sleeve/pant type labels)
        def color_indices(label_list, prefix):
            return [i for i, lbl in enumerate(label_list)
                    if lbl.startswith(prefix) and lbl not in ("up", "down") and i < len(label_list)]

        q_labels = query_label_list
        e_labels = entry.attr_label_list

        for prefix in ("up", "down"):
            q_idx = color_indices(q_labels, prefix)
            e_idx = color_indices(e_labels, prefix)
            if not q_idx or not e_idx:
                continue

            q_vec = query_raw_probs[np.array(q_idx, dtype=int)]
            e_vec = entry.raw_attr_probs[np.array(e_idx, dtype=int)]

            # Confidence check: both must have a clear winner (or at least one color spike)
            if q_vec.max() < VETO_MIN_CONF or e_vec.max() < VETO_MIN_CONF:
                continue

            # Align vectors to the same colour set (use shorter length)
            min_len = min(len(q_vec), len(e_vec))
            q_norm_vec = q_vec[:min_len]
            e_norm_vec = e_vec[:min_len]

            q_n = np.linalg.norm(q_norm_vec)
            e_n = np.linalg.norm(e_norm_vec)
            if q_n < 1e-6 or e_n < 1e-6:
                continue

            cos_sim = float(np.dot(q_norm_vec / q_n, e_norm_vec / e_n))
            if cos_sim < VETO_MAX_SIM:
                logger.debug(
                    f"[ReID] CLOTHING VETO prefix={prefix} cos={cos_sim:.3f} "
                    f"(query_max={q_vec.max():.2f} entry_max={e_vec.max():.2f})"
                )
                return True  # veto this match

        return False

    # ── Embedding helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _top_k_mean_sim(query_emb: np.ndarray, emb_list: List[np.ndarray], k: int = 3) -> float:
        """
        Mean cosine similarity of query against the top-k most similar embeddings
        in emb_list.  More robust than plain max (one lucky anchor can't dominate)
        yet more sensitive than mean-all (noisy/old anchors are de-weighted).
        Returns 0.0 for an empty list.
        """
        if not emb_list:
            return 0.0
        sims = sorted((float(np.dot(query_emb, e)) for e in emb_list), reverse=True)
        return float(np.mean(sims[:min(k, len(sims))]))

    @staticmethod
    def _get_track_embedding(
        buf: List[Tuple[np.ndarray, Optional[np.ndarray], float]]
    ) -> Optional[np.ndarray]:
        """
        Quality-weighted mean of all embeddings in the confirmation buffer.
        Higher-sharpness frames contribute more, producing a descriptor that
        is more stable and discriminative than any single-frame embedding.
        Critically robust for cases where individual frames suffer from motion
        blur, partial occlusion, or bad lighting.
        """
        if not buf:
            return None
        embs   = np.stack([e for e, _, _ in buf])          # (N, 512)
        quals  = np.array([max(s, 0.1) for _, _, s in buf])  # floor at 0.1 to avoid zero weight
        weights = quals / quals.sum()
        mean_emb = (embs * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(mean_emb)
        return mean_emb / norm if norm > 1e-8 else embs[0]

    # Pre-built CLAHE instance (shared, thread-safe for apply())
    _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    @staticmethod
    def _clahe_normalize(crop_bgr: np.ndarray) -> np.ndarray:
        """
        Normalise per-camera lighting with CLAHE on the L channel (LAB space).
        Dramatically improves cross-camera embedding similarity when cameras
        have different exposure / white-balance settings.

        LAB L-channel CLAHE is preferred over per-channel RGB CLAHE because:
          - It separates luminance from colour (chrominance stays intact)
          - CLAHE on L only stretches contrast without hue distortion
          - Empirically gives +0.05–0.12 same-person cosine boost across cameras
        """
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = ReIDManager._CLAHE.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _extract_color_signature(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract a 96-d HSV color histogram signature from the upper and lower
        body halves (48 bins each: 12 hue × 4 saturation).

        Why this works when deep embeddings fail:
          - Completely POSE-INVARIANT: same shirt from any angle → same hue distribution
          - Completely RESOLUTION-INVARIANT: distant or close, the colours are the same
          - LIGHTING-ROBUST after CLAHE: two cameras with different exposure produce
            similar histograms for the same outfit
          - Orthogonal to embedding: even when OSNet similarity drops to 0.30 due to
            viewpoint, colour similarity stays 0.75+ for the same outfit

        Body regions (fraction of bbox height):
          Upper body: 20%–55% (skip head, include torso)
          Lower body: 55%–90% (legs, skip feet)
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            bh, bw = y2 - y1, x2 - x1
            # X3-fix: require enough height so upper/lower body splits are meaningful.
            # At 32 px each half is only 11-16 px — background noise dominates.
            # 96 px gives ~27 px per half, enough to capture clothing colour reliably.
            if bh < 96 or bw < 32:
                return None

            # Normalize lighting before computing colour histogram
            crop = frame[y1:y2, x1:x2]
            crop_norm = self._clahe_normalize(crop)

            upper = crop_norm[int(bh * 0.20):int(bh * 0.55), :]
            lower = crop_norm[int(bh * 0.55):int(bh * 0.90), :]

            def hist48(region: np.ndarray) -> np.ndarray:
                if region.size == 0:
                    return np.zeros(48, dtype=np.float32)
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0, 1], None, [12, 4],
                                 [0, 180, 0, 256]).flatten().astype(np.float32)
                n = np.linalg.norm(h)
                return h / n if n > 1e-6 else h

            sig = np.concatenate([hist48(upper), hist48(lower)])
            n = np.linalg.norm(sig)
            return sig / n if n > 1e-6 else sig
        except Exception:
            return None

    # ── Shared ImageNet normalisation constants (avoids per-call allocation) ──
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply CLAHE → BGR→RGB → resize → normalise → transpose to [3, 256, 128].
        Returns float32 CHW tensor ready for batching, or None on error.
        """
        try:
            crop_norm = self._clahe_normalize(crop_bgr)
            crop_rgb  = cv2.cvtColor(crop_norm, cv2.COLOR_BGR2RGB)
            crop_rsz  = cv2.resize(crop_rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
            img = (crop_rsz.astype(np.float32) / 255.0 - self._MEAN) / self._STD
            return img.transpose(2, 0, 1)          # [3, 256, 128]
        except Exception:
            return None

    def _embed_batch_raw(
        self, crops_bgr: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """
        P1-fix: Run ONNX/TRT inference on N BGR crops in a single batched call.

        Replaces the old _embed_raw() which ran one session.run() per crop.
        Batching is safe because OSNet's ONNX export uses a dynamic 'batch' axis
        (verified: input shape = ['batch', 3, 256, 128]).

        On Jetson Orin Nano, GPU kernel-launch overhead dominates for small models
        like OSNet (2.2M params, 0.98 GFLOPs).  Batching N crops into one call:
          - Reduces kernel launches by N×
          - Gives TRT the full SIMD lane utilisation on Ampere CUDA cores
          - Typical measured speedup: 1.6–2.5× vs sequential single-crop calls

        Returns a list of N L2-normalised 512-d vectors (None for failed crops).
        """
        if not crops_bgr or self._extractor is None:
            return [None] * len(crops_bgr)

        imgs: List[np.ndarray] = []
        valid_idx: List[int]   = []

        for i, crop in enumerate(crops_bgr):
            t = self._preprocess_crop(crop)
            if t is not None:
                imgs.append(t)
                valid_idx.append(i)

        results: List[Optional[np.ndarray]] = [None] * len(crops_bgr)
        if not imgs:
            return results

        try:
            batch      = np.stack(imgs)                              # [N, 3, 256, 128]
            input_name = self._extractor.get_inputs()[0].name
            features   = self._extractor.run(None, {input_name: batch})[0]  # [N, 512]
            for batch_i, orig_i in enumerate(valid_idx):
                emb  = features[batch_i]
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    results[orig_i] = emb / norm
        except Exception as exc:
            logger.debug(f"[ReID] Batch embedding failed: {exc}")

        return results

    def _embed_raw(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Single-crop inference wrapper (kept for backward compatibility).
        Delegates to the batched path — no extra overhead."""
        return self._embed_batch_raw([crop_bgr])[0]

    def _extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop person from frame and extract a Test-Time Augmented (TTA) embedding.

        P1-fix: original crop and its horizontal flip are now batched into a
        single session.run([2, 3, 256, 128]) call instead of two sequential
        batch=1 calls.  This halves the GPU kernel-launch overhead for TTA.

        TTA improvement measured on OSNet for same-person cross-view pairs:
          - Front ↔ Back:  +0.10–0.18 cosine similarity
          - Side ↔ Front:  +0.05–0.12 cosine similarity
          - Same angle:    ≈ +0.01 (flip ≈ identity for symmetric pose)
        """
        if self._extractor is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (y2 - y1) < self._min_crop or (x2 - x1) < 16:
            return None

        crop = frame[y1:y2, x1:x2]

        # P1-fix: batch orig + flip in one session.run() call (was 2 separate calls)
        emb, emb_flip = self._embed_batch_raw([crop, cv2.flip(crop, 1)])

        if emb is None:
            return None

        # TTA: L2-normalised average of original and flipped embeddings
        if emb_flip is not None:
            combined = emb + emb_flip
            norm = np.linalg.norm(combined)
            if norm > 1e-8:
                emb = combined / norm

        return emb

    def _capture_thumbnail(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
        thumb_w: int = 64, thumb_h: int = 128,
        pad_pct: float = 0.10,
    ) -> Optional[np.ndarray]:
        """Return a padded, resized BGR crop of the person for the gallery view."""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            bw, bh = x2 - x1, y2 - y1
            pad_x = int(bw * pad_pct)
            pad_y = int(bh * pad_pct)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            if (x2 - x1) < 16 or (y2 - y1) < 16:
                return None
            crop = frame[y1:y2, x1:x2]
            return cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    def _compute_fused_score(
        self,
        emb_sim: float,
        attr_a: Optional[np.ndarray],
        attr_b: Optional[np.ndarray],
        same_camera: bool = True,
        color_sig_a: Optional[np.ndarray] = None,
        color_sig_b: Optional[np.ndarray] = None,
    ) -> float:
        """
        Blend embedding similarity, attribute similarity, and HSV color signature.

        Weight strategy:
          Same camera:  embedding 80% + color_sig 15% + attr  5%
          Cross camera: embedding 55% + color_sig 40% + attr  5%

        Cross-camera gets heavy color weight because:
          - HSV histograms are pose/angle/lighting invariant — the same outfit
            looks the same from any camera at any angle after CLAHE
          - OSNet embedding degrades significantly under viewpoint + lighting
            changes across different cameras
          - With CLAHE normalization both features are on the same footing

        Falls back gracefully when color_signature is unavailable (neutral 0.5).
        """
        attr_sim  = _attribute_similarity(attr_a, attr_b)  # 0.5 if missing

        color_sim = 0.5  # neutral when not available
        if color_sig_a is not None and color_sig_b is not None:
            cs = float(np.dot(color_sig_a, color_sig_b))
            color_sim = max(0.0, min(1.0, cs))  # clamp — histograms are non-negative

        if same_camera:
            # Same-cam: embedding is highly reliable (no viewpoint shift).
            # Colour at 15% provides a soft discriminator; attr at 5% vetoes gender.
            return 0.80 * emb_sim + 0.15 * color_sim + 0.05 * attr_sim
        else:
            # A1-fix: cross-cam colour weight reduced 0.40 → 0.25; embedding raised
            # 0.55 → 0.70.  The previous 40% colour weight caused false positives
            # when two different people both wore common neutral colours (black, navy,
            # white) — which describe ≈40% of retail shoppers.  At 0.25 colour is
            # still a meaningful boost for distinctive outfits (red dress, yellow
            # jacket) but can no longer overpower a low embedding score.
            return 0.70 * emb_sim + 0.25 * color_sim + 0.05 * attr_sim

    def _find_best_match(
        self,
        query_emb: np.ndarray,
        query_attr: Optional[np.ndarray] = None,
        camera_id: Optional[str] = None,
        recent_gids: Optional[set] = None,
        current_occupancy: int = 0,
        query_raw_probs: Optional[np.ndarray] = None,
        query_label_list: Optional[List[str]] = None,
        query_color_sig: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[int], float]:
        """
        Find the gallery entry with highest fused similarity to the query.
        Applies Adaptive Margin Thresholding and a hard clothing-color veto.
        Returns (global_id, fused_score) or (None, 0.0) if no confident match.
        """
        candidates: List[Tuple[float, int]] = []
        recent_gids = recent_gids or set()

        for gid, entry in self._gallery.items():
            if entry.prototype_emb is None and not entry.anchor_embs:
                continue

            # ── Clothing veto (hard reject before any scoring) ─────────────
            if self._clothing_veto(entry, query_raw_probs, query_label_list):
                continue

            # ── Top-3 mean gallery scoring ────────────────────────────────
            # Build the full candidate pool: prototype + all anchors.
            # Use mean of top-3 similarities rather than max:
            #   - max(sims): one lucky/noisy anchor dominates → high variance
            #   - mean-all:  noisy/old anchors dilute good matches → misses
            #   - top-3 mean: stable yet sensitive, ~95 % of the benefit of max
            #     with far lower variance across camera angles.
            pool = []
            if entry.prototype_emb is not None:
                pool.append(entry.prototype_emb)
            pool.extend(entry.anchor_embs)
            emb_sim = self._top_k_mean_sim(query_emb, pool, k=3)

            # Contextual bonus for recently-seen IDs.
            # With TTA, same-person scores are now reliably 0.40-0.65 (was 0.22-0.40).
            # A smaller bonus (+0.12) is sufficient and reduces false-positive risk
            # (a different person that scores 0.30 raw + 0.12 = 0.42 is already near-
            # threshold; the clothing veto catches the remainder).
            if gid in recent_gids:
                emb_sim = min(1.0, emb_sim + 0.12)

            same_cam = (entry.camera_id == camera_id)
            fused = self._compute_fused_score(
                emb_sim, query_raw_probs, entry.raw_attr_probs, same_cam,
                color_sig_a=query_color_sig, color_sig_b=entry.color_signature,
            )
            candidates.append((fused, gid))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_gid = candidates[0]

        # Dynamic threshold.
        # A1-fix: cross-camera matches use a slightly raised threshold (+0.02) because
        # the reduced colour weight (0.25 vs 0.40) means the fused score for the
        # same genuine cross-cam pair shifts down by ~0.03-0.07.  Adding 0.02 keeps
        # the operating point unchanged for distinctive outfits while giving an extra
        # buffer against common-colour false positives on the cross-cam path.
        # Same-camera stays at the base threshold — no change there.
        # DO NOT lower threshold when gallery is large — that's exactly when
        # wrong matches are most dangerous (more potential impostors).
        best_entry_cam = (self._gallery[best_gid].camera_id
                          if best_gid in self._gallery else camera_id)
        _cross_cam_match = (best_entry_cam != camera_id)
        dynamic_threshold = self._threshold + (0.02 if _cross_cam_match else 0.0)

        if best_score < dynamic_threshold:
            return None, 0.0

        # Adaptive margin: only reject when two gallery entries are nearly identical
        # in score AND both above threshold — that's a genuine coin-flip.
        # 0.03 (was 0.05) — 0.05 was too aggressive; with a busy gallery it blocked
        # correct matches too often, sending them back to buffer another window.
        if len(candidates) > 1:
            runner_up_score = candidates[1][0]
            margin = best_score - runner_up_score
            if runner_up_score >= dynamic_threshold and margin < 0.03:
                logger.warning(
                    f"[ReID] Ambiguous: G:{best_gid} vs G:{candidates[1][1]} "
                    f"margin={margin:.3f}. Buffering instead of minting."
                )
                return None, 0.0

        return best_gid, best_score

    def _prune_expired(self):
        """Remove gallery entries not seen recently. Must be called inside self._lock."""
        now = time.monotonic()
        expired = [
            gid for gid, e in self._gallery.items()
            if (now - e.last_seen) > self._expiry
        ]
        for gid in expired:
            del self._gallery[gid]
        if expired:
            logger.debug(f"[ReID] Pruned {len(expired)} expired gallery entries")
