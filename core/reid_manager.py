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

    # Weights for same-camera vs cross-camera fused similarity.
    # Clothing colour is more stable across viewpoints, so we trust it more cross-cam.
    EMBEDDING_W_SAME  = 0.98
    ATTRIBUTE_W_SAME  = 0.02
    EMBEDDING_W_CROSS = 0.85
    ATTRIBUTE_W_CROSS = 0.15

    # Max anchor embeddings per gallery entry (stores diverse angles)
    MAX_ANCHORS = 48

    # Confirmation window: number of frames to buffer before minting a new ID.
    # 5 frames is enough for the gallery match to fire on a clean crop while still
    # filtering one-frame ghost detections.  20 was so long that re-entries would
    # exhaust the window with bad crops and mint spurious new IDs.
    CONFIRMATION_WINDOW = 5

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

        thumb = None if skip_thumbnail else self._capture_thumbnail(frame, bbox)

        with self._lock:
            # Double-check (race condition guard)
            if key in self._local_to_global:
                return self._local_to_global[key]

            self._prune_expired()

            # ── Accumulate into confirmation window ────────────────────────
            buf = self._pending.setdefault(key, [])
            buf.append((embedding, raw_attr_probs.copy() if raw_attr_probs is not None else None, crop_sharpness))

            # Try matching with ALL buffered embeddings (best score wins)
            best_gid, best_score = None, 0.0
            if self._gallery:
                for buf_emb, buf_raw_probs, _ in buf:
                    gid, score = self._find_best_match(
                        buf_emb, query_raw_probs=buf_raw_probs,
                        camera_id=camera_id, recent_gids=recent_gids,
                        current_occupancy=current_occupancy,
                        query_label_list=attr_label_list,
                    )
                    if gid is not None and score > best_score:
                        best_gid, best_score = gid, score

            if best_gid is not None:
                # Match found — link and enrich gallery with best buffered embedding
                entry = self._gallery[best_gid]
                best_emb, _, best_sharp = max(buf, key=lambda x: x[2]) if buf else (embedding, None, crop_sharpness)
                self._update_entry(entry, best_emb, None, best_sharp, thumb,
                                   raw_probs=raw_attr_probs, label_list=attr_label_list)
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

            # Window exhausted with no match → mint new global_id using best-quality buffered embedding
            self._pending.pop(key, None)
            # M2-fix: attr_vec is not in scope here; use None as fallback
            best_emb, best_attr, _ = max(buf, key=lambda x: x[2]) if buf else (embedding, None, crop_sharpness)

            gid = self._next_global_id
            self._next_global_id += 1
            self._gallery[gid] = GalleryEntry(
                global_id=gid,
                prototype_emb=best_emb,
                anchor_embs=[best_emb],
                anchor_qualities=[crop_sharpness],
                camera_id=camera_id,
                local_track_id=local_track_id,
                thumbnails=[thumb] if thumb is not None else [],
                attribute_vec=None,
                raw_attr_probs=raw_attr_probs.copy() if raw_attr_probs is not None else None,
                attr_label_list=attr_label_list,
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
                sim = float(np.dot(embedding, entry.prototype_emb))
                # 0.40 threshold: only reject truly alien embeddings (different person
                # has drifted into the bounding box).  0.65 was too strict — it blocked
                # legitimate re-appearances from a different angle, preventing the anchor
                # bank from diversifying and causing re-entries to miss gallery matches.
                if sim < 0.40:
                    logger.warning(
                        f"[ReID] Drift detected for Local:{local_track_id} on {camera_id}! "
                        f"Cosine sim to GID {gid} is {sim:.3f}. Rejecting update."
                    )
                    return False

            added_new_angle = self._update_entry(entry, embedding, None, crop_sharpness, thumb,
                                                 raw_probs=raw_attr_probs, label_list=attr_label_list)
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
    ) -> bool:
        """
        Update a gallery entry's prototype, anchor bank, attribute vector,
        raw probability vector, and thumbnail.
        Returns True if a new distinct anchor angle was added.
        """
        added_new_angle = False

        if embedding is not None:
            # Update EMA prototype
            if entry.prototype_emb is None:
                entry.prototype_emb = embedding.copy()
                added_new_angle = True
            else:
                # 85% old, 15% new — adapts faster to better crops so a poor
                # first-frame prototype doesn't permanently poison matching.
                entry.prototype_emb = 0.85 * entry.prototype_emb + 0.15 * embedding
                norm = np.linalg.norm(entry.prototype_emb)
                if norm > 0:
                    entry.prototype_emb /= norm

            # Quality-aware anchor bank
            if not entry.anchor_embs:
                entry.anchor_embs.append(embedding.copy())
                entry.anchor_qualities.append(sharpness)
                added_new_angle = True
            else:
                max_sim = max(float(np.dot(a, embedding)) for a in entry.anchor_embs)
                # Require a novel angle (< 0.80 similarity) to not waste anchor slots on identical poses
                if max_sim < 0.80:
                    if len(entry.anchor_embs) < self.MAX_ANCHORS:
                        entry.anchor_embs.append(embedding.copy())
                        entry.anchor_qualities.append(sharpness)
                    else:
                        min_idx = int(np.argmin(entry.anchor_qualities))
                        if sharpness > entry.anchor_qualities[min_idx]:
                            entry.anchor_embs[min_idx] = embedding.copy()
                            entry.anchor_qualities[min_idx] = sharpness
                    added_new_angle = True

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

        entry.last_seen = time.monotonic()
        entry.hit_count += 1

        if thumb is not None and added_new_angle:
            entry.thumbnails.append(thumb)
            if len(entry.thumbnails) > 8:
                entry.thumbnails.pop(0)

        return added_new_angle

    def _merge_loop(self):
        """Background thread: deduplicates the gallery every 2 s."""
        while not self._stop.is_set():  # C3-fix: honour shutdown event
            self._stop.wait(timeout=2)
            if self._stop.is_set():
                break
            try:
                self._prune_expired()          # M3-fix: expire entries even when quiet
                self._merge_duplicates(merge_threshold=0.42)
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

        Lower threshold (0.48 vs old 0.55) ensures cross-camera same-person entries
        reliably merge in the background thread.
        """
        with self._lock:
            gids = list(self._gallery.keys())
            to_merge: List[Tuple[int, int, float]] = []

            for i in range(len(gids)):
                for j in range(i + 1, len(gids)):
                    e1 = self._gallery.get(gids[i])
                    e2 = self._gallery.get(gids[j])

                    if not e1 or not e2:
                        continue

                    # Max-anchor comparison: best cosine across all anchor pairs
                    emb_sim = self._max_anchor_similarity(e1, e2)

                    # Fallback to prototype if no anchors
                    if emb_sim == 0.0 and e1.prototype_emb is not None and e2.prototype_emb is not None:
                        emb_sim = float(np.dot(e1.prototype_emb, e2.prototype_emb))

                    if emb_sim == 0.0:
                        continue

                    # Cross-camera merge uses higher attribute weight (15%)
                    same_cam = (e1.camera_id == e2.camera_id)
                    f_score = self._compute_fused_score(emb_sim, e1.raw_attr_probs, e2.raw_attr_probs, same_cam)

                    if f_score >= merge_threshold:
                        # Safety check: hard clothing veto before merging
                        if self._clothing_veto(e1, e2.raw_attr_probs, e2.attr_label_list):
                            continue

                        if e1.hit_count >= e2.hit_count:
                            to_merge.append((gids[i], gids[j], f_score))
                        else:
                            to_merge.append((gids[j], gids[i], f_score))

            dropped: set = set()
            for keep_gid, drop_gid, f_score in to_merge:
                if drop_gid in dropped or keep_gid not in self._gallery or drop_gid not in self._gallery:
                    continue
                keeper  = self._gallery[keep_gid]
                dropper = self._gallery[drop_gid]

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

    def _extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop person from frame and run OSNet ONNX feature extraction."""
        if self._extractor is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Lowered thinness threshold to 16px to accept partial/distant bodies
        if (y2 - y1) < self._min_crop or (x2 - x1) < 16:
            return None

        try:
            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Preprocessing: resize to (256, 128), normalize with ImageNet mean/std
            crop_resized = cv2.resize(crop_rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
            img = crop_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            # HWC → CHW → NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0).astype(np.float32)

            input_name = self._extractor.get_inputs()[0].name
            features = self._extractor.run(None, {input_name: img})

            emb = features[0][0]  # (512,)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        except Exception as exc:
            logger.debug(f"[ReID] Embedding extraction failed: {exc}")
            return None

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
    ) -> float:
        """
        Blend embedding similarity with attribute similarity.
        Cross-camera queries use a higher attribute weight (7% vs 2%) because
        clothing colour is more stable across viewpoints than a single embedding angle.
        """
        if attr_a is None or attr_b is None:
            return emb_sim
        attr_sim = _attribute_similarity(attr_a, attr_b)
        if same_camera:
            return self.EMBEDDING_W_SAME * emb_sim + self.ATTRIBUTE_W_SAME * attr_sim
        else:
            return self.EMBEDDING_W_CROSS * emb_sim + self.ATTRIBUTE_W_CROSS * attr_sim

    def _find_best_match(
        self,
        query_emb: np.ndarray,
        query_attr: Optional[np.ndarray] = None,
        camera_id: Optional[str] = None,
        recent_gids: Optional[set] = None,
        current_occupancy: int = 0,
        query_raw_probs: Optional[np.ndarray] = None,
        query_label_list: Optional[List[str]] = None,
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

            # Best similarity across prototype and all anchors
            sims = []
            if entry.prototype_emb is not None:
                sims.append(float(np.dot(query_emb, entry.prototype_emb)))
            for a in entry.anchor_embs:
                sims.append(float(np.dot(query_emb, a)))
            emb_sim = max(sims) if sims else 0.0

            # Contextual bonus for IDs recently seen on this camera
            if gid in recent_gids:
                emb_sim = min(1.0, emb_sim + 0.02)

            same_cam = (entry.camera_id == camera_id)
            fused = self._compute_fused_score(emb_sim, query_raw_probs, entry.raw_attr_probs, same_cam)
            candidates.append((fused, gid))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_gid = candidates[0]

        # Dynamic threshold: stays flat, only provides a small bonus for
        # returning known IDs seen on the same camera.
        # DO NOT lower threshold when gallery is large — that's exactly when
        # wrong matches are most dangerous (more potential impostors).
        dynamic_threshold = self._threshold
        if current_occupancy > 0:
            # Mild bonus only for IDs recently-seen on this same camera
            pass  # bonus is handled by the recent_gids +0.02 in scoring above

        if best_score < dynamic_threshold:
            return None, 0.0

        # Adaptive margin: only reject when two gallery entries are nearly identical
        # in score AND both above threshold — that's a genuine coin-flip.
        # 0.03 (was 0.05) — 0.05 was too aggressive; with a busy gallery it blocked
        # correct matches too often, sending them back to buffer another window.
        if len(candidates) > 1:
            runner_up_score = candidates[1][0]
            margin = best_score - runner_up_score
            if runner_up_score >= self._threshold and margin < 0.03:
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
