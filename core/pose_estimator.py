"""
pose_estimator.py — Lightweight human pose estimation on person crops.

Uses YOLOv8n-pose (ultralytics, already in requirements) for 17-keypoint
COCO pose on a single BGR crop.  Model is ~6 MB and downloads once on
first use from the Ultralytics hub.

Public API:
    from core.pose_estimator import estimate_pose_both

    jpeg, data = estimate_pose_both(crop_bgr)  # single inference pass
"""

import cv2
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)

_LOAD_FAILED = object()          # sentinel — distinct from None
_model       = None              # None = not tried yet; _LOAD_FAILED = tried and failed
_model_lock  = threading.Lock()

# COCO 17-keypoint names
_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges for manual draw fallback
_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),           # face
    (5,6),(5,7),(7,9),(6,8),(8,10),    # arms
    (5,11),(6,12),(11,12),             # torso
    (11,13),(13,15),(12,14),(14,16),   # legs
]


def _get_model():
    """Lazy-load YOLOv8n-pose once; thread-safe.
    Returns the model on success, None on failure.
    After the first failed attempt the sentinel prevents repeated retries/log spam.
    """
    global _model
    if _model is not None:
        return None if _model is _LOAD_FAILED else _model
    with _model_lock:
        if _model is not None:          # double-checked
            return None if _model is _LOAD_FAILED else _model
        try:
            from ultralytics import YOLO
            _model = YOLO("yolov8n-pose.pt")   # auto-downloads ~6 MB on first use
            logger.info("[Pose] YOLOv8n-pose model loaded (%.1f MB)",
                        sum(p.numel() * 4 for p in _model.model.parameters()) / 1e6)
        except Exception as e:
            logger.error("[Pose] Could not load YOLOv8n-pose: %s — "
                         "run: pip install ultralytics", e)
            _model = _LOAD_FAILED
    return None if _model is _LOAD_FAILED else _model


def _angle(a, b, c) -> float:
    """Angle at joint b given three (x,y) points, in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def estimate_pose_both(crop_bgr: np.ndarray, min_conf: float = 0.3):
    """Run ONE model inference pass and return (jpeg_bytes, data_dict).
    This is the preferred entry point — twice as fast as calling
    estimate_pose_jpeg + estimate_pose_data separately.

    Returns:
        jpeg  — annotated JPEG bytes with skeleton overlay (None on failure)
        data  — dict with detected/keypoints/angles/posture keys
    """
    _empty = {"detected": False, "keypoints": {}, "angles": {}, "posture": "unknown"}
    model  = _get_model()
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return None, _empty

    h, w   = crop_bgr.shape[:2]
    scale  = max(1.0, 192 / min(h, w))
    inp    = crop_bgr
    if scale > 1.0:
        nh, nw = int(h * scale), int(w * scale)
        inp    = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w   = nh, nw

    try:
        results = model(inp, verbose=False, conf=min_conf)
        if not results or results[0].keypoints is None:
            return _no_detection_jpeg(inp), _empty

        kp_tensor = results[0].keypoints.data   # (N, 17, 3)
        if kp_tensor.shape[0] == 0:
            return _no_detection_jpeg(inp), _empty

        # ── JPEG ──────────────────────────────────────────────────────────
        annotated = results[0].plot(
            conf=False, labels=False, boxes=False,
            kpt_radius=5, line_width=2,
        )
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        jpeg = buf.tobytes() if ok else None

        # ── Data ──────────────────────────────────────────────────────────
        kp = kp_tensor[0].cpu().numpy()   # (17, 3)

        kp_dict = {}
        for i, name in enumerate(_KP_NAMES):
            x, y, c = kp[i]
            kp_dict[name] = {"x": float(x / w), "y": float(y / h), "conf": float(c)}

        def xy(name):
            pt = kp_dict.get(name, {})
            return (pt["x"], pt["y"]) if pt.get("conf", 0) >= min_conf else None

        ang = {}
        for label, (a, b, c) in [
            ("left_elbow",  ("left_shoulder",  "left_elbow",  "left_wrist")),
            ("right_elbow", ("right_shoulder", "right_elbow", "right_wrist")),
            ("left_knee",   ("left_hip",       "left_knee",   "left_ankle")),
            ("right_knee",  ("right_hip",      "right_knee",  "right_ankle")),
            ("left_hip",    ("left_shoulder",  "left_hip",    "left_knee")),
            ("right_hip",   ("right_shoulder", "right_hip",   "right_knee")),
        ]:
            pa, pb, pc = xy(a), xy(b), xy(c)
            if pa and pb and pc:
                ang[label] = round(_angle(pa, pb, pc), 1)

        posture = "unknown"
        lhip, rhip = xy("left_hip"),  xy("right_hip")
        lkn,  rkn  = xy("left_knee"), xy("right_knee")
        if lhip and rhip and lkn and rkn:
            hip_y  = (lhip[1] + rhip[1]) / 2
            knee_y = (lkn[1]  + rkn[1])  / 2
            ratio  = (knee_y - hip_y) / max(hip_y, 0.01)
            posture = "sitting" if ratio < 0.2 else "crouching" if ratio < 0.45 else "standing"

        data = {"detected": True, "keypoints": kp_dict, "angles": ang, "posture": posture}
        return jpeg, data

    except Exception as e:
        logger.error("[Pose] estimate_pose_both error: %s", e, exc_info=True)
        return None, _empty


def estimate_pose_jpeg(crop_bgr: np.ndarray, min_conf: float = 0.3) -> bytes | None:
    """
    Run pose estimation on a BGR person-crop.
    Returns an annotated JPEG (skeleton overlay) as bytes, or None on failure.
    """
    model = _get_model()
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return None

    # Upscale tiny crops so the model has enough resolution to work with
    h, w = crop_bgr.shape[:2]
    scale = max(1.0, 192 / min(h, w))
    if scale > 1.0:
        nh, nw = int(h * scale), int(w * scale)
        crop_bgr = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    try:
        results = model(crop_bgr, verbose=False, conf=min_conf)
        if not results or results[0].keypoints is None:
            return _no_detection_jpeg(crop_bgr)

        annotated = results[0].plot(
            conf=False, labels=False, boxes=False,
            kpt_radius=5, line_width=2,
        )
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None
    except Exception as e:
        logger.error("[Pose] Inference error: %s", e, exc_info=True)
        return None


def estimate_pose_data(crop_bgr: np.ndarray, min_conf: float = 0.3) -> dict:
    """
    Return structured pose data dict:
    {
      "detected": bool,
      "keypoints": {name: {"x": float, "y": float, "conf": float}, ...},
      "angles": {"left_elbow": float, "right_elbow": float,
                 "left_knee": float,  "right_knee": float, ...},
      "posture": "standing" | "sitting" | "crouching" | "unknown"
    }
    Coordinates are normalised to [0, 1] relative to the crop size.
    """
    model = _get_model()
    result = {"detected": False, "keypoints": {}, "angles": {}, "posture": "unknown"}
    if model is None or crop_bgr is None or crop_bgr.size == 0:
        return result

    h, w = crop_bgr.shape[:2]
    scale = max(1.0, 192 / min(h, w))
    inp = crop_bgr
    if scale > 1.0:
        nh, nw = int(h * scale), int(w * scale)
        inp = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = nh, nw

    try:
        results = model(inp, verbose=False, conf=min_conf)
        if not results or results[0].keypoints is None:
            return result

        kp = results[0].keypoints.data  # shape (N_persons, 17, 3)
        if kp.shape[0] == 0:
            return result

        # Use the first (most confident) person
        kp = kp[0].cpu().numpy()   # (17, 3) — x, y, conf
        result["detected"] = True

        kp_dict = {}
        for i, name in enumerate(_KP_NAMES):
            x, y, c = kp[i]
            kp_dict[name] = {"x": float(x / w), "y": float(y / h), "conf": float(c)}
        result["keypoints"] = kp_dict

        # Helper to get (x,y) if confident enough, else None
        def xy(name):
            pt = kp_dict.get(name, {})
            if pt.get("conf", 0) < min_conf:
                return None
            return (pt["x"], pt["y"])

        # Compute joint angles
        ang = {}
        pairs = [
            ("left_elbow",  ("left_shoulder",  "left_elbow",  "left_wrist")),
            ("right_elbow", ("right_shoulder", "right_elbow", "right_wrist")),
            ("left_knee",   ("left_hip",       "left_knee",   "left_ankle")),
            ("right_knee",  ("right_hip",      "right_knee",  "right_ankle")),
            ("left_hip",    ("left_shoulder",  "left_hip",    "left_knee")),
            ("right_hip",   ("right_shoulder", "right_hip",   "right_knee")),
        ]
        for label, (a, b, c) in pairs:
            pa, pb, pc = xy(a), xy(b), xy(c)
            if pa and pb and pc:
                ang[label] = round(_angle(pa, pb, pc), 1)
        result["angles"] = ang

        # Simple posture heuristic using hip and knee y-positions
        lhip, rhip = xy("left_hip"), xy("right_hip")
        lkn,  rkn  = xy("left_knee"), xy("right_knee")
        lank, rank = xy("left_ankle"), xy("right_ankle")
        if lhip and rhip and lkn and rkn:
            hip_y  = (lhip[1] + rhip[1]) / 2
            knee_y = (lkn[1] + rkn[1]) / 2
            ratio  = (knee_y - hip_y) / max(hip_y, 0.01)
            if ratio < 0.2:
                result["posture"] = "sitting"
            elif ratio < 0.45:
                result["posture"] = "crouching"
            else:
                result["posture"] = "standing"

        return result
    except Exception as e:
        logger.error("[Pose] Data extraction error: %s", e, exc_info=True)
        return result


def _no_detection_jpeg(img: np.ndarray) -> bytes | None:
    """Return the original crop with a 'No pose detected' label."""
    out = img.copy()
    cv2.putText(out, "No pose detected", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None
