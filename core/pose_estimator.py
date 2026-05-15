"""
pose_estimator.py — Human pose estimation for Nort edge AI.

PRIMARY backend: MediaPipe BlazePose (Google)
  License  : Apache 2.0 — free for commercial use
  Quality  : 33 landmarks, sub-pixel accuracy
  Install  : pip install mediapipe        (works on Jetson aarch64 JetPack 6.2+)

FALLBACK backend: YOLOv8n-pose (Ultralytics)
  License  : AGPL-3.0 — requires a paid enterprise licence for closed-source
             commercial products.  Use only for development / evaluation.
  Install  : pip install ultralytics

Public API (unchanged):
    from core.pose_estimator import estimate_pose_both
    jpeg, data = estimate_pose_both(crop_bgr)

data schema:
    {
      "detected" : bool,
      "landmarks": [{x,y,z,vis}, ...]    # 33 MediaPipe landmarks, normalised [0,1]
      "keypoints": {name: {x,y,conf}}    # COCO-17 subset (backward compat)
      "angles"   : {joint: degrees}
      "posture"  : "standing"|"sitting"|"crouching"|"unknown"
    }
"""

import cv2
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)

# ── Lazy-load sentinels ────────────────────────────────────────────────────────
_LOAD_FAILED = object()   # distinct from None (= not tried yet)

_mp_model      = None   # lazily initialised; set to _LOAD_FAILED on import error
_mp_lock       = threading.Lock()

_yolo_model    = None
_yolo_lock     = threading.Lock()

# ── MediaPipe 33-landmark → COCO-17 mapping ───────────────────────────────────
_MP_TO_COCO = {
     0: "nose",
     2: "left_eye",    5: "right_eye",
     7: "left_ear",    8: "right_ear",
    11: "left_shoulder",  12: "right_shoulder",
    13: "left_elbow",     14: "right_elbow",
    15: "left_wrist",     16: "right_wrist",
    23: "left_hip",       24: "right_hip",
    25: "left_knee",      26: "right_knee",
    27: "left_ankle",     28: "right_ankle",
}

_COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_EMPTY = {
    "detected": False, "landmarks": [],
    "keypoints": {}, "angles": {}, "posture": "unknown",
}


# ── Model loaders ──────────────────────────────────────────────────────────────

def _get_mp():
    """Lazy-load MediaPipe Pose once; thread-safe."""
    global _mp_model
    if _mp_model is not None:
        return None if _mp_model is _LOAD_FAILED else _mp_model
    with _mp_lock:
        if _mp_model is not None:
            return None if _mp_model is _LOAD_FAILED else _mp_model
        try:
            import mediapipe as mp
            _mp_model = mp.solutions.pose.Pose(
                static_image_mode=True,
                # complexity=2 (heavy model) segfaults on Jetson ARM with this mediapipe build.
                # complexity=1 (lite+full) is stable and still significantly better than 0.
                model_complexity=1,
                smooth_landmarks=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            logger.info("[Pose] MediaPipe BlazePose loaded — Apache 2.0, commercial use OK")
        except Exception as e:
            logger.error("[Pose] MediaPipe unavailable: %s  →  pip install mediapipe", e)
            _mp_model = _LOAD_FAILED
    return None if _mp_model is _LOAD_FAILED else _mp_model


def _get_yolo():
    """Lazy-load YOLOv8n-pose as fallback. Logs AGPL warning once."""
    global _yolo_model
    if _yolo_model is not None:
        return None if _yolo_model is _LOAD_FAILED else _yolo_model
    with _yolo_lock:
        if _yolo_model is not None:
            return None if _yolo_model is _LOAD_FAILED else _yolo_model
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n-pose.pt")
            logger.warning(
                "[Pose] Falling back to YOLOv8n-pose (AGPL-3.0). "
                "Install MediaPipe for commercial-safe pose: pip install mediapipe"
            )
        except Exception as e:
            logger.error("[Pose] YOLOv8n-pose also unavailable: %s", e)
            _yolo_model = _LOAD_FAILED
    return None if _yolo_model is _LOAD_FAILED else _yolo_model


# ── Helpers ────────────────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _no_detection_jpeg(img: np.ndarray) -> bytes | None:
    out = img.copy()
    cv2.putText(out, "No pose detected", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def _upscale(crop_bgr: np.ndarray):
    """Upscale tiny crops and add contextual border padding.

    MediaPipe BlazePose needs the full body visible with some surrounding
    context to reliably initialise its body-prior.  Two improvements vs the
    old code:
      1. Minimum short-side raised from 192 → 256 px for better keypoint
         localisation on typical CCTV crops.
      2. After scaling, add a 10 % border on all sides (filled with the mean
         edge colour so the model doesn't see a hard black cut-off).  This
         gives the hip/ankle detectors the context they need when the crop
         is tightly bounded to the person.
    """
    h, w = crop_bgr.shape[:2]
    scale = max(1.0, 256 / min(h, w))
    if scale > 1.0:
        nh, nw = int(h * scale), int(w * scale)
        crop_bgr = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = nh, nw

    # Add 10 % border so body extremities aren't cut at the crop edge
    pad_y = max(8, int(h * 0.10))
    pad_x = max(8, int(w * 0.10))
    # Use replicate border (mirrors edge pixels) rather than black — avoids
    # creating an artificial background that confuses body segmentation
    return cv2.copyMakeBorder(
        crop_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )


def _compute_angles_and_posture(kp_dict: dict, min_conf: float):
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
        posture = ("sitting" if ratio < 0.2 else
                   "crouching" if ratio < 0.45 else "standing")
    return ang, posture


# ── Primary path: MediaPipe ────────────────────────────────────────────────────

def _estimate_mediapipe(model, inp: np.ndarray, min_conf: float):
    try:
        import mediapipe as mp
        mp_drawing        = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose           = mp.solutions.pose

        rgb     = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        results = model.process(rgb)

        if not results.pose_landmarks:
            return _no_detection_jpeg(inp), dict(_EMPTY)

        # ── JPEG: draw MediaPipe skeleton on the crop ─────────────────────
        annotated = inp.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        jpeg = buf.tobytes() if ok else None

        # ── Data: store all 33 landmarks + COCO-17 subset ────────────────
        raw = results.pose_landmarks.landmark
        landmarks = [
            {"x": float(lm.x), "y": float(lm.y),
             "z": float(lm.z), "vis": float(lm.visibility)}
            for lm in raw
        ]

        kp_dict = {}
        for mp_idx, coco_name in _MP_TO_COCO.items():
            lm = raw[mp_idx]
            kp_dict[coco_name] = {
                "x": float(lm.x), "y": float(lm.y), "conf": float(lm.visibility),
            }

        ang, posture = _compute_angles_and_posture(kp_dict, min_conf)

        data = {
            "detected":  True,
            "landmarks": landmarks,
            "keypoints": kp_dict,
            "angles":    ang,
            "posture":   posture,
        }
        return jpeg, data

    except Exception as e:
        logger.error("[Pose] MediaPipe inference error: %s", e, exc_info=True)
        return None, dict(_EMPTY)


# ── Fallback path: YOLOv8n-pose (AGPL-3.0) ───────────────────────────────────

def _estimate_yolo(model, inp: np.ndarray, min_conf: float):
    try:
        h, w = inp.shape[:2]
        results = model(inp, verbose=False, conf=min_conf)
        if not results or results[0].keypoints is None:
            return _no_detection_jpeg(inp), dict(_EMPTY)

        kp_tensor = results[0].keypoints.data
        if kp_tensor.shape[0] == 0:
            return _no_detection_jpeg(inp), dict(_EMPTY)

        annotated = results[0].plot(
            conf=False, labels=False, boxes=False, kpt_radius=5, line_width=2,
        )
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        jpeg = buf.tobytes() if ok else None

        kp = kp_tensor[0].cpu().numpy()
        kp_dict = {}
        for i, name in enumerate(_COCO_KP_NAMES):
            x, y, c = kp[i]
            kp_dict[name] = {"x": float(x / w), "y": float(y / h), "conf": float(c)}

        ang, posture = _compute_angles_and_posture(kp_dict, min_conf)

        data = {
            "detected":  True,
            "landmarks": [],          # YOLO fallback has no 33-landmark data
            "keypoints": kp_dict,
            "angles":    ang,
            "posture":   posture,
        }
        return jpeg, data

    except Exception as e:
        logger.error("[Pose] YOLO inference error: %s", e, exc_info=True)
        return None, dict(_EMPTY)


# ── Public entry point ─────────────────────────────────────────────────────────

def estimate_pose_both(crop_bgr: np.ndarray, min_conf: float = 0.3):
    """Run one inference pass and return (jpeg_bytes, data_dict).

    Tries MediaPipe (Apache 2.0) first; falls back to YOLOv8n-pose (AGPL-3.0)
    if MediaPipe is not installed.

    Returns:
        jpeg — JPEG bytes of the crop with skeleton overlay drawn on it
        data — dict with detected/landmarks/keypoints/angles/posture
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, dict(_EMPTY)

    inp = _upscale(crop_bgr)

    model = _get_mp()
    if model is not None:
        return _estimate_mediapipe(model, inp, min_conf)

    yolo = _get_yolo()
    if yolo is not None:
        return _estimate_yolo(yolo, inp, min_conf)

    return _no_detection_jpeg(inp), dict(_EMPTY)


# ── Legacy single-function wrappers (kept for any direct callers) ──────────────

def estimate_pose_jpeg(crop_bgr: np.ndarray, min_conf: float = 0.3) -> bytes | None:
    jpeg, _ = estimate_pose_both(crop_bgr, min_conf)
    return jpeg


def estimate_pose_data(crop_bgr: np.ndarray, min_conf: float = 0.3) -> dict:
    _, data = estimate_pose_both(crop_bgr, min_conf)
    return data
