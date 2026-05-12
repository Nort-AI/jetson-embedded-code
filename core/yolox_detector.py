import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _onnx_path_to_engine(model_path: str) -> str:
    """Return the pre-built .engine path next to the .onnx file."""
    base, _ = os.path.splitext(model_path)
    return base + ".engine"


class YOLOXDetector:
    """
    YOLOX inference wrapper — TensorRT-first on Jetson, ORT fallback elsewhere.

    Loading priority:
      1. Pre-built .engine file next to the .onnx  → native TRT 10.x via
         TRTSession (core/trt_session.py).  Zero onnxruntime dependency.
      2. ONNXRuntime with TensorrtExecutionProvider → auto-engine-cache.
      3. ONNXRuntime with CUDAExecutionProvider.
      4. ONNXRuntime with CPUExecutionProvider (last resort).

    Path (1) is the normal Jetson path after running:
        python3 scripts/build_engines.py

    License: Apache 2.0 (no Ultralytics / AGPL dependency).
    """

    def __init__(self, model_path, providers=None, conf_threshold=0.25,
                 nms_threshold=0.45, force_trt=False):
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold

        # ── 1. Try native TRT session (Jetson-preferred path) ────────────────
        engine_path = _onnx_path_to_engine(model_path)
        self.session = None

        if os.path.exists(engine_path):
            try:
                from core.trt_session import TRTSession, is_available
                if is_available():
                    self.session = TRTSession(engine_path)
                    logger.info(
                        f"[YOLOX] ✓ TRT native session loaded: {os.path.basename(engine_path)}"
                    )
            except Exception as _e:
                logger.warning(f"[YOLOX] TRTSession failed ({_e}), falling back to ORT")
                self.session = None

        # ── 2. Fall back to ONNXRuntime ──────────────────────────────────────
        if self.session is None:
            self.session = self._load_ort_session(model_path, providers, force_trt)

        # ── Resolve input metadata ────────────────────────────────────────────
        self.input_name = self.session.get_inputs()[0].name
        input_shape     = self.session.get_inputs()[0].shape

        # YOLOX exports can have dynamic or fixed shapes
        self.input_size = (input_shape[2], input_shape[3])
        if isinstance(self.input_size[0], str) or self.input_size[0] is None:
            self.input_size = (640, 640)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_ort_session(model_path, providers, force_trt):
        """Load an ONNXRuntime session with the best available provider chain."""
        import onnxruntime as ort

        if providers is None:
            providers = YOLOXDetector._build_providers(model_path, force_trt=force_trt)

        available = ort.get_available_providers()
        valid_providers = []
        for p in providers:
            name = p if isinstance(p, str) else p[0]
            if name in available:
                valid_providers.append(p)
        if not valid_providers:
            valid_providers = ["CPUExecutionProvider"]

        logger.info(f"[YOLOX] ORT providers requested -> "
                    f"{[p if isinstance(p, str) else p[0] for p in valid_providers]}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            session = ort.InferenceSession(
                model_path, sess_options=so, providers=valid_providers
            )
        except Exception:
            # TRT EP DLL missing / incompatible → retry without it
            fallback = [
                p for p in valid_providers
                if (p if isinstance(p, str) else p[0]) != "TensorrtExecutionProvider"
            ]
            if not fallback:
                fallback = ["CPUExecutionProvider"]
            session = ort.InferenceSession(
                model_path, sess_options=so, providers=fallback
            )

        active = session.get_providers()
        logger.info(f"[YOLOX] ORT active providers   -> {active}")
        if "TensorrtExecutionProvider" in active:
            logger.info("[YOLOX] ORT TensorRT acceleration is ACTIVE (engine cached)")
        elif "CUDAExecutionProvider" in active:
            logger.info("[YOLOX] ORT CUDA acceleration is active")
        else:
            logger.warning("[YOLOX] Running on CPU only — performance will be limited")

        return session

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_providers(model_path, force_trt=False):
        """Build ORT provider list with TRT engine-cache dir next to the model."""
        import platform

        # On Jetson (aarch64), GPU inference is handled exclusively by TRTSession
        # (.engine files).  ORT's TensorrtExecutionProvider and CUDAExecutionProvider
        # are NOT used here because:
        #   - The pip onnxruntime-gpu wheel is x86_64 and has no real CUDA/TRT EP on arm64
        #   - ORT TRT EP conflicts with the system python3-libnvinfer native libs already
        #     loaded by TRTSession, causing a segfault during provider initialisation
        # CPU is the correct ORT fallback on Jetson; TRTSession handles all GPU work.
        if platform.machine() == "aarch64":
            return ["CPUExecutionProvider"]

        model_dir = os.path.dirname(os.path.abspath(model_path))
        trt_cache  = os.path.join(model_dir, "trt_engine_cache")
        os.makedirs(trt_cache, exist_ok=True)

        providers = []
        # Enable TRT EP automatically on Linux x86_64; opt-in on Windows.
        if os.name != "nt" or force_trt:
            providers.append((
                "TensorrtExecutionProvider", {
                    "trt_max_workspace_size":      str(4 * 1024 * 1024 * 1024),
                    "trt_fp16_enable":             "True",
                    "trt_engine_cache_enable":     "True",
                    "trt_engine_cache_path":       trt_cache,
                    "trt_max_partition_iterations": "10",
                    "trt_min_subgraph_size":       "5",
                }
            ))
        providers.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        return providers

    # ─────────────────────────────────────────────────────────────────────
    def preproc(self, img, input_size, swap=(2, 0, 1)):
        """Letterbox-pad and resize to the target YOLOX input size."""
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    # ─────────────────────────────────────────────────────────────────────
    def detect(self, img):
        """Run detection and return Nx6 array of [x1, y1, x2, y2, conf, cls]."""
        tensor, ratio = self.preproc(img, self.input_size)
        tensor = np.expand_dims(tensor, axis=0)

        # Inference
        outputs = self.session.run(None, {self.input_name: tensor})
        predictions = outputs[0][0]  # (N, 85)

        # Post-process
        boxes_raw = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        class_ids = np.argmax(scores, axis=1)
        class_scores = np.max(scores, axis=1)

        # Strict person-only filter (COCO class 0)
        mask = (class_scores > self.conf_threshold) & (class_ids == 0)
        boxes_raw = boxes_raw[mask]
        class_ids = class_ids[mask]
        class_scores = class_scores[mask]

        if len(boxes_raw) == 0:
            return np.empty((0, 6))

        # Convert cx,cy,w,h → x1,y1,x2,y2
        cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1) / ratio
        # H1-fix: rename for clarity — cv2.dnn.NMSBoxes expects TOP-LEFT [x,y,w,h],
        # not center-based xywh.  x1/y1 here are already top-left (cx−w/2, cy−h/2).
        boxes_tlwh = np.stack([x1, y1, w, h], axis=-1) / ratio

        # NMS — pass top-left [x, y, w, h] as required by cv2.dnn.NMSBoxes
        indices = cv2.dnn.NMSBoxes(
            boxes_tlwh.tolist(),
            class_scores.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return np.empty((0, 6))

        keep = indices.flatten()
        results = np.column_stack([
            boxes_xyxy[keep],
            class_scores[keep],
            class_ids[keep].astype(float),
        ])
        return results
