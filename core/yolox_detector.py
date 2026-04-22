import os
import cv2
import numpy as np
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)

class YOLOXDetector:
    """
    Pure ONNXRuntime wrapper for YOLOX with automatic TensorRT acceleration.

    Provider priority (best → worst):
      1. TensorrtExecutionProvider  – auto-converts ONNX → TensorRT engine,
         caches it to disk so subsequent boots are instant.
      2. CUDAExecutionProvider      – fast GPU inference without conversion.
      3. CPUExecutionProvider       – universal fallback.

    The TensorRT engine is GPU-architecture-specific: a cached engine built
    on an RTX 3050 (SM 8.6) will NOT work on a Jetson Orin Nano (SM 8.7).
    The cache directory uses the GPU name in its path to avoid collisions
    when the same model folder is shared (e.g. via Git).

    License: Apache 2.0 (no Ultralytics / AGPL dependency).
    """

    def __init__(self, model_path, providers=None, conf_threshold=0.25, nms_threshold=0.45, force_trt=False):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # ── Build the provider list with TensorRT caching ────────────────
        if providers is None:
            providers = self._build_providers(model_path, force_trt=force_trt)

        available = ort.get_available_providers()
        # Filter to only providers that are actually installed
        valid_providers = []
        for p in providers:
            name = p if isinstance(p, str) else p[0]
            if name in available:
                valid_providers.append(p)
        if not valid_providers:
            valid_providers = ['CPUExecutionProvider']

        logger.info(f"ONNX providers requested -> {[p if isinstance(p, str) else p[0] for p in valid_providers]}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Try loading with full provider chain; if TensorRT DLL fails,
        # silently retry without it (avoids noisy error spam on Windows)
        try:
            self.session = ort.InferenceSession(model_path, sess_options=so, providers=valid_providers)
        except Exception:
            fallback = [p for p in valid_providers
                        if (p if isinstance(p, str) else p[0]) != 'TensorrtExecutionProvider']
            if not fallback:
                fallback = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, sess_options=so, providers=fallback)

        active = self.session.get_providers()
        logger.info(f"ONNX active providers   -> {active}")
        if 'TensorrtExecutionProvider' in active:
            logger.info("TensorRT acceleration is ACTIVE (engine cached on disk)")
        elif 'CUDAExecutionProvider' in active:
            logger.info("CUDA acceleration is active")
        else:
            logger.warning("Running on CPU only - performance will be limited")

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        # YOLOX exports can have dynamic or fixed shapes
        self.input_size = (input_shape[2], input_shape[3])
        if isinstance(self.input_size[0], str) or self.input_size[0] is None:
            self.input_size = (640, 640)

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_providers(model_path, force_trt=False):
        """
        Build provider list with TensorRT cache directory next to the model.
        Engine files are stored per-GPU so switching between RTX 3050 and
        Jetson Orin Nano doesn't cause stale-engine crashes.
        """
        model_dir = os.path.dirname(os.path.abspath(model_path))
        trt_cache = os.path.join(model_dir, "trt_engine_cache")
        os.makedirs(trt_cache, exist_ok=True)

        providers = []
        # TRT is used on Jetson (Linux) automatically.
        # On Windows, only enabled when --force-trt flag is passed explicitly.
        if os.name != 'nt' or force_trt:
            providers.append(
                ('TensorrtExecutionProvider', {
                    'trt_max_workspace_size': str(4 * 1024 * 1024 * 1024),  # 4 GB for better TRT kernel selection
                    'trt_fp16_enable': 'True',
                    'trt_engine_cache_enable': 'True',
                    'trt_engine_cache_path': trt_cache,
                    'trt_max_partition_iterations': '10',
                    'trt_min_subgraph_size': '5',
                })
            )
        
        providers.extend([
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ])
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
        boxes_xywh = np.stack([x1, y1, w, h], axis=-1) / ratio

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
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
