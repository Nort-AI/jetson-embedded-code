"""
vlm_microservice.py — Isolated Moondream inference process.

Runs as a separate OS process with deliberately low priority so that the
Jetson GPU/memory bus is NOT saturated during inference, keeping the main
camera pipeline and MJPEG streaming completely fluid.

Key design decisions:
  - Process nice() is set to +19 (lowest CPU priority) on startup.
  - PyTorch CUDA memory fraction is limited to 40% of GPU memory so that the
    VideoCapture NVDEC decoder (which uses the remaining ~60%) is never starved.
  - All requests are serialized through a threading.Lock so requests that arrive
    while inference is running receive an immediate "busy" 503 with Retry-After
    header instead of queuing up and saturating memory.
  - Only one thread actually runs Moondream; Flask's threaded=True handles the
    HTTP accept/reject loop.
"""

import os
import io
import sys
import time
import logging
import threading

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_microservice")

# ── Step 1: Deprioritize this process IMMEDIATELY on startup ──────────────────
# This is critical on Jetson: if we don't lower our OS scheduling priority,
# PyTorch CUDA kernels will preempt the camera decode threads.
try:
    if hasattr(os, 'nice'):
        os.nice(19)          # Linux / Jetson — lowest CPU scheduling priority
        logger.info("[Priority] Set os.nice(19) — lowest CPU priority.")
    else:
        # Windows fallback (development only; production is always Linux/Jetson)
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(),
            0x00000040  # IDLE_PRIORITY_CLASS
        )
        logger.info("[Priority] Set Windows IDLE_PRIORITY_CLASS.")
except Exception as _pe:
    logger.warning(f"[Priority] Could not set low priority: {_pe}")

from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_model        = None
_model_loaded = False
_model_lock   = threading.Lock()   # serializes inference; only 1 job at a time

# ── Step 2: Load model with CUDA memory limits ────────────────────────────────

def _load_model() -> bool:
    """Load Moondream once, with CUDA memory capped at 40% of GPU memory."""
    global _model, _model_loaded
    if _model_loaded:
        return _model is not None

    try:
        # Limit CUDA memory fraction BEFORE importing moondream/torch.
        # This ensures that when PyTorch allocates its CUDA memory pool it only
        # takes 40% of the GPU RAM, leaving the rest for NVDEC / YOLO in the
        # main process.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.40)
                logger.info("[CUDA] Memory limited to 40%% of GPU RAM.")
        except Exception as _te:
            logger.warning(f"[CUDA] Could not limit memory fraction: {_te}")

        import moondream as md
        logger.info("[VLM] Loading local Moondream model…")
        _model = md.vl(local=True)
        _model_loaded = True
        logger.info("[VLM] Moondream loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"[VLM] Failed to load Moondream: {e}")
        _model_loaded = True
        return False


# ── Inference endpoint ────────────────────────────────────────────────────────

@app.route('/vlm/query', methods=['POST'])
def query_vlm():
    """
    POST /vlm/query
    Multipart form:
      image    — JPEG image file
      question — text prompt

    Returns immediately with 503 + Retry-After:2 if inference is already running,
    so the caller can retry without piling up requests.
    """
    # Try to acquire lock without blocking
    acquired = _model_lock.acquire(blocking=False)
    if not acquired:
        # Tell the caller to retry in 2 seconds
        resp = jsonify({"error": "busy", "retry_after": 2})
        resp.headers["Retry-After"] = "2"
        return resp, 503

    try:
        if not _load_model() or _model is None:
            return jsonify({"error": "VLM not available"}), 503

        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        question = request.form.get("question", "What is happening in this image?")

        img_bytes = request.files['image'].read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        start_t = time.time()

        # ── Step 3: Release CUDA cache before inference to reduce memory spike ──
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        encoded = _model.encode_image(pil_img)
        result  = _model.query(encoded, question)
        inference_ms = int((time.time() - start_t) * 1000)

        # Normalise result type
        if hasattr(result, "answer"):
            ans = result.answer
            ans_str = "".join(ans) if hasattr(ans, "__iter__") and not isinstance(ans, str) else str(ans)
        elif isinstance(result, dict):
            ans_str = result.get("answer", str(result))
        else:
            ans_str = str(result)

        logger.info(f"[VLM] Query done in {inference_ms}ms — '{question[:60]}'")
        return jsonify({"result": ans_str.strip(), "inference_ms": inference_ms})

    except Exception as e:
        logger.error(f"[VLM] Inference exception: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        _model_lock.release()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": _model_loaded}), 200


if __name__ == '__main__':
    logger.info("Starting NORT VLM Microservice on port 5055…")
    # threaded=True: Flask can accept the next request while still in the prior
    # request's handler thread.  The _model_lock above ensures only 1 inference
    # runs at a time, but HTTP connections are never dropped.
    app.run(host='127.0.0.1', port=5055, threaded=True)
