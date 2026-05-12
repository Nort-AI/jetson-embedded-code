"""
trt_session.py — TensorRT 10.x inference session.

Drop-in replacement for onnxruntime.InferenceSession.
Uses only the system tensorrt module (python3-libnvinfer) and ctypes —
no onnxruntime-gpu, no pycuda, no cuda-python required.

Pre-build engines once with:
    python3 scripts/build_engines.py

Then run.py automatically prefers .engine files over .onnx.
"""
import ctypes
import ctypes.util
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

# ── CUDA runtime via ctypes (no extra pip package needed) ────────────────────

def _load_cudart():
    """Load libcudart.so — searches standard Jetson/Linux paths."""
    candidates = [
        "libcudart.so.12",
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12",
    ]
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            lib.cudaMalloc.restype        = ctypes.c_int
            lib.cudaFree.restype          = ctypes.c_int
            lib.cudaMemcpy.restype        = ctypes.c_int
            lib.cudaMemcpyAsync.restype   = ctypes.c_int
            lib.cudaStreamCreate.restype  = ctypes.c_int
            lib.cudaStreamSynchronize.restype = ctypes.c_int
            return lib
        except OSError:
            continue
    return None


_cudart = _load_cudart()

# cudaMemcpy direction constants
_H2D = 1   # cudaMemcpyHostToDevice
_D2H = 2   # cudaMemcpyDeviceToHost


def _cuda_malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    err = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if err != 0:
        raise RuntimeError(f"cudaMalloc({nbytes}) failed: error {err}")
    return ptr.value


def _cuda_free(ptr: int):
    if ptr:
        _cudart.cudaFree(ctypes.c_void_p(ptr))


def _cuda_h2d(dst: int, src_array: np.ndarray, stream: int):
    err = _cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src_array.ctypes.data),
        ctypes.c_size_t(src_array.nbytes),
        ctypes.c_int(_H2D),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpyAsync H2D failed: error {err}")


def _cuda_d2h(dst_array: np.ndarray, src: int, stream: int):
    err = _cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst_array.ctypes.data),
        ctypes.c_void_p(src),
        ctypes.c_size_t(dst_array.nbytes),
        ctypes.c_int(_D2H),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpyAsync D2H failed: error {err}")


def _cuda_stream_create() -> int:
    stream = ctypes.c_void_p()
    err = _cudart.cudaStreamCreate(ctypes.byref(stream))
    if err != 0:
        raise RuntimeError(f"cudaStreamCreate failed: error {err}")
    return stream.value


def _cuda_sync(stream: int):
    _cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))


# ── tensorrt import (tries system python3-libnvinfer paths) ──────────────────

def _import_trt():
    """
    Import tensorrt, preferring the system python3-libnvinfer package over any
    pip-installed stub wheel.  The pip tensorrt wheels for Linux x86/aarch64
    may import successfully but lack the C bindings (no trt.Logger, no
    trt.Runtime, etc.) — we validate the import before returning it.

    python3-libnvinfer installs to system Python's dist-packages.  On Jetson
    the system Python and the conda Python share the same CPython ABI, so the
    extension can be loaded cross-environment once the path is added.
    """
    system_paths = [
        "/usr/lib/python3/dist-packages",
        "/usr/lib/python3.10/dist-packages",
        "/usr/lib/python3.11/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/usr/local/lib/python3.11/dist-packages",
    ]

    def _is_valid(trt) -> bool:
        """Return True only if the module has real C bindings (not a stub wheel)."""
        return hasattr(trt, "Logger") and hasattr(trt, "Runtime")

    # First try: whatever is already importable (conda env or system)
    try:
        import tensorrt as trt
        if _is_valid(trt):
            return trt
        # Stub wheel imported — purge it so we can try the system path
        sys.modules.pop("tensorrt", None)
        logger.debug("[TRT] pip tensorrt stub detected (no Logger/Runtime) — trying system path")
    except ImportError:
        pass

    # Second try: system dist-packages (python3-libnvinfer)
    for p in system_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        import tensorrt as trt
        if _is_valid(trt):
            return trt
    except ImportError:
        pass

    return None


# ── Availability check ────────────────────────────────────────────────────────

def is_available() -> bool:
    """
    True if both TensorRT Python module and libcudart are usable.
    Call this before constructing TRTSession.
    """
    if _cudart is None:
        logger.debug("[TRT] libcudart not found — TRT unavailable")
        return False
    trt = _import_trt()
    if trt is None:
        logger.debug("[TRT] tensorrt Python module not found — TRT unavailable")
        return False
    return True


# ── Mock NodeArg — matches onnxruntime.NodeArg interface ─────────────────────

class _NodeArg:
    def __init__(self, name: str, shape: tuple, dtype):
        self.name  = name
        self.shape = shape
        self.type  = str(dtype)


# ── Main session class ────────────────────────────────────────────────────────

class TRTSession:
    """
    TensorRT 10.x inference session with an onnxruntime.InferenceSession-
    compatible API.

    Calling code that does:
        session.get_inputs()[0].name
        session.run(None, {input_name: array})
        session.get_providers()

    …works identically with TRTSession and ORT InferenceSession.
    """

    def __init__(self, engine_path: str):
        trt = _import_trt()
        if trt is None:
            raise ImportError(
                "tensorrt Python module not found. "
                "Install JetPack TRT: sudo apt install python3-libnvinfer"
            )
        if _cudart is None:
            raise ImportError("libcudart not found — cannot run CUDA operations")

        self._trt = trt

        # ── Load serialised engine ────────────────────────────────────────────
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"TRT: failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self._stream  = _cuda_stream_create()

        # ── Enumerate tensors and pre-allocate GPU memory ─────────────────────
        self._inputs  = {}  # name → {'ptr', 'shape', 'dtype', 'nbytes'}
        self._outputs = {}

        for i in range(self.engine.num_io_tensors):
            name     = self.engine.get_tensor_name(i)
            shape    = tuple(self.engine.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Replace dynamic dims (-1) with 1 for initial allocation
            alloc_shape = tuple(max(1, d) for d in shape)
            nbytes      = int(np.prod(alloc_shape)) * np.dtype(np_dtype).itemsize
            ptr         = _cuda_malloc(nbytes)

            bucket = (
                self._inputs
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                else self._outputs
            )
            bucket[name] = {
                "ptr":    ptr,
                "shape":  shape,
                "dtype":  np_dtype,
                "nbytes": nbytes,
            }

        logger.info(
            f"[TRT] Loaded {os.path.basename(engine_path)} | "
            f"in={list(self._inputs)} out={list(self._outputs)}"
        )

    # ── onnxruntime-compatible interface ──────────────────────────────────────

    def get_inputs(self):
        return [
            _NodeArg(name, meta["shape"], meta["dtype"])
            for name, meta in self._inputs.items()
        ]

    def get_providers(self):
        return ["TensorrtExecutionProvider"]

    def run(self, output_names, input_feed: dict) -> list:
        # ── Upload inputs to GPU ──────────────────────────────────────────────
        for name, data in input_feed.items():
            if name not in self._inputs:
                continue
            meta = self._inputs[name]
            data = np.ascontiguousarray(data, dtype=meta["dtype"])

            if data.nbytes > meta["nbytes"]:            # grow buffer if needed
                # C6-fix: restore old pointer if realloc fails to avoid VRAM leak
                _old_ptr = meta["ptr"]
                try:
                    meta["ptr"]    = _cuda_malloc(data.nbytes)
                    meta["nbytes"] = data.nbytes
                    _cuda_free(_old_ptr)
                except RuntimeError:
                    meta["ptr"] = _old_ptr   # keep old buffer valid
                    raise

            _cuda_h2d(meta["ptr"], data, self._stream)
            self.context.set_tensor_address(name, meta["ptr"])

        # ── Set output addresses ──────────────────────────────────────────────
        for name, meta in self._outputs.items():
            self.context.set_tensor_address(name, meta["ptr"])

        # ── Inference ─────────────────────────────────────────────────────────
        if not self.context.execute_async_v3(self._stream):
            raise RuntimeError("TRT execute_async_v3 returned False")

        # ── Download outputs from GPU ─────────────────────────────────────────
        results = []
        targets = output_names or list(self._outputs.keys())
        for name in targets:
            meta = self._outputs[name]

            # Use runtime shape (handles dynamic output dims)
            try:
                actual_shape = tuple(self.context.get_tensor_shape(name))
            except Exception:
                actual_shape = tuple(max(1, d) for d in meta["shape"])

            out = np.empty(actual_shape, dtype=meta["dtype"])
            _cuda_d2h(out, meta["ptr"], self._stream)
            results.append(out)

        _cuda_sync(self._stream)
        return results

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def __del__(self):
        try:
            for meta in list(self._inputs.values()) + list(self._outputs.values()):
                _cuda_free(meta["ptr"])
        except Exception:
            pass
