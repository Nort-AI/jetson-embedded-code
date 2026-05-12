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

_TRT_SYSTEM_DIRS = [
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3.10/dist-packages",
    "/usr/lib/python3.11/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/local/lib/python3.11/dist-packages",
]

# Cache so we only do the import dance once per process
_trt_module = None


def _import_trt():
    """
    Return the tensorrt module with real C bindings (trt.Logger, trt.Runtime).

    Strategy:
    1. On aarch64 + Python ≥ 3.11: return None immediately.
       python3-libnvinfer on JetPack 6.x ships tensorrt.so compiled for Python 3.10
       only.  The .so exports PyInit_tensorrt with the CPython 3.10 ABI tag; calling
       any C-level object (trt.Logger(), trt.Runtime(), …) from a Python 3.11
       interpreter triggers an ABI mismatch that SEGFAULTS the process.
       hasattr() succeeds (the Python attribute object exists) but actually calling
       it is fatal — so we must skip before any hasattr check.
       Fix: run the pipeline in a Python 3.10 environment:
           conda create -n nort310 python=3.10
           conda activate nort310
           pip install -r requirements.txt
    2. Try whatever `import tensorrt` resolves to now.  If it has Logger+Runtime
       (i.e. it's the real python3-libnvinfer, not a pip stub) use it.
    3. If the pip stub was imported (no Logger/Runtime), pop it from sys.modules
       and retry after briefly prepending the JetPack system dist-packages path.
       We REMOVE the added paths afterwards so they don't pollute sys.path for
       the rest of the process (e.g. ORT would otherwise pick up the system
       onnxruntime instead of the conda one → segfault).

    The pip tensorrt stub on aarch64 Jetson does NOT load any native .so files
    (x86_64 bindings can't dlopen on arm64), so pop+reimport is safe here.
    """
    global _trt_module
    if _trt_module is not None:
        return _trt_module

    import platform as _platform

    # ── Guard: JetPack python3-libnvinfer is Python 3.10-only ABI ────────────
    # tensorrt.so at /usr/lib/python3.10/dist-packages/tensorrt/tensorrt.so is
    # compiled against CPython 3.10.  Any call to a C-level symbol (trt.Logger,
    # trt.Runtime, etc.) from Python 3.11+ causes an immediate SEGFAULT.
    # hasattr() returns True (the Python attribute wrapper exists) but the
    # underlying C call is fatal, so we must bail out before any hasattr check.
    if _platform.machine() == "aarch64" and sys.version_info >= (3, 11):
        logger.warning(
            "[TRT] Skipping TRT Python bindings on aarch64 + Python %d.%d: "
            "python3-libnvinfer only ships for Python 3.10 on JetPack 6.x "
            "(tensorrt.so has CPython 3.10 ABI — calling it from 3.11 segfaults). "
            "To enable GPU inference create a Python 3.10 env: "
            "  conda create -n nort310 python=3.10 && conda activate nort310 && "
            "  pip install -r requirements.txt",
            sys.version_info.major, sys.version_info.minor,
        )
        return None

    def _valid(trt) -> bool:
        return hasattr(trt, "Logger") and hasattr(trt, "Runtime")

    # ── Pass 1: try the default import ──────────────────────────────────────
    try:
        import tensorrt as trt
        if _valid(trt):
            _trt_module = trt
            return trt
        # pip stub imported — safe to purge on aarch64 (no native .so loaded)
        sys.modules.pop("tensorrt", None)
        logger.debug("[TRT] pip tensorrt stub detected (no Logger/Runtime) — trying system path")
    except ImportError:
        pass

    # ── Pass 2: temporarily prepend system dist-packages, import, then remove ─
    added = []
    for p in _TRT_SYSTEM_DIRS:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)

    try:
        import tensorrt as trt
        if _valid(trt):
            _trt_module = trt
            return trt
    except ImportError:
        pass
    finally:
        # Always remove added paths — do NOT leave them in sys.path.
        # Leaving system dist-packages in sys.path can cause other imports
        # (e.g. onnxruntime) to load system versions instead of conda ones.
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
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

    def get_outputs(self):
        return [
            _NodeArg(name, meta["shape"], meta["dtype"])
            for name, meta in self._outputs.items()
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
