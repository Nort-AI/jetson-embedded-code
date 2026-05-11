#!/usr/bin/env python3
"""
build_engines.py — Convert all Nort ONNX models to TensorRT .engine files.

Run ONCE on the Jetson after first deploy (or after a JetPack/TRT upgrade).
Each model takes 5-15 minutes. Engines are GPU-architecture-specific — do NOT
copy them between an Orin Nano, Orin NX, and AGX Orin (different SMs).

Usage:
    python3 scripts/build_engines.py [--fp32] [--workspace 4096]

After this script completes, run.py will automatically prefer .engine files
over .onnx files. No code changes needed.

Requirements:
    - trtexec from  sudo apt install libnvinfer-bin
    - Models must exist under assets/models/
"""
import argparse
import os
import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("build_engines")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model definitions ─────────────────────────────────────────────────────────
# Each entry: (display_name, onnx_path, extra_trtexec_flags)
MODELS = [
    (
        "YOLOX-M (detection)",
        os.path.join(ROOT, "assets", "models", "yolox_m.onnx"),
        # YOLOX has a single static input — no profile needed
        [],
    ),
    (
        "Attribute model (gender/age)",
        os.path.join(ROOT, "assets", "models", "attribute_model.onnx"),
        [],
    ),
    (
        "ReID OSNet-AIN x1.0",
        os.path.join(ROOT, "assets", "models", "osnet_ain_x1_0.onnx"),
        [],
    ),
]

# ── trtexec search paths ──────────────────────────────────────────────────────
_TRTEXEC_CANDIDATES = [
    "trtexec",                                  # on $PATH (JetPack default)
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/bin/trtexec",
]


def _find_trtexec() -> str:
    """Return the path to trtexec or raise."""
    for candidate in _TRTEXEC_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "--help"],
                capture_output=True, timeout=5,
            )
            if result.returncode in (0, 1):  # trtexec --help exits with 1 on some builds
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise FileNotFoundError(
        "trtexec not found. Install with:\n"
        "  sudo apt install libnvinfer-bin"
    )


def engine_path_for(onnx_path: str) -> str:
    """Return the .engine path next to the .onnx file."""
    base, _ = os.path.splitext(onnx_path)
    return base + ".engine"


def build_engine(
    trtexec: str,
    name: str,
    onnx_path: str,
    extra_flags: list,
    fp16: bool,
    workspace_mb: int,
    force: bool,
) -> bool:
    """
    Build one TRT engine via trtexec.
    Returns True on success, False on failure/skip.
    """
    if not os.path.exists(onnx_path):
        log.warning(f"[{name}] SKIPPED — model not found: {onnx_path}")
        return False

    out_path = engine_path_for(onnx_path)

    if os.path.exists(out_path) and not force:
        log.info(f"[{name}] Already built: {os.path.basename(out_path)} (use --force to rebuild)")
        return True

    log.info(f"[{name}] Building engine from {os.path.basename(onnx_path)} ...")
    log.info(f"  Output: {out_path}")
    log.info(f"  FP16  : {fp16}  |  workspace: {workspace_mb} MB")
    log.info(f"  This can take 5–15 minutes. Please wait...")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={out_path}",
        f"--workspace={workspace_mb}",
        "--verbose",
    ]
    if fp16:
        cmd.append("--fp16")
    cmd.extend(extra_flags)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.time() - t0

        # Always print last 30 lines of trtexec output for diagnostics
        lines = proc.stdout.strip().splitlines()
        tail = lines[-30:] if len(lines) > 30 else lines
        for line in tail:
            log.debug("  trtexec | " + line)

        if proc.returncode != 0:
            log.error(f"[{name}] FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
            log.error("  Last trtexec output:")
            for line in tail[-10:]:
                log.error("    " + line)
            return False

        if not os.path.exists(out_path):
            log.error(f"[{name}] trtexec exited 0 but engine file was not created")
            return False

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        log.info(f"[{name}] ✓ Done in {elapsed:.0f}s — engine: {size_mb:.1f} MB")
        return True

    except Exception as exc:
        log.error(f"[{name}] Exception: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nort ONNX models to TensorRT engines via trtexec"
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Use FP32 precision (default: FP16). FP16 is ~2x faster on Jetson."
    )
    parser.add_argument(
        "--workspace", type=int, default=4096,
        help="TensorRT workspace size in MiB (default: 4096 = 4 GB)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild engines even if .engine files already exist"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full trtexec output (very noisy)"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    fp16 = not args.fp32

    # ── Sanity checks ─────────────────────────────────────────────────────────
    import platform
    if not (platform.machine() == "aarch64" and platform.system() == "Linux"):
        log.warning("=" * 60)
        log.warning("WARNING: Not running on Jetson (aarch64 Linux)!")
        log.warning("Engines built here are NOT portable to the Jetson.")
        log.warning("Re-run this script on the actual Jetson hardware.")
        log.warning("=" * 60)
        ans = input("Continue anyway? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            sys.exit(0)

    try:
        trtexec = _find_trtexec()
        log.info(f"trtexec found: {trtexec}")
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ── Build all engines ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Nort TensorRT Engine Builder")
    log.info(f"Precision: {'FP16' if fp16 else 'FP32'}")
    log.info(f"Workspace: {args.workspace} MiB")
    log.info("=" * 60)

    results = {}
    for name, onnx_path, extra in MODELS:
        ok = build_engine(
            trtexec=trtexec,
            name=name,
            onnx_path=onnx_path,
            extra_flags=extra,
            fp16=fp16,
            workspace_mb=args.workspace,
            force=args.force,
        )
        results[name] = ok
        if ok:
            log.info(f"  Engine saved: {engine_path_for(onnx_path)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Summary:")
    all_ok = True
    for name, ok in results.items():
        icon = "✓ OK  " if ok else "✗ FAIL"
        log.info(f"  [{icon}]  {name}")
        if not ok:
            all_ok = False
    log.info("=" * 60)

    if all_ok:
        log.info("")
        log.info("All engines built. Run the pipeline with:")
        log.info("  python3 run.py --headless")
        log.info("")
        log.info("run.py will automatically prefer .engine files over .onnx.")
    else:
        log.warning("Some engines failed to build. Check the output above.")
        log.warning("Missing engines will fall back to ONNX (CPU execution).")
        sys.exit(1)


if __name__ == "__main__":
    main()
