#!/usr/bin/env bash
# smoke_test.sh — L6-fix: Basic deployment smoke test for Nort edge AI pipeline.
#
# Run ONCE after a fresh install or after an OTA update to verify the device
# is healthy before handing it over to the customer.
#
# Usage:
#   cd /opt/nort
#   bash scripts/smoke_test.sh
#
# Exit code 0 = all checks passed.
# Exit code 1 = one or more checks failed (details printed above).

set -euo pipefail

PASS=0
FAIL=0
INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

log_pass() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
log_fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
log_info() { echo "  [INFO] $1"; }

echo ""
echo "========================================================"
echo "  Nort Edge AI — Deployment Smoke Test"
echo "  Install dir: $INSTALL_DIR"
echo "========================================================"
echo ""

# ── 1. Python version ─────────────────────────────────────────────────────────
echo "[1] Python environment"
PY_VER=$($PYTHON --version 2>&1)
if [[ "$PY_VER" == Python\ 3* ]]; then
    log_pass "Python: $PY_VER"
else
    log_fail "Python 3 not found — got: $PY_VER"
fi

# ── 2. device.json present and parseable ─────────────────────────────────────
echo "[2] device.json"
DEVICE_JSON="$INSTALL_DIR/device.json"
if [ -f "$DEVICE_JSON" ]; then
    CLIENT_ID=$(python3 -c "import json; d=json.load(open('$DEVICE_JSON')); print(d.get('client_id','missing'))")
    if [ "$CLIENT_ID" != "missing" ] && [ "$CLIENT_ID" != "unknown" ]; then
        log_pass "device.json found, client_id=$CLIENT_ID"
    else
        log_fail "device.json exists but client_id is missing/unknown"
    fi
else
    log_fail "device.json not found — copy device.json.example and fill in values"
fi

# ── 3. cameras.json present ───────────────────────────────────────────────────
echo "[3] cameras.json"
if [ -f "$INSTALL_DIR/cameras.json" ]; then
    log_pass "cameras.json found"
else
    log_fail "cameras.json not found"
fi

# ── 4. YOLO model present ─────────────────────────────────────────────────────
echo "[4] YOLO model"
YOLO_ONNX="$INSTALL_DIR/assets/models/yolox_m.onnx"
YOLO_ENGINE="$INSTALL_DIR/assets/models/yolox_m.engine"
if [ -f "$YOLO_ENGINE" ]; then
    log_pass "yolox_m.engine found (TRT native path)"
elif [ -f "$YOLO_ONNX" ]; then
    log_pass "yolox_m.onnx found (ORT fallback path); run build_engines.py for best performance"
else
    log_fail "Neither yolox_m.engine nor yolox_m.onnx found in assets/models/"
fi

# ── 5. OSNet Re-ID model present ──────────────────────────────────────────────
echo "[5] OSNet Re-ID model"
OSNET_ENGINE="$INSTALL_DIR/assets/models/osnet_ain_x1_0.engine"
OSNET_ONNX="$INSTALL_DIR/assets/models/osnet_ain_x1_0.onnx"
if [ -f "$OSNET_ENGINE" ]; then
    log_pass "osnet_ain_x1_0.engine found"
elif [ -f "$OSNET_ONNX" ]; then
    log_pass "osnet_ain_x1_0.onnx found (run build_engines.py for GPU acceleration)"
else
    log_fail "OSNet model not found — Re-ID will be disabled"
fi

# ── 6. Python imports ─────────────────────────────────────────────────────────
echo "[6] Python imports"
cd "$INSTALL_DIR"
if $PYTHON -c "import run" 2>/dev/null; then
    log_pass "run.py imports cleanly"
else
    log_fail "run.py import failed — check dependencies and config"
fi

# ── 7. CUDA / TRT availability ────────────────────────────────────────────────
echo "[7] GPU / TRT availability"
if $PYTHON -c "import tensorrt; print('TRT', tensorrt.__version__)" 2>/dev/null; then
    TRT_VER=$($PYTHON -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null)
    log_pass "TensorRT available: $TRT_VER"
elif which trtexec >/dev/null 2>&1; then
    log_pass "trtexec binary found (TRT available)"
else
    log_fail "TensorRT not found — performance will be degraded on CPU"
fi

# ── 8. Disk space ─────────────────────────────────────────────────────────────
echo "[8] Disk space"
AVAIL_GB=$(df -BG "$INSTALL_DIR" | awk 'NR==2{gsub("G",""); print $4}')
if [ "$AVAIL_GB" -ge 5 ]; then
    log_pass "Disk: ${AVAIL_GB} GB available"
elif [ "$AVAIL_GB" -ge 2 ]; then
    log_info "Disk: ${AVAIL_GB} GB available (low — consider cleaning up old logs)"
    PASS=$((PASS+1))
else
    log_fail "Disk: only ${AVAIL_GB} GB available — system may run out of space quickly"
fi

# ── 9. Admin panel config ─────────────────────────────────────────────────────
echo "[9] Admin panel"
ADMIN_PORT=$($PYTHON -c "import json; d=json.load(open('$DEVICE_JSON')); print(d.get('admin_port',8080))" 2>/dev/null || echo 8080)
log_info "Admin panel will start on https://0.0.0.0:$ADMIN_PORT/ (accept self-signed cert warning)"
PASS=$((PASS+1))

# ── 10. systemd service file ──────────────────────────────────────────────────
echo "[10] systemd service"
if systemctl is-enabled nort >/dev/null 2>&1; then
    STATUS=$(systemctl is-active nort 2>/dev/null || echo "inactive")
    log_pass "nort.service is enabled, status: $STATUS"
elif [ -f "/etc/systemd/system/nort.service" ]; then
    log_pass "nort.service installed but not enabled — run: sudo systemctl enable nort"
else
    log_info "nort.service not installed — run system/install_service.sh to install"
    PASS=$((PASS+1))
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================================"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "  Fix the FAIL items above before deploying."
    exit 1
else
    echo "  All checks passed. Device is ready for deployment."
    exit 0
fi
