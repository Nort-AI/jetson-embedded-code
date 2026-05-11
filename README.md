# nort-jetson-embedded

Edge AI pipeline for Nort retail analytics. Runs YOLOX person detection, ByteTrack multi-object tracking, attribute classification (age/gender), cross-camera Re-ID, zone analytics, and entrance/exit counting on RTSP streams — then uploads structured CSV data to Google Cloud Storage.

> **Note:** `nort-jetson-deepstream` (V2) supersedes this system for deployments requiring >4 cameras or maximum throughput. This repo remains the primary choice for single-store deployments on Jetson Orin Nano/NX.

---

## What It Does

- Detects and tracks people across up to 6 simultaneous RTSP camera feeds
- Classifies age group and gender per tracked person using a ResNet50 attribute model
- Detects store entrance/exit events and zone interactions (polygon-based)
- Assigns persistent cross-camera global IDs via OSNet Re-ID
- Logs all tracking data as CSV → uploads to GCS → triggers the Cloud Functions analytics pipeline
- Sends heartbeats to Nort API for device fleet monitoring
- Handles OTA updates and remote commands (restart, snapshot, log upload) from the cloud
- Runs a local Gemini VLM microservice for scene-level natural-language descriptions
- Serves a local HTTPS admin panel for store staff (camera management, live metrics, zone editor)

---

## Architecture

```
cameras.json (RTSP URIs)
    ↓
run.py  →  1 CameraProcessor thread per camera
           │   ├── YOLOX  (TRTSession native → ORT TRT EP → ORT CUDA → CPU)
           │   ├── ByteTrack  (IoU-based multi-object tracker, per-camera)
           │   ├── Attribute model  (TRTSession → ORT, gender + age)
           │   └── Zone polygon + entrance line detection
           │
           ├── Shared ReIDManager  (OSNet AIN, TRTSession → ORT, cross-camera gallery)
           │       └── Background threads: merge duplicates, prune expired entries
           └── Shared DataHandler  (synchronous CSV writer with fsync)
                    ↓
             SyncManager thread  →  GCS upload (SQLite queue, dead-letter after 10 failures)
             Heartbeat thread    →  Nort API (300 s interval)
             OTA agent thread    →  Remote commands + validated extract + rollback
             Local admin thread  →  Flask HTTPS panel (port 8080, self-signed cert)
             Watchdog thread     →  sd_notify("WATCHDOG=1") every 10 s
             VLM microservice    →  Gemini scene analysis (port 5001, optional)
```

---

## Hardware & Software Requirements

| Component | Requirement |
|-----------|-------------|
| Board | NVIDIA Jetson Orin Nano / NX (8 GB recommended) |
| JetPack | 6.2 (CUDA 12.6, TensorRT 10.3) |
| Python | 3.10 (via `nort` conda env) |
| TRT Python | `python3-libnvinfer` system package (no pip wheel needed) |
| GCS auth | Service account key or `gcloud auth application-default login` |

> **onnxruntime-gpu is NOT available as a pip wheel on aarch64.** The pipeline uses a native `TRTSession` wrapper (`core/trt_session.py`) that calls TensorRT directly via the `python3-libnvinfer` system package. Pre-build `.engine` files with `scripts/build_engines.py` before first run.

---

## Project Structure

```
Jetson-Embedded-code-3/
├── run.py                        # Main entry point (multi-threaded, production)
├── master_processing.py          # All-in-one single-process variant
├── device.json                   # Device identity + secrets (git-ignored — never commit)
├── device.json.example           # Template; copy → device.json and fill in values
├── cameras.json                  # Camera source definitions (auto-created on first run)
├── zones_per_camera.json         # Zone polygons + entrance lines (edit via admin panel)
├── requirements.txt
├── env.yml                       # Conda environment definition
│
├── core/
│   ├── camera_processor.py       # Per-camera inference + analytics loop
│   ├── reid_manager.py           # Cross-camera Re-ID (OSNet AIN, TRT-first)
│   ├── yolox_detector.py         # YOLOX inference (TRTSession → ORT fallback)
│   ├── trt_session.py            # Native TensorRT session (no onnxruntime-gpu)
│   ├── frame_buffer.py           # Ring buffer of JPEG frames for event replay
│   ├── utils_body.py             # Attribute classification helpers
│   ├── polygon_zone.py           # NumPy polygon zone detection
│   ├── homography_manager.py     # Camera → world coordinate transform
│   └── render.py                 # Visualization helpers
│
├── data/
│   ├── data_handler.py           # Thread-safe synchronous CSV writer (fsync)
│   ├── sync_manager.py           # GCS upload queue (SQLite, dead-letter table)
│   ├── uploader.py               # CSV rotation + sync queue
│   └── spatial_logger.py         # Position log for heatmap queries
│
├── system/
│   ├── config.py                 # All configuration (loaded from device.json)
│   ├── ota_agent.py              # Remote commands: OTA, restart, snapshot, logs
│   ├── logger_setup.py           # RotatingFileHandler (10 MB × 5 backups)
│   └── nort.service              # systemd unit (MemoryMax, WatchdogSec)
│
├── admin/
│   ├── local_admin.py            # Flask HTTPS admin panel (self-signed cert)
│   ├── keys.py                   # i18n key validator
│   ├── templates/                # Jinja2 HTML templates
│   └── locales/                  # UI translations (en / pt / es)
│
├── assets/
│   ├── models/                   # ONNX + TRT engine files (git-ignored)
│   │   ├── yolox_m.onnx          # (download separately)
│   │   ├── yolox_m.engine        # (built by build_engines.py)
│   │   ├── net_last.pth          # Attribute model weights
│   │   ├── net_last.onnx         # (exported by export_attribute_onnx.py)
│   │   ├── net_last.engine       # (built by build_engines.py)
│   │   ├── osnet_ain_x1_0.onnx   # Re-ID model (exported by export_osnet_onnx.py)
│   │   └── osnet_ain_x1_0.engine # (built by build_engines.py)
│   ├── homographies.json
│   ├── label.json
│   └── attribute.json
│
├── scripts/
│   ├── build_engines.py          # Convert all .onnx → .engine via trtexec (run once)
│   ├── export_osnet_onnx.py      # Export OSNet Re-ID to ONNX
│   ├── export_attribute_onnx.py  # Export attribute model to ONNX
│   ├── smoke_test.sh             # Deployment readiness check (run after install)
│   └── nort.service              # systemd unit template (MemoryMax=6G, Watchdog=30s)
│
└── tracker_configs/
    ├── custom_bytetrack.yaml
    └── fast_botsort.yaml
```

---

## First-Time Setup

### 1. Create `device.json`

```bash
cp device.json.example device.json
```

Edit `device.json`:

```json
{
  "client_id": "acme-retail",
  "store_id": "store-01",
  "device_id": "jetson-store-01",
  "api_url": "https://your-nort-api.run.app",
  "device_key": "your-shared-secret",
  "gcs_bucket": "nort-data-your-project",
  "admin_pin": "your-pin",
  "admin_port": 8080,
  "enable_reid": true,
  "detect_every_n_frames": 1
}
```

> `admin_pin` is automatically hashed (scrypt via werkzeug) on first boot if supplied as plaintext. A default or empty PIN triggers generation of a random 6-digit PIN that is logged once to the console.

> `api_url` **must** start with `https://` — the pipeline warns at startup if it does not.

### 2. Install Python environment

```bash
# Jetson (recommended)
conda env create -f env.yml
conda activate nort

# Generic / dev machine
pip install -r requirements.txt
```

### 3. Export and build TRT engines (Jetson — run once)

```bash
# Export ONNX files from PyTorch weights
python scripts/export_osnet_onnx.py
python scripts/export_attribute_onnx.py

# Build .engine files (requires trtexec — included in JetPack TensorRT)
python scripts/build_engines.py

# Verify
ls assets/models/*.engine
```

On non-Jetson machines the pipeline falls back to ORT (CUDA → CPU). `.engine` files are device-specific and cannot be shared between GPU architectures.

### 4. Configure cameras

Edit `cameras.json` (or use the admin panel after first run):

```json
{
  "entrance_cam": {
    "name": "Entrance",
    "source": "rtsp://user:pass@192.168.1.100:554/stream1",
    "type": "entrance_camera",
    "enabled": true
  },
  "aisle_cam": {
    "name": "Aisle A",
    "source": "rtsp://user:pass@192.168.1.101:554/stream1",
    "type": "standard",
    "enabled": true
  }
}
```

For simulation / dev use local video files: `"source": "videos_1080p/CAM00.mp4"`

### 5. Configure zones

Draw zones in the admin panel after first run (`https://<device-ip>:8080`) or edit `zones_per_camera.json` directly. At minimum define one zone and an entrance line on the entrance camera.

### 6. Run

```bash
python run.py
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--headless` | Disable OpenCV GUI windows (required on servers / systemd) |
| `--no-gallery` | Disable Re-ID gallery debug window |
| `--no-heartbeat` | Disable heartbeat (dev / offline testing) |
| `--no-sync` | Disable GCS upload (dev / offline testing) |
| `--setup` | Run setup wizard only, don't start pipeline |

### 7. Verify deployment

```bash
bash scripts/smoke_test.sh
```

Checks Python version, `device.json`, camera config, model files, disk space, imports, TRT availability, and systemd service status. Exit code 0 = ready to deploy.

---

## Admin Panel

Accessible at `https://<device-ip>:8080` from any browser on the store LAN.

A self-signed TLS certificate is auto-generated on first run (`admin/admin_cert.pem`). Accept the browser security warning once — subsequent visits use the cached exception.

Features: live occupancy counter, per-camera FPS, CPU/GPU/disk/temp, sync status, camera table, zone editor, floor plan editor, force sync, restart pipeline, OTA status.

---

## Systemd Service (Production)

```bash
# Install the service
sudo cp scripts/nort.service /etc/systemd/system/nort.service
sudo systemctl daemon-reload
sudo systemctl enable nort
sudo systemctl start nort

# Monitor
sudo journalctl -u nort -f
```

The service unit includes:
- `MemoryMax=6G` / `MemoryHigh=5G` — leaves 2 GB headroom for kernel and GPU on 8 GB Orin Nano
- `WatchdogSec=30` + `NotifyAccess=main` — systemd restarts the service if the process hangs
- `OOMScoreAdjust=-500` — protects the process from the kernel OOM killer
- `ExecStartPre` — sets maximum GPU power mode and clocks via `nvpmodel` / `jetson_clocks`

---

## OTA Updates

Push a ZIP of the new codebase to the device via the Nort API. The OTA agent:
1. Downloads and SHA-256 verifies the archive
2. Backs up the current install to `INSTALL_DIR.bak`
3. Extracts the new code
4. Runs `python -c "import run"` to validate the new code imports cleanly
5. If validation passes: reports `COMPLETED` and restarts via `os._exit(0)` (systemd brings it back)
6. If validation fails: rolls back from backup automatically and reports the error

---

## Key Configuration

All runtime tuning is in `system/config.py`. The most commonly adjusted parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_CONF_THRESHOLD` | `0.15` | Detection confidence floor (lower = more detections, more FP) |
| `DETECT_EVERY_N_FRAMES` | `1` | Run YOLOX every N frames; ByteTrack Kalman-predicts between |
| `REID_SIMILARITY_THRESHOLD` | `0.52` | Cosine similarity for cross-camera ID match |
| `REID_EXPIRY_SECONDS` | `300` | Gallery entry TTL (seconds a person can leave and return) |
| `REID_MAX_EMBEDDINGS` | `45` | Max appearance embeddings stored per gallery entry |
| `REID_UPDATE_INTERVAL_FRAMES` | `30` | How often to refresh a live track's embedding |
| `TRACK_EMA_ALPHA` | `0.35` | EMA smoothing for zone assignment (reduce zone flickering) |
| `MIN_LAPLACIAN_SHARPNESS` | `10.0` | Minimum crop sharpness to extract a Re-ID embedding |

`detect_every_n_frames` can also be set per-device in `device.json` to tune throughput:
- `1` = every frame (highest accuracy)
- `2` = every other frame (~2× throughput, transparent for retail analytics)
- `3` = good for 4+ cameras on Orin Nano; ByteTrack handles occlusion gaps well

---

## Data Flow

```
CameraProcessor.run()
    → detect + track (YOLOX + ByteTrack)
    → classify attributes (ResNet50/TRT)
    → Re-ID (OSNet/TRT)
    → zone assignment (polygon test)
    → entrance/exit counting (line crossing)
    → DataHandler.write_data()  [fsync on every write]
        ↓
    tracking_log.csv (rotated every 5 min)
        ↓
    SyncManager → GCS landing_zone/client_id=.../store_id=.../YYYY/MM/DD/
        ↓
    Cloud Functions (BigQuery ingestion + dashboard refresh)
```

---

## Simulator / Dev Mode (Windows / no camera)

```bash
# Authenticate with GCP
gcloud auth application-default login

# Use local .mp4 files defined in cameras.json
python run.py

# Lightweight heartbeat + data simulator (no vision pipeline)
python scripts/jetson_simulator.py
```

---

## Contributing

| Branch | Purpose |
|--------|---------|
| `main` | Stable — deployed to devices via OTA |
| `dev` | Integration |
| `feat/<name>` | Feature work |
| `fix/<name>` | Bug fixes |

**PR checklist:**
- [ ] `bash scripts/smoke_test.sh` passes on a real or simulated device
- [ ] `python run.py --no-heartbeat --no-sync` starts without errors
- [ ] No `device.json`, `tracking_log.csv`, `*.db`, `*.engine`, `*.pem` committed
- [ ] `APP_VERSION` bumped in `system/config.py` if this targets an OTA release
- [ ] VLM microservice tested independently if `vlm_microservice.py` was touched
