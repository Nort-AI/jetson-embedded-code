# nort-jetson-embedded

NORT V1 edge processing system. Pure Python multi-threaded pipeline that runs YOLOX person detection, ByteTrack tracking, attribute classification, and cross-camera Re-ID on RTSP camera streams — then uploads tracking data to Google Cloud Storage.

> **Note:** `nort-jetson-deepstream` (V2) replaces this system for deployments requiring >4 cameras or the highest throughput. This repo remains active for single-store deployments and hardware without NVIDIA DeepStream SDK.

## What It Does

- Detects and tracks people across up to 6 simultaneous RTSP camera feeds
- Classifies age group and gender per tracked person
- Detects store entrance/exit events and zone interactions
- Assigns persistent cross-camera identity via OSNet Re-ID
- Logs all tracking data as CSV → uploads to GCS → triggers the Cloud Functions pipeline
- Sends heartbeats to Nort API for fleet monitoring
- Handles OTA updates and remote commands from the cloud
- Runs a local VLM microservice (Google Gemini) for scene-level natural-language descriptions
- Serves a local admin web panel on port 8080

## Architecture

```
cameras.json (RTSP URIs)
    ↓
run.py  →  1 CameraProcessor thread per camera
           │   ├── YOLOX ONNX (TensorRT auto-cached on Jetson)
           │   ├── ByteTrack (IoU-based, per-camera instance)
           │   ├── Attribute classification (ONNX ResNet50)
           │   └── Zone + entrance line detection
           │
           ├── Shared ReIDManager  (OSNet, cross-camera gallery)
           └── Shared DataHandler  (CSV writer thread)
                    ↓
             SyncManager thread  →  GCS upload (SQLite queue)
             Heartbeat thread    →  Nort API
             OTA agent thread    →  Remote commands
             Local admin thread  →  Flask web panel (port 8080)
             VLM microservice    →  Gemini scene analysis (port 5001)
```

`master_processing.py` is the integrated single-file variant used for direct device deployments where all logic runs in one process.

## Project Structure

```
Jetson-Embedded-code-3/
├── run.py                        # Main entry point (multi-threaded)
├── master_processing.py          # All-in-one deployment variant
├── vlm_microservice.py           # Gemini-based scene description service (port 5001)
├── reid_gallery_window.py        # Re-ID gallery inspection utility
├── export_tensorrt.py            # TensorRT engine export helper
├── requirements.txt
├── cameras.json                  # Camera source definitions
├── device.json                   # Device identity + cloud config (never commit)
├── device.json.example           # Template for device.json
├── zones_per_camera.json         # Zone polygons + entrance lines
├── env.yml                       # Conda environment definition
├── core/
│   ├── camera_processor.py       # Per-camera inference + analytics loop
│   ├── reid_manager.py           # Cross-camera Re-ID (OSNet ONNX)
│   ├── yolox_detector.py         # YOLOX ONNX inference wrapper
│   ├── utils_body.py             # Attribute classification
│   ├── polygon_zone.py           # NumPy zone detection
│   ├── homography_manager.py     # Camera → world coordinate transform
│   └── render.py                 # Visualization helpers
├── data/
│   ├── data_handler.py           # Async CSV writer
│   ├── sync_manager.py           # GCS upload queue (SQLite)
│   ├── uploader.py               # CSV rotation + sync queue
│   └── spatial_logger.py         # Position log for heatmap queries
├── system/
│   ├── config.py                 # All configuration (loaded from device.json)
│   ├── ota_agent.py              # Remote command handler
│   └── logger_setup.py           # Logging setup
├── admin/
│   ├── local_admin.py            # Flask admin web panel
│   └── locales/                  # Admin UI translations (pt/en/es)
├── assets/
│   ├── homographies.json         # Per-camera homography matrices
│   ├── label.json                # YOLO class labels
│   └── attribute.json            # Attribute classifier labels
├── models/                       # ONNX model files (not committed)
├── scripts/                      # Utility scripts (simulator, setup)
└── tracker_configs/
    ├── custom_bytetrack.yaml
    └── fast_botsort.yaml
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended — falls back to CPU automatically)
- Google Cloud credentials (`gcloud auth application-default login`)

```bash
# Conda (recommended on Jetson)
conda env create -f env.yml
conda activate nort

# Or pip
pip install -r requirements.txt
```

ONNX Runtime will auto-detect and use TensorRT on Jetson, CUDA on other NVIDIA GPUs, or CPU as fallback. TensorRT engines are cached on first run (`export_tensorrt.py` can pre-bake them).

## Setup

### 1. Configure the device

Create `device.json` in the project root:

```json
{
  "client_id": "your_client_id",
  "store_id": "your_store_id",
  "device_id": "jetson-store-01",
  "api_url": "https://your-nort-api.run.app",
  "device_key": "your-shared-secret",
  "gcs_bucket": "nort-data-your-project",
  "admin_pin": "your-pin",
  "admin_port": 8080,
  "enable_reid": true
}
```

See `device.json.example` for all available fields.

### 2. Configure cameras

Edit `cameras.json`:

```json
{
  "entrance_cam": {
    "name": "Entrance",
    "source": "rtsp://user:pass@192.168.1.100:554/stream1",
    "type": "entrance_camera",
    "enabled": true
  }
}
```

For simulation, use local video files: `"source": "videos/sample.mp4"`

### 3. Configure zones

Use the admin panel at `http://<device-ip>:8080` after first run, or edit `zones_per_camera.json` directly.

### 4. Run

```bash
# Multi-threaded (standard)
python run.py

# All-in-one variant (production devices)
python master_processing.py
```

### 5. VLM Microservice (optional)

The VLM microservice exposes a local HTTP API for Gemini-based scene descriptions. It runs independently and is polled by the main pipeline:

```bash
python vlm_microservice.py   # Starts on port 5001
```

Set `GOOGLE_API_KEY` in your environment before starting.

### 6. Install as systemd service (production)

```bash
sudo tee /etc/systemd/system/nort-tracking.service > /dev/null << 'EOF'
[Unit]
Description=NORT Analytics Tracking Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=nort
WorkingDirectory=/home/nort/nort-jetson-embedded
ExecStart=/home/nort/miniconda3/envs/nort/bin/python run.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nort-tracking

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nort-tracking
sudo systemctl start nort-tracking
sudo journalctl -u nort-tracking -f
```

## Local Admin Panel

`http://<device-ip>:8080` — live occupancy, per-camera FPS, system metrics, zone editor, force sync, restart.

## Running as a Simulator (Windows / no camera)

```bash
# Authenticate with GCP
gcloud auth application-default login

python run.py   # uses video files defined in cameras.json
```

Or use the lightweight heartbeat + data simulator (no vision pipeline):
```bash
python scripts/jetson_simulator.py
```

## Key Configuration Parameters

All config from `device.json`. Runtime tuning in `system/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `YOLO_CONF_THRESHOLD` | 0.15 | Detection confidence floor |
| `REID_SIMILARITY_THRESHOLD` | 0.52 | Cosine similarity for Re-ID |
| `REID_EXPIRY_SECONDS` | 300 | Gallery entry TTL |
| `LOG_ROTATION_INTERVAL` | 300s | CSV rotation + upload frequency |
| `HEARTBEAT_INTERVAL` | 300s | API heartbeat frequency |

## Collaboration

| Branch | Purpose |
|---|---|
| `main` | Stable — deployed to devices via OTA |
| `dev` | Integration |
| `feat/<name>` | Feature work |
| `fix/<name>` | Bug fixes |

**PR checklist:**
- [ ] Tested with at least one real or simulated camera stream
- [ ] `run.py` starts without errors
- [ ] No `device.json`, `tracking.log`, `*.csv`, `*.db` committed
- [ ] `APP_VERSION` bumped in `system/config.py` if releasing an OTA update
- [ ] VLM microservice tested independently if `vlm_microservice.py` was modified
