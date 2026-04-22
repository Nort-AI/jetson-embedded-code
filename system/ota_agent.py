"""
ota_agent.py — Remote command handler for the Jetson device.

Handles commands received via the heartbeat response from the Nort-API.
Supported commands:
  - RESTART:          Graceful process exit (systemd restarts)
  - CAPTURE_SNAPSHOT: Request camera snapshot
  - UPLOAD_LOGS:      Bundle + upload logs to GCS
  - GET_TELEMETRY:    Read tegrastats and report
  - OTA_UPGRADE:      Download → verify → backup → extract → restart
"""
import os
import sys
import time
import hashlib
import requests
import json
import zipfile
import shutil
import subprocess
import threading
import tempfile
from datetime import datetime
import logging
from system import config

from system.logger_setup import setup_logger
logger = setup_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
OTA_MAX_BUNDLE_SIZE = 500 * 1024 * 1024   # 500 MB hard limit
OTA_DOWNLOAD_TIMEOUT = 600                 # 10 min download timeout
OTA_CHUNK_SIZE = 8192                      # 8 KB download chunks

# Install directory — where the NORT code lives on the Jetson
INSTALL_DIR = getattr(config, "OTA_INSTALL_DIR", os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR  = INSTALL_DIR + ".bak"

# Files/dirs to preserve across OTA updates (never overwritten)
PRESERVE_LIST = [
    "device.json",
    "zones_per_camera.json",
    "sync_queue.db",
    "spatial_log.db",
    "tracking_log.csv",
    "pending_uploads",
    "tracking.log",
    "gcp-key.json",
]

# List of processors for interaction (global for this module, set by main)
_processors = []

def set_processors(processors):
    global _processors
    _processors = processors


# ── Main Command Dispatcher ──────────────────────────────────────────────────

def handle_remote_command(command_data):
    """
    Executes a command received from the Nort-API heartbeat response.
    Expected format: {"command_id": "...", "command_type": "...", "payload": {...}}
    """
    if not command_data:
        return

    command_id = command_data.get("command_id")
    cmd_type = command_data.get("command_type")
    payload = command_data.get("payload", {})

    # payload might arrive as a JSON string from JSONB column
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            payload = {}

    logger.info(f"Processing remote command: {cmd_type} (ID: {command_id})")

    status = "COMPLETED"
    error_message = None

    try:
        if cmd_type == "RESTART":
            logger.info("Restart command received. Graceful exit for systemd restart...")
            _report_result(command_id, "COMPLETED", None)
            # Delay slightly to allow result report to finish
            threading.Timer(2.0, lambda: os._exit(0)).start()
            return  # Don't report twice

        elif cmd_type == "CAPTURE_SNAPSHOT":
            camera_id = payload.get("camera_id")
            logger.info(f"Snapshot requested for {camera_id}")
            found = False
            for p in _processors:
                if p.camera_id == camera_id or not camera_id:
                    p.request_snapshot = True
                    found = True
            if not found:
                status = "FAILED"
                error_message = f"Camera {camera_id} not found or active"

        elif cmd_type == "UPLOAD_LOGS":
            logger.info("Uploading log bundle...")
            bundle_path = _bundle_logs()
            _upload_to_support_bucket(bundle_path, f"logs/{config.DEVICE_ID}_{int(time.time())}.zip")
            if os.path.exists(bundle_path):
                os.remove(bundle_path)

        elif cmd_type == "GET_TELEMETRY":
            logger.info("Fetching hardware telemetry...")
            telemetry = _get_tegrastats_data()
            _report_telemetry(command_id, telemetry)
            return  # telemetry reports its own result

        elif cmd_type == "OTA_UPGRADE":
            logger.info("=== OTA UPGRADE STARTED ===")
            _handle_ota_upgrade(command_id, payload)
            return  # OTA reports its own result

        elif cmd_type == "FORCE_SYNC":
            logger.info("Forcing Cloud Sync queue process...")
            from data.sync_manager import SyncManager
            SyncManager().process_queue()
            _report_result(command_id, "COMPLETED", None)
            return

        elif cmd_type == "RESTART_PIPELINE":
            logger.info("Restarting Pipeline requested...")
            _report_result(command_id, "COMPLETED", None)
            threading.Timer(2.0, lambda: os._exit(0)).start()
            return

        else:
            logger.warning(f"Unknown command type: {cmd_type}")
            status = "FAILED"
            error_message = f"Unknown command type: {cmd_type}"

    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        status = "FAILED"
        error_message = str(e)

    # Report result back to API
    _report_result(command_id, status, error_message)


# ── OTA Upgrade Handler ──────────────────────────────────────────────────────

def _handle_ota_upgrade(command_id, payload):
    """
    Full OTA upgrade pipeline:
      1. Validate payload (download_url, sha256, version)
      2. Download bundle ZIP to temp directory
      3. Verify SHA-256 checksum
      4. Create backup of current installation
      5. Extract new code (preserving device-specific files)
      6. Report success
      7. Restart process (systemd will restart it)

    On any failure → rollback from backup → report failure.
    """
    download_url = payload.get("download_url")
    expected_sha256 = payload.get("sha256")
    target_version = payload.get("version")

    # ── Step 0: Validate ──────────────────────────────────────────────────────
    if not download_url or not expected_sha256:
        _report_result(command_id, "FAILED", "Missing download_url or sha256 in payload")
        return

    logger.info(f"OTA target version: {target_version}")
    logger.info(f"Download URL: {download_url[:80]}...")
    logger.info(f"Expected SHA-256: {expected_sha256}")

    tmp_dir = tempfile.mkdtemp(prefix="nort_ota_")
    zip_path = os.path.join(tmp_dir, "ota_bundle.zip")

    try:
        # ── Step 1: Download ──────────────────────────────────────────────────
        logger.info("Step 1/5: Downloading OTA bundle...")
        _download_file(download_url, zip_path)
        file_size = os.path.getsize(zip_path)
        logger.info(f"  Downloaded {file_size / 1024 / 1024:.1f} MB")

        if file_size > OTA_MAX_BUNDLE_SIZE:
            raise OTAError(f"Bundle too large: {file_size} bytes (limit {OTA_MAX_BUNDLE_SIZE})")

        # ── Step 2: Verify SHA-256 ────────────────────────────────────────────
        logger.info("Step 2/5: Verifying SHA-256 checksum...")
        actual_sha256 = _sha256_file(zip_path)
        if actual_sha256 != expected_sha256.lower():
            raise OTAError(
                f"Checksum mismatch!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual_sha256}"
            )
        logger.info("  ✓ Checksum verified")

        # ── Step 3: Validate ZIP structure ────────────────────────────────────
        logger.info("Step 3/5: Validating ZIP structure...")
        if not zipfile.is_zipfile(zip_path):
            raise OTAError("Downloaded file is not a valid ZIP archive")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            # Basic safety: reject if it contains absolute paths or path traversal
            for name in names:
                if name.startswith('/') or '..' in name:
                    raise OTAError(f"Unsafe path in ZIP: {name}")
            logger.info(f"  ZIP contains {len(names)} files")

            # Check for main.py as a sanity check that this is a valid NORT bundle
            has_main = any(n.endswith('main.py') for n in names)
            if not has_main:
                raise OTAError("Bundle does not contain main.py — not a valid NORT release")

        # ── Step 4: Backup current installation ───────────────────────────────
        logger.info("Step 4/5: Creating backup...")
        if os.path.exists(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(INSTALL_DIR, BACKUP_DIR, symlinks=True)
        logger.info(f"  ✓ Backup created at {BACKUP_DIR}")

        # ── Step 5: Extract new code ──────────────────────────────────────────
        logger.info("Step 5/5: Extracting new code...")
        extract_dir = os.path.join(tmp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Find the root of the extracted code
        # Handle both flat ZIPs and ZIPs with a single root directory
        extracted_items = os.listdir(extract_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_items[0])):
            source_dir = os.path.join(extract_dir, extracted_items[0])
        else:
            source_dir = extract_dir

        # Save preserved files to a temp location
        preserved = {}
        for item in PRESERVE_LIST:
            src = os.path.join(INSTALL_DIR, item)
            if os.path.exists(src):
                preserve_dest = os.path.join(tmp_dir, "preserved", item)
                os.makedirs(os.path.dirname(preserve_dest), exist_ok=True)
                if os.path.isdir(src):
                    shutil.copytree(src, preserve_dest)
                else:
                    shutil.copy2(src, preserve_dest)
                preserved[item] = preserve_dest

        # Clear the install dir (except .git if present)
        for item in os.listdir(INSTALL_DIR):
            item_path = os.path.join(INSTALL_DIR, item)
            if item in ('.git', '__pycache__'):
                continue
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        # Copy new code into install dir
        for item in os.listdir(source_dir):
            src = os.path.join(source_dir, item)
            dst = os.path.join(INSTALL_DIR, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # Restore preserved files
        for item, preserve_path in preserved.items():
            dst = os.path.join(INSTALL_DIR, item)
            # Remove the new version's copy if it exists
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            if os.path.isdir(preserve_path):
                shutil.copytree(preserve_path, dst)
            else:
                shutil.copy2(preserve_path, dst)

        logger.info("  ✓ New code extracted and preserved files restored")

        # ── SUCCESS ───────────────────────────────────────────────────────────
        logger.info(f"=== OTA UPGRADE COMPLETE: v{target_version} ===")
        _report_result(command_id, "COMPLETED", None)

        # Clean up temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Restart process — systemd will restart it with the new code
        logger.info("Restarting process in 3 seconds...")
        threading.Timer(3.0, lambda: os._exit(0)).start()

    except OTAError as e:
        logger.error(f"OTA failed: {e}")
        _rollback(command_id, str(e), tmp_dir)

    except Exception as e:
        logger.error(f"OTA unexpected error: {e}", exc_info=True)
        _rollback(command_id, f"Unexpected error: {e}", tmp_dir)


def _rollback(command_id, error_message, tmp_dir):
    """Roll back to the backup if OTA extraction failed."""
    logger.warning("=== OTA ROLLBACK STARTED ===")
    try:
        if os.path.exists(BACKUP_DIR):
            # Clear whatever partial state is in install dir
            for item in os.listdir(INSTALL_DIR):
                item_path = os.path.join(INSTALL_DIR, item)
                if item in ('.git', '__pycache__'):
                    continue
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

            # Restore from backup
            for item in os.listdir(BACKUP_DIR):
                src = os.path.join(BACKUP_DIR, item)
                dst = os.path.join(INSTALL_DIR, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            logger.info("  ✓ Rollback complete — reverted to previous version")
        else:
            logger.error("  ✗ No backup directory found — cannot rollback!")
            error_message += " | CRITICAL: No backup available for rollback"

    except Exception as rb_err:
        logger.error(f"  ✗ Rollback itself failed: {rb_err}")
        error_message += f" | Rollback error: {rb_err}"

    # Clean up temp
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _report_result(command_id, "FAILED", error_message)


# ── Helper Functions ──────────────────────────────────────────────────────────

class OTAError(Exception):
    """Custom exception for OTA-specific failures (triggers rollback)."""
    pass


def _download_file(url, dest_path):
    """Download a file from URL with streaming and progress logging."""
    resp = requests.get(url, stream=True, timeout=OTA_DOWNLOAD_TIMEOUT, verify=True)
    resp.raise_for_status()

    # Check Content-Length if available
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > OTA_MAX_BUNDLE_SIZE:
        raise OTAError(f"Server reports file size {content_length} exceeds limit")

    downloaded = 0
    last_log = 0
    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=OTA_CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                # Log progress every 10 MB
                if downloaded - last_log > 10 * 1024 * 1024:
                    logger.info(f"  ... downloaded {downloaded / 1024 / 1024:.0f} MB")
                    last_log = downloaded

                # Safety: abort if we exceed max size during download
                if downloaded > OTA_MAX_BUNDLE_SIZE:
                    raise OTAError(f"Download exceeds {OTA_MAX_BUNDLE_SIZE} bytes limit")


def _sha256_file(file_path):
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            sha.update(block)
    return sha.hexdigest()


def _report_result(command_id, status, error_message):
    """Report command execution result back to the Nort-API."""
    try:
        requests.post(
            f"{config.API_URL}/api/v1/devices/command/result",
            json={
                "command_id": command_id,
                "status": status,
                "error_message": error_message
            },
            headers={"Authorization": f"Bearer {config.DEVICE_KEY}"},
            timeout=5,
            verify=True,
        )
        logger.info(f"Reported command result: {status}")
    except Exception as e:
        logger.error(f"Failed to report command result: {e}")


def _bundle_logs():
    """Zips the local logs for support upload."""
    zip_name = "/tmp/nort_logs_bundle.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add tracking.log
        log_file = os.path.join(INSTALL_DIR, "tracking.log")
        if os.path.exists(log_file):
            zipf.write(log_file, "tracking.log")

        # Add any .log files in the install dir
        for f in os.listdir(INSTALL_DIR):
            if f.endswith('.log') and f != "tracking.log":
                zipf.write(os.path.join(INSTALL_DIR, f), f)

    return zip_name


def _upload_to_support_bucket(local_path, destination_path):
    """Upload file to GCS support bucket."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket_name = os.environ.get("GCS_BUCKET", "nort-support-files")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_filename(local_path)
        logger.info(f"File uploaded to gs://{bucket_name}/{destination_path}")
    except Exception as e:
        logger.error(f"Support upload failed: {e}")


def _get_tegrastats_data():
    """Reads tegrastats if on Jetson hardware."""
    try:
        output = subprocess.check_output(
            ["tegrastats", "--interval", "10", "--count", "1"],
            timeout=5
        ).decode()
        return output.strip()
    except Exception:
        return "tegrastats not available"


def _report_telemetry(command_id, telemetry):
    """Sends telemetry data as a COMPLETED result."""
    try:
        requests.post(
            f"{config.API_URL}/api/v1/devices/command/result",
            json={
                "command_id": command_id,
                "status": "COMPLETED",
                "error_message": None,
                "telemetry": telemetry
            },
            headers={"Authorization": f"Bearer {config.DEVICE_KEY}"},
            timeout=5,
            verify=True,
        )
    except Exception as e:
        logger.error(f"Failed to report telemetry: {e}")
