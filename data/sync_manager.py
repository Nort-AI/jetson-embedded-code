import os
import time
import sqlite3
import requests
from datetime import datetime
from google.cloud import storage
from system import config
from system.logger_setup import setup_logger
logger = setup_logger(__name__)

class SyncManager:
    """
    Manages local data queuing and synchronization to GCS.
    Ensures zero data loss during offline periods.
    """
    def __init__(self, db_path="sync_queue.db", pending_dir="pending_uploads", last_upload_ref=None):
        self.db_path = db_path
        self.pending_dir = pending_dir
        self.last_upload_ref = last_upload_ref
        self.client_id = config.CLIENT_ID
        self.store_id = config.STORE_ID
        self.bucket_name = config.GCS_BUCKET
        
        if not os.path.exists(self.pending_dir):
            os.makedirs(self.pending_dir)
            
        self._init_db()
        self.storage_client = None
        # GCS-timeout-fix: storage.Client() calls google.auth.default() which, as a
        # last-resort credential probe, sends an HTTP request to the GCE metadata
        # server (169.254.169.254).  On non-GCE hardware (Jetson, dev laptops) that
        # address is unreachable and the call blocks indefinitely — freezing startup.
        # Fix: run the init in a daemon thread and join with a 5-second timeout.
        # The thread is daemon so it can't prevent process exit even if it's still
        # blocked when the OS later kills it.
        # KeyboardInterrupt during join() propagates normally (thread.join IS
        # interruptible), so Ctrl+C still shuts the process down cleanly.
        import threading as _threading
        _gcs_exc   = [None]
        _gcs_ready = [None]

        def _init_gcs():
            try:
                _gcs_ready[0] = storage.Client()
            except Exception as _e:
                _gcs_exc[0] = _e

        _t = _threading.Thread(target=_init_gcs, daemon=True, name="gcs-init")
        _t.start()
        _t.join(timeout=5.0)

        if _gcs_ready[0] is not None:
            self.storage_client = _gcs_ready[0]
        elif _gcs_exc[0] is not None:
            logger.warning(f"GCS Client init failed (safe if simulator/local): {_gcs_exc[0]}")
        else:
            # Thread still running → metadata server is unreachable; give up and
            # continue without cloud sync.  The daemon thread will be abandoned.
            logger.warning(
                "[SyncManager] GCS Client init timed out after 5 s "
                "(no GCE/ADC credentials found on this machine) — "
                "running without cloud sync.  Set GOOGLE_APPLICATION_CREDENTIALS "
                "to enable GCS uploads."
            )

    # H8-fix: stop retrying permanently broken files after this many attempts
    MAX_UPLOAD_ATTEMPTS = 10

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upload_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    destination_blob TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    attempts INTEGER DEFAULT 0,
                    last_attempt TIMESTAMP
                )
            """)
            # H8-fix: dead-letter table for files that exceeded MAX_UPLOAD_ATTEMPTS
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upload_dead_letter (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    destination_blob TEXT,
                    created_at TIMESTAMP,
                    attempts INTEGER,
                    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def queue_file(self, local_path, destination_blob):
        """Add a file to the local sync queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO upload_queue (file_path, destination_blob) VALUES (?, ?)",
                    (local_path, destination_blob)
                )
            logger.debug(f"Queued file for sync: {local_path}")
        except sqlite3.IntegrityError:
            pass # Already queued

    def process_queue(self):
        """Attempt to upload pending files in the queue."""
        if not self.storage_client:
            return

        # Enforce disk quota before processing (auto-evict oldest if over limit)
        self.check_disk_limit()

        with sqlite3.connect(self.db_path) as conn:
            # Get pending files, ordered by date
            rows = conn.execute(
                "SELECT id, file_path, destination_blob, attempts FROM upload_queue ORDER BY created_at ASC"
            ).fetchall()
            
            if not rows:
                return

            if len(rows) > 0:
                logger.debug(f"SyncManager: {len(rows)} files pending upload.")
            
            for row_id, file_path, dest_blob, attempts in rows:
                if not os.path.exists(file_path):
                    logger.warning(f"File missing from disk: {file_path}. Removing from queue.")
                    conn.execute("DELETE FROM upload_queue WHERE id = ?", (row_id,))
                    continue

                success = self._upload_file(file_path, dest_blob)

                if success:
                    os.remove(file_path)
                    conn.execute("DELETE FROM upload_queue WHERE id = ?", (row_id,))
                    logger.debug(f"Successfully synced: {dest_blob}")
                    # Keep processing next
                else:
                    new_attempts = attempts + 1
                    if new_attempts >= self.MAX_UPLOAD_ATTEMPTS:
                        # H8-fix: move to dead-letter after MAX_UPLOAD_ATTEMPTS failures
                        # so the queue doesn't grow unbounded and burn CPU forever
                        conn.execute(
                            "INSERT INTO upload_dead_letter "
                            "(file_path, destination_blob, created_at, attempts) VALUES (?, ?, ?, ?)",
                            (file_path, dest_blob,
                             datetime.now().isoformat(), new_attempts)
                        )
                        conn.execute("DELETE FROM upload_queue WHERE id = ?", (row_id,))
                        logger.error(
                            f"Upload permanently failed after {new_attempts} attempts: "
                            f"{file_path} — moved to dead-letter queue."
                        )
                    else:
                        conn.execute(
                            "UPDATE upload_queue SET attempts = attempts + 1, last_attempt = ? WHERE id = ?",
                            (datetime.now().isoformat(), row_id)
                        )
                        logger.warning(
                            f"Sync failed for {file_path} "
                            f"(attempt {new_attempts}/{self.MAX_UPLOAD_ATTEMPTS}). Will retry later."
                        )
                    break
            
            conn.commit()

    def _upload_file(self, local_path, destination_blob):
        """Direct upload to GCS with MD5 checksum verification."""
        import base64
        import hashlib
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(destination_blob)
            
            # GCS Data Integrity Validation (MD5 hash check)
            with open(local_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).digest()
            blob.md5_hash = base64.b64encode(file_hash).decode('utf-8')
            
            blob.upload_from_filename(local_path)
            if self.last_upload_ref is not None:
                self.last_upload_ref["ts"] = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Sync error (data integrity check may have failed): {e}")
            return False

    def check_disk_limit(self, limit_gb=1.0):
        """Guard against filling up the disk with unsent data."""
        # Calculate size of pending_dir
        total_size = 0
        for f in os.listdir(self.pending_dir):
            fp = os.path.join(self.pending_dir, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)

        size_gb = total_size / (1024**3)
        if size_gb > limit_gb:
            logger.critical(f"Disk storage limit for pending uploads exceeded ({size_gb:.2f} GB)!")
            self._enforce_disk_quota()
            return True
        return False

    def _enforce_disk_quota(self):
        """Delete oldest pending files if over quota."""
        try:
            files = sorted(
                [os.path.join(self.pending_dir, f) for f in os.listdir(self.pending_dir) if f.endswith('.csv')],
                key=os.path.getmtime
            )
            total_bytes = sum(os.path.getsize(f) for f in files)
            quota_bytes = 1 * 1024 * 1024 * 1024  # 1 GB

            while total_bytes > quota_bytes and files:
                oldest = files.pop(0)
                size = os.path.getsize(oldest)
                os.remove(oldest)
                total_bytes -= size
                logger.warning(f"Disk quota exceeded: deleted oldest file {os.path.basename(oldest)} ({size/1024/1024:.1f} MB)")
        except Exception as e:
            logger.error(f"Disk quota enforcement failed: {e}")
