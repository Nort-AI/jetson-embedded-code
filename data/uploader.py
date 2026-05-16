# uploader.py
# Rotates local tracking_log.csv into pending_uploads/ and queues it for
# background upload via the shared SyncManager owned by run.py.
#
# Design: this module is intentionally stateless — it never creates a
# SyncManager itself.  The caller (run.py's log-rotation thread) passes the
# shared instance so uploads drain through the same process_queue() loop that
# run.py already drives every 5 minutes.  This avoids the 5-second GCS init
# timeout that previously blocked the rotation thread on every call.
import os
from datetime import datetime
from system.logger_setup import setup_logger
from system import config

logger = setup_logger(__name__)


def queue_for_upload(sync_manager=None, source_files=None):
    """
    Rotate local log files and queue them for background GCS upload.

    Args:
        sync_manager: The shared SyncManager instance from run.py.  If None,
                      files are still rotated into pending_uploads/ but will
                      only be uploaded on the next process run that picks them
                      up from the queue DB.
        source_files: list of file paths to rotate (default: [CSV_FILENAME]).
    """
    if source_files is None:
        source_files = [config.CSV_FILENAME]

    for local_file in source_files:
        if not os.path.exists(local_file):
            continue

        try:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Destination path in GCS
            destination = (
                f"landing_zone/client_id={config.CLIENT_ID}/store_id={config.STORE_ID}/"
                f"year={now.strftime('%Y')}/month={now.strftime('%m')}/day={now.strftime('%d')}/"
                f"{timestamp}_{os.path.basename(local_file)}"
            )

            if sync_manager is None:
                # No shared manager available — still rotate the file so the
                # CSV writer can start a fresh file; upload will happen on the
                # next process run that has a live SyncManager.
                pending_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "pending_uploads",
                )
                os.makedirs(pending_dir, exist_ok=True)
                pending_path = os.path.join(pending_dir, f"{timestamp}_{os.path.basename(local_file)}")
                os.rename(local_file, pending_path)
                logger.debug("Rotated %s to %s (no SyncManager — will upload later)",
                             local_file, pending_path)
                continue

            # Move file into the shared manager's pending_uploads directory
            pending_path = os.path.join(sync_manager.pending_dir,
                                        f"{timestamp}_{os.path.basename(local_file)}")
            os.rename(local_file, pending_path)

            # Queue it in the shared DB — process_queue() in run.py will upload it
            sync_manager.queue_file(pending_path, destination)
            logger.debug("Queued %s for background sync.", local_file)

        except Exception as e:
            logger.error("Failed to queue '%s': %s", local_file, e)


if __name__ == "__main__":
    queue_for_upload()
