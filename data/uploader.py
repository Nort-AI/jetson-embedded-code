# uploader.py
# Rotates local tracking_log.csv into pending_uploads/ and queues it for
# background upload via SyncManager. The actual GCS upload is handled by
# the SyncManager instance owned by main.py — this module is intentionally
# stateless so it can be imported without triggering GCS/DB initialisation.
import os
from datetime import datetime
from system.logger_setup import setup_logger
import zipfile
from system import config  # loads device.json automatically
from data.sync_manager import SyncManager

logger = setup_logger(__name__)

def queue_for_upload(source_files=None):
    """
    Rotates local logs and queues them for the SyncManager.
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

            # Instantiate SyncManager scoped to this run to prevent circular GCS issues on raw imports
            sm = SyncManager()
            
            # Move file to pending_uploads with a unique name
            pending_filename = f"{timestamp}_{os.path.basename(local_file)}"
            pending_path = os.path.join(sm.pending_dir, pending_filename)
            
            os.rename(local_file, pending_path)
            
            # Queue it in the database
            sm.queue_file(pending_path, destination)
            logger.debug(f"Queued {local_file} for background sync.")
            
        except Exception as e:
            logger.error(f"Failed to queue '{local_file}': {e}")

if __name__ == "__main__":
    queue_for_upload()
