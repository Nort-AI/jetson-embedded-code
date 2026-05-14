# data_handler.py
# Fix for data_handler.py - Ensure all CSV fields are handled properly

import csv
import os
import time
from threading import Lock
from typing import List, Dict, Any

from system.logger_setup import setup_logger
from system.config import CSV_FILENAME, CSV_FIELDNAMES

logger = setup_logger(__name__)

# Flush to disk at most once per this many seconds OR once this many rows
# accumulate — whichever comes first.  Avoids per-row fsync that limits
# effective write throughput to ~2-5 FPS on Jetson eMMC.
_FLUSH_INTERVAL_SECS = 1.0
_FLUSH_ROWS_THRESHOLD = 30


class DataHandler:
    """
    Handles writing tracking data to a CSV file in a thread-safe manner.
    """
    def __init__(self):
        self.filename = CSV_FILENAME
        self.fieldnames = CSV_FIELDNAMES
        self._lock = Lock()
        self._unflushed_rows: int = 0
        self._last_flush: float = time.monotonic()
        self._initialize_file()

    def _initialize_file(self):
        """
        Creates the CSV file and writes the header if it doesn't exist.
        """
        with self._lock:
            if not os.path.exists(self.filename):
                try:
                    with open(self.filename, mode='w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                        writer.writeheader()
                    logger.info(f"CSV log file created: {self.filename}")
                except IOError as e:
                    logger.error(f"Failed to create CSV file: {e}")

    def write_data(self, data_rows: List[Dict[str, Any]]):
        """
        Writes multiple rows of tracking data to the CSV file.
        :param data_rows: A list of dictionaries, each representing a row.
        """
        with self._lock:
            try:
                # Process each row to handle None values and ensure all fields are present
                processed_rows = []
                for row in data_rows:
                    processed_row = {}
                    for field in self.fieldnames:
                        value = row.get(field)
                        # Convert None to empty string for CSV compatibility
                        # Handle boolean values properly
                        if value is None:
                            processed_row[field] = ''
                        elif isinstance(value, bool):
                            processed_row[field] = str(value).lower()  # true/false instead of True/False
                        else:
                            processed_row[field] = value
                    processed_rows.append(processed_row)

                with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writerows(processed_rows)

                    # Batch fsync: only sync to disk when enough rows have
                    # accumulated OR enough time has passed.  Per-row fsync
                    # saturates Jetson eMMC and limits tracking to ~2-5 FPS.
                    self._unflushed_rows += len(processed_rows)
                    elapsed = time.monotonic() - self._last_flush
                    if (self._unflushed_rows >= _FLUSH_ROWS_THRESHOLD
                            or elapsed >= _FLUSH_INTERVAL_SECS):
                        f.flush()
                        os.fsync(f.fileno())
                        self._unflushed_rows = 0
                        self._last_flush = time.monotonic()

                # logger.debug(f"Successfully wrote {len(data_rows)} rows to {self.filename}")
            except IOError as e:
                logger.error(f"Error writing to CSV file: {e}")
            except KeyError as e:
                logger.error(f"Missing field in data row: {e}")
                logger.error(f"Expected fields: {self.fieldnames}")
                logger.error(f"Data row keys: {list(data_rows[0].keys()) if data_rows else 'No data'}")
                # Log the problematic row for debugging
                if data_rows:
                    logger.error(f"Problematic row: {data_rows[0]}")
            except Exception as e:
                logger.error(f"Unexpected error writing to CSV: {e}")

    def flush(self):
        """Force an immediate fsync of any buffered rows.
        Call this from the shutdown path before closing to guarantee all data
        is on disk even if the batch threshold has not yet been reached."""
        with self._lock:
            if self._unflushed_rows == 0:
                return
            try:
                with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
                    f.flush()
                    os.fsync(f.fileno())
                self._unflushed_rows = 0
                self._last_flush = time.monotonic()
            except OSError as e:
                logger.error(f"Error flushing CSV to disk: {e}")