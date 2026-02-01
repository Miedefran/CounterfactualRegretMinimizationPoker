import multiprocessing
import time
import socket
from typing import Optional, Dict, Set, List, Tuple
import threading

# We need these to process data in the child process
from memray import SocketReader
from memray.reporters.tui import (
    Snapshot,
    aggregate_allocations,
    MAX_MEMORY_RATIO,
    _EMPTY_SNAPSHOT
)

# derived from https://github.com/bloomberg/memray/blob/main/src/memray/reporters/tui.py

def reader_process_func(port: int, queue: multiprocessing.Queue):
    """
    Runs in a separate process. Connects to Memray Tracker on port,
    reads snapshots, aggregates them, and sends safe-to-pickle data back.
    """
    print(f"Reader: Attempting to connect on port {port}...")

    for attempt in range(50):
        try:
            with SocketReader(port=port) as reader:
                print(f"Reader: Connected! PID={reader.pid}")

                while reader.is_active:
                    try:
                        # Read records
                        records = list(reader.get_current_snapshot(merge_threads=True))

                        # Process data (heavy lifting done here to save UI thread)
                        heap_size = sum(record.size for record in records)

                        # Check if native traces enabled on reader
                        has_native = reader.has_native_traces

                        records_by_location = aggregate_allocations(
                            records, MAX_MEMORY_RATIO * heap_size, has_native
                        )

                        # Convert to standard dict to avoid pickling lambda in defaultdict
                        records_by_location = dict(records_by_location)

                        # Extract thread names (tid -> name)
                        # records contains AllocationRecord objects which are NOT picklable.
                        thread_names = {r.tid: r.thread_name for r in records}

                        # Create a Snapshot with empty records to make it picklable
                        safe_snapshot = Snapshot(
                            heap_size=heap_size,
                            records=[],  # Empty list
                            records_by_location=records_by_location
                        )

                        # Send to UI
                        if not queue.full():
                            queue.put((safe_snapshot, thread_names, not reader.is_active))

                        time.sleep(0.5)

                    except Exception as e:
                        # print(f"Error in reader loop: {e}")
                        pass

                print("Reader: Tracker disconnected")
                return

        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)

    print("Reader: Failed to connect after 50 attempts")


class RemoteReaderClient:
    """
    Acts as the interface for the TUI to get data from the reader process queue.
    """

    def __init__(self, queue: multiprocessing.Queue):
        self.queue = queue
        self.latest_snapshot = _EMPTY_SNAPSHOT
        self.latest_thread_names = {}
        self.disconnected = False

    def poll(self):
        """
        Called by TUI thread to check for new data.
        Returns (snapshot, thread_names, disconnected)
        """
        try:
            # Drain queue to get latest
            while not self.queue.empty():
                snapshot, thread_names, disconnected = self.queue.get_nowait()
                self.latest_snapshot = snapshot
                self.latest_thread_names = thread_names
                self.disconnected = disconnected
        except Exception:
            pass

        return self.latest_snapshot, self.latest_thread_names, self.disconnected
