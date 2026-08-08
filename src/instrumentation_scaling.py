# instrumentation.py

import os
import time
import threading
import csv
from queue import Queue
from pathlib import Path
import psutil

# All possible event fields across the system. Adding new fields requires
# adding them here so the CSV schema stays consistent across runs.
_FIELDS = [
    # Universal
    "kind",
    "t",
    "thread",
    # GSM lock events
    "op",                # "read" | "write"
    "method",            # e.g. "add_state", "snapshot", ...
    "wait",              # seconds
    "hold",              # seconds
    "count",             # for queue draining methods
    "new_state_id",
    "state_id",
    "n_states",
    "n_models",
    # Producer events
    "gpu_id",
    "sim_time",
    "put_time",
    # Consumer events
    "queue_wait", #consumer wait time for a window to be available in the queue
    "queue_wait_analysis", #the window time sitting in the queue before analysis starts
    "analysis_time",
    "state",
    "is_new",
    "is_uncertain",
    # Both producer and consumer
    "traj_id",
    "window_idx",
    "windows_processed",
    "windows_produced",
    "parent_seed",
    # Main-loop events
    "n_active",
    "n_pending",
    "n",
    "origin",
    "final",
    "active_cap",
    "num_seeds",
    "total_budget_ns",
    "total_spent_ns",
    "initial_windows",
    "extension_windows",
    # System-level metrics
    "node_cpu_pct",
    "node_mem_pct",
    "node_mem_used_gb",
    "proc_cpu_pct",
    "proc_mem_gb",
]


class EventLogger:
    def __init__(self):
        self.queue = Queue()
        self.run_id = None
        self.outdir = None
        self.writer_thread = None
        self._running = False   # tracks whether writer thread is alive

    def start(self, run_id: str, outdir: str):
        if self._running:
            raise RuntimeError("EventLogger already started")
        self.run_id = run_id
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self._running = True
        self.writer_thread = threading.Thread(
            target=self._writer, name="EventLoggerWriter", daemon=True
        )
        self.writer_thread.start()

    def stop(self):
        if not self._running:
            return
        self.queue.put(None)
        self.writer_thread.join(timeout=30.0)
        self._running = False

    def log(self, kind: str, **fields):
        evt = {
            "kind": kind,
            "t": time.time(),
            "thread": threading.get_ident(),
        }
        evt.update(fields)
        self.queue.put(evt)

    def _writer(self):
        path = Path(self.outdir) / f"{self.run_id}_events.csv"
        with open(path, "w", newline="", buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
            writer.writeheader()
            while True:
                evt = self.queue.get()
                if evt is None:
                    break
                writer.writerow(evt)

    
logger = EventLogger()