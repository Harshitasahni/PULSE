#!/usr/bin/env python3
import os
import sys
import re
import json
import shutil
import threading
import subprocess
from datetime import datetime
from queue import Queue
import pickle
import glob
import time

from analysis import main as analysis_main  
from analysis_metadata import AnalysisMetadata, AnalysisResult
from global_state import global_state
from instrumentation_scaling import logger
import argparse
import random
import psutil

# ----------------------------
# Helpers
# ----------------------------
def start_resource_monitor(interval=5.0):
    """
    
    Sample node CPU + memory and process RSS every `interval`s into instrumentation.
    
    """
    def _monitor():
        proc = psutil.Process()
        psutil.cpu_percent(None)      # prime (first call returns 0.0)
        proc.cpu_percent(None)        # prime process-level too
        while True:
            node_cpu = psutil.cpu_percent(None)
            vm = psutil.virtual_memory()
            proc_cpu = proc.cpu_percent(None)
            proc_mem = proc.memory_info().rss / 1e9
            logger.log("resource_sample",
                       node_cpu_pct=node_cpu,
                       node_mem_pct=vm.percent,
                       node_mem_used_gb=vm.used / 1e9,
                       proc_cpu_pct=proc_cpu,
                       proc_mem_gb=proc_mem,
                       n_active=len(active))
            time.sleep(interval)
    t = threading.Thread(target=_monitor, name="ResourceMonitor", daemon=True)
    t.start()
    return t

# Per-window OpenMM sim cost (seconds) vs concurrency.

def sim_cost_for(n_active: int) -> float:
    _SIM_COST = {
      1: 38.0,  2: 40.5,  3: 43.0,  4: 45.0,
      5: 53.0,  6: 55.0,  7: 63.0,  8: 68.0,
      10:73.0,  12:78.0,  14:81.2,  16: 83.0, 18:85.5, 20:88.0, 22:89.2, 24:90.0,
      26:91.0,  28:92.0,  30:93.5,  32: 96.3,}
    keys = sorted(_SIM_COST)
    n = max(1, n_active)
    if n <= keys[0]:
        return _SIM_COST[keys[0]]
    if n >= keys[-1]:
        return _SIM_COST[keys[-1]]
    if n in _SIM_COST:
        return _SIM_COST[n]
    lo = max(k for k in keys if k < n)
    hi = min(k for k in keys if k > n)
    f = (n - lo) / (hi - lo)
    return _SIM_COST[lo] + f * (_SIM_COST[hi] - _SIM_COST[lo])
  
def now_iso():
    return datetime.now().isoformat()

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def get_available_gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
        n = len([l for l in out.strip().split("\n") if l.strip()])
        return max(n, -1)
    except Exception:
        return -1
           # analysis needs no GPU; slots are just labels
free_gpus = set()


gpu_lock = threading.Lock()

def assign_gpu():
    with gpu_lock:
      if not free_gpus:
        return None
      gid = min(free_gpus)
      free_gpus.discard(gid)
    return gid

def release_gpu(gpu_id):
  if gpu_id is None:
    return 
  with gpu_lock:
    free_gpus.add(gpu_id)

def find_seed_pdb(seed_folder: str):
    pdbs = sorted(glob.glob(os.path.join(seed_folder, "*.pdb")))
    return pdbs[0] if pdbs else None

def get_seed_folders(seeds_root: str) -> list:
    if not os.path.isdir(seeds_root):
        return []
    subs = []
    for name in os.listdir(seeds_root):
        p = os.path.join(seeds_root, name)
        if os.path.isdir(p):
            subs.append(p)
    subs.sort()
    return subs

def mark_seed_completed(seed_folder: str):
    open(os.path.join(seed_folder, ".COMPLETED"), "w").close()

def is_seed_completed(seed_folder: str) -> bool:
    return os.path.exists(os.path.join(seed_folder, ".COMPLETED"))

def get_next_seeds(seeds_root: str, n: int = 2) -> list:
    all_seeds = get_seed_folders(seeds_root)
    uncompleted = [s for s in all_seeds if not is_seed_completed(s)]
    return uncompleted[:n]

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="window exploration with per-seed budget")
parser.add_argument("--root", required=True, help="ROOT folder")
parser.add_argument("--total_ns", type=float, default=100.0, help="Total budget in ns (cumulative across all runs)")
parser.add_argument("--steps_per_frame", type=int, default=1000)
parser.add_argument("--frames_per_window", type=int, default=50)
parser.add_argument("--dt_fs", type=float, default=1.0)
parser.add_argument("--alpha_CA", action="store_true")
parser.add_argument("--active", type=int, default=3)
parser.add_argument("--num_seeds", type=int, default=8)
parser.add_argument("--seeds", type=str, default=None, 
                    help="Comma-separated seed names to run (e.g., 'seed_01,seed_03'). Overrides auto-selection.")
parser.add_argument("--initial_windows", type=int, default=20, help="Windows before checking for new state")
parser.add_argument("--extension_windows", type=int, default=20, help="Additional windows if new state found")
parser.add_argument("--fixed_windows", type=int, default=20, help="Fixed number of windows analyzed per trajectory")
parser.add_argument("--sim_jitter_frac", type=float, default=0.02, help="Uniform +/- fractional jitter around the median sim cost (0 = exact median)")
parser.add_argument("--no_descendants", action="store_true", help="Disable descendant collection (required in replay mode: descendants have no window files to replay)")

args = parser.parse_args()

ROOT = os.path.abspath(args.root)
RESIDUES = os.path.join(ROOT, "residues.txt")

SEEDS_ROOT = os.path.join(ROOT, "uncertain_seeds")
DESC_ROOT = os.path.join(ROOT, "uncertain_descendants")
safe_mkdir(SEEDS_ROOT)
safe_mkdir(DESC_ROOT)

free_gpus = set(range(args.active))


if not os.path.exists(RESIDUES):
    sys.exit(f"[ERROR] residues.txt not found at: {RESIDUES}")

# Load baseline
BASELINE_UNCERTAIN = os.path.join(ROOT, "variables_global_uncertain.pkl")
BASELINE_GLOBAL = os.path.join(ROOT, "variables_global.pkl")

baseline_blob = None
baseline_nstates = 0
old_states = set()

baseline_path = None
if os.path.exists(BASELINE_UNCERTAIN):
    baseline_path = BASELINE_UNCERTAIN
elif os.path.exists(BASELINE_GLOBAL):
    baseline_path = BASELINE_GLOBAL

if baseline_path:
    print(f"[INFO] Loading baseline from: {baseline_path}")
    with open(baseline_path, "rb") as f:
        baseline_blob = pickle.load(f)
    states_list = baseline_blob.get("states") or baseline_blob.get("States") or []
    baseline_nstates = len(states_list)
    old_states = set(range(baseline_nstates))
    global_state.restore(baseline_blob)

# Window calculation
STEPS_PER_WINDOW = args.steps_per_frame * args.frames_per_window

INITIAL_WINDOWS = args.initial_windows  
EXTENSION_WINDOWS = args.extension_windows 

N_WINDOWS_PER_SEED = 20
WINDOW_NS = (args.steps_per_frame * args.frames_per_window * args.dt_fs) / 1e6
BUDGET_PER_SEED  = N_WINDOWS_PER_SEED * WINDOW_NS  

# Budget tracking
BUDGET_FILE = os.path.join(ROOT, "total_budget.json")
budget_lock = threading.Lock()

if os.path.exists(BUDGET_FILE):
    with open(BUDGET_FILE, "r") as f:
        budget = json.load(f)
    print(f"[INFO] Resuming. Already spent: {budget['spent_ns']:.2f} ns")
else:
    budget = {"spent_ns": 0.0, "runs": []}

def save_budget():
    with open(BUDGET_FILE, "w") as f:
        json.dump(budget, f, indent=2)

# ----------------------------
# Trajectory ID Manager
# ----------------------------
class TrajectoryIDManager:
    def __init__(self, root_folder: str):
        self.root_folder = root_folder
        self.lock = threading.Lock()
        self._load_max_id()
    
    def _load_max_id(self):
        pat = os.path.join(self.root_folder, "variables_*.pkl")
        max_id = -1
        for p in glob.glob(pat):
            base = os.path.basename(p)
            m = re.match(r"variables_(\d+)\.pkl$", base)
            if m:
                max_id = max(max_id, int(m.group(1)))
        self.current_id = max_id + 1
        print(f"[ID MANAGER] Starting from: {self.current_id}")
    
    def get_next_id(self) -> int:
        with self.lock:
            tid = self.current_id
            self.current_id += 1
            return tid

traj_id_manager = TrajectoryIDManager(ROOT)

# ----------------------------
# seed -> traj_id audit log  
# ----------------------------
SEED_TRAJ_MAP_FILE = os.path.join(ROOT, "seed_traj_map.json")
seed_traj_map_lock = threading.Lock()
seed_traj_map = {}

def save_seed_traj_map(mapping: dict):
    with seed_traj_map_lock:
        with open(SEED_TRAJ_MAP_FILE, "w") as f:
            json.dump(mapping, f, indent=2)

def detect_existing_traj_id(run_folder: str):
    chk_files = glob.glob(os.path.join(run_folder, "traj*.chk"))
    if chk_files:
        base = os.path.basename(chk_files[0])
        m = re.match(r"traj_?(\d+)\.chk$", base)
        if m:
            return int(m.group(1))
    return None

# ----------------------------
# Analysis wrapper
# ----------------------------
def analysis(dcd_path: str, top_pdb: str, traj_id: int):
    """
    Calls analysis_main and handles return value.
    Returns: (state_id, metadata)
    """
    try:
        result = analysis_main(
            dcd_path, top_pdb, RESIDUES, args.alpha_CA,
            ROOT, ROOT, uncertainw=True
        )
        
        if isinstance(result, AnalysisResult):
            return result
        elif isinstance(result, tuple):
            return result
        else:
            return (result, AnalysisMetadata())
            
    except Exception as e:
        print(f"[ERROR] analysis failed for {dcd_path}: {e}")
        return (-1, AnalysisMetadata.uncertain())


seed_budgets = {} 
trajectory_to_seed = {}  
descendant_to_seed = {}  
seed_budgets_lock = threading.Lock()

# ----------------------------
# Producer-Consumer threads
# ----------------------------
def _window_index(p):
    m = re.search(r"window_(\d+)\.dcd$", os.path.basename(p))
    return int(m.group(1)) if m else 0

def producer_thread(run_folder: str, traj_id: int, working_pdb: str, gpu_id: int,
                    stop_flag: dict, q: Queue):

  windows = sorted(glob.glob(os.path.join(run_folder, "traj_*_window_*.dcd")), key=_window_index)
  windows = windows[:args.fixed_windows]
  logger.log("producer_start", traj_id=traj_id, gpu_id=gpu_id)
  for w, dcd_path in enumerate(windows):
    if stop_flag["stop"]:
      break
    n_now = len(active)
    base = sim_cost_for(n_now)
    if args.sim_jitter_frac > 0:
        base *= (1.0 + random.uniform(-args.sim_jitter_frac, args.sim_jitter_frac))
    dt = max(0.0, base)
    t_sim_start = time.perf_counter()
    time.sleep(dt)
    t_sim_end = time.perf_counter()
    t_put_start = time.perf_counter()
    q.put((dcd_path, traj_id, time.time(), w))
    t_put_end = time.perf_counter()

    logger.log("window_produced", traj_id=traj_id, window_idx=w, gpu_id=gpu_id, sim_time=t_sim_end - t_sim_start, put_time=t_put_end - t_put_start)
    print(f"[SIM] traj_{traj_id} | slot {gpu_id} | window {w}, n_active={n_now})")

  q.put(("exit", traj_id))
  logger.log("producer_exit", traj_id=traj_id, windows_produced=len(windows))



def consumer_thread(run_folder: str, traj_id: int, working_pdb: str,
                    stop_flag: dict, q: Queue, counters: dict):
    """
    Fixed-windows benchmark path: when fixed_windows is set,
    run exactly N windows per trajectory.
    """
    window_count = 0
    found_new_state = False
    logger.log("consumer_start", traj_id=traj_id)
    while True:
        t_get_start = time.perf_counter()
        item = q.get()
        t_get_end = time.perf_counter()

        if item[0] == "exit":
            q.task_done()
            logger.log("consumer_exit", traj_id=traj_id, windows_processed=window_count)
            return

        if stop_flag["stop"]:
            q.task_done()
            continue

        dcd_path, tid, t_put, window_idx = item
        queue_wait_analysis = time.time() - t_put   # real time this window waited in queue

        # Get state and metadata
        t_analysis_start = time.perf_counter()
        state, metadata = analysis(dcd_path, working_pdb, tid)
        t_analysis_end = time.perf_counter()
        window_count += 1

        logger.log("window_done", traj_id=tid, window_idx=window_count,
                   queue_wait=t_get_end - t_get_start,
                   queue_wait_analysis=queue_wait_analysis,
                   analysis_time=t_analysis_end - t_analysis_start,
                   state=state, is_new=getattr(metadata, "new_state_created", False),
                   is_uncertain=(state == -1))

        if args.fixed_windows is not None:
            if window_count >= args.fixed_windows:
                stop_flag["stop"] = True   
                print(f"[STOP] traj_{tid}: Reached fixed window limit ({args.fixed_windows})")
            q.task_done()
            continue   # skip everything below
        parent_seed = trajectory_to_seed.get(tid)
        seed_name = os.path.basename(parent_seed)
        q.task_done()
# ----------------------------
# Active trajectories
# ----------------------------
active = {}
def start_trajectory(run_folder: str, origin: str, parent_seed: str):
    """Start a single trajectory with seed tracking."""
    with seed_budgets_lock:
        if seed_budgets.get(parent_seed, 0) < WINDOW_NS:
            print(f"[SKIP] {origin} from {os.path.basename(parent_seed)}: seed budget exhausted")
            return False
    
    seed_pdb = find_seed_pdb(run_folder)
    if not seed_pdb:
        print(f"[SKIP] No PDB found in: {run_folder}")
        return False
    existing_traj_id = detect_existing_traj_id(run_folder)
    if existing_traj_id is not None:
        traj_id = existing_traj_id
        print(f"[RESUME] Detected traj_id={traj_id} from .chk in {os.path.basename(run_folder)}")
    else:
        traj_id = traj_id_manager.get_next_id()
        print(f"[NEW] Assigned traj_id={traj_id} for {os.path.basename(run_folder)}")
    seed_traj_map[run_folder] = traj_id
    save_seed_traj_map(seed_traj_map)
    trajectory_to_seed[traj_id] = parent_seed
    
    working_pdb = os.path.join(run_folder, f"traj{traj_id}.pdb")
    if not os.path.exists(working_pdb):
        shutil.copy(seed_pdb, working_pdb)

    try:
        rdest = os.path.join(run_folder, "residues.txt")
        if not os.path.exists(rdest):
            shutil.copy(RESIDUES, rdest)
    except Exception:
        pass

    gpu_id = assign_gpu()
    q = Queue()
    stop_flag = {"stop": False}
    counters = {"uncertain": 0, "old": 0}

    prod = threading.Thread(
        target=producer_thread,
        args=(run_folder, traj_id, working_pdb, gpu_id, stop_flag, q),
        daemon=True
    )
    cons = threading.Thread(
        target=consumer_thread,
        args=(run_folder, traj_id, working_pdb, stop_flag, q, counters),
        daemon=True
    )
    logger.log("triple_started", traj_id=traj_id, origin=origin, parent_seed=os.path.basename(parent_seed))
    prod.start()
    cons.start()

    active[traj_id] = {
        "folder": run_folder,
        "origin": origin,
        "prod": prod,
        "cons": cons,
        "stop_flag": stop_flag,
        "traj_id": traj_id,
        "counters": counters,
        "parent_seed": parent_seed,
        "gpu_id":gpu_id,
    }

    seed_name = os.path.basename(parent_seed)
    print(f"[START] {origin} | traj_id={traj_id} | seed={seed_name} | gpu={gpu_id}")
    return True

# ----------------------------
# Collect descendants
# ----------------------------
run_info = None

def find_parent_seed_for_descendant(desc_folder: str) -> str:
    """Find which seed this descendant belongs to."""
    print(f"[DEBUG] Finding parent seed for: {desc_folder}")
    for tid, info in active.items():
        parent_folder = info["folder"]
        if desc_folder.startswith(parent_folder) or parent_folder in desc_folder:
            print(f"[DEBUG] Found via active trajectory {tid}: {os.path.basename(info['parent_seed'])}")
            return info["parent_seed"]
    for tid, seed in trajectory_to_seed.items():
        # Try to match based on trajectory ID in the path
        if f"traj_{tid}" in desc_folder:
            print(f"[DEBUG] Found via trajectory_to_seed mapping traj_{tid}: {os.path.basename(seed)}")
            return seed
    desc_name = os.path.basename(desc_folder)
    if desc_name.startswith("traj_"):
        parts = desc_name.split("_")
        if len(parts) >= 2:
            try:
                parent_traj_id = int(parts[1])
                if parent_traj_id in trajectory_to_seed:
                    seed = trajectory_to_seed[parent_traj_id]
                    print(f"[DEBUG] Found via name pattern traj_{parent_traj_id}: {os.path.basename(seed)}")
                    return seed
            except ValueError:
                pass
    print(f"[DEBUG] Using fallback - assigning to first available seed")
    for seed in seed_budgets:
        if seed_budgets[seed] > 0:
            print(f"[DEBUG] Fallback seed: {os.path.basename(seed)}")
            return seed
    
    print(f"[DEBUG] No parent seed found!")
    return None

def collect_descendants() -> list:
    """Collect descendant folders with parent seed tracking."""
    jobs = global_state.pop_all_uncertain_jobs()
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [DESCENDANTS] Checking global_state: {len(jobs)} uncertain jobs found")
    
    if not jobs:
        return []

    folders = []
    for pdb_path, name in jobs:
        print(f"[DESCENDANTS] Processing job: {name} | pdb={pdb_path}")
        
        if not pdb_path or not os.path.exists(pdb_path):
            print(f"[DESCENDANTS] Skipping - PDB path invalid or doesn't exist")
            continue

        run_folder = os.path.dirname(pdb_path)
        
        if DESC_ROOT in run_folder and os.path.exists(run_folder):
            parent_seed = descendant_to_seed.get(run_folder)
            if not parent_seed:
                parent_seed = find_parent_seed_for_descendant(run_folder)
            
            if not parent_seed:
                print(f"[DESCENDANTS] Skipping - no parent seed found")
                continue
            descendant_to_seed[run_folder] = parent_seed
                
            seed_budget = seed_budgets.get(parent_seed, 0)
            print(f"[DESCENDANTS] Parent seed: {os.path.basename(parent_seed)} | budget: {seed_budget:.2f} ns")
            
            if parent_seed and seed_budget > 0:
                folders.append((run_folder, "desc", parent_seed))
                if run_info is not None:
                    run_info["descendants"].append(os.path.basename(run_folder))
                print(f"[DESCENDANTS] Added to queue: {os.path.basename(run_folder)}")
            else:
                print(f"[DESCENDANTS] Skipping - seed budget exhausted")

    if folders:
        print(f"\n[DESCENDANTS] Collected {len(folders)} folders total\n")
    else:
        print(f"[DESCENDANTS] No valid descendants collected\n")

    return folders

# ----------------------------
# Remove finished trajectories
# ----------------------------
def drain_finished():
    """Remove finished trajectories."""
    done = []
    for tid, info in active.items():
        if not info["prod"].is_alive() and not info["cons"].is_alive():
            done.append(tid)
            counters = info["counters"]
            seed_name = os.path.basename(info["parent_seed"])
            print(f"[FINISHED] traj_{tid} | seed={seed_name} | uncertain={counters['uncertain']}, old={counters['old']}")
    for tid in done:
        info = active.pop(tid, None)
        if info is not None:
          release_gpu(info["gpu_id"])
        

# ----------------------------
# Main execution
# ----------------------------

if args.seeds:
    seed_inputs = [s.strip() for s in args.seeds.split(',')]
    seed_folders = []
    for seed_input in seed_inputs:
        if os.path.isabs(seed_input) and os.path.isdir(seed_input):
            seed_folders.append(os.path.abspath(seed_input))
            print(f"[MANUAL] Selected seed (full path): {os.path.basename(seed_input)}")
        elif os.path.isdir(seed_input):
            seed_folders.append(os.path.abspath(seed_input))
            print(f"[MANUAL] Selected seed (relative path): {os.path.basename(seed_input)}")
        else:
            seed_path = os.path.join(SEEDS_ROOT, seed_input)
            if os.path.isdir(seed_path):
                seed_folders.append(seed_path)
                print(f"[MANUAL] Selected seed: {seed_input}")
            else:
                print(f"[WARNING] Seed not found: {seed_input}")
    
    if not seed_folders:
        print("[ERROR] No valid seeds specified!")
        sys.exit(1)
else:
    seed_folders = get_next_seeds(SEEDS_ROOT, args.num_seeds)
    
    if not seed_folders:
        print("[INFO] No uncompleted seeds found!")
        sys.exit(0)
with seed_budgets_lock:
    for sf in seed_folders:
        seed_budgets[sf] = BUDGET_PER_SEED
TOTAL_BUDGET_NS = len(seed_folders) * BUDGET_PER_SEED + WINDOW_NS   # global never cuts early

with budget_lock:
    remaining = TOTAL_BUDGET_NS - budget["spent_ns"]
    if remaining <= 0:
        print(f"[ERROR] Budget exhausted!")
        sys.exit(1)
print("=" * 60)
print("[WINDOW EXPLORATION DRIVER]")
print(f"ROOT: {ROOT}")
RUN_ID = os.environ.get("PULSE_RUN_ID", f"run_{int(time.time())}")
logger.start(run_id=RUN_ID, outdir=os.path.join(ROOT, "instrumentation_logs"))
logger.log("run_start", total_budget_ns=TOTAL_BUDGET_NS, active_cap=args.active,
           num_seeds=len(seed_folders), initial_windows=INITIAL_WINDOWS,
           extension_windows=EXTENSION_WINDOWS)

print(f"\n[INFO] Running {len(seed_folders)} seed(s):")
for sf in seed_folders:
    print(f"  - {os.path.basename(sf)} (budget: {BUDGET_PER_SEED:.2f} ns)")
print()

pending_folders = [(sf, "seed", sf) for sf in seed_folders]  # (folder, origin, parent_seed)

run_start_ns = budget["spent_ns"]
run_info = {
    "run_number": len(budget.get("runs", [])) + 1,
    "started_at": now_iso(),
    "seeds": [os.path.basename(sf) for sf in seed_folders],
    "descendants": [],
    "start_budget_ns": run_start_ns,
    "seed_budgets": {os.path.basename(sf): BUDGET_PER_SEED for sf in seed_folders},
}

budget_exhausted = False
last_descendant_check = time.time()
DESCENDANT_CHECK_INTERVAL = 50 # 5 minutes in seconds
start_resource_monitor(interval=5.0) 
while True:
    with budget_lock:
        if budget["spent_ns"] >= TOTAL_BUDGET_NS:
            print(f"\n[BUDGET EXHAUSTED]")
            budget_exhausted = True
            break

    drain_finished()
    logger.log("active_snapshot", n_active=len(active), n_pending=len(pending_folders))
    current_time = time.time()
    if not args.no_descendants and current_time - last_descendant_check >= DESCENDANT_CHECK_INTERVAL:
        desc_folders = collect_descendants()
        if desc_folders:
          logger.log("descendants_collected", n=len(desc_folders))
        pending_folders.extend(desc_folders)
        last_descendant_check = current_time

    while len(active) < args.active and pending_folders:
        with budget_lock:
            if budget["spent_ns"] >= TOTAL_BUDGET_NS:
                break
        
        folder, origin, parent_seed = pending_folders.pop(0)
        start_trajectory(folder, origin, parent_seed)

    if len(active) == 0 and len(pending_folders) == 0:
      if not args.no_descendants:
        final_desc = collect_descendants()
        if final_desc:
            logger.log("descendants_collected", n=len(final_desc), final=True)
            pending_folders.extend(final_desc)
            print(f"[INFO] Found {len(final_desc)} final descendants, continuing...")
            last_descendant_check = time.time()
            continue
      print("\n[COMPLETE] All finished")
      break

    save_budget()
    time.sleep(10.0) 
print("\n[INFO] Waiting for active trajectories...")
for tid, info in list(active.items()):
    info["prod"].join()
    info["cons"].join()

print("[INFO] Collecting final descendants...")
if not args.no_descendants:
  final_descendants = collect_descendants()
  if final_descendants:
      print(f"[WARNING] Found {len(final_descendants)} descendants after completion - they were not processed!")
      print("[INFO] Re-run with same seeds to process these descendants")
with seed_budgets_lock:
    for sf in seed_folders:
        if seed_budgets[sf] <= WINDOW_NS:
            # Budget exhausted - mark as complete
            mark_seed_completed(sf)
            print(f"[COMPLETED] {os.path.basename(sf)} - budget exhausted")
        else:
            print(f"[INCOMPLETE] {os.path.basename(sf)} - {seed_budgets[sf]:.2f} ns remaining, can resume")

run_info["finished_at"] = now_iso()
run_info["end_budget_ns"] = budget["spent_ns"]
run_info["spent_this_run_ns"] = budget["spent_ns"] - run_start_ns

# Record final seed budgets
with seed_budgets_lock:
    run_info["final_seed_budgets"] = {os.path.basename(sf): seed_budgets[sf] for sf in seed_folders}

if "runs" not in budget:
    budget["runs"] = []
budget["runs"].append(run_info)

save_budget()

final = global_state.snapshot()
final_out = os.path.join(ROOT, "variables_global_uncertain.pkl")
with open(final_out, "wb") as f:
    pickle.dump({
        "models": final.get("models", []),
        "states": final.get("states", []),
        "num_models": final.get("num_models", 0),
        "total_ns_spent": budget["spent_ns"],
        "finished_at": now_iso(),
    }, f)

print("\n" + "=" * 60)
print("[FINAL REPORT]")
print(f"States: {len(final.get('states', []))}")
print(f"Trajectory IDs used: {traj_id_manager.current_id}")
logger.log("run_end", total_spent_ns=budget["spent_ns"])
logger.stop()