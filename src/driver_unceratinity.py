#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from run_simulation_uncertain_glob import run_simulationU    #for VH use: from run_simulation_uncertain_villin import run_simulationU
from global_state import global_state
from instrumentation_scaling import logger
import argparse
import wandb
os.environ["WANDB_MODE"] = "offline"
wandb.init()

# ---------------------------
# GPU assignment (one per producer)
# ---------------------------
def get_available_gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
        n = len([l for l in out.strip().split("\n") if l.strip()])
        return max(n, -1)
    except Exception:
        return -1

def assign_gpu():
    with gpu_lock:
      if not free_gpus:
        return None
      gid = min(free_gpus)
      free_gpus.discard(gid)
      # gid = gpu_counter[0] % available_gpus
      # gpu_counter[0] += 1
    return gid

def release_gpu(gpu_id):
  if gpu_id is None:
    return 
  with gpu_lock:
    free_gpus.add(gpu_id)

available_gpus = get_available_gpu_count()
gpu_lock = threading.Lock()
free_gpus = set(range(available_gpus))


# ----------------------------
# Helpers
# ----------------------------
def now_iso():
    return datetime.now().isoformat()

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def find_seed_pdb(seed_folder: str):
    """Find the seed .pdb inside a seed folder."""
    pdbs = sorted(glob.glob(os.path.join(seed_folder, "*.pdb")))
    return pdbs[0] if pdbs else None

def get_seed_folders(seeds_root: str) -> list:
    """Return sorted list of seed folders."""
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
    """Mark seed folder as completed."""
    open(os.path.join(seed_folder, ".COMPLETED"), "w").close()

def is_seed_completed(seed_folder: str) -> bool:
    """Check if seed folder already completed."""
    return os.path.exists(os.path.join(seed_folder, ".COMPLETED"))

def get_next_seeds(seeds_root: str, n: int = 2) -> list:
    """Get next N uncompleted seed folders."""
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
parser.add_argument("--dt_fs", type=float, default=2.0)
parser.add_argument("--alpha_CA", action="store_true")
parser.add_argument("--active", type=int, default=3)
parser.add_argument("--num_seeds", type=int, default=10)
parser.add_argument("--seeds", type=str, default=None, 
                    help="Comma-separated seed names to run (e.g., 'seed_01,seed_03'). Overrides auto-selection.")
parser.add_argument("--initial_windows", type=int, default=20, help="Windows before checking for new state")
parser.add_argument("--extension_windows", type=int, default=20, help="Additional windows if new state found")

args = parser.parse_args()
ROOT = os.path.abspath(args.root)
RESIDUES = os.path.join(ROOT, "residues.txt")
SEEDS_ROOT = os.path.join(ROOT, "uncertain_seeds")
DESC_ROOT = os.path.join(ROOT, "uncertain_descendant")
safe_mkdir(SEEDS_ROOT)
safe_mkdir(DESC_ROOT)

if not os.path.exists(RESIDUES):
    sys.exit(f"[ERROR] residues.txt not found at: {RESIDUES}")

# ----------------------------
# Load Baseline 
# ----------------------------
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

# ----------------------------
# Budget and Window Calculations 
# ----------------------------
seed_budgets = {}  # seed_folder -> remaining_ns
trajectory_to_seed = {}  # traj_id -> seed_folder
descendant_to_seed = {}  # descendant_folder -> seed_folder
seed_budgets_lock = threading.Lock()

STEPS_PER_WINDOW = args.steps_per_frame * args.frames_per_window
WINDOW_NS = (STEPS_PER_WINDOW * args.dt_fs) / 1e6
TOTAL_BUDGET_NS = args.total_ns

INITIAL_WINDOWS = args.initial_windows  
EXTENSION_WINDOWS = args.extension_windows  
TOTAL_SEEDS = args.num_seeds
BUDGET_PER_SEED = TOTAL_BUDGET_NS / TOTAL_SEEDS  

print(f"[CONFIG] Window: {args.frames_per_window} frames = {WINDOW_NS:.4f} ns")
print(f"[CONFIG] Initial check: {INITIAL_WINDOWS} windows = {INITIAL_WINDOWS * WINDOW_NS:.4f} ns")
print(f"[CONFIG] Extension: {EXTENSION_WINDOWS} windows = {EXTENSION_WINDOWS * WINDOW_NS:.4f} ns")
print(f"[CONFIG] Budget per seed: {BUDGET_PER_SEED:.2f} ns")

# Budget tracking
BUDGET_FILE = os.path.join(ROOT, "total_budget.json")
budget_lock = threading.Lock()
SEED_BUDGET_FILE = os.path.join(ROOT, "seed_budgets.json")

if os.path.exists(BUDGET_FILE):
    with open(BUDGET_FILE, "r") as f:
        budget = json.load(f)
    print(f"[INFO] Resuming. Already spent: {budget['spent_ns']:.2f} ns")
else:
    budget = {"spent_ns": 0.0, "runs": []}

def save_budget():
    with open(BUDGET_FILE, "w") as f:
        json.dump(budget, f, indent=2)

def load_seed_budgets():
    if not os.path.exists(SEED_BUDGET_FILE):
        return {}

    with open(SEED_BUDGET_FILE, "r") as f:
        saved = json.load(f)

    return {
        os.path.abspath(seed_folder): float(remaining_ns)
        for seed_folder, remaining_ns in saved.items()
    }

def save_seed_budgets():
    with seed_budgets_lock:
        saved = {
            seed_folder: remaining_ns
            for seed_folder, remaining_ns in seed_budgets.items()
        }
    with open(SEED_BUDGET_FILE, "w") as f:
        json.dump(saved, f, indent=2)


with budget_lock:
    remaining = TOTAL_BUDGET_NS - budget["spent_ns"]
    if remaining <= 0:
        print(f"[ERROR] Budget exhausted!")
        sys.exit(1)

RUN_ID = os.environ.get("PULSE_RUN_ID", f"run_{int(time.time())}")
logger.start(run_id=RUN_ID, outdir=os.path.join(ROOT, "instrumentation_logs"))
logger.log("run_start",
          total_budget_ns=TOTAL_BUDGET_NS,
          active_cap=args.active,
          num_seeds=3,
          initial_windows=INITIAL_WINDOWS,
          extension_windows=EXTENSION_WINDOWS)

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

SEED_TRAJ_MAP_FILE = os.path.join(ROOT, "seed_traj_map.json")
seed_traj_map_lock = threading.Lock()
seed_traj_map = {} 

def save_seed_traj_map(mapping: dict):
    with seed_traj_map_lock:
        with open(SEED_TRAJ_MAP_FILE, "w") as f:
            json.dump(mapping, f, indent=2)

def detect_existing_traj_id(run_folder: str):
    """    
    Detect traj_id from the .chk file in the folder
    """
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
    Calls analysis_main and Returns: (state_id, metadata)
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

# ---------------------------
# Producer Block
# ---------------------------
def producer_thread(run_folder: str, traj_id: int, working_pdb: str, gpu_id: int,
                    stop_flag: dict, q: Queue):
    """Runs simulation windows."""
    simulation = None
    last_state = None

    # RESUME FIX: find last completed window and start from there
    existing = sorted(glob.glob(os.path.join(run_folder, f"traj_{traj_id}_window_*.dcd")))
    w = len(existing)
    logger.log("producer_start", traj_id=traj_id, gpu_id=gpu_id)
    if w > 0:
        print(f"[RESUME] traj_{traj_id}: resuming from window {w} ({w} already completed)")

    while True:
        if stop_flag["stop"]:
            break

        # Check global budget
        with budget_lock:
            if budget["spent_ns"] >= TOTAL_BUDGET_NS:
                stop_flag["stop"] = True
                break

        # Check seed budget
        parent_seed = trajectory_to_seed.get(traj_id)
        if parent_seed:
            with seed_budgets_lock:
                if seed_budgets.get(parent_seed, 0) <= 0:
                    stop_flag["stop"] = True
                    print(f"[PRODUCER STOP] traj_{traj_id}: Seed budget exhausted")
                    break

        dcd_path = os.path.join(run_folder, f"traj_{traj_id}_window_{w}.dcd")

        print(f"[SIM] traj_{traj_id} | GPU {gpu_id} | window {w} | global {budget['spent_ns']:.1f}/{TOTAL_BUDGET_NS:.1f} ns")
        t_sim_start = time.perf_counter()
        simulation, last_state = run_simulationU(
            dcd_path, w, working_pdb, traj_id,
            simulation=simulation, last_state=last_state, gpu_id=gpu_id
        )
        t_sim_end = time.perf_counter()
        t_put_start = time.perf_counter()
        q.put((dcd_path, traj_id))
        t_put_end = time.perf_counter()
        logger.log("window_produced",traj_id=traj_id,window_idx=w,gpu_id=gpu_id,sim_time=t_sim_end - t_sim_start, put_time=t_put_end - t_put_start)
        w += 1

    q.put(("exit", traj_id))
    logger.log("producer_exit", traj_id=traj_id, windows_produced=w)

# ---------------------------
# Consumer Block
# ---------------------------

def consumer_thread(run_folder: str, traj_id: int, working_pdb: str,
                    stop_flag: dict, q: Queue, counters: dict):

    window_count = 0
    found_new_state = False
    logger.log("consumer_start", traj_id=traj_id)
    while True:
        t_get_start = time.perf_counter()
        item, tid = q.get()
        t_get_end = time.perf_counter()

        if item == "exit":
            q.task_done()
            logger.log("consumer_exit", traj_id=traj_id, windows_processed=window_count)
            return

        if stop_flag["stop"]:
            q.task_done()
            continue

        # Get state and metadata
        t_analysis_start = time.perf_counter()
        state, metadata = analysis(item, working_pdb, tid)
        t_analysis_end = time.perf_counter()
        window_count += 1
        
        # Get parent seed
        parent_seed = trajectory_to_seed.get(tid)
        if not parent_seed:
            print(f"[ERROR] traj_{tid}: No parent seed found!")
            q.task_done()
            continue
        
        # Deduct budget
        with budget_lock:
            budget["spent_ns"] += WINDOW_NS
        
        with seed_budgets_lock:
            seed_budgets[parent_seed] -= WINDOW_NS
            remaining_seed_budget = seed_budgets[parent_seed]
        logger.log("window_done", traj_id=tid, window_idx=window_count, queue_wait=t_get_end - t_get_start, analysis_time=t_analysis_end - t_analysis_start, state=state, is_new=getattr(metadata, "new_state_created", False), is_uncertain=(state == -1), parent_seed=os.path.basename(parent_seed))
        save_seed_budgets()
        timestamp = datetime.now().strftime("%H:%M:%S")
        seed_name = os.path.basename(parent_seed)

        if state == -2:
            counters["uncertain"] = 0
            counters["old"] = 0
            if metadata.new_state_created:          # BREAK caused a new state
                found_new_state = True
                print(f"[{timestamp}] [ANALYSIS] traj_{tid} w{window_count}: BREAK + NEW STATE ✓ | seed={seed_name}")
            else:
                print(f"[{timestamp}] [ANALYSIS] traj_{tid} w{window_count}: BREAK | seed={seed_name}")

        elif metadata.new_state_created:
            counters["uncertain"] = 0
            counters["old"] = 0
            found_new_state = True
            print(f"[{timestamp}] [ANALYSIS] traj_{tid} w{window_count}: state={state} NEW STATE ✓ | seed={seed_name}")

        elif metadata.forced_reuse:
            counters["uncertain"] += 1
            counters["old"] += 1
            print(f"[{timestamp}] [ANALYSIS] traj_{tid} w{window_count}: state={state} FORCED REUSE | seed={seed_name}")

        elif state == -1:
            counters["uncertain"] += 1
            print(f"[{timestamp}] [ANALYSIS] traj_{tid} w{window_count}: UNCERTAIN | seed={seed_name} ({remaining_seed_budget:.2f} ns left)")

        else:
            if state in old_states:
                counters["old"] += 1
                counters["uncertain"] = 0
            print(f"[{timestamp}] [ANALYSIS] traj_{tid} w{window_count}: state={state} | seed={seed_name}")

        if window_count == INITIAL_WINDOWS:
            if found_new_state:
                print(f"[CONTINUE] traj_{tid}: New state found! Running {EXTENSION_WINDOWS} more windows")
            else:
                stop_flag["stop"] = True
                print(f"[STOP] traj_{tid}: No new state after {INITIAL_WINDOWS} windows ({INITIAL_WINDOWS * WINDOW_NS:.3f} ns)")

        elif window_count == (INITIAL_WINDOWS + EXTENSION_WINDOWS):
            stop_flag["stop"] = True
            print(f"[STOP] traj_{tid}: Completed {window_count} windows ({window_count * WINDOW_NS:.3f} ns)")

        # Budget checks
        if remaining_seed_budget <= 0:
            stop_flag["stop"] = True
            print(f"[STOP] traj_{tid}: Seed {seed_name} budget exhausted")

        with budget_lock:
            if budget["spent_ns"] >= TOTAL_BUDGET_NS:
                stop_flag["stop"] = True
                print(f"[STOP] traj_{tid}: Global budget exhausted")

        q.task_done()

# ----------------------------
# Active trajectories
# ----------------------------
active = {}
def start_trajectory(run_folder: str, origin: str, parent_seed: str):
    """Start trajectory with seed tracking."""
    # Check seed budget first
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
    # Producer and Consumer threads 
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
    
    # Check active trajectories first
    for tid, info in active.items():
        parent_folder = info["folder"]
        if desc_folder.startswith(parent_folder) or parent_folder in desc_folder:
            print(f"[DEBUG] Found via active trajectory {tid}: {os.path.basename(info['parent_seed'])}")
            return info["parent_seed"]
    
    # Check trajectory_to_seed mapping
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

# Initialize seed budgets
saved_seed_budgets = load_seed_budgets()
with seed_budgets_lock:
    for sf in seed_folders:
        sf = os.path.abspath(sf)
        if sf in saved_seed_budgets:
            seed_budgets[sf] = saved_seed_budgets[sf]
            print(
                f"[RESUME] {os.path.basename(sf)}: "
                f"{seed_budgets[sf]:.2f} ns remaining"
            )
        else:
            seed_budgets[sf] = BUDGET_PER_SEED


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
DESCENDANT_CHECK_INTERVAL = 50 # in seconds

while True:
    with budget_lock:
        if budget["spent_ns"] >= TOTAL_BUDGET_NS:
            print(f"\n[BUDGET EXHAUSTED]")
            budget_exhausted = True
            break

    drain_finished()
    logger.log("active_snapshot", n_active=len(active), n_pending=len(pending_folders))
    current_time = time.time()
    if current_time - last_descendant_check >= DESCENDANT_CHECK_INTERVAL:
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

# Collect any final descendants after all threads complete
print("[INFO] Collecting final descendants...")
final_descendants = collect_descendants()
if final_descendants:
    print(f"[WARNING] Found {len(final_descendants)} descendants after completion - they were not processed!")
    print("[INFO] Re-run with same seeds to process these descendants")

# Mark seeds complete based on budget exhaustion
# Complete = budget exhausted, won't be auto-picked in future runs
# Incomplete = budget remaining, can be resumed or manually specified
with seed_budgets_lock:
    for sf in seed_folders:
        if seed_budgets[sf] < WINDOW_NS:
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
save_seed_budgets()

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
print(f"Spent this run: {budget['spent_ns'] - run_start_ns:.2f} ns")
print(f"Total cumulative: {budget['spent_ns']:.2f}/{TOTAL_BUDGET_NS:.1f} ns")
print(f"States: {len(final.get('states', []))}")
print(f"Trajectory IDs used: {traj_id_manager.current_id}")
print("\nSeed budget usage:")
with seed_budgets_lock:
    for sf in seed_folders:
        used = BUDGET_PER_SEED - seed_budgets[sf]
        print(f"  {os.path.basename(sf)}: {used:.2f}/{BUDGET_PER_SEED:.2f} ns used")
print("=" * 60)
logger.log("run_end", total_spent_ns=budget["spent_ns"])
logger.stop()
