"""
Baseline simulation and analysis driver.

Usage:
    python driver_baseline.py <trajectory_input_dir> <output_dir>
"""

import os, sys, time, glob, pickle, threading, queue, re
import subprocess
from analysis import *                     # provides main for analyzing MD data; logic mostly from OBGL 
from run_simulation_villian import *       # provides run_simulation, load_progress, save_progress for VH, Change to _glob for Beta-lactoglobulin
from global_state import global_state



# ---------------------------
# CPU thread env
# ---------------------------
max_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "5"))
os.environ["OMP_NUM_THREADS"] = str(max_cpus)
os.environ["MKL_NUM_THREADS"] = str(max_cpus)
os.environ["OPENBLAS_NUM_THREADS"] = str(max_cpus)
os.environ["NUMBA_NUM_THREADS"] = str(max_cpus)

def is_seed_chunk(chunk_path: str) -> bool:
    base = os.path.basename(chunk_path)
    return base.startswith("traj_0_window_0")


# ---------------------------
# GPU assignment (one per producer)
# ---------------------------
def get_available_gpu_count():
    import subprocess

    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
        n = len([line for line in out.strip().split("\n") if line.strip()])
        return max(n, 1)
    except Exception:
        return -1


available_gpus = get_available_gpu_count()
gpu_lock = threading.Lock()
gpu_counter = [0]


def assign_gpu():
    with gpu_lock:
        gid = gpu_counter[0] % available_gpus
        gpu_counter[0] += 1

    return gid

# ---------------------------
# Important Hyperparameters 
# ---------------------------
WINDOW_FRAMES = 50  # Size of the chunk 
num_chunks = 5000   # Chunks to run simulation for the producer 
alpha_CA = False    # Only considering CA atoms
checkpoint_every = 25 # checkpoitning: global state is saved after every 25 successful analyses

# Global state persistence
analysis_lock = threading.Lock()
def save_global_state(folder: str):
    snap = global_state.snapshot()

    tmp = os.path.join(folder, "variables_global.pkl.tmp")
    final = os.path.join(folder, "variables_global.pkl")

    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "models": snap["models"],
                "States": snap["states"],
                "Nmodels": snap["num_models"],
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(tmp, final)

def try_restore_global_state(folder: str):
    global_pkl = os.path.join(folder, "variables_global.pkl")

    if os.path.exists(global_pkl):
        print(f"[RESTART] Loading global_state from {global_pkl}")

        with open(global_pkl, "rb") as f:
            saved = pickle.load(f)

        try:
            with global_state.rw_lock.gen_wlock():
                global_state.models = saved.get("models", [])
                global_state.states = saved.get("States", [])
                global_state.num_models = saved.get(
                    "Nmodels",
                    len(global_state.models),
                )
        except Exception:
            global_state.restore(saved)

        print(
            f"[RESTART] Restored {len(global_state.states)} states "
            f"and {len(global_state.models)} models."
        )

        if len(global_state.models) > 0 and len(global_state.states) > 0:
            seed_ready.set()

        return True

    print(
        "[RESTART] No variables_global.pkl found. "
        "Proceeding with EMPTY global_state."
    )
    return False


_chunk_re = re.compile(r"_window_(\d+)\.dcd$")

def extract_chunk_number_from_path(p: str):
    m = _chunk_re.search(os.path.basename(p))
    return int(m.group(1)) if m else None

# Reads variables_<traj>.pkl and converts fcurrent (frame index) to chunk index.
def load_analysis_done_chunk(
    folder: str,
    traj_id: int,
    window_frames: int = WINDOW_FRAMES,
) -> int:
    """
    Read variables_<traj>.pkl and convert fcurrent to the completed chunk index.
    """
    local_file = os.path.join(folder, f"variables_{traj_id}.pkl")

    if not os.path.exists(local_file):
        return 0

    try:
        with open(local_file, "rb") as f:
            d = pickle.load(f)

        fcurrent = int(d.get("fcurrent", 0))
        return fcurrent // window_frames

    except Exception as e:
        print(
            f"[WARN] Could not read analysis progress "
            f"for traj {traj_id}: {e}"
        )
        return 0
    
def enqueue_backlog(
    q: "queue.Queue",
    folder: str,
    traj_id: int,
    analysis_done: int,
    sim_done: int,
):
    """
    Enqueue existing simulation chunks that have not yet been analyzed.
    """
    paths = glob.glob(
        os.path.join(folder, f"traj_{traj_id}_window_*.dcd")
    )

    items = []

    for p in paths:
        c = extract_chunk_number_from_path(p)

        if c is None:
            continue

        if not (analysis_done <= c < sim_done):
            continue

        try:
            if os.path.getsize(p) == 0:
                continue
        except OSError:
            continue

        items.append((c, p))

    for _, p in sorted(items):
        q.put(p)
        print(
            f"[BACKLOG] Enqueued existing chunk for analysis: "
            f"{os.path.basename(p)}"
        )

#Function to call Analysis as a part of consumers 
def analysis(chunk_path: str, pdb_file: str, pdb_id: int, folder: str, folder_traj: str):
    """
      - traj0 window0 is allowed to run immediately (it should build seed model/state)
      - any other traj's window0 waits for seed_ready
    """
    base = os.path.basename(chunk_path)

    # Gate only window_0 for non-traj0
    if base.startswith("traj_") and "_window_0" in base and not base.startswith("traj_0_window_0"):
        while not seed_ready.is_set():
            if seed_failed.is_set():
                print("[FATAL] Seed (traj_0_window_0) failed. Aborting dependent analyses.")
                return -1
            time.sleep(0.05)

    residue_file = f"{folder_traj}/residues.txt"
    
    
    try:
        new_state = main(str(chunk_path), pdb_file, residue_file, alpha_CA, folder, folder_traj, uncertainw=False)

        if is_seed_chunk(chunk_path):
            snap = global_state.snapshot()
            if len(snap["models"]) > 0 and len(snap["states"]) > 0:
                seed_ready.set()
            else:
                seed_failed.set()
                raise RuntimeError("Seed chunk finished but model0/state0 not created in global_state")

        return new_state
    except Exception as e:
        import traceback
        print(f"[ERROR] Analysis failed for {chunk_path}: {e}")
        traceback.print_exc()
        if is_seed_chunk(chunk_path):
            seed_failed.set()
        return -1

# ---------------------------
# Producer Method: Produces chucks from simulation
# ---------------------------
def producer(q: "queue.Queue", pdb_file: str, pdb_id: int, folder: str):
    gpu_id = assign_gpu()
    print(f"[PRODUCER] {pdb_file} -> gpu {gpu_id}")

    simulation = None
    traj_tag = f"traj_{pdb_id}"

    start_chunk = load_progress(traj_tag, folder)
    print(f"[RESUME] {traj_tag} starting at chunk {start_chunk}/{num_chunks}")

    for chunk in range(start_chunk, num_chunks):
        chunk_filename = f"{folder}/traj_{pdb_id}_window_{chunk}.dcd"

        simulation = run_simulation(
            chunk_filename=chunk_filename,
            chunk_number=chunk,
            pdbfile=pdb_file,
            pdb_id=traj_tag,
            simulation=simulation,
            gpu_id=gpu_id,
            folder=folder,
        )

        save_progress(traj_tag, folder, chunk + 1)
        q.put(chunk_filename)

    q.put("exit")

# ---------------------------
# Consumer Method: Analyses produced chucks from the simulation 
# ---------------------------
def consumer(q: "queue.Queue", pdb_file: str, pdb_id: int, folder: str, folder_traj: str):
    checkpoint_every = 25
    count = 0

    while True:
        item = q.get()
        if item == "exit":
            q.task_done()
            print(f"[CONSUMER] traj {pdb_id} exit")
            return

        ns = analysis(item, pdb_file, pdb_id, folder, folder_traj)
        q.task_done()

        # Optional: if seed failed, you might want to exit quickly
        if seed_failed.is_set():
            print(f"[CONSUMER] traj {pdb_id} stopping because seed_failed is set")
            return

        if ns != -1:
            count += 1
            if count % checkpoint_every == 0:
                with analysis_lock:
                    save_global_state(folder)

# ---------------------------
# Main runner
# ---------------------------
def run(folder_traj: str, folder: str):
    print("PULSE Pipeline is starting......")
    print("Folder storing files:", folder)

    os.makedirs(folder, exist_ok=True)

    for f in glob.glob(os.path.join(folder, "lock*.txt")):
        try:
            os.remove(f)
            print(f"Deleted lock file: {f}")
        except OSError:
            pass

    try_restore_global_state(folder)
    pdbs = sorted(glob.glob(os.path.join(folder_traj, "*.pdb")))

    if not pdbs:
        raise RuntimeError(f"No pdbs found in {folder_traj}")

    queues = []
    producers = []
    consumers = []
    for pdb_id, pdb_file in enumerate(pdbs):
        q = queue.Queue()
        queues.append(q)

        # enqueue backlog if simulation is ahead of analysis ----
        traj_tag = f"traj_{pdb_id}"
        sim_done = load_progress(traj_tag, folder)                 # simulation completed chunks
        analysis_done = load_analysis_done_chunk(folder, pdb_id)   # analysis completed chunks

        if analysis_done < sim_done:
            print(f"[RESUME] traj_{pdb_id}: analysis behind (analysis={analysis_done}, sim={sim_done}) -> enqueue backlog")
            enqueue_backlog(q, folder, pdb_id, analysis_done, sim_done)

        # Start threads
        pt = threading.Thread(target=producer, args=(q, pdb_file, pdb_id, folder), daemon=False)
        ct = threading.Thread(target=consumer, args=(q, pdb_file, pdb_id, folder, folder_traj), daemon=False)

        pt.start()
        ct.start()

        producers.append(pt)
        consumers.append(ct)

    # wait producers
    for pt in producers:
        pt.join()

    # wait queues drain
    for q in queues:
        q.join()

    # wait consumers
    for ct in consumers:
        ct.join()

    # final save
    with analysis_lock:
        save_global_state(folder)
    print("[DONE] Saved global variable state")

    if seed_failed.is_set():
        raise SystemExit("Seed failed (traj_0_window_0). Run aborted.")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python driver_baseline.py <trajectory_input_dir> <output_dir>"
        )

    run(sys.argv[1], sys.argv[2])