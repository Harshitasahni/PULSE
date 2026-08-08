from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.app import CharmmPsfFile, CharmmParameterSet, PDBFile, DCDReporter, Simulation
import os
import threading
import subprocess
import numpy as np


# Detect available GPUs
def get_available_gpu_count():
    try:
        output = subprocess.check_output(['nvidia-smi', '-L']).decode()
        return len(output.strip().split('\n'))
    except:
        return 1


available_gpus = get_available_gpu_count()
gpu_lock = threading.Lock()
gpu_counter = [0]


def assign_gpu():
    with gpu_lock:
        gpu_id = gpu_counter[0] % available_gpus
        gpu_counter[0] += 1
    return gpu_id


# ---------------------------
# Progress file helpers 
# ---------------------------
def load_progress(traj_tag, folder):
    """Read the next chunk to run from <traj_tag>.progress; returns 0 if none."""
    progress_file = os.path.join(folder, f"{traj_tag}.progress")
    if os.path.exists(progress_file):
        try:
            with open(progress_file) as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return 0
    return 0


def save_progress(traj_tag, folder, chunk_number):
    """Atomically write the next chunk number to <traj_tag>.progress."""
    progress_file = os.path.join(folder, f"{traj_tag}.progress")
    tmp = progress_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(chunk_number))
    os.replace(tmp, progress_file)


# ---------------------------
# Simulation Block
# ---------------------------
def run_simulation(chunk_filename, chunk_number, pdbfile, pdb_id,
                   simulation=None, gpu_id=0, folder=None):
    psf_file = '../data/glob_water/bbl8.psf'
    top_file = '../data/glob_water/all_top.rtf'
    par_file = '../data/glob_water/parameters_all36.prm'

    temperature = 310 * kelvin
    pressure = 1.01325 * bar
    nonbondedMethod = PME
    nonbondedCutoff = 12 * angstroms
    switchDistance = 10 * angstroms
    ewaldErrorTolerance = 0.0005
    constraints = HBonds
    rigidWater = True
    constraintTolerance = 1e-6
    dt = 0.001 * picoseconds
    friction = 1.0 / picosecond
    frames_per_window = 50
    steps_per_frame = 1000
    total_steps = frames_per_window * steps_per_frame
    xlen, ylen, zlen = 7.47, 7.87, 7.39

    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}

    # pdb_id passed in from pipeline is traj_tag like "traj_0"
    chk_file = os.path.join(folder, f"{pdb_id}.chk") if folder else None

    if simulation is None:
        print(f"\n Initializing new simulation on GPU {gpu_id} for chunk {chunk_number}...")
        psf = CharmmPsfFile(psf_file)
        pdb = PDBFile(pdbfile)
        psf.setBox(xlen*nanometers, ylen*nanometers, zlen*nanometers)
        params = CharmmParameterSet(top_file, par_file)
        system = psf.createSystem(params,
                                  nonbondedMethod=nonbondedMethod,
                                  nonbondedCutoff=nonbondedCutoff,
                                  switchDistance=switchDistance,
                                  constraints=constraints,
                                  rigidWater=rigidWater,
                                  ewaldErrorTolerance=ewaldErrorTolerance,
                                  hydrogenMass=4*amu)
        system.addForce(MonteCarloBarostat(pressure, temperature))
        integrator = LangevinIntegrator(temperature, friction, dt)
        integrator.setConstraintTolerance(constraintTolerance)

        simulation = Simulation(psf.topology, system, integrator, platform, properties)
        simulation.context.setPositions(pdb.positions)

        # Sanity check positions
        state = simulation.context.getState(getPositions=True)
        test_positions = state.getPositions()
        pos_array = np.array([[pos.x, pos.y, pos.z] for pos in test_positions])
        if np.any(np.isnan(pos_array)):
            raise ValueError("NaN in initial positions!")

        if chk_file and os.path.exists(chk_file) and chunk_number > 0:
            print(f"[RESUME] Loading checkpoint {chk_file} for chunk {chunk_number}")
            with open(chk_file, "rb") as f:
                simulation.context.loadCheckpoint(f.read())
        elif chunk_number == 0:
            simulation.context.setVelocitiesToTemperature(temperature)
            print("Minimizing energy...")
            simulation.minimizeEnergy(maxIterations=5000)
        else:
            print(f"[WARN] chunk_number={chunk_number} but no checkpoint found at {chk_file}. "
                  f"Starting from fresh positions; results may differ.")
            simulation.context.setVelocitiesToTemperature(temperature)

    # Run the chunk
    reporter = DCDReporter(chunk_filename, steps_per_frame)
    simulation.reporters.append(reporter)
    simulation.step(total_steps)
    simulation.reporters.remove(reporter)

    # --- SAVE CHECKPOINT AFTER EACH CHUNK ---
    if chk_file:
        tmp = chk_file + ".tmp"
        with open(tmp, "wb") as f:
            f.write(simulation.context.createCheckpoint())
        os.replace(tmp, chk_file)

    print(f" Chunk {chunk_number} completed. Output: {chunk_filename}")
    return simulation