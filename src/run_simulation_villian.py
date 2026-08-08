# ===== run_simulation_villian.py =====
# Villin headpiece production MD (AMBER14 + TIP3P-FB)
# Chunked runner with safe cross-job restart via checkpoint + progress files.

import os
import numpy as np
from openmm import Platform, Vec3
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.app import (
    PDBFile, ForceField, Simulation,
    DCDReporter, CheckpointReporter, StateDataReporter,
)
from openmm.app import PME, HBonds
from openmm.unit import kelvin, picosecond, picoseconds, bar, nanometer, nanometers


# ==========================================================
# Progress / checkpoint file helpers
# ==========================================================
def checkpoint_path(pdb_id, folder):
    return os.path.join(folder, f"{pdb_id}.chk")


def progress_path(pdb_id, folder):
    return os.path.join(folder, f"{pdb_id}.progress")


def load_progress(pdb_id, folder):
    p = progress_path(pdb_id, folder)
    if os.path.exists(p):
        s = open(p).read().strip()
        if s:
            return int(s)
    return 0


def save_progress(pdb_id, folder, next_chunk_number):
    with open(progress_path(pdb_id, folder), "w") as f:
        f.write(str(int(next_chunk_number)))


def _ensure_box_vectors(topology, box_dims_nm):
    """Force orthorhombic periodic box vectors onto a topology, taken from PDB"""
    box_nm = [float(x) for x in box_dims_nm]
    a = Vec3(box_nm[0], 0.0, 0.0) * nanometers
    b = Vec3(0.0, box_nm[1], 0.0) * nanometers
    c = Vec3(0.0, 0.0, box_nm[2]) * nanometers
    topology.setPeriodicBoxVectors((a, b, c))


# ==========================================================
# Main runner
# ==========================================================
def run_simulation(
    chunk_filename,
    chunk_number,
    pdbfile,
    pdb_id,
    simulation=None,
    gpu_id=0,
    folder=".",
    box_dims_nm=None,
):
    # --------------------------------------------------------
    # Physical parameters
    # --------------------------------------------------------
    temperature     = 300 * kelvin
    friction        = 1.0 / picosecond
    dt              = 0.002 * picoseconds    # 2 fs (HBonds constrained)
    pressure        = 1.0 * bar

    nonbondedMethod = PME
    nonbondedCutoff = 1.0 * nanometer
    constraints     = HBonds
    rigidWater      = True
    constraintTolerance = 1e-6

    # --------------------------------------------------------
    # Sampling parameters
    # --------------------------------------------------------
    frames_per_window = 50                            # frames per chunk
    steps_per_frame   = 1000                          # DCD write every 2 ps
    total_steps       = frames_per_window * steps_per_frame  
    equilibration_ps       = 100                      # fresh-start equil
    checkpoint_interval_steps = 10000                 # ~20 ps, worst-case loss window

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------
    ckpt_file = checkpoint_path(pdb_id, folder)
    log_file  = os.path.join(folder, f"{pdb_id}.log")

    # --------------------------------------------------------
    # Platform
    # --------------------------------------------------------
    platform   = Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": str(gpu_id), "Precision": "mixed"}

    # ==========================================================
    # ONE-TIME INITIALIZATION (first chunk of a job)
    # ==========================================================
    if simulation is None:
        print(f"\nInitializing simulation on GPU {gpu_id} for chunk {chunk_number} (id={pdb_id})...")

        pdb = PDBFile(pdbfile)

        # PME requires periodic box vectors.
        # Use PDB CRYST1 if present. Otherwise require explicit box_dims_nm.
        if pdb.topology.getPeriodicBoxVectors() is None:
            if box_dims_nm is None:
                raise ValueError(
                    f"PME requires periodic box. {pdbfile} has no CRYST1 record "
                    f"and box_dims_nm was not provided. Fix the input PDB — "
                    f"no silent box fallback is allowed."
                )
            _ensure_box_vectors(pdb.topology, box_dims_nm)

        forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=constraints,
            rigidWater=rigidWater,
        )

        # Barostat must be on the system BEFORE Simulation is constructed
        # (required for consistent NPT, same for fresh start and resume)
        system.addForce(MonteCarloBarostat(pressure, temperature))

        integrator = LangevinMiddleIntegrator(temperature, friction, dt)
        integrator.setConstraintTolerance(constraintTolerance)

        simulation = Simulation(pdb.topology, system, integrator, platform, properties)

        # Safety: atom-count match
        if len(pdb.positions) != system.getNumParticles():
            raise ValueError(
                f"Atom mismatch: PDB has {len(pdb.positions)} atoms, "
                f"system expects {system.getNumParticles()} particles."
            )

        simulation.context.setPositions(pdb.positions)

        # NaN guard on initial positions
        state     = simulation.context.getState(getPositions=True)
        pos_array = np.array([[p.x, p.y, p.z] for p in state.getPositions()])
        if np.any(np.isnan(pos_array)):
            raise ValueError(f"NaN detected in initial positions for {pdb_id}")

        # ----------------------------------------------------
        # Resume from checkpoint, or do fresh minimize + equil
        # ----------------------------------------------------
        if os.path.exists(ckpt_file):
            print(f"Loading checkpoint: {ckpt_file}")
            simulation.loadCheckpoint(ckpt_file)
        else:
            print("Minimizing energy (maxIterations=10000)...")
            simulation.minimizeEnergy(maxIterations=10000)

            # Velocities set AFTER minimize (correct Maxwell-Boltzmann init)
            simulation.context.setVelocitiesToTemperature(temperature)

            # NPT equilibration, barostat already present on system
            print(f"Equilibrating {equilibration_ps} ps NPT...")
            simulation.step(int((equilibration_ps * picoseconds) / dt))

            simulation.saveCheckpoint(ckpt_file)

        # ----------------------------------------------------
        # Persistent reporters (stay across chunks)
        # ----------------------------------------------------
        simulation.reporters.append(
            CheckpointReporter(ckpt_file, checkpoint_interval_steps)
        )
        simulation.reporters.append(
            StateDataReporter(
                log_file,
                reportInterval=5000,       
                step=True,
                time=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                append=True,               # critical: preserve log across restarts
            )
        )

    # ==========================================================
    # CHUNK-SPECIFIC DCD REPORTER (one per chunk)
    # ==========================================================
    dcd = DCDReporter(chunk_filename, steps_per_frame)
    simulation.reporters.append(dcd)

    print(f"Running chunk {chunk_number} on GPU {gpu_id} "
          f"({total_steps} steps = 100 ps)")

    try:
        simulation.step(total_steps)
    except Exception:
        try:
            simulation.saveCheckpoint(ckpt_file)
        except Exception:
            pass
        raise
    finally:
        if dcd in simulation.reporters:
            simulation.reporters.remove(dcd)

    simulation.saveCheckpoint(ckpt_file)
    print(f"Chunk {chunk_number} completed. Output: {chunk_filename}")

    return simulation