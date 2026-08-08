from openmm import *
from openmm.app import *
from openmm.unit import kelvin, picoseconds, nanometer, bar
from openmm.unit import *
import numpy as np
import os


def _ensure_box_vectors(topology, box_dims_nm):
    """Force orthorhombic periodic box vectors onto a topology."""
    box_nm = [float(x) for x in box_dims_nm]
    a = Vec3(box_nm[0], 0.0, 0.0) * nanometers
    b = Vec3(0.0, box_nm[1], 0.0) * nanometers
    c = Vec3(0.0, 0.0, box_nm[2]) * nanometers
    topology.setPeriodicBoxVectors((a, b, c))


def run_simulationU(chunk_filename, chunk_number, pdbfile, pdb_id, simulation=None, last_state=None, gpu_id=0):
    """
    Runs one simulation chunk for the given PDB input on the assigned GPU
    Returns updated (simulation, last_state) 

    Change the paths for the simulation files below
    """

    box_dims_nm = (4.9163, 4.5981, 3.8869)  # Villin default box dimensions

    # --- simulation parameters ---
    temperature     = 300 * kelvin
    pressure        = 1.0 * bar
    nonbondedMethod = PME
    nonbondedCutoff = 1.0 * nanometer
    constraints     = HBonds
    rigidWater      = True
    constraintTolerance = 1e-6
    dt              = 0.002 * picoseconds      
    friction        = 1.0 / picosecond

    frames_per_window = 50
    steps_per_frame   = 1000
    total_steps       = frames_per_window * steps_per_frame 
    platform   = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}

    # ==============================================================
    # INITIALIZATION / RESTART HANDLING
    # ==============================================================
    if simulation is None:
        print(f"Initializing new simulation on GPU {gpu_id} for chunk {chunk_number}...")

        current_box  = box_dims_nm
        box_attempts = [box_dims_nm, (10.0, 10.0, 10.0)]

        for attempt, current_box in enumerate(box_attempts, 1):
            try:
                pdb = PDBFile(pdbfile)

                # Ensure periodic box vectors are set
                if pdb.topology.getPeriodicBoxVectors() is None:
                    _ensure_box_vectors(pdb.topology, current_box)

                forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

                system = forcefield.createSystem(
                    pdb.topology,
                    nonbondedMethod=nonbondedMethod,
                    nonbondedCutoff=nonbondedCutoff,
                    constraints=constraints,
                    rigidWater=rigidWater,
                )

                system.addForce(MonteCarloBarostat(pressure, temperature))

                integrator = LangevinMiddleIntegrator(temperature, friction, dt)
                integrator.setConstraintTolerance(constraintTolerance)

                simulation = Simulation(pdb.topology, system, integrator, platform, properties)
                simulation.context.setPositions(pdb.positions)

                # NaN check on initial positions
                state     = simulation.context.getState(getPositions=True)
                pos_array = np.array([[p.x, p.y, p.z] for p in state.getPositions()])
                if np.any(np.isnan(pos_array)):
                    raise ValueError("NaN detected in initial positions.")

                simulation.context.setVelocitiesToTemperature(temperature)
                simulation.minimizeEnergy(maxIterations=10000)

                print(f"[EQUIL] Running 20 ps NPT equilibration (attempt {attempt})...")
                equil_steps = int((20 * picoseconds) / dt)  # 10,000 steps
                simulation.step(equil_steps)

                state     = simulation.context.getState(getPositions=True)
                pos_array = np.array([[p.x, p.y, p.z] for p in state.getPositions()])
                if np.any(np.isnan(pos_array)):
                    raise ValueError("NaN detected after equilibration!")

                
                break  

            except Exception as e:
                if attempt < len(box_attempts):
                    next_box = box_attempts[attempt]
                    
                else:
                    raise e
                for obj in ('simulation', 'integrator', 'system'):
                    if obj in locals():
                        del locals()[obj]

    else:
        # Resume from last saved state
        simulation.context.setState(last_state)

    # ==============================================================
    # REPORTERS: TRAJECTORY + CHECKPOINT
    # ==============================================================
    reporter = DCDReporter(chunk_filename, steps_per_frame)
    simulation.reporters.append(reporter)

    chk_path     = os.path.join(os.path.dirname(chunk_filename), f"traj_{pdb_id}.chk")
    chk_reporter = CheckpointReporter(chk_path, total_steps)
    simulation.reporters.append(chk_reporter)

    # ==============================================================
    # RUN SIMULATION CHUNK
    # ==============================================================
    print(f"Running chunk {chunk_number} on GPU {gpu_id} ({total_steps} steps, {frames_per_window} frames)")
    simulation.step(total_steps)

    simulation.reporters.remove(reporter)
    simulation.reporters.remove(chk_reporter)
    del reporter
    del chk_reporter

    print(f"Chunk {chunk_number} completed. Output: {chunk_filename}")

    # Save final state for continuation
    last_state = simulation.context.getState(getPositions=True, getVelocities=True)

    return simulation, last_state