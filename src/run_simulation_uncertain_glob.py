from openmm import *
from openmm.app import *
from openmm.unit import kelvin, picoseconds, nanometer, bar
from openmm.unit import *
import numpy as np
import os
def run_simulationU(chunk_filename, chunk_number, pdbfile, pdb_id, simulation=None, last_state=None, gpu_id=0):
    """
    Runs one simulation chunk for the given PDB input on the assigned GPU
    Returns updated (simulation, last_state) 

    Change the paths for the simulation files below
    """

    psf_file = '../baseline_data/glob_water/bbl8.psf'
    top_file = '../baseline_data/glob_water/all_top.rtf'
    par_file = '../baseline_data/glob_water/parameters_all36.prm'
    
    # --- simulation parameters ---
    temperature = 310 * kelvin
    pressure = 1.0 * bar                            
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

    xlen, ylen, zlen = 7.47, 7.87, 7.39  # default box dimensions
   
    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': str(gpu_id), 'Precision': 'mixed'}
    

    # ==============================================================
    # INITIALIZATION / RESTART HANDLING
    # ==============================================================
    if simulation is None:
        print(f"Initializing new simulation on GPU {gpu_id} for chunk {chunk_number}...")

        current_pdb_path = pdbfile
        current_box = (xlen, ylen, zlen)
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                psf = CharmmPsfFile(psf_file)
                pdb = PDBFile(current_pdb_path)
                psf.setBox(current_box[0]*nanometers, current_box[1]*nanometers, current_box[2]*nanometers)
                params = CharmmParameterSet(top_file, par_file)

                system = psf.createSystem(
                    params,
                    nonbondedMethod=nonbondedMethod,
                    nonbondedCutoff=nonbondedCutoff,
                    switchDistance=switchDistance,
                    constraints=constraints,
                    rigidWater=rigidWater,
                    ewaldErrorTolerance=ewaldErrorTolerance,
                    hydrogenMass=4*amu
                )

                system.addForce(MonteCarloBarostat(pressure, temperature))
                integrator = LangevinIntegrator(temperature, friction, dt)
                integrator.setConstraintTolerance(constraintTolerance)
                integrator.setRandomNumberSeed(1000 + pdb_id)
                simulation = Simulation(psf.topology, system, integrator, platform, properties)
                simulation.context.setPositions(pdb.positions)
                state = simulation.context.getState(getPositions=True)
                pos_array = np.array([[p.x, p.y, p.z] for p in state.getPositions()])
                if np.any(np.isnan(pos_array)):
                    raise ValueError("NaN detected in initial positions.")

                simulation.context.setVelocitiesToTemperature(temperature)
                simulation.minimizeEnergy(maxIterations=5000)
                # run a quick stability check
                simulation.step(100)
                state = simulation.context.getState(getPositions=True)
                pos_array = np.array([[p.x, p.y, p.z] for p in state.getPositions()])
                if np.any(np.isnan(pos_array)):
                    raise ValueError("NaN detected after initial steps!")
                break 
            except Exception as e:
                if attempt == 0:
                    current_box = (10.0, 10.0, 10.0)
                elif attempt == 1:
                    current_box = (50.0, 50.0, 50.0)
                else:  
                  raise e

                if 'simulation' in locals():
                    del simulation
                if 'integrator' in locals():
                    del integrator
                if 'system' in locals():
                    del system

    else:
        # resume from last state
        simulation.context.setState(last_state)

    # ==============================================================
    # REPORTERS: TRAJECTORY + CHECKPOINT
    # ==============================================================
    reporter = DCDReporter(chunk_filename, steps_per_frame)
    simulation.reporters.append(reporter)
  
    chk_path = os.path.join(os.path.dirname(chunk_filename), f"traj_{pdb_id}.chk")
    chk_reporter = CheckpointReporter(chk_path, total_steps)
    simulation.reporters.append(chk_reporter)
    # ==============================================================
    # RUN SIMULATION CHUNK
    # ==============================================================
    print(f"Running chunk {chunk_number} on GPU {gpu_id} ({total_steps} steps, 50 frames)")
    simulation.step(total_steps)
    simulation.reporters.remove(reporter)
    simulation.reporters.remove(chk_reporter)
    del reporter
    del chk_reporter


    print(f"Chunk {chunk_number} completed. Output: {chunk_filename}")

    # get final state for continuation
    last_state = simulation.context.getState(getPositions=True, getVelocities=True)

    return simulation, last_state