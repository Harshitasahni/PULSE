# :trident: PULSE: Parallel Uncertainty-guided Landscape Sampling Engine

A scalable in-situ analysis framework for adaptive sampling of molecular dynamics (MD) simulations. PULSE couples online state discovery with uncertainty-driven adaptive sampling through a synchronized producer-consumer architecture, enabling real-time analysis of streaming trajectory data and dynamic redirection of compute toward unexplored conformational regions.

## Overview :beginner:
PULSE is data-agnostic. Use a different Simulator or tune analysis for your data. 

For Protein Analysis: 
PULSE addresses a core inefficiency in MD-based conformational sampling: simulations spend most of their time re-exploring already-characterized states. Rather than running fixed-length trajectories from pre-selected starting points, PULSE:

1. **Streams** trajectory windows from concurrent OpenMM simulations into a shared buffer
2. **Analyzes** windows in parallel using Online Boosted Gaussian Learners 
3. **Decides** in real time whether each trajectory has reached a well-characterized state (terminate) or an ambiguous region (spawn new simulation)
4. **Reallocates** compute from converged regions toward frontiers of conformational space



## 1. Repository structure :anchor:

```
PULSE/
├── README.md
├── LICENSE
├── baseline_data
├── sample_data
├── conda_analysis.yml            # Conda environment #1 
├── conda_openmm.yml              # Conda environment #2 
├── src/
│   ├── execution py files 
├── Zenodo traj.dcd files
├── Data_folder                   #  Reap/Count/Kinetic/Pulse .zip files from Zenodo unzipped here along with 4 trajectories files (.dcd extension)
│   ├── traj_1_merged.dcd
│   ├── pulse_latest
│   ├── kinectic_latest
│   ├── count_latest_glob
│   ├── om_REAP_glob1
│   ├── traj.npy



```

## 1.2 Dataset and Directory Structure 

We provide all the sample input data needed in the below DOI. We request you to pay close attention to this section as this is critical step. 
- Download the data into the PULSE directory your final directory structure should look like above.
- We provide datasets and neccessary files in zenodo DOI: [10.5281/zenodo.21815948](https://zenodo.org/records/21815948)
- 4 .dcd files in the root level of the DOI each of ~2.2GB are pre-simulated baselines trajectory co-ordinates
- The dataset also includes 4 zip files that contain pre-simulated trajectories along with their windows in .dcd extension for 4 methods that are used of comparison.
- Unzip the files in the parent folder (PULSE) 
- 4 Trajectory files (.dcd extendsion) also need to be inside Data_folder



## 2. Requirements :roller_coaster:

- Python 3.9+
- OpenMM 8.0+ (for MD simulation)
- PyEMMA 2.5+ (for TICA, MSM, clustering)
- MDTraj 1.9+
- NumPy (version 1.26.4) , SciPy, scikit-learn, matplotlib

> Please note that numpy version is very critical. Please make sure to double check this. 

**Note**: PyEMMA and OpenMM have conflicting dependency pins. PULSE uses two separate conda environments:

- `openmm` — OpenMM-based environment for simulation
- `analysis` — PyEMMA-based environment for online state discovery

The producer (simulation) and consumer (analysis) threads load only the modules from their respective environments.

## 3. Installation :passport_control:

```bash
git clone https://github.com/Harshitasahni/PULSE
cd PULSE
```

### 3.1 Create Anaconda Environment

```bash
conda env create --name openmm -f conda_openmm.yml
conda activate pulse_sim
```

### 3.2 Analysis environment

```bash
conda env create --name analysis -f conda_analysis.yml
conda activate pulse_analysis

```


> If using any other virtual environment like python venv please update your steps in batch job files as well. 

## 4. Quickstart :link:
 
### 4.1 Prepare a baseline trajectory

PULSE requires a short baseline simulation of the system being studied. Use any MD package; an example with OpenMM:

```bash
python src/driver_baseline.py \
    --pdb_dir baseline_data/protein_type \ # example glob_water for beta-lactoglobulin
    --total_time_ns 200 \                  #time for simulation
    --output_dir baseline_results_protein_type/
```

### 4.2 Run PULSE adaptive sampling

```bash
python src/driver_uncertain.py \
    --root baseline_results_protein_type/ \
    --total_ns 100 \
    --active 2 \
    --num_seeds 10  # or use the folders names to restart for selected seed
``` 

Key arguments:
- `--root`: working directory containing seed PDBs and analysis state
- `--total_ns`: aggregate simulation budget in nanoseconds
- `--active`: number of concurrent trajectories (typically = number of GPUs)
- `--num_seeds`: exact seed structures to spawn from


## 5. How PULSE works :paperclip: 

### 5.1 Producer-consumer architecture

```
┌─────────────┐                        ┌─────────────┐
│  Producer   │   trajectory windows   │  Consumer   │
│ (OpenMM MD) │──────────────────────▶│  analysis    │
│             │                        │             │
└─────────────┘                        └──────┬──────┘
       ▲                                      │
       │   uncertainty signal                 │
       └──────────────────────────────────────┘
            (terminate / continue / spawn)
```

- **Producers** run OpenMM simulations in independent threads, each on its own GPU
- **Consumers** retrieve trajectory windows from a shared queue and run incremental state discovery
- The consumer returns one of three signals: `STATE` (known state, continue), `NEW STATE` (new state found, continue), or `STOP` (terminate; budget exhausted or saturation)
- When a frame is flagged as uncertain, the consumer saves it as a seed PDB to spawn a new trajectory

### 5.2 Adaptive termination

Each trajectory runs for at most `initial_windows + extension_windows` windows (each window is 50 frames × 1000 steps = 100 ps at a 2 fs timestep). After `initial_windows`, the trajectory stops if no new state has been found; otherwise it continues for `extension_windows` more.

### 5.3 Uncertainty-driven spawning

Frames flagged as uncertain (max reconstruction error) are saved as new seed PDBs. The driver picks these up and launches new trajectories from them, creating a multi-generation tree of adaptive exploration.


## 6. Citation

If you use PULSE in your research, please cite:

https://github.com/Harshitasahni/PULSE



## 7. Gap Exploration and Analysis :microscope: 

### 7.1 Requirements 

For running scaling experiments for through and GSM coordination time, we require servers with below specifications: 
- CPU server with at least 10 cores and 30GB of RAM for heavy tICA computation 

### 7.2 Steps 
We provide a python file compute_tica.py which can be run using the anaconda/virtual environment describe in Section 3. 

- Activate the conda environment analysis as described in the Section 3.2.
- Make sure you have downloaded the datasets as described in Section 1.2
- Run command: 
```bash
python calculate_gap_exploration.py  #Change the path in the script(line #17-38)
```
> The python script assumes a fixed directory structure for input datasets and configuration.

### 7.3 Inputs 

We provide precomputed tICA coordinates (gray dots in Figure 2 and 3) along with python scripts that calculates gap exploration rates reported in Figure 3 and Table 1. We provide data folders that contains .dcd files or precomputed simulations for each method in (count|kinetic|reap|pulse) for $\beta$-lactoglobulin viral system.  


### 7.4 Outputs 

The output from task are 1) graphs (equivalent to Figure 3) for each individual method along with 2) corresponding Gap exploration rate, projected onto the two leading tICs of the baseline trajectories (gray dots). We also print overall gap exploration rate on screen for each of the individual method. Please note, the metrics reported in Table 1 are a mean over 3 runs. 


## 8. Performance Modeling :dart:

### 8.1 Requirements 

For running these experiments to capture GSM coordination time, we used TAMU Aces Cluster with below specifications: 
- At least 8 cores CPU 8 NVIDIA H100 80GB GPUs. 
- RedHat Enterprise Linux 8 
- 250 GB of RAM 

> Please note above requirements are needed to complete reproduction. Using a different GPU version with memory size may provide deviations in performance metrics. More details of our system are available [here](https://hprc.tamu.edu/kb/User-Guides/ACES/Hardware/), refer Sapphire Rapids Nodes. 

### 8.2 Steps 
We provide easy and ready batch job file [submit_pulse_metrics.sh](src/submit_pulse_metrics.sh), before submitting, make below changes:
- Update/remove steps from line 21-30 for environment setup. (use environment openmm as described in Section 3.1)
- Change the ACTIVE_LEVELS variable from 1-8 (space separated) 

It shall take 100-150 minutes for each entry in ACTIVE_LEVELS to complete. 

Submit the batch job as: ```sbatch submit_pulse_metrics.sh``` 
> Same batch job can be run directly if you are not using a scheudler like ```bash submit_pulse_gsm_scaling.sh```. 

### 8.3 Inputs 

Folder sample_data has simulation for 8 starting structures (20 windows for each) to record scaling metrics. 


### 8.4 Outputs 

Each batch job for ACTIVE_LEVELS will create output folder with name active(N)_(p), where N is the number define in ACTIVE_LEVELS and p is the iteration. Each folder will have intrumentation_logs file that captures system metrics in a csv file. The time to run each experiment varies. These csv log file can then be used to plot graphs that show metrics reported in Figures 4, 5 & 6. 

### 8.5 Plotting Graphs 

To plot throughput metrics on a graph run ```python plot_scaling_gsm.py ``` command. 

## 9. Synthetic scaling with up to 32 concurrent trajectories :chart_with_upwards_trend:

### 9.1 Requirements 

For running scaling experiments for through and GSM coordination time, we require servers with below specifications: 
- CPU server with at least 10 cores and 30GB of RAM. 

### 9.2 Steps 
We provide easy and ready batch job file [submit_scaling32.sh](src/submit_scaling32.sh), before submitting, make below changes:
- Update/remove steps from line 13-21 for environment setup.  (use environment openmm as described in Section 3.1)
- Change root directory on line #25
- Change output directory on line #27
- Change the ACTIVE_LEVELS variable from 1-32 (space separated) 
- Change iterations on line #32

Submit the batch job as: ```sbatch submit_scaling32.sh``` 

> Same batch job can be run directly if you are not using a scheudler like ```bash submit_scaling32.sh```. 

### 9.3 Inputs 

Folder scaling_data has simulation for 8 starting structures (20 windows for each) to record scaling metrics. 


### 9.4 Outputs 

The batch job will create folder with name active(N)_run(p), where N is the number define in ACTIVE_LEVELS and p is the iteration. Each folder will have intrumentation_logs file that captures system metrics in a csv file. The time to run each experiment varies and increases with the number of ACTIVE_LEVELS. For a full run that contains all levels from 1 to 32. A single run with all 32 levels is expected to complete in ~7 Hours.  

### 9.5 Plotting Graphs 

To plot throughput metrics on a graph run ```python plot_scaling.py --input_dir /path/to/output``` command. ```--input-dir``` provide the folder that contains sub-folders  active(N)_run(p). The file will plot only those ACTIVE_LEVELS for which data is avaiable. 

## 10. Acknowledgments :clap:
We would like to acknowledge the resources provided by UNM CARC and ACCESS-CI grant CIS240709.

