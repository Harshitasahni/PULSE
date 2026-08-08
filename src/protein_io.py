"""Topology loading and PDB output functions."""

import os
import random as _random

import mdtraj as md
import numpy as np

from openmm.app import CharmmPsfFile, PDBFile
from openmm.unit import nanometer

from protein_config import get_protein_config


def get_reference_topology(protein_name):
    config = get_protein_config(protein_name)
    topology_path = config["topology_path"]

    if config["topology_type"] == "pdb":
        return PDBFile(topology_path).topology

    if config["topology_type"] == "psf":
        psf = CharmmPsfFile(topology_path)
        box_dimensions = config["box_dimensions_nm"]

        if box_dimensions is not None:
            psf.setBox(
                *(dimension * nanometer for dimension in box_dimensions)
            )

        return psf.topology

    raise ValueError(
        f"Unsupported topology type: {config['topology_type']}"
    )


def save_uncertain_frame(
    protein_name,
    local_variables,
    fcurrent,
    traj_path,
    fmax,
    folder,
    traj_num,
    local_window,
):
    config = get_protein_config(protein_name)
    topology_path = config["topology_path"]

    local_variables["frames2states"][fcurrent] = [-1]

    traj = md.load(traj_path, top=topology_path)
    fmax_frame = fmax
    frame = traj[fmax_frame]

    uncertain_folder = os.path.join(
        folder,
        config["uncertain_folder"],
    )

    frame_name = (
        f"traj_{traj_num}_window_{local_window}_frame_{fmax_frame}"
    )
    frame_folder = os.path.join(uncertain_folder, frame_name)
    os.makedirs(frame_folder, exist_ok=True)

    output_pdb = os.path.join(
        frame_folder,
        f"{frame_name}.pdb",
    )

    if np.any(np.isnan(frame.xyz)):
        raise ValueError(
            f"NaNs found in frame {fmax_frame} of {traj_path}"
        )

    positions = frame.openmm_positions(0)
    topology = get_reference_topology(protein_name)
    topology_atom_count = len(list(topology.atoms()))

    if topology_atom_count != len(positions):
        raise ValueError(
            f"Atom mismatch: topology has {topology_atom_count} atoms, "
            f"but the frame has {len(positions)} positions"
        )

    with open(output_pdb, "w") as file:
        PDBFile.writeFile(
            topology,
            positions,
            file,
            keepIds=True,
        )

    print(f"[UNCERTAIN] Saved fmax frame: {frame_name}")

    return output_pdb, frame_name


def save_frame_only(
    protein_name,
    traj_path,
    fmax,
    folder,
    traj_num,
    local_window,
):
    config = get_protein_config(protein_name)
    topology_path = config["topology_path"]

    traj = md.load(traj_path, top=topology_path)
    fmax_frame = fmax
    frame = traj[fmax_frame]

    save_folder = os.path.join(
        folder,
        config["forced_reuse_folder"],
    )

    frame_name = (
        f"traj_{traj_num}_window_{local_window}_frame_{fmax_frame}"
    )
    frame_folder = os.path.join(save_folder, frame_name)
    os.makedirs(frame_folder, exist_ok=True)

    output_pdb = os.path.join(
        frame_folder,
        f"{frame_name}.pdb",
    )

    if np.any(np.isnan(frame.xyz)):
        raise ValueError(
            f"NaNs found in frame {fmax_frame} of {traj_path}"
        )

    positions = frame.openmm_positions(0)
    topology = get_reference_topology(protein_name)
    topology_atom_count = len(list(topology.atoms()))

    if topology_atom_count != len(positions):
        raise ValueError(
            f"Atom mismatch: topology has {topology_atom_count} atoms, "
            f"but the frame has {len(positions)} positions"
        )

    with open(output_pdb, "w") as file:
        PDBFile.writeFile(
            topology,
            positions,
            file,
            keepIds=True,
        )

    print(f"[FORCED-REUSE] Saved fmax frame: {frame_name}")

    return output_pdb, frame_name


def save_random_uncertain_frameio(
    protein_name,
    traj_path,
    folder,
    traj_num,
    local_window,
    fmax_frame,
):
    config = get_protein_config(protein_name)
    topology_path = config["topology_path"]

    traj = md.load(traj_path, top=topology_path)
    nframes = traj.n_frames

    if nframes <= 1:
        random_frame = 0
    else:
        candidates = list(range(nframes))

        if 0 <= fmax_frame < nframes:
            candidates.remove(fmax_frame)

        random_frame = _random.choice(candidates)

    frame = traj[random_frame]

    if np.any(np.isnan(frame.xyz)):
        raise ValueError(
            f"NaNs found in frame {random_frame} of {traj_path}"
        )

    positions = frame.openmm_positions(0)
    topology = get_reference_topology(protein_name)
    topology_atom_count = len(list(topology.atoms()))

    if topology_atom_count != len(positions):
        raise ValueError(
            f"Atom mismatch: topology has {topology_atom_count} atoms, "
            f"but the frame has {len(positions)} positions"
        )

    random_frames_folder = os.path.join(
        folder,
        config["uncertain_folder"],
        "random_frames",
    )
    os.makedirs(random_frames_folder, exist_ok=True)

    random_pdb_name = (
        f"traj_{traj_num}_window_{local_window}_"
        f"frame_{random_frame}.pdb"
    )
    output_pdb = os.path.join(
        random_frames_folder,
        random_pdb_name,
    )

    with open(output_pdb, "w") as file:
        PDBFile.writeFile(
            topology,
            positions,
            file,
            keepIds=True,
        )

    print(f"[RANDOM] Saved random frame: {random_pdb_name}")

    return output_pdb