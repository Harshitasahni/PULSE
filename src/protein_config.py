"""
Protein-specific topology and output configuration: hardcoded paths
Update the paths below to match your local system

"""

PROTEIN_CONFIGS = {
    "villian": {
        "topology_type": "pdb",
        "topology_path": (
            "../baseline_data/villian"
            "/villin.pdb"
        ),
        "box_dimensions_nm": None,
        "uncertain_folder": "uncertain_descendant",
        "forced_reuse_folder": "forced_reuse_descendant",
    },
    "glob": {
        "topology_type": "psf",
        "topology_path": (
            "../baseline_data/glob_water"
            "/bbl8.psf"
        ),
        "box_dimensions_nm": (7.47, 7.87, 7.39),
        "uncertain_folder": "uncertain_descendant",
        "forced_reuse_folder": "forced_reuse_descendant",
    },
}


def get_protein_config(protein_name):
    if protein_name not in PROTEIN_CONFIGS:
        raise ValueError(
            f"Unknown protein configuration: {protein_name}"
        )

    return PROTEIN_CONFIGS[protein_name]