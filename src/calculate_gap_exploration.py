import warnings

warnings.filterwarnings("ignore")

import glob
import os
from itertools import product

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pyemma
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


ORIGINAL_FOLDER = (
    "../"
    "Data_folder/"
)
UNCERTAIN_BASE_FOLDER = (
    "../"
)
TOP_FILE = (
    "../baseline_data/"
    "glob_water/bbl8-Copy1.pdb"
)
PRECOMPUTED_TICA_PATH = (
    "../Data_folder"
    "tica_projection.npy"
)
METHOD_PATHS = {
    "pulse": (
        "../Data_folder/"
        "pulse_latest/"
    )
}
METHODS = [
    "count_latest_glob",
    "kinetic_latest",
    "om_REAP_glob1",
    "pulse",
]
OUTPUT_FOLDER = "."
OUTPUT_SUFFIX = "plot"

TICA_LAG = 20
TICA_DIM = 3
SAVE_FIGURES = True


class StreamlinedGapAnalyzer:
    def __init__(
        self,
        original_folder: str,
        uncertain_base_folder: str,
        top_file: str,
        methods: list = None,
        precomputed_tica_path: str = None,
        method_paths: dict = None,
        output_folder: str = ".",
        output_suffix: str = "New2",
    ):
        self.original_folder = original_folder
        self.uncertain_base_folder = uncertain_base_folder
        self.top_file = top_file
        self.precomputed_tica_path = precomputed_tica_path
        self.method_paths = method_paths or {}
        self.methods = methods or [
            "count_based_villian",
            "kinetic_based_villian",
            "reap_based_villian",
        ]
        self.output_folder = output_folder
        self.output_suffix = output_suffix

        self.topo = md.load_topology(top_file)
        self._setup_featurizer()

        self.tica_model = None
        self.tica_original = None
        self.tica_uncertainty = {}

    def _setup_featurizer(self):
        loop1 = self.topo.select("residue 89 90 and name CA")
        loop2 = self.topo.select("residue 19 43 54 105 122 and name CA")
        atom_pairs = [pair for pair in product(loop1, loop2) if pair[0] < pair[1]]

        if not atom_pairs:
            raise ValueError(
                "No atom pairs found. Check the residue indices in _setup_featurizer()."
            )

        self.feat = pyemma.coordinates.featurizer(self.top_file)
        self.feat.add_distances(atom_pairs)

    def _collect_original_files(self) -> list:
        return sorted(glob.glob(os.path.join(self.original_folder, "*.dcd")))

    def _collect_uncertainty_files(self, method: str) -> list:
        if method in self.method_paths:
            spec = self.method_paths[method]
            pattern = spec if "*" in spec else os.path.join(spec, "*.dcd")
            return sorted(glob.glob(pattern))

        method_dir = os.path.join(self.uncertain_base_folder, method)
        if not os.path.isdir(method_dir):
            return []

        files = []
        for frame_dir in sorted(glob.glob(os.path.join(method_dir, "*"))):
            files.extend(sorted(glob.glob(os.path.join(frame_dir, "*.dcd"))))
        return files

    def compute_tica_original(self, original_files: list, lag: int = 20, dim: int = 3):
        src = pyemma.coordinates.source(
            original_files,
            features=self.feat,
            top=self.top_file,
        )
        self.tica_model = pyemma.coordinates.tica(src, lag=lag, dim=dim)

        if self.precomputed_tica_path is not None:
            self.tica_original = np.load(self.precomputed_tica_path)
        else:
            self.tica_original = np.vstack(self.tica_model.get_output())

        return self.tica_model

    def _project_uncertainty(self, files: list):
        if not files:
            return None

        projected = []
        for file_path in files:
            try:
                src = pyemma.coordinates.source(
                    file_path,
                    features=self.feat,
                    top=self.top_file,
                )
                for chunk in src.get_output():
                    if len(chunk) > 0:
                        projected.append(self.tica_model.transform(chunk))
            except Exception:
                continue

        if not projected:
            return None

        return np.vstack(projected)

    def _gap_metrics(self, tica_unc: np.ndarray, method_name: str = "") -> dict:
        if tica_unc is None or tica_unc.size == 0:
            return {}

        kde = gaussian_kde(self.tica_original[:, :2].T)
        density_unc = kde(tica_unc[:, :2].T)
        density_original = kde(self.tica_original[:, :2].T)
        threshold = np.percentile(density_original, 10)

        gap_mask = density_unc < threshold
        gap_rate = gap_mask.sum() / len(gap_mask)

        print(f"INFO-----------------------------------{method_name}: {gap_rate:.1%}")

        return {
            "fraction_in_gaps": gap_rate,
            "n_gap_frames": int(gap_mask.sum()),
            "total_frames": len(gap_mask),
            "mean_density": float(np.mean(density_unc)),
            "median_density": float(np.median(density_unc)),
            "mean_density_ratio": float(
                np.mean(density_unc) / np.mean(density_original)
            ),
            "low_density_threshold": float(threshold),
            "exploring_new_regions": gap_rate > 0.30,
            "_density_unc": density_unc,
            "_density_ori": density_original,
            "_gap_mask": gap_mask,
        }

    def _build_density_grid(self, kde, pad_frac=0.10, n_grid=100):
        x_min, x_max = self.tica_original[:, 0].min(), self.tica_original[:, 0].max()
        y_min, y_max = self.tica_original[:, 1].min(), self.tica_original[:, 1].max()
        x_pad = pad_frac * (x_max - x_min)
        y_pad = pad_frac * (y_max - y_min)
        x_values = np.linspace(x_min - x_pad, x_max + x_pad, n_grid)
        y_values = np.linspace(y_min - y_pad, y_max + y_pad, n_grid)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        density = kde(
            np.vstack([grid_x.ravel(), grid_y.ravel()])
        ).reshape(grid_x.shape)
        return grid_x, grid_y, density

    def create_plots(self, method: str, results: dict, save: bool = True):
        if not results:
            return

        uncertainty = self.tica_uncertainty.get(method)
        if uncertainty is None:
            return

        plt.rc("font", size=18)
        plt.rc("axes", titlesize=18, labelsize=18)
        plt.rc("xtick", labelsize=18)
        plt.rc("ytick", labelsize=18)
        plt.rc("legend", fontsize=18)
        plt.rc("figure", titlesize=18)

        kde = gaussian_kde(self.tica_original[:, :2].T)
        threshold = results["low_density_threshold"]
        gap_mask = results["_gap_mask"]
        grid_x, grid_y, density = self._build_density_grid(
            kde,
            pad_frac=0.12,
            n_grid=150,
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.contourf(grid_x, grid_y, density, levels=30, cmap="Greys", alpha=0.85)
        ax.contour(
            grid_x,
            grid_y,
            density,
            levels=[threshold],
            colors="#e74c3c",
            linestyles="--",
            linewidths=1.8,
        )

        if (~gap_mask).any():
            ax.scatter(
                uncertainty[~gap_mask, 0],
                uncertainty[~gap_mask, 1],
                c="#e67e22",
                s=6,
                alpha=0.45,
                linewidths=0,
                label=f"Uncertainty — dense ({(~gap_mask).sum():,})",
                zorder=3,
            )

        if gap_mask.any():
            ax.scatter(
                uncertainty[gap_mask, 0],
                uncertainty[gap_mask, 1],
                c="#2980b9",
                s=8,
                alpha=0.75,
                linewidths=0,
                label=f"Uncertainty — gap ({gap_mask.sum():,})",
                zorder=4,
            )

        ax.set_xlabel("tIC 1", fontsize=19)
        ax.set_ylabel("tIC 2", fontsize=19)
        ax.set_xlim(-5, 3)
        ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3])

        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="--",
                color="#e74c3c",
                linewidth=1.8,
                label="Gap boundary (10th pct)",
            )
        )
        labels.append("Gap boundary (10th pct)")
        ax.legend(
            handles=handles,
            labels=labels,
            fontsize=14,
            markerscale=2,
            loc="upper left",
            framealpha=0.9,
        )

        plt.tight_layout()

        if save:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = os.path.join(
                self.output_folder,
                f"gap_exploration_{method}_{self.output_suffix}.png",
            )
            plt.savefig(output_path, dpi=500, bbox_inches="tight")

        plt.show()

    def run_analysis(self, lag: int = 20, dim: int = 3) -> dict:
        original_files = self._collect_original_files()
        if not original_files:
            raise FileNotFoundError(
                f"No original DCD files found in: {self.original_folder}"
            )

        self.compute_tica_original(original_files, lag=lag, dim=dim)

        all_results = {}
        for method in self.methods:
            uncertainty_files = self._collect_uncertainty_files(method)
            tica_uncertainty = self._project_uncertainty(uncertainty_files)
            self.tica_uncertainty[method] = tica_uncertainty
            all_results[method] = self._gap_metrics(
                tica_uncertainty,
                method_name=method,
            )

        for method in self.methods:
            self.create_plots(
                method,
                all_results.get(method, {}),
                save=SAVE_FIGURES,
            )

        clean_results = {
            method: {
                key: value
                for key, value in method_results.items()
                if not key.startswith("_")
            }
            for method, method_results in all_results.items()
        }

        return {
            "tica_model": self.tica_model,
            "gap_results": clean_results,
            "original_shape": (
                self.tica_original.shape
                if self.tica_original is not None
                else None
            ),
            "uncertainty_shapes": {
                method: values.shape
                for method, values in self.tica_uncertainty.items()
                if values is not None
            },
        }


if __name__ == "__main__":
    analyzer = StreamlinedGapAnalyzer(
        original_folder=ORIGINAL_FOLDER,
        uncertain_base_folder=UNCERTAIN_BASE_FOLDER,
        top_file=TOP_FILE,
        methods=METHODS,
        precomputed_tica_path=PRECOMPUTED_TICA_PATH,
        method_paths=METHOD_PATHS,
        output_folder=OUTPUT_FOLDER,
        output_suffix=OUTPUT_SUFFIX,
    )
    results = analyzer.run_analysis(lag=TICA_LAG, dim=TICA_DIM)