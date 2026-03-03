from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Root directory with sequence folders like aaaaaa_nocap, aaggaa_nocap, ...
BASE_DIR = Path("/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped")

SEQUENCE_GLOB = "*_nocap"
DPS_GLOB = "*_nocap"

# Subdirectory where Markov model lives
MARKOV_SUBDIR_TEMPLATE = "markov_{temp_tag}"   # e.g. markov_T300K

# Temperatures you actually analyse
TEMPERATURES: List[float] = [300.0]

N_EIGS: int = 20
MAX_SOURCES_FOR_FULL_MFPT: int | None = None
RNG_SEED: int = 1234


@dataclass
class MarkovFilePaths:
    base_dir: Path  # DPS dir
    T: float

    @property
    def temp_tag(self) -> str:
        return f"T{int(round(self.T))}K"

    @property
    def markov_dir(self) -> Path:
        return self.base_dir / MARKOV_SUBDIR_TEMPLATE.format(temp_tag=self.temp_tag)

    # microscopic matrices
    @property
    def B_path(self) -> Path:
        return self.markov_dir / f"B_{self.temp_tag}.npz"

    @property
    def K_path(self) -> Path:
        return self.markov_dir / f"K_{self.temp_tag}.npz"

    @property
    def Q_path(self) -> Path:
        return self.markov_dir / f"Q_{self.temp_tag}.npz"

    @property
    def tau_path(self) -> Path:
        return self.markov_dir / f"tau_{self.temp_tag}.npy"

    @property
    def pi_path(self) -> Path:
        return self.markov_dir / f"pi_{self.temp_tag}.npy"

    @property
    def energies_path(self) -> Path:
        return self.markov_dir / f"energies_{self.temp_tag}.npy"

    @property
    def entropies_path(self) -> Path:
        return self.markov_dir / f"entropies_{self.temp_tag}.npy"

    @property
    def retained_mask_path(self) -> Path:
        return self.markov_dir / f"retained_mask_{self.temp_tag}.npy"

    @property
    def orig_ids_path(self) -> Path:
        return self.markov_dir / f"original_min_ids_{self.temp_tag}.npy"

    @property
    def pygt_dir(self) -> Path:
        return self.markov_dir

    @property
    def barrier_matrix_path(self) -> Path:
        return self.markov_dir / f"barrier_matrix_{self.temp_tag}.npz"

    @property
    def summary_json_path(self) -> Path:
        return self.markov_dir / f"summary_{self.temp_tag}.json"


def iter_dps_dirs() -> list[Path]:
    dps_dirs: list[Path] = []
    for seq_dir in BASE_DIR.glob(SEQUENCE_GLOB):
        if not seq_dir.is_dir():
            continue
        for dps_dir in seq_dir.glob(DPS_GLOB):
            if not dps_dir.is_dir():
                continue
            if (dps_dir / "min.data").exists() and (dps_dir / "ts.data").exists():
                dps_dirs.append(dps_dir)
    return sorted(dps_dirs)
