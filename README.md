# Hexapeptide Kinetic Transition Networks

Graph reduction and machine learning for hexapeptide energy landscapes at 300 K.

This repository studies kinetics across a collection of hexapeptide landscapes (my senior thesis!). Each local minimum becomes a network state. Transition states connect those minima. The resulting kinetic transition network, or KTN, is treated as a continuous-time Markov chain.

The main workflow builds microscopic KTNs from PATHSAMPLE files, reduces them with PyGT, checks the reduced kinetics, and tests whether network structure predicts kinetic behavior. The included validation table covers 43 landscapes from 30 sequences.

## Research focus

Large energy landscapes can contain tens of thousands of minima. Direct kinetic analysis is expensive and can be numerically delicate. Graph transformation removes low-priority states while retaining the A and B endpoint sets. The reduced network is then checked against the microscopic model.

Two analysis tracks follow:

- Classical models use sequence, distance, spectral, centrality, community, path, and topology features.
- Graph neural networks use node and edge features to predict committors, MFPTs, and graph-level targets.

## Network model

The code uses the column convention `i <- j`. For a rate matrix `K` and mean waiting times `tau`,

$$
B_{ij}=K_{ij}\tau_j, \qquad
Q_{ij}=K_{ij}\;(i\ne j), \qquad
Q_{jj}=-\tau_j^{-1}, \qquad
Q\pi=0.
$$

`A` and `B` are endpoint sets read from `min.A` and `min.B`. The forward committor is fixed to zero on A and one on B. Mean first-passage times, or MFPTs, measure the expected travel time between the sets.

## Repo guide

| Area | Main files | Role |
|---|---|---|
| KTN construction | `build_markov_model.py`, `io_markov.py`, `stationary_point_io.py` | Read PATHSAMPLE data and save sparse microscopic Markov models. |
| Reduction and checks | `build_gt_kept_models.py`, `mfpt_analysis.py`, `quantitative_keeplist_checks.py`, `ktn_utils.py` | Build GT-kept networks and test kinetic accuracy across reduction choices. |
| Feature models | `graph_features.py`, `ml_regression.py`, `ml_permutation_test.py` | Extract graph descriptors, fit regressors, and run uncertainty checks. |
| Graph learning | `ktn_dataset.py`, `gnn_models.py`, `train_gnn.py`, `train_gnn_v2.py` | Build PyG graphs and train node-level or graph-level GNNs. |
| Follow-up tests | `committor_linear_baseline.py`, `landscape_class_tests.py`, `gnn_ablation_sweep.py` | Compare baselines, landscape classes, and GNN settings. |
| Review | `thesis_analysis.ipynb`, `make_micro_report.py` | Collect results and produce figures, tables, and summaries. |

The `.sbatch` and `.slurm` files are Princeton Research Computing jobs for the longer runs.

## Setup

The pinned environment uses Python 3.11.15. A clean virtual environment is recommended.

```bash
git clone https://github.com/hareS1234/senior-thesis-code.git
cd senior-thesis-code

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Exact top-level versions are recorded in `requirements.txt`. The interpreter version is recorded in `.python-version`. PyGT is also available from the [PyGT source repository](https://github.com/tomswinburne/PyGT). PyTorch Geometric may need a platform-specific PyTorch build on GPU systems. Its [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) covers those cases.

Set `BASE_DIR` in `config.py` to the local PATHSAMPLE data root. Commands that accept `--root` can override it directly. The cluster scripts still contain the original Princeton paths. `prep_all_sequences.slurm` also accepts `THESIS_CODE_DIR` and `THESIS_DATA_ROOT` from the submission environment.

Run the small numerical test suite after setup:

```bash
python -m unittest discover -s tests -v
```

## Data layout

Each DPS directory needs `min.data` and `ts.data`. Kinetic endpoint calculations also need `min.A` and `min.B`.

```text
LAMMPS_uncapped/
└── aaaaaa_nocap/
    └── aaaaaa_99idps_nocap/
        ├── min.data
        ├── ts.data
        ├── min.A
        ├── min.B
        └── markov_T300K/
            ├── B_T300K.npz
            ├── K_T300K.npz
            ├── Q_T300K.npz
            ├── pi_T300K.npy
            ├── tau_T300K.npy
            └── GT_kept_T300K/
```

Raw PATHSAMPLE data and generated matrix files are not stored in this repository.

## First run

Define two paths for a single landscape:

```bash
DATA_ROOT=/path/to/LAMMPS_uncapped
DPS_DIR="$DATA_ROOT/aaaaaa_nocap/aaaaaa_99idps_nocap"
```

Build the microscopic model:

```bash
python build_markov_model.py --data-dir "$DPS_DIR" --T 300
```

Build every microscopic model with resume support:

```bash
python run_all_build.py --root "$DATA_ROOT" --temperatures 300
```

Build coarse models for every available landscape:

```bash
python build_gt_kept_models.py \
  --root "$DATA_ROOT" \
  --T 300 \
  --style hybrid \
  --percent-retained 5 \
  --min-kept 200 \
  --block 50 \
  --screen
```

Calculate microscopic and coarse kinetics for the selected landscape:

```bash
python mfpt_analysis.py --data-dir "$DPS_DIR" --T 300 --max-eigs 10
python mfpt_analysis.py --data-dir "$DPS_DIR" --T 300 --coarse --max-eigs 10
```

Validate all completed coarse models:

```bash
python analyze_micro_vs_coarse_T300K.py \
  --root "$DATA_ROOT" \
  --T 300 \
  --out-dir .
```

The committed validation report records 26 passing networks, 10 flagged networks, and 7 incomplete cases. Passing networks have close microscopic and coarse MFPTs under the configured tolerances.

Test sensitivity to the basin barrier cutoff:

```bash
python quantitative_keeplist_checks.py \
  --data-dir "$DPS_DIR" \
  --deltaE-grid 10,15,20,25,30,40 \
  --E-window 3 \
  --temperature 300 \
  --out-csv robustness_vs_deltaE.csv
```

Each cutoff gets its own PyGT reduction. The output records the requested and retained state counts, A-to-B and B-to-A MFPTs, and the leading relaxation times.

## Feature analysis

`graph_features.py` reads `BASE_DIR` from `config.py`. The lighter preset skips the most expensive feature groups.

```bash
python graph_features.py \
  --T 300 \
  --out graph_features_coarse_T300K_lite.csv \
  --lite \
  --resume

python ml_regression.py \
  --features-csv graph_features_coarse_T300K_lite.csv \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --out-dir ml_results_cpu \
  --targets log_MFPT_AB log_MFPT_BA log_t1 t1_over_t2
```

Statistical follow-ups are available through `ml_permutation_test.py` and `landscape_class_tests.py`.

## Graph learning

Build the cached PyTorch Geometric dataset:

```bash
python ktn_dataset.py \
  --root ktn_pyg_data_v2 \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --T 300
```

Run the lighter CPU comparison across GCN, GAT, and NNConv:

```bash
python train_gnn_v2.py \
  --root ktn_pyg_data_v2 \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --task committor \
  --top-k 20 \
  --conv-types gcn gat nnconv \
  --epochs 100 \
  --device cpu \
  --out-dir gnn_results_v2
```

`train_gnn.py` also supports graph-level prediction and node-to-graph multitask training. Run `python train_gnn.py --help` for the full set of options.

## Cluster runs

`prep_all_sequences.slurm` now builds both the microscopic and GT-kept models. Paths can be supplied at submission time:

```bash
export THESIS_CODE_DIR=/scratch/gpfs/JERELLE/harry/thesis_code
export THESIS_DATA_ROOT=/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped
sbatch prep_all_sequences.slurm
```

After that job finishes, continue with:

```bash
sbatch mfpt_micro_all.slurm
sbatch mfpt_coarse_all.slurm
sbatch run_GT_validation_T300K.sbatch
sbatch run_graph_features.sbatch
sbatch run_ml_regression.sbatch
sbatch run_gnn_v2.sbatch
```

Check the resource requests, email address, environment name, and absolute paths before submitting a job.

## Result review

Open `thesis_analysis.ipynb` after the CSV files and model outputs are ready:

```bash
jupyter lab thesis_analysis.ipynb
```

The notebook collects the landscape overview, kinetic observables, spectral results, graph features, regression results, GNN results, and final thesis figures.
