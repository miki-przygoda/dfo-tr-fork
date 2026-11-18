# DFO-TR fork for CT reconstruction

THIS IS A PUBLIC REPOSITORY FOR EDUCATIONAL PURPOSES. IT IS A PART OF A MODULE COURSEWORK. THE CODE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. THIS WAS CREATED FOR THE PURPOSE OF LEARNING AND FOR A MODULE COURSEWORK. -- COMP1805

This is a fork of the DFO-TR code from the paper. I've added a few features (e.g., a better initialization and a different disturbance rate. The code is structured to allow easy experimentation with different search strategies. Code was written in Python 3.14, and tested on macOS (Apple Silicon). It uses ASTRA for the system matrix and SIRT reconstruction.

I've also added a few experiments and notebooks to rerun the algorithm  from the paper, and to compare the different search strategies.

Generative AI was used to help write the code and the documentation, ideas and original concepts are mine.

## Setup

ASTRA has no pip wheel for every platform, so the environment comes from conda (Python 3.14; tested on macOS, Apple Silicon):

```bash
conda env create -f environment.yml
conda activate dfo-tr-fork
```

## Run

```bash
python runner.py --method v2.3 --size 32 --angles 6 --fes 1000000 --sirt   # one run, with progress
python scripts/run_experiments.py --quick                                    # every experiment, small (≈20 s)
python scripts/run_experiments.py                                            # every experiment, full
jupyter lab notebooks/                                                       # 01, 02, 03
python -m pytest                                                             # tests
DFO_TR_SCRIPT=path/to/dfo_tr.py python -m pytest                             # + check against the papers' script
```

The papers' script (`dfo_tr.py`) is not included; with it, the last command checks that `dfoct/paper.py` gives the
same fitness values for the same seed.

## Versions

| Method | Start | Disturbance | Rate Δ | Code |
|---|---|---|---|---|
| DFO-TR (paper) | uniform on [0, 255] | uniform on [0, 255] | 0.001 | `dfoct/paper.py` |
| v1.3 | uniform on [0, 255] | uniform on [0, 255] | 0.001 | `dfoct.V1_3` |
| v2.3 | flat level fitted to the sinogram ± 16 | best fly's value ± 0.05 | 0.003 | `dfoct.V2_3` |

All three use N = 2 flies, φ = √3 and a clamp to [0, 255]. v1.3 is the same search as the paper's, with delta
evaluation (one fitness evaluation = one fly move).

## Reproducing the results

1. `python scripts/run_experiments.py` writes `results/*.csv` and the final images (`results/*.npz`).
2. The notebooks read those files and write the figures and LaTeX tables in `figures/`.

Finished runs are cached in `results/runs/`, so an interrupted run continues where it stopped.

## Layout

| Path | What |
|---|---|
| `dfoct/problem.py` | phantom, ASTRA system matrix, fitness e1, scores, SIRT-box |
| `dfoct/paper.py` | DFO-TR as in the papers' script (re-implemented as a function) |
| `dfoct/search.py` | the delta-evaluated search: v1.3 and v2.3 |
| `dfoct/analysis.py` | loading results, tables and plot helpers for the notebooks |
| `scripts/run_experiments.py` | produces `results/*.csv` |
| `notebooks/` | 01 problem and baseline, 02 the fork, 03 results |
| `runner.py` | one run from the terminal |
| `tests/` | tests, including reproduction of the development versions |

## Data

- `data/phantoms/`: the phantoms from the DFO-TR papers.
- `data/lodopab/`: 12 slices (6 whole, 6 crops; 12 patients) from the validation split of LoDoPaB-CT, downsampled to 32² and 64²; `manifest.csv` gives each slice's source. LoDoPaB-CT: Leuschner, J., Schmidt, M., Otero Baguer, D. and Maass, P. (2021) *Scientific Data* 8, 109, doi:10.1038/s41597-021-00893-z; data: doi:10.5281/zenodo.3384092, licensed CC BY 4.0. It is derived from the LIDC-IDRI database (Armato et al. 2011, doi:10.1118/1.3528204; The Cancer Imaging Archive, CC BY 3.0).

## Licence

GPL-3.0-or-later (see `LICENSE`). The search descends from Dispersive Flies Optimisation,
https://github.com/mohmaj/DFO (GPL-3.0).
