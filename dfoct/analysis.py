"""Loading the results (results/*.csv) and the tables and figures the notebooks share."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .problem import SETTINGS, UPPER

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"DFO-TR (paper)": "black", "v1.3": "#d62728", "v2.3": "#1f77b4", "SIRT-box": "grey"}


def results_dir() -> Path:
    """results/, or results/quick/ (from `run_experiments.py --quick`) when the full results aren't there."""
    full = ROOT / "results"
    if (full / "main.csv").exists():
        return full
    print("results/main.csv not found: using the quick results (2 seeds, 100k FEs)")
    return full / "quick"


def load() -> dict:
    d = results_dir()
    out = {name: pd.read_csv(d / f"{name}.csv") for name in ("main", "tuning", "sirt")}
    timing = d / "timing.csv"
    if not timing.exists():  # timing is a separate step, run with nothing else on the machine
        print("results/timing.csv not found: using the quick run's timings")
        timing = ROOT / "results" / "quick" / "timing.csv"
    out["timing"] = pd.read_csv(timing)
    out["images"] = dict(np.load(d / "main_images.npz"))
    out["sirt_images"] = dict(np.load(d / "sirt_images.npz"))
    if (d / "heldout.csv").exists():  # the held-out part (run_experiments.py --parts heldout)
        out["heldout"] = pd.read_csv(d / "heldout.csv")
        out["heldout_images"] = dict(np.load(d / "heldout_images.npz"))
    return out


HELDOUT_SETS = {  # rows of the held-out table: name -> which images it pools
    "Real CT, whole slices": "ct-whole",
    "Real CT, crops": "ct-crop",
    "Phantom 1 re-greyed (64/192)": "phantom1-grey",
    "Phantom 1 binary": "phantom1",
    "Phantom 5, new seeds": "phantom5",
}


def heldout_set(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df[df.image.str.startswith(key)] if key.startswith("ct-") else df[df.image == key]


def paired_wins(df: pd.DataFrame, a: str, b: str, metric: str = "psnr") -> tuple[int, int]:
    """Runs where `a` beats `b` on the same image and seed, and how many such pairs there are."""
    k = ["image", "size", "alpha", "seed"]
    m = df[df.method == a].merge(df[df.method == b], on=k, suffixes=("_a", "_b"))
    return int((m[f"{metric}_a"] > m[f"{metric}_b"]).sum()), len(m)


def label(size: int, alpha: int) -> str:
    return f"{size}² / {alpha} angles"


def settings():
    return [(s, a) for _p, s, a in SETTINGS]


def finals(main: pd.DataFrame) -> pd.DataFrame:
    """Each run's scores at its last checkpoint (the full budget)."""
    last = main.groupby(["method", "size", "alpha", "seed"]).fes.transform("max")
    return main[main.fes == last]


def at(main: pd.DataFrame, fes: int) -> pd.DataFrame:
    return main[main.fes == fes]


def compare(df: pd.DataFrame, a: str, b: str, metric: str = "psnr") -> pd.DataFrame:
    """Per setting: medians of `a` and `b`, the share of seed pairs where a is better, and Mann-Whitney's p."""
    from scipy.stats import mannwhitneyu

    rows = []
    for size, alpha in settings():
        x = df[(df.method == a) & (df["size"] == size) & (df.alpha == alpha)][metric].to_numpy()
        y = df[(df.method == b) & (df["size"] == size) & (df.alpha == alpha)][metric].to_numpy()
        better = np.mean(x[:, None] > y[None, :]) if metric != "e1" else np.mean(x[:, None] < y[None, :])
        rows.append({
            "setting": label(size, alpha), f"{a} median": np.median(x), f"{b} median": np.median(y),
            f"{a} better (pairs)": f"{better:.0%}", "Mann-Whitney p": mannwhitneyu(x, y).pvalue,
        })  # fmt: skip
    return pd.DataFrame(rows).set_index("setting")


def median_seed(df: pd.DataFrame, method: str, size: int, alpha: int, metric: str = "psnr") -> int:
    """The seed whose score is the median one (the lower middle for an even count)."""
    sub = df[(df.method == method) & (df["size"] == size) & (df.alpha == alpha)].sort_values(metric)
    return int(sub.seed.iloc[(len(sub) - 1) // 2])


def show(ax, x, size: int, title: str = ""):
    ax.imshow(np.reshape(x, (size, size)), cmap="gray", vmin=0, vmax=UPPER, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def show_error(ax, x, truth, size: int):
    err = np.reshape(np.asarray(x, dtype=np.float64) - truth.ravel(), (size, size))
    return ax.imshow(err, cmap="RdBu_r", vmin=-128, vmax=128, interpolation="nearest")


def to_tex(df: pd.DataFrame, path: Path, digits: int = 2) -> None:
    """Write a table as LaTeX (booktabs), with ², Δ, ± and µ as math so it compiles under any font."""
    tex = df.to_latex(na_rep="--", float_format=lambda v: f"{v:.{digits}f}" if abs(v) < 1e4 else f"{v:.3g}")
    for a, b in (
        ("²", "$^2$"),
        ("Δ", "$\\Delta$"),
        ("±", "$\\pm$"),
        ("µ", "$\\mu$"),
        ("ℓ", "$\\ell$"),
        ("→", "$\\to$"),
        ("∞", "$\\infty$"),
        ("%", "\\%"),
    ):
        tex = tex.replace(a, b)
    Path(path).write_text(tex)
