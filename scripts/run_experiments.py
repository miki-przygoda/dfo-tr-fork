"""Run every experiment behind the notebooks and write the results to results/*.csv.

    python scripts/run_experiments.py                 # everything (see --help for parts and sizes)
    python scripts/run_experiments.py --quick         # 2 seeds and 100k FEs: a check that it all runs

Parts:
- main:   DFO-TR (the papers' script), v1.3 and v2.3 on the three settings, 10 seeds, 1M FEs each;
          scores at log-spaced checkpoints -> results/main.csv, final images -> results/main_images.npz
- tuning: the v2.3 family (start, disturbance radius and rate) at 1M FEs -> results/tuning.csv
- sirt:   SIRT-box, 50k iterations -> results/sirt.csv, images -> results/sirt_images.npz
- timing: seconds for a 1M-FE run, one thread, nothing else running -> results/timing.csv
- heldout: v1.3, v2.3 and the grid's ±0.1 Δ0.003 cell on images never used in development (phantom 5 on new
          seeds, phantom 1 binary and re-greyed, 12 LoDoPaB-CT slices), plus SIRT-box on each
          -> results/heldout.csv, images -> results/heldout_images.npz
Finished runs are cached in results/runs/, so an interrupted run continues where it stopped.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per run (ASTRA too)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dfoct
from dfoct import Settings
from dfoct.problem import checkpoints

RESULTS = ROOT / "results"
PAPER = "DFO-TR (paper)"
MAIN = {PAPER: None, "v1.3": dfoct.V1_3, "v2.3": dfoct.V2_3}


def tuning_variants() -> dict[str, Settings]:
    """Every v2.3-family setting the notebooks show, by name (the same setting may appear under two names)."""
    v = {"v1.3": dfoct.V1_3, "clean start ±16": Settings(spread=16.0)}
    v["no restarts"] = Settings(spread=16.0, delta=0.0)
    for r in (16.0, 4.0, 1.0, 0.25, 0.1, 0.05):
        v[f"restart ±{r:g}"] = Settings(spread=16.0, radius=r)
    for r in (0.05, 0.1, 0.25, 1.0):
        for d in (0.0003, 0.001, 0.003, 0.01):
            v[f"±{r:g} Δ{d:g}"] = Settings(spread=16.0, radius=r, delta=d)
    v["v2.3"] = dfoct.V2_3
    return v


def _key(st: Settings | None, setting, fes: int, seed: int, n_cps: int) -> str:
    spec = "paper" if st is None else f"s{st.spread}_r{st.radius}_d{st.delta}"
    return f"{spec}_{setting[0]}_{setting[1]}_{setting[2]}_{fes}_{seed}_c{n_cps}"


def _one(job):
    """Run (or load) one job in a worker process; returns its rows of scores and its final image."""
    name, st, setting, fes, seed, cps = job
    path = RESULTS / "runs" / f"{_key(st, setting, fes, seed, len(cps))}.npz"
    held_out = isinstance(setting[0], str)  # an image name (held-out part) rather than a phantom number
    p = dfoct.load_image_problem(*setting) if held_out else dfoct.load_problem(*setting)
    if path.exists():
        z = np.load(path)
        snaps, x, seconds = z["snaps"], z["x"], float(z["seconds"])
    else:
        if st is None:
            from dfoct.paper import PaperDFO

            r = PaperDFO(*setting).run(fes, seed, cps)
        else:
            r = dfoct.run(p, st, fes, seed, cps)
        snaps, x, seconds = r.snaps.astype(np.float32), r.x, r.seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, snaps=snaps, x=x, seconds=seconds)
        tmp.replace(path)
    rows = [
        {
            "method": name,
            **({"image": setting[0]} if held_out else {}),
            "size": setting[1],
            "alpha": setting[2],
            "seed": seed,
            "fes": int(c),
            **dfoct.score(p, s),
        }
        for c, s in zip(cps, snaps, strict=True)
    ]
    return rows, x


def run_jobs(jobs, workers: int, label: str):
    rows, images = [], {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_one, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futures), 1):
            name, _st, setting, _fes, seed, _cps = futures[fut]
            r, x = fut.result()
            rows += r
            prefix = f"{setting[0]}|" if isinstance(setting[0], str) else ""
            images[f"{prefix}{name}|{setting[1]}|{setting[2]}|{seed}"] = x.astype(np.float32)
            print(f"\r{label}: {k}/{len(jobs)} runs, {time.perf_counter() - t0:.0f} s", end="", flush=True)
    print()
    return pd.DataFrame(rows), images


def part_main(seeds, fes, workers):
    cps = checkpoints(fes)
    jobs = [(m, st, s, fes, seed, cps) for m, st in MAIN.items() for s in dfoct.SETTINGS for seed in seeds]
    jobs.sort(key=lambda j: j[1] is not None)  # the slow paper runs first
    df, images = run_jobs(jobs, workers, "main")
    df.sort_values(["method", "size", "alpha", "seed", "fes"]).to_csv(RESULTS / "main.csv", index=False)
    np.savez_compressed(RESULTS / "main_images.npz", **images)


def part_tuning(seeds, fes, workers):
    variants = tuning_variants()
    unique = {st: name for name, st in variants.items()}  # run each setting once
    jobs = [(name, st, s, fes, seed, np.array([fes])) for st, name in unique.items()
            for s in dfoct.SETTINGS for seed in seeds]  # fmt: skip
    df, _ = run_jobs(jobs, workers, "tuning")
    rows = []
    for name, st in variants.items():
        sub = df[df.method == unique[st]].assign(method=name)
        rows.append(sub.assign(spread=st.spread, radius=st.radius, delta=st.delta))
    pd.concat(rows).to_csv(RESULTS / "tuning.csv", index=False)


def part_sirt(iterations=50_000):
    rows, images = [], {}
    for s in dfoct.SETTINGS:
        p = dfoct.load_problem(*s)
        t = time.perf_counter()
        x = dfoct.sirt_box(p, iterations)
        rows.append({"method": "SIRT-box", "size": s[1], "alpha": s[2], "iterations": iterations,
                     "seconds": time.perf_counter() - t, **dfoct.score(p, x)})  # fmt: skip
        images[f"SIRT-box|{s[1]}|{s[2]}"] = x.astype(np.float32)
    pd.DataFrame(rows).to_csv(RESULTS / "sirt.csv", index=False)
    np.savez_compressed(RESULTS / "sirt_images.npz", **images)


def part_timing(fes, paper_fes, reps=3):
    """One thread, one run at a time: seconds per run of `fes` FEs (the paper's script timed on fewer)."""
    from dfoct.paper import PaperDFO

    rows = []
    for s in dfoct.SETTINGS:
        p = dfoct.load_problem(*s)
        paper = PaperDFO(*s)
        paper.run(50, 0)  # warm up
        for m, st in MAIN.items():
            if st is None:
                n = paper_fes
                secs = [paper.run(n, seed).seconds for seed in range(reps)]
            else:
                dfoct.run(p, st, 1_000, 0)  # compile
                n = fes
                secs = [dfoct.run(p, st, n, seed).seconds for seed in range(reps)]
            rows.append({"method": m, "size": s[1], "alpha": s[2], "fes": n, "seconds": float(np.median(secs)),
                         "us_per_fe": 1e6 * float(np.median(secs)) / n, "machine": platform.node()})  # fmt: skip
            print(f"timing {m} {p.label}: {rows[-1]['us_per_fe']:.3f} µs per FE", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "timing.csv", index=False)


HELDOUT = {"v1.3": dfoct.V1_3, "v2.3": dfoct.V2_3, "±0.1 Δ0.003": Settings(spread=16.0, radius=0.1, delta=0.003)}


def _sirt_one(job):
    image, size, alpha, iterations = job
    p = dfoct.load_image_problem(image, size, alpha)
    t = time.perf_counter()
    x = dfoct.sirt_box(p, iterations)
    row = {"method": "SIRT-box", "image": image, "size": size, "alpha": alpha, "seed": -1, "fes": iterations,
           "seconds": time.perf_counter() - t, **dfoct.score(p, x)}  # fmt: skip
    return row, x


def part_heldout(fes, workers, phantom_seeds, ct_seeds, sirt_iterations=50_000):
    """Fresh seeds (from 100) so no run repeats one used in development; CT gets fewer seeds but 12 images."""
    cps = np.array(sorted({min(100_000, fes), fes}))
    jobs = []
    for image in dfoct.HELDOUT_IMAGES:
        seeds = ct_seeds if image.startswith("ct-") else phantom_seeds
        for _ph, size, alpha in dfoct.SETTINGS:
            jobs += [(m, st, (image, size, alpha), fes, seed, cps) for m, st in HELDOUT.items() for seed in seeds]
    jobs.sort(key=lambda j: -(j[2][1] ** 2 * j[2][2]))  # the biggest problems first
    df, images = run_jobs(jobs, workers, "heldout")
    rows = []
    sirt_jobs = [(im, s, a, sirt_iterations) for im in dfoct.HELDOUT_IMAGES for _ph, s, a in dfoct.SETTINGS]
    with ProcessPoolExecutor(workers) as pool:
        for row, x in pool.map(_sirt_one, sirt_jobs):
            rows.append(row)
            images[f"{row['image']}|SIRT-box|{row['size']}|{row['alpha']}|-1"] = x.astype(np.float32)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.sort_values(["image", "method", "size", "alpha", "seed", "fes"]).to_csv(RESULTS / "heldout.csv", index=False)
    np.savez_compressed(RESULTS / "heldout_images.npz", **images)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="+", default=["main", "tuning", "sirt", "heldout", "timing"],
                    choices=["main", "tuning", "sirt", "heldout", "timing"])  # fmt: skip
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--fes", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="2 seeds, 100k FEs (writes to results/quick/)")
    args = ap.parse_args()
    global RESULTS
    if args.quick:
        args.seeds, args.fes, RESULTS = 2, 100_000, RESULTS / "quick"
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seeds)
    if "main" in args.parts:
        part_main(seeds, args.fes, args.workers)
    if "tuning" in args.parts:
        part_tuning(seeds, args.fes, args.workers)
    if "sirt" in args.parts:
        part_sirt(5_000 if args.quick else 50_000)
    if "heldout" in args.parts:
        n = args.seeds
        part_heldout(args.fes, args.workers, range(100, 100 + n), range(100, 100 + max(1, n // 2)),
                     5_000 if args.quick else 50_000)  # fmt: skip
    if "timing" in args.parts:
        part_timing(args.fes, 2_000 if args.quick else 20_000)


if __name__ == "__main__":
    main()
