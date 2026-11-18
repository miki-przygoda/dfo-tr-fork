"""Run one method on one setting and print its progress.

    python runner.py --method v2.3 --size 32 --angles 6 --fes 1000000
    python runner.py --method paper --size 32 --angles 6 --fes 20000 --sirt

Prints FEs, FEs per second, e1, PSNR and SSIM (against the truth, for evaluation only) at log-spaced
points, then saves the final image (truth | result [| SIRT-box]) as a PNG and the progress as a CSV.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import dfoct
from dfoct.problem import checkpoints

METHODS = {"paper": "DFO-TR (paper)", "v1.3": "v1.3", "v2.3": "v2.3"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", choices=list(METHODS), default="v2.3")
    ap.add_argument("--phantom", type=int, default=5)
    ap.add_argument("--size", type=int, default=32, choices=[32, 64])
    ap.add_argument("--angles", type=int, default=6)
    ap.add_argument("--fes", type=int, default=1_000_000, help="function evaluations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sirt", action="store_true", help="also run SIRT-box (50k iterations)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "runner"))
    args = ap.parse_args()

    name = METHODS[args.method]
    p = dfoct.load_problem(args.phantom, args.size, args.angles)
    print(f"{name}: phantom {args.phantom}, {p.label}: {p.D:,} unknowns, {p.ref.size:,} measurements; "
          f"{args.fes:,} FEs, seed {args.seed}")  # fmt: skip
    print(f"{'FEs':>12} {'seconds':>9} {'FEs/s':>11} {'e1':>12} {'PSNR (dB)':>10} {'SSIM':>6}")
    if args.method == "paper":
        from dfoct.paper import PaperDFO

        cps = checkpoints(args.fes, 12)
        r = PaperDFO(args.phantom, args.size, args.angles).run(args.fes, args.seed, cps)
        rows = [{"fes": int(c), **dfoct.score(p, x)} for c, x in zip(cps, r.snaps, strict=True)]
        for row in rows:
            print(
                f"{row['fes']:12,} {'':>9} {'':>11} {row['e1']:12.4g} {row['psnr']:10.2f} {row['ssim']:6.3f}"
            )
        x, seconds, fes = r.x, r.seconds, r.fes
    else:
        search = dfoct.Search(p, dfoct.METHODS[name], args.seed)
        dfoct.Search(p, dfoct.METHODS[name], args.seed).advance(10)  # compile before the clock starts
        rows = []
        for c in checkpoints(args.fes, 12):
            if c > search.fes:
                search.advance(c - search.fes)
            sc = dfoct.score(p, search.x)
            rows.append({"fes": search.fes, "seconds": search.seconds, **sc})
            rate = search.fes / max(search.seconds, 1e-9)
            print(f"{search.fes:12,} {search.seconds:9.2f} {rate:11,.0f} {sc['e1']:12.4g} {sc['psnr']:10.2f} "
                  f"{sc['ssim']:6.3f}", flush=True)  # fmt: skip
        x, seconds, fes = search.x, search.seconds, search.fes
    final = dfoct.score(p, x)
    print(f"final: {final['psnr']:.2f} dB, SSIM {final['ssim']:.3f}, {fes:,} FEs in {seconds:.2f} s")

    images = [("truth", p.phantom.ravel()), (f"{name}\n{final['psnr']:.2f} dB", x)]
    if args.sirt:
        t = time.perf_counter()
        xs = dfoct.sirt_box(p)
        sb = dfoct.score(p, xs)
        print(
            f"SIRT-box (50k iterations): {sb['psnr']:.2f} dB, SSIM {sb['ssim']:.3f}, {time.perf_counter() - t:.1f} s"
        )
        images.append((f"SIRT-box\n{sb['psnr']:.2f} dB", xs))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.phantom}_{args.size}_{args.angles}_seed{args.seed}"
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3.3), layout="constrained")
    for ax, (title, img) in zip(axes, images, strict=True):
        ax.imshow(np.reshape(img, (p.size, p.size)), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(out / f"{stem}.png", dpi=120)
    print(f"saved {out / stem}.png and .csv")


if __name__ == "__main__":
    main()
