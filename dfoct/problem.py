"""The reconstruction problem: a phantom, its sinogram, the fitness e1, and the scores used in the results.

A problem is a phantom image (size x size grey levels in [0, 255]) seen from `alpha` angles by ASTRA's
linear parallel-beam projector, as in the DFO-TR papers. The projector is kept as its sparse system
matrix W (one row per ray, one column per pixel), so the search loops can be compiled with Numba.
The fitness of an image x is e1 = ||ref - W x||^2, the squared sinogram error, where ref = W phantom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

LOWER, UPPER = 0.0, 255.0  # grey-level box every method is kept in
DATA = Path(__file__).resolve().parent.parent / "data"
SETTINGS = [(5, 32, 6), (5, 64, 6), (5, 32, 32)]  # (phantom, size, angles) used in the results


@dataclass(frozen=True)
class Problem:
    phantom_no: int
    size: int
    alpha: int
    phantom: np.ndarray  # (size, size) float64 in [0, 255]
    indptr: np.ndarray  # W as CSR: rays x pixels
    indices: np.ndarray
    values: np.ndarray
    ref: np.ndarray  # the measured sinogram, W @ phantom (noise-free)
    col_ptr: np.ndarray  # W as CSC: the rays through pixel j are col_rows[col_ptr[j]:col_ptr[j + 1]]
    col_rows: np.ndarray
    col_vals: np.ndarray

    @property
    def D(self) -> int:
        """Number of unknowns (pixels)."""
        return self.size * self.size

    @property
    def label(self) -> str:
        return f"{self.size}² / {self.alpha} angles"

    def e1(self, x: np.ndarray) -> float:
        x = np.ascontiguousarray(x, dtype=np.float64)
        return float(e1(self.indptr, self.indices, self.values, self.ref, x, np.empty(self.ref.size)))

    def residual(self, x: np.ndarray) -> np.ndarray:
        """ref - W x, computed in full."""
        r = np.empty(self.ref.size)
        e1(self.indptr, self.indices, self.values, self.ref, np.ascontiguousarray(x, dtype=np.float64), r)
        return self.ref - r


@dataclass(frozen=True)
class Run:
    x: np.ndarray  # best image found, flattened
    e1: float
    fes: int  # function evaluations used (fewer than the budget only if e1 reached 0)
    snaps: np.ndarray  # (len(checkpoints), D): the best image when the FE count reached each checkpoint
    seconds: float = 0.0  # wall-clock time of the search (setup excluded)


def geometry(size: int, alpha: int):
    """ASTRA volume and parallel-beam projection geometry: `alpha` angles over [0, pi), one detector per pixel."""
    import astra

    vol_geom = astra.create_vol_geom(size, size)
    proj_geom = astra.create_proj_geom("parallel", 1.0, size, np.linspace(0, np.pi, alpha, endpoint=False))
    return vol_geom, proj_geom


def load_phantom(phantom_no: int, size: int) -> np.ndarray:
    from PIL import Image

    path = DATA / "phantoms" / f"Phantom_{phantom_no:02d}_{size}x{size}.bmp"
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def load_problem(phantom_no: int, size: int, alpha: int) -> Problem:
    return problem_from_image(load_phantom(phantom_no, size), alpha, phantom_no)


# Held-out images (never used while developing or tuning the fork):
# - "phantom1": binary (0 or 255 only), so the [0, 255] box alone pins most of it;
# - "phantom1-grey": the same shapes at 64 and 192, so the box no longer gives the answer;
# - "ct-whole00".."05", "ct-crop00".."05": LoDoPaB-CT validation slices (data/lodopab/, CC BY 4.0), 12 patients.
CT_IMAGES = [f"ct-{kind}{i:02d}" for kind in ("whole", "crop") for i in range(6)]
HELDOUT_IMAGES = ["phantom5", "phantom1", "phantom1-grey", *CT_IMAGES]


def load_image(name: str, size: int) -> np.ndarray:
    """A held-out truth image by name (see HELDOUT_IMAGES), size x size grey levels in [0, 255]."""
    from PIL import Image

    if name.startswith("ct-"):
        path = DATA / "lodopab" / f"{name[3:]}_{size}x{size}.png"
        return np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    if name == "phantom1-grey":
        return np.where(load_phantom(1, size) > 127, 192.0, 64.0)
    if name.startswith("phantom"):
        return load_phantom(int(name[len("phantom"):]), size)
    raise ValueError(f"unknown image {name!r}")


def load_image_problem(name: str, size: int, alpha: int) -> Problem:
    return problem_from_image(load_image(name, size), alpha)


def problem_from_image(phantom: np.ndarray, alpha: int, phantom_no: int = 0) -> Problem:
    """The problem of reconstructing `phantom` (the truth) from its noise-free sinogram at `alpha` angles."""
    import astra

    size = phantom.shape[0]
    vol_geom, proj_geom = geometry(size, alpha)
    projector_id = astra.create_projector("linear", proj_geom, vol_geom)
    matrix_id = astra.projector.matrix(projector_id)
    W = astra.matrix.get(matrix_id).tocsr()
    astra.matrix.delete(matrix_id)
    astra.projector.delete(projector_id)
    Wc = W.tocsc()
    return Problem(
        phantom_no, size, alpha, phantom,
        W.indptr.astype(np.int64), W.indices.astype(np.int64), W.data.astype(np.float64),
        W @ phantom.ravel(),
        Wc.indptr.astype(np.int64), Wc.indices.astype(np.int64), Wc.data.astype(np.float64),
    )  # fmt: skip


@njit(nogil=True, cache=True)
def e1(indptr, indices, values, ref, x, sino):
    """||ref - W x||^2, projecting every pixel (a full evaluation); W x is left in `sino`."""
    total = 0.0
    for r in range(indptr.shape[0] - 1):
        acc = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            acc += values[k] * x[indices[k]]
        sino[r] = acc
        d = ref[r] - acc
        total += d * d
    return total


def flat_level(problem: Problem) -> float:
    """The constant image that best fits the measurements (least squares; uses only the sinogram)."""
    w = problem.ref - problem.residual(np.ones(problem.D))  # W @ 1: each ray's total weight
    return float(np.clip(problem.ref @ w / (w @ w), LOWER, UPPER))


def sirt_box(problem: Problem, iterations: int = 50_000) -> np.ndarray:
    """SIRT kept in [0, 255] (ASTRA's SIRT with min/max constraints): the classical reference method."""
    import astra

    p = problem
    vol_geom, proj_geom = geometry(p.size, p.alpha)
    projector_id = astra.create_projector("linear", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, p.ref.reshape(p.alpha, p.size))
    rec_id = astra.data2d.create("-vol", vol_geom, 0.0)
    cfg = astra.astra_dict("SIRT")
    cfg["ReconstructionDataId"], cfg["ProjectionDataId"], cfg["ProjectorId"] = rec_id, sino_id, projector_id
    cfg["option"] = {"MinConstraint": LOWER, "MaxConstraint": UPPER}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)
    rec = astra.data2d.get(rec_id).astype(np.float64)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete([rec_id, sino_id])
    astra.projector.delete(projector_id)
    return np.clip(rec, LOWER, UPPER).ravel()


def score(problem: Problem, x: np.ndarray) -> dict:
    """Quality against the phantom (evaluation only; the search never sees it), with no rescaling."""
    from skimage.metrics import structural_similarity

    truth = problem.phantom.ravel()
    x = np.asarray(x, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((truth - x) ** 2)))
    return {
        "e1": problem.e1(x),
        "psnr": float("inf") if rmse == 0 else 20 * np.log10(UPPER / rmse),
        "ssim": float(
            structural_similarity(
                truth.reshape(problem.size, -1), x.reshape(problem.size, -1), data_range=UPPER
            )
        ),
        "rmse": rmse,
        "on_bound": float(np.mean((x <= LOWER) | (x >= UPPER))),  # share of pixels stuck at 0 or 255
    }


def checkpoints(budget: int, n: int = 30) -> np.ndarray:
    """FE counts to snapshot the best image at: n log-spaced from 10 to `budget`, plus 100k and the budget."""
    pts = np.unique(np.round(np.logspace(1, np.log10(budget), n)).astype(np.int64))
    return np.unique(np.concatenate([pts, [b for b in (100_000, budget) if b <= budget]])).astype(np.int64)
