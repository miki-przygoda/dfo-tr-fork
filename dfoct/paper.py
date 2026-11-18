"""DFO-TR as in the papers' script (al-Rifaie's `dfo_tr.py`), re-implemented as a function.

It performs the script's operations in the same order with the same NumPy and ASTRA calls: N = 2 flies,
float32 sinograms from ASTRA's forward projector (`OpTomo.FP`), fitness = the squared sinogram error
summed in float32 by NumPy (the script's `from numpy import *` makes `sum` NumPy's), U(0, sqrt 3),
delta = 0.001, clamping to [0, 255]. Changes, none of which alter the search: a `seed` argument instead of the clock, an FE budget
argument, and snapshots kept in memory instead of PNG files. With the same seed it reproduces the
script's e1 exactly (checked in tests/test_paper.py against a copy of the script, when one is given).
"""

from __future__ import annotations

import time

import numpy as np

from .problem import LOWER, UPPER, Run, geometry, load_phantom

N, DELTA, PHI = 2, 0.001, float(np.sqrt(3))


class PaperDFO:
    """The script's setup; `run(fes, seed, checkpoints)` is its main loop."""

    def __init__(self, phantom_no: int, size: int, alpha: int):
        import astra

        self.size, self.D = size, size * size
        phantom = load_phantom(phantom_no, size).astype(np.uint8)
        vol_geom, proj_geom = geometry(size, alpha)
        self.projector_id = astra.create_projector("linear", proj_geom, vol_geom)
        self.W = astra.OpTomo(self.projector_id)
        self.x_sino = np.zeros((alpha, size), dtype=np.float32)
        ref_sino = np.zeros((alpha, size), dtype=np.float32)
        self.W.FP(phantom, out=ref_sino)
        self.ref = ref_sino.flatten()

    def eval(self, x):
        self.W.FP(x, out=self.x_sino)
        # the script's `from numpy import *` makes `sum` and `pow` NumPy's: a float32 pairwise sum
        return np.sum(np.power(self.ref - self.x_sino.flatten(), 2))

    def run(self, fes: int, seed: int, checkpoints=()) -> Run:
        rng = np.random.RandomState(seed)  # the script seeds NumPy's global generator; same stream
        cps = list(np.asarray(checkpoints, dtype=np.int64))
        snaps = np.empty((len(cps), self.D))
        t0 = time.perf_counter()
        X = rng.uniform(low=LOWER, high=UPPER, size=(N, self.D))
        fitness = [self.eval(X[i]) for i in range(N)]
        used = N
        s = int(np.argmin(fitness))
        c = 0
        while c < len(cps) and used >= cps[c]:
            snaps[c] = X[s]
            c += 1
        while used < fes:
            if fitness[s] == 0:
                break
            for i in range(N):
                if i == s:
                    continue  # elitist
                left, right = (i - 1) % N, (i + 1) % N
                nb = right if fitness[right] < fitness[left] else left
                U = rng.uniform(low=0, high=PHI, size=self.D)
                X[i] = np.add(X[nb], np.multiply(U, np.subtract(X[s], X[i])))
                restart = rng.uniform(low=LOWER, high=UPPER, size=self.D)
                U = rng.uniform(low=0, high=1, size=self.D)
                X[i] = np.where(U < DELTA, restart, X[i])
                X[i] = np.where(X[i] > UPPER, UPPER, X[i])
                X[i] = np.where(X[i] < LOWER, LOWER, X[i])
                fitness[i] = self.eval(X[i])
                used += 1
                s = int(np.argmin(fitness))
                while c < len(cps) and used >= cps[c]:
                    snaps[c] = X[s]
                    c += 1
        seconds = time.perf_counter() - t0
        self.X = X  # both flies at the end (for checks)
        while c < len(cps):
            snaps[c] = X[s]
            c += 1
        return Run(X[s].copy(), float(fitness[s]), used, snaps, seconds)

    def close(self):
        import astra

        astra.projector.delete(self.projector_id)
