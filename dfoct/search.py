"""DFO-TR's search with delta evaluation: the engine behind v1.3 and v2.3.

The search is DFO-TR with N = 2 flies (as in the papers): the best fly s never moves (elitism), and the
other fly i moves every pixel d to
    X[i, d] = X[s, d] + U(0, phi) * (X[s, d] - X[i, d]),   phi = sqrt(3),
except that, with probability delta, a pixel is disturbed ("restarted") instead. Values are clamped to
[0, 255], the moved fly is scored, and the better of the two becomes s. One move = one function
evaluation (FE).

How it is computed (v1.3; none of this changes which images are visited, only how fast):
- Delta evaluation: each fly keeps its residual r = ref - W x. A pixel moving by dv changes only the rays
  through it, and e1 changes by dv * (dv * |w|^2 - 2 r . w), so a move costs the pixels that change,
  not the whole image. The result is still the exact e1 of the new image.
- Only pixels where the two flies disagree can change (X[s] - X[i] = 0 elsewhere), so the engine keeps
  that set and visits only it.
- The projector hits at most two neighbouring rays per pixel and angle, so each pixel's column of W is
  stored as (first ray, two weights) per angle.
- Random numbers come from xoshiro256+; the gap between restarts is drawn once as Geometric(delta)
  instead of one coin per pixel.

What v2.3 changes (the fork; the `Settings` below):
- start: both flies start at the flat grey level that best fits the sinogram, +/- `spread`
  (DFO-TR: uniform on [0, 255]);
- restart: a disturbed pixel goes to the best fly's value +/- `radius` (DFO-TR: uniform on [0, 255]);
- rate: delta = 0.003 (DFO-TR: 0.001).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, uint64

from .problem import LOWER, UPPER, Problem, Run, e1, flat_level

PHI = float(np.sqrt(3.0))
RESUM_EVERY = 1024  # moves between recomputing e1 = r . r from the residual (stops rounding drift)
INV_2_53 = 1.0 / 9007199254740992.0


@dataclass(frozen=True)
class Settings:
    """How a run starts and how it disturbs pixels. `None` means DFO-TR's uniform draw on [0, 255]."""

    spread: float | None = None  # start: flat level +/- spread
    radius: float | None = None  # restart: best fly's value +/- radius
    delta: float = 0.001  # restart rate per pixel per move


# ---------------------------------------------------------------- compiled kernels


@njit(nogil=True, cache=True, inline="always")
def _next(q0, q1, q2, q3):
    """One xoshiro256+ step (Blackman & Vigna) on state held in locals: (uniform in [0, 1), new state)."""
    result = q0 + q3
    t = q1 << uint64(17)
    q2 ^= q0
    q3 ^= q1
    q1 ^= q2
    q0 ^= q3
    q2 ^= t
    q3 = (q3 << uint64(45)) | (q3 >> uint64(19))
    return float(result >> uint64(11)) * INV_2_53, q0, q1, q2, q3


@njit(nogil=True, cache=True, fastmath=True)
def _move_pixel(first, weights, col_sq, r, j, dv):
    """Move pixel j by dv: update the residual r of the rays through it; returns the change in e1."""
    dot = 0.0
    for a in range(first.shape[1]):
        b = np.uintp(first[j, a])
        b1 = b + np.uintp(1)
        w0 = np.float64(weights[j, a, 0])
        w1 = np.float64(weights[j, a, 1])
        r0 = r[b]
        r1 = r[b1]
        dot += r0 * w0 + r1 * w1
        r[b] = r0 - dv * w0
        r[b1] = r1 - dv * w1
    return dv * (dv * col_sq[j] - 2.0 * dot)


@njit(nogil=True, cache=True)
def _resync(indptr, indices, values, ref, x, r):
    """Recompute the residual and e1 in full."""
    f = e1(indptr, indices, values, ref, x, r)  # r now holds W x
    for row in range(ref.shape[0]):
        r[row] = ref[row] - r[row]
    return f


@njit(cache=True)
def _band(col_ptr, col_rows, col_vals, D, size, alpha):
    """Per (pixel, angle): the first ray hit, and the weights of it and the next ray."""
    first = np.full((D, alpha), -1, dtype=np.int32)
    weights = np.zeros((D, alpha, 2))
    col_sq = np.zeros(D)
    for j in range(D):
        for k in range(col_ptr[j], col_ptr[j + 1]):
            a = col_rows[k] // size
            if first[j, a] < 0 or col_rows[k] < first[j, a]:
                first[j, a] = col_rows[k]
        for k in range(col_ptr[j], col_ptr[j + 1]):
            a = col_rows[k] // size
            off = col_rows[k] - first[j, a]
            if off > 1:
                raise ValueError("a pixel hits more than two adjacent detectors at one angle")
            weights[j, a, off] = col_vals[k]
            col_sq[j] += col_vals[k] * col_vals[k]
        for a in range(alpha):
            if first[j, a] < 0:  # the pixel misses this angle: zero weights on a valid ray
                first[j, a] = a * size
    return first, weights, col_sq


@njit(nogil=True, cache=True)
def _remove(A, pos, n_a, d):
    """Drop pixel d from the disagreement set A (swap with the last entry)."""
    k = pos[d]
    last = A[n_a - 1]
    A[k] = last
    pos[last] = k
    pos[d] = -1
    return n_a - 1


@njit(nogil=True, cache=True)
def _start(indptr, indices, values, ref, D, seed, level, spread, delta):
    """Seed the generator (splitmix64), place both flies, score them, and build the disagreement set."""
    z = uint64(seed)
    rng = np.empty(4, dtype=np.uint64)
    for k in range(4):
        z = z + uint64(0x9E3779B97F4A7C15)
        x = z
        x = (x ^ (x >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> uint64(27))) * uint64(0x94D049BB133111EB)
        rng[k] = x ^ (x >> uint64(31))
    q0, q1, q2, q3 = rng[0], rng[1], rng[2], rng[3]
    X = np.empty((2, D))
    R = np.zeros((2, ref.shape[0] + 1))  # residual per fly, plus one padding entry (always 0)
    fitness = np.empty(2)
    for i in range(2):
        for d in range(D):
            u, q0, q1, q2, q3 = _next(q0, q1, q2, q3)
            X[i, d] = min(max((level - spread) + (2.0 * spread) * u, LOWER), UPPER)
        fitness[i] = _resync(indptr, indices, values, ref, X[i], R[i])
    A = np.empty(D, dtype=np.int64)  # pixels where the flies disagree, in no order
    pos = np.full(D, -1, dtype=np.int64)  # each pixel's index in A, or -1
    n_a = 0
    for d in range(D):
        if X[0, d] != X[1, d]:
            A[n_a] = d
            pos[d] = n_a
            n_a += 1
    u, q0, q1, q2, q3 = _next(q0, q1, q2, q3)
    next_restart = int(np.log(1.0 - u) / np.log(1.0 - delta)) if delta > 0.0 else 1 << 62
    rng[0], rng[1], rng[2], rng[3] = q0, q1, q2, q3
    ints = np.array([n_a, next_restart, 0], dtype=np.int64)  # set size, next restart slot, moves made
    return rng, X, R, fitness, A, pos, ints


@njit(nogil=True, cache=True)
def _moves(
    indptr, indices, values, first, weights, col_sq, ref, D, delta, local, radius, moves,
    rng, X, R, fitness, A, pos, ints,
):  # fmt: skip
    """Make `moves` moves (fewer if e1 reaches 0); returns the best fly."""
    log_keep = np.log(1.0 - delta) if delta > 0.0 else -1.0
    q0, q1, q2, q3 = rng[0], rng[1], rng[2], rng[3]
    n_a, next_restart, move = ints[0], ints[1], ints[2]
    restart = np.zeros(D, dtype=np.bool_)
    restarts = np.empty(D, dtype=np.int64)
    s = np.argmin(fitness)
    for _ in range(moves):
        if fitness[s] <= 0.0:
            break
        i = 1 - s  # elitist: only the other fly moves, towards the best
        move += 1
        f = fitness[i]

        n_r = 0  # the pixels disturbed in this move (slots move * D + pixel; geometric gaps)
        while next_restart < D:
            restart[next_restart] = True
            restarts[n_r] = next_restart
            n_r += 1
            u, q0, q1, q2, q3 = _next(q0, q1, q2, q3)
            next_restart += int(np.log(1.0 - u) / log_keep) + 1
        next_restart -= D

        k = n_a - 1  # the DFO step, for the pixels where the flies disagree (backwards: safe removal)
        while k >= 0:
            d = A[k]
            if not restart[d]:
                u, q0, q1, q2, q3 = _next(q0, q1, q2, q3)
                v = X[s, d] + PHI * u * (X[s, d] - X[i, d])
                v = min(max(v, LOWER), UPPER)
                if v != X[i, d]:
                    f += _move_pixel(first, weights, col_sq, R[i], d, v - X[i, d])
                    X[i, d] = v
                if v == X[s, d]:
                    n_a = _remove(A, pos, n_a, d)
            k -= 1

        for t in range(n_r):  # the disturbance
            d = restarts[t]
            restart[d] = False
            u, q0, q1, q2, q3 = _next(q0, q1, q2, q3)
            if local:  # v2.3: near the best fly's value
                v = min(max(X[s, d] + radius * (2.0 * u - 1.0), LOWER), UPPER)
            else:  # DFO-TR: anywhere in [0, 255]
                v = LOWER + (UPPER - LOWER) * u
            if v != X[i, d]:
                f += _move_pixel(first, weights, col_sq, R[i], d, v - X[i, d])
                X[i, d] = v
            if v == X[s, d]:
                if pos[d] >= 0:
                    n_a = _remove(A, pos, n_a, d)
            elif pos[d] < 0:
                A[n_a] = d
                pos[d] = n_a
                n_a += 1

        if f < 1e-6:  # near e1 = 0 rounding decides: recompute in full
            f = _resync(indptr, indices, values, ref, X[i], R[i])
        elif move % RESUM_EVERY == 0:
            f = R[i] @ R[i]
        fitness[i] = f
        s = np.argmin(fitness)
    rng[0], rng[1], rng[2], rng[3] = q0, q1, q2, q3
    ints[0], ints[1], ints[2] = n_a, next_restart, move
    return s


# ---------------------------------------------------------------- Python interface


def columns(problem: Problem):
    """W's columns in the banded layout: float32 weights (as ASTRA's), 16-bit ray indices when they fit."""
    p = problem
    first, weights, col_sq = _band(p.col_ptr, p.col_rows, p.col_vals, p.D, p.size, p.alpha)
    index = np.uint16 if p.ref.size + 1 <= np.iinfo(np.uint16).max else np.uint32
    return first.astype(index), weights.astype(np.float32), col_sq


class Search:
    """One run that can be continued: `advance(moves)` makes more moves; `x`, `e1` and `fes` read it."""

    def __init__(self, problem: Problem, settings: Settings, seed: int, cols=None):
        p, st = problem, settings
        self.problem, self.settings, self.seed = p, st, seed
        self.cols = columns(p) if cols is None else cols
        level, spread = (127.5, 127.5) if st.spread is None else (flat_level(p), float(st.spread))
        self.state = _start(p.indptr, p.indices, p.values, p.ref, p.D, seed, level, spread, st.delta)
        self.best = int(np.argmin(self.state[3]))
        self.seconds = 0.0

    def advance(self, moves: int) -> int:
        """Make up to `moves` moves; returns how many were made (fewer only if e1 reached 0)."""
        p, st = self.problem, self.settings
        before, t = self.moves, time.perf_counter()
        local, radius = st.radius is not None, 0.0 if st.radius is None else float(st.radius)
        args = (p.indptr, p.indices, p.values, *self.cols, p.ref, p.D, st.delta, local, radius, int(moves))
        self.best = int(_moves(*args, *self.state))
        self.seconds += time.perf_counter() - t
        return self.moves - before

    @property
    def moves(self) -> int:
        return int(self.state[6][2])

    @property
    def fes(self) -> int:
        return self.moves + 2  # the two starting flies are evaluated once each

    @property
    def x(self) -> np.ndarray:
        return self.state[1][self.best].copy()

    @property
    def e1(self) -> float:
        return float(self.state[3][self.best])


def run(problem: Problem, settings: Settings, fes: int, seed: int, checkpoints=()) -> Run:
    """A run of `fes` function evaluations, with the best image saved at each checkpoint (in FEs)."""
    search = Search(problem, settings, seed)
    cps = np.asarray(checkpoints, dtype=np.int64)
    snaps = np.empty((cps.size, problem.D))
    for c, at in enumerate(cps):
        if at > search.fes:
            search.advance(min(at, fes) - search.fes)
        snaps[c] = search.x
    if fes > search.fes:
        search.advance(fes - search.fes)
    return Run(search.x, search.e1, search.fes, snaps, search.seconds)
