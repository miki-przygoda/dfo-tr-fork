import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import dfoct
from dfoct import search
from dfoct.paper import PaperDFO

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


@pytest.fixture(scope="module", params=dfoct.SETTINGS, ids=lambda s: "{}-{}-{}".format(*s))
def problem(request):
    return dfoct.load_problem(*request.param)


def test_problem_sinogram_is_w_times_phantom(problem):
    assert problem.e1(problem.phantom.ravel()) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(problem.residual(np.zeros(problem.D)), problem.ref)


def test_flat_level_is_the_least_squares_constant(problem):
    c = search.flat_level(problem)
    for d in (-0.5, 0.5):
        assert problem.e1(np.full(problem.D, c)) <= problem.e1(np.full(problem.D, c + d))


@pytest.mark.parametrize("run", GOLDEN["runs"], ids=lambda r: "{method}-{setting}-seed{seed}".format(**r))
def test_reproduces_the_versions_the_results_were_developed_with(run):
    """Same images as the versions in the development repo (mhbench), seed for seed."""
    p = dfoct.load_problem(*run["setting"])
    r = dfoct.run(p, dfoct.METHODS[run["method"]], run["fes"], run["seed"])
    assert r.e1 == run["e1"]
    assert hashlib.sha256(np.ascontiguousarray(r.x).tobytes()).hexdigest() == run["sha256"]


@pytest.mark.parametrize("method", list(dfoct.METHODS))
@pytest.mark.parametrize("fes", [50, 5_000, 60_000])
def test_delta_evaluation_matches_full_evaluation(problem, method, fes):
    s = dfoct.Search(problem, dfoct.METHODS[method], seed=7)
    s.advance(fes - 2)
    _rng, X, R, fitness, A, _pos, ints = s.state
    assert sorted(A[: ints[0]]) == list(np.flatnonzero(X[0] != X[1]))  # the set is where the flies differ
    for i in range(2):
        assert np.allclose(R[i, :-1], problem.residual(X[i]), atol=1e-6 * np.abs(problem.ref).max())
        assert fitness[i] == pytest.approx(problem.e1(X[i]), rel=1e-9, abs=1e-6)
    assert ((X >= dfoct.LOWER) & (X <= dfoct.UPPER)).all()


def test_advancing_in_pieces_is_the_same_as_one_run(problem):
    a = dfoct.Search(problem, dfoct.V2_3, seed=3)
    a.advance(10_000)
    b = dfoct.Search(problem, dfoct.V2_3, seed=3)
    for _ in range(10):
        b.advance(1_000)
    assert np.array_equal(a.x, b.x) and a.e1 == b.e1


def test_v1_3_start_and_restarts_are_uniform():
    p = dfoct.load_problem(5, 32, 6)
    X = dfoct.Search(p, dfoct.V1_3, seed=0).state[1]
    assert X.min() >= 0 and X.max() <= 255 and abs(X.mean() - 127.5) < 5


def test_paper_dfo_is_deterministic_and_scores_its_image():
    p = PaperDFO(5, 32, 6)
    a, b = p.run(300, seed=4), p.run(300, seed=4)
    assert a.e1 == b.e1 and np.array_equal(a.x, b.x) and a.fes == 300
    full = dfoct.load_problem(5, 32, 6).e1(a.x)  # float64 e1 of the same image
    assert a.e1 == pytest.approx(full, rel=1e-5)


@pytest.mark.skipif("DFO_TR_SCRIPT" not in os.environ, reason="set DFO_TR_SCRIPT to the papers' dfo_tr.py")
def test_paper_dfo_reproduces_the_script(tmp_path):
    """The script, with its clock seed replaced by a fixed one and a smaller budget, gives the same e1."""
    src = Path(os.environ["DFO_TR_SCRIPT"]).read_text()
    src = src.replace("seed = int(start_time)+np.random.randint(1000000)", "seed = 3")
    src = src.replace("maxFEs = 100000", "maxFEs = 2000").replace("        if True:", "        if False:")
    (tmp_path / "script.py").write_text(src)
    (tmp_path / "input").symlink_to(dfoct.problem.DATA / "phantoms")
    out = subprocess.run(
        [sys.executable, "script.py", "5", "32", "6"], cwd=tmp_path, capture_output=True, text=True, check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    ).stdout.strip().splitlines()[-1].split(",")  # fmt: skip
    r = PaperDFO(5, 32, 6).run(2000, seed=3)
    assert np.float32(out[4]) == np.float32(r.e1) and int(out[6]) == r.fes


def test_heldout_images_load():
    from dfoct.problem import HELDOUT_IMAGES, load_image

    assert set(np.unique(load_image("phantom1-grey", 32))) == {64.0, 192.0}
    assert set(np.unique(load_image("phantom1", 32))) == {0.0, 255.0}
    for name in HELDOUT_IMAGES:
        for size in (32, 64):
            x = load_image(name, size)
            assert x.shape == (size, size) and x.min() >= 0 and x.max() <= 255
