"""
tests/test_resample.py

resample() computes its three interpolation passes as banded matmuls into reused buffers.
Before that it used three dense np.einsum passes.  Both evaluate the same dot products,
so the output should be bit for bit the same; this checks it against the einsum version,
kept below as the reference.

Whether BLAS really gives identical bits depends on its kernels, so run this on every
machine that writes production files (OpenBLAS on x86 and Arm, Accelerate on macOS):

    pytest tests/test_resample.py -q
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from jetscape.bulk_sources import (Grid, _linear_weights, _weight_matrix, resample,
                                   resolve_out_grid)


def _einsum_resample(arr, src, out):
    """resample() as of js-contrib 309bb90: the same arithmetic, dense np.einsum passes."""
    n_feat, ntau_src, tol = arr.shape[-1], arr.shape[0], 1e-6
    fx = (out.axis("x") - src.x_min) / src.dx
    fy = (out.axis("y") - src.y_min) / src.dy
    if src.boost_invariant or src.neta <= 1:
        feta, eta_ok = np.zeros(out.neta), np.ones(out.neta, dtype=bool)
    else:
        feta = (out.axis("eta") - src.eta_min) / src.deta
        eta_ok = (feta >= -tol) & (feta <= src.neta - 1 + tol)
    x_ok = (fx >= -tol) & (fx <= src.nx - 1 + tol)
    y_ok = (fy >= -tol) & (fy <= src.ny - 1 + tol)
    mask = (x_ok[:, None, None] & y_ok[None, :, None] & eta_ok[None, None, :])
    mask = mask[..., None].astype(np.float32)

    def axis_matrix(f, n):
        i0, i1, w = _linear_weights(f, n)
        lo, hi = int(min(i0.min(), i1.min())), int(max(i0.max(), i1.max())) + 1
        return slice(lo, hi), _weight_matrix(i0 - lo, i1 - lo, w, hi - lo)

    sx, mx = axis_matrix(fx, src.nx)
    sy, my = axis_matrix(fy, src.ny)
    se, me = axis_matrix(feta, max(int(src.neta), 1))

    def spatial(k):
        a = np.asarray(arr[k, sx, sy, se, :], dtype=np.float64)
        a = np.einsum("ke,xyef->xykf", me, a, optimize=True)
        a = np.einsum("jy,xykf->xjkf", my, a, optimize=True)
        a = np.einsum("ix,xjkf->ijkf", mx, a, optimize=True)
        return a.astype(np.float32)

    result = np.zeros((out.ntau, out.nx, out.ny, out.neta, n_feat), dtype=np.float32)
    for j in range(out.ntau):
        ft = (out.tau_min + j * out.dtau - src.tau_min) / src.dtau if src.dtau else 0.0
        if ft < -tol or ft > ntau_src - 1 + tol:
            continue
        ft = min(max(ft, 0.0), ntau_src - 1.0)
        k0 = min(int(ft), ntau_src - 1)
        k1 = min(k0 + 1, ntau_src - 1)
        w = ft - k0
        v = spatial(k0)
        if w and k1 != k0:
            v = (1.0 - w) * v + w * spatial(k1)
        result[j] = v * mask
    return result


# MUSIC's Au+Au grid and the FNO output grid of prod_AuAu_0_10 (grid_fno.yaml)
MUSIC = Grid(nx=100, ny=100, neta=60, ntau=6, x_min=-15.0, dx=0.3, y_min=-15.0, dy=0.3,
             eta_min=-6.0, deta=0.2, tau_min=0.4, dtau=0.1)
FNO = Grid.from_bounds((-12.20625, 12.20625, 64), (-12.20625, 12.20625, 64),
                       (-4.84375, 4.84375, 32), tau_min=0.5, dtau=0.1)


def _source(src, seed=1):
    """Hydro-like frames: e >= 0 with an all-zero vacuum margin, signed velocities."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((src.ntau, src.nx, src.ny, max(src.neta, 1), 4))
    a[..., 0] = np.abs(a[..., 0]) * 10.0
    a[:, : src.nx // 5] = 0.0
    return a.astype(np.float32)


CASES = {
    "fno_from_music": (MUSIC, FNO),
    "music_native": (MUSIC, MUSIC),
    # finer in tau (blends two source frames), partly outside the box in x and eta
    "shifted_finer_tau": (MUSIC, Grid(nx=37, ny=29, neta=11, ntau=9, x_min=-16.1, dx=0.91,
                                      y_min=-9.0, dy=0.61, eta_min=-7.0, deta=1.3,
                                      tau_min=0.43, dtau=0.05)),
    "boost_invariant": (Grid(nx=40, ny=30, neta=1, ntau=4, x_min=-6.0, dx=0.3, y_min=-4.5,
                             dy=0.3, eta_min=0.0, deta=0.0, tau_min=0.6, dtau=0.2,
                             boost_invariant=True),
                        Grid(nx=21, ny=17, neta=5, ntau=4, x_min=-5.0, dx=0.5, y_min=-4.0,
                             dy=0.5, eta_min=-1.0, deta=0.5, tau_min=0.6, dtau=0.2)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_resample_is_bit_identical_to_the_einsum_version(case):
    src, out = CASES[case]
    out = resolve_out_grid(src, out)
    data = _source(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got, ref = resample(data, src, out), _einsum_resample(data, src, out)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert np.any(ref != 0)
    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
