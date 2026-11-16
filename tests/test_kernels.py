# SPDX-License-Identifier: Apache-2.0
"""The optional C kernels against the Python reference (skipped when the
kernels are not built: python -m ac4.build_kernels)."""
import ctypes
import os

import numpy as np
import pytest

from ac4 import build_kernels as BK
from ac4 import qmf as Q


def _lib(name):
    p = BK.lib_path(name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not built")
    return ctypes.CDLL(p)


def test_qmf_kernel_matches_python():
    dll = _lib("qmf_kernel")
    dll.qmf_kernel_abi.restype = ctypes.c_int32
    assert int(dll.qmf_kernel_abi()) == 1
    f64 = np.ctypeslib.ndpointer(np.float64, flags="C")
    for fn in ("qmf_analyse_f64", "qmf_synthesise_f64"):
        g = getattr(dll, fn)
        g.restype = ctypes.c_int32
        g.argtypes = [f64, ctypes.c_int32, f64, f64, f64]
    w = Q.qwin()
    M = np.ascontiguousarray(Q.analysis_matrix(), np.complex128)
    N = np.ascontiguousarray(Q.synthesis_matrix(255), np.complex128)
    x = np.random.default_rng(7).standard_normal(64 * 120)
    nts = len(x) // 64
    Qc = np.empty((64, nts), np.complex128)
    assert dll.qmf_analyse_f64(np.ascontiguousarray(x), nts, w,
                               M.view(np.float64), Qc.view(np.float64)) == 0
    y = np.empty(nts * 64, np.float64)
    assert dll.qmf_synthesise_f64(Qc.view(np.float64), nts, w,
                                  N.view(np.float64), y) == 0
    # the Python loops are the reference (force them, kernel or not)
    saved = Q._QMF_STATE["lib"]
    Q._QMF_STATE["lib"] = False
    try:
        An = Q.analyse(x, w, M)
        Sn = Q.synthesise(An, w, N)
    finally:
        Q._QMF_STATE["lib"] = saved
    assert np.abs(An - Qc).max() < 1e-10
    assert np.abs(Sn - y).max() < 1e-10


def test_spectral_kernel_hand_built_stream():
    """Two-symbol book '0'->0, '10'->1 (dim 2, mod 2, off 1): bits 1 0 0
    decode to lines [-1, 0, -1, -1] and leave the cursor at bit 3."""
    dll = _lib("spectral_kernel")
    dll.spectral_kernel_abi.restype = ctypes.c_int32
    assert int(dll.spectral_kernel_abi()) == 1
    i32 = np.ctypeslib.ndpointer(np.int32, flags="C")
    i64 = np.ctypeslib.ndpointer(np.int64, flags="C")
    u8 = np.ctypeslib.ndpointer(np.uint8, flags="C")
    dll.asf_spectral_i32.restype = ctypes.c_int32
    dll.asf_spectral_i32.argtypes = [u8, ctypes.c_int64, i64,
                                     i32, ctypes.c_int32, i32, ctypes.c_int32,
                                     i32, i32, i32, i32, i32,
                                     i32, ctypes.c_int32]
    tree = np.array([-(0 + 1), 1, -(1 + 1), 0], np.int32)
    troot = np.full(12, -1, np.int32)
    troot[1] = 0
    cb_dim = np.zeros(12, np.int32); cb_dim[1] = 2
    cb_mod = np.zeros(12, np.int32); cb_mod[1] = 2
    cb_off = np.zeros(12, np.int32); cb_off[1] = 1
    buf = np.array([0b10000000], np.uint8)
    sects = np.array([1, 0, 2], np.int32)
    offsets = np.array([0, 2, 4], np.int32)
    lines = np.zeros(4, np.int32)
    pos = np.zeros(1, np.int64)
    rc = dll.asf_spectral_i32(buf, 1, pos, sects, 1, offsets, 3,
                              cb_dim, cb_mod, cb_off, tree, troot, lines, 4)
    assert rc == 0
    assert pos[0] == 3
    assert list(lines) == [-1, 0, -1, -1]
