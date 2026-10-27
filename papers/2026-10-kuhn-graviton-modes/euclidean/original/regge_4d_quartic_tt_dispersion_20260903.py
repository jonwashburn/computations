#!/usr/bin/env python3
"""Quartic (k^4) transverse-traceless dispersion of the linearised Regge action
on the periodic Freudenthal (Kuhn) triangulation of flat R^4.

Object.  S = sum_h A_h delta_h (Regge action, hinges h = triangles).  At the
flat background the second variation in squared edge lengths a_e is
  d2S = sum_h dA_h dDelta_h                    (Schlaefli kills A d2Delta)
with exact symbolic gradients (Heron for A, Gram/dihedral for delta) from
regge_4d_exact_m2_table_20260721.assemble_exact_grads on the 50-hinge
translation-fundamental domain.

Bloch matrix.  Edges of one unit cell are labelled by the 0/1 displacement
mask m = 1..15 (popcount 1: axis, 2: face diagonal, 3: body diagonal, 4: the
hyperbody diagonal).  For a strain plane wave  eps_e = c_{m(e)} exp(i k.mid_e)
the second variation per unit cell is  sigma(c,k) = (1/2) c^dagger M(k) c,
  M(k)_{m m'} = sum_{couplings (e,e')} w_{ee'} exp(i k.(mid_e - mid_{e'})),
  w_{ee'}     = (1/2) sum_h (dA_h/da_e dDelta_h/da_{e'} + (e <-> e')).
M(k) is a 15x15 trigonometric polynomial, so it is defined for complex k, and
its Taylor coefficients in k are exact rationals times powers of the
components.

Effective operator.  Uniform metric perturbations H (10) are exact zero
modes of M(0); so is the hyperbody-diagonal strain e_15 (Rocek-Williams'
fifth zero mode), for every k.  The remaining 4 edge directions are
non-metric with O(1) eigenvalues.  The Schur complement onto the metric
space, in a Frobenius-orthonormal basis of Sym(4),
    E(k) = P^T M P - P^T M Q (Q^T M Q)^{-1} Q^T M P     (10x10),
is expanded to fourth order in k.  Gauge modes u(x)v + v(x)u are exact null
vectors of E at every order.  On the cone u = iT + n (T, n real orthonormal)
the two TT polarisations tau have  tau^T E_2 tau = 0  and their quartic branch
    Lambda = F4_tt - F3_tc (F2_cc)^{-1} F3_ct                 (2x2)
gives  omega = k (1 - kappa k^2),  kappa = 2 Lambda / |E_2 normalisation|.

Claims tested here (each is a printed receipt):
  R1  M(k) e_15 = 0 for all k; no coupling involves mask 15.
  R2  M(k) = M(-k): odd Taylor orders vanish identically.
  R3  kernel of M(k) is 5-dimensional at generic real k, 11 at k = 0.
  R4  TT quadratic form  x^T E_2 x = -(1/4)(u.u)|H|_F^2  for every real
      direction and every TT polarisation (isotropic cone).
  R5  On the cone, for random orthonormal (T, n):  Lambda = (1/48) sum_mu
      u_mu^4 * I_2  (polarisation-blind, hyperoctahedral quartic), with
      F3 = 0, E2 g0 = 0 and tau^T E4 g0 = 0 for the leading gauge vectors.
  R6  Time axis admissibility: Im sum_mu u_mu^4 = 0 for all n orthogonal to T
      iff reflection through T^perp is a lattice symmetry (T an axis or a
      face diagonal).
  R7  Nonperturbative root: the exact trigonometric E(i omega T + k n) has a
      TT null vector at omega*(k) with (k - omega*)/k^3 -> (1 + sum n_i^4)/24
      for T a lattice axis.
  R7b The exact TT root coincides, to working precision and for |k| up to
      2.5 in several frames (complex roots included), with the root of the
      nearest-neighbour Laplacian symbol  L(u) = sum_mu 4 sin^2(u_mu/2):
      the TT branch of the Regge graviton on this lattice is exactly
      L(i omega T + k n) = 0.  (Roček-Williams' transformed action has the
      factor 8 - Sigma - bar Sigma = L(k) in front of the tensor structure.)
  R8  Finite differences of the full nonlinear Regge action on the N = 4
      torus reproduce M(k) at lattice-allowed k.
  R9  Phenomenology: SME d = 6 coefficients, Kostelecky-Tasson Cherenkov
      bound, threshold parton energy for a given lattice spacing.

Usage (repo root, on the build box):
  python3 original/regge_4d_quartic_tt_dispersion_20260903.py
Writes state/qg_full_theory/regge_4d_quartic_tt_dispersion_20260903.json
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import mpmath as mp
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from regge_4d_exact_m2_table_20260721 import (  # type: ignore
    assemble_exact_grads,
    midpoint_of,
)
from regge_4d_gauge_discrimination_probe_20260721 import (  # type: ignore
    action_from_field,
    build_complex,
    central_second,
    mask_to_disp,
    popcount,
)

mp.mp.dps = 40
ORDER = 4

# ---------------------------------------------------------------------------
# 1. Couplings from exact gradients
# ---------------------------------------------------------------------------


def build_couplings():
    """dict (m, m', 2*Delta) -> mpf weight; also whether dDelta/da_15 == 0."""
    grads, fund, by = assemble_exact_grads(4)
    assert len(fund) == 50
    raw: Dict[Tuple[int, int, Tuple[int, int, int, int]], sp.Expr] = defaultdict(
        lambda: sp.Integer(0)
    )
    dD15_nonzero = 0
    dA15_nonzero = 0
    for gA, gD in grads:
        for ep, vd in gD.items():
            if ep[1] == 15 and vd != 0:
                dD15_nonzero += 1
        for e, va in gA.items():
            if e[1] == 15 and va != 0:
                dA15_nonzero += 1
        for e, va in gA.items():
            if va == 0:
                continue
            me = midpoint_of(e)
            for ep, vd in gD.items():
                if vd == 0:
                    continue
                mep = midpoint_of(ep)
                d2 = tuple(int(round(2 * (me[i] - mep[i]))) for i in range(4))
                w = sp.Rational(1, 2) * va * vd
                raw[(e[1], ep[1], d2)] += w
                nd2 = tuple(-x for x in d2)
                raw[(ep[1], e[1], nd2)] += w
    coup = {}
    table = []
    for key, wexpr in raw.items():
        wv = sp.N(wexpr, 60)
        wf = mp.mpf(str(wv))
        if abs(wf) < mp.mpf("1e-45"):
            continue
        coup[key] = wf
        table.append({"D": list(mask_to_disp(key[0])), "Dprime": list(mask_to_disp(key[1])),
                      "two_times_midpoint_separation": list(key[2]),
                      "weight_exact": str(wexpr), "weight_40digits": mp.nstr(wf, 40)})
    # supplementary table: the algebraic numbers that define M(k)
    tpath = Path("state/qg_full_theory/regge_4d_kuhn_coupling_table_20260903.json")
    tpath.parent.mkdir(parents=True, exist_ok=True)
    with open(tpath, "w") as f:
        json.dump({"convention": "M(k)_{D D'} = sum_v w(D,D',v) exp(i k.v/2), v = 2*(midpoint(D',x) - midpoint(D,0)); "
                               "second-order action per cell = (1/2) c^T M c on squared-length amplitudes",
                   "count": len(table), "couplings": table}, f, indent=1)
    return coup, dD15_nonzero, dA15_nonzero


# ---------------------------------------------------------------------------
# 2. Bloch matrix and Taylor coefficients
# ---------------------------------------------------------------------------


def bloch_matrix(coup, k) -> mp.matrix:
    M = mp.matrix(15, 15)
    for (m, mp_, d2), w in coup.items():
        ph = sum(k[i] * mp.mpf(d2[i]) / 2 for i in range(4))
        M[m - 1, mp_ - 1] += w * mp.exp(1j * ph)
    return M


def bloch_taylor(coup, u, order: int) -> List[mp.matrix]:
    Ms = [mp.matrix(15, 15) for _ in range(order + 1)]
    for (m, mp_, d2), w in coup.items():
        ph = sum(u[i] * mp.mpf(d2[i]) / 2 for i in range(4))
        x = 1j * ph
        pw = mp.mpc(1)
        for n in range(order + 1):
            Ms[n][m - 1, mp_ - 1] += w * pw / mp.factorial(n)
            pw = pw * x
    return Ms


# ---------------------------------------------------------------------------
# 3. Frobenius-orthonormal metric embedding, gauge, polarisations
# ---------------------------------------------------------------------------

SYM_PAIRS = [(a, b) for a in range(4) for b in range(a, 4)]  # 10
SQ2 = mp.sqrt(2)


def metric_coords(H) -> List:
    """Frobenius-orthonormal coordinates of a symmetric 4x4 H:
    x_aa = H_aa, x_ab = sqrt(2) H_ab (a<b); |x|^2 = |H|_F^2."""
    out = []
    for (a, b) in SYM_PAIRS:
        out.append(H[a][b] if a == b else SQ2 * H[a][b])
    return out


def P_metric_matrix() -> mp.matrix:
    """15x10: strain of edge class m under unit metric basis tensor."""
    P = mp.matrix(15, 10)
    for m in range(1, 16):
        D = [mp.mpf(int(x)) for x in mask_to_disp(m)]
        for j, (a, b) in enumerate(SYM_PAIRS):
            # D^T B_j D with B_aa = e_a e_a, B_ab = (e_a e_b + e_b e_a)/sqrt2
            P[m - 1, j] = D[a] * D[a] if a == b else SQ2 * D[a] * D[b]
    return P


def gram_schmidt(vectors: List[mp.matrix], against: List[mp.matrix] = ()) -> List[mp.matrix]:
    """Real Gram-Schmidt; returns orthonormal vectors independent of `against`."""
    out = []
    for v in vectors:
        v = v.copy()
        for w in list(against) + out:
            v = v - ((w.T * v)[0] / (w.T * w)[0]) * w
        nrm = mp.sqrt((v.T * v)[0])
        if nrm > mp.mpf("1e-20"):
            out.append(v / nrm)
    return out


def unit_vectors(n: int) -> List[mp.matrix]:
    vs = []
    for e in range(n):
        v = mp.matrix(n, 1)
        v[e] = 1
        vs.append(v)
    return vs


def cols(vectors: List[mp.matrix]) -> mp.matrix:
    n = vectors[0].rows
    M = mp.matrix(n, len(vectors))
    for j, v in enumerate(vectors):
        for i in range(n):
            M[i, j] = v[i]
    return M


def vec(xs) -> mp.matrix:
    v = mp.matrix(len(xs), 1)
    for i, x in enumerate(xs):
        v[i] = x
    return v


def outer_sym(u, v):
    return [[u[a] * v[b] + v[a] * u[b] for b in range(4)] for a in range(4)]


def tensor(a, b):
    return [[a[i] * b[j] for j in range(4)] for i in range(4)]


def addm(A, B, ca=1, cb=1):
    return [[ca * A[i][j] + cb * B[i][j] for j in range(4)] for i in range(4)]


def orthonormal_complement_4d(vectors: List[List]) -> List[List]:
    basis = [vec([mp.mpf(x) for x in v]) for v in vectors]
    out = gram_schmidt(unit_vectors(4), against=basis)
    return [[o[i] for i in range(4)] for o in out]


def mnorm(M: mp.matrix) -> mp.mpf:
    return max(abs(M[i, j]) for i in range(M.rows) for j in range(M.cols))


def fmt(z) -> str:
    z = mp.mpc(z)
    if abs(z.imag) < mp.mpf("1e-25"):
        return mp.nstr(z.real, 18)
    return f"{mp.nstr(z.real, 18)} + {mp.nstr(z.imag, 18)}i"


def rat_guess(x, maxden: int = 100000) -> str:
    x = mp.mpc(x)
    if abs(x.imag) > mp.mpf("1e-25"):
        return "complex"
    fr = Fraction(str(mp.nstr(x.real, 30))).limit_denominator(maxden)
    err = abs(mp.mpf(fr.numerator) / mp.mpf(fr.denominator) - x.real)
    return f"{fr}" if err < mp.mpf("1e-22") else "not-rational"


# ---------------------------------------------------------------------------
# 4. Truncated matrix series (Schur complement)
# ---------------------------------------------------------------------------


def series_mul(A, B, order):
    out = []
    for n in range(order + 1):
        acc = None
        for i in range(n + 1):
            j = n - i
            if i < len(A) and j < len(B):
                term = A[i] * B[j]
                acc = term if acc is None else acc + term
        if acc is None:
            acc = mp.matrix(A[0].rows, B[0].cols)
        out.append(acc)
    return out


def series_inv(D, order):
    X = D[0] ** -1
    n = D[0].rows
    R = [mp.matrix(n, n)] + [D[k] for k in range(1, order + 1)]
    RX = series_mul(R, [X] + [mp.matrix(n, n)] * order, order)
    negRX = [-(RX[k]) for k in range(order + 1)]
    term = [mp.eye(n)] + [mp.matrix(n, n)] * order
    total = [mp.eye(n)] + [mp.matrix(n, n)] * order
    for _ in range(order):
        term = series_mul(term, negRX, order)
        total = [total[k] + term[k] for k in range(order + 1)]
    return series_mul([X] + [mp.matrix(n, n)] * order, total, order)


def effective_series(Ms, P, Q, order):
    A = [P.T * M * P for M in Ms]
    B = [P.T * M * Q for M in Ms]
    Bt = [Q.T * M * P for M in Ms]
    D = [Q.T * M * Q for M in Ms]
    Dinv = series_inv(D, order)
    corr = series_mul(series_mul(B, Dinv, order), Bt, order)
    return [A[n] - corr[n] for n in range(order + 1)]


def effective_exact(coup, P, Q, u) -> mp.matrix:
    """Exact trigonometric Schur complement onto the metric space."""
    M = bloch_matrix(coup, u)
    A = P.T * M * P
    B = P.T * M * Q
    D = Q.T * M * Q
    return A - B * (D ** -1) * (Q.T * M * P)


# ---------------------------------------------------------------------------
# 5. On-shell TT quartic branch in a frame (Taylor)
# ---------------------------------------------------------------------------


def frame_vectors(T, n):
    """Polarisations (Frobenius unit), leading gauge vectors, completing set."""
    T = [mp.mpf(x) for x in T]
    n = [mp.mpf(x) for x in n]
    u = [1j * T[i] + n[i] for i in range(4)]
    a, b = orthonormal_complement_4d([T, n])
    e_plus = addm(tensor(a, a), tensor(b, b), 1 / SQ2, -1 / SQ2)
    e_cross = [[x / SQ2 for x in row] for row in outer_sym(a, b)]
    tau = [vec(metric_coords(e_plus)), vec(metric_coords(e_cross))]
    gauge = []
    for vidx in range(4):
        v = [mp.mpf(1) if i == vidx else mp.mpf(0) for i in range(4)]
        gauge.append(vec(metric_coords(outer_sym(u, v))))
    # 4 real vectors completing (tau, gauge) to a basis of C^10
    comp_all = gram_schmidt(unit_vectors(10), against=tau)
    chosen = []
    for cand in comp_all:
        if len(chosen) == 4:
            break
        trial = cols(tau + gauge + chosen + [cand])
        sv = mp.svd_c(trial, compute_uv=False)
        if min(sv) > mp.mpf("1e-12"):
            chosen.append(cand)
    assert len(chosen) == 4
    return u, tau, gauge, chosen


def schur_tt(F: mp.matrix) -> mp.matrix:
    """2x2 TT Schur complement of a 6x6 block matrix ordered (tau, comp)."""
    Ftt = F[0:2, 0:2]
    Ftc = F[0:2, 2:6]
    Fcc = F[2:6, 2:6]
    Fct = F[2:6, 0:2]
    return Ftt - Ftc * (Fcc ** -1) * Fct


def frame_dispersion(coup, P, Q, T, n, order=ORDER):
    u, tau, gauge, comp = frame_vectors(T, n)
    Ms = bloch_taylor(coup, u, order)
    E = effective_series(Ms, P, Q, order)
    # The exact gauge null vector is k-dependent, g(k) = g0 + k^2 g2 + ...;
    # order by order: E2 g0 = 0 exactly, and E4 g0 = -E2 g2 lies in the range
    # of E2, so it is orthogonal to the TT polarisations.  Both are receipts.
    gauge_res = max(mnorm(E[2] * g) for g in gauge)
    gauge_tt4 = max(abs((t.T * E[4] * g)[0]) for g in gauge for t in tau)
    W = cols(tau + comp)
    F = [W.T * E[k] * W for k in range(order + 1)]
    F2 = F[2]
    Lam = F[4][0:2, 0:2] - F[3][0:2, 2:6] * (F2[2:6, 2:6] ** -1) * F[3][2:6, 0:2]
    sum_u4 = sum(x ** 4 for x in u)
    pred = sum_u4 / 48
    dev = max(abs(Lam[i, j] - (pred if i == j else 0)) for i in range(2) for j in range(2))
    # E_2 on TT: -(1/4)(u.u); on the cone u.u = 0 so F2_tt = 0.
    # kappa: -(1/4)(k^2 - w^2) + Lam k^4 = 0  ->  w = k(1 - 2 Lam k^2)
    kappas = [2 * Lam[0, 0], 2 * Lam[1, 1]]
    return {
        "T": [fmt(x) for x in T],
        "n": [fmt(x) for x in n],
        "E2_gauge_annihilation_max": str(gauge_res),
        "tau_E4_gauge_max": str(gauge_tt4),
        "F1_max": str(mnorm(F[1])),
        "F3_max": str(mnorm(F[3])),
        "F2_tt_max": str(mnorm(F2[0:2, 0:2])),
        "F2_tc_max": str(mnorm(F2[0:2, 2:6])),
        "Lambda": [[fmt(Lam[i, j]) for j in range(2)] for i in range(2)],
        "sum_u4_over_48": fmt(pred),
        "Lambda_minus_pred_max": str(dev),
        "kappa_phase": [fmt(k) for k in kappas],
        "kappa_rational": [rat_guess(k) for k in kappas],
    }


# ---------------------------------------------------------------------------
# 6. Nonperturbative TT root of the exact trigonometric operator
# ---------------------------------------------------------------------------


def tt_root(coup, P, Q, T, n, kmag, kappa_guess):
    """Solve for omega with a TT null vector of exact E(i omega T + k n)."""
    _, tau, gauge, comp = frame_vectors(T, n)
    W = cols(tau + comp)
    kmag = mp.mpf(kmag)
    T = [mp.mpf(x) for x in T]
    n = [mp.mpf(x) for x in n]

    def lam_min(omega):
        u = [1j * omega * T[i] + kmag * n[i] for i in range(4)]
        E = effective_exact(coup, P, Q, u)
        L = schur_tt(W.T * E * W)
        ev = mp.eig(L)[0]
        return min(ev, key=lambda z: abs(z)), E

    w0 = kmag * (1 - kappa_guess * kmag ** 2)
    w1 = kmag * (1 - mp.mpf("1.5") * kappa_guess * kmag ** 2)
    f0, _ = lam_min(w0)
    f1, _ = lam_min(w1)
    for _ in range(60):
        if abs(f1 - f0) == 0:
            break
        w2 = w1 - f1 * (w1 - w0) / (f1 - f0)
        w0, f0 = w1, f1
        w1 = w2
        f1, _ = lam_min(w1)
        if abs(f1) < mp.mpf("1e-32"):
            break
    omega = w1
    # independent confirmation: 6th smallest singular value of E at the root
    u = [1j * omega * T[i] + kmag * n[i] for i in range(4)]
    E = effective_exact(coup, P, Q, u)
    sv = sorted(mp.svd_c(E, compute_uv=False))
    return omega, f1, [mp.nstr(s, 5) for s in sv[:7]]


# ---------------------------------------------------------------------------
# 7. Finite-difference cross-check on the N=4 torus
# ---------------------------------------------------------------------------


def strain_field(N, eps, cvec, m_mode, phase):
    k = 2.0 * math.pi * m_mode / N
    field = {}
    for base in itertools.product(range(N), repeat=4):
        xb = np.array(base, dtype=float)
        for mask in range(1, 16):
            D = mask_to_disp(mask)
            mid = xb + 0.5 * D
            field[(base[0], base[1], base[2], base[3], mask)] = float(
                popcount(mask) + eps * cvec[mask - 1] * math.cos(float(k @ mid) + phase)
            )
    return field


def fd_check(coup, N, trials, rng):
    incidences, _ = build_complex(N)
    rows = []
    modes = [
        np.array([1.0, 0, 0, 0]),
        np.array([1.0, 1, 0, 0]),
        np.array([1.0, 2, 0, 0]),
        np.array([1.0, 1, 1, 0]),
        np.array([2.0, 1, 0, 1]),
        np.array([1.0, 1, 1, 1]),
    ]
    for t in range(trials):
        m_mode = modes[t % len(modes)]
        cvec = rng.normal(size=15)
        cvec /= np.linalg.norm(cvec)
        phase = float(rng.uniform(0, 2 * math.pi))
        k = [mp.mpf(2) * mp.pi * mp.mpf(float(m_mode[i])) / N for i in range(4)]
        M = bloch_matrix(coup, k)
        cm = vec([mp.mpf(float(x)) for x in cvec])
        pred = ((mp.mpf(N) ** 4 / 2) * (cm.T * M * cm)[0]).real

        def f(e):
            S, _ = action_from_field(strain_field(N, e, cvec, m_mode, phase), incidences, N)
            return S

        # eps = 3e-3: double-precision roundoff in the nonlinear action scales as
        # eps^-2 (it dominated at eps = 1e-4, giving ~1e-5); Richardson removes eps^2.
        d2_big, d2_small, _ = central_second(f, 3e-3)
        rich = (4 * d2_small - d2_big) / 3
        rows.append(
            {
                "m_mode": m_mode.tolist(),
                "fd_d2S": rich,
                "bloch_d2S": float(pred),
                "rel_err": abs(rich - float(pred)) / max(abs(float(pred)), 1e-30),
            }
        )
        print(f"  FD {t+1}/{trials}: m={m_mode.tolist()} fd={rich:.10e} bloch={float(pred):.10e} rel={rows[-1]['rel_err']:.2e}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 8. Phenomenology numbers
# ---------------------------------------------------------------------------


def phenomenology():
    """SME d=6 map and Kostelecky-Tasson Cherenkov reach.

    kappa(n) = (1 + A)/24, A = sum n_i^4; 1 - v_ph = kappa (k a)^2.
    KT: n_refr = 1 + (1/2) s6(n) |l|^2  =>  s6(n) = 2 kappa a^2 = (1 + A) a^2 / 12.
    Spherical: s6(n) = sum_jm Y_jm(n) s6_jm  (isotropic part s6_00 / sqrt(4 pi)).
    A - 3/5 = c4 [Y_40 + sqrt(5/14)(Y_44 + Y_4-4)] with c4 = (16 sqrt(pi))/(15 ... )
    (computed numerically below).
    Bound (KT eq. 44): s6(p) < sqrt(F_psi(6)/(G_N E_f^7 L)), with the calligraphic
    F_psi(d) = (d-2)(d-3)(2d-3)/(4(2d^2-7d+9)) of KT eq. (43) entering directly
    (KT eq. (25) defines it as (2d-5)/F^w, and (43) already IS the calligraphic one).
    F_psi(6) = 9/13.  Check: KT's conservative isotropic bound s6_00 < 2e-31 GeV^-2
    is reproduced from Fly's Eye 320 EeV, iron N=56, r=0.1 (E_f = E/560), L = 10 Mpc,
    and s6_00 = sqrt(4 pi) s6(p) (only the j=0 harmonic kept).
    """
    mp.mp.dps = 30
    GeV_per_m_inv = mp.mpf("5.067730716e15")  # 1 m = 5.0677e15 GeV^-1 (hbar c)
    Mpc_m = mp.mpf("3.0856775814913673e22")
    EP = mp.mpf("1.220890e19")  # GeV, CODATA 2018
    GN = 1 / EP ** 2
    Fpsi6 = mp.mpf(4 * 3 * 9) / (4 * (72 - 42 + 9))
    Fprime = Fpsi6  # numerator of KT eq. (44)

    # spherical decomposition of A(n) = sum n_i^4
    def A_of(th, ph):
        x, y, z = mp.sin(th) * mp.cos(ph), mp.sin(th) * mp.sin(ph), mp.cos(th)
        return x ** 4 + y ** 4 + z ** 4

    def Y40(th):
        c = mp.cos(th)
        return (3 / (16 * mp.sqrt(mp.pi))) * (35 * c ** 4 - 30 * c ** 2 + 3)

    def Y44re(th, ph):
        # Re Y_44 = (Y_44 + Y_4-4)/2 = (3/16) sqrt(35/(2 pi)) sin^4 cos 4phi.  Since A is real
        # and even in phi, <A, Re Y_44> = <A, Y_44>: the coefficient below is the standard
        # complex-basis s_44 (= s_4,-4).  The unit-normalized real harmonic is sqrt2 * Re Y_44,
        # whose coefficient is sqrt2 times this.
        return (3 / 16) * mp.sqrt(35 / (2 * mp.pi)) * mp.sin(th) ** 4 * mp.cos(4 * ph)

    mean_A = mp.quad(lambda th, ph: A_of(th, ph) * mp.sin(th), [0, mp.pi], [0, 2 * mp.pi]) / (4 * mp.pi)
    c40 = mp.quad(lambda th, ph: A_of(th, ph) * Y40(th) * mp.sin(th), [0, mp.pi], [0, 2 * mp.pi])
    c44 = mp.quad(lambda th, ph: A_of(th, ph) * Y44re(th, ph) * mp.sin(th), [0, mp.pi], [0, 2 * mp.pi])
    out = {
        "mean_A": fmt(mean_A),
        "A_Y40_coefficient": fmt(c40),
        "A_Y44re_coefficient": fmt(c44),
        "ratio_c44_over_c40": fmt(c44 / c40),
        "sqrt_5_over_14_times_sqrt2": fmt(mp.sqrt(mp.mpf(5) / 14) * mp.sqrt(2)),
        "F_psi_6": fmt(Fpsi6),
        "F_prime_6": fmt(Fprime),
    }
    # s6 spherical coefficients per a^2:  s6(n) = (1 + A) a^2 / 12
    s00_per_a2 = mp.sqrt(4 * mp.pi) * (1 + mean_A) / 12
    s40_per_a2 = c40 / 12
    s44re_per_a2 = c44 / 12
    out["s6_00_per_a2"] = fmt(s00_per_a2)
    out["s6_40_per_a2"] = fmt(s40_per_a2)
    out["s6_44re_per_a2"] = fmt(s44re_per_a2)
    # Reproduce KT's conservative isotropic bound from their stated inputs.
    L10 = mp.mpf(10) * Mpc_m * GeV_per_m_inv
    Ef_KT = mp.mpf("3.2e20") / 560 / mp.mpf("1e9")  # GeV
    s_dir_KT = mp.sqrt(Fprime / (GN * L10 * Ef_KT ** 7))  # bound on s6(p-hat)
    out["KT_reproduction"] = {
        "Ef_GeV": fmt(Ef_KT), "L_Mpc": 10, "F_psi_6": fmt(Fprime),
        "s6_direction_bound_GeV-2": fmt(s_dir_KT),
        "s6_00_bound_GeV-2_from_sqrt4pi": fmt(mp.sqrt(4 * mp.pi) * s_dir_KT),
        "KT_published_s6_00": "2e-31",
    }
    # Lattice bound.  Direction bound s6(p) < s_dir applied in the weakest lattice
    # direction (body diagonal, 1+A = 4/3) gives the direction-independent a_max;
    # the isotropic-coefficient reading uses s6_00 = s00_per_a2 a^2 < 2e-31.
    a2_max_body = 12 * s_dir_KT / (1 + mp.mpf(1) / 3)
    a2_max_axis = 12 * s_dir_KT / 2
    a2_max = mp.mpf("2e-31") / s00_per_a2
    out["KT_s00_bound_GeV-2"] = "2e-31"
    out["a_max_isotropic_GeV-1"] = fmt(mp.sqrt(a2_max))
    out["a_max_isotropic_m"] = fmt(mp.sqrt(a2_max) / GeV_per_m_inv)
    out["a_max_body_direction_GeV-1"] = fmt(mp.sqrt(a2_max_body))
    out["a_max_body_direction_m"] = fmt(mp.sqrt(a2_max_body) / GeV_per_m_inv)
    out["a_max_body_direction_over_lP"] = fmt(mp.sqrt(a2_max_body) * EP)
    out["a_max_axis_direction_m"] = fmt(mp.sqrt(a2_max_axis) / GeV_per_m_inv)
    out["a_max_axis_direction_over_lP"] = fmt(mp.sqrt(a2_max_axis) * EP)
    out["inv_a_min_GeV_body"] = fmt(1 / mp.sqrt(a2_max_body))
    # spacings to evaluate
    spacings = {
        "planck_a=1/EP": 1 / EP,
        "sqrt_pi_over_EP": mp.sqrt(mp.pi) / EP,
    }
    rows = {}
    for name, a in spacings.items():
        a2 = a * a
        s_axis = (1 + 1) * a2 / 12
        s_body = (1 + mp.mpf(1) / 3) * a2 / 12
        s_mean = (1 + mean_A) * a2 / 12
        r = {
            "a_GeV-1": fmt(a),
            "a_m": fmt(a / GeV_per_m_inv),
            "s6_axis_GeV-2": fmt(s_axis),
            "s6_body_GeV-2": fmt(s_body),
            "s6_mean_GeV-2": fmt(s_mean),
            "s6_00_GeV-2": fmt(s00_per_a2 * a2),
            "margin_vs_KT_conservative": fmt(mp.mpf("2e-31") / (s00_per_a2 * a2)),
            "margin_direction_axis_vs_KT": fmt(s_dir_KT / s_axis),
        }
        for L_Mpc in (10, 100):
            L = mp.mpf(L_Mpc) * Mpc_m * GeV_per_m_inv
            for lab, s in (("axis", s_axis), ("body", s_body)):
                Ef = (Fprime / (GN * L * s * s)) ** (mp.mpf(1) / 7)
                r[f"Ef_threshold_{lab}_{L_Mpc}Mpc_GeV"] = fmt(Ef)
                r[f"Ef_threshold_{lab}_{L_Mpc}Mpc_eV"] = fmt(Ef * mp.mpf("1e9"))
        rows[name] = r
    out["spacings"] = rows
    # excluded spacing for a given parton energy and distance (axis direction, A=1)
    lP = 1 / EP
    excl = {}
    for Ef_eV, lab in ((mp.mpf("3.2e19"), "FlyEye_320EeV_r0.1"), (mp.mpf("1.0e20"), "100EeV_parton"),
                       (mp.mpf("3.2e20"), "FlyEye_320EeV_r1"), (mp.mpf("3.2e20") / 560, "FlyEye_320EeV_iron_r0.1")):
        Ef = Ef_eV / mp.mpf("1e9")
        for L_Mpc in (10, 100):
            L = mp.mpf(L_Mpc) * Mpc_m * GeV_per_m_inv
            s_thr = mp.sqrt(Fprime / (GN * L * Ef ** 7))
            a_thr = mp.sqrt(12 * s_thr / 2)  # axis direction: s = (1+1) a^2 / 12
            excl[f"{lab}_{L_Mpc}Mpc"] = {"Ef_eV": fmt(Ef_eV), "s_thr_GeV-2": fmt(s_thr), "a_thr_over_lP": fmt(a_thr / lP), "a_thr_m": fmt(a_thr / GeV_per_m_inv)}
    out["excluded_spacing_above"] = excl
    # GW150914 dispersion bound on d=6 (Kostelecky-Mewes 2016): |k6| < 1e-9 m^2
    out["KM2016_1e-9_m2_in_GeV-2"] = fmt(mp.mpf("1e-9") * GeV_per_m_inv ** 2)
    # Cherenkov kinematics at Planck spacing: proton 1e20 eV
    Ep = mp.mpf("1e11")
    mp_GeV = mp.mpf("0.938272")
    out["proton_1e20eV_one_minus_v"] = fmt(mp_GeV ** 2 / (2 * Ep ** 2))
    for name, a in spacings.items():
        kmin = mp.sqrt((mp_GeV ** 2 / (2 * Ep ** 2)) / (mp.mpf(1) / 12)) / a  # axis direction kappa=1/12
        out[f"graviton_emission_opens_above_GeV_{name}"] = fmt(kmin)
    mp.mp.dps = 40
    return out


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.time()
    random.seed(20260903)
    print("=== couplings from exact gradients ===", flush=True)
    coup, dD15, dA15 = build_couplings()
    masks15 = sum(1 for (m, m2, _) in coup if 15 in (m, m2))
    print(f"ordered couplings={len(coup)}; couplings touching mask 15: {masks15}; "
          f"hinge gradients: dDelta/da_15 nonzero entries={dD15}, dA/da_15 nonzero entries={dA15}", flush=True)
    # symmetry of the coupling table under Delta -> -Delta
    asym = mp.mpf(0)
    for (m, m2, d2), w in coup.items():
        nd = tuple(-x for x in d2)
        asym = max(asym, abs(w - coup.get((m, m2, nd), mp.mpf(0))))
    print(f"R2 coupling table |w(m,m',D) - w(m,m',-D)| max = {mp.nstr(asym, 5)}", flush=True)
    # coupling census by class pair
    census = defaultdict(int)
    for (m, m2, d2) in coup:
        census[(popcount(m), popcount(m2))] += 1
    print("coupling census by (popcount m, popcount m'):", dict(sorted(census.items())), flush=True)

    e15 = mp.matrix(15, 1)
    e15[14] = 1
    Pm = P_metric_matrix()
    M0 = bloch_matrix(coup, [mp.mpf(0)] * 4)
    print(f"|M0 P| = {mp.nstr(mnorm(M0 * Pm), 5)}  (10 metric zero modes at k=0)", flush=True)

    # R1, R3: kernel dimensions
    kern_rows = []
    for t in range(4):
        k = [mp.mpf(random.uniform(-3, 3)) for _ in range(4)]
        M = bloch_matrix(coup, k)
        sv = sorted(mp.svd_c(M, compute_uv=False))
        me15 = mnorm(M * e15)
        herm = mnorm(M - M.H)
        realpart = mnorm(M - M.T)  # symmetric complex? M real symmetric if even
        imagmax = max(abs(M[i, j].imag) for i in range(15) for j in range(15))
        kern_rows.append({"k": [fmt(x) for x in k], "sv_smallest_7": [mp.nstr(s, 4) for s in sv[:7]],
                          "M_e15": str(me15), "hermiticity": str(herm), "max_imag_entry": str(imagmax)})
        print(f"R1/R3 k={[mp.nstr(x,3) for x in k]}: sv[:7]={[mp.nstr(s,3) for s in sv[:7]]} |M e15|={mp.nstr(me15,3)} max|Im M|={mp.nstr(imagmax,3)}", flush=True)
    sv0 = sorted(mp.svd_c(M0, compute_uv=False))
    print(f"k=0 singular values: {[mp.nstr(s,4) for s in sv0]}", flush=True)

    # metric ⊕ e15 span, complement Q (4 columns)
    Pfull_cols = gram_schmidt([Pm[:, j] for j in range(10)] + [e15])
    assert len(Pfull_cols) == 11
    Q_cols = gram_schmidt(unit_vectors(15), against=Pfull_cols)
    assert len(Q_cols) == 4
    Q = cols(Q_cols)
    P = Pm  # metric columns (Frobenius-orthonormal basis of Sym(4)); e15 has zero row/col
    d0 = Q.T * M0 * Q
    d0_eigs = mp.eigsy(mp.matrix([[d0[i, j].real for j in range(4)] for i in range(4)]))[0]
    print("non-metric block eigenvalues at k=0:", [mp.nstr(x, 12) for x in d0_eigs], flush=True)

    # R4 quadratic form on real directions
    print("=== R4: quadratic TT form on real directions ===", flush=True)
    real_dirs = {
        "axis(1,0,0,0)": [1, 0, 0, 0], "face(1,1,0,0)": [1, 1, 0, 0], "body(1,1,1,0)": [1, 1, 1, 0],
        "diag(1,1,1,1)": [1, 1, 1, 1], "generic(1,2,3,4)": [1, 2, 3, 4], "generic(3,-1,2,0)": [3, -1, 2, 0],
    }
    quad_rows = []
    c2_dev = mp.mpf(0)
    odd_max = mp.mpf(0)
    for name, d in real_dirs.items():
        u = [mp.mpf(x) for x in d]
        nrm = mp.sqrt(sum(x * x for x in u))
        u = [x / nrm for x in u]
        E = effective_series(bloch_taylor(coup, u, ORDER), P, Q, ORDER)
        a, b, c = orthonormal_complement_4d([u])
        pols = {
            "aa-bb": addm(tensor(a, a), tensor(b, b), 1 / SQ2, -1 / SQ2),
            "ab": [[x / SQ2 for x in r] for r in outer_sym(a, b)],
            "ac": [[x / SQ2 for x in r] for r in outer_sym(a, c)],
            "bc": [[x / SQ2 for x in r] for r in outer_sym(b, c)],
            "aa-cc": addm(tensor(a, a), tensor(c, c), 1 / SQ2, -1 / SQ2),
        }
        row = {"dir": name}
        for pn, pol in pols.items():
            x = vec(metric_coords(pol))
            q2 = (x.T * E[2] * x)[0]
            q4 = (x.T * E[4] * x)[0]
            row[f"E2_{pn}"] = fmt(q2)
            row[f"E4_euclid_{pn}"] = fmt(q4)
            c2_dev = max(c2_dev, abs(q2 + mp.mpf(1) / 4))
        E2h = mp.matrix([[E[2][i, j] for j in range(10)] for i in range(10)])
        row["E2_spectrum"] = [mp.nstr(z.real, 10) for z in sorted(mp.eigh(E2h)[0], key=lambda z: z.real)]
        odd_max = max(odd_max, mnorm(E[1]), mnorm(E[3]))
        gres = max(mnorm(E[2] * vec(metric_coords(outer_sym(u, [1 if i == v else 0 for i in range(4)])))) for v in range(4))
        row["E2_gauge_annihilation_max"] = str(gres)
        quad_rows.append(row)
        print(f"  {name}: E2 on TT pols = {[row[f'E2_{pn}'] for pn in pols]} ; E4(euclid) = {[row[f'E4_euclid_{pn}'] for pn in pols]}", flush=True)
        print(f"     E2 spectrum {row['E2_spectrum']}  gauge {mp.nstr(gres,3)}", flush=True)
    print(f"R4: max |E2_TT + 1/4| = {mp.nstr(c2_dev, 5)};  R2: max |E1|,|E3| over real dirs = {mp.nstr(odd_max, 5)}", flush=True)

    # R11: E2(u) = -(1/4) G(u) as a polynomial identity in u in C^4, where G is the
    # linearized Einstein operator in Fourier space,
    #   h'.G(u)h = (u.u)<h',h> - 2 (h'u).(hu) + tr h (u h' u) + tr h' (u h u) - (u.u) tr h tr h',
    # in the Frobenius-orthonormal basis of Sym(4).  Both sides are quadratic matrix
    # polynomials (10 coefficients per entry); agreement at 12 points whose Veronese
    # matrix has rank 10 is a polynomial identity.
    print("=== R11: E2(u) = -G(u)/4 at random complex u (polynomial identity) ===", flush=True)
    sym_basis = []
    for (a_, b_) in SYM_PAIRS:
        Bm = [[mp.mpf(0)] * 4 for _ in range(4)]
        if a_ == b_:
            Bm[a_][a_] = mp.mpf(1)
        else:
            Bm[a_][b_] = Bm[b_][a_] = 1 / SQ2
        sym_basis.append(Bm)

    def G_matrix(u):
        uu = sum(x * x for x in u)

        def mv(B):
            return [sum(B[i][j] * u[j] for j in range(4)) for i in range(4)]

        def tr(B):
            return sum(B[i][i] for i in range(4))

        def uBu(B):
            return sum(u[i] * B[i][j] * u[j] for i in range(4) for j in range(4))

        G = mp.matrix(10, 10)
        for i_, Bi in enumerate(sym_basis):
            for j_, Bj in enumerate(sym_basis):
                ip = sum(Bi[p][q] * Bj[p][q] for p in range(4) for q in range(4))
                G[i_, j_] = (uu * ip - 2 * sum(x * y for x, y in zip(mv(Bi), mv(Bj)))
                             + tr(Bj) * uBu(Bi) + tr(Bi) * uBu(Bj) - uu * tr(Bi) * tr(Bj))
        return G

    r11_dev = mp.mpf(0)
    r11_pts = []
    ver = mp.matrix(12, 10)
    for t in range(12):
        u = [mp.mpc(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)) for _ in range(4)]
        E2u = effective_series(bloch_taylor(coup, u, 2), P, Q, 2)[2]
        dev = mnorm(E2u + G_matrix(u) / 4)
        r11_dev = max(r11_dev, dev)
        for j_, (a_, b_) in enumerate(SYM_PAIRS):
            ver[t, j_] = u[a_] * u[b_]
        r11_pts.append({"u": [fmt(x) for x in u], "dev": fmt(dev), "E2_norm": fmt(mnorm(E2u))})
        print(f"  u={[mp.nstr(x,4) for x in u]}  |E2 + G/4| = {mp.nstr(dev,4)}  |E2| = {mp.nstr(mnorm(E2u),4)}", flush=True)
    ver_sv = sorted([abs(s) for s in mp.svd_c(ver, compute_uv=False)])
    print(f"R11: max |E2(u) + G(u)/4| over 12 complex points = {mp.nstr(r11_dev,5)}; Veronese 12x10 smallest singular value = {mp.nstr(ver_sv[0],5)} (rank 10 iff > 0)", flush=True)

    # R5 on-shell frames: random orthonormal (T, n)
    print("=== R5: on-shell TT quartic form, random frames ===", flush=True)
    r5 = []
    r5_dev = mp.mpf(0)
    r5_gauge = mp.mpf(0)
    r5_f3 = mp.mpf(0)
    r5_f2tt = mp.mpf(0)
    for t in range(12):
        v1 = [mp.mpf(random.gauss(0, 1)) for _ in range(4)]
        v2 = [mp.mpf(random.gauss(0, 1)) for _ in range(4)]
        Tn = gram_schmidt([vec(v1), vec(v2)])
        T = [Tn[0][i] for i in range(4)]
        n = [Tn[1][i] for i in range(4)]
        r = frame_dispersion(coup, P, Q, T, n)
        r5.append(r)
        r5_dev = max(r5_dev, mp.mpf(r["Lambda_minus_pred_max"]))
        r5_gauge = max(r5_gauge, mp.mpf(r["E2_gauge_annihilation_max"]), mp.mpf(r["tau_E4_gauge_max"]))
        r5_f3 = max(r5_f3, mp.mpf(r["F3_max"]))
        r5_f2tt = max(r5_f2tt, mp.mpf(r["F2_tt_max"]), mp.mpf(r["F2_tc_max"]))
        print(f"  frame {t}: Lambda={r['Lambda']}  pred sum u^4/48={r['sum_u4_over_48']}  dev={r['Lambda_minus_pred_max'][:9]}", flush=True)
    print(f"R5: max |Lambda - (sum u^4/48) I| = {mp.nstr(r5_dev,5)}; |E2 g0|,|tau E4 g0| {mp.nstr(r5_gauge,5)}; |F3| {mp.nstr(r5_f3,5)}; |F2_tt|,|F2_tc| on shell {mp.nstr(r5_f2tt,5)}", flush=True)

    # named frames
    print("=== named frames: kappa_phase = 2 Lambda ===", flush=True)
    sqrt2, sqrt3, sqrt6, sqrt14 = mp.sqrt(2), mp.sqrt(3), mp.sqrt(6), mp.sqrt(14)
    named = {
        "axis-time n=(1,0,0)": ([1, 0, 0, 0], [0, 1, 0, 0]),
        "axis-time n=(1,1,0)/r2": ([1, 0, 0, 0], [0, 1 / sqrt2, 1 / sqrt2, 0]),
        "axis-time n=(1,1,1)/r3": ([1, 0, 0, 0], [0, 1 / sqrt3, 1 / sqrt3, 1 / sqrt3]),
        "axis-time n=(1,2,3)/r14": ([1, 0, 0, 0], [0, 1 / sqrt14, 2 / sqrt14, 3 / sqrt14]),
        "face-time T=(1,1,0,0)/r2 n=(1,-1,0,0)/r2": ([1 / sqrt2, 1 / sqrt2, 0, 0], [1 / sqrt2, -1 / sqrt2, 0, 0]),
        "face-time T=(1,1,0,0)/r2 n=(0,0,1,0)": ([1 / sqrt2, 1 / sqrt2, 0, 0], [0, 0, 1, 0]),
        "face-time T=(1,1,0,0)/r2 n=(1,-1,2,0)/r6": ([1 / sqrt2, 1 / sqrt2, 0, 0], [1 / sqrt6, -1 / sqrt6, 2 / sqrt6, 0]),
        "body-time T=(1,1,1,0)/r3 n=(1,-1,0,0)/r2": ([1 / sqrt3, 1 / sqrt3, 1 / sqrt3, 0], [1 / sqrt2, -1 / sqrt2, 0, 0]),
        "body-time T=(1,1,1,0)/r3 n=(1,1,-2,0)/r6": ([1 / sqrt3, 1 / sqrt3, 1 / sqrt3, 0], [1 / sqrt6, 1 / sqrt6, -2 / sqrt6, 0]),
        "diag-time T=(1,1,1,1)/2 n=(1,-1,0,0)/r2": ([mp.mpf(1) / 2] * 4, [1 / sqrt2, -1 / sqrt2, 0, 0]),
        "diag-time T=(1,1,1,1)/2 n=(1,1,-2,0)/r6": ([mp.mpf(1) / 2] * 4, [1 / sqrt6, 1 / sqrt6, -2 / sqrt6, 0]),
    }
    named_rows = {}
    for name, (T, n) in named.items():
        r = frame_dispersion(coup, P, Q, T, n)
        named_rows[name] = r
        print(f"  {name}: kappa = {r['kappa_phase']} ({r['kappa_rational']})  dev {r['Lambda_minus_pred_max'][:8]}", flush=True)

    # R6 admissible time axes: Im sum u^4 over random n ⊥ T
    print("=== R6: reality of the quartic term for candidate time axes ===", flush=True)
    axes = {
        "axis (1,0,0,0)": [1, 0, 0, 0],
        "face (1,1,0,0)/r2": [1 / sqrt2, 1 / sqrt2, 0, 0],
        "body (1,1,1,0)/r3": [1 / sqrt3, 1 / sqrt3, 1 / sqrt3, 0],
        "hyperbody (1,1,1,1)/2": [mp.mpf(1) / 2] * 4,
        "generic (0.3,0.9,0.2,0.27)": [mp.mpf("0.3"), mp.mpf("0.9"), mp.mpf("0.2"), mp.mpf("0.27")],
    }
    r6 = {}
    for name, T in axes.items():
        Tv = vec([mp.mpf(x) for x in T])
        Tv = Tv / mp.sqrt((Tv.T * Tv)[0])
        worst = mp.mpf(0)
        minre = None
        maxre = None
        for _ in range(200):
            nv = vec([mp.mpf(random.gauss(0, 1)) for _ in range(4)])
            nv = gram_schmidt([nv], against=[Tv])[0]
            s4 = sum((1j * Tv[i] + nv[i]) ** 4 for i in range(4))
            worst = max(worst, abs(s4.imag))
            minre = s4.real if minre is None else min(minre, s4.real)
            maxre = s4.real if maxre is None else max(maxre, s4.real)
        r6[name] = {"max_Im_sum_u4": str(worst), "min_Re": fmt(minre), "max_Re": fmt(maxre)}
        print(f"  {name}: max |Im sum u^4| = {mp.nstr(worst,4)};  Re range [{mp.nstr(minre,5)}, {mp.nstr(maxre,5)}]  (kappa = Re/24)", flush=True)

    # R7 nonperturbative roots, axis time
    print("=== R7: exact-operator TT roots, T = axis ===", flush=True)
    r7 = {}
    dirs7 = {
        "n=(1,0,0)": [0, 1, 0, 0],
        "n=(1,1,0)/r2": [0, 1 / sqrt2, 1 / sqrt2, 0],
        "n=(1,1,1)/r3": [0, 1 / sqrt3, 1 / sqrt3, 1 / sqrt3],
        "n=(1,2,3)/r14": [0, 1 / sqrt14, 2 / sqrt14, 3 / sqrt14],
    }
    for name, n in dirs7.items():
        A = sum(mp.mpf(x) ** 4 for x in n)
        kap_pred = (1 + A) / 24
        rows = []
        kmeas = {}
        for kmag in ("0.4", "0.2", "0.1", "0.05"):
            omega, resid, sv = tt_root(coup, P, Q, [1, 0, 0, 0], n, kmag, kap_pred)
            km = mp.mpf(kmag)
            kap_meas = (km - omega.real) / km ** 3
            # nearest-neighbour Laplacian root: 4 sinh^2(w/2) = sum 4 sin^2(k_i/2)
            rhs = sum(4 * mp.sin(km * mp.mpf(x) / 2) ** 2 for x in n)
            w_lap = 2 * mp.asinh(mp.sqrt(rhs) / 2)
            rows.append({"k": kmag, "omega": fmt(omega), "resid": str(abs(resid)), "sv_E": sv,
                         "kappa_meas": fmt(kap_meas), "omega_minus_laplacian": fmt(omega.real - w_lap)})
            kmeas[kmag] = kap_meas
            print(f"  {name} k={kmag}: omega*={fmt(omega)}  kappa_meas={mp.nstr(kap_meas,12)}  Im/Re={mp.nstr(abs(omega.imag)/abs(omega.real),3)}  omega*-omega_lap={mp.nstr(omega.real - w_lap,6)}  sv(E)[:7]={sv}", flush=True)
        rich = (4 * kmeas["0.05"] - kmeas["0.1"]) / 3
        print(f"  {name}: predicted kappa=(1+A)/24={mp.nstr(kap_pred,15)}  Richardson(0.1,0.05)={mp.nstr(rich,15)}  diff={mp.nstr(rich-kap_pred,3)}", flush=True)
        r7[name] = {"A": fmt(A), "kappa_pred": fmt(kap_pred), "rows": rows, "kappa_richardson": fmt(rich)}

    # R7b exact roots at finite k in several frames vs the nearest-neighbour
    # Laplacian symbol L(u) = sum_mu 4 sin^2(u_mu/2) = 0 (complex roots allowed)
    print("=== R7b: exact TT roots vs Laplacian roots, finite k, several frames ===", flush=True)

    def Lsym(u):
        return sum(4 * mp.sin(x / 2) ** 2 for x in u)

    frames7b = {
        "axis T, n=(1,2,3)/r14": ([1, 0, 0, 0], [0, 1 / sqrt14, 2 / sqrt14, 3 / sqrt14]),
        "face T=(1,1,0,0)/r2, n=(1,-1,2,0)/r6": ([1 / sqrt2, 1 / sqrt2, 0, 0], [1 / sqrt6, -1 / sqrt6, 2 / sqrt6, 0]),
        "face T=(1,1,0,0)/r2, n=(1,-1,0,0)/r2": ([1 / sqrt2, 1 / sqrt2, 0, 0], [1 / sqrt2, -1 / sqrt2, 0, 0]),
        "body T=(1,1,1,0)/r3, n=(1,1,-2,0)/r6": ([1 / sqrt3, 1 / sqrt3, 1 / sqrt3, 0], [1 / sqrt6, 1 / sqrt6, -2 / sqrt6, 0]),
    }
    r7b = {}
    r7b_maxdiff = mp.mpf(0)
    for name, (T, n) in frames7b.items():
        T = [mp.mpf(x) for x in T]
        n = [mp.mpf(x) for x in n]
        _, tau, gauge, comp = frame_vectors(T, n)
        W = cols(tau + comp)
        rows = []
        for kmag in ("0.3", "1.0", "1.8", "2.5"):
            km = mp.mpf(kmag)
            fl = lambda w: Lsym([1j * w * T[i] + km * n[i] for i in range(4)])

            def lam(w):
                u = [1j * w * T[i] + km * n[i] for i in range(4)]
                Ee = effective_exact(coup, P, Q, u)
                ev = mp.eig(schur_tt(W.T * Ee * W))[0]
                return min(ev, key=abs)

            try:
                wl = mp.findroot(fl, mp.mpc(km, 0))
                we = mp.findroot(lam, wl, tol=mp.mpf("1e-30"))
                u = [1j * we * T[i] + km * n[i] for i in range(4)]
                sv = sorted(mp.svd_c(effective_exact(coup, P, Q, u), compute_uv=False))
                diff = abs(we - wl)
                r7b_maxdiff = max(r7b_maxdiff, diff)
                rows.append({"k": kmag, "omega_laplacian": fmt(wl), "omega_exact": fmt(we), "diff": str(diff), "sv_E_4_to_6": [mp.nstr(s, 4) for s in sv[4:7]]})
                print(f"  {name} k={kmag}: omega_lap={fmt(wl)} omega_exact={fmt(we)} diff={mp.nstr(diff,3)} sv[4:7]={rows[-1]['sv_E_4_to_6']}", flush=True)
            except Exception as ex:  # root finder failure is reported, not hidden
                rows.append({"k": kmag, "error": str(ex)[:120]})
                print(f"  {name} k={kmag}: root search failed: {str(ex)[:100]}", flush=True)
        r7b[name] = rows
    print(f"R7b: max |omega_exact - omega_laplacian| over frames and k = {mp.nstr(r7b_maxdiff, 4)}", flush=True)

    # R10: single-branch check on random complex u.  Nullity of the full 15x15
    # M(u) must be 5 (4 gauge + hyperbody) whenever L(u) != 0 and 7 when L(u) = 0;
    # the body-diagonal block must stay invertible.
    print("=== R10: nullity of M(u) at random complex u vs L(u) ===", flush=True)
    r10 = []
    rngc = np.random.default_rng(7)
    body_idx = [6, 10, 12, 13]  # masks 7, 11, 13, 14 -> 0-based
    def Lfun(u):
        return sum(4 * mp.sin(ui / 2) ** 2 for ui in u)
    for trial in range(8):
        re = rngc.uniform(-2.5, 2.5, 4)
        im = rngc.uniform(-1.5, 1.5, 4)
        u = [mp.mpc(float(re[i]), float(im[i])) for i in range(4)]
        if trial >= 4:  # put u on the branch L(u)=0 by solving for u4
            s3 = sum(mp.sin(ui / 2) ** 2 for ui in u[:3])
            u[3] = 2 * mp.asin(1j * mp.sqrt(s3))
        Mu = bloch_matrix(coup, u)
        sv = mp.svd_c(Mu, compute_uv=False)
        svl = sorted([abs(s) for s in sv])
        Dblk = mp.matrix(4, 4)
        for a_, ia in enumerate(body_idx):
            for b_, ib in enumerate(body_idx):
                Dblk[a_, b_] = Mu[ia, ib]
        deig = mp.eig(Dblk, left=False, right=False)
        Lu = Lfun(u)
        nnull = sum(1 for s in svl if s < mp.mpf("1e-30"))
        r10.append({"u": [mp.nstr(x, 6) for x in u], "L": mp.nstr(Lu, 6), "nullity": nnull,
                    "sv_low8": [mp.nstr(s, 3) for s in svl[:8]], "min_abs_eig_body_block": mp.nstr(min(abs(e) for e in deig), 6)})
        print(f"  trial {trial}: L(u)={mp.nstr(Lu, 5)} nullity={nnull} sv[:8]={[mp.nstr(s, 3) for s in svl[:8]]} min|eig D|={mp.nstr(min(abs(e) for e in deig), 5)}", flush=True)

    # R8 finite differences
    print("=== R8: finite-difference cross-check of M(k) on the N=4 torus ===", flush=True)
    rng = np.random.default_rng(20260903)
    fd_rows = fd_check(coup, 4, 6, rng)
    fd_max = max(r["rel_err"] for r in fd_rows)

    # R9 phenomenology
    print("=== R9: phenomenology ===", flush=True)
    ph = phenomenology()
    print(json.dumps(ph, indent=1), flush=True)

    out = {
        "script": "original/regge_4d_quartic_tt_dispersion_20260903.py",
        "date": "2026-09-03",
        "mp_dps": mp.mp.dps,
        "n_ordered_couplings": len(coup),
        "couplings_touching_mask15": masks15,
        "dDelta_da15_nonzero_entries": dD15,
        "dA_da15_nonzero_entries": dA15,
        "coupling_table_reflection_asymmetry": str(asym),
        "coupling_census_by_popcount_pair": {f"{k[0]},{k[1]}": v for k, v in census.items()},
        "kernel_rows": kern_rows,
        "k0_singular_values": [str(s) for s in sv0],
        "nonmetric_block_eigs_k0": [str(x) for x in d0_eigs],
        "R4_max_dev_E2_TT_plus_quarter": str(c2_dev),
        "R2_max_odd_order": str(odd_max),
        "quadratic_rows": quad_rows,
        "R5_random_frames": r5,
        "R5_max_dev": str(r5_dev),
        "named_frames": named_rows,
        "R6_time_axes": r6,
        "R7_roots": r7,
        "R7b_exact_vs_laplacian": r7b,
        "R7b_max_diff": str(r7b_maxdiff),
        "R10_complex_u_nullity": r10,
        "R11_E2_equals_minus_G_over_4": {"points": r11_pts, "max_dev": str(r11_dev),
                                          "veronese_min_sv": str(ver_sv[0])},
        "fd_check": {"rows": fd_rows, "max_rel_err": fd_max},
        "phenomenology": ph,
        "elapsed_sec": time.time() - t0,
    }
    outp = Path("state/qg_full_theory/regge_4d_quartic_tt_dispersion_20260903.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", outp, f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
