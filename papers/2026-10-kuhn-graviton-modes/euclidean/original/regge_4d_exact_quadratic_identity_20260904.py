#!/usr/bin/env python3
"""Exact (rational, symbolic) certificates for two statements in
papers/Lattice_Graviton_Dispersion_20260903:

  (A) Proposition 3.2 at k = 0: the characteristic polynomial of the 15x15
      symbol M(0) is  lambda^11 (lambda + 2)^3 (lambda + 8),  so the kernel
      has dimension exactly 11.
  (B) Theorem 3.3: the quadratic part of the reduced symbol is
      E2(u) = P^T M2(u) P  exactly (the Schur correction starts at order u^4
      because P^T M(0) = 0 and M is even), and  E2(u) = -(1/4) G(u)  as an
      identity of 10x10 matrices of quadratic polynomials in u in C^4, G the
      linearized Einstein operator.

Input: the exact coupling table written by regge_4d_quartic_tt_dispersion_20260903.py.
All arithmetic is in sympy over Q; no floating point anywhere.
"""
import json
import sys
from pathlib import Path

import sympy as sp

TABLE = Path("state/qg_full_theory/regge_4d_kuhn_coupling_table_20260903.json")
OUT = Path("state/qg_full_theory/regge_4d_exact_quadratic_identity_20260904.json")

CLASSES = [tuple(int(b) for b in f"{m:04b}"[::-1]) for m in range(1, 16)]  # D in {0,1}^4, mask bits
IDX = {D: i for i, D in enumerate(CLASSES)}
SYM_PAIRS = [(a, b) for a in range(4) for b in range(a, 4)]  # h_ab, a<=b (unnormalized coordinates)


def main():
    tab = json.load(open(TABLE))
    u = sp.symbols("u1:5")
    M0 = sp.zeros(15, 15)
    M2 = sp.zeros(15, 15)
    nonrational = 0
    for r in tab["couplings"]:
        w = sp.nsimplify(sp.sympify(r["weight_exact"]), rational=True)
        if not w.is_Rational:
            nonrational += 1
        D = tuple(int(round(x)) for x in r["D"])
        Dp = tuple(int(round(x)) for x in r["Dprime"])
        v = r["two_times_midpoint_separation"]
        i, j = IDX[D], IDX[Dp]
        ph = sum(u[a] * sp.Rational(v[a], 2) for a in range(4))  # k.v/2
        M0[i, j] += w
        M2[i, j] += w * (sp.I * ph) ** 2 / 2  # second Taylor coefficient of w*exp(i ph)
    M2 = sp.expand(M2)
    assert M0 == M0.T

    # (A) exact characteristic polynomial of M(0)
    lam = sp.symbols("lam")
    cp = sp.factor(M0.charpoly(lam).as_expr())
    target = lam**11 * (lam + 2) ** 3 * (lam + 8)
    charpoly_ok = sp.simplify(cp - target) == 0

    # metric strain map in unnormalized h coordinates: da_D = D^T h D
    P = sp.zeros(15, 10)
    for i, D in enumerate(CLASSES):
        for j, (a, b) in enumerate(SYM_PAIRS):
            P[i, j] = D[a] * D[a] if a == b else 2 * D[a] * D[b]
    hyp = IDX[(1, 1, 1, 1)]
    hyp_row_zero = all(M0[hyp, j] == 0 for j in range(15)) and all(M2[hyp, j] == 0 for j in range(15))
    PM0_zero = (P.T * M0) == sp.zeros(10, 15)  # strains are exact null vectors of M(0)

    # (B) E2 = P^T M2 P, compared with -(1/4) G in the same coordinates
    E2 = sp.expand(P.T * M2 * P)
    basis = []
    for (a, b) in SYM_PAIRS:
        B = sp.zeros(4, 4)
        if a == b:
            B[a, a] = 1
        else:
            B[a, b] = B[b, a] = 1
        basis.append(B)
    uv = sp.Matrix(u)
    uu = (uv.T * uv)[0]

    def form(Bi, Bj):
        ip = sum(Bi[p, q] * Bj[p, q] for p in range(4) for q in range(4))
        Biu, Bju = Bi * uv, Bj * uv
        return (uu * ip - 2 * (Biu.T * Bju)[0] + Bj.trace() * (uv.T * Bi * uv)[0]
                + Bi.trace() * (uv.T * Bj * uv)[0] - uu * Bi.trace() * Bj.trace())

    G = sp.Matrix(10, 10, lambda i, j: sp.expand(form(basis[i], basis[j])))
    R = sp.expand(E2 + G / 4)
    identity_ok = R == sp.zeros(10, 10)

    res = {
        "table": str(TABLE), "n_couplings": len(tab["couplings"]), "nonrational_weights": nonrational,
        "A_charpoly_M0": str(cp), "A_charpoly_equals_lam11_lam2_3_lam8": bool(charpoly_ok),
        "hypotenuse_row_zero_in_M0_and_M2": bool(hyp_row_zero),
        "PT_M0_zero": bool(PM0_zero),
        "B_E2_equals_minus_G_over_4_identically": bool(identity_ok),
        "E2_sample_entry_00": str(E2[0, 0]), "G_sample_entry_00": str(G[0, 0]),
    }
    OUT.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k}: {v}")
    ok = charpoly_ok and identity_ok and hyp_row_zero and PM0_zero and nonrational == 0
    print("ALL EXACT CHECKS PASS" if ok else "SOME CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
