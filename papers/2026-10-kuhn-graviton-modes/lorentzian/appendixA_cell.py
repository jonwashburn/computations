#!/usr/bin/env python3
"""
appendixA_cell.py -- the Lorentzian Kuhn cell, from geometry to the assembled symbol.

Companion to: J. Washburn and P. Beltracchi, "Graviton Modes on the Kuhn
Lattice: Euclidean Continuation and Lorentzian Propagation".

Reproduces, in this order:
  Section 6.1  squared intervals n(1 - n), null axis edges, nondegenerate simplices,
               timelike triangles with spacelike normal planes, the angles pi/6 and 2 pi/3
               (Euclidean pi/4 and pi/2), and angle sums of 2 pi around every hinge.
  Appendix A   the chain Gram matrix G_ij = min(i,j) - ij, the Hessian cell C = 2R/sqrt(3)
               with R the printed matrix (eq:lorentz-cell), its symmetry (Schlaefli), the
               assembly rule (eq:lorentz-assembly) and its cosine form.
  Lemma 2.2(iii) in Lorentzian form: the gauge (lattice Bianchi) identity M_L g_xi = 0.
  Appendix B   self-test of the same code at kappa = 0 against the published Euclidean
               302-coupling table, its weight census and its zero hypotenuse row.

Run:  python appendixA_cell.py [outputs/appendixA_cell.json] [path/to/regge_4d_kuhn_coupling_table_20260903.json]
Exact arithmetic throughout (SymPy). Runtime: under a minute.
"""
import itertools
import json
import os
import sys
from collections import Counter
from fractions import Fraction

import sympy as sp

import kuhn_lorentz_geometry as geo
import kuhn_lorentz_symbol as sym
from report import Report, out_path

# eq:lorentz-cell, as printed in Appendix A (rows and columns in slot order 01,02,03,04,12,13,14,23,24,34).
PAPER_32R = [
    [2, -2, 1, 0, 0, 3, -2, -6, 2, 2],
    [-2, -4, 4, -1, 6, -6, 2, 3, -2, 2],
    [1, 4, 0, -2, -6, 0, 2, -3, 2, -2],
    [0, -1, -2, 2, 2, 2, -2, 2, -1, 0],
    [0, 6, -6, 2, -6, 6, -3, 0, 3, -6],
    [3, -6, 0, 2, 6, 0, 0, 6, -6, 3],
    [-2, 2, 2, -2, -3, 0, 0, -6, 4, 1],
    [-6, 3, -3, 2, 0, 6, -6, -6, 6, 0],
    [2, -2, 2, -1, 3, -6, 4, 6, -4, -2],
    [2, 2, -2, 0, -6, 3, 1, 0, -2, 2],
]

# Appendix B, "Exact gradients": the Euclidean weights take only these values, with these counts.
PAPER_EUCLID_WEIGHT_CENSUS = {Fraction(-3, 2): 4, Fraction(-1): 6, Fraction(-1, 2): 4,
                              Fraction(1, 2): 48, Fraction(1, 4): 96, Fraction(-1, 4): 144}
PAPER_EUCLID_LENGTH_CENSUS = {(1, 1): 52, (1, 2): 96, (1, 3): 48, (2, 2): 54, (2, 3): 48, (3, 3): 4}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TABLE = os.path.join(HERE, "..", "euclidean", "reference", "regge_4d_kuhn_coupling_table_20260903.json")


def section_6_1_geometry(rep):
    rep.section("Section 6.1: the Lorentzian simplex on g_L = I - J")
    g = geo.background_metric(1)
    rep.check("g_L^{-1} = I - J/3 (eq:direct-metric)", g.inv() == sp.eye(4) - sp.ones(4, 4) / 3)
    ok = True
    for D in itertools.product((0, 1), repeat=4):
        if any(D):
            n = sum(D)
            ok &= geo.squared_interval(D, g) == n * (1 - n)
    rep.check("every edge D has squared interval n(1 - n), n = number of ones", ok)
    rep.check("the four axis edges are null",
              all(geo.squared_interval(tuple(int(i == k) for i in range(4)), g) == 0 for k in range(4)))

    s = geo.simplex(1)
    G = s.gram(0, [1, 2, 3, 4])
    rep.check("chain Gram matrix G_ij = min(i,j) - ij (Appendix A)",
              G == sp.Matrix(4, 4, lambda i, j: min(i + 1, j + 1) - (i + 1) * (j + 1)), G.tolist())
    rep.value("det G (four-simplex)", G.det())

    tri_dets = {t: s.triangle_gram_det(t) for t in geo.TRIANGLES}
    rep.check("every triangle is timelike: det G_h < 0", all(d < 0 for d in tri_dets.values()), tri_dets)
    tet_ok = True
    for tet in itertools.combinations(range(5), 4):
        tet_ok &= s.gram(tet[0], list(tet[1:])).det() != 0
    rep.check("all five tetrahedra are nondegenerate", tet_ok)
    normals_ok = True
    for t in geo.TRIANGLES:
        N = s.normal_plane_gram(t)
        normals_ok &= N[0, 0] > 0 and N[1, 1] > 0 and N.det() > 0
    rep.check("every hinge normal plane is spacelike (positive-definite normal Gram)", normals_ok)

    ang = {t: s.angle(t) for t in geo.TRIANGLES}
    eang = {t: geo.simplex(0).angle(t) for t in geo.TRIANGLES}
    rep.note("interior angles by hinge (Lorentzian | Euclidean):")
    for t in geo.TRIANGLES:
        rep.note("   hinge %s: %s | %s" % ("".join(map(str, t)), ang[t], eang[t]))
    rep.check("end triangles 012 and 234 of the chain have angle pi/6 (Euclidean pi/4)",
              ang[(0, 1, 2)] == sp.pi / 6 and ang[(2, 3, 4)] == sp.pi / 6
              and eang[(0, 1, 2)] == sp.pi / 4 and eang[(2, 3, 4)] == sp.pi / 4)
    rep.check("middle triangle 123 has angle 2 pi/3 (Euclidean pi/2)",
              ang[(1, 2, 3)] == 2 * sp.pi / 3 and eang[(1, 2, 3)] == sp.pi / 2)

    sums = geo.hinge_angle_sums(1)
    rep.check("angles around each of the 50 hinge classes sum to 2 pi (zero deficits)",
              len(sums) == 50 and all(v[0] == 2 * sp.pi for v in sums.values()),
              "simplices per hinge: %s" % dict(Counter(v[1] for v in sums.values())))
    esums = geo.hinge_angle_sums(0)
    rep.check("Euclidean check: the same 50 sums are 2 pi at kappa = 0",
              all(v[0] == 2 * sp.pi for v in esums.values()))


def appendix_A_cell(rep):
    rep.section("Appendix A: the Hessian cell C = 2R/sqrt(3)")
    s = geo.simplex(1)
    rep.check("single-simplex Schlaefli identity sum_h A_h d(theta_h) = 0 at the flat simplex",
              all(x == 0 for x in s.schlaefli_residual()))
    C = geo.hessian_cell(1)
    rep.check("the cell is symmetric (mixed second derivatives commute)",
              (C - C.T).applyfunc(sp.simplify) == sp.zeros(10, 10))
    R = sym.rational_cell(1)
    rep.check("R = (sqrt 3/2) C is rational", all(x.is_Rational for x in R))
    diff = [((geo.SLOT_NAMES[i], geo.SLOT_NAMES[j]), 32 * R[i, j], PAPER_32R[i][j])
            for i in range(10) for j in range(10) if 32 * R[i, j] != PAPER_32R[i][j]]
    rep.check("32R from the geometry equals the printed matrix (eq:lorentz-cell), all 100 entries",
              not diff, diff if diff else "")
    rep.note("32R from the geometry, rows in slot order " + ",".join(geo.SLOT_NAMES) + ":")
    for i in range(10):
        rep.note("   " + " ".join("%3d" % (32 * R[i, j]) for j in range(10)))
    rep.value("cell C entry (01,01) = 2R/sqrt3", C[0, 0])

    same = all(geo.hessian_cell(1, perm) == C for perm in geo.PERMUTATIONS)
    rep.check("each of the 24 simplices, computed from its own coordinates, has the same cell in slot order",
              same)


def appendix_A_assembly(rep):
    rep.section("Appendix A: assembly rule and the 15 x 15 symbol M_L")
    table, M = sym.lorentz_symbol(1, per_permutation=True)
    R = sym.rational_cell(1)
    table2, M2 = sym.lorentz_symbol(1, per_permutation=False)
    rep.check("assembly from the printed cell rule 2R equals the per-simplex geometric assembly",
              table == table2)
    rep.value("nonzero couplings (D, D', v) of M_L", len(table))
    even, symm = sym.check_real_symmetric(M)
    rep.check("coefficients of v and -v agree, so every entry is a cosine sum (eq:lorentz-assembly)", even)
    rep.check("M_L is symmetric", symm)
    # The cosine form exactly as printed: sum 2 R_{f'f} cos(q.(m_f - m_f')).
    cosform = [[sym.TrigPoly() for _ in range(15)] for _ in range(15)]
    for perm in geo.PERMUTATIONS:
        slots = [geo.slot_vector(perm, f) for f in range(10)]
        for f in range(10):
            for fp in range(10):
                w = 2 * R[fp, f]
                if w != 0:
                    D, m2 = slots[f]
                    Dp, mp2 = slots[fp]
                    term = sym.TrigPoly.cos(tuple(a - b for a, b in zip(m2, mp2)), Fraction(int(w.p), int(w.q)))
                    cosform[sym.INDEX[D]][sym.INDEX[Dp]] = cosform[sym.INDEX[D]][sym.INDEX[Dp]] + term
    rep.check("the printed cosine formula reproduces every entry of the exponential assembly",
              all((cosform[i][j] - M[i][j]).is_zero() for i in range(15) for j in range(15)))
    b = sym.check_bianchi(M)
    rep.check("gauge identity M_L(q) g_xi(q) = 0 for all q, xi = e_1..e_4 (lattice Bianchi)", all(b.values()), b)
    hyp = sym.INDEX[(1, 1, 1, 1)]
    rep.check("unlike the Euclidean case, the hypercube-diagonal row of M_L is not zero",
              not all(M[hyp][j].is_zero() for j in range(15)))
    return table, M


def euclidean_self_test(rep, table_path):
    rep.section("Self-test at kappa = 0 against the published Euclidean coupling table")
    table, M = sym.lorentz_symbol(0, per_permutation=True)
    # The Euclidean cell is used unscaled: M_{DD'} = sum C_{ff'} exp(i q.(m_f' - m_f)).
    rep.value("nonzero Euclidean couplings from this code", len(table))
    rep.check("302 nonzero couplings (Appendix B)", len(table) == 302)
    census = Counter(table.values())
    rep.check("weight census -3/2 (4), -1 (6), -1/2 (4), 1/2 (48), 1/4 (96), -1/4 (144)",
              dict(census) == PAPER_EUCLID_WEIGHT_CENSUS, dict(census))
    lc = Counter(tuple(sorted((sum(D), sum(Dp)))) for (D, Dp, v) in table)
    rep.check("census by squared lengths (1,1):52 (1,2):96 (1,3):48 (2,2):54 (2,3):48 (3,3):4",
              dict(lc) == PAPER_EUCLID_LENGTH_CENSUS, dict(lc))
    rep.check("no Euclidean coupling touches the hypercube diagonal (Theorem 3.1)",
              all(D != (1, 1, 1, 1) and Dp != (1, 1, 1, 1) for (D, Dp, v) in table))
    rep.check("reflection asymmetry max|w(D,D',v) - w(D,D',-v)| is exactly 0",
              all(table.get((D, Dp, tuple(-x for x in v)), 0) == w for (D, Dp, v), w in table.items()))
    if os.path.exists(table_path):
        with open(table_path) as fh:
            ref = json.load(fh)
        refd = {}
        for row in ref["couplings"]:
            key = (tuple(int(x) for x in row["D"]), tuple(int(x) for x in row["Dprime"]),
                   tuple(int(x) for x in row["two_times_midpoint_separation"]))
            refd[key] = Fraction(row["weight_exact"])
        rep.check("every coupling equals the published 302-entry table (same D, D', v and weight)",
                  refd == table, "table: %s" % os.path.basename(table_path))
    else:
        rep.check("published Euclidean table found", False, table_path)


def main(argv):
    rep = Report("Appendix A: the Lorentzian Kuhn cell and its symbol (kappa = 1)")
    section_6_1_geometry(rep)
    appendix_A_cell(rep)
    appendix_A_assembly(rep)
    euclidean_self_test(rep, argv[2] if len(argv) > 2 else DEFAULT_TABLE)
    return rep.finish(out_path(argv, None))


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv) else 0)
