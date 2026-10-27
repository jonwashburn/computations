#!/usr/bin/env python3
"""
section7_source_constraints.py -- the two scalar sources that violate the null-space conditions.

Companion to: J. Washburn and P. Beltracchi, "Graviton Modes on the Kuhn
Lattice: Euclidean Continuation and Lorentzian Propagation", Section 7
("Matter sources must satisfy the gravitational constraints").

This section is Euclidean: the background is the flat unit Kuhn lattice and the
scalar action is eq:joint-action with identity links and one real field component.
The program computes, from the simplex geometry:

  eq:scalar-flat-metric   G_ij = min(i,j), E = G^{-1}, V_sigma = 1/24 on every chain simplex
  eq:scalar-edge-variation  the edge source J_e = dS_m/da_e, by the paper's formula and,
                          independently, by differentiating S_m exactly
  first obstruction       phi = lambda (x1 + x2): the printed B_sigma and delta G, and
                          J_{1111} = -kappa_m lambda^2 / 12 summed over the 24 simplices of one
                          hypercube, while dA_h/da_{1111} = 0 for every hinge
  second obstruction      phi = lambda (x1^3 - 3 x1 x2^2) and the vertex p = (0,1,0,0) moved along
                          xi = e1 - e3: g^T J = -18 kappa_m lambda^2 over the 120 simplices
                          containing p; the same contraction vanishes at the origin; hypercube
                          diagonals at p have D.xi = 0
  Prop. 7.1 inputs        the hypercube-diagonal strain and the vertex-displacement vector are null
                          vectors of the assembled Euclidean Hessian H (checked in position space)

Results are reported in units kappa_m = lambda = 1 (both sources are quadratic in lambda and
linear in kappa_m). Exact arithmetic (SymPy). Runtime: under a minute.
Run:  python section7_source_constraints.py [outputs/section7_source_constraints.json]
"""
import itertools
import sys
from collections import defaultdict

import sympy as sp

import kuhn_lorentz_geometry as geo
from report import Report, out_path

EUCLID = 0       # kappa = 0 in g = I - kappa J: the Euclidean unit Kuhn lattice


def lattice_simplices_containing(p):
    """All Kuhn simplices (base y, ordering perm) with p among their five vertices."""
    out = []
    for d in itertools.product((0, 1), repeat=4):
        y = tuple(pi - di for pi, di in zip(p, d))
        for perm in geo.PERMUTATIONS:
            verts = [tuple(int(c) + yy for c, yy in zip(v, y)) for v in geo.chain_vertices(perm)]
            if tuple(p) in verts:
                out.append((y, perm, verts))
    return out


def simplex_metric(a):
    """eq:simplex-metric from the ten squared lengths a[(i,j)] of a simplex with ordered vertices 0..4."""
    def sq(i, j):
        return sp.Integer(0) if i == j else sp.sympify(a[(min(i, j), max(i, j))])
    G = sp.Matrix(4, 4, lambda i, j: (sq(0, i + 1) + sq(0, j + 1) - sq(i + 1, j + 1)) / 2)
    return G


def scalar_action_simplex(a, w):
    """S_m^sigma = (kappa_m/2) V_sigma sum_ij E^ij w_i w_j with kappa_m = 1 (eq:joint-action)."""
    G = simplex_metric(a)
    E = G.inv()
    Vs = sp.sqrt(G.det()) / 24
    wv = sp.Matrix(w)
    return sp.Rational(1, 2) * Vs * (wv.T * E * wv)[0, 0]


def flat_squared_lengths(verts):
    return {(i, j): sp.Integer(sum((verts[j][k] - verts[i][k])**2 for k in range(4)))
            for i, j in geo.EDGE_SLOTS}


def edge_source_formula(verts, phi):
    """J for each slot by eq:scalar-edge-variation: (1/2) V [ (1/2) tr(E dG) tr(EB) - tr(E dG E B) ]."""
    a = flat_squared_lengths(verts)
    G = simplex_metric(a)
    E = G.inv()
    Vs = sp.sqrt(G.det()) / 24
    w = [phi(verts[i]) - phi(verts[0]) for i in range(1, 5)]
    B = sp.Matrix(4, 4, lambda i, j: w[i] * w[j])
    out = []
    for slot in geo.EDGE_SLOTS:
        dG = geo.KuhnSimplex.gram_derivative(0, [1, 2, 3, 4], slot)
        out.append(sp.Rational(1, 2) * Vs * (sp.Rational(1, 2) * (E * dG).trace() * (E * B).trace()
                                             - (E * dG * E * B).trace()))
    return out, B


def edge_source_direct(verts, phi):
    """J for each slot by changing that squared length alone, a_f -> a_f + h, and differentiating
    S_m^sigma(h) at h = 0. This does not use the variation formula."""
    h = sp.Symbol("h")
    w = [phi(verts[i]) - phi(verts[0]) for i in range(1, 5)]
    flat = flat_squared_lengths(verts)
    out = []
    for s in geo.EDGE_SLOTS:
        a = dict(flat)
        a[s] = a[s] + h
        out.append(sp.nsimplify(sp.diff(scalar_action_simplex(a, w), h).subs(h, 0)))
    return out


def main(argv):
    rep = Report("Section 7: scalar sources and the null-space conditions (Euclidean)")
    lam = 1

    rep.section("eq:scalar-flat-metric on the unit Kuhn simplex")
    verts = [tuple(int(c) for c in v) for v in geo.chain_vertices(geo.CANONICAL)]
    G = simplex_metric(flat_squared_lengths(verts))
    rep.check("G_ij = min(i,j)", G == sp.Matrix(4, 4, lambda i, j: min(i, j) + 1))
    rep.check("E = G^{-1} is the printed tridiagonal matrix",
              G.inv() == sp.Matrix([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]]))
    rep.check("V_sigma = sqrt(det G)/24 = 1/24", sp.sqrt(G.det()) / 24 == sp.Rational(1, 24))
    rep.check("the same G holds on all 24 chain simplices",
              all(simplex_metric(flat_squared_lengths([tuple(int(c) for c in v) for v in geo.chain_vertices(p)])) == G
                  for p in geo.PERMUTATIONS))

    rep.section("First obstruction: phi = lambda (x1 + x2) and the hypercube diagonal")
    phi1 = lambda v: lam * (v[0] + v[1])
    lap1 = sum(phi1(tuple(sp.Symbol("x%d" % k) + (s if k == mu else 0) for k in range(4)))
               for mu in range(4) for s in (1, -1)) - 8 * phi1(tuple(sp.Symbol("x%d" % k) for k in range(4)))
    rep.check("phi is lattice harmonic (eq:scalar-field-equation with identity links)", sp.expand(lap1) == 0)
    J, B = edge_source_formula(verts, phi1)
    rep.check("B_sigma on the simplex 0,e1,e1+e2,... is the printed matrix",
              B == sp.Matrix([[1, 2, 2, 2], [2, 4, 4, 4], [2, 4, 4, 4], [2, 4, 4, 4]]))
    dG = geo.KuhnSimplex.gram_derivative(0, [1, 2, 3, 4], (0, 4))
    rep.check("delta G for the hypercube diagonal is the printed matrix",
              dG == sp.Matrix([[0, 0, 0, sp.Rational(1, 2)], [0, 0, 0, sp.Rational(1, 2)],
                               [0, 0, 0, sp.Rational(1, 2)], [sp.Rational(1, 2)] * 3 + [1]]))
    total = sp.Integer(0)
    agree = True
    for perm in geo.PERMUTATIONS:
        vv = [tuple(int(c) for c in v) for v in geo.chain_vertices(perm)]
        Jf, _ = edge_source_formula(vv, phi1)
        Jd = edge_source_direct(vv, phi1)
        agree &= all(sp.simplify(p - q) == 0 for p, q in zip(Jf, Jd))
        total += Jf[geo.EDGE_SLOTS.index((0, 4))]
    rep.check("eq:scalar-edge-variation agrees with direct differentiation of S_m on all 24 simplices", agree)
    rep.check("dS_m/da_{1111} = -kappa_m lambda^2/12 summed over the 24 simplices of the hypercube",
              sp.simplify(total + sp.Rational(1, 12)) == 0, total)
    s = geo.simplex(EUCLID)
    rep.check("dA_h/da_{1111} = 0 for every hinge (Theorem 3.1), so the gravitational side is 0",
              all(s.area_gradient(t)[geo.EDGE_SLOTS.index((0, 4))] == 0 for t in geo.TRIANGLES))

    rep.section("Second obstruction: phi = lambda (x1^3 - 3 x1 x2^2), vertex p = (0,1,0,0), xi = e1 - e3")
    phi2 = lambda v: lam * (v[0]**3 - 3 * v[0] * v[1]**2)
    X = tuple(sp.Symbol("x%d" % k) for k in range(4))
    lap2 = sum(phi2(tuple(X[k] + (s if k == mu else 0) for k in range(4)))
               for mu in range(4) for s in (1, -1)) - 8 * phi2(X)
    rep.check("phi is lattice harmonic", sp.expand(lap2) == 0)
    xi = (1, 0, -1, 0)

    def contraction(p):
        sims = lattice_simplices_containing(p)
        tot = sp.Integer(0)
        for y, perm, vv in sims:
            Jf, _ = edge_source_formula(vv, phi2)
            for f, (i, j) in enumerate(geo.EDGE_SLOTS):
                D = tuple(vv[j][k] - vv[i][k] for k in range(4))
                Dxi = sum(d * x for d, x in zip(D, xi))
                if vv[j] == tuple(p):
                    tot += 2 * Dxi * Jf[f]           # p is the head of the edge
                elif vv[i] == tuple(p):
                    tot -= 2 * Dxi * Jf[f]           # p is the tail
        return sp.simplify(tot), len(sims)

    val, n = contraction((0, 1, 0, 0))
    rep.check("120 simplices contain p", n == 120, n)
    rep.check("g^T J = -18 kappa_m lambda^2 (eq:obstruction-gauge)", val == -18, val)
    val0, _ = contraction((0, 0, 0, 0))
    rep.check("the same contraction vanishes at the origin", val0 == 0, val0)
    rep.check("every hypercube diagonal at p has D.xi = 0", sum((1, 1, 1, 1)[k] * xi[k] for k in range(4)) == 0)

    # Independent route: move p by eps*xi in the embedding and differentiate the scalar action.
    eps = sp.Symbol("eps")
    total_eps = sp.Integer(0)
    for y, perm, vv in lattice_simplices_containing((0, 1, 0, 0)):
        pos = [tuple(sp.Integer(c) + (eps * xi[k] if tuple(v) == (0, 1, 0, 0) else 0) for k, c in enumerate(v))
               for v in vv]
        a = {(i, j): sum((pos[j][k] - pos[i][k])**2 for k in range(4)) for i, j in geo.EDGE_SLOTS}
        w = [phi2(vv[i]) - phi2(vv[0]) for i in range(1, 5)]     # fields stay fixed at the vertices
        total_eps += sp.diff(scalar_action_simplex(a, w), eps).subs(eps, 0)
    rep.check("moving the vertex in the embedding and differentiating S_m gives the same -18",
              sp.simplify(total_eps + 18) == 0, sp.simplify(total_eps))

    rep.section("Proposition 7.1 inputs: both vectors are null vectors of the Euclidean Hessian")
    C = geo.hessian_cell(EUCLID)
    # Position-space Hessian contracted with the gauge vector g of the displaced vertex p.
    p = (0, 1, 0, 0)
    Hg = defaultdict(lambda: sp.Integer(0))
    for y, perm, vv in lattice_simplices_containing(p):
        gvec = []
        for f, (i, j) in enumerate(geo.EDGE_SLOTS):
            D = tuple(vv[j][k] - vv[i][k] for k in range(4))
            Dxi = sum(d * x for d, x in zip(D, xi))
            gvec.append(2 * Dxi if vv[j] == p else (-2 * Dxi if vv[i] == p else 0))
        for f, (i, j) in enumerate(geo.EDGE_SLOTS):
            edge = (vv[i], vv[j])
            Hg[edge] += sum(C[f, fp] * gvec[fp] for fp in range(10))
    rep.check("(H g)_e = 0 on every edge e near p: vertex displacements are null vectors",
              all(sp.simplify(v) == 0 for v in Hg.values()), "%d edges" % len(Hg))
    hyp = geo.EDGE_SLOTS.index((0, 4))
    rep.check("the hypercube-diagonal row and column of every Euclidean cell vanish: H n_1111 = 0",
              all(C[hyp, f] == 0 and C[f, hyp] == 0 for f in range(10)))
    return rep.finish(out_path(argv, None))


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv) else 0)
