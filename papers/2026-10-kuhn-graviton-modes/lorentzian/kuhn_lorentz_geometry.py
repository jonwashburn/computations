#!/usr/bin/env python3
"""
kuhn_lorentz_geometry.py -- one Kuhn 4-simplex on the flat background g = I - kappa*J.

Companion to: J. Washburn and P. Beltracchi, "Graviton Modes on the Kuhn
Lattice: Euclidean Continuation and Lorentzian Propagation".
Supports Section 6.1 ("Geometry, time and momentum") and the first half of
Appendix A ("The Lorentzian cell and the paired orbit block").

Everything here starts from vertex coordinates and the background metric.
Nothing is read from a stored table.

  * The 24 Kuhn simplices of a unit hypercube (Section 2.1): for an ordering
    pi of the four axes, v_0 = 0 and v_j = v_{j-1} + e_{pi(j)}.
  * Squared intervals a = D^T g D. For g = I - kappa*J and an edge D in {0,1}^4
    with n ones this is n(1 - kappa*n). At kappa = 1 the four axis edges are null.
  * The edge Gram matrix from a base vertex b: G_ij = (a_bi + a_bj - a_ij)/2.
  * Hinge (triangle) areas A_h = (1/2) sqrt(|det G_h|). In the Lorentzian case
    det G_h < 0 (every triangle is timelike) and this is the paper's
    "one half of the positive square root of minus the triangle Gram determinant".
  * Interior dihedral angles from the inverse Gram matrix X = G^{-1}
    (Appendix B, "Exact gradients"):
        cos(theta) = -X_lm / sqrt(X_ll X_mm),
    where l and m are the two vertices not on the hinge. In the Lorentzian case
    the 2x2 block [[X_ll, X_lm], [X_lm, X_mm]] is positive definite: the normal
    plane of a timelike triangle is spacelike, so theta is an ordinary angle.
  * First derivatives of areas and angles with respect to the ten squared edge
    lengths, using dX = -X dG X. All values are exact algebraic numbers.
  * The per-simplex Hessian cell, equation (eq:hessian) restricted to one simplex:
        C_{ff'} = sum_{h in simplex} dA_h/da_{f'} * d(delta_h)/da_f,
    with delta_h = 2 pi - sum(theta), so d(delta_h)/da_f = -d(theta_h)/da_f.
    Appendix A states C = 2R/sqrt(3) at kappa = 1, with R printed as 32R.

kappa = 1 gives the paper's Lorentzian background g_L = I - J (eq:direct-metric).
kappa = 0 gives the Euclidean Kuhn lattice. We use kappa = 0 only as a self-test
of the conventions against the published 302-entry Euclidean coupling table.

Exact arithmetic: SymPy rationals and square roots throughout.
"""
import itertools
from functools import lru_cache

import sympy as sp

# Appendix A: "In a chain simplex use the ordered edge slots (01,02,03,04,12,13,14,23,24,34)."
EDGE_SLOTS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
SLOT_NAMES = ["%d%d" % s for s in EDGE_SLOTS]
# The ten hinges (triangles) of a 4-simplex, as triples of local vertex labels 0..4.
TRIANGLES = list(itertools.combinations(range(5), 3))
# The 24 orderings of the axes; index pi(j) - 1 in the paper's 1-based notation.
PERMUTATIONS = list(itertools.permutations(range(4)))
CANONICAL = (0, 1, 2, 3)


def background_metric(kappa):
    """g = I_4 - kappa J_4 (eq:direct-metric at kappa = 1; Euclidean at kappa = 0)."""
    return sp.eye(4) - sp.Rational(kappa) * sp.ones(4, 4)


def chain_vertices(perm):
    """Kuhn simplex for the axis ordering perm (Section 2.1): v_0 = 0, v_j = v_{j-1} + e_{perm(j)}."""
    verts = [sp.zeros(4, 1)]
    for axis in perm:
        nxt = verts[-1].copy()
        nxt[axis] += 1
        verts.append(nxt)
    return verts


def squared_interval(D, g):
    """a = D^T g D. For D in {0,1}^4 with n ones and g = I - kappa J this is n(1 - kappa n)."""
    D = sp.Matrix(D)
    return (D.T * g * D)[0, 0]


class KuhnSimplex:
    """One flat 4-simplex with exact squared intervals between its five vertices."""

    def __init__(self, vertices, g):
        self.vertices = [sp.Matrix(v) for v in vertices]
        self.g = g
        self.sq = {}
        for i, j in itertools.combinations(range(5), 2):
            a = squared_interval(self.vertices[j] - self.vertices[i], g)
            self.sq[(i, j)] = self.sq[(j, i)] = a

    def a(self, i, j):
        return sp.Integer(0) if i == j else self.sq[(i, j)]

    # --- Gram matrices and their derivatives in squared-edge coordinates ----------
    def gram(self, base, others):
        """G_rc = (a_{b,o_r} + a_{b,o_c} - a_{o_r,o_c}) / 2 (eq:simplex-metric)."""
        n = len(others)
        return sp.Matrix(n, n, lambda r, c: (self.a(base, others[r]) + self.a(base, others[c])
                                             - self.a(others[r], others[c])) / 2)

    @staticmethod
    def gram_derivative(base, others, slot):
        """dG/da_slot: G is linear in the squared lengths, so this is a constant matrix."""
        def d(i, j):
            return sp.Integer(1) if i != j and {i, j} == set(slot) else sp.Integer(0)
        n = len(others)
        return sp.Matrix(n, n, lambda r, c: (d(base, others[r]) + d(base, others[c])
                                             - d(others[r], others[c])) / 2)

    # --- hinge areas ------------------------------------------------------------
    def triangle_gram_det(self, tri):
        p, q, r = tri
        return self.gram(p, [q, r]).det()

    def area(self, tri):
        """A_h = (1/2) sqrt(|det G_h|) (Heron's formula, eq:heron, in Gram form)."""
        return sp.sqrt(sp.Abs(self.triangle_gram_det(tri))) / 2

    def area_gradient(self, tri):
        """dA_h/da_f for the ten slots f. With s = sign(det G_h): dA = s d(det G_h) / (8 A)."""
        p, q, r = tri
        G = self.gram(p, [q, r])
        det = G.det()
        s = sp.sign(det)
        A = self.area(tri)
        adj = G.adjugate()
        grad = []
        for slot in EDGE_SLOTS:
            dG = self.gram_derivative(p, [q, r], slot)
            ddet = (adj * dG).trace()          # Jacobi's formula
            grad.append(sp.radsimp(s * ddet / (8 * A)))
        return grad

    # --- dihedral angles --------------------------------------------------------
    def _angle_data(self, tri):
        """Base b on the hinge, inverse Gram X, and the indices of the two off-hinge vertices."""
        b = tri[0]
        others = [v for v in range(5) if v != b]
        l, m = [v for v in range(5) if v not in tri]
        G = self.gram(b, others)
        X = G.inv()
        return b, others, X, others.index(l), others.index(m)

    def normal_plane_gram(self, tri):
        """Gram matrix of the two inward facet normals at a hinge: [[X_ll, X_lm], [X_lm, X_mm]]."""
        _, _, X, il, im = self._angle_data(tri)
        return sp.Matrix([[X[il, il], X[il, im]], [X[il, im], X[im, im]]])

    def cos_angle(self, tri):
        _, _, X, il, im = self._angle_data(tri)
        return sp.radsimp(-X[il, im] / sp.sqrt(X[il, il] * X[im, im]))

    def angle(self, tri):
        """Interior dihedral angle theta_h (an ordinary rotation angle in the spacelike normal plane)."""
        return sp.acos(self.cos_angle(tri))

    def angle_gradient(self, tri):
        """d(theta_h)/da_f for the ten slots, from dX = -X dG X and theta = arccos(c)."""
        b, others, X, il, im = self._angle_data(tri)
        c = self.cos_angle(tri)
        sin_theta = sp.sqrt(1 - c ** 2)
        root = sp.sqrt(X[il, il] * X[im, im])
        grad = []
        for slot in EDGE_SLOTS:
            dX = -X * self.gram_derivative(b, others, slot) * X
            dc = -(dX[il, im] / root
                   - X[il, im] * (dX[il, il] / X[il, il] + dX[im, im] / X[im, im]) / (2 * root))
            grad.append(sp.radsimp(-dc / sin_theta))
        return grad

    # --- the per-simplex Hessian (eq:hessian restricted to one simplex) ---------
    def hessian_cell(self):
        """C_{ff'} = sum_h dA_h/da_{f'} * d(delta_h)/da_f, with d(delta_h) = -d(theta_h)."""
        dA = {t: self.area_gradient(t) for t in TRIANGLES}
        dth = {t: self.angle_gradient(t) for t in TRIANGLES}
        C = sp.zeros(10, 10)
        for f in range(10):
            for fp in range(10):
                C[f, fp] = sp.radsimp(sp.expand(sum(dA[t][fp] * (-dth[t][f]) for t in TRIANGLES)))
        return C

    def schlaefli_residual(self):
        """sum_h A_h d(theta_h)/da_f for each slot f; the Schlaefli identity says this is zero."""
        out = []
        for f in range(10):
            out.append(sp.radsimp(sp.expand(sum(self.area(t) * self.angle_gradient(t)[f]
                                                for t in TRIANGLES))))
        return out


@lru_cache(maxsize=None)
def simplex(kappa, perm=CANONICAL):
    """The Kuhn simplex of ordering perm at base vertex 0 on the background g = I - kappa J."""
    return KuhnSimplex(chain_vertices(perm), background_metric(kappa))


@lru_cache(maxsize=None)
def hessian_cell(kappa, perm=CANONICAL):
    return simplex(kappa, perm).hessian_cell()


def slot_vector(perm, f):
    """Edge vector D_{pi f} = v_j - v_i and twice the midpoint 2 m_{pi f} = v_i + v_j (Appendix A)."""
    verts = chain_vertices(perm)
    i, j = EDGE_SLOTS[f]
    D = tuple(int(x) for x in (verts[j] - verts[i]))
    two_m = tuple(int(x) for x in (verts[i] + verts[j]))
    return D, two_m


def hinge_angle_sums(kappa):
    """Sum of interior angles around each of the 50 hinge classes (Section 6.1: 'still sum to 2 pi').

    A hinge class is represented with its lowest vertex at the origin: vertices 0, D1, D1 + D2 with
    D1, D2 disjoint nonempty 0-1 vectors (Lemma 2.1). Every lattice simplex containing the origin has
    its base vertex in {-1, 0}^4.
    """
    angles = {perm: {t: simplex(kappa, perm).angle(t) for t in TRIANGLES} for perm in PERMUTATIONS}
    classes = []
    for D1 in itertools.product((0, 1), repeat=4):
        for D2 in itertools.product((0, 1), repeat=4):
            if any(D1) and any(D2) and not any(a and b for a, b in zip(D1, D2)):
                classes.append((D1, D2))
    result = {}
    for D1, D2 in classes:
        tri = {(0, 0, 0, 0), D1, tuple(a + b for a, b in zip(D1, D2))}
        total, count = sp.Integer(0), 0
        for y in itertools.product((-1, 0), repeat=4):
            for perm in PERMUTATIONS:
                verts = [tuple(int(c) + yy for c, yy in zip(v, y)) for v in chain_vertices(perm)]
                if tri <= set(verts):
                    local = tuple(sorted(verts.index(v) for v in tri))
                    total += angles[perm][local]
                    count += 1
        result[(D1, D2)] = (sp.simplify(total), count)
    return result


if __name__ == "__main__":
    s = simplex(1)
    print("kappa = 1 chain simplex, squared intervals by slot:",
          {SLOT_NAMES[f]: s.a(*EDGE_SLOTS[f]) for f in range(10)})
    print("Gram matrix G_ij = min(i,j) - ij:", s.gram(0, [1, 2, 3, 4]).tolist())
    print("angles:", {t: s.angle(t) for t in TRIANGLES})
