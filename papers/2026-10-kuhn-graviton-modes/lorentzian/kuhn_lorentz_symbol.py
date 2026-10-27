#!/usr/bin/env python3
"""
kuhn_lorentz_symbol.py -- assembling the Hessian cell into the 15 x 15 midpoint symbol.

Companion to: J. Washburn and P. Beltracchi, "Graviton Modes on the Kuhn
Lattice: Euclidean Continuation and Lorentzian Propagation".
Supports equation (eq:symbol) of Section 2.3, equation (eq:lorentz-assembly)
of Appendix A, and the gauge (lattice Bianchi) identity used in Section 6.

The midpoint symbol. A plane-wave edge perturbation da_D(x) = c_D exp(i q.(x + D/2))
(eq. aperturbation) turns the translation-invariant Hessian into a 15 x 15 matrix
of trigonometric polynomials,

    M_{DD'}(q) = sum over simplices pi and slot pairs (f, f') with D_{pi f} = D, D_{pi f'} = D'
                 of C_{ff'} exp(i q.(m_{pi f'} - m_{pi f})),

where C is the per-simplex Hessian cell and m the edge midpoints. We store each entry
exactly as a dictionary {v: coefficient} meaning sum coefficient * exp(i q.v/2), with
v = 2 (m_{pi f'} - m_{pi f}) an integer vector. Because the lattice is symmetric under
x -> -x, the coefficients of v and -v agree, which gives the cosine form of Appendix A:

    (M_L(q))_{DD'} = sum 2 R_{f'f} cos(q.(m_{pi f} - m_{pi f'})).

M_L includes the overall factor sqrt(3) of Section 6.1: M_L = sqrt(3) M at kappa = 1.

The gauge vectors. Moving the vertices by a displacement field zeta exp(i q.x) changes the
squared interval of edge D by 4i (D.xi) sin(q.D/2) in midpoint coordinates, where xi = g zeta
is the covector of the displacement (Lemma 2.2(iii); Section 6.2 proof). Flat displaced vertices
stay flat, so M(q) annihilates every such vector: this is the lattice Bianchi identity.
"""
import itertools
from collections import defaultdict
from fractions import Fraction

import sympy as sp

import kuhn_lorentz_geometry as geo

# The 15 edge classes D in {0,1}^4 \ {0}: 4 axes, 6 face diagonals, 4 body diagonals, 1 hypercube diagonal.
EDGE_CLASSES = sorted((D for D in itertools.product((0, 1), repeat=4) if any(D)),
                      key=lambda D: (sum(D), [-x for x in D]))
INDEX = {D: i for i, D in enumerate(EDGE_CLASSES)}


def class_name(D):
    return "e" + "".join(str(i + 1) for i in range(4) if D[i])


class TrigPoly:
    """Exact trigonometric polynomial  sum_v c_v exp(i q.v/2)  with rational c_v and integer v."""

    __slots__ = ("terms",)

    def __init__(self, terms=None):
        self.terms = {}
        if terms:
            for v, c in terms.items():
                if c != 0:
                    self.terms[tuple(v)] = Fraction(c)

    def __add__(self, other):
        out = dict(self.terms)
        for v, c in other.terms.items():
            out[v] = out.get(v, 0) + c
        return TrigPoly(out)

    def __sub__(self, other):
        return self + other.scale(-1)

    def scale(self, s):
        return TrigPoly({v: c * Fraction(s) for v, c in self.terms.items()})

    def __mul__(self, other):
        out = defaultdict(Fraction)
        for v, c in self.terms.items():
            for w, d in other.terms.items():
                out[tuple(a + b for a, b in zip(v, w))] += c * d
        return TrigPoly(out)

    def is_zero(self):
        return not self.terms

    def map_keys(self, f):
        out = defaultdict(Fraction)
        for v, c in self.terms.items():
            out[f(v)] += c
        return TrigPoly(out)

    def is_even(self):
        """Coefficient of v equals coefficient of -v: the polynomial is a cosine sum (real for real q)."""
        return all(self.terms.get(tuple(-x for x in v), 0) == c for v, c in self.terms.items())

    def cosine_form(self):
        """{v: a_v} with sum a_v cos(q.v/2) equal to this even polynomial; one representative per +-v."""
        assert self.is_even()
        out = {}
        for v, c in self.terms.items():
            neg = tuple(-x for x in v)
            if all(x == 0 for x in v):
                out[v] = c
            elif v > neg:
                out[v] = 2 * c
        return dict(sorted(out.items()))

    def evaluate(self, q, mp):
        """Numerical value at real or complex q (mpmath)."""
        return mp.fsum(mp.mpf(c.numerator) / c.denominator * mp.expj(mp.fsum(a * b for a, b in zip(q, v)) / 2)
                       for v, c in self.terms.items())

    @staticmethod
    def cos(v, coeff=1):
        """coeff * cos(q.v/2)."""
        v = tuple(v)
        if all(x == 0 for x in v):
            return TrigPoly({v: coeff})
        neg = tuple(-x for x in v)
        return TrigPoly({v: Fraction(coeff) / 2, neg: Fraction(coeff) / 2})

    @staticmethod
    def two_i_sin(v, coeff=1):
        """coeff * 2i sin(q.v/2) = coeff * (exp(i q.v/2) - exp(-i q.v/2))."""
        v = tuple(v)
        neg = tuple(-x for x in v)
        return TrigPoly({v: coeff, neg: -Fraction(coeff)})


def rational_cell(kappa):
    """The scaled cell. kappa = 1: R = (sqrt 3 / 2) C, so that C = 2R/sqrt(3) (Appendix A).
    kappa = 0 (Euclidean): the cell C itself is rational and we return C/2 so that the same
    assembly rule 2R applies."""
    C = geo.hessian_cell(kappa)
    scale = sp.sqrt(3) / 2 if kappa == 1 else sp.Rational(1, 2)
    R = (C * scale).applyfunc(lambda x: sp.nsimplify(sp.radsimp(x)))
    for x in R:
        assert x.is_Rational, "cell entry is not rational after scaling: %s" % x
    return R


def couplings(cell_of_perm):
    """The coupling table {(D, D', v): w}, v = 2(m_{pi f'} - m_{pi f}), w = sum of cell entries.

    cell_of_perm(perm) gives the 10 x 10 cell of the simplex with that axis ordering, already in the
    normalization used for the symbol (2R for M_L).
    """
    table = defaultdict(Fraction)
    for perm in geo.PERMUTATIONS:
        cell = cell_of_perm(perm)
        slots = [geo.slot_vector(perm, f) for f in range(10)]
        for f in range(10):
            D, m2 = slots[f]
            for fp in range(10):
                Dp, mp2 = slots[fp]
                w = cell[f, fp]
                if w != 0:
                    table[(D, Dp, tuple(b - a for a, b in zip(m2, mp2)))] += Fraction(int(w.p), int(w.q))
    return {k: w for k, w in table.items() if w != 0}


def symbol_from_couplings(table):
    """M_{DD'}(q) = sum_v w(D, D', v) exp(i q.v/2), a 15 x 15 array of TrigPoly."""
    M = [[TrigPoly() for _ in EDGE_CLASSES] for _ in EDGE_CLASSES]
    for (D, Dp, v), w in table.items():
        M[INDEX[D]][INDEX[Dp]] = M[INDEX[D]][INDEX[Dp]] + TrigPoly({v: w})
    return M


def lorentz_symbol(kappa=1, per_permutation=True):
    """The scaled Lorentzian midpoint symbol M_L (Section 6.1, eq:lorentz-assembly).

    With per_permutation=True every one of the 24 simplices has its own cell computed from its own
    vertex coordinates, rather than assuming it is the coordinate permutation of the canonical one.
    """
    if per_permutation:
        cells = {}
        for perm in geo.PERMUTATIONS:
            C = geo.hessian_cell(kappa, perm)
            scale = sp.sqrt(3) if kappa == 1 else sp.Integer(1)
            cells[perm] = (C * scale).applyfunc(lambda x: sp.nsimplify(sp.radsimp(x)))
        table = couplings(lambda perm: cells[perm])
    else:
        R = rational_cell(kappa)
        table = couplings(lambda perm: 2 * R)
    return table, symbol_from_couplings(table)


def matvec(M, c):
    """(M c)_D for a vector c of TrigPoly or rational entries."""
    out = []
    for i in range(len(M)):
        acc = TrigPoly()
        for j in range(len(M)):
            cj = c[j] if isinstance(c[j], TrigPoly) else TrigPoly({(0,) * _dim(M): c[j]})
            if not cj.is_zero() and not M[i][j].is_zero():
                acc = acc + M[i][j] * cj
        out.append(acc)
    return out


def _dim(M):
    for row in M:
        for e in row:
            if e.terms:
                return len(next(iter(e.terms)))
    return 4


def gauge_vector(xi):
    """g_xi(q)_D = 4i (D.xi) sin(q.D/2) = 2 (D.xi) [2i sin(q.D/2)] (Lemma 2.2(iii), Section 6.2)."""
    return [TrigPoly.two_i_sin(D, 2 * sum(d * x for d, x in zip(D, xi))) for D in EDGE_CLASSES]


def check_bianchi(M):
    """M(q) g_xi(q) = 0 identically in q, for xi = e_1, ..., e_4."""
    results = {}
    for mu in range(4):
        xi = [1 if k == mu else 0 for k in range(4)]
        res = matvec(M, gauge_vector(xi))
        results["xi=e%d" % (mu + 1)] = all(r.is_zero() for r in res)
    return results


def check_real_symmetric(M):
    even = all(M[i][j].is_even() for i in range(15) for j in range(15))
    sym = all((M[i][j] - M[j][i]).is_zero() for i in range(15) for j in range(15))
    return even, sym


def paired(M):
    """Restrict to the paired covector q = (x, x, y, y): key v -> (v1 + v2, v3 + v4).
    The term exp(i (p x + r y)/2) is then labelled (p, r); cos((p x + r y)/2) is the paper's C_{pr}."""
    return [[e.map_keys(lambda v: (v[0] + v[1], v[2] + v[3])) for e in row] for row in M]


def numeric_matrix(M, q, mp):
    return mp.matrix([[M[i][j].evaluate(q, mp) for j in range(15)] for i in range(15)])


if __name__ == "__main__":
    table, M = lorentz_symbol(1)
    print("nonzero Lorentzian couplings (D, D', v):", len(table))
    print("cosine form and symmetry:", check_real_symmetric(M))
    print("Bianchi identity:", check_bianchi(M))
