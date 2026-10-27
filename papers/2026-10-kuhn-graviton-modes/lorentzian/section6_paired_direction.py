#!/usr/bin/env python3
"""
section6_paired_direction.py -- the two graviton branches along the paired direction.

Companion to: J. Washburn and P. Beltracchi, "Graviton Modes on the Kuhn
Lattice: Euclidean Continuation and Lorentzian Propagation", Section 6
("Direct Lorentzian propagation on the same triangulation") and the second
half of Appendix A (the paired orbit block and its determinant).

The symbol M_L is assembled from the geometry by kuhn_lorentz_symbol.py; nothing
below reads a stored table. The program reproduces, in the paper's order:

  6.1  the clock tau and spatial direction sigma, their normalization, the paired
       covector q = (z+t, z+t, z-t, z-t) with K = q(sigma) = 2t and a*omega = q(tau) = 2z/sqrt3,
       the null cone z = sqrt3 t and the phase speed z/(sqrt3 t)          (eq:paired-clock)
  6.2  the cross column c_D = 2(D1-D2)(D3-D4) (eq:cross-vector), the exact eigenvalue
       s_x (eq:cross-scalar) with coefficients -2, 3/2, 3/2, -1, the characteristic
       equation (eq:cross-direct), G_x (eq:cross-poly), dG_x/dV(0,0) = -1, the
       coefficient +5/24 (eq:cross-speed), and the nongauge argument
  6.3  the six even orbits, the block B = O^T M_L O (eq:plus-block), its determinant
       (eq:plus-det) with the polynomial P, the identity P(1-2V, 1-2U) = G_+ (eq:plus-poly),
       dG_+/dV(0,0) = 2, the coefficient -59/24 and the splitting 8/3 (eq:paired-split),
       and the reconstruction of a full zero mode of M_L from a zero mode of B
  6.3  the Euclidean comparison v_E = 1 - (1+A) K^2/24 (axis time) and 1 + K^2/24
       (hypercube-diagonal time with spatial direction sigma)

Exact work uses SymPy; the numerical branch and kernel checks use mpmath at 50 digits.
Run:  python section6_paired_direction.py [outputs/section6_paired_direction.json]
Runtime: about a minute.
"""
import sys
from fractions import Fraction as Fr

import mpmath as mp
import sympy as sp

import kuhn_lorentz_symbol as sym
from kuhn_lorentz_symbol import EDGE_CLASSES, INDEX, TrigPoly
from report import Report, out_path

z, t, x, y, U, V, alpha, K = sp.symbols("z t x y U V alpha K", real=True)
a, b = sp.symbols("a b", nonzero=True)       # a = exp(i x/2), b = exp(i y/2)

# ---------------------------------------------------------------------------------------------
# What the paper prints (typed from the manuscript; the program compares the geometry with these)
# ---------------------------------------------------------------------------------------------
ORBITS = [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]           # Appendix A, in this order
AXIS_ORBITS = [(1, 0), (0, 1)]


def C(p, q, coeff=1):
    """coeff * C_pq = coeff * cos((p x + q y)/2), the paper's notation in eq:plus-block."""
    return TrigPoly.cos((p, q), coeff)


ONE = TrigPoly({(0, 0): 1})
PAPER_B = {
    (0, 0): ONE.scale(-2), (0, 1): C(1, 1, -6), (0, 2): C(1, 0, 2), (0, 3): C(2, 2, -1),
    (0, 4): C(2, 1, 1), (0, 5): ONE.scale(Fr(1, 2)) + C(2, 0, Fr(-1, 2)),
    (1, 1): ONE.scale(-8) + C(0, 2, -6) + C(2, 0, -6) + C(2, 2, -4),
    (1, 2): C(0, 1, 4) + C(2, 1, 2), (1, 3): C(1, 1, -6), (1, 4): C(1, 0, 4) + C(1, 2, 2),
    (1, 5): C(1, -1, 2) + C(1, 1, -2),
    (2, 2): C(2, 0, 1), (2, 3): C(1, 2, 1), (2, 4): C(1, 1, 2), (2, 5): C(1, 0, -3),
    (3, 3): ONE.scale(-2), (3, 4): C(0, 1, 2), (3, 5): ONE.scale(Fr(1, 2)) + C(0, 2, Fr(-1, 2)),
    (4, 4): C(0, 2, 1), (4, 5): C(0, 1, -3),
    (5, 5): ONE.scale(3),
}

r_, s_ = sp.symbols("r s")
PAPER_P = (16 * r_**6 - 8 * s_ * r_**5 - 8 * s_**2 * r_**4 - 42 * r_**4 + 2 * s_**3 * r_**3 + 39 * s_ * r_**3
           + 3 * s_**2 * r_**2 + 25 * r_**2 - s_**3 * r_ - 35 * s_ * r_ + 7 * s_**2 + 2)
PAPER_G_PLUS = (U**3 * (128 * V**3 - 192 * V**2 + 80 * V - 8)
                + U**2 * (-512 * V**4 + 832 * V**3 - 432 * V**2 + 88 * V + 20)
                + U * (-512 * V**5 + 1792 * V**4 - 1584 * V**3 + 280 * V**2 + 20 * V - 6)
                + 1024 * V**6 - 2816 * V**5 + 2400 * V**4 - 648 * V**3 + 44 * V**2 + 2 * V)
PAPER_G_CROSS = 4 * V**2 - V + 3 * U - 6 * U * V
PAPER_CROSS_COEFF = sp.Rational(5, 24)
PAPER_PLUS_COEFF = sp.Rational(-59, 24)


# ---------------------------------------------------------------------------------------------
# Section 6.1: clock, spatial direction and the paired covector
# ---------------------------------------------------------------------------------------------
def section_6_1_clock(rep):
    rep.section("6.1  Clock, spatial direction and the paired covector (eq:paired-clock)")
    g = sp.eye(4) - sp.ones(4, 4)
    ginv = g.inv()
    tau = sp.Matrix([1, 1, 1, 1]) / (2 * sp.sqrt(3))
    sigma = sp.Matrix([1, 1, -1, -1]) / 2
    G = lambda u, w: sp.simplify((u.T * g * w)[0, 0])
    rep.check("g_L(tau, tau) = -1", G(tau, tau) == -1)
    rep.check("g_L(sigma, sigma) = 1", G(sigma, sigma) == 1)
    rep.check("g_L(tau, sigma) = 0", G(tau, sigma) == 0)
    ones = sp.Matrix([1, 1, 1, 1])
    rep.value("g_L((1,1,1,1), (1,1,1,1)) (timelike; hence the 2 sqrt3 normalization)", G(ones, ones))
    q = sp.Matrix([z + t, z + t, z - t, z - t])
    rep.check("K = q(sigma) = 2t", sp.simplify((q.T * sigma)[0, 0] - 2 * t) == 0)
    rep.check("a omega = q(tau) = 2z/sqrt3", sp.simplify((q.T * tau)[0, 0] - 2 * z / sp.sqrt(3)) == 0)
    cone = sp.expand((q.T * ginv * q)[0, 0])
    rep.check("g_L^{-1}(q, q) = 4t^2 - (4/3) z^2, so the continuum null cone is z = sqrt3 t",
              sp.simplify(cone - (4 * t**2 - sp.Rational(4, 3) * z**2)) == 0, cone)
    rep.check("phase speed (a omega)/K = z/(sqrt3 t)",
              sp.simplify((q.T * tau)[0, 0] / (q.T * sigma)[0, 0] - z / (sp.sqrt(3) * t)) == 0)
    # The perpendicular plane: covectors annihilating tau and sigma. q has no component there.
    rep.check("q vanishes on the g_L-orthogonal complement of tau and sigma",
              all(sp.simplify((q.T * w)[0, 0]) == 0 for w in
                  (sp.Matrix([1, -1, 0, 0]), sp.Matrix([0, 0, 1, -1]))))


# ---------------------------------------------------------------------------------------------
# Small-momentum coefficient of a branch G(U, V) = 0 through the origin
# ---------------------------------------------------------------------------------------------
def phase_coefficient(G):
    """Insert z = sqrt3 t (1 + alpha t^2), U = sin^2(t/2), V = sin^2(z/2) and expand in t.

    The t^2 term vanishes because the branch starts on the null cone; the t^4 term is linear in alpha.
    The phase speed is v = z/(sqrt3 t) = 1 + alpha t^2 = 1 + (alpha/4) K^2 with K = 2t.
    """
    zz = sp.sqrt(3) * t * (1 + alpha * t**2)
    expr = G.subs({U: sp.sin(t / 2)**2, V: sp.sin(zz / 2)**2})
    ser = sp.expand(sp.series(expr, t, 0, 6).removeO())
    c2 = sp.simplify(ser.coeff(t, 2))
    c4 = sp.simplify(ser.coeff(t, 4))
    sol = sp.solve(sp.Eq(c4, 0), alpha)
    return c2, c4, sp.nsimplify(sol[0] / 4)


def cos_to_sines(expr):
    """Write cos z and cos t through V = sin^2(z/2), U = sin^2(t/2): cos z = 1 - 2V, cos t = 1 - 2U."""
    return sp.expand(expr.subs({sp.cos(z): 1 - 2 * V, sp.cos(t): 1 - 2 * U}))


# ---------------------------------------------------------------------------------------------
# Section 6.2: the cross polarization
# ---------------------------------------------------------------------------------------------
def swap(D, i, j):
    D = list(D)
    D[i], D[j] = D[j], D[i]
    return tuple(D)


def section_6_2_cross(rep, MP):
    rep.section("6.2  The exact cross-polarization branch (Proposition 6.1)")
    c = [2 * (D[0] - D[1]) * (D[2] - D[3]) for D in EDGE_CLASSES]
    support = {sym.class_name(D): c[INDEX[D]] for D in EDGE_CLASSES if c[INDEX[D]] != 0}
    rep.check("c is supported on the four mixed face diagonals with values +2, -2, -2, +2",
              support == {"e13": 2, "e14": -2, "e23": -2, "e24": 2}, support)
    rep.check("c is odd under the swap 1<->2 and under the swap 3<->4",
              all(c[INDEX[swap(D, 0, 1)]] == -c[INDEX[D]] and c[INDEX[swap(D, 2, 3)]] == -c[INDEX[D]]
                  for D in EDGE_CLASSES))
    # The swaps commute with M_L on a paired covector.
    for (i, j) in ((0, 1), (2, 3)):
        rep.check("the swap %d<->%d commutes with M_L(x,x,y,y)" % (i + 1, j + 1),
                  all((MP[INDEX[swap(D, i, j)]][INDEX[swap(E, i, j)]] - MP[INDEX[D]][INDEX[E]]).is_zero()
                      for D in EDGE_CLASSES for E in EDGE_CLASSES))
    # Dimensions of the four joint sign sectors of the two swaps.
    dims = {}
    for s1 in (1, -1):
        for s2 in (1, -1):
            rows = []
            for D in EDGE_CLASSES:
                for (i, j, s) in ((0, 1, s1), (2, 3, s2)):
                    r = [0] * 15
                    r[INDEX[swap(D, i, j)]] += 1
                    r[INDEX[D]] -= s
                    rows.append(r)
            dims[(s1, s2)] = 15 - sp.Matrix(rows).rank()
    rep.check("the common odd sector of the two swaps is one-dimensional", dims[(-1, -1)] == 1,
              "sector dimensions (even/odd under 1<->2, 3<->4): %s" % {str(k): v for k, v in dims.items()})
    res = sym.matvec(MP, c)
    s_cross = ONE.scale(-2) + C(2, 0, Fr(3, 2)) + C(0, 2, Fr(3, 2)) + C(2, 2, -1)
    rep.check("M_L(x,x,y,y) c = s_x(x,y) c with s_x = -2 + (3/2)cos x + (3/2)cos y - cos(x+y) (eq:cross-scalar)",
              all((res[i] - s_cross.scale(c[i])).is_zero() for i in range(15)))
    coeffs = res[INDEX[(1, 0, 1, 0)]].scale(Fr(1, 2)).cosine_form()
    rep.check("contraction coefficients -2, 3/2, 3/2, -1 at frequencies 0, y, x, x+y and nothing else",
              coeffs == {(0, 0): -2, (0, 2): Fr(3, 2), (2, 0): Fr(3, 2), (2, 2): -1},
              {"cos((%d x + %d y)/2)" % k: str(v) for k, v in coeffs.items()})

    s_sym = -2 + sp.Rational(3, 2) * sp.cos(x) + sp.Rational(3, 2) * sp.cos(y) - sp.cos(x + y)
    s_zt = sp.expand(sp.expand_trig(s_sym.subs({x: z + t, y: z - t})))
    rep.check("s_x(z+t, z-t) = -(2cos^2 z - 3 cos z cos t + 1), so eq:cross-direct",
              sp.simplify(s_zt + (2 * sp.cos(z)**2 - 3 * sp.cos(z) * sp.cos(t) + 1)) == 0)
    lhs = cos_to_sines(2 * sp.cos(z)**2 - 3 * sp.cos(z) * sp.cos(t) + 1)
    rep.check("2cos^2 z - 3 cos z cos t + 1 = 2 G_x(U, V), G_x = 4V^2 - V + 3U - 6UV (eq:cross-poly)",
              sp.expand(lhs - 2 * PAPER_G_CROSS) == 0)
    rep.check("dG_x/dV at the origin is -1 (implicit-function theorem applies)",
              sp.diff(PAPER_G_CROSS, V).subs({U: 0, V: 0}) == -1)
    c2, c4, coeff = phase_coefficient(PAPER_G_CROSS)
    rep.check("t^2 term vanishes: the branch leaves the origin along the null cone", c2 == 0)
    rep.value("t^4 coefficient of G_x on z = sqrt3 t (1 + alpha t^2)", c4)
    rep.check("v_x(K) = 1 + (5/24) K^2 + o(K^2) (eq:cross-speed)", coeff == PAPER_CROSS_COEFF, coeff)
    return c


# ---------------------------------------------------------------------------------------------
# Section 6.3 and Appendix A: the plus polarization
# ---------------------------------------------------------------------------------------------
def orbit_of(D):
    return (D[0] + D[1], D[2] + D[3])


def orbit_matrix():
    """O: indicator columns of the six non-axis orbits, without normalization (Appendix A)."""
    return [[1 if orbit_of(D) == o else 0 for o in ORBITS] for D in EDGE_CLASSES]


def block_B(MP):
    O = orbit_matrix()
    MO = [[TrigPoly() for _ in ORBITS] for _ in EDGE_CLASSES]
    for i in range(15):
        for j in range(6):
            for k in range(15):
                if O[k][j]:
                    MO[i][j] = MO[i][j] + MP[i][k]
    B = [[TrigPoly() for _ in ORBITS] for _ in ORBITS]
    for i in range(6):
        for j in range(6):
            for k in range(15):
                if O[k][i]:
                    B[i][j] = B[i][j] + MO[k][j]
    return O, MO, B


def laurent(tp):
    """exp(i (p x + q y)/2) -> a^p b^q."""
    return sum(sp.Rational(c.numerator, c.denominator) * a**p * b**q for (p, q), c in tp.terms.items())


def section_6_3_plus(rep, MP):
    rep.section("6.3  The plus polarization: orbit block, determinant, branch (Appendix A)")
    sizes = {o: sum(1 for D in EDGE_CLASSES if orbit_of(D) == o) for o in AXIS_ORBITS + ORBITS}
    rep.value("orbit sizes (axis orbits (1,0), (0,1) first)", sizes)
    rep.check("the even sector is spanned by 8 orbit indicators: 2 axis orbits and the 6 printed orbits",
              sum(sizes.values()) == 15 and len(sizes) == 8)
    O, MO, B = block_B(MP)
    mism = [(i + 1, j + 1) for i in range(6) for j in range(6)
            if not (B[i][j] - PAPER_B[(min(i, j), max(i, j))]).is_zero()]
    rep.check("B = O^T M_L O from the geometry equals the printed block (eq:plus-block), all 36 entries",
              not mism, mism if mism else "")
    for i in range(6):
        rep.note("   row %d: %s" % (i + 1, "; ".join(
            " + ".join("%s C%d%d" % (v, k[0], k[1]) if k != (0, 0) else str(v)
                       for k, v in B[i][j].cosine_form().items()) or "0" for j in range(6))))
    # Swap invariance: within an orbit all rows of M_L O agree.
    same = True
    for o in AXIS_ORBITS + ORBITS:
        rows = [MO[INDEX[D]] for D in EDGE_CLASSES if orbit_of(D) == o]
        same &= all((r[j] - rows[0][j]).is_zero() for r in rows for j in range(6))
    rep.check("within each orbit, every row of M_L O is the same (so B w = 0 kills each non-axis residual)", same)
    # The Bianchi identity on the slice, contracted with the lifted columns.
    bianchi = True
    for mu in range(4):
        xi = [1 if k == mu else 0 for k in range(4)]
        gv = [e.map_keys(lambda v: (v[0] + v[1], v[2] + v[3])) for e in sym.gauge_vector(xi)]
        for j in range(6):
            acc = TrigPoly()
            for k in range(15):
                acc = acc + gv[k] * MO[k][j]
            bianchi &= acc.is_zero()
    rep.check("g_xi^T M_L O = 0 on the slice for xi = e_1..e_4 (the identity that recovers the axis rows)",
              bianchi)

    # The determinant.
    N = sp.Matrix(6, 6, lambda i, j: sp.expand(a**2 * b**2 * laurent(B[i][j])))
    detN = sp.expand(N.det(method="berkowitz"))
    detB = detN / (a**12 * b**12)
    cz = (a * b + 1 / (a * b)) / 2          # cos z, z = (x + y)/2
    ct = (a / b + b / a) / 2                # cos t, t = (x - y)/2
    sin2x = -(a - 1 / a)**2 / 4             # sin^2(x/2)
    sin2y = -(b - 1 / b)**2 / 4
    rhs = -36 * sin2x * sin2y * PAPER_P.subs({r_: cz, s_: ct})
    rep.check("det B(z+t, z-t) = -36 sin^2((z+t)/2) sin^2((z-t)/2) P(cos z, cos t) (eq:plus-det)",
              sp.expand(sp.numer(sp.together(detB - rhs))) == 0)
    rep.value("det(a^2 b^2 B) factored", sp.factor(detN))
    GP = sp.expand(PAPER_P.subs({r_: 1 - 2 * V, s_: 1 - 2 * U}))
    rep.check("P(1 - 2V, 1 - 2U) = G_+(U, V) (eq:plus-poly)", sp.expand(GP - PAPER_G_PLUS) == 0)
    rep.check("dG_+/dV at the origin is 2", sp.diff(PAPER_G_PLUS, V).subs({U: 0, V: 0}) == 2)
    rep.check("dG_+/dU at the origin is -6, so V = 3U to first order (the null cone)",
              sp.diff(PAPER_G_PLUS, U).subs({U: 0, V: 0}) == -6)
    rep.value("G_+(0, V) factored (t = 0: only the root V = 0 is near the origin)", sp.factor(PAPER_G_PLUS.subs(U, 0)))
    c2, c4, coeff = phase_coefficient(PAPER_G_PLUS)
    rep.check("t^2 term vanishes on the null cone", c2 == 0)
    rep.value("t^4 coefficient of G_+ on z = sqrt3 t (1 + alpha t^2)", c4)
    rep.check("v_+(K) = 1 - (59/24) K^2 + o(K^2) (eq:paired-split)", coeff == PAPER_PLUS_COEFF, coeff)
    rep.check("v_x - v_+ = (8/3) K^2 + o(K^2)", PAPER_CROSS_COEFF - coeff == sp.Rational(8, 3))

    # Where the sine prefactor of det B vanishes, a vertex displacement survives the axis gauge.
    rep.note("Prefactor check: at x = 0, xi = (1,1,0,0) gives a pure-gauge vector with zero axis")
    rep.note("components, constant on orbits; at y = 0 the same holds for xi = (0,0,1,1).")
    ok = True
    for fixed in (0, 1):          # fixed = 0: x = 0 with xi = (1,1,0,0); fixed = 1: y = 0 with xi = (0,0,1,1)
        # Restrict B to x = 0 (fixed = 0) or y = 0 (fixed = 1): drop that frequency component.
        Bres = [[e.map_keys(lambda v, f=fixed: tuple(0 if k == f else v[k] for k in range(2))) for e in row]
                for row in B]
        # Orbit values of g_xi: (D.xi) * 2i sin(q.D/2), with q.D/2 = (r y)/2 at x = 0 or (p x)/2 at y = 0.
        w = []
        for o in ORBITS:
            Dxi = o[0] if fixed == 0 else o[1]
            other = o[1] if fixed == 0 else o[0]
            vec = (0, other) if fixed == 0 else (other, 0)
            w.append(TrigPoly.two_i_sin(vec, Dxi) if other else TrigPoly())
        for i in range(6):
            acc = TrigPoly()
            for j in range(6):
                acc = acc + Bres[i][j] * w[j]
            ok &= acc.is_zero()
    rep.check("B(0, y) and B(x, 0) annihilate these gauge vectors: the sin^2 prefactors are gauge zeros", ok)
    return O, B


# ---------------------------------------------------------------------------------------------
# Numerical branches, full-kernel reconstruction and gauge independence (50 digits)
# ---------------------------------------------------------------------------------------------
def numeric_branches(rep, M, MP, B, c_cross):
    rep.section("Branches at finite K (mpmath, 50 digits): coefficients, full kernels, nongauge test")
    mp.mp.dps = 50
    sqrt3 = mp.sqrt(3)

    def z_cross(tt):
        ct = mp.cos(tt)
        return mp.acos((3 * ct + mp.sqrt(9 * ct**2 - 8)) / 4)    # root of 2c^2 - 3c cos t + 1 through c = 1

    Gp = sp.lambdify((U, V), PAPER_G_PLUS, modules="mpmath")

    def z_plus(tt):
        # The near-cone root has V close to 3U. G_+ also has lattice-scale roots near V = 0.19 and 0.25,
        # so bracket the root between V = 2U and V = 4U instead of starting an open iteration.
        uu = mp.sin(tt / 2)**2
        lo, hi = 2 * uu, 4 * uu
        assert Gp(uu, lo) * Gp(uu, hi) < 0
        vv = mp.findroot(lambda vv: Gp(uu, vv), (lo, hi), solver="anderson")
        return 2 * mp.asin(mp.sqrt(vv))

    for name, zf, exact in (("cross", z_cross, PAPER_CROSS_COEFF), ("plus", z_plus, PAPER_PLUS_COEFF)):
        est = {}
        # (v - 1)/K^2 = c + d K^2 + O(K^4). The plus branch has a large d (about 20), so K must be
        # well below 0.1 before the K^2 law is accurate; Richardson extrapolation removes the d K^2 term.
        for tt in (mp.mpf(1) / 100, mp.mpf(1) / 200, mp.mpf(1) / 20):
            KK = 2 * tt
            v = zf(tt) / (sqrt3 * tt)
            est[float(KK)] = (v - 1) / KK**2
            rep.value("%s phase speed at K = %s" % (name, mp.nstr(KK, 3)), mp.nstr(v, 25), show=False)
            rep.note("%s: K = %s   v = %s   (v - 1)/K^2 = %s" %
                     (name, mp.nstr(KK, 3), mp.nstr(v, 20), mp.nstr(est[float(KK)], 12)))
        rich = (4 * est[0.01] - est[0.02]) / 3
        rep.check("%s: Richardson value from K = 0.02, 0.01 agrees with %s to 1e-4" % (name, exact),
                  abs(rich - mp.mpf(exact.p) / exact.q) < mp.mpf("1e-4"), mp.nstr(rich, 12))

    # Full-kernel reconstruction at K = 0.1 on the plus branch.
    tt = mp.mpf(1) / 20
    for name, zf in (("plus", z_plus), ("cross", z_cross)):
        zz = zf(tt)
        xx, yy = zz + tt, zz - tt
        q = (xx, xx, yy, yy)
        Mnum = mp.matrix(15, 15)
        for i in range(15):
            for j in range(15):
                Mnum[i, j] = mp.re(M[i][j].evaluate(q, mp))
        scale = max(abs(Mnum[i, j]) for i in range(15) for j in range(15))
        if name == "plus":
            Bnum = mp.matrix(6, 6)
            for i in range(6):
                for j in range(6):
                    Bnum[i, j] = mp.re(B[i][j].evaluate((xx, yy), mp))
            Ub, Sb, Vb = mp.svd_r(Bnum)
            k = min(range(6), key=lambda i: Sb[i])
            w = [Vb[k, j] for j in range(6)]
            rep.note("plus: singular values of B at the branch point: %s" % [mp.nstr(Sb[i], 5) for i in range(6)])
            cvec = [0 if orbit_of(D) in AXIS_ORBITS else w[ORBITS.index(orbit_of(D))] for D in EDGE_CLASSES]
        else:
            cvec = [mp.mpf(v) for v in c_cross]
        cnorm = max(abs(v) for v in cvec)
        r = [mp.fsum(Mnum[i, j] * cvec[j] for j in range(15)) for i in range(15)]
        axis_idx = [i for i, D in enumerate(EDGE_CLASSES) if sum(D) == 1]
        res_axis = max(abs(r[i]) for i in axis_idx) / (scale * cnorm)
        res_rest = max(abs(r[i]) for i in range(15) if i not in axis_idx) / (scale * cnorm)
        rep.check("%s: lifted vector solves all 15 rows of M_L c = 0 (non-axis rows %s, axis rows %s)" %
                  (name, mp.nstr(res_rest, 3), mp.nstr(res_axis, 3)),
                  res_rest < mp.mpf("1e-40") and res_axis < mp.mpf("1e-40"))
        Um, Sm, Vm = mp.svd_r(Mnum)
        small = sorted(Sm[i] / scale for i in range(15))
        n_null = sum(1 for s in small if s < mp.mpf("1e-40"))
        rep.check("%s: M_L has exactly 5 null directions there (4 gauge + this mode)" % name, n_null == 5,
                  "relative singular values: smallest six %s" % [mp.nstr(s, 3) for s in small[:6]])
        # Gauge vectors (real form): 4 (D.xi) sin(q.D/2).
        G = mp.matrix(15, 5)
        for mu in range(4):
            for i, D in enumerate(EDGE_CLASSES):
                G[i, mu] = 4 * D[mu] * mp.sin(mp.fsum(q[k] * D[k] for k in range(4)) / 2)
        for i in range(15):
            G[i, 4] = cvec[i] / cnorm
        Ug, Sg, Vg = mp.svd_r(G)
        rep.check("%s: the mode is not a gauge vector (smallest singular value of [g_1..g_4, c] is %s)" %
                  (name, mp.nstr(min(Sg[i] for i in range(5)), 5)), min(Sg[i] for i in range(5)) > mp.mpf("1e-3"))
        rep.note("%s: sin(x/2) = %s, sin(y/2) = %s (both nonzero, as the reconstruction requires)" %
                 (name, mp.nstr(mp.sin(xx / 2), 8), mp.nstr(mp.sin(yy / 2), 8)))


# ---------------------------------------------------------------------------------------------
# The Euclidean comparison in Section 6.3
# ---------------------------------------------------------------------------------------------
def euclidean_comparison(rep):
    rep.section("6.3  Euclidean continuation for comparison (the exact Laplacian branch of Section 4)")
    cc, w = sp.symbols("c w")
    n1, n2 = sp.symbols("n1 n2", real=True)
    n3 = sp.sqrt(1 - n1**2 - n2**2)

    def series_c(kvec):
        """Lap(k) = sum 4 sin^2(k_mu/2) = 0 with frequency w = K (1 + c K^2): solve for c at order K^4."""
        L = sum(4 * sp.sin(k / 2)**2 for k in kvec).subs(w, K * (1 + cc * K**2))
        ser = sp.expand(sp.series(L, K, 0, 6).removeO())
        c2, c4 = sp.simplify(ser.coeff(K, 2)), sp.expand(ser.coeff(K, 4))
        return c2, sp.simplify(sp.solve(sp.Eq(c4, 0), cc)[0])

    # Axis time T = e_4, spatial direction (n1, n2, n3): k = (K n1, K n2, K n3, i w).
    c2, cax = series_c([K * n1, K * n2, K * n3, sp.I * w])
    A = n1**4 + n2**4 + n3**4
    rep.check("axis time: Lap = 0 at order K^2 on the cone", sp.simplify(c2) == 0)
    rep.check("axis time: v_E = 1 - (1 + A) K^2/24, A = sum n_j^4, same for both polarizations",
              sp.simplify(cax + (1 + A) / 24) == 0, sp.simplify(cax))
    # Hypercube-diagonal time T = (1,1,1,1)/2 with spatial direction sigma = (1,1,-1,-1)/2.
    kv = [(sp.I * w + K) / 2, (sp.I * w + K) / 2, (sp.I * w - K) / 2, (sp.I * w - K) / 2]
    c2h, chy = series_c(kv)
    rep.check("hypercube time with sigma: v_E = 1 + K^2/24 for both modes", chy == sp.Rational(1, 24), chy)
    rep.note("Neither Euclidean coefficient splits the polarizations; the Lorentzian ones are +5/24 and -59/24.")


def main(argv):
    rep = Report("Section 6: direct Lorentzian propagation along the paired direction (kappa = 1)")
    table, M = sym.lorentz_symbol(1, per_permutation=False)
    MP = sym.paired(M)
    section_6_1_clock(rep)
    c_cross = section_6_2_cross(rep, MP)
    O, B = section_6_3_plus(rep, MP)
    numeric_branches(rep, M, MP, B, c_cross)
    euclidean_comparison(rep)
    return rep.finish(out_path(argv, None))


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv) else 0)
