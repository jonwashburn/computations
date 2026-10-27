#!/usr/bin/env python3
"""
sections8_9_bound_numbers.py -- every number in the conditional Cherenkov bound and the spacing section.

Companion to: J. Washburn and P. Beltracchi, "Graviton Modes on the Kuhn
Lattice: Euclidean Continuation and Lorentzian Propagation", Section 8
("A conditional gravitational Cherenkov bound") and Section 9 ("A candidate
lattice spacing").

These sections use the AXIS-TIME EUCLIDEAN branch, omega/p = 1 - kappa(p) a^2 p^2 with
kappa = (1 + A)/24 and A = sum_j n_j^4 (Section 5, eq:kappa). They do not use the direct
Lorentzian branches of Section 6. The emission rate is imported from Kostelecky and Tasson
(the factor F_psi), with their matter assumptions. This program recomputes each printed number
from those inputs and the constants of Appendix B, and compares it with the printed value at the
printed number of significant digits.

Inputs (Appendix B, "Constants", and Section 8): E_P = 1.220890e19 GeV, hbar c = 1.97327e-16 GeV m,
1 Mpc = 3.0857e22 m, the Fly's Eye energy 3.2e20 eV. The proton mass 0.938272 GeV used for 1 - v
is the standard value; the paper does not print it.
Run:  python sections8_9_bound_numbers.py [outputs/sections8_9_bound_numbers.json]
Runtime: a few seconds.
"""
import sys

import mpmath as mp
import sympy as sp

from report import Report, out_path

mp.mp.dps = 30
E_P = mp.mpf("1.220890e19")          # GeV
HBARC = mp.mpf("1.97327e-16")        # GeV m  (1 GeV^-1 = 1.97327e-16 m)
MPC = mp.mpf("3.0857e22")            # m
M_PROTON = mp.mpf("0.938272")        # GeV
E_FLYS_EYE = mp.mpf("3.2e20")        # eV
G_N = 1 / E_P**2                     # GeV^-2
L_P_M = HBARC / E_P                  # Planck length in m
F_PSI6 = sp.Rational(9, 13)


def printed_match(value, printed):
    """True when value rounds to the printed decimal (same significant digits)."""
    mant, _, exp = printed.lower().partition("e")
    digits = len(mant.replace("-", "").replace(".", "").lstrip("0"))
    p = mp.mpf(printed)
    e10 = mp.floor(mp.log10(abs(p)))
    half_ulp = mp.mpf(10) ** (e10 - digits + 1) / 2
    return abs(value - p) <= half_ulp * (1 + mp.mpf("1e-12"))


def num(rep, label, value, printed):
    return rep.check("%s = %s (printed %s)" % (label, mp.nstr(value, 6), printed), printed_match(value, printed))


def L_gev(mpc):
    return mpc * MPC / HBARC                       # GeV^-1


def s_bound(E_f_eV, L_mpc):
    """eq:kt: s^(6) < (F_psi(6) / (G_N E_f^7 L))^(1/2), in GeV^-2."""
    E = E_f_eV / mp.mpf(10)**9
    return mp.sqrt(mp.mpf(9) / 13 / (G_N * E**7 * L_gev(L_mpc)))


def a_thr(E_f_eV, L_mpc, A):
    """eq:athr: a_thr = (12/(1 + A))^(1/2) (F_psi/(G_N E_f^7 L))^(1/4), in GeV^-1."""
    return mp.sqrt(12 / (1 + A)) * mp.sqrt(s_bound(E_f_eV, L_mpc))


def attenuation_energy(a_gev, A, L_mpc):
    """Parton energy (eV) at which s^(6) = (1 + A) a^2/12 equals the survival limit of eq:kt."""
    s = (1 + A) * a_gev**2 / 12
    E = (mp.mpf(9) / 13 / (G_N * L_gev(L_mpc) * s**2)) ** (mp.mpf(1) / 7)
    return E * mp.mpf(10)**9


def main(argv):
    rep = Report("Sections 8 and 9: the conditional bound and the candidate spacing (axis-time branch)")
    Asym, asym, ysym, psym = sp.symbols("A a y p", positive=True)

    rep.section("8.1  SME coefficients of the axis-time branch")
    kappa = (1 + Asym) / 24
    rep.check("s^(6) = 2 kappa a^2 = (1 + A) a^2/12 (eq:s6)", sp.simplify(2 * kappa - (1 + Asym) / 12) == 0)
    rep.check("F_psi(6) = (d-2)(d-3)(2d-3)/(4(2d^2-7d+9)) at d = 6 is 9/13",
              sp.Rational(4 * 3 * 9, 4 * (72 - 42 + 9)) == F_PSI6)
    s00 = 4 * mp.sqrt(mp.pi) / 15
    s40 = mp.sqrt(mp.pi) / 45
    num(rep, "s00/a^2 = 4 sqrt(pi)/15", s00, "0.4727")
    num(rep, "s40/a^2 = sqrt(pi)/45", s40, "0.03939")
    num(rep, "s4,+-4/a^2 = sqrt(5/14) s40", mp.sqrt(mp.mpf(5) / 14) * s40, "0.02354")
    num(rep, "real cos(4 phi) harmonic: sqrt(5/7) s40", mp.sqrt(mp.mpf(5) / 7) * s40, "0.03329")
    num(rep, "s44/s40 = sqrt(5/14)", mp.sqrt(mp.mpf(5) / 14), "0.5976")
    rep.check("mean of s^(6) over directions is (1 + 3/5) a^2/12 = 2 a^2/15",
              (1 + sp.Rational(3, 5)) / 12 == sp.Rational(2, 15))
    # Appendix B, "Spherical components": A(n) = 3/5 + (4 sqrt(pi)/15)[Y40 + sqrt(5/14)(Y44 + Y4,-4)].
    worst = mp.mpf(0)
    for th, ph in ((0.3, 1.1), (1.2, 2.5), (2.0, -0.7), (0.9, 0.4), (2.7, 3.0)):
        th, ph = mp.mpf(th), mp.mpf(ph)
        n = (mp.sin(th) * mp.cos(ph), mp.sin(th) * mp.sin(ph), mp.cos(th))
        A = sum(c**4 for c in n)
        rhs = mp.mpf(3) / 5 + 4 * mp.sqrt(mp.pi) / 15 * (mp.spherharm(4, 0, th, ph) + mp.sqrt(mp.mpf(5) / 14) *
                                                         (mp.spherharm(4, 4, th, ph) + mp.spherharm(4, -4, th, ph)))
        worst = max(worst, abs(A - rhs))
    rep.check("A(n) = 3/5 + (4 sqrt(pi)/15)[Y40 + sqrt(5/14)(Y44 + Y4,-4)] at five directions",
              worst < mp.mpf("1e-25"), mp.nstr(worst, 3))
    rep.check("on an axis: 3/5 + (4 sqrt(pi)/15)(3/(2 sqrt(pi))) = 1",
              sp.Rational(3, 5) + 4 * sp.sqrt(sp.pi) / 15 * 3 / (2 * sp.sqrt(sp.pi)) == 1)
    rep.check("minimum of s^(6) is a^2/9 on the body diagonal (A = 1/3)", (1 + sp.Rational(1, 3)) / 12 == sp.Rational(1, 9))

    rep.section("8.3  The bound from the Fly's Eye event (iron, r = 0.1, L = 10 Mpc)")
    E_f = E_FLYS_EYE / 560
    num(rep, "E_f = E/560 (eV)", E_f, "5.7e17")
    s = s_bound(E_f, 10)
    num(rep, "s^(6)(p) bound (GeV^-2), eq:sdir", s, "5.8e-32")
    num(rep, "s00 = sqrt(4 pi) s bound (GeV^-2)", mp.sqrt(4 * mp.pi) * s, "2.0e-31")
    a_max = mp.sqrt(9 * s)
    num(rep, "a bound (GeV^-1), eq:bound", a_max, "7.2e-16")
    num(rep, "a bound (m)", a_max * HBARC, "1.42e-31")
    num(rep, "a bound / l_P", a_max * E_P, "8.8e3")
    num(rep, "1/a bound (GeV)", 1 / a_max, "1.39e15")
    num(rep, "axis-direction bound (m)", mp.sqrt(6 * s) * HBARC, "1.16e-31")
    num(rep, "isotropic-only benchmark from s00 alone (m)",
        mp.sqrt(mp.sqrt(4 * mp.pi) * s * 15 / (4 * mp.sqrt(mp.pi))) * HBARC, "1.3e-31")

    rep.section("Table 5: excluded spacings a_thr / l_P along an axis (A = 1)")
    rows = [("iron, r = 0.1", E_FLYS_EYE / 560, "5.7e17", "7.2e3", "4.0e3"),
            ("proton, r = 0.1", E_FLYS_EYE / 10, "3.2e19", "6.3", "3.5"),
            ("proton, E_f = 1e20 eV", mp.mpf("1e20"), "1.0e20", "0.85", "0.48"),
            ("proton, r = 1", E_FLYS_EYE, "3.2e20", "0.11", "0.063")]
    for label, Ef, pE, p10, p100 in rows:
        num(rep, label + ": E_f (eV)", Ef, pE)
        num(rep, label + ": a_thr/l_P, 10 Mpc", a_thr(Ef, 10, 1) * E_P, p10)
        num(rep, label + ": a_thr/l_P, 100 Mpc", a_thr(Ef, 100, 1) * E_P, p100)
    # The iron row sits close to a rounding boundary. a_thr scales as E_f^(-7/4), so the 0.25 percent
    # difference between E/560 = 5.714e17 eV and its rounded form 5.7e17 eV moves it by 0.44 percent:
    # 4036 (printed 4.0e3) with the paper's E_f = E/560, 4054 with the rounded value.
    rep.note("iron row with E_f = E/560 = %s eV: %s (10 Mpc), %s (100 Mpc)" %
             (mp.nstr(E_FLYS_EYE / 560, 6), mp.nstr(a_thr(E_FLYS_EYE / 560, 10, 1) * E_P, 6),
              mp.nstr(a_thr(E_FLYS_EYE / 560, 100, 1) * E_P, 6)))
    rep.note("iron row with the rounded E_f = 5.7e17 eV: %s (10 Mpc), %s (100 Mpc)" %
             (mp.nstr(a_thr(mp.mpf("5.7e17"), 10, 1) * E_P, 6), mp.nstr(a_thr(mp.mpf("5.7e17"), 100, 1) * E_P, 6)))
    rep.note("The bound (eq:bound) and Section 9 numbers above match the printed values only with E_f = E/560;")
    rep.note("with 5.7e17 eV they would read 1.43e-31 m, 1.38e15 GeV and 1.87e7.")
    num(rep, "body-diagonal factor (2/(4/3))^(1/2)", mp.sqrt(mp.mpf(2) / (mp.mpf(4) / 3)), "1.22")
    num(rep, "r = 0.31 for E_f = 1e20 eV", mp.mpf("1e20") / E_FLYS_EYE, "0.31")
    num(rep, "Planck spacing excluded by factor (proton r = 1, 10 Mpc)", 1 / (a_thr(E_FLYS_EYE, 10, 1) * E_P), "9")
    num(rep, "Planck spacing excluded by factor (proton r = 1, 100 Mpc)", 1 / (a_thr(E_FLYS_EYE, 100, 1) * E_P), "16")

    rep.section("8.4  What a Planck-scale spacing predicts")
    aP = 1 / E_P
    num(rep, "s^(6) minimum at a = l_P (GeV^-2)", aP**2 / 9, "7.5e-40")
    num(rep, "s^(6) maximum at a = l_P (GeV^-2)", aP**2 / 6, "1.1e-39")
    num(rep, "factor inside eq:sdir", s / (aP**2 / 6), "5e7")
    num(rep, "attenuation energy, axis, 10 Mpc (eV)", attenuation_energy(aP, 1, 10), "9.1e19")
    num(rep, "attenuation energy, body diagonal, 10 Mpc (eV)", attenuation_energy(aP, mp.mpf(1) / 3, 10), "1.02e20")
    num(rep, "attenuation energy, axis, 100 Mpc (eV)", attenuation_energy(aP, 1, 100), "6.6e19")
    one_minus_v = M_PROTON**2 / (2 * mp.mpf("1e11")**2)
    num(rep, "1 - v for a 1e20 eV proton", one_minus_v, "4.4e-23")
    num(rep, "graviton wave number where the axis phase speed drops below that proton (GeV)",
        mp.sqrt(12 * one_minus_v) * E_P, "2.8e8")
    rep.check("|l| a <= E_f a ~ 1e-8 at a Planck spacing (E_f = 1e11 GeV)",
              mp.nint(mp.log10(mp.mpf("1e11") / E_P)) == -8, mp.nstr(mp.mpf("1e11") / E_P, 3))
    rep.check("relative higher-order correction (l a)^2 is of order 1e-16",
              mp.nint(mp.log10((mp.mpf("1e11") / E_P)**2)) == -16, mp.nstr((mp.mpf("1e11") / E_P)**2, 3))
    k_gw = 2 * mp.pi * 100 / mp.mpf("2.99792458e8")
    num(rep, "k = 2 pi x 100 Hz / c (1/m)", k_gw, "2e-6")
    rep.check("defect kappa k^2 a^2 at a Planck spacing is of order 1e-82 (axis kappa = 1/12)",
              mp.nint(mp.log10(k_gw**2 * L_P_M**2 / 12)) == -82, mp.nstr(k_gw**2 * L_P_M**2 / 12, 3))

    rep.section("9  The bit spacing a = sqrt(4 ln 2) l_P")
    r = mp.sqrt(4 * mp.log(2))
    num(rep, "sqrt(4 ln 2)", r, "1.665")
    num(rep, "a (m)", r * L_P_M, "2.6912e-35")
    num(rep, "E_lat = E_P/sqrt(4 ln 2) (GeV)", E_P / r, "7.3322e18")
    ab = r / E_P
    num(rep, "s^(6) minimum (GeV^-2)", ab**2 / 9, "2.067e-39")
    num(rep, "s^(6) maximum (GeV^-2)", ab**2 / 6, "3.100e-39")
    num(rep, "factor below the directional limit", s / (ab**2 / 6), "1.86e7")
    num(rep, "attenuation energy, axis, 10 Mpc (eV)", attenuation_energy(ab, 1, 10), "6.82e19")
    num(rep, "attenuation energy, body diagonal, 10 Mpc (eV)", attenuation_energy(ab, mp.mpf(1) / 3, 10), "7.66e19")
    num(rep, "attenuation energy, axis, 100 Mpc (eV)", attenuation_energy(ab, 1, 100), "4.91e19")
    rep.check("Kuhn triangle areas 1/2, sqrt2/2, sqrt3/2, 1 for legs (1,1),(1,2),(1,3),(2,2)",
              [sp.sqrt(r1) * sp.sqrt(q1) / 2 for r1, q1 in ((1, 1), (1, 2), (1, 3), (2, 2))]
              == [sp.Rational(1, 2), sp.sqrt(2) / 2, sp.sqrt(3) / 2, 1])

    rep.section("9.1  Photon and graviton propagation (collinear energy balance)")
    w_g = lambda k: k - (1 + Asym) * asym**2 * k**3 / 24          # axis-time graviton, leading order
    w_g1 = lambda k: k - Asym * asym**2 * k**3 / 24               # model gamma-1
    rep.check("v_ph,gamma1 - v_ph,g = a^2 k^2/24",
              sp.simplify(w_g1(psym) / psym - w_g(psym) / psym - asym**2 * psym**2 / 24) == 0)
    bal1 = sp.expand(w_g1(psym) - w_g1((1 - ysym) * psym) - w_g(ysym * psym))
    rep.check("gamma-1 excess energy = (a^2 p^3/24) y (y^2 + 3Ay - 3A) (eq:photon-emission)",
              sp.expand(bal1 - asym**2 * psym**3 / 24 * ysym * (ysym**2 + 3 * Asym * ysym - 3 * Asym)) == 0)
    ystar = (sp.sqrt(9 * Asym**2 + 12 * Asym) - 3 * Asym) / 2
    rep.check("y_* = (sqrt(9A^2 + 12A) - 3A)/2 is the positive root",
              sp.simplify(ystar**2 + 3 * Asym * ystar - 3 * Asym) == 0)
    rep.check("y_*(1/3) = (sqrt5 - 1)/2 and y_*(1) = (sqrt21 - 3)/2",
              sp.simplify(ystar.subs(Asym, sp.Rational(1, 3)) - (sp.sqrt(5) - 1) / 2) == 0
              and sp.simplify(ystar.subs(Asym, 1) - (sp.sqrt(21) - 3) / 2) == 0)
    rep.check("y_* increases with A on [1/3, 1], so it lies between those endpoints",
              all(sp.diff(ystar, Asym).subs(Asym, sp.Rational(k, 30)) > 0 for k in range(10, 31)))
    num(rep, "y_*(1/3)", mp.mpf(float(sp.N(ystar.subs(Asym, sp.Rational(1, 3)), 20))), "0.6180")
    num(rep, "y_*(1)", mp.mpf(float(sp.N(ystar.subs(Asym, 1), 20))), "0.7913")
    bal2 = sp.expand(w_g(psym) - w_g((1 - ysym) * psym) - w_g(ysym * psym))
    rep.check("gamma-2 (= the axis-time law) excess = (a^2 p^3/8)(1 + A) y (y - 1) (eq:photon-2)",
              sp.expand(bal2 - asym**2 * psym**3 / 8 * (1 + Asym) * ysym * (ysym - 1)) == 0)
    return rep.finish(out_path(argv, None))


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv) else 0)
