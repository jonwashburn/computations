"""Finite-momentum edge modes of the published axis-time Kuhn symbol.

See MODE_CONSTRUCTION.txt for the gauge, phase convention and derivation.
Only Python's standard library is needed. No continuum tensor is substituted
directly into the edge field: lattice differences and W(k) are applied first.
"""
import cmath
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path

PAIRS = list(itertools.combinations(range(4), 2))
AXIS_FACE = [0, 1, 3, 7] + [(1 << i) + (1 << j) - 1 for i, j in PAIRS]
BODY = [6, 10, 12, 13]
TABLE_SHA = 'c2faba52960f1cf4de7b8bb4c2e0da599b2335dc6b7f76743ed90a95157e1b31'


@lru_cache(maxsize=1)
def couplings():
    path = Path(__file__).resolve().parent / 'reference/regge_4d_kuhn_coupling_table_20260903.json'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TABLE_SHA:
        raise ValueError('The coupling table does not match the published release')
    compiled = []
    for row in json.loads(data)['couplings']:
        d, e, v = (row[key] for key in ('D', 'Dprime', 'two_times_midpoint_separation'))
        i, j = [sum(int(x) << a for a, x in enumerate(vec)) - 1 for vec in (d, e)]
        powers = [(v[a] + d[a] - e[a]) / 2 for a in range(4)]
        if any(int(q) != q for q in powers):
            raise ValueError('Nonintegral vertex-phase exponent')
        compiled.append((i, j, float(Fraction(row['weight_exact'])), [int(q) for q in powers]))
    if len(compiled) != 302:
        raise ValueError('Incomplete coupling table')
    return compiled


def edge_symbol(z):
    matrix = [[0j for _ in range(15)] for _ in range(15)]
    for i, j, weight, powers in couplings():
        term = complex(weight)
        for a in range(4):
            term *= z[a] ** powers[a]
        matrix[i][j] += term
    return matrix


def norm(v):
    return math.sqrt(sum(abs(x) ** 2 for x in v))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def encode(values):
    return [[float(x.real), float(x.imag)] for x in values]


def modes(k, n, omega):
    p = [k*x for x in n] + [1j*omega]
    z = [cmath.exp(1j*x) for x in p]
    q = [2*math.sin(x/2) for x in p[:3]]
    qnorm = norm(q)
    direction = [x/qnorm for x in q] if qnorm else list(n)
    a = cross(direction, [0, 0, 1] if abs(direction[2]) < .9 else [0, 1, 0])
    anorm = norm(a)
    a = [x/anorm for x in a]
    b = cross(direction, a)
    matrix = edge_symbol(z)
    matrix_norm = norm([x for row in matrix for x in row])
    result = {}
    for name in ('plus', 'cross'):
        h = [[0.0 for _ in range(4)] for _ in range(4)]
        for i in range(3):
            for j in range(3):
                h[i][j] = (a[i]*a[j]-b[i]*b[j] if name == 'plus'
                           else a[i]*b[j]+b[i]*a[j])
        # y is trace reversed. Here trace(H)=0 and all time components vanish.
        y = [complex(h[i][i]) for i in range(4)]
        y += [cmath.exp(1j*(p[i]+p[j])/2)*h[i][j] for i, j in PAIRS]
        cy = [(z[i]-1)*y[i] for i in range(4)]
        for index, (i, j) in enumerate(PAIRS):
            cy[i] += (1-1/z[j])*y[4+index]
            cy[j] += (1-1/z[i])*y[4+index]
        # X = W y for the ten axis/face classes; trace(y_diagonal)=0.
        x = [0j]*15
        for i in range(4):
            x[(1 << i)-1] = z[i]*y[i]
        for index, (i, j) in enumerate(PAIRS):
            x[(1 << i)+(1 << j)-1] = (2*y[4+index] - z[i]*z[j]
                *sum(y[r] for r in range(4) if r not in (i, j)))
        # The body block is -I/2 at all momenta. Set the inert 15th edge to zero.
        for i in BODY:
            x[i] = 2*sum(matrix[i][j]*x[j] for j in AXIS_FACE)
        residual = [sum(matrix[i][j]*x[j] for j in range(15)) for i in range(15)]
        relative_residual = norm(residual)/(matrix_norm*norm(x))
        constraint_residual = norm(cy)/max(norm(y), 1e-300)
        if relative_residual > 1e-10 or constraint_residual > 1e-10:
            raise ArithmeticError('The lattice mode failed its field-equation check')
        # Linear length strain = squared-length perturbation / (2 background a_D).
        strains = [x[mask-1]/(2*bin(mask).count('1')) for mask in range(1, 16)]
        peak = max(abs(value) for value in strains[:7])
        if peak <= 0:
            raise ArithmeticError('A zero spatial mode cannot be visualized')
        result[name] = {'strains': encode([value/peak for value in strains]),
                        'squared_edge_amplitudes': encode(x), 'y': encode(y),
                        'edge_equation_residual': relative_residual,
                        'constraint_residual': constraint_residual}
    return {'polarizations': result, 'lattice_transverse_direction': direction,
            'kind': 'propagating' if k > 0 else 'uniform-limit',
            'normalization': 'Peak spatial edge-strain amplitude is one; overall amplitude is arbitrary.'}
