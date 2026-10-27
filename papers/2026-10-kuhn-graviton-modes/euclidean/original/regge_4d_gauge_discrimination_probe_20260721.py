#!/usr/bin/env python3
"""Exact 4D Regge second-variation gauge/TT discrimination probe (2026-07-21).

Independent of Bloch-fold covectors: builds the Kuhn/Freudenthal N^4 torus,
evaluates S = sum_h Area_h * deficit_h from Heron areas and Gram-projection
dihedral angles, and measures
d^2S/dε^2 by central finite differences.

Tables and conventions: simplex vertex masks and axis permutations, edge classes with their
displacements, the gauge and TT parts of an edge field, the symbol direction, the class pairing
edgeLoad = D^T H D, and the hinge orbit classification by triangle index triples.

Does NOT import or reuse fold deficit/area covectors.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

PERM_AXES = [
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1),
    (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2),
    (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0),
    (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1),
    (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0),
]

TRI_INDEX = [
    (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
    (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
]

AXIS_TT_PLUS = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, -1]], dtype=float,
)

SYMBOL_DIR = np.array([1.0, 1.0, 0.0, 0.0])
AXIS_WAVE = np.array([1.0, 0.0, 0.0, 0.0])
E0 = np.array([1.0, 0.0, 0.0, 0.0])
E1 = np.array([0.0, 1.0, 0.0, 0.0])
E2 = np.array([0.0, 0.0, 1.0, 0.0])
E3 = np.array([0.0, 0.0, 0.0, 1.0])


def mask_to_disp(mask: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(4)], dtype=float)


def popcount(mask: int) -> int:
    return bin(mask).count("1")


def vertex_mask(s: int) -> List[int]:
    axes = PERM_AXES[s]
    out = [0]
    acc = 0
    for k in range(4):
        acc |= 1 << axes[k]
        out.append(acc)
    return out


def class_disp(d: int) -> np.ndarray:
    return mask_to_disp(d + 1)


def edge_load(H: np.ndarray, D: np.ndarray) -> float:
    return float(D @ H @ D)


def gauge_part(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.outer(m, v) + np.outer(v, m)


def heron_area(la: float, lb: float, lc: float) -> float:
    s = 0.5 * (la + lb + lc)
    val = s * (s - la) * (s - lb) * (s - lc)
    return math.sqrt(max(val, 0.0))


def hinge_gram_det(a: np.ndarray) -> float:
    return 4 * a[0] * a[1] - (a[0] + a[1] - a[4]) ** 2


def apex_dot_num(a: np.ndarray) -> float:
    den = hinge_gram_det(a)
    return (
        (-2) * a[0] * (a[1] + a[2] - a[7]) * (a[1] + a[3] - a[8])
        + (-2) * a[1] * (a[0] + a[2] - a[5]) * (a[0] + a[3] - a[6])
        + den * (a[2] + a[3] - a[9])
        + (a[0] + a[1] - a[4]) * (a[0] + a[2] - a[5]) * (a[1] + a[3] - a[8])
        + (a[0] + a[1] - a[4]) * (a[0] + a[3] - a[6]) * (a[1] + a[2] - a[7])
    )


def apex3_norm_sq_num(a: np.ndarray) -> float:
    den = hinge_gram_det(a)
    return (
        (-a[0]) * (a[1] + a[2] - a[7]) ** 2
        + (-a[1]) * (a[0] + a[2] - a[5]) ** 2
        + a[2] * den
        + (a[0] + a[1] - a[4]) * (a[0] + a[2] - a[5]) * (a[1] + a[2] - a[7])
    )


def apex4_norm_sq_num(a: np.ndarray) -> float:
    den = hinge_gram_det(a)
    return (
        (-a[0]) * (a[1] + a[3] - a[8]) ** 2
        + (-a[1]) * (a[0] + a[3] - a[6]) ** 2
        + a[3] * den
        + (a[0] + a[1] - a[4]) * (a[0] + a[3] - a[6]) * (a[1] + a[3] - a[8])
    )


def cos_dihedral(a: np.ndarray) -> float:
    num = apex_dot_num(a)
    den = 2.0 * math.sqrt(max(apex3_norm_sq_num(a) * apex4_norm_sq_num(a), 0.0))
    if den == 0.0:
        return 1.0
    return float(np.clip(num / den, -1.0, 1.0))


def sq_edges_for_hinge(d2: np.ndarray, i: int, j: int, k: int, p: int, q: int) -> np.ndarray:
    idx = (i, j, k, p, q)
    a = np.zeros(10, dtype=float)
    pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    ]
    for s, (u, v) in enumerate(pairs):
        a[s] = d2[idx[u], idx[v]]
    return a


def orbit_type_of_masks(m0: int, m1: int, m2: int) -> str:
    a = popcount(m0 ^ m1)
    b = popcount(m1 ^ m2)
    return {
        (1, 1): "t11", (1, 2): "t12", (2, 1): "t21",
        (1, 3): "t13", (3, 1): "t31", (2, 2): "t22",
    }.get((a, b), f"other_{a}_{b}")


def build_complex(N: int):
    simp_masks = [vertex_mask(s) for s in range(24)]
    incidences = []
    for cell in itertools.product(range(N), repeat=4):
        for s in range(24):
            masks = simp_masks[s]
            verts = []
            for m in masks:
                D = mask_to_disp(m).astype(int)
                verts.append(tuple((cell[i] + int(D[i])) % N for i in range(4)))
            for t, (i, j, k) in enumerate(TRI_INDEX):
                hi = frozenset((verts[i], verts[j], verts[k]))
                trip = sorted([(popcount(masks[u]), masks[u], u) for u in (i, j, k)])
                mm = [trip[0][1], trip[1][1], trip[2][1]]
                orb = orbit_type_of_masks(mm[0], mm[1], mm[2])
                apexes = [u for u in range(5) if u not in (i, j, k)]
                incidences.append({
                    "cell": cell,
                    "s": s,
                    "t": t,
                    "hinge": hi,
                    "orbit": orb,
                    "ijk": (i, j, k),
                    "apex": (apexes[0], apexes[1]),
                    "verts_mod": verts,
                    "masks": masks,
                })
    return incidences, simp_masks


def edge_lookup_key(u, v, N: int):
    du = tuple((v[i] - u[i]) % N for i in range(4))
    if all(d in (0, 1) for d in du) and any(d == 1 for d in du):
        base, disp = u, du
    else:
        dv = tuple((u[i] - v[i]) % N for i in range(4))
        base, disp = v, dv
    mask = sum(int(disp[i]) << i for i in range(4))
    return (base[0], base[1], base[2], base[3], mask)


def flat_l2_field(N: int) -> Dict[Tuple, float]:
    field = {}
    for base in itertools.product(range(N), repeat=4):
        for mask in range(1, 16):
            field[(base[0], base[1], base[2], base[3], mask)] = float(popcount(mask))
    return field


def lengths_from_vertex_gauge(N: int, eps: float, m_mode: np.ndarray, v: np.ndarray, phase: str):
    k = 2.0 * math.pi * m_mode / N
    field = {}
    for base in itertools.product(range(N), repeat=4):
        xb = np.array(base, dtype=float)
        phase_b = float(k @ xb)
        amp_b = math.sin(phase_b) if phase == "sin" else math.cos(phase_b)
        p0 = xb + eps * v * amp_b
        for mask in range(1, 16):
            D = mask_to_disp(mask)
            end = tuple((base[i] + int(D[i])) % N for i in range(4))
            xe = np.array(end, dtype=float)
            phase_e = float(k @ xe)
            amp_e = math.sin(phase_e) if phase == "sin" else math.cos(phase_e)
            p1 = xb + D + eps * v * amp_e
            field[(base[0], base[1], base[2], base[3], mask)] = float(np.sum((p1 - p0) ** 2))
    return field


def lengths_from_metric_wave(N: int, eps: float, H: np.ndarray, m_mode: np.ndarray):
    k = 2.0 * math.pi * m_mode / N
    field = {}
    for base in itertools.product(range(N), repeat=4):
        xb = np.array(base, dtype=float)
        for mask in range(1, 16):
            D = mask_to_disp(mask)
            mid = xb + 0.5 * D
            c = edge_load(H, D)
            l2 = float(popcount(mask) + eps * c * math.cos(float(k @ mid)))
            field[(base[0], base[1], base[2], base[3], mask)] = l2
    return field


def simplex_d2_from_field(field, cell, masks, N: int) -> np.ndarray:
    d2 = np.zeros((5, 5), dtype=float)
    verts_mod = []
    for m in masks:
        D = mask_to_disp(m).astype(int)
        verts_mod.append(tuple((cell[i] + int(D[i])) % N for i in range(4)))
    for a in range(5):
        for b in range(a + 1, 5):
            key = edge_lookup_key(verts_mod[a], verts_mod[b], N)
            d2[a, b] = d2[b, a] = field[key]
    return d2


def action_from_field(field, incidences, N: int, collect_orbits: bool = False):
    angle_sum = defaultdict(float)
    area = {}
    orbit_of = {}
    for inc in incidences:
        masks = inc["masks"]
        d2 = simplex_d2_from_field(field, inc["cell"], masks, N)
        i, j, k = inc["ijk"]
        p, q = inc["apex"]
        a = sq_edges_for_hinge(d2, i, j, k, p, q)
        theta = math.acos(cos_dihedral(a))
        hk = inc["hinge"]
        angle_sum[hk] += theta
        if hk not in area:
            la = math.sqrt(max(d2[i, j], 0.0))
            lb = math.sqrt(max(d2[i, k], 0.0))
            lc = math.sqrt(max(d2[j, k], 0.0))
            area[hk] = heron_area(la, lb, lc)
            orbit_of[hk] = inc["orbit"]
    S = 0.0
    max_def = 0.0
    orbit_S = defaultdict(float)
    for hk, thsum in angle_sum.items():
        delta = 2.0 * math.pi - thsum
        contrib = area[hk] * delta
        S += contrib
        max_def = max(max_def, abs(delta))
        if collect_orbits:
            orbit_S[orbit_of[hk]] += contrib
    if collect_orbits:
        return S, max_def, dict(orbit_S)
    return S, max_def


def central_second(f, eps: float):
    f0 = f(0.0)
    d2_big = (f(eps) - 2.0 * f0 + f(-eps)) / (eps ** 2)
    e2 = 0.5 * eps
    d2_small = (f(e2) - 2.0 * f0 + f(-e2)) / (e2 ** 2)
    return d2_big, d2_small, f0


def k_sq(N: int, m_mode: np.ndarray) -> float:
    k = 2.0 * math.pi * m_mode / N
    return float(k @ k)


def run_flat_validation(N: int, incidences):
    field = flat_l2_field(N)
    S, max_def = action_from_field(field, incidences, N)
    key = (0, 0, 0, 0, 1)
    h = 1e-6
    field_p = dict(field)
    field_p[key] = field[key] + h
    field_m = dict(field)
    field_m[key] = field[key] - h
    Sp, _ = action_from_field(field_p, incidences, N)
    Sm, _ = action_from_field(field_m, incidences, N)
    grad = (Sp - Sm) / (2.0 * h)
    return {"S_flat": S, "max_abs_deficit": max_def, "grad_sample_edge_01": grad}


def gauge_second_deriv(N, incidences, m_mode, v, phase: str, eps: float):
    def f(e):
        S, _ = action_from_field(
            lengths_from_vertex_gauge(N, e, m_mode, v, phase), incidences, N
        )
        return S

    d2_big, d2_small, S0 = central_second(f, eps)
    _, _, orb = action_from_field(
        lengths_from_vertex_gauge(N, eps, m_mode, v, phase),
        incidences, N, collect_orbits=True,
    )
    ks = k_sq(N, m_mode)
    return {
        "phase": phase,
        "d2S_eps": d2_big,
        "d2S_eps_half": d2_small,
        "S0": S0,
        "richardson_rel": abs(d2_big - d2_small) / max(abs(d2_small), 1e-30),
        "orbit_S_at_eps": orb,
        "k_sq": ks,
        "N4": N ** 4,
        "norm_coeff": d2_small / (N ** 4 * ks) if ks > 0 else None,
    }


def tt_second_deriv(N, incidences, H, m_mode, eps: float):
    def f(e):
        S, _ = action_from_field(
            lengths_from_metric_wave(N, e, H, m_mode), incidences, N
        )
        return S

    d2_big, d2_small, S0 = central_second(f, eps)
    ks = k_sq(N, m_mode)
    norm = d2_small / (N ** 4 * ks) if ks > 0 else None
    fold_m2 = -0.25
    return {
        "d2S_eps": d2_big,
        "d2S_eps_half": d2_small,
        "S0": S0,
        "richardson_rel": abs(d2_big - d2_small) / max(abs(d2_small), 1e-30),
        "k_sq": ks,
        "N4": N ** 4,
        "norm_coeff": norm,
        "fold_m2_target": fold_m2,
        "ratio_norm_to_fold_m2": (norm / fold_m2) if norm is not None else None,
    }


def self_check():
    a = np.array([1, 2, 3, 4, 1, 2, 3, 1, 2, 1], dtype=float)
    assert abs(cos_dihedral(a) - 1 / math.sqrt(2)) < 1e-12
    m = np.array([1.0, 1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0, 0.0])
    H = gauge_part(m, v)
    for d in range(15):
        D = class_disp(d)
        lhs = edge_load(H, D)
        rhs = 2 * float(m @ D) * float(v @ D)
        assert abs(lhs - rhs) < 1e-12, (d, lhs, rhs)
    return True


def main():
    t0 = time.time()
    self_check()
    eps = 1e-4
    results = {
        "script": "original/regge_4d_gauge_discrimination_probe_20260721.py",
        "eps": eps,
        "fold_banked": {
            "axisTTPlus_symbolDir_distinct_hinge_m2": -0.25,
            "gauge_m1100_e2_distinct_hinge_m2_approx": -1.0,
            "decoy_gauge_axisWave_e0_m2": 0.0,
            "lesson": "L-p1-transported-gauge-zero-false-for-general-v",
        },
    }

    for N in (4, 6):
        print(f"=== Building complex N={N} ===", flush=True)
        incidences, _ = build_complex(N)
        print(f"incidences={len(incidences)}", flush=True)
        flat = run_flat_validation(N, incidences)
        print("flat", flat, flush=True)
        results[f"N{N}"] = {"flat": flat, "gauges": {}, "tt": {}}

        gauge_cases = [
            ("counterexample_m1100_e2", np.array([1.0, 1.0, 0.0, 0.0]), E2),
            ("decoy_axisWave_e0", AXIS_WAVE.copy(), E0),
            ("m1100_e3", np.array([1.0, 1.0, 0.0, 0.0]), E3),
            ("m1000_e1", np.array([1.0, 0.0, 0.0, 0.0]), E1),
        ]
        for name, m_mode, v in gauge_cases:
            print(f"gauge {name} N={N}", flush=True)
            out_sin = gauge_second_deriv(N, incidences, m_mode, v, "sin", eps)
            out_cos = gauge_second_deriv(N, incidences, m_mode, v, "cos", eps)
            results[f"N{N}"]["gauges"][name] = {"sin": out_sin, "cos": out_cos}
            print(
                f"  sin d2={out_sin['d2S_eps_half']:.6e} norm={out_sin['norm_coeff']}",
                flush=True,
            )

        print(f"TT axisTTPlus / symbolDir N={N}", flush=True)
        tt = tt_second_deriv(N, incidences, AXIS_TT_PLUS, SYMBOL_DIR, eps)
        results[f"N{N}"]["tt"]["axisTTPlus_symbolDir"] = tt
        print(
            f"  d2={tt['d2S_eps_half']:.6e} norm={tt['norm_coeff']} "
            f"ratio_to_-1/4={tt['ratio_norm_to_fold_m2']}",
            flush=True,
        )

    g = results["N6"]["gauges"]["counterexample_m1100_e2"]["sin"]
    decoy = results["N6"]["gauges"]["decoy_axisWave_e0"]["sin"]
    tt = results["N6"]["tt"]["axisTTPlus_symbolDir"]
    g4 = results["N4"]["gauges"]["counterexample_m1100_e2"]["sin"]
    g_norm = g["norm_coeff"]
    fold_gauge = -1.0
    disc = abs(g_norm - g4["norm_coeff"]) / max(abs(g_norm), 1e-30)

    if abs(g_norm) < 0.05:
        verdict = "H_fold"
        reason = (
            "True Hessian quadratic form on counterexample gauge mode is near zero "
            f"(norm_coeff={g_norm:.4e}); fold's m2≈-1 is a fold-transport artifact."
        )
    elif abs(g_norm) > 0.3 and math.copysign(1, g_norm) == math.copysign(1, fold_gauge):
        verdict = "H_pred"
        reason = (
            "True Hessian is nonzero on the counterexample gauge mode "
            f"(norm_coeff={g_norm:.4e}, fold m2≈-1); gauge-zero predicate over-broad; "
            "fold vindicated at leading continuum order."
        )
    else:
        if abs(g_norm) < abs(g_norm - fold_gauge):
            verdict = "H_fold"
            reason = f"Closer to 0 than to -1 (norm_coeff={g_norm:.4e}); favor H_fold."
        else:
            verdict = "H_pred"
            reason = f"Closer to fold -1 than to 0 (norm_coeff={g_norm:.4e}); favor H_pred."

    results["verdict"] = {
        "label": verdict,
        "reason": reason,
        "gauge_counterexample_norm_coeff_N6_sin": g_norm,
        "gauge_counterexample_d2S_N6_sin": g["d2S_eps_half"],
        "gauge_counterexample_norm_coeff_N4_sin": g4["norm_coeff"],
        "gauge_N4_N6_rel_disc": disc,
        "decoy_norm_coeff_N6_sin": decoy["norm_coeff"],
        "decoy_d2S_N6_sin": decoy["d2S_eps_half"],
        "tt_norm_coeff_N6": tt["norm_coeff"],
        "tt_d2S_N6": tt["d2S_eps_half"],
        "tt_ratio_to_fold_m2_N6": tt["ratio_norm_to_fold_m2"],
        "tt_ratio_to_fold_m2_N4": results["N4"]["tt"]["axisTTPlus_symbolDir"][
            "ratio_norm_to_fold_m2"
        ],
        "orbit_S_counterexample_N6": g["orbit_S_at_eps"],
    }
    results["elapsed_sec"] = time.time() - t0

    out_json = "state/qg_full_theory/regge_4d_gauge_discrimination_20260721.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("WROTE", out_json)
    print("VERDICT", verdict)
    print(reason)
    return results


if __name__ == "__main__":
    main()
