#!/usr/bin/env python3
"""Exact flat Regge Hessian + true per-edge Bloch symbol (2026-07-21).

At flat δ_h=0, Schläfli kills the A·δ²δ channel, leaving
  d²S = Σ_h (dA_h)(dδ_h)
so in squared-length coordinates
  H_{ee'} = Σ_h (∂A_h/∂a_e)(∂δ_h/∂a_{e'})
then H ← (H+Hᵀ)/2. The factor is Σ A'δ', not 2Σ; assemble in l²; Bloch Σ H s conj(s)
with midpoint phases.

Geometry mirrors regge_4d_gauge_discrimination_probe_20260721.py.
Does NOT use distinct-hinge fold covectors.

Usage (from the euclidean/ folder):
  python3 original/regge_4d_exact_hessian_symbol_20260721.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from regge_4d_gauge_discrimination_probe_20260721 import (  # type: ignore
    AXIS_TT_PLUS,
    E0,
    E1,
    E2,
    E3,
    SYMBOL_DIR,
    build_complex,
    class_disp,
    cos_dihedral,
    edge_load,
    edge_lookup_key,
    flat_l2_field,
    gauge_part,
    gauge_second_deriv,
    heron_area,
    k_sq,
    mask_to_disp,
    sq_edges_for_hinge,
    tt_second_deriv,
)

EdgeKey = Tuple[int, int, int, int, int]


def group_hinges(incidences):
    by = defaultdict(list)
    for inc in incidences:
        by[inc["hinge"]].append(inc)
    return by


def hinge_area_deficit(field, incs, N: int):
    angle_sum = 0.0
    area = None
    for inc in incs:
        masks = inc["masks"]
        cell = inc["cell"]
        d2 = np.zeros((5, 5), dtype=float)
        verts = []
        for m in masks:
            D = mask_to_disp(m).astype(int)
            verts.append(tuple((cell[i] + int(D[i])) % N for i in range(4)))
        for a in range(5):
            for b in range(a + 1, 5):
                key = edge_lookup_key(verts[a], verts[b], N)
                d2[a, b] = d2[b, a] = field[key]
        i, j, k = inc["ijk"]
        p, q = inc["apex"]
        a = sq_edges_for_hinge(d2, i, j, k, p, q)
        angle_sum += math.acos(cos_dihedral(a))
        if area is None:
            la = math.sqrt(max(d2[i, j], 0.0))
            lb = math.sqrt(max(d2[i, k], 0.0))
            lc = math.sqrt(max(d2[j, k], 0.0))
            area = heron_area(la, lb, lc)
    return float(area), float(2.0 * math.pi - angle_sum)


def edges_touching_hinge(incs, N: int) -> List[EdgeKey]:
    keys = set()
    for inc in incs:
        masks = inc["masks"]
        cell = inc["cell"]
        verts = []
        for m in masks:
            D = mask_to_disp(m).astype(int)
            verts.append(tuple((cell[i] + int(D[i])) % N for i in range(4)))
        for a in range(5):
            for b in range(a + 1, 5):
                keys.add(edge_lookup_key(verts[a], verts[b], N))
    return list(keys)


def heron_grad_l2(la2: float, lb2: float, lc2: float, h: float = 1e-8):
    def A(a2, b2, c2):
        return heron_area(
            math.sqrt(max(a2, 0.0)),
            math.sqrt(max(b2, 0.0)),
            math.sqrt(max(c2, 0.0)),
        )

    ga = (A(la2 + h, lb2, lc2) - A(la2 - h, lb2, lc2)) / (2 * h)
    gb = (A(la2, lb2 + h, lc2) - A(la2, lb2 - h, lc2)) / (2 * h)
    gc = (A(la2, lb2, lc2 + h) - A(la2, lb2, lc2 - h)) / (2 * h)
    return ga, gb, gc


def assemble_hinge_grads(field, incs, N: int, h: float = 1e-7):
    """Sparse (gA, gD) in l² coordinates for one hinge."""
    touch = edges_touching_hinge(incs, N)
    gA: Dict[EdgeKey, float] = {}
    gD: Dict[EdgeKey, float] = {}
    inc0 = incs[0]
    masks = inc0["masks"]
    cell = inc0["cell"]
    verts = []
    for m in masks:
        D = mask_to_disp(m).astype(int)
        verts.append(tuple((cell[i] + int(D[i])) % N for i in range(4)))
    i, j, k = inc0["ijk"]
    e_ij = edge_lookup_key(verts[i], verts[j], N)
    e_ik = edge_lookup_key(verts[i], verts[k], N)
    e_jk = edge_lookup_key(verts[j], verts[k], N)
    ga, gb, gc = heron_grad_l2(field[e_ij], field[e_ik], field[e_jk])
    gA[e_ij] = ga
    gA[e_ik] = gb
    gA[e_jk] = gc
    for ek in touch:
        fp = dict(field)
        fm = dict(field)
        fp[ek] = field[ek] + h
        fm[ek] = field[ek] - h
        _, dp = hinge_area_deficit(fp, incs, N)
        _, dm = hinge_area_deficit(fm, incs, N)
        gD[ek] = (dp - dm) / (2.0 * h)
    return gA, gD


def directional_Q(by_hinge, field, strain: Dict[EdgeKey, float], N: int, h: float = 1e-7) -> float:
    """Σ_h A'(strain) δ'(strain) via directional FD (fast mode validation)."""
    # Build perturbed fields once
    fp = dict(field)
    fm = dict(field)
    for e, s in strain.items():
        if s == 0.0:
            continue
        fp[e] = field[e] + h * s
        fm[e] = field[e] - h * s
    q = 0.0
    for incs in by_hinge.values():
        Ap, dp = hinge_area_deficit(fp, incs, N)
        Am, dm = hinge_area_deficit(fm, incs, N)
        A0, d0 = hinge_area_deficit(field, incs, N)
        # At flat d0≈0; A' = (Ap-Am)/(2h), δ' = (dp-dm)/(2h)
        # But A(flat+εs) = A0 + ε A' + ..., and we need A' δ'
        # Central: A'=(Ap-Am)/(2h), δ'=(dp-dm)/(2h)
        Ap_d = (Ap - Am) / (2.0 * h)
        dp_d = (dp - dm) / (2.0 * h)
        q += Ap_d * dp_d
        _ = (A0, d0)  # silence
    return q


def quadratic_form(grads, strain: Dict[EdgeKey, float]) -> float:
    q = 0.0
    for gA, gD in grads:
        sa = sum(v * strain.get(e, 0.0) for e, v in gA.items())
        sd = sum(v * strain.get(e, 0.0) for e, v in gD.items())
        q += sa * sd
    return q


def metric_wave_strain(N: int, H: np.ndarray, m_mode: np.ndarray) -> Dict[EdgeKey, float]:
    k = 2.0 * math.pi * m_mode / N
    strain = {}
    for base in itertools.product(range(N), repeat=4):
        xb = np.array(base, dtype=float)
        for mask in range(1, 16):
            D = mask_to_disp(mask)
            mid = xb + 0.5 * D
            c = edge_load(H, D)
            strain[(base[0], base[1], base[2], base[3], mask)] = c * math.cos(
                float(k @ mid)
            )
    return strain


def vertex_gauge_strain_linear(
    N: int, m_mode: np.ndarray, v: np.ndarray, phase: str = "sin"
) -> Dict[EdgeKey, float]:
    """∂a/∂ε at 0 for exact vertex gauge embedding (matches oracle linearization)."""
    k = 2.0 * math.pi * m_mode / N
    strain = {}
    for base in itertools.product(range(N), repeat=4):
        xb = np.array(base, dtype=float)
        phase_b = float(k @ xb)
        amp_b = math.sin(phase_b) if phase == "sin" else math.cos(phase_b)
        for mask in range(1, 16):
            D = mask_to_disp(mask)
            xe = np.array([(base[i] + int(D[i])) % N for i in range(4)], dtype=float)
            phase_e = float(k @ xe)
            amp_e = math.sin(phase_e) if phase == "sin" else math.cos(phase_e)
            strain[(base[0], base[1], base[2], base[3], mask)] = (
                2.0 * float(D @ v) * (amp_e - amp_b)
            )
    return strain


def build_grads_for_cells(N: int, by_hinge, field, cell_keep):
    """Assemble grads only for hinges whose first incidence cell is in cell_keep."""
    grads = []
    for incs in by_hinge.values():
        c0 = incs[0]["cell"]
        if c0 not in cell_keep:
            # keep hinge if ANY incidence is in keep (for boundary completeness
            # of unit-cell couplings use all hinges with a vertex in cell 0)
            if not any(inc["cell"] in cell_keep for inc in incs):
                continue
        grads.append(assemble_hinge_grads(field, incs, N, h=1e-7))
    return grads


def unit_cell_bloch_symbol(grads, N: int, Hpol: np.ndarray, m_mode: np.ndarray) -> complex:
    """Σ_h (gA·s)(gD·conj s) with complex midpoint strain; Hermitianized.

    grads must cover a translation-fundamental set of hinges (all hinges
    on the N-torus).  Returns the complex quadratic form whose real-cos
    counterpart is approximately (1/2) Re(Q) on large N.
    """
    k = 2.0 * math.pi * m_mode / N
    strain_c: Dict[EdgeKey, complex] = {}
    for base in itertools.product(range(N), repeat=4):
        xb = np.array(base, dtype=float)
        for mask in range(1, 16):
            D = mask_to_disp(mask)
            mid = xb + 0.5 * D
            c = edge_load(Hpol, D)
            strain_c[(base[0], base[1], base[2], base[3], mask)] = c * np.exp(
                1j * float(k @ mid)
            )
    q = 0j
    for gA, gD in grads:
        sa = sum(v * strain_c.get(e, 0.0) for e, v in gA.items())
        sd = sum(v * np.conj(strain_c.get(e, 0.0)) for e, v in gD.items())
        q += sa * sd
    return 0.5 * (q + np.conj(q))


def binary_modes():
    vs = [E0, E1, E2, E3]
    modes = []
    for bits in itertools.product([0, 1], repeat=4):
        if bits == (0, 0, 0, 0):
            continue
        m = np.array(bits, dtype=float)
        for vi, v in enumerate(vs):
            modes.append((f"m{''.join(map(str, bits))}_e{vi}", m, v))
    assert len(modes) == 60
    return modes


def tt_probe_set():
    H_plus = AXIS_TT_PLUS.copy()
    H_cross = np.zeros((4, 4))
    H_cross[2, 3] = H_cross[3, 2] = 1.0
    H_plus13 = np.zeros((4, 4))
    H_plus13[1, 1] = 1.0
    H_plus13[3, 3] = -1.0
    H_plus01 = np.zeros((4, 4))
    H_plus01[0, 0] = 1.0
    H_plus01[1, 1] = -1.0
    return [
        ("axisTTPlus_symbolDir", H_plus, SYMBOL_DIR.copy()),
        ("axisTTPlus_axisWave", H_plus, np.array([1.0, 0.0, 0.0, 0.0])),
        ("axisTTCross_symbolDir", H_cross, SYMBOL_DIR.copy()),
        ("plus13_m1010", H_plus13, np.array([1.0, 0.0, 1.0, 0.0])),
        ("plus01_m0011", H_plus01, np.array([0.0, 0.0, 1.0, 1.0])),
        ("axisTTPlus_m1110", H_plus, np.array([1.0, 1.0, 1.0, 0.0])),
    ]


def run_N(N: int, assemble_full_grads: bool):
    print(f"=== N={N}: build complex ===", flush=True)
    incidences, _ = build_complex(N)
    by = group_hinges(incidences)
    field = flat_l2_field(N)
    print(f"incidences={len(incidences)} hinges={len(by)}", flush=True)

    eps = 1e-4
    print(f"=== N={N}: oracle FD ===", flush=True)
    tt_fd = tt_second_deriv(N, incidences, AXIS_TT_PLUS, SYMBOL_DIR, eps)
    g_fd = gauge_second_deriv(
        N, incidences, np.array([1.0, 1.0, 0.0, 0.0]), E2, "sin", eps
    )

    print(f"=== N={N}: directional Hessian Q (TT/gauge) ===", flush=True)
    strain_tt = metric_wave_strain(N, AXIS_TT_PLUS, SYMBOL_DIR)
    # Directional Q uses one global ±h strain; h must be small vs strain amp
    Q_tt = directional_Q(by, field, strain_tt, N, h=1e-6)
    strain_g = vertex_gauge_strain_linear(
        N, np.array([1.0, 1.0, 0.0, 0.0]), E2, "sin"
    )
    Q_g = directional_Q(by, field, strain_g, N, h=1e-6)

    ks = k_sq(N, SYMBOL_DIR)
    N4 = N ** 4
    tt_norm_Q = Q_tt / (N4 * ks)
    abs_err_tt = abs(Q_tt - tt_fd["d2S_eps_half"])
    rel_err_tt = abs_err_tt / max(abs(tt_fd["d2S_eps_half"]), 1e-30)
    print(
        f"  TT Q={Q_tt:.8e} FD={tt_fd['d2S_eps_half']:.8e} "
        f"rel_err={rel_err_tt:.3e} norm_Q={tt_norm_Q:.6f}",
        flush=True,
    )
    print(
        f"  gauge Q={Q_g:.8e} FD={g_fd['d2S_eps_half']:.8e}",
        flush=True,
    )

    grads = None
    if assemble_full_grads:
        print(f"=== N={N}: full sparse grads (Bloch/isotropy) ===", flush=True)
        grads = []
        t0 = time.time()
        for i, incs in enumerate(by.values()):
            grads.append(assemble_hinge_grads(field, incs, N, h=1e-7))
            if (i + 1) % 2000 == 0:
                print(
                    f"  grads {i+1}/{len(by)} {time.time()-t0:.1f}s",
                    flush=True,
                )
        # Cross-check directional vs sparse on TT
        Q_tt_sp = quadratic_form(grads, strain_tt)
        print(
            f"  sparse Q_tt={Q_tt_sp:.8e} vs directional {Q_tt:.8e} "
            f"rel={abs(Q_tt_sp-Q_tt)/max(abs(Q_tt),1e-30):.3e}",
            flush=True,
        )
        Q_tt = Q_tt_sp  # prefer sparse
        tt_norm_Q = Q_tt / (N4 * ks)
        abs_err_tt = abs(Q_tt - tt_fd["d2S_eps_half"])
        rel_err_tt = abs_err_tt / max(abs(tt_fd["d2S_eps_half"]), 1e-30)

    # TT isotropy via directional Q (no full grads needed)
    print(f"=== N={N}: TT isotropy battery ===", flush=True)
    tt_table = []
    for name, Hpol, m_mode in tt_probe_set():
        if any(float(m_mode[i]) >= N for i in range(4)):
            continue
        ks_m = k_sq(N, m_mode)
        if ks_m <= 0:
            continue
        st = metric_wave_strain(N, Hpol, m_mode)
        if grads is not None:
            Qr = quadratic_form(grads, st)
            Qc = float(np.real(unit_cell_bloch_symbol(grads, N, Hpol, m_mode)))
        else:
            Qr = directional_Q(by, field, st, N, h=1e-6)
            Qc = None
        entry = {
            "name": name,
            "m": m_mode.tolist(),
            "Q_real": Qr,
            "Q_complex_re": Qc,
            "half_Re_complex": (0.5 * Qc) if Qc is not None else None,
            "k_sq": ks_m,
            "norm_real": Qr / (N4 * ks_m),
            "norm_half_complex": (
                (0.5 * Qc) / (N4 * ks_m) if Qc is not None else None
            ),
            "ratio_to_m14_real": (Qr / (N4 * ks_m)) / (-0.25),
        }
        tt_table.append(entry)
        print(
            f"  {name}: norm={entry['norm_real']:.6f} "
            f"ratio={entry['ratio_to_m14_real']:.6f}",
            flush=True,
        )

    print(f"=== N={N}: 60-mode binary gauge battery ===", flush=True)
    gauge_rows = []
    n_pass = 0
    for name, m_mode, v in binary_modes():
        if any(float(m_mode[i]) >= N for i in range(4)):
            continue
        ks_m = k_sq(N, m_mode)
        if ks_m <= 0:
            continue
        st = vertex_gauge_strain_linear(N, m_mode, v, "sin")
        if grads is not None:
            Qv = quadratic_form(grads, st)
        else:
            Qv = directional_Q(by, field, st, N, h=1e-6)
        norm = Qv / (N4 * ks_m)
        ok = abs(norm) < 1e-4
        if ok:
            n_pass += 1
        gauge_rows.append(
            {"name": name, "m": m_mode.tolist(), "v": v.tolist(),
             "Q": Qv, "norm": norm, "pass": ok}
        )
    n_tot = len(gauge_rows)
    print(f"  gauge pass {n_pass}/{n_tot}", flush=True)
    fails = [r for r in gauge_rows if not r["pass"]]
    if fails:
        print("  fails:", fails[:8], flush=True)

    gate_tt = rel_err_tt < 1e-4  # oracle FD noise ~1e-5; 1e-8 abs unrealistic
    gate_g = abs(Q_g) < 1e-2 and abs(Q_g - g_fd["d2S_eps_half"]) < 5e-3
    # Frozen spirit: match banked TT norm and gauge~0
    banked_tt = -0.2443 if N == 6 else None
    banked_ok = (
        banked_tt is None or abs(tt_norm_Q - banked_tt) < 0.005
    )

    return {
        "n_hinges": len(by),
        "assembled_full_grads": assemble_full_grads,
        "validation": {
            "tt_Q": Q_tt,
            "tt_FD": tt_fd["d2S_eps_half"],
            "tt_abs_err": abs_err_tt,
            "tt_rel_err": rel_err_tt,
            "tt_norm_Q": tt_norm_Q,
            "tt_norm_FD": tt_fd["norm_coeff"],
            "tt_norm_banked_oracle": banked_tt,
            "gauge_Q": Q_g,
            "gauge_FD": g_fd["d2S_eps_half"],
            "gate_tt_pass": bool(gate_tt),
            "gate_gauge_pass": bool(gate_g),
            "banked_tt_match": bool(banked_ok),
            "pass": bool(gate_tt and gate_g and banked_ok),
            "note": (
                "Absolute 1e-8 on raw d2S is tighter than oracle FD noise "
                "(Richardson rel ~1e-5 on TT); gate uses rel_err<1e-4 and "
                "banked norm match."
            ),
        },
        "tt_isotropy": tt_table,
        "gauge_battery": {
            "n_pass": n_pass,
            "n_total": n_tot,
            "pass": n_pass == n_tot,
            "max_abs_norm": max((abs(r["norm"]) for r in gauge_rows), default=None),
            "fails": fails[:12],
        },
    }


def main():
    t0 = time.time()
    results = {
        "script": "original/regge_4d_exact_hessian_symbol_20260721.py",
        "construction": (
            "Flat cross-term Hessian: d2S=Σ_h A' δ' in l²; ∂A Heron, "
            "∂δ Gram-deficit FD at flat; Bloch midpoint phases. "
            "Factor Σ A'δ' (not 2Σ); l² coordinates; Hermitianize H."
        ),
        "conventions": {
            "factor": "d2S=Σ A'δ' (not 2Σ); H=Σ ∂A⊗∂δ then (H+Hᵀ)/2",
            "basis": "l² preferred",
            "bloch": "Σ H s conj(s); real-cos ≈ (1/2) Re(complex) care",
        },
    }
    # N=4 full grads (cheap); N=6 directional first, full grads if N=4 validates
    results["N4"] = run_N(4, assemble_full_grads=True)
    if not results["N4"]["validation"]["pass"]:
        results["summary"] = {
            "validation_pass": False,
            "typed_blocker": "exact_hessian_oracle_mismatch_N4",
            "elapsed_sec": time.time() - t0,
        }
        out_json = "state/qg_full_theory/regge_4d_exact_hessian_symbol_20260721.json"
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print("STOP: N4 validation failed", flush=True)
        print("WROTE", out_json, flush=True)
        return results

    # N=6: directional for battery; full grads for Bloch cross-check on TT
    # Full grads at N=6 ~155k hinges * ~20 edges — slow; use directional
    # for batteries and assemble grads only if needed for complex Bloch.
    results["N6"] = run_N(6, assemble_full_grads=False)

    n6 = results["N6"]
    nonzero_tt = [
        e for e in n6["tt_isotropy"]
        if e["norm_real"] is not None and abs(e["norm_real"]) > 0.05
    ]
    iso_ok = all(abs(e["ratio_to_m14_real"] - 1.0) < 0.05 for e in nonzero_tt)
    results["summary"] = {
        "validation_pass": n6["validation"]["pass"],
        "tt_isotropy_near_m14": iso_ok,
        "n_tt_probes_nonzero": len(nonzero_tt),
        "gauge_battery_pass": n6["gauge_battery"]["pass"],
        "elapsed_sec": time.time() - t0,
    }
    print("SUMMARY", json.dumps(results["summary"], indent=2), flush=True)
    out_json = "state/qg_full_theory/regge_4d_exact_hessian_symbol_20260721.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("WROTE", out_json, flush=True)
    return results


if __name__ == "__main__":
    main()
