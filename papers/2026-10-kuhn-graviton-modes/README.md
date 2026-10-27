# Computational supplement

**Graviton Modes on the Kuhn Lattice: Euclidean Continuation and Lorentzian Propagation**
Jonathan Washburn and Philip Beltracchi, Recognition Physics Institute, Austin, Texas

This folder holds every program behind the paper's calculations, with the
output each one produced. The `lorentzian/` programs rebuild the direct
Lorentzian result of Section 6 from the geometry of the Kuhn simplices: areas
and angles on the background g = I − J, the Hessian cell, the 15 × 15 symbol,
the two graviton branches along the paired direction and their phase-speed
coefficients +5/24 and −59/24. They also recompute the numbers of Sections 7
to 9. The `euclidean/` folder is the published package for the Euclidean
calculation of Sections 2 to 5.

Start with `lorentzian/READERS_GUIDE.md` (typeset copy:
`lorentzian/READERS_GUIDE.pdf`). It explains what each step of Section 6
computes and why, and names the function that does it.

## Layout

```
README.md            this file
requirements.txt     Python dependencies
CITATION.cff         how to cite the paper and this supplement
LICENSE              licence (placeholder; see below)
SHA256SUMS.txt       hashes of every file in this folder
lorentzian/          programs for Section 6, Appendix A and Sections 7-9, and the reader's guide
outputs/             the full output of one run of each lorentzian/ program
euclidean/           the Euclidean package, release 2026.09.16.1
```

## Running

Python 3.10 or newer.

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python lorentzian/run_all.py
```

`run_all.py` runs the four programs below once each. For every program it
writes `outputs/<name>.txt` (everything printed) and `outputs/<name>.json`
(each check with PASS or FAIL and the values reported), and then
`outputs/RUN_RECORD.json` with runtimes, library versions and the SHA-256 hash
of every program and output. Each program also runs on its own:
`python lorentzian/section6_paired_direction.py`.

## What each program checks

| Program | Paper | What it computes | Checks | Time |
|---|---|---|---|---|
| `appendixA_cell.py` | Sec. 6.1, App. A, App. B | squared intervals n(1 − n); timelike triangles and spacelike normal planes; angles π/6 and 2π/3; 2π around all 50 hinges; the Hessian cell C = 2R/√3 against the printed 32R, all 100 entries; the assembly rule and its cosine form; the gauge identity M_L g_ξ = 0; the same code on the Euclidean lattice against the published 302-coupling table | 28 | 12 s |
| `section6_paired_direction.py` | Sec. 6, App. A | clock τ and direction σ; the cross column and its exact eigenvalue s_×; the characteristic equation and G_×; the orbit block B, all 36 entries; det B and the polynomials P and G_+; the coefficients 5/24 and −59/24 and the splitting 8/3; reconstruction of a full zero mode from a zero of B; 50-digit branch, kernel and gauge-independence checks; the Euclidean coefficients −(1 + A)/24 and +1/24 | 43 | 17 s |
| `section7_source_constraints.py` | Sec. 7 | the scalar simplex metric; the edge source of φ = λ(x₁ + x₂) on the hypercube diagonal, −κ_m λ²/12; the vertex-displacement contraction gᵀJ = −18 κ_m λ² for φ = λ(x₁³ − 3x₁x₂²), computed two independent ways; both vectors are null vectors of the Euclidean Hessian | 18 | 50 s |
| `sections8_9_bound_numbers.py` | Secs. 8, 9 | every printed number of the conditional Cherenkov bound, Table 5 and the candidate spacing; the photon energy balances, equations (119) and (120), and the range of y_* | 66 | 1 s |

The two library modules, `kuhn_lorentz_geometry.py` (one simplex: Gram
matrices, areas, angles, their derivatives, the Hessian cell) and
`kuhn_lorentz_symbol.py` (assembly into the midpoint symbol, exact
trigonometric polynomials, gauge vectors), are imported by the programs.

Exact results use SymPy rationals and square roots. The finite-momentum
checks in Section 6 use mpmath at 50 digits. Sections 8 and 9 use mpmath at
30 digits with the constants of Appendix B.

## Expected output

One run of `run_all.py` with Python 3.10.12, SymPy 1.12 and mpmath 1.2.1 on
Linux produced the files in `outputs/`: 155 checks, 154 passing.

The one failing check is a printed value, not the calculation. In Table 5 the
iron row at 100 Mpc is printed as 4.1 × 10³. With the paper's own input
E_f = E/560 = 5.714 × 10¹⁷ eV the threshold is 4036 l_P, which rounds to
4.0 × 10³. Rounding E_f to 5.7 × 10¹⁷ eV before using it gives 4054, which
rounds to 4.1 × 10³. Every other number in Sections 8 and 9 matches the
unrounded input and not the rounded one. `sections8_9_bound_numbers.txt`
prints both values.

Your own run should reproduce the `.txt` and `.json` files exactly, apart
from timings in `RUN_RECORD.json`.

## Scope

The Lorentzian programs work on the paper's stated model: the fixed Kuhn
triangulation, the background g = I − J (the κ = 1 member of the family
n(1 − κn)), and the paired direction q = (z+t, z+t, z−t, z−t). The branch
statements hold for sufficiently small positive K, near the light cone, with
sin(x/2) and sin(y/2) nonzero; finite periodic lattices need at least three
cells in each direction. The programs do not compute the Lorentzian
characteristic set in a general direction, which remains open. Sections 8 and
9 use the Euclidean axis-time branch and the imported emission rate of
Kostelecký and Tasson, with their matter assumptions.

## The Euclidean package

`euclidean/` is release 2026.09.16.1 of the Euclidean software as distributed from the
paper's download page, with comments that referred to internal tools removed (one
output key is renamed `rational_table`); every computation and printed number is unchanged. Its own `README.txt` explains
each program; `euclidean/SHA256SUMS.txt` lists its hashes. It needs NumPy in
addition to SymPy and mpmath. Its full numerical program,
`original/regge_4d_quartic_tt_dispersion_20260903.py`, is considerably slower
than the exact checks.

Appendix B also names `VerifyCurrentDraft.py` and `VerifyScalarCoupling.py`.
They are not part of release 2026.09.16.1 and are not included here.

## Licence and citation

`LICENSE` is a placeholder; the proposed licence is MIT. Cite the paper as
given in `CITATION.cff`.

Questions: jon@recognitionphysics.org
