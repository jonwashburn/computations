LATTICE GRAVITON DISPERSION: SCIENTIFIC SOFTWARE
Release 2026.09.16.1 | Recognition Physics Institute
https://recognitionphysics.org/qg/

This package accompanies Lattice Graviton Dispersion on the Kuhn
Triangulation, Jonathan Washburn and Philip Beltracchi, September 2026.
It corresponds to the September 15 proposed collaborator revision.

QUICK START (Python 3.10 or newer)

macOS / Linux:

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python run_checks.py

Windows (PowerShell or Command Prompt):

  py -3 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe run_checks.py

No environment activation or PowerShell execution-policy change is needed.
For the individual commands below, replace python with
.\.venv\Scripts\python.exe on Windows. Use the same interpreter to install
dependencies and run the programs. If py is unavailable, use an installed
Python 3.10+ interpreter's full path to create the environment. Recreate
the environment after reinstalling Python; do not reuse a broken old one.

Run these commands from the extracted package directory. The exact checks
take seconds to minutes, depending on the machine. Reference native runs
used Python 3.10, SymPy 1.12, mpmath 1.2.1 and NumPy 1.21.5. The ranges in
requirements.txt permit installation on newer Python versions. The website
uses pinned Pyodide 0.28.3 and reports its actual bundled package versions.

INDIVIDUAL CHECKS

1. Exact quadratic operator and zero-momentum kernel
   python original/regge_4d_exact_quadratic_identity_20260904.py
   First run run_checks.py to stage the coupling table under state/.
   Output: state/qg_full_theory/regge_4d_exact_quadratic_identity_20260904.json

2. Finite-momentum action factorization, constraints and gauge map
   python VerifyPhilipDraft.py .
   Output: PhilipExactChecks.json
   The boolean printed_F_beta_formula_is_correct is EXPECTED FALSE.
   This deliberately rejects an old missing-factor-two formula. The
   corrected_F_beta_formula_is_correct boolean must be TRUE.

3. Full off-shell transverse-traceless quartic eigenvalues
   python VerifyQuartic.py reference/regge_4d_kuhn_coupling_table_20260903.json
   Output: QuarticSpectrum.json
   Axis: 11/384 (multiplicity 2), 31/768 (1), 1/192 (2).
   Face: 1/24 (2), 1/192 (1), 13/384 (1), 1/96 (1).

LIVE EXPLORER

  python explorer.py --azimuth 45 --elevation 0 --k 1.2

explorer.py is a lightweight companion evaluating the paper's closed-form
axis-time branch. It is not the full geometric reconstruction. In units
a=c=1, omega = 2 asinh sqrt(sum_j sin(k*n_j/2)^2). The group-speed output
is the magnitude of the gradient, which need not be parallel to momentum.
The zero-momentum values of both speed readouts are their limiting value 1.
Angles range from 0 to 90 degrees; k is clipped to the first zone.

lattice_modes.py constructs two finite-momentum modes with the paper's
lattice constraint and exact edge map. Every explorer update checks all
15 edge equations. The page displays normalized edge strains on fixed
reference geometry. Its colors are gauge dependent; their scale does not
specify a physical wave amplitude. At zero momentum the pattern is a
static uniform-strain limit. See MODE_CONSTRUCTION.txt for the derivation.

An independent check using the midpoint symbol and a gauge-rank test:
  python verify_modes.py
Reference: reference/ModeChecks.json (156 settings, 312 mode checks).

FULL GEOMETRY AND NUMERICAL PROGRAM

  python -X utf8 original/regge_4d_quartic_tt_dispersion_20260903.py

Run from this directory. This reconstructs the geometric Hessian, performs
high-precision root searches and finite differences on a 4^4 periodic
lattice, and writes its results to state/qg_full_theory/. It is considerably
more expensive than the exact checks. All three imported geometry modules
are included beside the original program. No private repository is needed.
Keep -X utf8 when running the original program, especially when redirecting
its output on Windows: its progress messages contain Greek symbols.

For an independent raw-assembly comparison against the reference table:
  python audit_geometry.py
Output: results/geometry_audit.json

VERSION CORRECTIONS

Release 2026.09.16.1 fixes the Windows failure in audit_geometry.py: host
metadata now uses platform.node() instead of the Unix-only os.uname().
The audit also escapes unsupported progress-log characters when Windows
uses a legacy output encoding. Git checkouts preserve the release's line
endings so the exact coupling-table checksum remains valid on Windows.
The geometry calculation and reference results are unchanged.

The original programs in original/ keep every computation unchanged; comments that
referred to internal tools were removed.
Some historical quartic entries are diagonal Rayleigh values, not the
full five-dimensional off-shell spectrum. Use VerifyQuartic.py for the
corrected spectrum, including off-diagonal terms. This does not change
the equal-polarization on-shell branch shown by the explorer.

The original numerical program's optional entropy anchor uses sqrt(pi).
The corrected anchor is sqrt(4*log(2)). VerifyPhilipDraft.py contains
the updated substitutions. Physical bounds in that script retain the
paper's conventional-matter and minimal-coupling assumptions.

REFERENCE DATA AND PROVENANCE

reference/ contains the 302-entry exact rational coupling table, the
September 15 exact outputs and September 16 mode checks. These are stored reference results, not
fresh results from your own machine. Browser results are separately labeled
and downloadable with runtime, timestamp and SHA-256 receipts.

Coupling table SHA-256:
c2faba52960f1cf4de7b8bb4c2e0da599b2335dc6b7f76743ed90a95157e1b31

SHA256SUMS.txt covers the package files. manifest.json names their roles.
The browser checks run the same unchanged scripts that you can download.
The browser needs access to cdn.jsdelivr.net to load Python and its packages.

SCIENTIFIC SCOPE

Free waves of the linearized four-dimensional Kuhn Regge model, weak-field
in amplitude and exact in lattice momentum, with an axis-time analytic
continuation of the Euclidean symbol. The edge modes are constructed from
the exact factorization and evaluated in floating-point arithmetic. The
visualization shows normalized edge strain in a specified gauge, not a
literal embedding of curved spacetime, nonlinear evolution or quantum
fluctuations. c denotes the long-wavelength model speed; a comparison with
photons needs a specified photon law. The free branch alone does not
establish a physical interacting matter theory.

Questions: jon@recognitionphysics.org
