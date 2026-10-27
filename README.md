# Computational supplements

Programs that accompany papers from the Recognition Physics Institute. Each paper has its own folder with the programs,
a README that says which section each program supports and how to run it, the stored output of a reference run, and
SHA-256 checksums. Run a folder's programs and compare your output with its `outputs/`.

| Paper | Folder |
|---|---|
| J. Washburn and P. Beltracchi, *Graviton Modes on the Kuhn Lattice: Euclidean Continuation and Lorentzian Propagation* (working draft, October 2026) | [`papers/2026-10-kuhn-graviton-modes`](papers/2026-10-kuhn-graviton-modes) |

Folders are named by date and a short title. A released folder is not edited in place: a correction is a new commit,
and each paper version is tagged, so a tag always reproduces what that version of the paper printed.

The programs need Python 3.10 or later; each folder's `requirements.txt` lists its packages. Licence: MIT (see `LICENSE`).
