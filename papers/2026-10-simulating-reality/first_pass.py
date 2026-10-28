"""Histories that finish their first complete pass of the cube at each tick, counted exactly.

Three accounts each carry one digit, the parity of its entries, so the record stands on a corner of a cube with eight
corners, 000 to 111. Every entry, a debit or a credit on one account, flips that account's digit: each step to a
neighbouring corner is taken by two entries. A history finishes its first pass at tick n when, after n entries, it has
stood on all eight corners and is back at 000 for the first time with all eight visited.

The program carries, tick by tick, how many histories stand at each state (the set of corners visited, the corner
now), moves each history to the three neighbouring corners (two entries each), and takes out those that finish.

    python3 first_pass.py                 print tick, count, share of the 6^n histories (exact fraction) for 8..24
    python3 first_pass.py FILE.csv        check FILE (columns: tick, count; lines starting # skipped)

Pure Python, exact integers, no dependencies.
"""
import csv
import sys
from fractions import Fraction

NMAX = 24
ALL = 0b11111111


def first_pass(nmax=NMAX):
    done = [0] * (nmax + 1)
    now = {(1 << 0, 0): 1}                    # (corners visited as a bit set, corner now): histories
    for n in range(1, nmax + 1):
        nxt = {}
        for (seen, at), c in now.items():
            for axis in range(3):
                to = at ^ (1 << axis)
                s2 = seen | (1 << to)
                if s2 == ALL and to == 0:
                    done[n] += 2 * c
                else:
                    nxt[(s2, to)] = nxt.get((s2, to), 0) + 2 * c
        now = nxt
    return done


def check(path):
    done = first_pass()
    with open(path, newline="", encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    reader = csv.reader(lines)
    next(reader)
    rows, bad = 0, []
    for rec in reader:
        try:
            n, c = int(rec[0]), int(rec[1])
        except (ValueError, IndexError):
            print(f"{path} is not a table of tick and count; use the page's stage 3 download 'When histories finish their first pass'")
            return False
        rows += 1
        if n > NMAX or c != done[n]:
            bad.append((n, c))
    print(f"{rows} rows: " + ("all agree" if not bad else f"{len(bad)} disagree, first {bad[0]}"))
    return not bad


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(0 if check(sys.argv[1]) else 1)
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["tick", "histories finishing their first pass", "share of the 6^n histories"])
    for n, c in enumerate(first_pass()):
        if n >= 8:
            w.writerow([n, c, Fraction(c, 6 ** n)])
