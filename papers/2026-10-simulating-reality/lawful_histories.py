"""Lawful histories reaching each balance of three accounts, counted exactly.

A history is a sequence of entries; each entry is a debit or a credit on one of three accounts, so n entries give 6^n
histories. A history ends at the balances (x, y, z): credits minus debits on each account. This program counts, for
every n and every (x, y, z), the histories ending there, by the closed form

    N(n; x, y, z) = sum over n1 + n2 + n3 = n of  n! / (n1! n2! n3!) * C(n1, (n1 + x)/2) * C(n2, (n2 + y)/2) * C(n3, (n3 + z)/2),

(choose which entries go to which account, then which of each account's entries are credits; a term is zero unless
n_i >= |x_i| and n_i - x_i is even), and checks the counts against a file of the same numbers.

    python3 lawful_histories.py                 print every count for n = 0..24 as CSV
    python3 lawful_histories.py --ticks         print, per tick, the histories (6^n), the distinct records (the ways to
                                                share n entries among six kinds, C(n + 5, 5)), the balances reached and
                                                the histories back in balance
    python3 lawful_histories.py FILE.csv        check FILE (columns: tick, x, y, z, count; lines starting # skipped)

Pure Python, exact integers, no dependencies.
"""
import csv
import sys
from math import comb, factorial

NMAX = 24


def count(n, x, y, z):
    """Histories of n entries ending at balances (x, y, z), by the closed form above."""
    total = 0
    for n1 in range(abs(x), n + 1, 1):
        if (n1 - x) % 2:
            continue
        for n2 in range(abs(y), n - n1 + 1):
            if (n2 - y) % 2:
                continue
            n3 = n - n1 - n2
            if n3 < abs(z) or (n3 - z) % 2:
                continue
            total += (factorial(n) // (factorial(n1) * factorial(n2) * factorial(n3))
                      * comb(n1, (n1 + x) // 2) * comb(n2, (n2 + y) // 2) * comb(n3, (n3 + z) // 2))
    return total


def balances(n):
    """Every (x, y, z) some history of n entries can reach: |x| + |y| + |z| <= n with the parity of n."""
    for x in range(-n, n + 1):
        for y in range(-(n - abs(x)), n - abs(x) + 1):
            r = n - abs(x) - abs(y)
            for z in range(-r, r + 1):
                if (abs(x) + abs(y) + abs(z) - n) % 2 == 0:
                    yield x, y, z


def table():
    for n in range(NMAX + 1):
        for x, y, z in balances(n):
            yield n, x, y, z, count(n, x, y, z)


def check(path):
    """Compare every row of a CSV (tick, x, y, z, count) with the closed form; also check each tick's rows add up to 6^n
    and cover every reachable balance exactly once."""
    rows, seen, bad = 0, {}, []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    reader = csv.reader(lines)
    next(reader, None)                             # the header row
    for rec in reader:
        try:
            n, x, y, z, c = (int(v) for v in rec[:5])
        except ValueError:
            print(f"{path} is not a table of tick, x, y, z and count; use the page's download 'Histories per square'")
            return False
        rows += 1
        if (x, y, z) in seen.get(n, {}):
            bad.append(("duplicate row", n, x, y, z))
        seen.setdefault(n, {})[(x, y, z)] = c
        if c != count(n, x, y, z):
            bad.append((n, x, y, z, c))
    if not seen:
        print(f"{path} has no data rows")
        return False
    for n, got in sorted(seen.items()):
        if set(got) != set(balances(n)):
            bad.append(("balances differ at tick", n))
        if sum(got.values()) != 6 ** n:
            bad.append(("total differs at tick", n))
    print(f"{rows} rows over ticks {min(seen)} to {max(seen)}: " + ("all agree with the closed form" if not bad else f"{len(bad)} disagree, first {bad[0]}"))
    return not bad


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ticks":
        w = csv.writer(sys.stdout, lineterminator="\n")
        w.writerow(["tick", "histories", "distinct records", "balances reached", "back in balance"])
        for n in range(1, NMAX + 1):
            w.writerow([n, 6 ** n, comb(n + 5, 5), sum(1 for _ in balances(n)), count(n, 0, 0, 0)])
        sys.exit(0)
    if len(sys.argv) > 1:
        sys.exit(0 if check(sys.argv[1]) else 1)
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["tick", "x", "y", "z", "histories"])
    for row in table():
        w.writerow(row)
