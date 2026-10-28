# Computational supplement

**Simulating Reality**
Recognition Physics Institute, Austin, Texas. Web page: <https://recognitionphysics.org/simulation/>

The page computes its first figures in your browser, and each figure has a button that saves the numbers it computed
as a CSV file. The two programs here compute the same numbers again, in plain Python with exact integers and
independently of the page's code, and check a saved file row by row.

## Layout

```
README.md             this file
requirements.txt      no packages: the standard library is enough
CITATION.cff          how to cite the page and this supplement
SHA256SUMS.txt        hashes of every file in this folder
lawful_histories.py   the hero's histories per balance, and stage 2's table
first_pass.py         stage 3's first complete passes
outputs/              one run of each program, and the checks of the files in page_files/
page_files/           three files the page saved on 28 October 2026 at 01:00 UTC
```

## Running

Python 3.10 or newer. Nothing to install.

```
python3 lawful_histories.py              > lawful_histories.csv
python3 lawful_histories.py --ticks      > lawful_histories_ticks.csv
python3 first_pass.py                    > first_pass.csv
python3 lawful_histories.py page_files/simulation-hero-cloud-28-Oct-2026-01-00-UTC.csv
python3 first_pass.py page_files/simulation-s3-chart-28-Oct-2026-01-00-UTC.csv
```

Each command finishes in about a second. To check the page yourself, save a figure's numbers with its download button
and give the file to the program in place of the files in `page_files/`. A check exits with status 0 when every row
agrees.

## What each program computes

A history is a sequence of entries. Each entry is a debit or a credit on one of three accounts, so n entries give 6ⁿ
histories. A history ends at the balances (x, y, z): credits minus debits on each account.

| Program | On the page | What it computes | What its check does |
|---|---|---|---|
| `lawful_histories.py` | The hero, "Histories per square"; stage 2, "Every lawful history, carried at once" | For every n from 0 to 24 and every reachable (x, y, z), the histories ending there, by the closed form N(n; x, y, z) = Σ over n₁ + n₂ + n₃ = n of n!/(n₁! n₂! n₃!) · C(n₁, (n₁ + x)/2) · C(n₂, (n₂ + y)/2) · C(n₃, (n₃ + z)/2). With `--ticks`, per tick: the 6ⁿ histories, the C(n + 5, 5) distinct records, the balances reached and the histories back in balance | Compares every row of the hero's file with the closed form, and checks that each tick's rows add up to 6ⁿ and cover every reachable balance exactly once |
| `first_pass.py` | Stage 3, "When histories finish their first pass" | Each account's digit is the parity of its entries, so the record stands on one of the eight corners of a cube and every entry moves it to a neighbouring corner. For every tick from 8 to 24: the histories that have stood on all eight corners and are back at 000 for the first time, with their share of the 6ⁿ histories as an exact fraction | Compares every row of stage 3's file with the program's counts |

The page's stage 2 table shows ticks 1 to 4, 8, 12, 16 and 24; `lawful_histories.py --ticks` prints every tick from
1 to 24. Stage 3's other download, "When one recognizer completes its first pass", holds a different quantity, given
as bounds to tick 40; these programs do not compute it.

## Expected output

One run with Python 3.14.6 produced the files in `outputs/`. `outputs/checks.txt` records both checks of the files in
`page_files/`: all 70,525 rows of the hero's file and all 24 rows of stage 3's file agree. The hero's data rows are
also identical, character for character, to `outputs/lawful_histories.csv`, and every row of the stage 2 table
matches the same tick in `outputs/lawful_histories_ticks.csv`. Your run should reproduce `outputs/` exactly.

## Scope

The programs count the histories of the stated model exactly, with no floating point. They check the page's arithmetic
for these three figures. The rest of the page, including what the counts mean physically, is argued on the page and
in its stage papers.

## Licence and citation

MIT, as for the whole repository (see `LICENSE` at its top). Cite the page as given in `CITATION.cff`.

Questions: jon@recognitionphysics.org
