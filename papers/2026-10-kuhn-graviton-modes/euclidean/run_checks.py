"""Run the three exact paper checks from this standalone package.

Usage: python run_checks.py
Outputs are written beneath this package, never into a user's other files.
The full geometry reconstruction is separate; see README.txt.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
os.chdir(root)
table = Path('reference/regge_4d_kuhn_coupling_table_20260903.json')
state = Path('state/qg_full_theory')
state.mkdir(parents=True, exist_ok=True)
shutil.copyfile(table, state / table.name)
for script, arguments in [
    ('original/regge_4d_exact_quadratic_identity_20260904.py', []),
    ('VerifyPhilipDraft.py', ['.']),
    ('VerifyQuartic.py', [str(table)]),
]:
    print('\nRunning ' + script, flush=True)
    subprocess.run([sys.executable, script, *arguments], check=True)
print('\nAll three programs completed. Inspect the JSON results described in README.txt.')
