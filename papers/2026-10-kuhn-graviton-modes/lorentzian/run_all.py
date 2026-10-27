#!/usr/bin/env python3
"""
run_all.py -- run every program in this folder once and store its full output.

Writes, for each program, outputs/<name>.txt (everything it prints) and outputs/<name>.json
(each check with PASS or FAIL, and the values it reports), then outputs/RUN_RECORD.json with the
runtime, the Python and library versions, and the SHA-256 hash of every program and output.

Run from anywhere:  python lorentzian/run_all.py
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
PROGRAMS = ["appendixA_cell", "section6_paired_direction", "section7_source_constraints",
            "sections8_9_bound_numbers"]
LIBRARIES = ["kuhn_lorentz_geometry.py", "kuhn_lorentz_symbol.py", "report.py", "run_all.py"]


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    import mpmath
    import sympy
    record = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "python": platform.python_version(), "sympy": sympy.__version__, "mpmath": mpmath.__version__,
              "platform": platform.platform(), "programs": {}, "sha256": {}}
    failed = 0
    for name in PROGRAMS:
        txt = os.path.join(OUT, name + ".txt")
        js = os.path.join(OUT, name + ".json")
        t0 = time.time()
        with open(txt, "w") as fh:
            proc = subprocess.run([sys.executable, os.path.join(HERE, name + ".py"), js],
                                  cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        with open(js) as fh:
            summary = json.load(fh)
        n_fail = sum(1 for c in summary["checks"] if not c["pass"])
        failed += n_fail
        record["programs"][name] = {"exit_code": proc.returncode, "seconds": round(dt, 1),
                                    "checks": len(summary["checks"]), "failed": n_fail,
                                    "failed_checks": [c["check"] for c in summary["checks"] if not c["pass"]]}
        print("%-30s %3d checks, %d failed, %.1f s" % (name, len(summary["checks"]), n_fail, dt))
    for f in sorted(PROGRAMS):
        record["sha256"]["lorentzian/" + f + ".py"] = sha(os.path.join(HERE, f + ".py"))
    for f in LIBRARIES:
        record["sha256"]["lorentzian/" + f] = sha(os.path.join(HERE, f))
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".txt", ".json")) and f != "RUN_RECORD.json":
            record["sha256"]["outputs/" + f] = sha(os.path.join(OUT, f))
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(OUT, "RUN_RECORD.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print("total failed checks: %d (see outputs/RUN_RECORD.json)" % failed)
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
