"""Small helper for printing and recording checks in the companion programs."""
import json
import sys


class Report:
    def __init__(self, title):
        self.title = title
        self.checks = []
        self.values = {}
        print("=" * 78)
        print(title)
        print("=" * 78)

    def section(self, name):
        print()
        print("-- " + name + " " + "-" * max(0, 74 - len(name)))

    def note(self, text):
        print("   " + text)

    def value(self, key, val, show=True):
        self.values[key] = str(val)
        if show:
            print("   %s = %s" % (key, val))

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.checks.append({"check": name, "pass": ok, "detail": str(detail)})
        print("   [%s] %s%s" % ("PASS" if ok else "FAIL", name, (": " + str(detail)) if detail else ""))
        return ok

    def finish(self, json_path=None):
        n_fail = sum(1 for c in self.checks if not c["pass"])
        print()
        print("%s: %d checks, %d failed" % (self.title, len(self.checks), n_fail))
        if json_path:
            with open(json_path, "w") as fh:
                json.dump({"title": self.title, "checks": self.checks, "values": self.values}, fh, indent=1)
        return n_fail


def out_path(argv, default):
    """Optional first argument: path of the JSON summary to write."""
    return argv[1] if len(argv) > 1 else default
