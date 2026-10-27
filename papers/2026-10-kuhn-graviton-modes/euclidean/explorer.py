"""Closed-form axis-time graviton branch for the public QG explorer.

Equations: Lattice Graviton Dispersion on the Kuhn Triangulation,
September 15, 2026 revision, 'Time along an axis'. Units: a = c = 1.
This companion evaluates the derived branch and constructs two lattice modes
from the exact coupling table. It does not rebuild the geometric Hessian.
The original programs are supplied in original/.
"""
import json
import math
from lattice_modes import modes


def branch(k, n):
    if k == 0:
        return {"omega": 0.0, "phase": 1.0, "group": 1.0, "radial_group": 1.0,
                "group_vector": list(n)}
    p = [k * x for x in n]
    s = sum(math.sin(x / 2) ** 2 for x in p)
    omega = 2 * math.asinh(math.sqrt(s))
    vg = [math.sin(x) / (2 * math.sqrt(s * (1 + s))) for x in p]
    return {"omega": omega, "phase": omega / k,
            "group": math.sqrt(sum(v * v for v in vg)),
            "radial_group": sum(v * x for v, x in zip(vg, n)), "group_vector": vg}


def explore(azimuth, elevation, k):
    azimuth, elevation, k = map(float, (azimuth, elevation, k))
    if not all(map(math.isfinite, (azimuth, elevation, k))):
        raise ValueError("Parameters must be finite")
    if not (0 <= azimuth <= 90 and 0 <= elevation <= 90):
        raise ValueError("Angles must lie between 0 and 90 degrees")
    az, el = math.radians(azimuth), math.radians(elevation)
    n = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    norm = math.sqrt(sum(x * x for x in n))
    n = [x / norm for x in n]
    kmax = math.pi / max(abs(x) for x in n)
    k = max(0, min(k, kmax))
    result = branch(k, n)
    result.update({"n": n, "k": k, "kmax": kmax,
                   "kappa": (1 + sum(x ** 4 for x in n)) / 24,
                   "curve": [[kmax * i / 180, branch(kmax * i / 180, n)["omega"]]
                             for i in range(181)]})
    result['modes'] = modes(k, n, result['omega'])
    return result


def explore_json(parameters):
    return json.dumps(explore(**json.loads(parameters)), allow_nan=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--azimuth", type=float, default=0)
    parser.add_argument("--elevation", type=float, default=0)
    parser.add_argument("--k", type=float, default=1.2)
    print(json.dumps(explore(**vars(parser.parse_args())), indent=2, allow_nan=False))
