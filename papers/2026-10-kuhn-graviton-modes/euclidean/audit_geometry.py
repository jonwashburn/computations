#!/usr/bin/env python3
"""Independent checks on the original unsymmetrized Hessian and exact table.

Run from this directory. Does not replace the supplied numerical
program: audits its inputs before its explicit symmetrization step.
"""
import collections
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "original"))
import sympy as sp
import regge_4d_quartic_tt_dispersion_20260903 as original

def key_of(e, ep):
    v = original.midpoint_of(e) - original.midpoint_of(ep)
    return e[1], ep[1], tuple(int(round(2*x)) for x in v)

def reverse(key):
    a,b,v = key
    return b,a,tuple(-x for x in v)

def main():
    # Imported geometry code prints Greek symbols. Redirected Windows output
    # can use a legacy encoding; escape unsupported log characters safely.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    start = time.time()
    grads, fund, _ = original.assemble_exact_grads(4)
    raw = collections.defaultdict(lambda: sp.Integer(0))
    for area, deficit in grads:
        for e,a in area.items():
            for ep,d in deficit.items():
                raw[key_of(e,ep)] += a*d
    raw = {k:sp.simplify(v) for k,v in raw.items() if sp.simplify(v) != 0}
    asym = {str(k):str(sp.simplify(v-raw.get(reverse(k),0))) for k,v in raw.items()
            if sp.simplify(v-raw.get(reverse(k),0)) != 0}
    ref = json.loads((ROOT/'reference/regge_4d_kuhn_coupling_table_20260903.json').read_text())
    def mask(D): return sum(int(x)<<i for i,x in enumerate(D))
    table = {(mask(r['D']),mask(r['Dprime']),tuple(r['two_times_midpoint_separation'])):
             sp.sympify(r['weight_exact']) for r in ref['couplings']}
    mismatches = [str(k) for k in raw.keys() | table.keys() if raw.get(k,0) != table.get(k,0)]
    # Non-vacuous negative controls: perturb one existing weight, and empty table.
    bad = dict(table)
    bad[next(iter(bad))] += sp.Rational(1,100)
    def matches(t): return bool(t) and t == raw
    assert matches(table) and not matches(bad) and not matches({})
    body = [7,11,13,14]
    u = sp.symbols('u0:4', real=True)
    bodyblock = sp.Matrix(4,4,lambda i,j:sp.simplify(sum(w*sp.exp(sp.I*sum(u[l]*v[l]/2 for l in range(4)))
             for (a,b,v),w in raw.items() if a==body[i] and b==body[j])))
    result = {
      'host':platform.node(),'started_unix':start,'finished_unix':time.time(),
      'fundamental_hinges':len(fund),'unsymmetrized_nonzero_couplings':len(raw),
      'unsymmetrized_reciprocity_failures':asym,'reference_table_mismatches':mismatches,
      'exact_weight_census':dict(collections.Counter(str(w) for w in raw.values())),
      'nonrational_weights':sum(not w.is_Rational for w in raw.values()),
      'exact_body_diagonal_block':str(bodyblock),
      'body_block_equals_minus_half_identity':bodyblock == -sp.eye(4)/2,
      'negative_controls_reject_perturbed_and_empty_table':True,
      'dA_hypercube_nonzero':sum(v!=0 for a,d in grads for (base,m),v in a.items() if m==15),
      'dDelta_hypercube_nonzero':sum(v!=0 for a,d in grads for (base,m),v in d.items() if m==15),
    }
    out = ROOT/'results/geometry_audit.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert not asym and not mismatches and result['body_block_equals_minus_half_identity']

if __name__ == '__main__': main()
