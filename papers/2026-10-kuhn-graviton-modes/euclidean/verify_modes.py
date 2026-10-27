"""Independent NumPy audit of the displayed edge modes.

Run: python3 verify_modes.py
Reassembles the MIDPOINT symbol directly from the rational table, converts
the explorer's VERTEX amplitudes, and checks gauge quotient independence.
"""
import cmath
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import numpy as np
from explorer import explore
from lattice_modes import modes

ROOT = Path(__file__).resolve().parent
TABLE = json.loads((ROOT/'reference/regge_4d_kuhn_coupling_table_20260903.json').read_text())['couplings']
EDGES = np.array([[(mask >> i) & 1 for i in range(4)] for mask in range(1,16)])


def midpoint_symbol(p):
    matrix = np.zeros((15,15),complex)
    for row in TABLE:
        i,j=[sum(int(x)<<a for a,x in enumerate(row[key]))-1 for key in ('D','Dprime')]
        v=np.array(row['two_times_midpoint_separation'])/2
        matrix[i,j] += float(Fraction(row['weight_exact']))*cmath.exp(1j*np.dot(p,v))
    return matrix


def decode(values):
    return np.array([complex(*x) for x in values])


def run():
    rng=random.Random(20260916)
    configurations=[(az,el,f) for az,el in [(0,0),(45,0),(45,math.degrees(math.asin(1/math.sqrt(3)))),(90,0),(0,90),(23,37)]
                    for f in [0,1e-5,.01,.3,.7,1]]
    configurations += [(rng.uniform(0,90),rng.uniform(0,90),rng.uniform(.01,1)) for _ in range(120)]
    worst=0; worst_constraint=0; min_physical_singular=1; min_bad=1; worst_reality=0
    for az,el,f in configurations:
        # First call determines the actual direction-dependent zone boundary.
        kmax=explore(az,el,0)['kmax']
        r=explore(az,el,f*kmax)
        p=np.array([*(r['k']*np.array(r['n'])),1j*r['omega']])
        matrix=midpoint_symbol(p)
        phases=np.exp(-1j*(EDGES@p)/2)
        reverse=modes(r['k'],[-x for x in r['n']],-r['omega'])
        vectors=[]
        for name,mode in r['modes']['polarizations'].items():
            # The rendered spatial field is real: reversing momentum/frequency
            # conjugates its amplitudes. The chosen cross basis reverses sign.
            mirror=decode(reverse['polarizations'][name]['squared_edge_amplitudes'])[:7]
            expected=np.conj(decode(mode['squared_edge_amplitudes'])[:7])*(1 if name=='plus' else -1)
            reality_error=float(np.linalg.norm(mirror-expected)/np.linalg.norm(expected))
            assert reality_error<1e-12
            worst_reality=max(worst_reality,reality_error)
            x=decode(mode['squared_edge_amplitudes'])*phases
            err=np.linalg.norm(matrix@x)/(np.linalg.norm(matrix)*np.linalg.norm(x))
            assert err<2e-12, (az,el,f,err)
            worst=max(worst,float(err)); worst_constraint=max(worst_constraint,mode['constraint_residual'])
            assert abs(max(abs(decode(mode['strains'])[:7]))-1)<1e-12
            vectors.append(x/np.linalg.norm(x))
            # Known-wrong field: modify an interacting body edge.
            bad=x.copy();bad[6]+=.5*np.linalg.norm(x)
            bad_err=np.linalg.norm(matrix@bad)/(np.linalg.norm(matrix)*np.linalg.norm(bad))
            assert bad_err>1e-5
            min_bad=min(min_bad,float(bad_err))
        if f>0:
            # Four vertex-displacement columns, plus the inert hyperdiagonal.
            gauge=2*EDGES*(np.exp(1j*(EDGES@p))[:,None]-1)*phases[:,None]
            gauge/=np.linalg.norm(gauge,axis=0)
            inert=np.eye(15,dtype=complex)[:,14]
            combined=np.column_stack([gauge,*vectors,inert])
            smallest=np.linalg.svd(combined,compute_uv=False)[-1]
            assert smallest>1e-4, (az,el,f,smallest)
            min_physical_singular=min(min_physical_singular,float(smallest))
    # Exact counterexample from the earlier assessment, now repaired.
    face=explore(45,0,math.sqrt(2)*math.pi)
    assert face['modes']['polarizations']['plus']['constraint_residual']<1e-12
    # The plotting routine cannot silently accept a missing/zero mode.
    assert all(np.linalg.norm(decode(mode['strains']))>0 for mode in face['modes']['polarizations'].values())
    return {'configurations':len(configurations),'mode_equation_checks':2*len(configurations),
            'max_independent_midpoint_residual':worst,'max_constraint_residual':worst_constraint,
            'min_seven_column_independence_singular_value':min_physical_singular,
            'min_corrupted_field_residual':min_bad,
            'max_spatial_conjugate_pair_error':worst_reality,
            'face_zone_plus_constraint_repaired':True,
            'scope':'Numerical full-symbol verification plus non-gauge independence; construction is derived in MODE_CONSTRUCTION.txt.'}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
