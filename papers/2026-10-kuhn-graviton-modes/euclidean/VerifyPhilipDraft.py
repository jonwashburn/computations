"""Run on the authorized research host, never as local scientific compute.

Usage: python3 VerifyPhilipDraft.py <September-10 reproducibility directory>
Writes the receipt in the current directory. No source files are mutated.
"""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import sympy as s
import mpmath as mp

start = time.time()
root = Path(sys.argv[1])
table_path = root / 'reference/regge_4d_kuhn_coupling_table_20260903.json'
table = json.loads(table_path.read_text())['couplings']
z = s.symbols('z0:4', nonzero=True)
pairs = list(itertools.combinations(range(4), 2))
masks = [1 << i for i in range(4)] + [(1 << i) + (1 << j) for i,j in pairs]
M = s.zeros(15)
for row in table:
    D,E,v = (row[k] for k in ('D','Dprime','two_times_midpoint_separation'))
    a,b = [sum(int(x)<<i for i,x in enumerate(vec)) for vec in (D,E)]
    powers = [s.Rational(v[i]+int(D[i])-int(E[i]),2) for i in range(4)]
    assert all(p.q == 1 for p in powers)
    M[a-1,b-1] += s.Rational(row['weight_exact'])*s.prod(z[i]**powers[i] for i in range(4))
def sharp(A):
    return A.subs(dict(zip(z,[1/x for x in z])), simultaneous=True).T
def zero(A):
    return all(s.cancel(x)==0 for x in A)
I,J = [m-1 for m in masks], [m-1 for m in (7,11,13,14)]
assert M.extract(J,J) == -s.eye(4)/2
R = M.extract(I,I)+2*M.extract(I,J)*M.extract(J,I)
beta = s.eye(4)-s.ones(4)/2
V = s.diag(beta/2,s.eye(6))
C,W = s.zeros(4,10),s.zeros(10)
for i in range(4):
    C[i,i] = z[i]-1
    for j in range(4): W[i,j] = z[i]*beta[i,j]
for a,(i,j) in enumerate(pairs):
    C[i,4+a] = 1-1/z[j]
    C[j,4+a] = 1-1/z[i]
    W[4+a,4+a] = 2
    for r in range(4):
        if r not in (i,j): W[4+a,r] = -z[i]*z[j]
L = sum(2-zi-1/zi for zi in z)
K = L*V-sharp(C)*C
checks = {}
checks['Philip_squared_edge_factorization_all_100_entries'] = zero(sharp(W)*R*W+K/2)
checks['constraint_identity'] = zero(C*V.inv()*sharp(C)-L*s.eye(4))
checks['detW'] = str(s.factor(W.det()))
Xg = s.zeros(10,4)
for i in range(4): Xg[i,i]=2*(z[i]-1)
for a,(i,j) in enumerate(pairs):
    Xg[4+a,i]=Xg[4+a,j]=2*(z[i]*z[j]-1)
checks['gauge_map_y_equals_minus_VinvCsharp_xi'] = zero(-W*V.inv()*sharp(C)-Xg)
checks['hypotenuse_row_column_all_momenta'] = zero(M[14,:]) and zero(M[:,14])
M0 = M.subs(dict(zip(z,[1]*4)))
checks['M0_characteristic_polynomial'] = str(s.factor(M0.charpoly().as_expr()))
f=[zi-1 for zi in z]; t=[1-1/zi for zi in z]
F=s.diag(*f)
printed=s.Matrix(4,4,lambda i,j:-(1 if i==j else 0)*f[i]*t[i]+f[i]*t[j])
correct=s.Matrix(4,4,lambda i,j:-2*(1 if i==j else 0)*f[i]*t[i]+f[i]*t[j])
checks['printed_F_beta_formula_is_correct'] = zero(2*F*beta*sharp(F)-printed)
checks['corrected_F_beta_formula_is_correct'] = zero(2*F*beta*sharp(F)-correct)

# Quartic harmonic identity in a lattice-aligned frame, on the unit sphere.
x,y,u=s.symbols('x y u', real=True)
Y40=3*(35*u**4-30*u**2+3)/(16*s.sqrt(s.pi))
Y44sum=3*s.sqrt(s.Rational(35,2))*(x**4-6*x*x*y*y+y**4)/(8*s.sqrt(s.pi))
harm=s.Rational(3,5)+4*s.sqrt(s.pi)/15*(Y40+s.sqrt(s.Rational(5,14))*Y44sum)
checks['spherical_harmonic_identity'] = s.rem(s.expand(harm-x**4-y**4-u**4),u*u+x*x+y*y-1,u)==0

mp.mp.dps=70
EP=mp.mpf('1.220890e19'); hbarc=mp.mpf('1.973269804e-16'); lp=hbarc/EP
Mpc=mp.mpf('3.0856775814913673e22')/hbarc
Fpsi=mp.mpf(9)/13
def sbound(E,L):return mp.sqrt(Fpsi*EP**2/((E/mp.mpf(1e9))**7*L*Mpc))
def athr(E,L,A):return mp.sqrt(12/(1+A)*sbound(E,L))
def energy(a,L,A):return (Fpsi*EP**2/(((1+A)*a*a/12)**2*L*Mpc))**(mp.mpf(1)/7)*mp.mpf(1e9)
def fmt(v):return mp.nstr(v,24)
E=mp.mpf('3.2e20')
b=sbound(E/560,10)
anchor=mp.sqrt(4*mp.log(2))/EP
numbers={'s_bound_GeV_minus2':fmt(b),'isotropic_s00_bound':fmt(mp.sqrt(4*mp.pi)*b),
    'direction_independent_a_m':fmt(athr(E/560,10,mp.mpf(1)/3)*hbarc),
    'axis_a_m':fmt(athr(E/560,10,1)*hbarc),
    'axis_rows_a_over_lP':[[fmt(athr(e,l,1)*EP) for l in (10,100)] for e in (E/560,E/10,mp.mpf('1e20'),E)],
    'planck_loss_energies_eV':[fmt(energy(1/EP,l,a)) for a,l in ((1,10),(mp.mpf(1)/3,10),(1,100))],
    'anchored_loss_energies_eV':[fmt(energy(anchor,l,a)) for a,l in ((1,10),(mp.mpf(1)/3,10),(1,100))],
    'anchored_a_m':fmt(anchor*hbarc),'anchored_E_GeV':fmt(1/anchor),
    'anchored_s_range':[fmt((1+a)*anchor**2/12) for a in (mp.mpf(1)/3,1)],
    's00_per_a2':fmt(4*mp.sqrt(mp.pi)/15),'s40_per_a2':fmt(mp.sqrt(mp.pi)/45),
    's44_per_a2':fmt(mp.sqrt(mp.mpf(5)/14)*mp.sqrt(mp.pi)/45)}
assert all(v is True for k,v in checks.items() if isinstance(v,bool) and k!='printed_F_beta_formula_is_correct')
assert checks['printed_F_beta_formula_is_correct'] is False
result={'started_unix':start,'ended_unix':time.time(),'sympy':s.__version__,'mpmath':mp.__version__,
    'table_sha256':hashlib.sha256(table_path.read_bytes()).hexdigest(),
    'checks':checks,'numbers':numbers,'scope':'Exact algebra and numerical substitutions; not a formal proof or matter-coupling derivation.'}
Path('PhilipExactChecks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
