"""Exact off-shell TT spectrum; run remotely with the reference table path."""
import itertools,json,sys
from pathlib import Path
import sympy as s
tab=json.loads(Path(sys.argv[1]).read_text())['couplings']
M0=s.zeros(15)
for r in tab:
 i,j=[sum(int(x)<<a for a,x in enumerate(r[k]))-1 for k in ('D','Dprime')]
 M0[i,j]+=s.Rational(r['weight_exact'])
Mi=21*M0/64+5*M0*M0/128
assert M0*Mi*M0==M0
def strain(H):
 return s.Matrix([(s.Matrix([(m>>i)&1 for i in range(4)]).T*H*s.Matrix([(m>>i)&1 for i in range(4)]))[0] for m in range(1,16)])
def calc(u,a,b,c):
 M2,M4=s.zeros(15),s.zeros(15)
 for r in tab:
  i,j=[sum(int(x)<<k for k,x in enumerate(r[key]))-1 for key in ('D','Dprime')]
  d=sum(u[k]*s.Rational(r['two_times_midpoint_separation'][k],2) for k in range(4))
  w=s.Rational(r['weight_exact'])
  M2[i,j]+=-w*d*d/2;M4[i,j]+=w*d**4/24
 pol=[(a*a.T-b*b.T)/s.sqrt(2),(a*a.T+b*b.T-2*c*c.T)/s.sqrt(6)]
 pol +=[(v*w.T+w*v.T)/s.sqrt(2) for v,w in ((a,b),(a,c),(b,c))]
 B=s.Matrix.hstack(*[strain(H) for H in pol])
 Q=s.simplify(B.T*(M4-M2*Mi*M2)*B)
 return {'matrix':str(Q),'eigenvalues':{str(k):int(v) for k,v in Q.eigenvals().items()},'numerical_eigenvalues':{str(k.evalf(16)):int(v) for k,v in Q.eigenvals().items()}}
e=[s.eye(4)[:,i] for i in range(4)]
out={'axis':calc(e[0],e[1],e[2],e[3]),'face':calc((e[0]+e[1])/s.sqrt(2),(e[0]-e[1])/s.sqrt(2),e[2],e[3])}
Path('QuarticSpectrum.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
