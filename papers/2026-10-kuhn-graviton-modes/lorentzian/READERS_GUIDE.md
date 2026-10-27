# Reading the Lorentzian section with the code

This guide accompanies Section 6 and Appendix A of *Graviton Modes on the Kuhn
Lattice: Euclidean Continuation and Lorentzian Propagation*, together with the
numbers in Sections 7 to 9. Each step names the function that computes it, so
you can go from a paragraph of the paper to the code that produces its result.

The programs start from vertex coordinates and the background metric. They
compute every area, angle and derivative, assemble the Hessian, and only then
compare with what the paper prints. Exact results use SymPy rationals and
square roots. The finite-momentum checks use mpmath at 50 digits. Equation,
section and table numbers below refer to the unmarked draft of 27 October 2026.
The programs name equations by their LaTeX labels (for example `eq:plus-det`),
which stay fixed if the paper is renumbered.

```
python lorentzian/run_all.py                    # all four programs, about 80 s
python lorentzian/section6_paired_direction.py  # one program and its checks
```

## Seven things the prose leaves implicit

These are the places where the paper relies on a convention or a computation
it does not spell out. Each is worth reading before the section-by-section
tables below.

### 1. Why the clock is the hypercube diagonal, and where $2\sqrt3$ comes from

The background is $g_L = I_4 - J_4$. Every coordinate axis is a null
direction, and an edge $D\in\{0,1\}^4$ with $n$ ones has squared interval
$n(1-n)$. The lattice's symmetries are the 24 permutations of the axes and the
inversion $x\to -x$. The only direction that every permutation leaves fixed is
$(1,1,1,1)$. Its interval is $g_L((1,1,1,1),(1,1,1,1)) = 4 - 16 = -12$, so it
is timelike, and a unit clock along it is $\tau = (1,1,1,1)/(2\sqrt3)$. The
factor $2\sqrt3$ is $\sqrt{12}$.

The spatial direction $\sigma = (1,1,-1,-1)/2$ has components summing to zero,
so the $J_4$ part of the metric does not see it: $g_L(\sigma,\sigma) = |\sigma|^2 = 1$
and $g_L(\tau,\sigma) = 0$. The pair $(\tau,\sigma)$ is one unit of time and
one unit of length, orthogonal to each other. A wave covector $q$ then has
frequency $a\omega = q(\tau)$ and wave number $K = q(\sigma)$. On
$q = (z+t,\,z+t,\,z-t,\,z-t)$ these are $a\omega = 2z/\sqrt3$ and $K = 2t$, the
light cone $g_L^{-1}(q,q) = 4t^2 - \tfrac43 z^2 = 0$ is $z=\sqrt3\,t$, and the
phase speed is $z/(\sqrt3\,t)$.

This is why the paper fixes the clock before comparing coefficients. The
numbers $+5/24$ and $-59/24$ are coefficients of $K^2$ in these units. A
different clock rescales $\omega$ and $K$ differently and changes them.

The parameter $a$ sets the scale of the metric. It cannot be the length of an
axis edge, because axis edges have zero interval.

Code: `section_6_1_clock` in `section6_paired_direction.py`.

### 2. Why the paired direction

On a covector of the form $q = (x,x,y,y)$, swapping coordinates
$1\leftrightarrow2$ or $3\leftrightarrow4$ leaves $q$ unchanged. Both swaps are
lattice symmetries, so they commute with the symbol $\mathsf M_L(q)$. The 15
edge amplitudes therefore split into four sectors by their parity under the two
swaps, with dimensions 8 (even, even), 3, 3 and 1 (odd, odd). The program
computes these dimensions.

The one-dimensional (odd, odd) sector is what makes the cross polarization
exact. $\mathsf M_L(q)$ maps that sector to itself, so its single vector is an
eigenvector at every $q$ on the slice, and the dispersion relation is one scalar
equation instead of a determinant.

The paired covectors are exactly those that vanish on the two transverse
directions introduced below: $q = z(1,1,1,1) + t(1,1,-1,-1)$. Such a wave
travels along $\sigma$ in the rest frame of the clock $\tau$.

Code: `section_6_2_cross` (the sector dimensions and the swap commutation).

### 3. What the cross column is

Equation (78), $c_D = 2(D_1-D_2)(D_3-D_4)$, is the edge strain of the
continuum cross polarization. The transverse directions $b_1 = (1,-1,0,0)$ and
$b_2 = (0,0,1,-1)$ are $g_L$-orthogonal to $\tau$ and $\sigma$. A constant metric
perturbation $h$ changes the squared interval of edge $D$ by $D^{\mathsf T}hD$,
and for $h = b_1\otimes b_2 + b_2\otimes b_1$ this is
$2(D\cdot b_1)(D\cdot b_2) = 2(D_1-D_2)(D_3-D_4)$. The lattice keeps this
vector exactly; only its eigenvalue $s_\times(x,y)$ carries the lattice
corrections.

The plus polarization $b_1\otimes b_1 - b_2\otimes b_2$ is even under both
swaps, so it lives in the eight-dimensional even sector. There it mixes with
the other even edge strains, which is why it needs the orbit block.

### 4. What the paired orbit block is

A vector in the (even, even) sector takes the same value on every edge class
that the swaps carry into each other. These orbits are labelled by
$(D_1+D_2,\,D_3+D_4)$. There are eight: two axis orbits $\{e_1,e_2\}$ and
$\{e_3,e_4\}$, and the six non-axis orbits $(0,2), (1,1), (1,2), (2,0), (2,1), (2,2)$
of Appendix A, with sizes 1, 4, 2, 1, 2, 1.

The axis components can be set to zero by a gauge choice. A vertex
displacement contributes $4i\,\xi_\mu\sin(q_\mu/2)$ to axis edge $\mu$, so
whenever $\sin(x/2)$ and $\sin(y/2)$ are nonzero a displacement can cancel any
axis components, and choosing $\xi_1=\xi_2$, $\xi_3=\xi_4$ keeps the vector
even. What remains is six orbit values $w$. The matrix
$B = O^{\mathsf T}\mathsf M_L O$, with $O$ the six unnormalized orbit indicator
columns, is the quadratic action restricted to those vectors.

Code: `block_B` builds $O$ and $B$; `section_6_3_plus` compares all 36 entries
with equation (124).

### 5. Why the full-kernel reconstruction matters beyond a determinant root

Gauge fixing removes the four axis variables. It does not remove the four axis
equations of motion. A zero of $\det B$ says only that $O^{\mathsf T}r = 0$ for
the residual $r = \mathsf M_L O w$: the residual is orthogonal to the six
orbit indicators. Two more facts make $r$ vanish entirely.

1. Symmetry. $r$ is even under both swaps, so it is constant on each orbit. A
   zero orbit sum then means each non-axis entry of $r$ is zero. The program
   checks that the rows of $\mathsf M_L O$ agree within every orbit.
2. The lattice Bianchi identity. For every vertex displacement $\xi$,
   $g_\xi^{\mathsf T} r = g_\xi^{\mathsf T}\mathsf M_L c = (\mathsf M_L g_\xi)^{\mathsf T} c = 0$.
   With the non-axis entries already zero, this reads
   $\sum_\mu 4i\,\xi_\mu \sin(q_\mu/2)\, r_\mu = 0$ for every $\xi$, so each axis
   entry $r_\mu$ vanishes where $\sin(q_\mu/2)\neq0$.

This is the lattice form of a familiar statement: the constraint equations
follow from the dynamical ones. Finally, $c$ has zero axis components, and a
gauge vector with zero axis components has $\xi = 0$, so $c$ is a wave, not a
coordinate change.

The sine prefactor in equation (125) shows why the argument is needed.
$\det B$ contains $\sin^2(x/2)\sin^2(y/2)$. At $x = 0$ the displacement
$\xi = (1,1,0,0)$ produces a strain with no axis components that is constant on
orbits. It sits inside the gauge-fixed space, and $B$ annihilates it. Those
zeros of $\det B$ are pure gauge. The program checks this at $x=0$ and at $y=0$
(last check of `section_6_3_plus`).

At $K = 0.1$, on each branch, the lifted vector solves all 15 equations to
about $10^{-50}$ relative to the matrix entries, $\mathsf M_L$ has exactly five
null directions (four gauge and the mode), and the mode is independent of the
gauge vectors. Code: `numeric_branches`.

### 6. Where $\sqrt3$ and the factor 2 in $2R/\sqrt3$ come from

The Hessian of $S = \sum_h A_h\delta_h$, restricted to one simplex, is
$C_{ff'} = \sum_h (\partial A_h/\partial a_{f'})(\partial\delta_h/\partial a_f)$,
with $\delta_h = 2\pi - \sum\theta$, so $\partial\delta_h = -\partial\theta_h$.
On the chain simplex the triangle Gram determinants are $-1, -4, -9$ and $-12$,
and the interior angles are $\pi/6, \pi/3, \pi/2$ and $2\pi/3$. The square roots
combine so that every entry of $C$ is a rational multiple of $1/\sqrt3$. The
paper writes $C = 2R/\sqrt3$ and prints $32R$. Multiplying the whole symbol by
$\sqrt3$ gives $\mathsf M_L$, whose assembly rule has coefficients $2R$. The
kernel is unchanged.

Equation (122) uses the midpoint frame: an edge of class $D$ at
vertex $x$ carries the phase $e^{iq\cdot(x+D/2)}$. The coupling between two
edges of one simplex then depends only on the difference of their midpoints,
and because the lattice is symmetric under $x\to-x$ each term pairs with its
mirror image. That gives the cosine form.

Code: `KuhnSimplex.hessian_cell` in `kuhn_lorentz_geometry.py`;
`appendix_A_cell` and `appendix_A_assembly` in `appendixA_cell.py`.

### 7. What the source constraints mean physically, and what stays open

Section 7 is Euclidean. The linearized coupled equation is
$\gamma\,\mathsf H\,\delta a = -J$, where $J_e = \partial S_m/\partial a_e$ is the
force the matter exerts on edge $e$. $\mathsf H$ is symmetric, so for every null
vector $n$ of $\mathsf H$,
$n^{\mathsf T}J = -\gamma\,(\mathsf H n)^{\mathsf T}\delta a = 0$. This is the
lattice analogue of the requirement that the matter stress be conserved: in
general relativity the Bianchi identity forces $\nabla_\mu T^{\mu\nu} = 0$.

$\mathsf H$ has two kinds of null vectors. The hypercube-diagonal strain is null
because the diagonal is always the hypotenuse of a right triangle, so no area
depends on it at first order. The matter action does depend on the diagonal,
because the scalar simplex metric uses all ten edges. For
$\phi = \lambda(x_1+x_2)$ the program finds
$\partial S_m/\partial a_{1111} = -\kappa_m\lambda^2/12$ summed over one
hypercube: the field pushes on an edge that gravity cannot respond to.

The second example survives deleting the diagonal. Moving the vertex
$p = (0,1,0,0)$ along $\xi = e_1 - e_3$ is a symmetry of the gravitational
action but changes the scalar action at first order:
$g^{\mathsf T}J = -18\,\kappa_m\lambda^2$ for $\phi = \lambda(x_1^3 - 3x_1x_2^2)$.
The program computes this twice: from equation (96), and by
moving the vertex in the embedding and differentiating $S_m$ directly. A field
that is harmonic on the background is not automatically a consistent source.

What stays open is the Lorentzian characteristic set in a general direction.
Off the paired slice there is no swap symmetry. The problem is then the full
$15\times15$ symbol with its four gauge directions, and the physical modes are
the $q$ at which $\mathsf M_L(q)$ has more than four null vectors. The paired ray
establishes that the two polarizations split along $\sigma$. It does not say
how the splitting depends on direction, which a radiation calculation needs,
because emission sums over directions. None of the programs here attempts the
general direction.

## Section by section

### Section 6.1: geometry, time and momentum

| Paper | What is computed | Function |
|---|---|---|
| $g_L = I_4 - J_4$, $g_L^{-1} = I_4 - J_4/3$ | the inverse metric | `section_6_1_geometry` |
| interval $n(1-n)$; axis edges null | all 15 edge classes | `squared_interval` |
| nondegenerate simplices, timelike triangles | Gram determinants of triangles, tetrahedra, simplex | `KuhnSimplex.gram` |
| angles $\pi/6$, $2\pi/3$ (Euclidean $\pi/4$, $\pi/2$) | $\cos\theta = -X_{lm}/\sqrt{X_{ll}X_{mm}}$, $X = G^{-1}$ | `KuhnSimplex.angle` |
| angle sum $2\pi$ around each hinge | all 50 hinge classes, 4 or 6 simplices each | `hinge_angle_sums` |
| $\tau$, $\sigma$, $q$, $K = 2t$, $a\omega = 2z/\sqrt3$, cone $z=\sqrt3t$ | exact | `section_6_1_clock` |

The end triangles of the chain are 012 and 234 (angle $\pi/6$); the middle
triangle is 123 (angle $2\pi/3$). The normal plane of every triangle is
spacelike, so each angle is an ordinary rotation angle and the Euclidean
formula applies.

### Section 6.2: the exact cross branch

| Paper | Result | Function |
|---|---|---|
| (78) | $c$ on the four mixed face diagonals, values $+2,-2,-2,+2$ | `section_6_2_cross` |
| (79) | $\mathsf M_L(x,x,y,y)\,c = s_\times c$ exactly; coefficients $-2, 3/2, 3/2, -1$ | `matvec` |
| (80) | $s_\times(z+t,z-t) = -(2\cos^2z - 3\cos z\cos t + 1)$ | `section_6_2_cross` |
| (82) | $2\cos^2 z - 3\cos z\cos t + 1 = 2G_\times(U,V)$; $\partial_V G_\times(0,0) = -1$ | `section_6_2_cross` |
| (81) | $v_\times = 1 + \tfrac{5}{24}K^2 + o(K^2)$ | `phase_coefficient` |
| nongauge | axis components zero, sines nonzero | `numeric_branches` |

The sine variables are $U = \sin^2(t/2)$ and $V = \sin^2(z/2)$, so
$\cos t = 1-2U$ and $\cos z = 1-2V$. Zero momentum is $U = V = 0$, and the light
cone starts along $V = 3U$. `phase_coefficient` inserts
$z = \sqrt3\,t(1+\alpha t^2)$ into $G(U,V)$, expands in $t$, and solves the
$t^4$ term for $\alpha$. The $t^2$ term vanishes because the branch starts on
the cone. The phase speed is $1 + \alpha t^2 = 1 + (\alpha/4)K^2$.

### Section 6.3 and Appendix A: the plus branch and the comparison

| Paper | Result | Function |
|---|---|---|
| (121) | $32R$ from the geometry equals the printed matrix, all 100 entries | `appendix_A_cell` |
| (122) | printed cosine rule equals the exponential assembly; $\mathsf M_L$ symmetric | `appendix_A_assembly` |
| Lemma 2.2(iii), Lorentzian form | $\mathsf M_L(q)\,g_\xi(q) = 0$ for all $q$ | `check_bianchi` |
| (124) | $B = O^{\mathsf T}\mathsf M_L O$ equals the printed block, all 36 entries | `block_B` |
| (125) | $\det B = -36\sin^2\frac{z+t}2\sin^2\frac{z-t}2\,P(\cos z,\cos t)$ | `section_6_3_plus` |
| (83) | $P(1-2V,1-2U) = G_+(U,V)$; $\partial_V G_+(0,0) = 2$ | `section_6_3_plus` |
| (84) | $v_+ = 1 - \tfrac{59}{24}K^2 + o(K^2)$; $v_\times - v_+ = \tfrac83K^2 + o(K^2)$ | `phase_coefficient` |
| reconstruction | orbit rows equal; Bianchi on the slice; full kernel at $K = 0.1$ | `numeric_branches` |
| Euclidean comparison | axis time $1-(1+A)K^2/24$; hypercube time with $\sigma$: $1+K^2/24$ | `euclidean_comparison` |

The determinant is checked as an identity of Laurent polynomials in
$a = e^{ix/2}$ and $b = e^{iy/2}$, as the paper suggests:
$\cos z = (ab + 1/(ab))/2$ and $\cos t = (a/b + b/a)/2$.

At finite $K$ the plus branch has a large next-order term: $(v-1)/K^2$ is
$-2.249$ at $K = 0.1$ and $-2.449$ at $K = 0.02$, against the limit $-2.458$.
The program therefore uses $K = 0.02$ and $0.01$ for its Richardson check. This
is a numerical observation about the branch, not a statement in the paper.

The Euclidean comparison uses the exact Laplacian factor of the continued
Euclidean symbol (Section 4) and expands $\mathcal L(i\omega T + K\hat n) = 0$.
Both Euclidean polarizations share that factor, so neither continuation splits
them.

The self-test at the end of `appendixA_cell.py` runs the same geometry code on
the Euclidean lattice and reproduces the published 302-entry coupling table
exactly, weight by weight. That fixes the conventions (midpoint separations,
the sign of $\delta$, which index carries the area derivative) independently of
the Lorentzian result.

### Section 7: the two source obstructions

| Paper | Result | Function |
|---|---|---|
| (86) | $G = \min(i,j)$, the printed $E$, $V_\sigma = 1/24$ on all 24 simplices | `main` |
| printed $B_\sigma$ and $\delta G$ | for $\phi = \lambda(x_1+x_2)$ on the canonical simplex | `edge_source_formula` |
| first obstruction | $\partial S_m/\partial a_{1111} = -\kappa_m\lambda^2/12$; $\partial A_h/\partial a_{1111} = 0$ | `edge_source_direct` |
| (104) | $g^{\mathsf T}J = -18\kappa_m\lambda^2$ over 120 simplices; 0 at the origin | `contraction` |
| Proposition 7.1 inputs | $\mathsf H g = 0$ near $p$; hypercube-diagonal row of $\mathsf H$ is zero | `main` |

### Sections 8 and 9: the conditional bound and the candidate spacing

These sections use the Euclidean axis-time branch, not the Lorentzian branches,
and import the fermion emission rate of Kostelecký and Tasson.
`sections8_9_bound_numbers.py` recomputes every printed number from $E_P$,
$\hbar c$, the megaparsec and the event energy, and compares at the printed
number of significant digits.

One printed value does not follow from the paper's own inputs. In Table 5 the
iron row at 100 Mpc is printed as $4.1\times10^3$. With
$E_f = E/560 = 5.714\times10^{17}$ eV the threshold is $4036\,\ell_P$, which
rounds to $4.0\times10^3$. Rounding $E_f$ to $5.7\times10^{17}$ eV first gives
4054, which rounds to $4.1\times10^3$. Every other number in these sections,
including $a < 1.42\times10^{-31}$ m, $1/a > 1.39\times10^{15}$ GeV and the
factor $1.86\times10^7$, matches only the unrounded $E/560$. The program reports
this entry as its one failed check and prints both values.

## Conventions in one place

- Coordinates: the Lorentzian section works directly with squared-edge
  amplitudes $c_D$. The Frobenius coordinates $\mathrm{vec}_F$ of Sections 3 to 5
  describe metric perturbations in the Euclidean calculation and are not used
  here.
- Edge classes: the 15 nonzero $D\in\{0,1\}^4$, ordered as 4 axes, 6 face
  diagonals, 4 body diagonals and 1 hypercube diagonal (`EDGE_CLASSES`).
- Edge slots in a simplex: $(01,02,03,04,12,13,14,23,24,34)$, as in Appendix A
  (`EDGE_SLOTS`).
- Deficit: $\delta_h = 2\pi - \sum\theta$, with $\theta$ the interior angle, so
  $\partial\delta = -\partial\theta$.
- Hessian: $\mathsf H_{ee'} = \sum_h (\partial A_h/\partial a_{e'})(\partial\delta_h/\partial a_e)$
  (equation (4)); the second-order action is $\tfrac12 c^\dagger\mathsf M c$
  per cell.
- Plane waves: midpoint convention, $e^{iq\cdot(x+D/2)}$.
- Gauge vectors: $g_\xi(q)_D = 4i\,(D\cdot\xi)\sin(q\cdot D/2)$, with
  $\xi = g\zeta$ the covector of the vertex displacement $\zeta$.
- Trigonometric polynomials: a dictionary $\{v: c\}$ means
  $\sum c\, e^{iq\cdot v/2}$ with integer $v$ (`TrigPoly`). On the paired slice
  the key $(p,r)$ means $e^{i(px+ry)/2}$, and the paper's $C_{pr}$ is
  $\cos((px+ry)/2)$.
