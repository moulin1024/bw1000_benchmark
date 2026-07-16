# Mode-Adaptive Truncated SPIKE for Distributed Spectral–FD Pressure-Poisson Solves

**Implementation:** `poisson3d_distributed.py --method spike-adaptive`
**Status:** validated to machine precision on 8 (emulated) devices; see §8.

---

## 1. Motivation

The pressure-Poisson solve in a horizontally periodic, wall-bounded flow solver
(channel / atmospheric boundary layer LES) has the classical structure

1. 2-D real FFT over the periodic horizontal directions,
2. one independent tridiagonal system in the wall-normal direction $z$ for
   every horizontal Fourier mode $(k_x, k_y)$,
3. inverse 2-D FFT.

With the flow fields distributed as **z-slabs** ($n_z/P$ levels per GPU), the
FFTs are communication-free, but the tridiagonal systems couple the full $z$
extent. The standard remedy — transposing to a mode-partitioned layout and
back — moves **two full spectral fields per solve** (≈ 1.9 GiB aggregate at
$1024^2 \times 128$, float64, $P = 8$).

SPIKE substructuring removes the transposes: each GPU eliminates its local
rows, and blocks couple only through a small interface system whose per-solve
data is **two scalars per mode per GPU**. The method documented here goes one
step further: it exploits the *mode-dependent* diagonal dominance of the
Helmholtz-like vertical operator to close **almost every mode with
nearest-neighbour communication only**, retaining the exact global interface
solve for a small, statically known set of low-wavenumber modes. Communication
volume for the global collective then no longer scales with the horizontal
resolution at all.

The three ingredients are individually classical — SPIKE partitioning
(Sameh-style substructuring), the truncated/PDD closure for diagonally
dominant systems, and precomputed factorizations for time-invariant
operators. The contribution is their per-mode adaptive combination in a
spectral Poisson context, with an explicit machine-precision truncation
threshold, and an implementation in pure JAX collectives.

## 2. Discrete problem

Per horizontal mode with wavenumber magnitude $k_h^2 = k_x^2 + k_y^2$, the
vertical system has $n_z + 1$ unknowns $x_0, \dots, x_{n_z}$ (pressure at
half-staggered levels) and rows

$$
\begin{aligned}
\text{row } 0 &: \; -x_0 + x_1 = 0
  &&\text{(bottom Neumann; } x_0 = 0 \text{ for } k_h^2 = 0\text{)}\\
\text{rows } 1 \le j \le n_z - 1 &: \;
  \frac{x_{j-1} - 2x_j + x_{j+1}}{\Delta z^2} - k_h^2\, x_j = d_j
  &&\text{(interior)}\\
\text{row } n_z &: \; -x_{n_z - 1} + x_{n_z} = 0
  &&\text{(rigid-lid top)}
\end{aligned}
$$

with $d_j$ the transformed divergence at physical level $j - 1$. The operator
is time-invariant: everything derivable from $(a, b, c)$ alone may be
precomputed once.

## 3. Block partition and SPIKE substructuring

Let $m = n_z / P$. **Row 0 joins the interface system**; GPU $k$ owns rows
$[km + 1,\, (k{+}1)m]$. This partition has two structural benefits:

* the padded-row bookkeeping of a uniform partition of $n_z + 1$ rows
  disappears ($1 + Pm = n_z + 1$ exactly), and
* block rows align *exactly* with the z-slab layout of both the physical RHS
  (row $j$ reads level $j - 1$) and the output pressure levels — no data
  movement besides the interface exchange.

Write the block system as $A_k x_k + a_f L_k e_0 + c_l R_k e_{m-1} = d_k$,
where $a_f, c_l$ are the couplings leaving the block, $L_k$ is the last
unknown of the previous block ($x_0$ for $k = 0$) and $R_k$ the first unknown
of the next. With

$$
y_k = A_k^{-1} d_k, \qquad
w_k = A_k^{-1} (a_f e_0), \qquad
v_k = A_k^{-1} (c_l e_{m-1}),
$$

the exact block solution is the rank-2 correction

$$
x_k = y_k - w_k L_k - v_k R_k .
$$

Only $y_k$ depends on the RHS; the **spike vectors $w_k, v_k$ are
precomputed**, as are the selected local solver factors. The implementation
supports parallel cyclic reduction (PCR, $\lceil \log_2 m \rceil$ steps) and
Thomas forward/backward scans. This local choice does not change the SPIKE
interface equations. Collecting the first/last entries of each block yields
the $(2P{+}1)$-row **interface system** in the unknowns
$u = (x_0, \alpha_0, \beta_0, \dots, \alpha_{P-1}, \beta_{P-1})$ with
$\alpha_k = x_k[0]$, $\beta_k = x_k[m{-}1]$:

$$
\begin{aligned}
b_0 x_0 + c_0 \alpha_0 &= 0,\\
\alpha_k + w_k[0]\, L_k + v_k[0]\, \alpha_{k+1} &= y_k[0],\\
\beta_k + w_k[m{-}1]\, L_k + v_k[m{-}1]\, \alpha_{k+1} &= y_k[m{-}1].
\end{aligned}
$$

The interface matrix is static per mode, so its factors are precomputed once.
Per solve, the only communicated data are the RHS endpoints
$(y_k[0], y_k[m{-}1])$.

For the exact plain-SPIKE path, the implementation now avoids forming this
dense inverse by default. Eliminating $x_0$ with the bottom boundary equation
leaves $P$ unknown blocks $(\alpha_k,\beta_k)$ and an exact $2\times2$
block-tridiagonal system. Its block-Thomas factorization stores six real
coefficients per block and mode.

Applying those factors with a forward and backward scan saves memory but
serializes $2P$ full-mode operations at runtime. The default `selected-rows`
path instead observes that GPU $d$ consumes only its own pair $(L_d,R_d)$.
During setup it applies the **transpose** of the structured block
factorization to those two selection vectors, directly producing

$$
R_d = E_d S,
$$

where $S$ maps the $2P$ endpoint RHS values to all interface neighbour values.
No dense inverse is formed, even temporarily. Every timed solve then evaluates
the exact response with one parallel $2\times 2P$ contraction. For an
all-gather layout this stores $4P$ real coefficients per mode and GPU; for an
all-to-all layout each GPU stores all $2P$ response rows for its $1/P$ mode
shard, resulting in the same aggregate storage.

The runtime-scan path remains available as
`--spike-interface-solver block-thomas`, and the full dense path as
`--spike-interface-solver dense`, for validation and A/B measurements. The
adaptive method still uses dense inverses only for its small low-$k_h$ box.

The complete FFT--vertical-solve--inverse-FFT pipeline can also be executed in
two ways. `--pipeline-execution monolithic` builds one mapped executable for
the whole solve. `--pipeline-execution staged` dispatches the already compiled
FFT, vertical-solve, and inverse-FFT executables in sequence without
synchronizing their intermediate arrays; only the final result is blocked.
This preserves asynchronous device execution while allowing backends whose
whole-graph buffer scheduling is unfavourable to avoid the monolithic
executable. The BW1000 submission script defaults to `staged`, while the MN5
H100 script keeps `monolithic`; both remain runtime overrides for A/B tests.

Thomas runtime substitution can optionally group several dependent z rows into
one outer-scan body with `--thomas-chunk C`. The original path is `C=1`.
For `C>1`, each outer iteration statically unrolls C forward or backward
recurrences, allowing the compiler to keep the inter-row carry inside the
kernel instead of returning it through a row-at-a-time scan. The factors,
forward workspace, interface equations, and communication payload are
unchanged; no additional full-field array is stored. The BW1000 script uses
`C=16` as the initial candidate, while `THOMAS_CHUNK=1` remains the exact
baseline and MN5 default.

The local field layout is also selectable. The original `--data-layout xyz`
stores physical slabs as $(x,y,z)$ and spectral slabs as $(k_x,y,z)$, which
makes the horizontal FFT axes strided in C-order device memory. The native
`--data-layout z-first` path instead stores

$$
\text{physical}: (z,y,x), \qquad
\text{spectral}: (z,y,k_x).
$$

The horizontal transforms then operate on the two contiguous trailing axes,
and Thomas consumes the leading z axis directly. This is a true end-to-end
layout: RHS generation, MMS fields, FFTs, local SPIKE factors, residuals, and
the transpose method all use z-first arrays. Only the two-dimensional SPIKE
endpoint fields are reordered into canonical $(k_x,k_y)$ interface order, so
no full physical or spectral field transpose and no additional field-sized
buffer are introduced. The BW1000 script defaults to `z-first`; MN5 retains
`xyz` until the H100 A/B result is measured.

## 4. Decay of the spikes: the truncation lemma

The homogeneous solutions of the interior recurrence are $\mu^{\pm j}$ with
$\mu + \mu^{-1} = 2 + k_h^2 \Delta z^2$, i.e.

$$
\ln \mu = 2\,\operatorname{asinh}\!\left(\frac{k_h \Delta z}{2}\right).
$$

A spike vector excited at one end of a block therefore attenuates across the
block's $m$ rows by

$$
\rho(k_h) = \mu^{-m} = \exp\!\left(-2m\,\operatorname{asinh}\!\left(\frac{k_h \Delta z}{2}\right)\right)
\approx e^{-k_h L_z / P} \;\; \text{for } k_h \Delta z \ll 1 .
$$

The *far* endpoints $v_k[0]$ and $w_k[m-1]$ — a spike evaluated at the end
opposite to its excitation — are $O(\rho)$. The *near* endpoints
$w_k[0], v_k[m-1]$ are $O(1)$ and are always retained.

**Truncation rule.** Choose a target exponent
$\tau = -\ln \varepsilon_{\text{mach}} + 4$ (float64: $\tau \approx 40$;
float32: $\tau \approx 20$). For every mode with $\rho(k_h) \le e^{-\tau}$,
dropping the far endpoints perturbs the interface system by less than machine
precision with margin $e^{-4}$. The cutoff wavenumber is

$$
k_c = \frac{2}{\Delta z}\, \sinh\!\left(\frac{\tau}{2m}\right).
$$

## 5. PDD closure for dominant modes

With the far endpoints dropped, the interface system decouples into
independent $2 \times 2$ systems, one per block boundary — precisely the
Parallel Diagonal Dominant (PDD) structure:

$$
\begin{pmatrix} 1 & v_k[m{-}1] \\ w_{k+1}[0] & 1 \end{pmatrix}
\begin{pmatrix} \beta_k \\ \alpha_{k+1} \end{pmatrix}
=
\begin{pmatrix} y_k[m{-}1] \\ y_{k+1}[0] \end{pmatrix},
$$

solved in closed form with the precomputed determinant
$\delta_k = 1 - v_k[m{-}1]\, w_{k+1}[0]$:

$$
\beta_k = \frac{y_k[m{-}1] - v_k[m{-}1]\, y_{k+1}[0]}{\delta_k}, \qquad
\alpha_{k+1} = \frac{y_{k+1}[0] - w_{k+1}[0]\, y_k[m{-}1]}{\delta_k}.
$$

Block $k$ computes its own $L_k = \beta_{k-1}$ and $R_k = \alpha_{k+1}$ from
*one value received from each neighbour* ($y_{k-1}[m{-}1]$ from below,
$y_{k+1}[0]$ from above) plus precomputed neighbour spike endpoints. The
bottom boundary closes locally on block 0: the row-0 equation gives
$x_0 = -(c_0/b_0)\,\alpha_0$, hence
$\alpha_0 = y_0[0]\,/\,(1 - w_0[0]\, c_0/b_0)$ (the pinned $k_h^2 = 0$ mode
has $c_0 = 0$, so $x_0 = 0$ automatically — but that mode never satisfies the
dominance criterion and always takes the exact path).

Communication for the PDD path: **two `ppermute` calls** (send-up and
send-down of one scalar field each), the cheapest and most scalable
collective available — no all-to-all, no all-gather, single hop, naturally
contention-free across nodes.

## 6. Adaptive mode partition

Modes are split statically at setup time:

* **Dominant modes** ($k_h \ge k_c$): PDD closure, neighbour-only
  communication. At $1024 \times 1024 \times 128$, $L_z = 1$, $P = 8$,
  float64 this is **≈ 98.3 %** of all modes.
* **Low-$k_h$ box** ($k_x < k_c$ *and* $|k_y| < k_c$, a rectangular superset
  of the disk $k_h < k_c$ chosen for static array shapes): exact global
  interface solve. The box endpoints are exchanged with one **all-gather** of
  $2 \times$ (box size) scalars, the prefactored $(2P{+}1)$-row systems are
  solved redundantly on every GPU (one small batched matvec), and each GPU
  extracts its own $(L, R)$ entries locally. At the production scale the box
  holds ≈ 9000 of 525k modes (**1.7 %**), an all-gather payload of ≈ 2 MiB.

The two closures write into the same $(L, R)$ fields (`box_scatter`
overwrites the box region), followed by the identical rank-2 correction.
Degenerate limits are handled by clamping: if no mode is dominant (small
$m$, small grids) the box covers the whole spectrum and the method reduces
exactly to plain SPIKE.

**Resolution independence.** The box size depends only on
$k_c(\tau, m, \Delta z)$ and the domain lengths — *not* on $n_x, n_y$. Grid
refinement in the horizontal grows the PDD share and leaves the global
collective payload constant.

## 7. Algorithm summary and cost

**Setup (once; operator is time-invariant):**
1. local PCR or Thomas factors of $A_k$ (couplings leaving the block zeroed);
2. spike vectors $w_k, v_k$ (two local solves);
3. one `ppermute` pair to exchange static neighbour endpoints, PDD
   determinants $\delta$;
4. all-gather of box spike endpoints, assembly and batched inversion of the
   box interface matrices.

**Per solve:**

| step | operation | communication |
|---|---|---|
| 1 | $y = A_k^{-1} d$ (local PCR or Thomas) | — |
| 2 | endpoint exchange | 2 × `ppermute` (one scalar field each) |
| 3 | PDD $2\times2$ closures (all modes) | — |
| 4 | box endpoints | 1 × all-gather ($2\times$ box, ≈ MiB) |
| 5 | box interface matvec, scatter into $(L,R)$ | — |
| 6 | $x = y - wL - vR$; Nyquist filter | — |

Per-GPU communication at $1024^2 \times 128$, float64, $P = 8$:

| method | payload / solve | collectives | pattern |
|---|---|---|---|
| transpose | ≈ 1.9 GiB | 2 | all-to-all (full field) |
| SPIKE | ≈ 30 MiB | 2 | all-to-all (endpoints) |
| **SPIKE-adaptive** | ≈ 17 MiB + 2 MiB | 2 + 1 | **ppermute + tiny all-gather** |

Memory overhead per GPU (float64, production grid): local factors
($2\lceil\log_2 m\rceil{+}1$ full arrays for PCR or $2$ full arrays plus one
compact $m$-element subdiagonal vector for Thomas), spike vectors ($2$ full
arrays), and box interface inverses ($\text{box} \times (2P{+}1)^2$). A full
array has shape $(n_x/2{+}1) \times n_y \times m$. Thomas substantially
reduces factor storage and timed HBM traffic, while PCR exposes more
parallelism along $z$.

For plain SPIKE with an all-gather interface, the exact block-Thomas interface
solver stores $6P$ real coefficients per mode. At $P=8$ this replaces 289
entries of a dense $17\times17$ inverse with 48 entries, reducing the
full-mode interface factors by about $6\times$ without changing the
communication pattern or mathematical solution.

## 8. Validation

Three complementary checks, all layout-independent and run on every
configuration (8 emulated CPU devices; `run` = the actual benchmark script):

1. **Eigenmode MMS** (`--mms --mms-kind modes`): five separable terms whose
   vertical profiles $\cos(m_z \pi \zeta)$, $\zeta = (j - \tfrac12)/(n_z-1)$,
   are exact discrete eigenfunctions of the FD stencil (node symmetry makes
   the one-sided Neumann rows hold exactly). The solver must reproduce the
   field to roundoff.
2. **Broadband MMS** (`--mms --mms-kind broadband`): one random-phase field
   per $z$ level, shaped by a Kolmogorov-like envelope
   $(1 + k_h^2/k_0^2)^{-2/3}$ over *every* resolved mode; the RHS is the
   exact discrete operator applied in spectral space (BC rows enforced by
   level aliasing). Exercises the full spectrum, including modes straddling
   $k_c$.
3. **Equation residual** $\max |A x - d| / \max |d|$ on a white-spectrum
   random RHS, computed with the *full* coupling coefficients — for the
   adaptive method this measures the truncation error directly.

Measured ($32 \times 32 \times 128$; "PDD active" uses $L_z = 100$ so that
$k_c$ falls inside the resolved range, leaving 1.5 % of modes in the box):

| configuration | residual | broadband MMS error |
|---|---|---|
| transpose, float64 | 2.1e-14 | 1.34e-14 |
| SPIKE, float64 | 2.0e-15 | 1.25e-14 |
| adaptive, degenerate (100 % box) | 2.0e-15 | 1.252e-14 (bit-identical to SPIKE) |
| **adaptive, PDD active (1.5 % box)** | **4.4e-16** | **2.053e-12** |
| SPIKE (exact) same conditions | 4.4e-16 | 2.053e-12 (**bit-identical**) |

The elevated MMS level in the PDD-active rows is reproduced *exactly* by the
untruncated methods and is a property of the test problem's conditioning at
$L_z = 100$ (the near-singular Neumann column amplifies FFT roundoff), not of
the truncation: the adaptive solution agrees with the exact solvers to every
printed digit. At the physical configuration ($L_z = 1$) all methods sit at
1.3e-14 (float64).

## 9. Relation to prior work

Partitioned tridiagonal solvers originate with Sameh & Kuck and Wang's
partition method; the spike-vector formulation and the name are due to
Polizzi & Sameh (SPIKE). The truncated closure for diagonally dominant
systems is Sun's Parallel Diagonal Dominant (PDD) algorithm; GPU batches of
Thomas/PCR hybrids are analysed by László, Giles & Appleyard, and
transpose-free tridiagonal solves for CFD appear in PaScaL_TDMA (Kim et al.).
The elements specific to this work: (i) the *per-mode adaptive* PDD/SPIKE
split driven by the spectral Helmholtz shift with an explicit
machine-precision threshold $k_c = (2/\Delta z)\sinh(\tau/2m)$; (ii) full
amortization of the time-invariant operator (local PCR/Thomas factors, spike
vectors, PDD determinants, prefactored interface inverses), reducing per-solve
interface data to the information-theoretic minimum of two scalars per mode
per direction; and
(iii) a pure-JAX collective implementation (ppermute / all-gather) validated
by discrete manufactured solutions. *(Verify citation details before
publication.)*

## 10. Limitations and extensions

* **Stretched vertical grids** change only the row coefficients $(a, b, c)$;
  the SPIKE/PDD machinery is untouched. The decay rate becomes
  $\sum_j \ln \mu_j$ over the block; a conservative cutoff uses the minimum
  local $\ln\mu_j$. (The uniform-grid closed form above is what is currently
  implemented.)
* **Multi-node**: the PDD path is nearest-neighbour and maps directly onto
  node boundaries; only the ≈ MiB box all-gather crosses the fabric globally.
* **float32**: $\tau \approx 20$ enlarges the PDD share further; combined
  with mixed-precision iterative refinement (residual computable locally from
  the returned $(L, R)$) this halves all bandwidth-bound costs.
* The interface matrix inversion uses batched `jnp.linalg.inv` at setup; if a
  backend lacks batched LU, the box is small enough to invert on the host.
* Requires $n_z / P \ge 2$ and $n_z \bmod P = 0$; the $k_h^2 = 0$ pinned mode
  is always inside the box by construction.
