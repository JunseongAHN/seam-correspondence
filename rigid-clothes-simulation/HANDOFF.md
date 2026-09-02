# Isometric Shell space — handoff

## What it is

The shape a sewing pattern implies on its own, before gravity gets a say.

Take the flat panels, close the seams, do not stretch the cloth, and stay outside
the body it was cut for. That is the whole definition. No gravity, no cloth
physics, no folds, no collision response beyond an analytic body.

It sits between two spaces that are already familiar:

```
  flat pattern  ──────────►  ISOMETRIC SHELL  ──────────►  drape
  (2D panels)                (3D, sewn, unstretched)       (3D, worn, wrinkled)
```

Naming it by what it lacks -- "the drape without gravity" -- is the fastest way
to explain it, and the wrong way to think about it. The shell is where the
pattern's metric is still intact; the drape is that same surface after gravity
has folded information out of it. Going right loses something, and the numbers
below say how much.

## Why it exists

A geometry image is a field of 3D positions over the flat panel. It is only as
good as the map from the panel to the surface. The drape distorts that map; the
shell does not.

Same mesh, same domain, only the 3D values differ (`gi_complexity.py`):

| | Isometric Shell | drape |
|---|---|---|
| distance from isometry, p50 \|σ−1\| | **0.0042** | 0.0177 |
| p90 | **0.0280** | 0.0872 |
| p99 | **0.1082** | 0.3088 |
| one pixel's surface area, p99 | **1.125** | 1.240 |
| surface under-sampled by >10% | **1.5%** | 9.0% |
| physical size of a fixed kernel, max/min | **1.11×** | 1.40× |
| pixels needed for a given surface density | **1.12×** | 1.24× |

The third-from-last line is the one to quote to anyone building a CNN: in the
drape the *same* 3×3 kernel covers a physical patch that varies by 1.40× across
the garment, so a convolution means different things in different places. In the
shell that variation is 11%.

Where the drape wrinkles, the shell is also cheaper to represent. Projecting each
panel onto the first k eigenmodes of its own flat Laplacian:

```
  skirt_front (10344 verts)     k=10    k=30    k=100   k=300     RMS error, cm
    Isometric Shell            3.453   1.454   0.532   0.252
    drape                      3.607   2.159   1.404   0.612
```

100 modes reach 0.53 cm in the shell; the drape is still at 0.61 cm with 300.

**On body-fitted panels the two are equal.** Characteristic wavelength ratios are
0.97 (ftorso), 0.93 (sleeve), 1.05 (hood), 0.86 (waistband) -- the drape is
already smooth there because the body holds it flat, and the shell is marginally
worse in places, from folding in a 5 cm band at the waist seam. The honest claim
is "3-4× closer to isometric everywhere, and 2.6× more compressible where the
drape wrinkles", not "simpler in every respect".

## How to reproduce it

```
python run_garment.py --garment <id> --outdir <dir> --amp 0 \
                      --body --sym --mu 0.02
```

lambda_b runs 1e-1 down to 1e-8 in eight rungs of 400 iterations, 3400 in all,
with one matrix factorisation per rung and none per iteration. About 6 minutes
for a 18k-vertex garment, 18 for 30k.

Running several at once, cap the BLAS threads. Eight processes each spawning
sixteen OpenMP threads on sixteen cores ran **6× slower** than eight processes
with two:

```
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
```

### The pieces

| file | what it does |
|---|---|
| `gcd_io.py` | reads a GarmentCode sample: flat panels, seam correspondence, placement. The drape is loaded only for comparison and returned under `drape` so any accidental use is visible |
| `assembly.py` | ARAP with per-triangle rotations, Bergou quadratic bending, seam stitching, the lambda_b continuation |
| `body.py` | the analytic body: 9 plain cylinders and 1 sphere, from the measurements file and the placement, never from the drape |
| `run_garment.py` | the driver |
| `gi_complexity.py` | the geometry-image measurements above |
| `perturb_pattern.py` | Lipschitz test: does a nearby pattern give a nearby shell |
| `render_patches.py` | per-panel colours, panel outlines in black |
| `measure_all.py`, `analyze.py` | the full measurement battery |

### Results in the tree

```
result/lambda_sweep/   the lambda_b endpoint sweep, on the true flat metric
result/v4c_skirt2/     six runs of rand_00YONAPXZE, measurements, renders, README
                       -- produced WITH the deleted ease, so they describe a
                       pattern whose waist gather was erased.  Regenerate.
result/v4d_skirt/      the same, coloured per panel with black outlines
proxy/<garment_id>/    run_batch.py output, one directory per garment.  Not in
                       the repo: gitignored, and rebuilt by run_batch.py
result/v1_noconstraint/  v2_hardclamp/  v3_plane/  v4a_fatarm/  v4b_mu1/
                       earlier attempts, kept because their failures are the
                       argument for the current design
```

## What the solver does, and the three things that had to be right

**ARAP with per-triangle rotations.** Never a vertex one-ring. `R = U V^T` on the
Stiefel manifold V2(R^3) with no determinant fix, because V2(R^3) is connected
unlike SO(3) in O(3). Computed in closed form as `R = F (F^T F)^{-1/2}` -- F^T F
is 2×2, so its inverse square root is explicit -- which matches `np.linalg.svd` to
9e-13 and is much faster over 57k triangles.

**One factorisation per lambda_b rung.** The seam weight continuation rides on a
Woodbury update, and the obstacle penalty is written so the matrix does not depend
on the active set: `mu` is carried by *every* constrained vertex, violating or
not, and the active set enters only through the right-hand side.

**Three things that were wrong and had to be fixed**, each of which produced a
garment that looked plausible in outline and was ruined in detail:

1. *A hard clamp cannot be used.* Projecting onto the feasible side after each
   global step drives triangles flat onto the constraint surface (σ = 0), opens
   seam gaps to 6 cm, and breaks descent -- 1507 monotonicity violations. The
   one-sided quadratic penalty replaced it and gives 0.

2. *The sphere test was missing `sd > 0`.* It kept a vertex whenever the head
   sphere was "deeper" than any cylinder, without also requiring that the sphere
   contain it. Both depths are negative outside everything, so 5315 vertices --
   18% of the mesh, mostly torso -- were teleported onto the head sphere every
   iteration. The bodice came out 4.6 cm tall against a 45.5 cm placement.

3. *The obstacle must fit inside the garment, not just inside the body.* The arm
   cylinder was r = 5.15 where the cuff ring can only wrap r = 3.17, so the cuff
   tore open by 1.6×. This dress has no ease at all -- its torso ring measures
   99.90 cm against a measured bust of 99.84 -- so an obstacle even slightly too
   large has no isometric solution and the solver answers by tearing.

## Two design decisions worth understanding before changing anything

**The body proxy is inscribed, twice over.** Every primitive fits inside the body
part it stands for *and* inside what the garment can wrap, so it can never push
the garment further than either would. The torso radius is the ellipse's
semi-minor axis (the half-depth), not `C/2π`: taking the circumference would make
the torso 32.9 cm deep at the hips where the body is 25.6 and wedge the front and
back panels apart. The aspect ratio 1.548 comes from the one genuine chord width
in the measurements file, `shoulder_w`, paired with the chest circumference the
back arc implies. The `*_back_width` entries are arcs, not widths -- no ellipse of
circumference 103.48 reaches a semi-axis of 27.41.

**The seam ease was removed, and the audit that removed it.**  A `seam_ease.py`
scaled the rest metric near a seam so that neither side is ever compressed,
gated by a RELATIVE threshold of 25%.  That was wrong on three counts and the
module is deleted; git history has it.

*It eased gathers.*  This repo already classifies seams by an absolute
tau_l = 10 mm (`run_baseline.py`, from arXiv 2607.21213): dart/self, equal
(<=1 mm), ease (1-10 mm), gather (>10 mm).  Of 54 stitches, 6 are gathers, and
the modification eased **2 of them** -- both waist seams, dL = 30.31 mm, three
times tau_l, applied at s = 1.0750 and 1.0744.  They slipped through because a
25% relative gate cannot see a 30 mm mismatch on a 42 cm seam, so the gate
systematically misses gathers on long seams.  86.2% of all the rest area the
modification added came from those two seams.

*Most of the rest was noise.*  13 of the 16 applied scales are smaller than the
mesh's own UV calibration error (median 0.35%, p90 0.74%), and some flip sign
against the specification -- `left_hood`/`left_ftorso` was eased because the mesh
says ftorso is 0.43 mm longer while the spec says the hood is 0.06 mm longer.

*And it was applied at the wrong granularity.*  `seam_scales` keys on panel
pairs, not stitches, and 10 of 29 pairs bundle two or three stitches. An `equal`
stitch received 0.90 pp more correction than its own mismatch while an `ease`
stitch received 6.68 pp less.

Removing it costs nothing.  Measured against the TRUE flat metric, the sweep run
without it is better than the run with it -- max |sigma-1| 0.4069 against 0.6319,
p50 0.0029 against 0.0042, and on the 710 gathered-seam faces p50 0.0400 against
0.0640.  The ease was absorbing about half the designed gather at the median and
making those faces merely look unstretched.

One fact cuts the other way and is recorded because it is true: the waist seam
carries no designed excess.  Specification arc lengths sum to 84.333800 cm on
both sides, identical to six decimals, `design.waistband.waist` is 1.0 and there
is no waist ruffle parameter.  The 30.31 mm is a front/back distribution
difference -- waistband cut 45.198/39.136, skirt 42.167/42.167 -- that an
absolute per-edge rule reads as a gather.  The classification is correct by the
stated rule; the design-intent reading it invites is not.  The module is still
removed, because the two other defects stand on their own.

## Parameter sensitivity

Measured, not guessed:

| parameter | tried | effect |
|---|---|---|
| `--mu` obstacle weight | 1.0 / 0.1 / 0.02 | p50 \|σ−1\| 0.0186 / 0.0041 / 0.0016. **Monotone**, smaller is better |
| `lambda_b` ladder start | 1e-1 / 1e1 / 1e3 / 1e6 | 1e-1 is right. **1e6 crumples the garment into a 22 cm ball** -- isometric, E_arap 19.4, and useless |
| `--amp` perturbation | 0.01 / 0.10 / 0.30 | 0.01 is right; larger only worsens the stretch |
| soft-first ladder | no stiff rung | no effect whatsoever |
| `--anchor` to the placement | 0.05 / 1e-3 / 1e-4 | 0.05 pins the panels so hard they cannot sew (E_anchor 9.1e4 against E_arap 3.9e4). Left off |
| `lambda_b` ladder END | 1e-4 / 1e-6 / 1e-8 | all three converge, 0 monotonicity violations, and **smaller is better**: max \|σ−1\| 0.4573 / 0.4054 / 0.4069, p99 0.0864 / 0.0641 / 0.0548.  The skirt is unchanged -- 2 lobes and a 20.2 cm hem radius in all three -- so the buckling wavelength does not depend on it |

The dangerous knobs are the continuation schedule -- `lambda_b` start and
perturbation amplitude -- where two decades gives a qualitatively different
answer. `mu` is sensitive but monotone, which is the benign kind.

`mu` deserves its own note: with it on every vertex, the term is proximal,
`mu (x − x_prev)`, on every vertex that is not violating anything, and that damps
exactly the long-wavelength modes which decide the shape. At `mu = 1` E_arap was
still falling 9% per rung after 3400 iterations and had only reached 210; at
`mu = 0.02` it reaches 15. **The distortion was never the constraint's price, it
was arriving early.** Anything that looks like "the constraint costs too much"
should be checked against convergence first.

## Generality

`assembly.py` reads no panel names at all. The single place a panel name is read is three lines in `body.py` that
locate the arm axis from the sleeve panels, because taking `arm_pose_angle` from
the vertical puts the axis 13 cm above the cuff. A sleeveless garment gets no arm
cylinders and works unchanged.

Three further garments ran with no tuning:

| garment | verts | structure | max \|σ−1\| | seam gap | mono viol |
|---|---|---|---|---|---|
| `rand_00YONAPXZE` | 29784 | the original | 0.631 | 3.9e-5 | 0 |
| `rand_023FMIGQK0` | 17879 | no sleeves, collar, 3-tier skirt | **0.150** | 7.3e-6 | 0 |
| `rand_05PQL1BU0S` | 12487 | sleeves + collar + waistband, no skirt | 0.441 | 2.6e-5 | 0 |
| `rand_0A36YXPNV0` | 18002 | sleeves + cuff-skirt, no waistband | **0.257** | 9.1e-6 | 0 |

All three came out *better* than the garment the pipeline was built on, which has
zero ease and a 1.84 gather. The dataset holds 3450 samples.

## Known limits

**The skirt becomes a flat disc, and that is correct.** Its panel is a flat
annulus -- `r(s) = r0 + s` holds to 1% for the first 30 cm, which is a full circle
skirt. The unique low-bending isometric embedding of a flat annulus is a flat
disc. Without gravity there is nothing that prefers hanging folds; folding costs
bending, spreading does not. `lambda_b` cannot fix this: 1e-8 and 1e-4 give hem
radii identical to three digits. Only a gravity term would, and it is deliberately
absent.

**Folding concentrates in a 5 cm band at the waist seam.** 8.2% of edges there
exceed 45°, against 0.0% for the drape; away from seams the shell is far smoother
than the drape (dihedral p50 0.16° against 7.05°). The excess circumference has to
go somewhere, and with the skirt spread flat it all lands at the waist. This is
the one place the shell is worse, and it is why the fitted panels do not win in
the spectral comparison.

**A few triangles remain badly stretched.** p99 of |σ−1| is 0.11 but the max is
0.4-1.6 depending on the seed. A network trained with an L2 loss will chase that
tail; it should be characterised before it is trusted.

**The geometry-image comparison is n = 1.** Sections above are measured on
`rand_00YONAPXZE` only.

## What to do next, in order

1. **Measure smoothness in *pattern* space.** This is the important one and it has
   not been done. The 1.375 cm spread across seeds says the map is a function of
   the pattern; it says nothing about whether it is a *continuous* function of the
   pattern, which is what a network needs. Perturb a GarmentCode design parameter
   slightly, regenerate, and see whether the shell moves slightly. If the shell
   jumps anywhere -- and the envelope-to-round transition in the skirt is a
   candidate -- training will fail there and no amount of capacity will help.

2. **Scale to ~100 garments and count failures.** Three is not a sample. Log
   `max |σ−1|`, seam gap and monotonicity violations for each and look at the
   tail, not the median.

3. **Fix the waist-seam band.** It is the only measurement the drape wins. The
   cause is understood -- excess circumference with nowhere to go once the skirt
   spreads -- so the fix is whatever makes the skirt not spread, which currently
   means gravity, which is excluded. Worth revisiting as a soft directional prior
   rather than a physical force.

4. **Repeat `gi_complexity.py` across the dataset** so the headline numbers have
   error bars.

5. **Then train something small on both spaces and compare.** Every number in this
   document is a proxy for that experiment.
