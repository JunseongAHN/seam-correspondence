# v4c — the runs that came out right, and what made them right

## How each file here was produced

Every run starts from the specification placement plus a 1% random perturbation.
The drape is never an input.

```
python run_garment.py --seed {1,2,3,4,5} --body --sym --ease 5.0 --mu 0.02
python run_garment.py --seed 1 --inflate 1.5 --body --sym --ease 5.0 --mu 0.02
```

lambda_b runs 1e-1 down to 1e-8, eight rungs of 400 iterations, 3400 in total,
eight factorisations -- one per rung, none per iteration.

`mu1_previous/` holds the earlier six at `mu = 1.0`, and `assembly_probe_mu0.02`
is the short 1800-iteration probe that first showed the difference.

Also here: `body_proxy.ply` (the obstacle), `placed_init.ply` (the initial
state), `rest_flat.ply` (the flat pattern), `drape_reference.ply` (the
GarmentCode simulation, comparison only), `measurements.txt` (the full battery).

## Results

|                             | no constraint | mu = 1.0 | **mu = 0.02** | drape |
|---|---|---|---|---|
| p50 \|sigma-1\| off gathers | 0.0004 | 0.0186 | **0.0042** | - |
| p99                          | 0.0061 | 0.3633 | **0.0934** | - |
| front panels behind the back | 48.2%  | 0.0%   | **0.0%**   | 0% |
| left panels on the right     | 23.5%  | 6.4%   | **1.4%**   | 0% |
| seam gap, max                | 1.5e-5 | 7.5e-5 | **2.0e-5** cm | - |
| waistband droop below its seam | -    | 2.57   | **0.00** cm | 0.00 |
| waistband seam-to-seam        | -      | 10.25  | **15.48** cm | 15.53 |
| initial-value spread, p50     | 6.40   | 5.64   | **1.375** cm | - |
| monotonicity violations       | 0      | 0      | **0**      | - |
| intersecting triangle pairs   | 2093   | 205    | 857        | - |

Seam gap tolerance is 1e-4 x the 230.4 cm characteristic length = 2.3e-2 cm, so
every run passes it by three orders of magnitude.

The spread across the six runs falling to 1.375 cm is the result that matters for
using this as a space to learn over: the same pattern lands in the same place.

The 857 intersections are up from 205, but they are the two halves meeting at the
centre line -- left_ftorso against right_ftorso (113), skirt_back against
skirt_front (92), the two hood halves (86) -- which is contact from the garment
sitting closer to the body, not the collapse that produced 2093.  Self-collision
is counted, never resolved.

## What separates these from the runs that looked wrong

Three of the four differences are not parameters at all — they are defects that
were fixed, and every earlier run carries at least one of them.

**The sphere test was missing a condition.** `take = sd > dep` kept a vertex
whenever the head sphere was "deeper" than any cylinder, without also requiring
that the sphere actually contain it.  Both depths are negative outside
everything, so any vertex merely nearer the head than to any cylinder was
teleported onto the sphere every iteration — 5315 of them, 18% of the mesh,
mostly torso panels.  The bodice came out 4.6 cm tall against a 45.5 cm
placement.  Fixed by `(sd > dep) & (sd > 0)`.

**The arm cylinder was wider than the sleeve can wrap.**  The proxy was built
inscribed in the *body*, but never checked against the *garment*.  The cuff ring
is 19.9 cm around (r = 3.17) and the arm was r = 5.15 there, so the cuff was torn
open by 1.6x — p50 |sigma-1| = 1.49 on those faces.  This dress has no ease at
all (its torso ring measures 99.90 cm against a measured bust of 99.84), so an
obstacle even slightly too large has no isometric solution and the solver answers
by tearing.  `body.ring_radius` now caps the arm at what the cuff can wrap.

**A hard clamp cannot be used.**  Projecting onto the feasible side after each
global step drives triangles onto the constraint surface (max |sigma-1| = 1, i.e.
sigma = 0), opens seam gaps to 6 cm and breaks descent (1507 monotonicity
violations).  The one-sided quadratic penalty replaces it: `mu` is carried by
every constrained vertex rather than only the violating ones, so the matrix does
not depend on the active set and the single factorisation per rung still holds.

**The seam ease** (`--ease 5.0`) is the one real addition.  The waist seam does
not match — 45.23 cm of waistband front against 42.10 of skirt front, and 39.16
of waistband back against 42.10 of skirt back, though the totals agree to 0.2%.
A side that must lengthen resolves it in plane; a side that must shorten buckles,
and that buckling is the pinch.  Setting the shared rest length to the longer of
the two removes it.  Measured across the seam, 3D/rest went from 0.995
(compressed) to 1.025 (in tension) — no side is compressed any more.  Only the
rest METRIC is edited, over a 5 cm band and along the seam tangent only;
`d["rest"]`, the geometry-image domain, is untouched.

## Is it parameter sensitive?

Yes, and the sensitivity is not spread evenly.  What was measured:

| parameter | tried | effect |
|---|---|---|
| `mu` obstacle weight | 1.0 / 0.1 / 0.02 | p50 \|sigma-1\| 0.0186 / 0.0041 / 0.0016 — **monotone**, smaller is better, and 0.02 also removed the 2.57 cm waistband droop entirely |
| `lambda_b` ladder start | 1e-1 / 1e1 / 1e3 / 1e6 | 1e-1 is right.  1e3 changes the skirt section ratio 4.92 -> 3.90; **1e6 crumples the whole garment into a 22 cm ball** (isometric, E_arap 19.4, and useless) |
| perturbation `--amp` | 0.01 / 0.10 / 0.30 | 0.01 is right; 0.10 and 0.30 only make the stretch worse |
| soft-first ladder | 1e-1 .. 1e-7 with no stiff rung | no effect at all, identical to the baseline |
| `--anchor` to the placement | 0.05 / 1e-3 / 1e-4 | 0.05 is far too strong (E_anchor 9.1e4 against E_arap 3.9e4 — the panels stay pinned to the placement and cannot sew).  Left off. |
| `--ease` band width | 5 cm only | **not tested** |

So the dangerous knobs are the continuation schedule, `lambda_b` start and
perturbation amplitude, where being wrong by a couple of decades gives a
qualitatively different answer.  `mu` is sensitive but monotone and well behaved,
which is the benign kind: there is no knife edge, the answer just keeps improving
as it goes down, at the cost of letting the garment settle a little deeper into
the body (max penetration 0.95 -> 2.63 cm).

Why `mu` matters so much is worth stating: with `mu` on every vertex, the term is
proximal -- `mu (x - x_prev)` -- on every vertex that is not violating anything,
and that damps exactly the long-wavelength modes which decide the shape.  At
`mu = 1` E_arap was still falling 9% per rung at the end of 3400 iterations and
had reached 210; at `mu = 0.02` it reaches 15.  The twelvefold difference in
distortion was never the constraint's price, it was arriving early.


