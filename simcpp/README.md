# simcpp — the isometric assembly solver in C++ / WebAssembly

A port of the numerical core of `../rigid-clothes-simulation/` (`assembly.py`,
plus the obstacle projector from `body.py`) to C++17 + Eigen, buildable both
natively and to WebAssembly with Emscripten.

`../rigid-clothes-simulation/` is read-only here and nothing in it was changed.

Agreement with the Python on real garments, after the full 5-rung `lambda_b`
ladder and 350 iterations:

| garment | verts | max &#124;diff&#124; vs Python | rel. to bbox |
|---|---|---|---|
| `rand_1328ERLDIC` (trousers, 6 panels) | 10 186 | **9.94e-11 cm** | 1.1e-12 |
| `rand_0B1T21D8NX` (shirt, 14 panels)   |  8 377 | **2.99e-09 cm** | 3.3e-11 |

All four gates (`mono_violations`, `seam_gap_max`, `max_sigma_dev`, the
energies) agree to 1e-9 relative or better.  See "Validation" below.

## What is ported

Everything in `assembly.py`:

* `shape_gradients` — per-triangle rest gradients and **|signed area|**.  The
  absolute value is not a patch: every back panel in this dataset is
  parametrised mirrored relative to its 3D winding, and mirroring sends
  `F -> F M` for a reflection `M`, which leaves the singular values — and hence
  `||F-R||^2` over the Stiefel manifold — unchanged.  Signed areas would put
  negative weights on half the mesh and make the stiffness matrix indefinite.
* `best_rotations` — the closed-form 2x2 polar decomposition, kept exactly as in
  the Python (`sqrt(C) = (C + sqrt(det C) I) / sqrt(tr C + 2 sqrt(det C))`, then
  a 2x2 inverse).  No SVD, no determinant fix.
* `build_hinges` + `hinge_stencils` — Bergou quadratic bending over **every**
  interior edge of the welded topology, with the two triangles unfolded into a
  common plane across seams and their own rest frames reused within a panel.
  The hinge enumeration preserves Python's dict insertion order, so the hinge
  list is index-for-index identical (verified: 0 mismatches on both garments).
* `Assembly` — assembling `L0 = K_cot + lambda_b H_bend + eps I + diag(mu+nu)`,
  the sparse Cholesky (`Eigen::SimplicialLDLT` in place of SciPy's `splu`; `L0`
  is SPD so this is the better factorisation, not merely an equivalent one), and
  the Woodbury update carrying the whole `w_s` continuation on one factorisation.
* `solve` / `solve_annealed` — the `lambda_b` ladder, the geometric `w_s`
  schedule, the monotonicity gate, the convergence break, `recenter`.
* the `mu`/`clamp` obstacle penalty and the `nu`/`anchor` term, with the same
  "`mu` is constant so the matrix is independent of the active set" structure.
* `body.py`'s `penetration` / `projector` — the cylinder+sphere closest-point
  projector, including the first-max/first-min tie-breaking that `np.argmax` /
  `np.argmin` imply and the "sphere must be deeper **and** contain the vertex"
  condition.

The clamp is a `std::function<void(Mat3X&)>` and the penalty is a plain
diagonal vector, so any other projector can be dropped in; `BodyProxy` is just
the one this dataset needs.

## What is deliberately left out

* **`gcd_io.py`** — as instructed.  Reading `_sim.ply`, the `_orig_lens.pickle`
  and the `_body_measurements.yaml`, the UV calibration, the welding, the
  connected-component panel assignment and the specification placement all stay
  on the host side.
* **The construction of the body proxy** from the measurements yaml
  (`body.primitives`, `aspect_ratio`, `half_depth`, `ring_radius`, `arm_axis`).
  That is data preparation: it reads a yaml and the placement, and produces a
  handful of cylinders and one sphere.  The solver takes those primitives as
  input and ports the part that runs *inside the iteration* — the projector.
  So a `--body`-equivalent run is fully possible (and is what the numbers above
  come from); what the C++ cannot do is derive the primitives itself.
* `half_space_penalty` / `half_space_sets` — likewise host-side; they only build
  a `mu` vector and a clamp, and `--half-lr`/`--half-fb` runs work through the
  same dumped `mu` + a trivial clamp. The C++ has no baked-in half-space clamp.
* `initial()`'s random perturbation, `plyio`, `analyze.py`, the reporting.
* `Assembly(woodbury=False)` exists but the no-seam path is untested.

## Layout

```
simcpp/
  src/sim.h        types + the whole public surface
  src/sim.cpp      the numerics: ARAP, hinges, Assembly, solve_annealed, BodyProxy
  src/io.cpp       the flat input reader and a minimal .npy writer
  src/main.cpp     native CLI driver (also builds to a node CLI under emcc)
  src/wasm.cpp     the C API (_sim_build/_sim_solve/...) + an embind mirror
  CMakeLists.txt   single build file, native and Emscripten
  build.sh         the exact command lines used here
  dump_garment.py  host-side: gcd_io.load(...) -> the flat .bin the solver reads
  selftest.py      stage-by-stage diff of C++ vs Python intermediates
  compare_ref.py   vertex-by-vertex diff of a result against the Python .npy
  test_wasm.js     drives build-wasm/simcpp.js through the C API under node
  data/            dumped garments (trousers.bin, shirt.bin)
  ref/             the Python reference runs
```

### Input format

`dump_garment.py` writes a flat little-endian binary: magic `SIMCPP01`, then
eight `int32` (`n`, `n_faces`, `n_pairs`, `n_cyl`, `n_sph`, `has_mu`, `has_nu`,
`recenter`), then `faces (M,3) i32`, `wid (n) i32`, `panel_of_face (M) i32`,
`rest (n,2) f64`, `P0 (n,3) f64`, `pairs (K,2) i32`, optional `mu (n) f64`,
optional `nu (n) f64` + `anchor (n,3) f64`, `cyl (NC,7) f64` (p0, p1, r) and
`sph (NS,4) f64` (centre, r).  A `.json` sidecar carries the counts.

## Building

Eigen is header-only and already on this machine at `C:\repos\eigen`
(**5.0.1-dev**, `EIGEN_WORLD/MAJOR/MINOR/PATCH = 3/5/0/1`, git
`63b079a1cf37d92ab5fe32062783c9eedd3e74b2`).  It is used in place, read-only —
nothing was vendored or downloaded.  Note that Eigen 5 needs a real C++17
standard library (`std::invoke_result_t`, `std::destroy_at`); the GCC 5.3 that
`conda install m2w64-toolchain` provides is too old and ICEs on it.

### Native

```sh
g++ -std=c++17 -O3 -DEIGEN_NO_DEBUG -static \
    -I src -I C:/repos/eigen \
    src/sim.cpp src/io.cpp src/main.cpp -o build-native/simcpp.exe
```

or `./build.sh native`.  `CMakeLists.txt` covers both targets
(`cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`, or
`emcmake cmake -B build-em`), but **it is untested** — this machine has no
`cmake`.  The command lines above and in `build.sh` are the ones actually used.

### Emscripten / WASM

```sh
export EMSDK=/c/repos/emsdk
export EMSDK_PYTHON="C:/repos/emsdk/python/3.13.3_64bit/python.exe"
export PATH="/c/repos/emsdk:/c/repos/emsdk/upstream/emscripten:$PATH"

em++ -std=c++17 -O3 -DEIGEN_NO_DEBUG -I src -I /c/repos/eigen \
  src/sim.cpp src/io.cpp src/wasm.cpp -o build-wasm/simcpp.js \
  -lembind -sMODULARIZE=1 -sEXPORT_NAME=createSim -sALLOW_MEMORY_GROWTH=1 \
  -sINITIAL_MEMORY=268435456 -sSTACK_SIZE=1048576 \
  -sEXPORTED_FUNCTIONS='["_sim_build","_sim_solve","_sim_n_hinges","_sim_mono_violations","_sim_seconds","_sim_energy","_sim_free","_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["HEAPF64","HEAP32","HEAPU8","ccall","cwrap"]' \
  -sENVIRONMENT=web,worker,node
```

Use `em++`, not `emcc` — `emcc` will not link the C++ runtime.
`build.sh wasm` also builds `simcpp_node.js`, the same CLI as the native binary
with `-sNODERAWFS=1`, which is handy for running the dumps under node.

## Running

```sh
PY="$USERPROFILE/miniconda3/envs/agentic-ai/python.exe"

# host side: turn a garment directory into the solver's input
"$PY" dump_garment.py --garment <gcd_data dir> --out data/trousers.bin \
      --body --mu 0.02 --amp 0 --sym

# native
./build-native/simcpp.exe --in data/trousers.bin --out out/native_trousers --per-lambda 30

# wasm, through the JS C API
node test_wasm.js data/trousers.bin out/wasm_api_trousers.npy
```

The JS surface is

```js
const Mod = await createSim();
Mod._sim_build(n, nFaces, pFaces, pWid, pPanelOfFace, pRest,
               nPairs, pPairs, pMu, nCyl, pCyl, nSph, pSph);   // -> n_hinges
Mod._sim_solve(pP0, nLadder, pLadder, w0, w1, factor,
               itersPerStage, perLambda, maxIter, tol, recenter, verbose, pOut);
```

with `Mod.simBuild` / `Mod.simSolve` as embind mirrors.  All pointers are wasm
heap offsets; `test_wasm.js` is a complete worked example.

## Validation

Reference (both garment directories read-only):

```sh
cd ../rigid-clothes-simulation
"$PY" run_garment.py --garment <dir> --outdir <scratch> --amp 0 --body --sym \
      --mu 0.02 --per-lambda 30 --lam-start 1e-3 --lam-stop 1e-5 --tag ref
```

**Stage by stage** (`selftest.py`) — each intermediate recomputed in Python from
the same dump and diffed against what `simcpp --selftest` wrote:

```
shape_gradients G/area     max abs 8.9e-16 / 0.0      (area bit-identical)
build_hinges hinges        0 integer mismatches
hinge_stencils Kb/wb       bit-identical
deformation_gradients      max abs 5.7e-14
best_rotations R           max abs 3.0e-14
best_rotations sigma       max abs 7.5e-09   (see "suspect" below)
body clamp Z               bit-identical
arap_rhs + mu Z            max abs 2.3e-14
solve_global (one step)    max abs 1.5e-11
```

**End to end** (`compare_ref.py`), trousers / shirt:

```
max |diff|        9.94e-11 cm      2.99e-09 cm
p50 |diff|        6.43e-11 cm      7.98e-11 cm
mono_violations   0 == 0           0 == 0
seam_gap_max      rel 2.1e-10      rel 4.4e-10
max_sigma_dev     rel 4.2e-12      rel 1.1e-11
E_arap / E_bend   rel <6e-12       rel <1.2e-11
```

That residual is roundoff accumulated over 350 iterations, not a modelling
difference: at the same iteration count the printed per-stage energies agree to
all eight digits the log shows.

## Runtime

Same run (350 iterations, 5 factorisations), same machine:

| | trousers (10.2k v, 480 pairs) | shirt (8.4k v, 704 pairs) |
|---|---|---|
| Python + SciPy | 13.90 s | 15.19 s |
| **native C++** (best of 3) | **6.00 s** | **6.89 s** |
| WASM under node | 10.91 s | 15.64 s |

Native is ~2.2x the Python.  WASM is ~1.8x/2.3x the native, which is the usual
WASM penalty on dense float work — it lands roughly at Python's speed here, so
the win in the browser is that it runs at all, not that it is fast.

### The Woodbury dense operator — measured, not assumed

The note that motivated this: in Python, `np.linalg.inv` on the dense
`(n_pairs x n_pairs)` Woodbury matrix dominated runtime, and replacing it with a
per-application triangular solve made things *slower*; eigendecomposing once per
factorisation instead is what is committed.  `--wood eigh|inverse` measures the
same choice in C++:

| | trousers | shirt |
|---|---|---|
| `--wood eigh` (default) | 6.00 s total, **0.25 s** in the operator | 6.89 s, **0.82 s** |
| `--wood inverse` | 6.34 s total, 0.55 s in the operator | 7.69 s, 1.70 s |

Same conclusion as Python, for the same reason — the eigendecomposition is paid
5 times (once per `lambda_b` rung) where the inverse is paid ~25 (once per
distinct `w_s`) — but the stake is much smaller here: 4–12% of total runtime
rather than a majority of it, because `n_pairs` is only 480/704 on these two
garments and Eigen's dense kernels are far quicker than NumPy's per-call
overhead at that size.  Both give identical results.

## Things in the Python that were ambiguous or look suspect

1. **Dead code in `build_hinges`.**  The lines

   ```python
   b0, b1, b3 = fB[cB], fB[(cB + 1) % 3], fB[(cB + 2) % 3]
   if wid[b0] != wid[a0]:
       b0, b1 = b1, b0
   ```

   compute and swap `b0`/`b1`, which are then never read — the orientation that
   actually matters is redone a few lines later on `ib`.  Only `b3` survives.
   Harmless, but it reads as if it were doing something.  The port keeps only
   the `ib` swap.

2. **`best_rotations` loses ~8 digits on the singular values of near-conformal
   triangles**, and `max_sigma_dev` is a gate.  `disc = sqrt(tr^2 - 4 det)`
   cancels catastrophically when the two singular values are close (exactly the
   case at a rigid placement, where both are 1): a 1e-16 relative error in `tr^2`
   becomes ~2e-8 in `disc` and ~7e-9 in each `sigma`.  This is inherent to the
   Python formula, not introduced by the port — the selftest above shows both
   implementations landing 7.5e-9 apart on sigma while `R` itself agrees to
   3e-14, because `R` never goes through `disc`.  It does not matter at the
   reported gate magnitudes (0.12 and 0.91), but a run whose `max_sigma_dev`
   gate sat near 1e-8 would be reporting noise.  `sig[:,1]` could be recovered
   as `sqrt(det)/sig[:,0]` instead if that ever mattered.

3. **Non-manifold edges are silently dropped.**  `build_hinges` skips any welded
   edge with `len(lst) != 2`, so an edge shared by three or more faces
   contributes no bending term at all rather than raising.  The port reproduces
   this exactly, but it is a silent failure mode: a mesh defect shows up only as
   a missing bending term, never as an error.

4. **`geometric_schedule` cannot terminate if `factor <= 1`** — `while w < w1:
   w *= factor` spins forever.  Not reachable from `run_garment.py`, and the
   port has the same shape.

5. **The convergence break only fires on the last stage** (`and stage ==
   len(schedule) - 1`), so the ~20 warm-up stages of rung 0 always burn their
   full `iters_per_stage` even when converged.  Deliberate-looking, but worth
   knowing when reading iteration counts.

6. **`initial()`'s `--sym` branch** contains a `... if False else ...` and a
   `np.ones((24,3)) * 0 + 1.0`, i.e. leftovers from an edit.  With `--amp 0` the
   function returns before reaching it, which is the path used here.  Not ported
   (host side).
