# HANDOFF — seam-correspondence

Written across two sessions. Read this first, then the two component-level docs it points
at (§11). Everything below is either *measured on this machine* or explicitly flagged as
unverified.

Repo: `C:\repos\seam-correspondence` · Branch: **`feature/demo`**

---

## 0. Hard constraints (from the user, still in force)

- **Python is exactly one interpreter.** `$PY = "$env:USERPROFILE\miniconda3\envs\agentic-ai\python.exe"`,
  invoked as `& $PY ...`. **Never `conda activate`.**
- **The GCD data folders are read-only.** `C:\Users\POMCHECKER\Downloads\garments_5000_0\...\data`
  and `C:\Users\POMCHECKER\gcd_data`.
- **`autosew/HANDOFF.md` is authoritative** for the AutoSew reproduction. If a prompt conflicts
  with it, follow the doc and report the conflict. Its §2 hyperparameters are paper-specified:
  **do not change them.** (§4 below is a deliberate, user-approved departure — read it.)
- **`rigid-clothes-simulation/` is the user's own code**, with its own `HANDOFF.md` and `README.md`.
  Treat as read-only unless asked. (I once overwrote its README — see §9.)
- Emscripten `EMSDK=/c/repos/emsdk`, `EMSDK_PYTHON=C:/repos/emsdk/python/3.13.3_64bit/python.exe`.
  Eigen at `C:/repos/eigen`, header-only, not vendored.
- Do not `npm install` new packages without asking. `fbx-parser@2.1.3` and `dxf-parser` are
  installed and the user chose both — **use them; do not hand-roll FBX or DXF parsing in the
  web path.**

---

## 1. What this is

Reproduction of **AutoSew** (arXiv 2602.22052, WACV 2026): a 2D sewing pattern in, the stitch
correspondence between panel edges out. Around that core sit an assembly simulator, a browser
demo, and a DXF/FBX industrial input path.

```
specification.json ──► parse ──► 24-dim edge features ──► GraphSAGE + Sinkhorn ──► stitch pairs
   or CLO DXF                                                                           |
                                                                                        v
                                                                     ARAP shell assembly ──► 3D mesh
```

---

## 2. Component status

| Component | Where | State | Verified how |
|---|---|---|---|
| AutoSew model | `autosew/` | **Working.** r1 test TF1 **0.9490**, GSP 0.7203 | full test-set eval |
| Assembly solver (Python) | `rigid-clothes-simulation/` | **Working**, user's code | README gates pass |
| Solver unit tests | `rigid-clothes-simulation/test_assembly.py` | **78 tests, OK** | 54 s with data |
| C++ port | `simcpp/` | **Working**, 2.2× Python | matches reference to 1e-11 cm |
| WASM port | `simcpp/build-wasm-simd/` | **Working in a real browser** | §3 |
| Browser demo | `webdemo/` | **Working**, all three input modes | headless Chrome + CDP |
| vtk.js 3D views | `webdemo/src/{Viewer3D,WeldGT,SimViewer}.tsx` | **Working** | software WebGL2 only |
| DXF input path | `dxfcheck/`, `webdemo/src/lib/parseDxf.ts` | **Working end to end** | §5 |
| CLO seam ground truth | `dxfcheck/export_weld_gt.py` | **Solved and eye-verified** | §6 |
| 2D↔3D panel matching | — | **Blocked on mirror pairs** | §7 |

### Model results

| run | train garments | encoding | panel ids | test TF1 | GSP |
|---|---|---|---|---|---|
| r0 | 2,760 | tagged | index | 0.8750 | 0.4058 |
| **r1 (shipped)** | 24,024 | tagged | index | **0.9490** | 0.7203 |
| paper | 102,660 | tagged | index | 0.9706 | 0.806 |
| s12_tagged | 2,000 | tagged | index | 0.8324 | 0.205 |
| s12_sagitta | 2,000 | sagitta | index | 0.8191 | 0.155 |
| rand_tagged | 2,000 | tagged | **random** | 0.7468 | 0.050 |
| **rand_sagitta** | 2,000 | sagitta | **random** | **0.7501** | 0.105 |

r1 is the shipped model: `webdemo/public/model/autosew.onnx`, 7.1 MB, opset 17, epoch 5.
The four small runs are the controlled 2×2 of §4 — same 2,000 garments, 12 epochs, nothing
else differing.

---

## 3. Simulation — measured, and now seen in a browser

350 iterations, 5 factorizations, `--per_lambda 30`, identical input:

| | Python | native C++ | wasm baseline | wasm `-msimd128 -msse4.2` |
|---|---|---|---|---|
| trousers (10,186 v / 480 pairs) | 13.90 s | **6.28 s** | 11.03 s | **10.04 s** |
| shirt (8,377 v / 704 pairs) | 15.19 s | **7.20 s** | 14.56 s | **11.46 s** |

**The wasm build runs in a real browser** (Chrome 152, headless, CDP-driven): 350 iters,
`monoViolations 0`, wall 10.37 s, `max|diff|` vs the Python reference **9.95e-11 cm**, and
bit-identical to the Node run of the same wasm. SIMD proven three ways: the shipped
`webdemo/public/sim/simcpp.wasm` is sha256-identical to `simcpp/build-wasm-simd/`, a rebuild
with `-msimd128 -msse4.2` reproduces those bytes, and an Eigen probe reports
`EIGEN_VECTORIZE=yes, packet<double>=2`. `wasm-dis` counts 1826 `f64x2` ops (0 in the
non-SIMD build).

### The SIMD flag finding (non-obvious — don't lose this)

**`-msimd128` alone does nothing for Eigen.** Eigen has no wasm-SIMD backend, so packet size
stays 1. You must *also* pass `-msse2` or higher to engage emscripten's SSE→wasm128 emulation:

| flags | `EIGEN_VECTORIZE` | packet size (double) |
|---|---|---|
| *(none)* | no | 1 |
| `-msimd128` | **no** | 1 |
| `-msimd128 -msse2` | yes | 2 |

Net gain is only −9% / −19%, far short of native's 2.2×. The gain being larger on the garment
with more seam pairs confirms the **dense Woodbury operator** is what SIMD helps. The rest is
SparseLU indirect addressing, which SIMD cannot touch. **~10 s is the floor for this
structure.** Do **not** add `-ffast-math`: the monotonicity gate in `solve()` and the 1e-300
guards in `best_rotations` depend on IEEE semantics.

`build.sh` has **not** been updated with the SIMD flags, and `simcpp/build-wasm-simd/` and
`build-wasm-simdonly/` are **not** covered by `.gitignore` (only `build-wasm/` and
`build-native/` are). Extend the ignore rule or delete them before committing.

Known nit: the browser solve is synchronous on the main thread and freezes the page for ~10 s.
A Worker is the natural fix.

---

## 4. The domain gap — the biggest finding, and it was not what we expected

Predictions on the real CLO DXF were poor for every model. Two causes were measured, and
**the one everybody would guess last is 15–20× larger than the one that looks obvious.**

### 4.1 Panel ordering (dominant)

GCD's `panel_u` (dim 23) is the panel's position in the specification file — which is
*generation order*, so related panels sit next to each other. A DXF has no such thing: the id
is block order, an arbitrary number. Perturbing `panel_u` at inference on 200 test garments:

| model | baseline TF1 | shuffled | zeroed | reversed | CLO prediction stability |
|---|---|---|---|---|---|
| r1 | 0.9086 | 0.7105 | **0.6110** | 0.7333 | Jaccard 0.49–0.62 |
| s12_tagged | 0.8346 | 0.6277 | 0.5919 | 0.6189 | 0.41–0.52 |
| s12_sagitta | 0.8210 | 0.6452 | 0.6273 | 0.6473 | 0.39–0.62 |
| **rand_sagitta** | 0.7552 | 0.7588 | **0.7577** | 0.7532 | **0.96–1.00** |

r1's GSP collapses 0.445 → 0.085 when `panel_u` is zeroed — an 81% drop.

**So the shipped models borrow 0.19–0.30 TF1 from a signal that does not exist in a DXF.**
Their real DXF-regime performance is ~0.59–0.63, not 0.82–0.91.

**The fix already existed in the codebase:** `panel_id_mode="random_norm"`. Training with
per-sample shuffled ids forces the model onto geometry. `rand_sagitta` is then *completely*
order-independent — ±0.004 TF1 across every perturbation, and on the CLO DXF zeroing or
reversing the ids changes **not one prediction** (Jaccard 1.000).

| | nominal TF1 | **TF1 in the DXF regime** |
|---|---|---|
| `index_norm` models | 0.82–0.91 | 0.59–0.63 |
| `random_norm` models | 0.7501 | **0.7501** |

Give up ~0.07 nominal on GCD, gain ~0.13 where it actually matters, and the prediction becomes
deterministic. Caveat: `PatternDataset.from_dir` shuffles **once** at dataset build, not per
epoch, so this is randomised assignment rather than true per-epoch augmentation. The test
split gets its own assignment, so the test number is honest.

Note `autosew/HANDOFF.md` §3 records *"u 인코딩(idx/37 vs raw)도 무차이"* — that ablation
compared **encodings of the id**, not whether the **ordering carries signal**. Different
question; first measured 2026-09-03.

### 4.2 Curvature encoding (secondary, but it is what makes DXF input possible at all)

Dims 7..17 are a **tagged union**: `k_t` selects the meaning of ten parameter slots
(circle → `[radius, large_arc, sweep]`; quadratic → one control point; cubic → two). Fine
when `k_t` comes from a spec. Catastrophic when it must be *refitted* from a polyline: a flip
redefines eleven dimensions at once. (The old note that dim 7 is a "one-hot" is wrong — it is
a single raw ordinal 0..5, which is worse, since it also imposes a false ordering.)

**On the real CLO DXF the refit does not merely flip — it fails outright.** Every one of the
28 curved edges falls through to `KT_CUBIC` with a cubic residual 10× to 600× outside
tolerance, because a CLO boundary run between turn points is a free-form curve that no single
Bézier or arc represents. The synthetic `spec_to_dxf` round trip never reproduced this: it
emits single-Bézier edges, so its F1 of 0.783 was **optimistic**.

`curvature_encoding="sagitta"` (`autosew/autosew/curves.py`) replaces dims 7..17 with K=11
signed sagitta values — perpendicular deviation from the chord at uniform arclength fractions,
divided by chord length. No type is ever decided. Measured:

| | curvature block max&#124;diff&#124;, spec vs DXF | other 13 dims |
|---|---|---|
| tagged | **1.101** | 6e-8 |
| sagitta | **7.7e-4** | 6e-8 |

Sampling-density behaviour is the clean statement of why: resampling the same curve at
3→128 points, sagitta's error falls monotonically (0.120 → 4.3e-5) while tagged sits at
**1.0 regardless** — more samples never fix a categorical flip. CLO samples adaptively
(straight edges get exactly 2 points, curved ones 16–88; correlation between point count and
`|sagitta|max` is +0.55), so vertex count is not itself a problem.

Distribution shift of the CLO features against 300 GCD training garments:

| encoding | worst shift | CLO edges outside GCD's 1–99% |
|---|---|---|
| tagged | `k4` **1.45 σ** | **16.7%** |
| sagitta | `panel_u` 0.50 σ | **0.0% on all 24 dims** |

**With sagitta there is no measurable per-feature domain gap.** Structure matches too:
10 panels (GCD mean 10.3), 6.0 edges/panel (GCD 6.97, range 4–12 inside GCD's 3–33).

### 4.3 The controlled 2×2

2,000 garments × 12 epochs, nothing else differing:

| | `index_norm` | `random_norm` |
|---|---|---|
| tagged | 0.8324 | 0.7468 |
| sagitta | 0.8191 | **0.7501** |

Once the ordering crutch is gone the two encodings are **equal** (sagitta +0.003), and sagitta
is the less damaged by removing it (−0.069 vs −0.086) with a better TP/TR balance
(0.749/0.751 vs 0.786/0.712) and double the GSP. **The industrial track is
`curvature_encoding=sagitta` + `panel_id_mode=random_norm`.** sagitta's value is not accuracy
— it is that DXF input becomes possible at all.

### 4.4 Track split (user-approved departure from `autosew/HANDOFF.md` §2)

The paper track (defaults: `tagged` + `index_norm`) is untouched and still reproduces. The
industrial track is two config flags, same codebase, separate checkpoints. Everything else —
GNN, Sinkhorn, loss, all hyperparameters, and the **training GT, which comes from the GCD
spec's `stitches` as before** — is identical. FBX/OBJ welds are *evaluation* ground truth for
the one real garment, never training data.

---

## 5. A real parser bug, found and fixed

`gcd_parser.py` transformed quadratic/cubic control points when a panel is traversed backwards
but left circle params `[radius, large_arc, sweep]` untouched. Traversing an arc backwards
flips its sweep, so for arcs on reversed panels dim 10 described the *original* direction while
dims 0..3 described the *reversed* one. Measured on the sagitta round trip: reversed arcs were
off by **0.554**; with the flip, **5.4e-9**.

Frequency: arcs are ~7% of edges and about half sit on reversed panels, so **~3.5% of edges
carried one wrong boolean**. r1 was trained with the bug, so r1 and the current parser now
disagree slightly; r2 resolves it. **`webdemo/src/lib/parseSpec.ts` deliberately still has the
bug**, to stay bit-consistent with the deployed r1 — fix it together with the r2 model swap.

---

## 6. The CLO seam ground truth — solved

The user exported the same garment twice, with CLO's weld option off and on
(`clo_example/unweld.obj`, `weld.obj`). **CLO's own weld operation is the correspondence.**
Welding merges coincident boundary vertices without moving them, so a group of unweld vertices
sharing a position is a sewn point.

- unweld 44,696 v / weld 44,455 v → **CLO merged 241 vertices**.
- Clustering the garment panels' vertices by **bit-exact** position gives 224 cross-panel
  groups with `sum(len−1) = 241` — **exactly CLO's merge count**. That is the validation gate,
  and `dxfcheck/export_weld_gt.py` refuses to export unless it passes.

**This settles the counting convention the previous handoff left open.** Candidates were 265
(all pairs), 241 (`sum(n_verts−1)`) and 240 (`sum(n_components−1)`); the earlier doc picked
240. **The answer is 241.** The discriminator is one `(5 verts, 4 components)` group whose
extra vertex is a within-panel weld that `sum(n_components−1)` misses.

Cross-validated: group breakdown `214×(2v,2c), 4×(3v,3c), 5×(4v,4c), 1×(5v,4c)`, panel sizes
`942/933/925/901/132/132/111/111/24/20`, 4,231 verts, 7,855 tris — every number matches the
independent FBX analysis exactly.

**These are real seams, not drape contact.** Welded pairs are at distance 0 (bit-exact); the
nearest *non*-welded cross-panel vertex pair is **10.98 mm** away, with nothing in between. A
proximity weld would leave a continuum of sub-millimetre distances. There is none.

### Vertex welds convert cleanly to edge-level seams

Walking each panel's 3D boundary loop and cutting it into maximal runs sharing a partner panel,
run lengths split into two disjoint groups with **nothing in between**: 1 vertex, or 5–23.
The 1-vertex runs are corners where 3+ panels meet (partner set of size ≥2) — seam *endpoints*,
not seams. Every 5–23v run joins **exactly one** partner panel.

So the garment's seam structure **is edge-like**, which is what the model's whole-edge
assumption needs. Conversion rule: drop the 1-vertex runs; each remaining run is one side of an
edge-level seam. No threshold to tune. Sanity check: the two big bodice panels are joined by
**four** runs (18v, 5v, 5v, 18v) = left/right shoulder + left/right side seam.

**Corrected count: ~18 real seams, not 21.** Three of the 21 panel pairs from raw vertex
counting only touch at a corner.

`webdemo/src/WeldGT.tsx` renders this: panels exploded outward (sewn vertices coincide in the
drape, so an unexploded seam line has zero length), weld lines drawn between them. **The user
has eye-verified it.**

---

## 7. What is still blocked: 2D↔3D panel matching

No shared ids exist anywhere. Checked and ruled out:

- **DXF `# NN` TEXT labels are not correspondence.** They are per-document corner numbering.
  Of 29 labels, 24 sit in a single piece; the 5 shared ones are shared only between **mirror
  duplicate pieces** (`8_M`↔`11_M`, `6_M`↔`10_M`), and repeats within a piece are its own
  left-right symmetry (the palindromic `9,10,…,15,…,10`). No body↔sleeve sharing exists.
  The previous handoff's conclusion was right; the apparent cross-piece lead is a mirror
  artifact. **Layer 14 is a duplicate of layer 1**, not sewing info. Modelspace TEXT is CLO
  metadata (`UNITS: METRIC` confirms mm).
- **No FBX carries panel names.** `draped_seperated.fbx`, `seperated.fbx` and `panel_sep.fbx`
  all hold the garment as a *single* mesh (`panel_sep`, 4231 v / 7855 polys). Greps for
  `Pattern_634078`, `12_M` return **0 hits**.
- **The OBJ `usemtl` names are fabrics** (2 of them), not panels.

Shape matching (perimeter + area, both near-invariant under draping) resolves the **groups**
cleanly — {`3_M`,`9_M`}↔{942,933}, {`Pattern_*`}↔{925,901}, {`10_M`,`6_M`}↔{132,132},
{`8_M`,`11_M`}↔{111,111}, `12_M`↔24, `13_M`↔20 — but **cannot** resolve within a mirror pair:
assignment margins are 0.0000 or negative, because a left and a right sleeve have identical
perimeter and area *by construction*. This is HANDOFF's documented "identical panels" failure
mode reappearing in the matching problem.

Remaining options: break the tie by 3D left/right vs 2D layout left/right (an assumption, not
a guarantee), or evaluate up to the mirror symmetry. **Not yet decided.** Everything
downstream — edge-level scoring of the CLO prediction, and edge-to-edge seam rendering in 3D —
waits on this.

Note the 3D perimeters run 4–7% larger than the DXF (207 vs 196 cm), worst on thin strips
(`12_M` 15%). Drape stretch plus boundary sampling; it does not affect group matching.

---

## 8. Dataset

Official split at
`C:\Users\POMCHECKER\gcd_data\GarmentCodeData_v2_official_train_valid_test_data_split.json`
(102,660 train / 6,270 valid / 6,265 test).

Downloaded into `C:\Users\POMCHECKER\gcd_data\{train,valid,test}\part<N>\<garment>\`. The
download pipeline **finished on 2026-09-03**; all tarballs are deleted (0 left on disk):

| split | garments | official | progress |
|---|---|---|---|
| train | **99,666** | 102,660 | 97.1% |
| valid | 6,096 | 6,270 | 97.2% |
| test | 6,078 | 6,265 | 97.0% |

**`part20` is missing.** 35 part dirs are present — `part0`–`part19` and `part21`–`part35`. The
~2,990 missing train garments are exactly one part's worth, so the shortfall is entirely part20,
not a partial extraction elsewhere. Re-fetch it if you want the complete official split;
otherwise the set is usable as-is at 97%.

**train/valid keep the spec JSON only; test keeps every asset including `.ply`** (user's
decision, so 3D checking stays possible).

GCD v2 is served as **one `data.tar.gz` per part (~4.7 GiB)** on libdrive.ethz.ch — *not* as
per-garment WebDAV folders. `autosew/scripts/fetch_gcd_specs.py` is committed but **does not
work against this share** (its PROPFIND walk returns zero garments). Use
`extract_specs_from_tar.py` (streams via `tarfile.open(path, "r|gz")`) plus `organize_split.py`.

Trap in `organize_split.py`: `if sum(n.values()) == 0: sys.exit(1)` reports a **false failure**
when a part is already fully organised.

`train.py` supports explicit split dirs via `--val_dir` / `--test_dir`; when either is given,
`--val_frac` / `--test_frac` are ignored.

---

## 9. Traps and mistakes already made — do not repeat

- **`train.log` looks empty.** stdout is block-buffered when redirected. Monitor
  `runs/<name>/history.jsonl` instead.
- **cProfile lied.** A `cho_factor`/`cho_solve` "optimisation" turned out **46% slower**;
  the profiler inflated the run and misattributed the cost. **Always A/B on wall clock, two
  runs each, alternating.**
- **`sed`-based path substitution failed silently**, and an "r1" evaluation actually re-ran
  r0's checkpoint. Rewrite files in Python with an `assert old in src` anchor.
- **Root `.gitignore:17` has a `lib/` rule** that swallowed `webdemo/src/lib/`.
  `!webdemo/src/lib/` is now present. **Run `git status --short` and confirm new files are
  visible before committing.**
- **A new feature-affecting config must be added to `train.py`'s `FEAT_KEYS`**, or a cache
  built under one setting is silently reused under another. `curvature_encoding` and
  `sagitta_samples` are now in the list; the next one must be too.
- **`ONNX EyeLike` has no onnxruntime kernel.** Replaced with an arange comparison, and the
  in-place `Cbar` slice writes with `torch.cat`, as a *runtime patch in the export script*.
- **Edge midpoints.** `sample_edge` returns only `[p0, p1]` for a straight edge, so
  `pts[len//2]` lands on the *end vertex*. `render.ts` and the 3D path use arclength
  interpolation. `visualize.py`'s `mid()` is still imprecise — **reported, not changed**.
- **Heredocs break on this content**, and `python` is not on PATH in the Bash tool (only
  `& $PY`). Use the Write/Edit tools for files, not heredocs.
- **Background `bash script.sh &` dies when the tool call ends.** Launch it as the foreground
  command of a `run_in_background` call.
- **The dev server is fine; a bare `vite` is not.** A background launch once failed with exit
  127, which looks like a broken toolchain but is not. `vite` exists only as
  `webdemo/node_modules/.bin/vite`, never on `PATH`. Always `cd webdemo` and run
  `npm run dev` (verified 2026-09-03: ready in 801 ms, HTTP 200 on `--port 5199 --strictPort`).
- **numpy 2.x removed `ndarray.ptp()`** — use `np.ptp(a)`.
- **`evaluate_batch(logP, batch, cfg, acc)`** — accumulator is the *last* argument.
- Use `$PY` and explicit `encoding="utf-8"` for every file rewrite (cp949 otherwise).
- **I overwrote `rigid-clothes-simulation/README.md`** because I checked whether it was tracked
  *before* the user's rebase and never re-checked. **Re-check tracked status immediately before
  writing to any file you did not create.**

---

## 10. Suggested order of work

1. **r2 on the industrial track**: `--set curvature_encoding=sagitta panel_id_mode=random_norm`
   over the 99,666 downloaded train garments, 18 epochs. r1 reached 0.9086 from 24k with the
   ordering crutch; this should land well above 0.7501 and — unlike r1 — **that number holds on
   DXF input.** Then export to ONNX and swap it into the demo, fixing `parseSpec.ts`'s sweep
   bug in the same change (§5).
2. **Decide the mirror tie-break** (§7). It unblocks edge-level scoring of the CLO prediction
   and edge-to-edge seam rendering in 3D — the last two things standing between us and a
   measured F1 on real industrial data.
3. Consider **per-epoch** id shuffling (`dataset.py`) rather than once at build; §4.1.
4. Fold the SIMD flags into `simcpp/build.sh`, gitignore the two extra build dirs (§3).
5. Move the browser solve to a Web Worker so the page stops freezing for 10 s.

---

## 11. Component docs

- `autosew/HANDOFF.md` — **authoritative** for the model. §2 fixed hyperparameters, §3 gap
  decisions, §5-1 the identical-panel behaviour, §6 prescriptions. Note §4 above is a
  user-approved departure from its §2, and §4.1 above corrects the reading of its §3.
- `rigid-clothes-simulation/HANDOFF.md` and `README.md` — the user's own. Solver gates:
  `mono_violations == 0`, `seam_gap_max < 1e-3`, `max_sigma_dev < 1.0`, finite energies.
- `simcpp/README.md` — build invocations and the Woodbury measurement writeup.

---

## 12. Short version

A reproduction of AutoSew (WACV 2026) — 2D sewing pattern in, stitch correspondence out —
extended to take **real CLO exports** (DXF + FBX/OBJ) instead of only the synthetic
GarmentCodeData specification format. Reproduction reaches **TF1 0.9490** against the paper's
0.9706 at a quarter of the training data. Around it: an ARAP assembly solver ported to C++ and
WebAssembly, and a browser demo that runs the whole pipeline client-side.

The headline result is that **making it work on real CLO data was not a modelling problem.**
Two concrete defects in how the input is represented accounted for everything, and the
larger of the two was invisible until it was measured.

<details>
<summary><b>1. The model was quietly using a signal that does not exist in a CLO export</b></summary>

GarmentCodeData numbers its panels in *generation* order, so related panels sit next to each
other. That number is fed to the network as a feature. A DXF numbers panels by block order —
arbitrary.

Zeroing that one feature at inference: **TF1 0.9086 → 0.6110**, GSP **0.445 → 0.085**. So
roughly a quarter of the reported accuracy was borrowed from an artifact of the synthetic
dataset, and it evaporates on real data.

Training with randomised panel ids removes the dependence completely: ±0.004 TF1 across every
perturbation, and on the CLO file, zeroing or reversing the ids changes **not one prediction**.
Nominal accuracy drops ~0.07; accuracy *in the regime that matters* rises ~0.13, and the
prediction becomes deterministic.

</details>

<details>
<summary><b>2. The curvature features could not survive a DXF at all</b></summary>

The paper encodes an edge's curve as a type tag plus ten slots whose meaning the tag selects
(radius+flags for an arc, control points for a Bézier). A DXF stores curves as sampled points,
so the type must be re-estimated — and a wrong guess silently redefines eleven feature
dimensions at once.

On the real CLO file it is worse than a wrong guess: **every one of the 28 curved edges fails
to fit**, by 10× to 600× the tolerance, because a CLO boundary between corner points is a
free-form curve that no single arc or Bézier represents. Our own synthetic DXF round-trip had
never shown this — it emits single-Bézier edges, so it flattered the result.

Replacing those dimensions with a **sagitta profile** — signed deviation from the chord at 11
uniform positions, normalised by chord length — removes the type decision entirely. Round-trip
error drops from **1.101 to 7.7e-4**, and CLO's features land **inside** the training
distribution on all 24 dimensions (0.0% of edges outside the 1–99% range, versus 16.7% before).

The clean way to state why: sagitta's error is *continuous* in sampling density (0.120 at 3
points → 4.3e-5 at 128), while the tagged encoding sits at ~1.0 **no matter how many samples
you give it** — more data cannot repair a categorical flip.

</details>

<details>
<summary><b>3. Ground truth from CLO's own weld, validated against CLO's own count</b></summary>

Exporting the same garment twice — weld off and weld on — turns CLO's weld operation into the
seam ground truth: 44,696 → 44,455 vertices means **241 merges**. Clustering the unwelded
panels by bit-exact position reproduces **exactly 241**, which is the validation gate.

That also settles a counting convention an earlier analysis had guessed wrong (241, not 240 —
the discriminator is a single 5-vertex/4-panel group containing a within-panel weld).

These are genuine seams, not panels touching in the drape: welded pairs are at distance 0 and
the nearest *non*-welded cross-panel pair is **10.98 mm**, with nothing in between.

Walking each panel's 3D boundary, the welds fall into runs of either **1 vertex** (a corner
where 3+ panels meet) or **5–23** (a seam) — nothing in between, so converting vertex welds to
edge-level correspondence needs no tuned threshold. The two bodice panels are joined by exactly
four runs: two shoulders, two side seams.

</details>

<details>
<summary><b>4. What is measured versus what is claimed</b></summary>

| claim | how it was checked |
|---|---|
| WASM solver correct in-browser | headless Chrome + CDP; `max&#124;diff&#124;` **9.95e-11 cm** vs the Python reference, `monoViolations 0`, bit-identical to Node |
| SIMD actually enabled | sha256 match to the SIMD build, reproducible rebuild, Eigen probe (`packet<double>=2`), 1826 `f64x2` opcodes vs 0 |
| `-msimd128` alone does nothing | Eigen has no wasm-SIMD backend; needs `-msse2`+ to engage the SSE→wasm128 emulation. Probed directly |
| encoding choice is what matters | controlled 2×2, 2,000 garments × 12 epochs, one variable at a time |
| seam GT is real | reproduces CLO's own merge count exactly; refuses to export otherwise |

A parser bug was also found and fixed along the way: arcs on reversed panels kept their
original sweep flag, so ~3.5% of edges carried one inconsistent boolean. Off by 0.554 before,
5.4e-9 after.

</details>

<details>
<summary><b>5. What is still open</b></summary>

Matching the 10 DXF pieces to the 10 draped 3D panels has no shared identifier anywhere — not
in the DXF labels, not in any of the three FBX files, not in the OBJ materials. Shape
invariants resolve the size *groups* cleanly but **cannot** separate a left sleeve from a right
one: perimeter and area are identical by construction, and the assignment margins are 0.

That is the same "geometrically identical panels" limitation the paper reports for the model
itself, showing up again one level up. Until it is broken — by 3D position, or by evaluating
up to the mirror symmetry — the CLO prediction cannot be scored edge-by-edge.

</details>
