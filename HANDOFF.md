# HANDOFF — seam-correspondence

Two audiences, one document.

**§A is for an interview-prep agent**: what the work is, what it found, what can be
demonstrated live, and — most importantly — where the honest limits are. Everything in it
is measured; nothing is aspirational.

**§B onward is for whoever continues the engineering.** Constraints, component status,
traps already hit, and what to do next.

Repo: `C:\repos\seam-correspondence` · Branch: **`feature/demo`** · Last commit `4570e1e`

---
---

# §A — For interview preparation

## A1. What this is, in three sentences

A reproduction of **AutoSew** (WACV 2026) — a 2D sewing pattern in, the stitch
correspondence between panel edges out — extended so it can take **real CLO exports** (a
DXF of the flat pattern, FBX/OBJ of the drape) instead of only the synthetic
GarmentCodeData the paper uses. Around the model sit an ARAP assembly solver ported to
C++ and WebAssembly, and a browser demo that runs the whole pipeline client-side. The
reproduction reaches **TF1 0.9490** against the paper's 0.9706 on a quarter of the
training data.

**The interesting result is not the reproduction. It is that the benchmark did not
predict real-world performance, and the four specific reasons why.**

## A2. The one-paragraph headline

Getting the model to work on a real garment was a separate problem from reproducing the
paper, and **it is not solved**. Four concrete defects in how the input is represented
were found and measured, three fixed. The model still scores **0.33–0.49 on a real CLO
export against 0.75–0.95 on the benchmark** — and the ordering inverts: the model that
scores *best* on GarmentCodeData scores *worst* on the real garment, and training on 44×
more synthetic data made real-world performance **drop**.

## A3. The findings, each with its number

| # | finding | evidence | status |
|---|---|---|---|
| 1 | The model leaned on **panel ordering**, which a DXF has not got | zeroing that one feature: TF1 **0.9086 → 0.6110**, GSP **0.445 → 0.085** | fixed |
| 2 | Curvature features **cannot survive a DXF** | on the real file **all 28 curved edges fail to refit**, by 10× to 600× tolerance | fixed |
| 3 | CLO splits one seam across two edges on the **mirror axis** | 12 of 60 turn points are artifacts; the two populations separate by **40°** | fixed |
| 4 | Dim 4 is the **chord**; a seam matches the **arc** | collar seams **66.7% → 100%** agreeing within 10% | fixed |
| 5 | Real seams join curves that **do not interlock** | **0 of 32** attempts across four models | **open** |

### 1 — panel ordering (the biggest, and invisible to a distribution check)

GarmentCodeData numbers panels in *generation* order, so related panels sit next to each
other, and that number is a feature. A DXF numbers panels by block order — arbitrary.
About a quarter of the reported accuracy was borrowed from that artifact.

**Why this is the interesting one:** the feature's *values* on CLO data are perfectly in
range (0.50σ shift, 0% of edges outside the training 1–99%). Its *meaning* is destroyed.
No distribution check finds this; only a perturbation experiment does.

The fix was a config option nobody had exercised — `panel_id_mode="random_norm"`, which
shuffles the ids per sample. The model then becomes *completely* order-independent:
±0.004 TF1 under every perturbation, and on the CLO file zeroing or reversing the ids
changes **not one prediction**.

### 2 — the curvature encoding

The paper encodes an edge as a type tag plus ten slots whose meaning the tag selects. A
DXF stores curves as sampled points, so the type must be re-estimated, and a wrong guess
redefines eleven dimensions at once. On the real file it is worse than a wrong guess:
every curved edge fails to fit any single arc or Bézier, because a CLO boundary between
corners is free-form.

Replacement: a **sagitta profile** — signed deviation from the chord at 11 uniform
arclength positions, over the chord length. No type is ever decided. Round-trip error
**1.101 → 7.7e-4**, and CLO's features land inside the training distribution on all 24
dimensions.

**The sharpest way to put it:** sagitta's error is *continuous* in sampling density
(0.120 at 3 points → 4.3e-5 at 128); the tagged encoding sits at **~1.0 no matter how many
samples it gets**, because more data cannot repair a categorical flip.

**Its value is not accuracy.** With panel order removed the two encodings tie (0.7501 vs
0.7468). Its value is that DXF input becomes possible at all.

### 5 — the open one

Ground truth for the CLO garment was drawn by hand in the demo (20 stitches, 40 of 48
edges). It is extremely regular: **11 stitches match in length to 1.000, 18 of 20 to
within 3%**. Sorting by how well the two edges' shapes interlock splits the model's
success cleanly:

| shape mismatch | stitches | found over 4 models | rate |
|---|---|---|---|
| interlocking (< 0.05) | 12 | 29 of 48 | 0.60 |
| **unrelated (> 0.5)** | **8** | **0 of 32** | **0.00** |

The eight are both waist seams, all four sleeve attachments and both straps — *the seams
that make a garment a garment*. Their lengths match to 1.000. What fails is that a concave
armhole is eased onto a convex sleeve cap, and a nearly straight bodice waist onto a
flared skirt.

**Ruled out with numbers**, in case it is asked: not edge length (the missed seams match
to 1.000); not length ambiguity (the model does *worst*, 0 of 4, on uniquely-lengthed
seams); not edge count (GCD stitched panels share one 33.3% of the time against 29.3% at
random); not an unseen seam type — on GCD its armhole recall is **0.993**.

**That last one is the live question**: on GarmentCodeData the model handles
shape-mismatched seams well; on a real garment it gets none of them. Whatever cue it uses
there is absent here, and it has not been identified. *Say this plainly; do not invent an
answer.*

## A4. What can be shown live

The demo runs entirely in the browser — `npm run dev` in `webdemo/`, or the deployed copy
at https://junseongahn.github.io/seam-correspondence/, which ships this same r3 model.

| what | where |
|---|---|
| import a CLO DXF, get predicted stitching **and scored** against hand-drawn truth |
  the drop zone, or "1. CLO tutorial example" |
| the same on a held-out GCD garment, **scored** | "2. GarmentCode test data" |
| five more held-out garments, **F1 0.00 to 1.00** | the second row of links |
| each of those beside its **ground-truth assembly, in red** | the right pane there |
| draw ground truth by hand, click edge to edge | "draw the ground truth by hand" |
| the wasm assembly solver, input → solved | right pane of example 2, and of either
  exactly-predicted garment in the second row |
| CLO's own drape with weld-derived seams | right pane of example 1 |

Two details worth mentioning unprompted, because they show judgement:

- The example garments were picked by **scoring 250 held-out garments with the model the
  page actually runs and taking a spread** — 0th, 24th, 57th, 64th percentile — not by
  choosing what looks good.
- The red pane is the **ground truth's** shape, solved on the host from the correct
  stitching, and is labelled as such in the pane itself. It is the only thing on the
  page that is not the model's output, and a 3D shape is persuasive enough to be taken
  for one, so it is red and says so. Where the prediction is exact it is equally the
  prediction's shape; where it is not, the note says a wrong stitching would tear the
  mesh. Seam direction is not guessed anywhere: the vertex pairing comes from welding
  the simulated mesh, and the solve starts from the specification's per-panel 3D
  placement, so panels begin on their own side of the body and seams close untwisted.
- The assembly solve is **disabled unless the prediction exactly equals the ground
  truth**. The dump's constraints are vertex pairs built from the GT and it carries no
  edge→vertex mapping, so a wrong stitching cannot honestly be assembled — and would tear
  the mesh if it could.

## A5. Numbers to have ready

| run | train garments | encoding | panel ids | GCD test TF1 | **CLO TF1** |
|---|---|---|---|---|---|
| paper | 102,660 | tagged | index | 0.9706 | — |
| r1 (reproduction) | 24,024 | tagged | index | **0.9490** | 0.467 |
| rand_sagitta | 2,000 | sagitta | random | 0.7501 | **0.485** |
| r2 | 87,697 | sagitta | random | 0.8030 | 0.333 |
| **r3 (+ arc features) — shipped** | 99,666 | sagitta | random | **0.8040** | **0.471** |

Per-garment distribution on 1,500 held-out garments (the shipped model): mean **0.846**,
median **0.904**, **34.1% exactly 1.000**, **5.4% below 0.5**. Mostly-perfect with a real
tail — not uniformly mediocre.

**r2's 0.8030 is not comparable to r1's 0.9490.** r1's number includes the panel-order
signal and drops to 0.61 without it; r2 never uses it.

Verification, if pressed on rigour:

| claim | how |
|---|---|
| WASM solver correct in-browser | headless Chrome + CDP; max abs diff **9.95e-11 cm** vs Python, `monoViolations 0` |
| SIMD actually on | sha256 match, reproducible rebuild, Eigen probe (`packet<double>=2`), 1826 `f64x2` opcodes vs 0 |
| browser sees training's features | TS vs Python dump: **0.000e+00** on both a spec and the CLO DXF, all dims |
| ONNX export faithful | 39/40 identical; the one difference is a tie-break, the two
  runtimes agreeing on that pair's score to **1.15e-07** |
| the seam ground truth is real | reproduces CLO's own merge count of **241** exactly; the exporter refuses otherwise |

## A6. Things to be honest about

- **The CLO garment's panels were deliberately cut so every seam is one-to-one.** The
  model's output space is one-to-one edge pairs and the training data contains no
  multi-edge stitches at all. Real patterns have gathers. Nothing here handles them.
- **Inference now keeps only mutually-agreed pairs**, which guarantees one-to-one and
  never scored lower. That is right *for this garment* and wrong for a pattern with
  gathers.
- **One garment is not a benchmark.** Every CLO number rests on a single hand-annotated
  example.
- **r3 vs r2 is confounded**: r3 has both the arc features and 14% more data, because the
  download was still running. The arc-feature effect alone is not isolated. What is clear
  is that it bought nothing on GCD (0.8030 → 0.8040) and did not touch the 0-of-32.
- **Two mistakes were made and caught**, both worth telling if the conversation turns to
  method (they read as rigour, not as failure):
  1. The diagnostic scripts read edges straight from the DXF while the pipeline
     canonicalises each panel to anticlockwise first, so on every reversed panel they
     named a different edge than the model was given. This produced a confident, wrong
     analysis — length ratios up to 3.4×, a story about the garment being out of
     distribution — until the drawing contradicted it. Fix: one shared accessor
     (`dxf_to_features.panel_edges`) instead of two implementations of one ordering.
  2. Four models agreed on a stitch pairing that disagreed with the ground truth, and that
     was written up as a model failure. **The ground truth was wrong.** Independent models
     agreeing is evidence *for* an answer; the agreement should have prompted a check of
     the label.

## A7. Where to read further

`REPORT.md` (TL;DR → changes → evidence) with figures in `report/`. Every figure is a
measurement and is regenerated by `dxfcheck/make_report_figures.py`.

---
---

# §B — For continuing the engineering

## B1. Hard constraints (from the user, still in force)

- **Python is exactly one interpreter.** `$PY = "$env:USERPROFILE\miniconda3\envs\agentic-ai\python.exe"`,
  invoked as `& $PY ...`. **Never `conda activate`.**
- **The GCD data folders are read-only.** `C:\Users\POMCHECKER\gcd_data`.
- **`autosew/HANDOFF.md` is authoritative** for the reproduction. Its §2 hyperparameters
  are paper-specified: do not change them. The industrial track's departures are config
  flags, and the paper track's defaults are untouched.
- **`rigid-clothes-simulation/` is the user's own code.** Read-only unless asked.
- Emscripten `EMSDK=/c/repos/emsdk`; Eigen at `C:/repos/eigen`, header-only, not vendored.
- **Do not `npm install` without asking.** `fbx-parser` and `dxf-parser` are installed and
  were chosen by the user — use them, do not hand-roll either format in the web path.
- **Do not hand-edit `clo_example/panel_seperated_gt.json`.** It is the user's hand-drawn
  ground truth. Redraw in the demo and re-export instead; editing it by hand across a
  conversation is how it got corrupted once already.

## B2. Component status

| component | where | state |
|---|---|---|
| AutoSew model | `autosew/` | working; r1 TF1 0.9490 |
| Assembly solver (Python) | `rigid-clothes-simulation/` | working, user's code |
| C++ / WASM port | `simcpp/` | working; wasm verified in a real browser |
| Browser demo | `webdemo/` | working, all input modes |
| DXF input path | `dxfcheck/`, `webdemo/src/lib/parseDxf.ts` | working end to end |
| CLO seam ground truth | `dxfcheck/export_weld_gt.py` | solved, eye-verified |
| ONNX export | `autosew/scripts/export_onnx.py` | working, 40/40 verified |
| 2D↔3D panel matching | — | **blocked on mirror pairs** (B5) |

## B3. Key configuration

The industrial track is three flags on the same codebase; the paper track is the defaults.

```
--set curvature_encoding=sagitta panel_id_mode=random_norm arc_features=true
```

`train.py` sets `cfg.in_dim = feature_dim(cfg)` automatically. **Any new feature-affecting
config must go into `train.py`'s `FEAT_KEYS`**, or a cache built under one setting is
silently reused under another.

## B4. The SIMD finding (non-obvious — don't lose this)

**`-msimd128` alone does nothing for Eigen.** Eigen has no wasm-SIMD backend, so packet
size stays 1. You must *also* pass `-msse2` or higher to engage emscripten's SSE→wasm128
emulation. Net gain is only −9% / −19% against native's 2.2×; **~10 s is the floor for
this structure**. Do **not** add `-ffast-math`: the monotonicity gate in `solve()` and the
1e-300 guards in `best_rotations` depend on IEEE semantics.

`simcpp/build.sh` has **not** been updated with the flags, and `build-wasm-simd/` and
`build-wasm-simdonly/` are **not** in `.gitignore`.

## B5. What is blocked

No shared identifier exists between the 10 DXF pieces and the 10 draped 3D panels — not in
the DXF labels (per-document corner numbering; the apparent cross-piece sharing is between
*mirror duplicate* pieces), not in any of the three FBX files (the garment is a single mesh
in all of them), not in the OBJ materials (two fabrics, not ten panels).

Shape invariants resolve the size *groups* cleanly but cannot separate a left sleeve from a
right one: perimeter and area are identical by construction and the assignment margins are
0.0000. Sagitta *sign* does distinguish them, which is a lead. Until this is broken, the
CLO prediction cannot be scored automatically — the current scoring rests on the
hand-drawn ground truth.

## B6. Traps already hit — do not repeat

- **`train.log` looks empty.** stdout is block-buffered when redirected. Monitor
  `runs/<name>/history.jsonl`.
- **Diagnostics must use the pipeline's ordering.** `dxf_to_features.panel_edges()` exists
  for this. Reading `read_panels()` directly skips the anticlockwise canonicalisation.
- **React StrictMode double-invokes state updaters.** A toggle inside one
  (`has ? delete : add`) runs twice and cancels itself; side effects inside one fire twice.
- **cProfile lied** once: a "faster" change was 46% slower. A/B on wall clock, two runs
  each, alternating.
- **numpy 2.x removed `ndarray.ptp()`** — use `np.ptp(a)`.
- **`evaluate_batch(logP, batch, cfg, acc)`** — the accumulator is the *last* argument.
- **`ONNX EyeLike` has no onnxruntime-web kernel**; the exporter patches it at export time
  and asserts the patch is bit-identical first.
- Heredocs break on this content, and `python` is not on PATH in the Bash tool.
  Background `bash script.sh &` dies when the tool call ends.
- **I overwrote `rigid-clothes-simulation/README.md`** once by checking tracked status
  before a rebase and not re-checking. Re-check immediately before writing to any file you
  did not create.

## B7. Suggested order of work

1. **Finish r3** and score it on CLO. The question is whether arc features move the
   **0 of 32** in §A3-5. So far they have not.
2. **Attack the open finding.** Everything length- and curvature-shaped has been tried and
   measured; that avenue is closed. The next idea should be relational or structural.
3. **Break the mirror ambiguity** (B5) — sagitta sign is the lead.
4. **A second annotated garment.** Every CLO number rests on one example.
5. Fold the SIMD flags into `simcpp/build.sh`; gitignore the two extra build dirs.
7. Redeploy the demo — the live copy predates the current model and UI.

## B8. Dataset

Official split at
`C:\Users\POMCHECKER\gcd_data\GarmentCodeData_v2_official_train_valid_test_data_split.json`
(102,660 / 6,270 / 6,265). Downloaded to `gcd_data\{train,valid,test}\part<N>\`, **~99,666
train specs as of 2026-09-03**, and it was still growing during this session — which is
what confounds r2 against r3.

GCD v2 is served as **one `data.tar.gz` per part (~4.7 GiB)**, *not* as per-garment WebDAV
folders; `autosew/scripts/fetch_gcd_specs.py` does not work against that share. Use
`extract_specs_from_tar.py` plus `organize_split.py` (whose `sys.exit(1)` reports a false
failure when a part is already organised).

Parsing 99,666 specs takes ~35 minutes at ~0.8 MB/s — that is per-file overhead on an
NVMe SSD, most likely Defender scanning each small file. Excluding `gcd_data` from
real-time scanning would speed every run's first epoch considerably.

## B9. Component docs

- `autosew/HANDOFF.md` — authoritative for the model.
- `rigid-clothes-simulation/HANDOFF.md`, `README.md` — the user's own. Solver gates:
  `mono_violations == 0`, `seam_gap_max < 1e-3`, `max_sigma_dev < 1.0`, finite energies.
- `simcpp/README.md` — build invocations and the Woodbury measurement.
