# AutoSew web demo

Drop a GarmentCodeData `*_specification.json` and the predicted stitching is drawn
over the flat pattern. Everything runs in the browser: the ONNX model is fetched
once and executed with onnxruntime-web (WASM), so no data leaves the machine.

```bash
npm install
npm run dev
```

## What it does

1. `src/lib/parseSpec.ts` — port of `autosew/gcd_parser.py`. Panel-local coordinates
   (bbox lower-left at the origin), curvature typed to `k_t`, and the ACW
   canonicalisation that reverses clockwise panels.
2. `src/lib/features.ts` — port of `autosew/features.py`. 24 numbers per edge plus
   the prev/next cycle graph inside each panel.
3. `src/lib/infer.ts` — runs `public/model/autosew.onnx`, then the hard assignment
   that cannot live in the graph: `P' = (P + P^T)/2`, per-row argmax, plus every
   entry over `tau = 0.4`.
4. `src/lib/render.ts` — lays the panels out by their spec transform (back panels
   shifted right) and draws the overlay: green correct, red false positive,
   black dashed false negative.

The ground truth is read from the same file, so the counts shown are real, not an
estimate.

## Keeping the port honest

The features must match the Python pipeline exactly or the prediction silently
degrades. Checked on four garments (54 to 122 edges, straight / circular /
quadratic / cubic edges, reversed and non-reversed panels):

| | result |
|---|---|
| node order, cycle neighbours, GT pairs | identical |
| 24-dim features | `max|diff| = 0.000e+00` |
| predicted pair set (vs PyTorch) | identical |

If `gcd_parser.py` or `features.py` change, re-check before trusting this demo.

## The model

`public/model/autosew.onnx` (7.1 MB, opset 17) is exported from the best
checkpoint of the 24k-garment run (`runs/r1`, epoch 5). On the official GCD
test split it scores TF1 0.917 and gets 1143 of 2072 garments exactly right;
the earlier 2.8k-garment model scored TF1 0.879. It takes `x (B,M,24)`, `nbr (B,M,2)`, `mask (B,M)`
and returns `logP (B,M+1,M+1)`; `B` and `M` are dynamic. No custom operators.

Two things had to be avoided at export time for onnxruntime to run it: `torch.eye`
with a bool dtype becomes `EyeLike`, which onnxruntime has no kernel for, and the
in-place slice assignments building `Cbar` become `ScatterND`. Both are replaced
with `arange` comparison and `concat` in the export script.
