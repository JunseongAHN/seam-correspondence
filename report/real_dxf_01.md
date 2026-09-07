# Seam-matching failure report

Most of the 18 failures come from a cluster of straight, identical-length edges (10.5cm and 21.0cm, curvature 0.00, shape_mismatch 0.00) around panels 6/8/10/11, where the model can't tell apart several geometrically indistinguishable edge pairs and swaps partners between them, producing 6 false positives and 4 misses that are really "right shape, wrong twin" confusions. A second group of misses involves genuinely mismatched shapes (shape_mismatch 0.73-1.00) where the model correctly refuses the pair and sends one or both edges to the dustbin at high probability -- this is the model behaving reasonably given real shape disagreement, not a confidence bug. A third small group (3_M#2/9_M#6 chain) shows a runaway high-confidence false positive (p=0.994) dragging two other edges (9_M#6<->6_M#3, 3_M#2<->11_M#1) into misses because those edges' true partners get stolen at very high probability.

## Failure categories

- **ambiguous identical-shape twins (straight, equal-length duplicate edges)** (10) — Multiple straight edges of identical arc length (10.5cm or 21.0cm) with curvature 0.00 and shape_mismatch 0.00 exist across panels 6, 8, 10, 11. Because these edges are geometrically indistinguishable (a straight segment has no shape signature to disambiguate which of several equal-length straight edges is the true partner), the model matches by chance among near-degenerate candidates, causing both the false positives (10_M#1<->10_M#3, 11_M#0<->11_M#2, 8_M#1<->6_M#2, 8_M#3<->6_M#0, 12_M#2<->13_M#0) and the corresponding true pairs to be missed when the wrong twin is chosen instead (6_M#0<->11_M#0, 6_M#2<->11_M#2, 8_M#1<->10_M#1, 8_M#3<->10_M#3). This is not shape mismatch (values are 0.00 or the mismatch is purely the trivial length-swap case, e.g. 10_M#1<->10_M#3 where the same panel's two edges of different lengths get confused), not arc/chord confusion (lengths already agree exactly), and not mirror-axis splitting (these are separate panels, not a seam split by a mirror line).
  - 8_M#1 <-> 6_M#2
  - 8_M#3 <-> 6_M#0
  - 10_M#1 <-> 10_M#3
- **genuine shape/curvature disagreement correctly rejected** (6) — These pairs show large shape_mismatch (0.73-1.00) with curvature values that visibly disagree in sign or magnitude (e.g. 0.54 vs 0.00, -0.04 vs 0.15, 0.15 vs -0.02), meaning the sagitta profiles genuinely do not interlock under any of the four alignments tested. The model assigns these edges to the dustbin at high probability (0.71-1.00) rather than forcing a bad match -- this is a correct refusal given real geometric mismatch ('shape mismatch' by name), not a confidence-calibration failure. No evidence of arc/chord confusion here since lengths are close (e.g. 45.2/45.2, 43.7/43.5); no evidence of mirror-axis splitting since these are distinct labeled panels, not a single panel's seam split across its own mirror line.
  - 3_M#0 <-> 12_M#2
  - 3_M#4 <-> Pattern_634078_M#1
  - Pattern_669883_M#3 <-> 9_M#0
- **high-confidence false-positive cascade stealing true partners** (2) — The false positive 3_M#2<->9_M#6 (p=0.994, shape_mismatch only 0.27) is a strong, near-plausible-shape mismatch that outcompetes the true partners of both edges: 9_M#6's real match 6_M#3 and 3_M#2's real match 11_M#1 both get pushed to the dustbin (p=0.47-0.93) because their edge is already claimed elsewhere. This is a shape-mismatch-driven ranking error (0.27 is low enough to look plausible but is not a true fit) rather than an arc/chord length issue (lengths 19.4/20.5 and 20.1/20.5 are close) or a mirror-axis artifact (no evidence these are split mirror edges of one panel).
  - 9_M#6 <-> 6_M#3
  - 3_M#2 <-> 11_M#1

## Next actions (ranked)

1. Add a tie-breaking feature or secondary geometric cue (e.g. adjacent-edge context, panel topology, or edge ordering) to disambiguate straight equal-length edges, since 10 of 18 failures stem from indistinguishable straight-edge twins across panels 6/8/10/11 with shape_mismatch 0.00.
2. Investigate and raise the decision margin/threshold for accepting a high-confidence match like 3_M#2<->9_M#6 (p=0.994, shape_mismatch 0.27) so it does not silently starve the true partners 6_M#3 and 11_M#1 into the dustbin -- consider a bipartite/global assignment pass instead of independent per-edge argmax.
3. Leave the dustbin-routed high-shape_mismatch cases (shape_mismatch 0.73-1.00, e.g. 3_M#0<->12_M#2, 3_M#4<->Pattern_634078_M#1) as-is or use them as negative-training exemplars, since these are correct rejections of genuinely non-interlocking shapes, not a bug to fix.

_model=claude-sonnet-5 · in=3878 out=2061 tokens · 21.7s_