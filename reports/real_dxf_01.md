# Seam-matching failure report

The model is failing on three distinct geometric patterns rather than random noise: nearly half the errors come from CAD files splitting one physical seam into two edges at a panel's mirror line, causing both wrong half-edge matches and missed whole-edge matches. Another third of the errors involve genuine concave-to-convex seam eases where curvature sign or magnitude disagrees enough to reject a correct match despite matching lengths. The remainder stem from mixing arc-length and chord-length measurements on curved seams, so small length and curvature discrepancies push true matches below the similarity threshold.

## Failure categories

- **mirror axis artifacts (panel-splitting on straight seams, len 10.5/21.0 pairs)** (8) — Panels 6, 8, 10, 11 all show a straight seam of length 21.0 that also appears split into two 10.5 halves within the same panel set (e.g. 10_M#1/10_M#3, 11_M#0/11_M#2). This is the classic CAD mirror-line export artifact: a single physical seam crossing the panel's mirror axis gets emitted as two edges, so the matcher both wrongly pairs half-edges as false positives (8_M#1<->6_M#2, 8_M#3<->6_M#0, 10_M#1<->10_M#3, 11_M#0<->11_M#2, 12_M#2<->13_M#0) and misses the correct whole-edge or half-to-half pairings (6_M#0<->11_M#0, 6_M#2<->11_M#2, 8_M#1<->10_M#1, 8_M#3<->10_M#3). The zero curvature on all of these confirms they are straight seams, consistent with a mirrored fold line rather than a genuine shape difference.
  - 8_M#1 <-> 6_M#2
  - 10_M#1 <-> 10_M#3
  - 11_M#0 <-> 11_M#2
- **shape mismatch (concave/convex sagitta sign disagreement)** (6) — Several missed matches have similar lengths but curvature values that differ sharply in magnitude or flip sign (e.g. 0.54 vs 0.00, 0.17 vs 0.00, -0.04 vs 0.15, 0.15 vs -0.02), indicating a concave edge on one panel is being eased onto a convex or flat edge on the other. The matcher's curvature-similarity threshold is rejecting these true seams because sagitta sign/magnitude disagreement is treated as a mismatch, even though lengths align well enough to be a real seam pair.
  - 3_M#0 <-> 12_M#2
  - 9_M#4 <-> 13_M#0
  - 3_M#4 <-> Pattern_634078_M#1
- **arc vs chord length discrepancy on paired curved seams** (4) — These misses (and the related false positive 3_M#2<->9_M#6) involve edges with near-identical curvature (~0.14-0.20 vs ~-0.09 to 0.10) and lengths that are close but not exact (19.4/18.9, 20.5/20.1), suggesting one edge's length was measured along the sewn arc and the other along the straight chord. The small length deltas combined with curvature sign flips push otherwise-correct matches below the similarity threshold, producing both a false pairing (3_M#2 wrongly linked to 9_M#6 instead of 11_M#1) and its correct counterpart being missed.
  - 9_M#6 <-> 6_M#3
  - 9_M#2 <-> 10_M#0
  - 3_M#6 <-> 8_M#2

## Next actions (ranked)

1. Add mirror-axis detection: pre-merge or flag co-linear straight edges within the same panel that sum to a known paired-panel seam length (e.g. 10.5+10.5=21.0) before matching, to eliminate the 8 mirror-related failures.
2. Relax or reweight the curvature similarity check to tolerate sign flips when comparing concave-to-convex eased seams, and add a sagitta-magnitude-only comparison mode to recover the 6 shape-mismatch misses.
3. Switch edge length comparison to use arc length (sewn length) consistently instead of mixing chord and arc measurements, which should resolve the 4 arc-vs-chord related failures and likely fix the related false positive.

_model=claude-sonnet-5 · in=2116 out=1595 tokens · 16.8s_