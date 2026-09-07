# Seam-matching failure report

Most of these 18 failures come from a cluster of edges (8_M, 6_M, 10_M, 11_M) that are geometrically identical straight edges of the same two lengths (10.5 and 21.0 cm) with zero curvature, so the model cannot tell them apart and swaps partners among the group -- shape_mismatch of 0.00 confirms the edges themselves fit perfectly, it's an identity/ambiguity problem, not a shape problem. A second cluster (3_M#2/9_M#6/6_M#3/3_M#0/12_M#2 etc.) shows genuine shape mismatches (shape_mismatch 0.73-1.00) where curved edges of different sagitta are being asked to mate, and the model correctly refuses some (dustbin) but wrongly accepts one high-confidence bad pair. A third small group of misses are true low-confidence dustbin refusals on long, nearly-matching arcs (len 43.5-45.2, shape_mismatch 0.73-0.87) that sit right at the edge of the shape-mismatch threshold. There is no evidence of an arc-vs-chord length problem anywhere in this set, since paired lengths already agree closely (e.g. 21.0/21.0, 45.2/45.2) wherever geometry is the issue.

## Failure categories

- **shape mismatch (ambiguous duplicate straight edges, low signal)** (10) — Several edges in this garment are perfectly straight (curvature 0.00) and come in duplicate lengths (10.5 cm and 21.0 cm), so shape_mismatch is 0.00 for many wrong pairings too -- the model has no geometric signal to distinguish which of several identical-length straight edges is the true partner. It picks whichever duplicate scores marginally higher (p around 0.47-0.65, far from confident) and the true partner loses out to a same-shaped rival (e.g. 8_M#3 chose 6_M#0 p=0.65 while the true partner 10_M#3 got only p=0.48). This is an identity/context problem, not a curvature or arc-length problem.
  - 8_M#1 <-> 6_M#2
  - 10_M#1 <-> 10_M#3
  - 6_M#0 <-> 11_M#0
- **shape mismatch (genuine curvature/sagitta disagreement)** (6) — True shape_mismatch is high (0.73-1.00) because the two edges' sagitta profiles genuinely don't interlock (e.g. curvature 0.54 vs 0.00, or 0.20 vs 0.10) even though lengths are close. The model correctly sends the weaker edge to dustbin at high probability (0.80-1.00) in most of these, showing it is properly refusing bad-shape pairs; the residual miss is because the true partner is unavailable (already consumed by a wrong high-confidence match elsewhere), not a shape-mismatch scoring failure per se.
  - 3_M#0 <-> 12_M#2
  - 9_M#4 <-> 13_M#0
  - 3_M#2 <-> 11_M#1
- **false positive: high-confidence bad accept despite real shape mismatch** (1) — shape_mismatch is 0.27 (moderate, not negligible) and curvature differs (0.20 vs 0.14) yet the model assigns p=0.994 to both sides reciprocally -- this pair anchors a chain reaction that starves the true partners (11_M#1, 6_M#3) of their correct match, turning one bad accept into multiple downstream misses in the 'ambiguous duplicate' and 'curvature disagreement' groups above.
  - 3_M#2 <-> 9_M#6
- **shape mismatch (long-arc curvature disagreement, borderline dustbin)** (1) — Long edges (~20 cm) with opposite-signed curvature (0.14 vs -0.09) give shape_mismatch 0.74, and the model dustbins both sides independently (p=1.00 and p=0.79) rather than pairing them -- consistent with genuine shape disagreement rather than a length or mirror-axis artifact, since arc lengths already match closely (20.5 vs 20.1).
  - 9_M#2 <-> 10_M#0

## Next actions (ranked)

1. Add a disambiguating feature (e.g. panel adjacency / topological neighbor context, or edge index within panel) so duplicate same-length straight edges (10.5 cm and 21.0 cm cluster across 6_M, 8_M, 10_M, 11_M) stop being interchangeable -- this alone accounts for 10 of 18 failures.
2. Investigate the one false positive with shape_mismatch 0.27 and p=0.994 (3_M#2 <-> 9_M#6): tighten the acceptance threshold or add a shape_mismatch penalty term for scores above ~0.15, since this single bad accept cascades into at least 3 downstream misses.
3. No action needed on arc-vs-chord length or mirror-axis splitting: all failures here show closely matching arc lengths between true partners (e.g. 21.0/21.0, 45.2/45.2), and no pair shows the half-length/double-edge signature expected from a mirror-axis split, so effort should not be spent there.

_model=claude-sonnet-5 · in=3878 out=1938 tokens · 21.4s_