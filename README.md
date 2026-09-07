# seam-correspondence

A reproduction of **AutoSew** (WACV 2026) — a 2D sewing pattern in, the stitch
correspondence between panel edges out — extended to take real CLO exports, with an ARAP
assembly solver in C++/WebAssembly and a browser demo that runs the whole pipeline
client-side. Live at <https://junseongahn.github.io/seam-correspondence/>.

Start with [REPORT.md](REPORT.md) for what was found and [HANDOFF.md](HANDOFF.md) for how
to continue the work.

## Failure report (Anthropic API)

Give it an eval JSON — edge pairs with `pred`/`gt`, arc lengths, sagitta, what the model
chose instead and how sure it was, exported by the demo's own button — and it returns a
Markdown breakdown: failures grouped into categories with a cause each, plus three ranked
next actions. Run it locally with `npm run report -- eval/real_dxf_01.json --out
reports/real_dxf_01.md`, or press **show LLM evaluation report** on the
[live demo](https://junseongahn.github.io/seam-correspondence/) and generate a fresh one.
The key is never in the page: locally it comes from `process.env.ANTHROPIC_API_KEY` (a
git-ignored `api.env`, see [api.env.example](api.env.example)); in the browser the call
goes to a Cloudflare Worker that enforces one report per IP per 30 s and forwards to a
Vercel function that holds the key — Anthropic refuses the Cloudflare edge outright, which
is why the call is not made there. The gate exits non-zero unless every MISS and
FALSE_POSITIVE is assigned to exactly one category, the three established causes (shape
mismatch, arc-vs-chord, mirror axis) are addressed, and there are exactly three next
actions; it checks the report's shape, not whether the reasoning is right. Generated
report: [reports/real_dxf_01.md](reports/real_dxf_01.md).
