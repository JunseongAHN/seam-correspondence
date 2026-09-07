# seam-correspondence

A reproduction of **AutoSew** (WACV 2026) — a 2D sewing pattern in, the stitch
correspondence between panel edges out — extended to take real CLO exports, with an ARAP
assembly solver in C++/WebAssembly and a browser demo that runs the whole pipeline
client-side. Live at <https://junseongahn.github.io/seam-correspondence/>.

Start with [REPORT.md](REPORT.md) for what was found and [HANDOFF.md](HANDOFF.md) for how
to continue the work.

## Failure report (Anthropic API)

Give it an eval JSON (edge pairs with `pred`/`gt`, arc lengths and sagitta, exported by the
demo's "save the evaluation as JSON" button) and it returns a reviewer-readable Markdown
breakdown: failures grouped into categories with a likely cause each, plus three ranked
next actions. Run it with `npm run report -- eval/real_dxf_01.json --out
reports/real_dxf_01.md`. It is a **local CLI** — the key comes from
`process.env.ANTHROPIC_API_KEY`, via a shell `export` or a git-ignored `api.env` (see
[api.env.example](api.env.example)), and the browser demo never calls the API. The gate exits
non-zero unless every MISS and FALSE_POSITIVE is assigned to exactly one category, the
three established causes (shape mismatch, arc-vs-chord, mirror axis) are named, and there
are exactly three next actions. Generated report:
[reports/real_dxf_01.md](reports/real_dxf_01.md).
