# seam-correspondence

A reproduction of **AutoSew** (WACV 2026) — a 2D sewing pattern in, the stitching between
its panel edges out — extended to take real CLO exports, with an ARAP assembly solver in
C++/WebAssembly and a browser demo that runs the whole pipeline client-side.

**Live: <https://junseongahn.github.io/seam-correspondence/>**
<img width="1056" height="848" alt="image" src="https://github.com/user-attachments/assets/ac05c2bc-e0a9-4f62-940e-6c10c99e5745" />

TF1 0.9490 against the paper's 0.9706 on a quarter of the training data. The interesting
result is not the reproduction: it is that the benchmark did not predict real-world
performance, and the four measured reasons why — [REPORT.md](REPORT.md) for those,
[HANDOFF.md](HANDOFF.md) to continue the work.

Only one-to-one correspondence is supported: the training data contains no multi-edge
stitches, so gathers cannot be expressed.

| | |
|---|---|
| `autosew/` | the model — features, training, ONNX export |
| `rigid-clothes-simulation/` | the assembly solver (Python) |
| `simcpp/` | its C++ / WebAssembly port |
| `webdemo/` | the browser demo |
| `dxfcheck/` | the CLO DXF path, and the diagnostics behind REPORT.md |
| `tools/`, `api/`, `worker/` | the failure report, below |

## Failure report (Anthropic API)

Takes only the failures out of an evaluation — edge pairs with `pred`/`gt`, arc lengths,
sagitta, and what the model chose instead — forces a schema, and checks the answer in
code: every miss and false positive assigned to exactly one category, the three
established causes addressed, exactly three ranked next actions. A run that misses any of
those exits non-zero. **One run: ~3,900 in / ~2,000 out tokens, ~20 s, about $0.03.**

```sh
npm run report -- eval/real_dxf_01.json --out reports/real_dxf_01.md
```

Sample output: [reports/real_dxf_01.md](reports/real_dxf_01.md).

The gate checks the report's *shape*, not whether the reasoning is right.

The key comes from `ANTHROPIC_API_KEY` — a shell export or a git-ignored `api.env`
([api.env.example](api.env.example)). The live demo generates reports too: the browser
calls a Cloudflare Worker that allows one per IP per 30 s and forwards to a Vercel
function holding the key, because Anthropic refuses the Cloudflare edge outright.
