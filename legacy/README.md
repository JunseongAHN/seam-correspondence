# legacy

Everything here predates the AutoSew work and nothing in the active tree imports it. It
is kept rather than deleted because it is where several of the current choices came from
— the depth and tolerance sweeps, the optimal-transport matcher, the first viewers — and
a claim in a report is worth more when the run behind it is still on disk.

Nothing here is maintained. If something is needed again, move it back rather than
importing across the boundary.

| | |
|---|---|
| `*.py` at this level | the first pass: baselines, depth oracles, tolerance sweeps, exporters |
| `*.json` at this level | their outputs, kept as fixtures |
| `index.html`, `viewer_template.html`, `make_viewer.py` | the viewer before `webdemo/` |
| `noise-inject/` | perturbing patterns to see what the matcher tolerated |
| `rigid-simulation/` | the first rigid solver, superseded by `rigid-clothes-simulation/` |
| `results/` | its outputs |

`rigid-clothes-simulation/HANDOFF.md` cites `run_baseline.py` for the origin of
`tau_l = 10 mm`; that file is now `legacy/run_baseline.py`.

The active tree is `autosew/` (the model), `rigid-clothes-simulation/` and `simcpp/` (the
solver, Python and C++/WASM), `webdemo/` (the browser demo), `dxfcheck/` (the CLO DXF path
and the diagnostics behind REPORT.md), and `tools/` + `api/` + `worker/` (the failure
report).
