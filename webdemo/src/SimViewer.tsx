/* The assembly simulation, seen rather than printed.

   Three stages of the same garment, drawn from the SIMCPP01 dump and the wasm solve:
     flat      the 2D panels the solver starts from (dump.rest)
     placed    their initial 3D placement before solving (dump.P0)
     solved    the wasm result

   The body is not a mesh: the solver models it as analytic capsules and spheres, so
   those are drawn from their own parameters (cylinders are p0,p1,r; spheres c,r) rather
   than from any geometry file.

   This is the prepared GCD garment the dump was built from, not the CLO example -- the
   CLO garment has no simulation input yet, because that needs edge-level seams and the
   2D<->3D panel matching, which the mirror-pair ambiguity still blocks. */
import { useCallback, useEffect, useRef, useState } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkCylinderSource from "@kitware/vtk.js/Filters/Sources/CylinderSource";
import vtkSphereSource from "@kitware/vtk.js/Filters/Sources/SphereSource";
import { loadDump, solveParsed, simdSupported, type Dump, type SimResult } from "./lib/sim";

type Stage = "flat" | "placed" | "solved";

const PANEL_COLORS: [number, number, number][] = [
  [0.20, 0.47, 0.85], [0.95, 0.55, 0.15], [0.20, 0.65, 0.35], [0.85, 0.25, 0.30],
  [0.55, 0.35, 0.75], [0.60, 0.42, 0.30], [0.90, 0.45, 0.70], [0.45, 0.45, 0.45],
  [0.75, 0.75, 0.20], [0.20, 0.70, 0.80],
];

/** Vertex positions for a stage, always as (n,3). */
function stagePoints(d: Dump, r: SimResult | null, stage: Stage): Float32Array {
  const out = new Float32Array(d.n * 3);
  if (stage === "flat") {
    for (let i = 0; i < d.n; i++) {
      out[i * 3] = d.rest[i * 2]; out[i * 3 + 1] = d.rest[i * 2 + 1]; out[i * 3 + 2] = 0;
    }
  } else {
    const src = stage === "placed" || !r ? d.P0 : r.positions;
    for (let i = 0; i < d.n * 3; i++) out[i] = src[i];
  }
  return out;
}

function meshActors(d: Dump, pts: Float32Array, edges: boolean) {
  const byPanel = new Map<number, number[]>();
  for (let f = 0; f < d.M; f++) {
    const p = d.panelOfFace[f];
    let a = byPanel.get(p);
    if (!a) { a = []; byPanel.set(p, a); }
    a.push(d.faces[f * 3], d.faces[f * 3 + 1], d.faces[f * 3 + 2]);
  }
  const actors: vtkActor[] = [];
  for (const [p, tris] of [...byPanel].sort((a, b) => a[0] - b[0])) {
    const polys = new Uint32Array((tris.length / 3) * 4);
    for (let t = 0, o = 0; t < tris.length; t += 3, o += 4) {
      polys[o] = 3; polys[o + 1] = tris[t]; polys[o + 2] = tris[t + 1]; polys[o + 3] = tris[t + 2];
    }
    const pd = vtkPolyData.newInstance();
    pd.getPoints().setData(pts, 3);
    pd.getPolys().setData(polys);
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(pd);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    const c = PANEL_COLORS[p % PANEL_COLORS.length];
    actor.getProperty().setColor(...c);
    if (edges) { actor.getProperty().setEdgeVisibility(true); actor.getProperty().setEdgeColor(0.1, 0.1, 0.1); }
    actors.push(actor);
  }
  return actors;
}

/** The solver's body proxy: capsules between p0 and p1, plus spheres. */
function bodyActors(d: Dump) {
  const actors: vtkActor[] = [];
  for (let k = 0; k < d.NC; k++) {
    const a = [d.cyl[k * 7], d.cyl[k * 7 + 1], d.cyl[k * 7 + 2]];
    const b = [d.cyl[k * 7 + 3], d.cyl[k * 7 + 4], d.cyl[k * 7 + 5]];
    const r = d.cyl[k * 7 + 6];
    const dir = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const h = Math.hypot(...dir) || 1e-9;
    const src = vtkCylinderSource.newInstance({
      height: h, radius: r, resolution: 32, capping: true,
      center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2],
      direction: [dir[0] / h, dir[1] / h, dir[2] / h],
    });
    const mapper = vtkMapper.newInstance();
    mapper.setInputConnection(src.getOutputPort());
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(0.72, 0.72, 0.76);
    actor.getProperty().setOpacity(0.35);
    actors.push(actor);
  }
  for (let k = 0; k < d.NS; k++) {
    const src = vtkSphereSource.newInstance({
      center: [d.sph[k * 4], d.sph[k * 4 + 1], d.sph[k * 4 + 2]],
      radius: d.sph[k * 4 + 3], phiResolution: 24, thetaResolution: 32,
    });
    const mapper = vtkMapper.newInstance();
    mapper.setInputConnection(src.getOutputPort());
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(0.72, 0.72, 0.76);
    actor.getProperty().setOpacity(0.35);
    actors.push(actor);
  }
  return actors;
}

/** Stitch constraints, as lines between the paired vertices. */
function seamActor(d: Dump, pts: Float32Array) {
  const lines = new Uint32Array(d.K * 3);
  for (let k = 0; k < d.K; k++) {
    lines[k * 3] = 2; lines[k * 3 + 1] = d.pairs[k * 2]; lines[k * 3 + 2] = d.pairs[k * 2 + 1];
  }
  const pd = vtkPolyData.newInstance();
  pd.getPoints().setData(pts, 3);
  pd.getLines().setData(lines);
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(pd);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setColor(0.05, 0.05, 0.05);
  actor.getProperty().setLineWidth(1.5);
  return actor;
}

/** The dump's stitch constraints are vertex pairs, built from the GROUND TRUTH before
    export; nothing in it says which vertices belong to which panel edge, so a predicted
    correspondence cannot be turned into constraints without a mapping the file does not
    carry.  The honest thing is therefore to solve only when the prediction is identical
    to the ground truth -- then the assembly is the prediction's, unambiguously -- and to
    refuse otherwise rather than show a solve of the wrong stitching. */
export default function SimViewer({ exact = true, fp = 0, fn = 0 }:
                                  { exact?: boolean; fp?: number; fn?: number }) {
  const host = useRef<HTMLDivElement>(null);
  const [dump, setDump] = useState<Dump | null>(null);
  const [result, setResult] = useState<SimResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // open on the input, not the answer: the panels as the solver receives them, sitting
  // around the body it actually models
  const [stage, setStage] = useState<Stage>("placed");
  const [showBody, setShowBody] = useState(true);
  const [showSeams, setShowSeams] = useState(false);
  const [wire, setWire] = useState(false);

  /* Parsing the dump is milliseconds, so the input geometry is on screen immediately;
     only the solve costs ~10 s and it is not needed to show what goes IN. */
  useEffect(() => {
    loadDump("trousers.bin").then(setDump)
      .catch((e) => setErr(String(e?.message ?? e)));
  }, []);

  const run = useCallback(async () => {
    if (!dump) return;
    setBusy(true); setErr(null);
    await new Promise((r) => setTimeout(r, 30));   // let the button repaint first
    try { setResult(await solveParsed(dump)); setStage("solved"); }
    catch (e: any) { setErr(String(e?.stack ?? e?.message ?? e)); }
    finally { setBusy(false); }
  }, [dump]);

  useEffect(() => {
    if (!dump || !host.current) return;
    const grw = vtkGenericRenderWindow.newInstance({ background: [1, 1, 1] });
    grw.setContainer(host.current);
    const renderer = grw.getRenderer();
    const pts = stagePoints(dump, result, stage);
    for (const a of meshActors(dump, pts, wire)) renderer.addActor(a);
    if (showSeams) renderer.addActor(seamActor(dump, pts));
    if (showBody && stage !== "flat") for (const a of bodyActors(dump)) renderer.addActor(a);
    renderer.resetCamera();
    renderer.getActiveCamera().setViewUp(0, 1, 0);
    renderer.resetCameraClippingRange();
    grw.resize();
    grw.getRenderWindow().render();
    const onResize = () => { grw.resize(); grw.getRenderWindow().render(); };
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); grw.delete(); };
  }, [dump, result, stage, showBody, showSeams, wire]);

  return (
    <>
      <div className="examples">
        <button onClick={run} disabled={busy || !exact}>
          {busy ? "solving… (the page freezes for ~10 s)"
                : exact ? "assemble it — the prediction, solved"
                        : "cannot assemble this prediction"}
        </button>
        <span className="note">
          {exact
            ? "the prediction for this garment is identical to its ground truth — 24 of 24 "
              + "stitches, no false positives — so the solve below assembles exactly what "
              + "the model predicted. The body is not a mesh: the solver models it as 7 "
              + "capsules and a sphere, and that is what is drawn."
            : `the prediction has ${fp} false positive${fp === 1 ? "" : "s"} and `
              + `${fn} missed stitch${fn === 1 ? "" : "es"}. The dump's constraints are `
              + "vertex pairs built from the ground truth, and it carries no mapping from "
              + "a panel edge to its vertices, so a wrong stitching cannot be assembled — "
              + "and would tear the mesh if it could."}
        </span>
      </div>
      {err && <div className="err">{err}</div>}
      {!dump && !err && <div className="drop">loading the garment…</div>}
      {dump && (
        <>
          <div className="bar">
            <span className="pill name">trousers.bin</span>
            <span className="pill">{dump.n} verts</span>
            <span className="pill">{dump.M} faces</span>
            <span className="pill">{dump.K} stitch pairs</span>
            <span className="pill">{dump.NC} cylinders · {dump.NS} spheres</span>
            {result ? (
              <>
                <span className={`pill ${result.monoViolations === 0 ? "ok" : "fp"}`}>
                  mono violations {result.monoViolations}
                </span>
                <span className="pill t">{(result.wallMs / 1000).toFixed(2)} s</span>
              </>
            ) : (
              <span className="pill">not solved yet</span>
            )}
            <span className="pill">simd {String(simdSupported())}</span>
          </div>
          <div className="toggles">
            <span className="seg">
              {(["flat", "placed", "solved"] as const).map((s) => (
                <button key={s} className={stage === s ? "on" : ""}
                        disabled={s === "solved" && !result}
                        onClick={() => setStage(s)}>
                  {s === "flat" ? "2D panels in" : s === "placed" ? "initial placement" : "solved output"}
                </button>
              ))}
            </span>
            <label><input type="checkbox" checked={showBody}
                          onChange={(e) => setShowBody(e.target.checked)} /> body capsules</label>
            <label><input type="checkbox" checked={showSeams}
                          onChange={(e) => setShowSeams(e.target.checked)} /> stitch constraints</label>
            <label><input type="checkbox" checked={wire}
                          onChange={(e) => setWire(e.target.checked)} /> mesh edges</label>
          </div>
          {result && (
            <pre className="energies">
{`E_arap   = ${result.energies[0]}
E_bend   = ${result.energies[1]}
E_stitch = ${result.energies[2]}
E_obst   = ${result.energies[3]}`}
            </pre>
          )}
          <div className="vtk" ref={host} />
        </>
      )}
    </>
  );
}
