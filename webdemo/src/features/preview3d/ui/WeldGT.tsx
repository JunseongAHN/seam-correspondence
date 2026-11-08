/* Does the weld-derived seam ground truth actually look like seams?

   CLO's weld merges coincident boundary vertices without moving them, so on the draped
   mesh a sewn pair sits at one point and a seam line would have zero length.  The panels
   are therefore exploded outward from the garment centroid before drawing, which turns
   every weld into a visible line between the two panel boundaries it joins.

   Data comes from dxfcheck/export_weld_gt.py, which refuses to write the file unless the
   weld groups reproduce CLO's own merge count exactly. */
import { useEffect, useRef, useState } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";

const URL = `${import.meta.env.BASE_URL}example/weld_gt.json`;

type Gt = {
  source: string; merged: number;
  verts: number[]; tris: number[]; panelOf: number[];
  panelSizes: number[]; welds: number[][];
};

const PANEL_COLORS: [number, number, number][] = [
  [0.20, 0.47, 0.85], [0.95, 0.55, 0.15], [0.20, 0.65, 0.35], [0.85, 0.25, 0.30],
  [0.55, 0.35, 0.75], [0.60, 0.42, 0.30], [0.90, 0.45, 0.70], [0.45, 0.45, 0.45],
  [0.75, 0.75, 0.20], [0.20, 0.70, 0.80],
];

/** Move every panel away from the garment centroid, along its own centroid direction. */
function explode(gt: Gt, k: number): Float32Array {
  const n = gt.panelOf.length;
  const np = gt.panelSizes.length;
  const out = new Float32Array(gt.verts);
  const sum = new Float64Array(np * 3), cnt = new Float64Array(np);
  const all = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const p = gt.panelOf[i];
    for (let d = 0; d < 3; d++) { sum[p * 3 + d] += gt.verts[i * 3 + d]; all[d] += gt.verts[i * 3 + d]; }
    cnt[p]++;
  }
  for (let d = 0; d < 3; d++) all[d] /= n;
  const off = new Float64Array(np * 3);
  for (let p = 0; p < np; p++)
    for (let d = 0; d < 3; d++) off[p * 3 + d] = (sum[p * 3 + d] / cnt[p] - all[d]) * k;
  for (let i = 0; i < n; i++) {
    const p = gt.panelOf[i];
    for (let d = 0; d < 3; d++) out[i * 3 + d] += off[p * 3 + d];
  }
  return out;
}

function actorFrom(pd: vtkPolyData, color: [number, number, number],
                   width: number, opacity = 1) {
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(pd);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setColor(...color);
  actor.getProperty().setLineWidth(width);
  actor.getProperty().setOpacity(opacity);
  return actor;
}

export default function WeldGT() {
  const host = useRef<HTMLDivElement>(null);
  const [gt, setGt] = useState<Gt | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [k, setK] = useState(0.55);
  const [surface, setSurface] = useState(true);

  useEffect(() => {
    fetch(URL)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${URL}: HTTP ${r.status}`))))
      .then(setGt)
      .catch((e) => setErr(String(e?.message ?? e)));
  }, []);

  useEffect(() => {
    if (!gt || !host.current) return;
    const grw = vtkGenericRenderWindow.newInstance({ background: [1, 1, 1] });
    grw.setContainer(host.current);
    const renderer = grw.getRenderer();
    const pts = explode(gt, k);

    // one surface actor per panel, so each panel keeps its own colour
    if (surface) {
      const byPanel: number[][] = gt.panelSizes.map(() => []);
      for (let t = 0; t < gt.tris.length; t += 3)
        byPanel[gt.panelOf[gt.tris[t]]].push(gt.tris[t], gt.tris[t + 1], gt.tris[t + 2]);
      byPanel.forEach((tris, p) => {
        if (!tris.length) return;
        const polys = new Uint32Array((tris.length / 3) * 4);
        for (let f = 0, o = 0; f < tris.length; f += 3, o += 4) {
          polys[o] = 3; polys[o + 1] = tris[f]; polys[o + 2] = tris[f + 1]; polys[o + 3] = tris[f + 2];
        }
        const pd = vtkPolyData.newInstance();
        pd.getPoints().setData(pts, 3);
        pd.getPolys().setData(polys);
        renderer.addActor(actorFrom(pd, PANEL_COLORS[p % PANEL_COLORS.length], 1, 0.5));
      });
    }

    // every weld group drawn as a star from its first member
    const segs: number[] = [];
    for (const g of gt.welds) for (let i = 1; i < g.length; i++) segs.push(g[0], g[i]);
    const lines = new Uint32Array((segs.length / 2) * 3);
    for (let s = 0, o = 0; s < segs.length; s += 2, o += 3) {
      lines[o] = 2; lines[o + 1] = segs[s]; lines[o + 2] = segs[s + 1];
    }
    const lpd = vtkPolyData.newInstance();
    lpd.getPoints().setData(pts, 3);
    lpd.getLines().setData(lines);
    renderer.addActor(actorFrom(lpd, [0.05, 0.05, 0.05], 2));

    renderer.resetCamera();
    renderer.getActiveCamera().setViewUp(0, 1, 0);
    renderer.resetCameraClippingRange();
    grw.resize();
    grw.getRenderWindow().render();

    const onResize = () => { grw.resize(); grw.getRenderWindow().render(); };
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); grw.delete(); };
  }, [gt, k, surface]);

  if (err) return <div className="err">weld GT: {err}</div>;
  if (!gt) return <div className="drop">loading the weld ground truth…</div>;

  const segs = gt.welds.reduce((s, g) => s + g.length - 1, 0);
  return (
    <>
      <div className="bar">
        <span className="pill name">{gt.source}</span>
        <span className="pill">{gt.panelSizes.length} panels</span>
        <span className="pill">{gt.welds.length} weld groups</span>
        <span className="pill ok">{segs} merged vertices</span>
        <span className="pill">{segs === gt.merged ? "matches CLO's merge count" : "MISMATCH"}</span>
      </div>
      <div className="toggles">
        <label>
          explode
          <input type="range" min={0} max={1.5} step={0.05} value={k}
                 onChange={(e) => setK(Number(e.target.value))} />
          {k.toFixed(2)}
        </label>
        <label>
          <input type="checkbox" checked={surface}
                 onChange={(e) => setSurface(e.target.checked)} />
          panel surfaces
        </label>
      </div>
      <div className="vtk" ref={host} />
    </>
  );
}
