/* The ground-truth drape, beside the predicted stitching.

   It is drawn RED and labelled as ground truth on purpose.  Everything else in this
   page is a prediction, and a 3D shape is persuasive in a way a stitch list is not --
   so the one pane that is *not* the model's output has to be impossible to mistake for
   it.

   The shape is solved from the ground-truth seams, on the host, by the native build of
   the same solver the demo ships to wasm.  Two things make that a fair reference:

   - the pairing along a seam is not inferred.  gcd_io welds the coincident vertices of
     the simulated mesh, so which vertex meets which is recovered exactly.
   - the solve starts from the specification's own per-panel 3D placement, never from a
     drape.  Each panel therefore begins on its own side of the body, which is what stops
     a seam from closing inside-out and taking the surface through itself. */
import { useEffect, useRef, useState } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkCylinderSource from "@kitware/vtk.js/Filters/Sources/CylinderSource";
import vtkSphereSource from "@kitware/vtk.js/Filters/Sources/SphereSource";
import { loadDrape, type Drape } from "./lib/drape";

const RED: [number, number, number] = [0.86, 0.13, 0.15];

function bodyActors(d: Drape) {
  const actors: vtkActor[] = [];
  const grey = (a: vtkActor) => {
    a.getProperty().setColor(0.72, 0.72, 0.76);
    a.getProperty().setOpacity(0.28);
  };
  for (let k = 0; k < d.cyl.length / 7; k++) {
    const a = [d.cyl[k * 7], d.cyl[k * 7 + 1], d.cyl[k * 7 + 2]];
    const b = [d.cyl[k * 7 + 3], d.cyl[k * 7 + 4], d.cyl[k * 7 + 5]];
    const dir = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const h = Math.hypot(dir[0], dir[1], dir[2]) || 1e-9;
    const src = vtkCylinderSource.newInstance({
      height: h, radius: d.cyl[k * 7 + 6], resolution: 32, capping: true,
      center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2],
      direction: [dir[0] / h, dir[1] / h, dir[2] / h],
    });
    const mapper = vtkMapper.newInstance();
    mapper.setInputConnection(src.getOutputPort());
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper); grey(actor); actors.push(actor);
  }
  for (let k = 0; k < d.sph.length / 4; k++) {
    const src = vtkSphereSource.newInstance({
      center: [d.sph[k * 4], d.sph[k * 4 + 1], d.sph[k * 4 + 2]],
      radius: d.sph[k * 4 + 3], phiResolution: 24, thetaResolution: 32,
    });
    const mapper = vtkMapper.newInstance();
    mapper.setInputConnection(src.getOutputPort());
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper); grey(actor); actors.push(actor);
  }
  return actors;
}

export default function GtDrape({ id, exact, fp, fn }:
                                { id: string; exact: boolean; fp: number; fn: number }) {
  const host = useRef<HTMLDivElement>(null);
  const [drape, setDrape] = useState<Drape | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showBody, setShowBody] = useState(true);
  const [wire, setWire] = useState(false);

  useEffect(() => {
    setDrape(null); setErr(null);
    let live = true;
    loadDrape(id).then((d) => { if (live) setDrape(d); })
                 .catch((e) => { if (live) setErr(String(e?.message ?? e)); });
    return () => { live = false; };
  }, [id]);

  useEffect(() => {
    if (!drape || !host.current) return;
    const grw = vtkGenericRenderWindow.newInstance({ background: [1, 1, 1] });
    grw.setContainer(host.current);
    const renderer = grw.getRenderer();

    const polys = new Uint32Array(drape.M * 4);
    for (let t = 0, o = 0; t < drape.M; t++, o += 4) {
      polys[o] = 3;
      polys[o + 1] = drape.faces[t * 3];
      polys[o + 2] = drape.faces[t * 3 + 1];
      polys[o + 3] = drape.faces[t * 3 + 2];
    }
    const pd = vtkPolyData.newInstance();
    pd.getPoints().setData(drape.pos, 3);
    pd.getPolys().setData(polys);
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(pd);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(...RED);
    if (wire) {
      actor.getProperty().setEdgeVisibility(true);
      actor.getProperty().setEdgeColor(0.35, 0.05, 0.06);
    }
    renderer.addActor(actor);
    if (showBody) for (const a of bodyActors(drape)) renderer.addActor(a);

    renderer.resetCamera();
    renderer.getActiveCamera().setViewUp(0, 1, 0);
    renderer.resetCameraClippingRange();
    grw.resize();
    grw.getRenderWindow().render();
    const onResize = () => { grw.resize(); grw.getRenderWindow().render(); };
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); grw.delete(); };
  }, [drape, showBody, wire]);

  return (
    <>
      <p className="note gtnote">
        <strong>Red is the ground truth, not the prediction.</strong> This garment was
        assembled from its ground-truth stitching by the same solver the demo ships,
        run once on the host, so what you see is the shape the correct answer produces.
        It is an isometric assembly around the body proxy, not a cloth drape under
        gravity.
        {exact
          ? " The prediction for this garment is identical to that ground truth, so this"
            + " is equally the drape of what the model predicted."
          : ` The prediction here has ${fp} false positive${fp === 1 ? "" : "s"} and `
            + `${fn} missed stitch${fn === 1 ? "" : "es"}, and a wrong stitching cannot`
            + " be assembled: sewing an edge to the wrong partner drags the surface"
            + " through itself and tears the mesh. Only the ground truth is solved here."}
      </p>
      <p className="note">
        A seam can also be sewn the right pair but the wrong way round. Nothing here
        guesses that: the vertex-to-vertex correspondence comes from welding the
        coincident vertices of the simulated mesh, and the solve starts from the
        specification's own per-panel 3D placement rather than from any drape — so each
        panel begins on its own side of the body and the seams close without twisting.
      </p>
      {err && <div className="err">{err}</div>}
      {!drape && !err && <div className="drop">loading the ground-truth drape…</div>}
      {drape && (
        <>
          <div className="bar">
            <span className="pill name gt">ground-truth drape</span>
            {/* the headline fact about this pane, where the eye already is */}
            <span className={`pill ${exact ? "ok" : "fp"}`}>
              {exact ? "prediction identical — this is its assembly too"
                     : `prediction differs by ${fp + fn} stitch${fp + fn === 1 ? "" : "es"}`}
            </span>
            <span className="pill">{drape.n} verts</span>
            <span className="pill">{drape.M} faces</span>
            <span className="pill">solved on the host, native build</span>
          </div>
          <div className="toggles">
            <label><input type="checkbox" checked={showBody}
                          onChange={(e) => setShowBody(e.target.checked)} /> body capsules</label>
            <label><input type="checkbox" checked={wire}
                          onChange={(e) => setWire(e.target.checked)} /> mesh edges</label>
          </div>
          <div className="vtk" ref={host} />
        </>
      )}
    </>
  );
}
