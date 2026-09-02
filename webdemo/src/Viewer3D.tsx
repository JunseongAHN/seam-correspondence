import { useEffect, useRef } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import { panelMeshes, stitchLines } from "./lib/panels3d";
import type { Pattern } from "./lib/parseSpec";

function polyData(points: Float32Array, lines: Uint32Array) {
  const pd = vtkPolyData.newInstance();
  pd.getPoints().setData(points, 3);
  pd.getLines().setData(lines);
  return pd;
}

function lineActor(points: Float32Array, lines: Uint32Array,
                   color: [number, number, number], width = 1) {
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(polyData(points, lines));
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setColor(...color);
  actor.getProperty().setLineWidth(width);
  return actor;
}

const PANEL_COLORS: [number, number, number][] = [
  [0.20, 0.47, 0.85], [0.95, 0.55, 0.15], [0.20, 0.65, 0.35], [0.85, 0.25, 0.30],
  [0.55, 0.35, 0.75], [0.60, 0.42, 0.30], [0.90, 0.45, 0.70], [0.45, 0.45, 0.45],
  [0.75, 0.75, 0.20], [0.20, 0.70, 0.80],
];

export default function Viewer3D({
  pattern, keys, pred, gt, show,
}: {
  pattern: Pattern; keys: Array<[string, number]>;
  pred: Set<string>; gt: Set<string>;
  show: { correct: boolean; fp: boolean; fn: boolean };
}) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!host.current) return;
    const grw = vtkGenericRenderWindow.newInstance({ background: [1, 1, 1] });
    grw.setContainer(host.current);
    const renderer = grw.getRenderer();

    const meshes = panelMeshes(pattern);
    meshes.forEach((m, i) =>
      renderer.addActor(lineActor(m.points, m.lines, PANEL_COLORS[i % PANEL_COLORS.length], 2)));

    const ok = [...pred].filter((k) => gt.has(k));
    const fp = [...pred].filter((k) => !gt.has(k));
    const fn = [...gt].filter((k) => !pred.has(k));
    const overlay: Array<[string[], [number, number, number], number, boolean]> = [
      [ok, [0.09, 0.64, 0.29], 2, show.correct],
      [fp, [0.86, 0.15, 0.15], 3, show.fp],
      [fn, [0.10, 0.10, 0.10], 2, show.fn],
    ];
    for (const [set, color, w, on] of overlay) {
      if (!on || !set.length) continue;
      const { points, lines } = stitchLines(meshes, keys, set);
      if (points.length) renderer.addActor(lineActor(points, lines, color, w));
    }

    renderer.resetCamera();
    // look at the garment from the front, y up
    const cam = renderer.getActiveCamera();
    cam.setViewUp(0, 1, 0);
    cam.azimuth(0);
    renderer.resetCameraClippingRange();
    grw.resize();
    grw.getRenderWindow().render();

    const onResize = () => { grw.resize(); grw.getRenderWindow().render(); };
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); grw.delete(); };
  }, [pattern, keys, pred, gt, show.correct, show.fp, show.fn]);

  return <div className="vtk" ref={host} />;
}
