/* Where every panel, edge and label sits in SVG user units.  The pattern's own frame has y
   up; the SVG's has y down, so everything is flipped about the top of the bounding box. */
import type { Pt } from "@/core-algorithm/pattern";
import { edgeMid, edgePolyline, type Placed } from "./render";

export type Frame = { x0: number; y1: number; pad: number; W: number; H: number };

export function svgFrame(placed: Placed[]): Frame {
  const pts = placed.flatMap((p) => p.verts);
  const xs = pts.map((v) => v[0]), ys = pts.map((v) => v[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pad = 12;
  return { x0, y1, pad, W: x1 - x0 + 2 * pad, H: y1 - y0 + 2 * pad };
}

/* Path strings, midpoints and arc lengths depend on the pattern, not on the viewBox, so
   they are built once.  Recomputing 48 polylines on every pan frame is what made
   dragging stutter even after the coordinate feedback was fixed. */
export function patternGeometry(placed: Placed[], { x0, y1, pad }: Frame) {
  const tx = (p: Pt) => `${p[0] - x0 + pad},${y1 - p[1] + pad}`;
  return placed.map((p) => {
    const cx = p.verts.reduce((s, q) => s + q[0], 0) / p.verts.length - x0 + pad;
    const cy = y1 - p.verts.reduce((s, q) => s + q[1], 0) / p.verts.length + pad;
    const edges = p.edges.map((_: unknown, i: number) => {
      const poly = edgePolyline(p, i);
      let arc = 0;
      for (let q = 1; q < poly.length; q++)
        arc += Math.hypot(poly[q][0] - poly[q - 1][0], poly[q][1] - poly[q - 1][1]);
      if (p.blowUp) arc /= p.blowUp;      // undo the display stretch on a thin piece
      const m = edgeMid(p, i);
      return { d: `M ${poly.map(tx).join(" L ")}`, arc,
               mx: m[0] - x0 + pad, my: y1 - m[1] + pad };
    });
    return { name: p.name, blowUp: p.blowUp, cx, cy, edges };
  });
}

export type PanelGeom = ReturnType<typeof patternGeometry>[number];

/** node -> its edge's midpoint, where stitch lines start and end */
export function midpointsByNode(geom: PanelGeom[], node: Map<string, number>) {
  const m = new Map<number, [number, number]>();
  geom.forEach((g) => g.edges.forEach((e, i) => {
    const n = node.get(`${g.name}#${i}`);
    if (n !== undefined) m.set(n, [e.mx, e.my]);
  }));
  return m;
}
