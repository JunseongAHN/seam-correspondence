/* Panels placed in 3D from the specification transform, as polylines.
   Same convention as autosew/viz/visualize3d.py: xyz euler in degrees, R = Rz Ry Rx,
   then translate.  Units are cm. */
import type { Pattern, Pt } from "./parseSpec";
import { edgePolylineLocal } from "./render";

export type V3 = [number, number, number];

function eulerXYZ(rot: number[]): number[][] {
  const [rx, ry, rz] = rot.map((d) => (d * Math.PI) / 180);
  const [cx, sx] = [Math.cos(rx), Math.sin(rx)];
  const [cy, sy] = [Math.cos(ry), Math.sin(ry)];
  const [cz, sz] = [Math.cos(rz), Math.sin(rz)];
  const Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]];
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
  const mm = (A: number[][], B: number[][]) =>
    A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  return mm(mm(Rz, Ry), Rx);
}

export interface PanelMesh {
  name: string;
  points: Float32Array;     // flat xyz
  lines: Uint32Array;       // vtk cell array: [n, i0..in-1, ...]
  edgeMid: V3[];            // one per edge, in ORIGINAL json edge order
}

/** Polyline geometry for every panel, plus the 3D midpoint of each edge. */
export function panelMeshes(p: Pattern): PanelMesh[] {
  return p.panels.map((panel) => {
    const R = eulerXYZ(panel.rotation);
    const t = panel.translation;
    const to3 = (q: Pt): V3 => [
      R[0][0] * q[0] + R[0][1] * q[1] + t[0],
      R[1][0] * q[0] + R[1][1] * q[1] + t[1],
      R[2][0] * q[0] + R[2][1] * q[1] + t[2],
    ];
    const pts: number[] = [];
    const lines: number[] = [];
    const mids: V3[] = [];
    for (let i = 0; i < panel.rawEdges.length; i++) {
      const poly = edgePolylineLocal(panel, i).map(to3);
      const base = pts.length / 3;
      for (const v of poly) pts.push(v[0], v[1], v[2]);
      lines.push(poly.length, ...poly.map((_, k) => base + k));
      // arclength midpoint, so a straight edge does not report its end vertex
      let total = 0; const cum = [0];
      for (let k = 1; k < poly.length; k++) {
        total += Math.hypot(poly[k][0] - poly[k - 1][0], poly[k][1] - poly[k - 1][1],
                            poly[k][2] - poly[k - 1][2]);
        cum.push(total);
      }
      const half = total / 2;
      let k = 1; while (k < cum.length - 1 && cum[k] < half) k++;
      const f = (half - cum[k - 1]) / Math.max(cum[k] - cum[k - 1], 1e-12);
      mids.push([
        poly[k - 1][0] + f * (poly[k][0] - poly[k - 1][0]),
        poly[k - 1][1] + f * (poly[k][1] - poly[k - 1][1]),
        poly[k - 1][2] + f * (poly[k][2] - poly[k - 1][2]),
      ]);
    }
    return { name: panel.name, points: new Float32Array(pts),
             lines: new Uint32Array(lines), edgeMid: mids };
  });
}

/** Straight segments between the midpoints of paired edges, as one polydata. */
export function stitchLines(meshes: PanelMesh[], keys: Array<[string, number]>,
                            pairs: Iterable<string>) {
  const byName = new Map(meshes.map((m) => [m.name, m]));
  const pts: number[] = []; const lines: number[] = [];
  for (const k of pairs) {
    const [a, b] = k.split("-").map(Number);
    const [pa, ea] = keys[a], [pb, eb] = keys[b];
    const ma = byName.get(pa)?.edgeMid[ea], mb = byName.get(pb)?.edgeMid[eb];
    if (!ma || !mb) continue;
    const i = pts.length / 3;
    pts.push(ma[0], ma[1], ma[2], mb[0], mb[1], mb[2]);
    lines.push(2, i, i + 1);
  }
  return { points: new Float32Array(pts), lines: new Uint32Array(lines) };
}
