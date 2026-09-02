/* DXF (ASTM-ish, as CLO exports) -> the same Pattern shape parseSpec.ts produces.
   Port of dxfcheck/dxf_to_features.py; keep the two in step.

   What a DXF gives us and a specification.json does not: nothing.  What it costs:
   the boundary arrives as a polyline, so the curve TYPE and its control points are
   gone.  For the tagged encoding they have to be refitted, and that refit is where
   this path loses accuracy -- a circular arc and a weakly-curved quadratic Bezier fit
   the samples about equally well, so k_t flips and eleven feature dimensions change
   meaning at once.  Measured end to end on a synthetic round trip: F1 1.000 -> 0.783.

   There is also no stitch ground truth in the file, so the caller gets an empty
   `stitches` and the UI has nothing to score against. */
import DxfParser from "dxf-parser";
import { KT, type Edge, type Panel, type Pattern, type Pt } from "./parseSpec";

const MM = 10.0;               // CLO exports mm; the model works in cm
const BOUNDARY_LAYERS = ["8", "1"];   // preferred first
const TURN_LAYER = "2";

type Raw = { name: string; pts: Pt[]; turns: Pt[] };

function blockPolyline(ents: any[]): Pt[] | null {
  for (const layer of BOUNDARY_LAYERS) {
    const p = ents.find((e) => e.type === "POLYLINE" && String(e.layer) === layer);
    if (p?.vertices?.length) return p.vertices.map((v: any) => [v.x / MM, v.y / MM] as Pt);
  }
  return null;
}

function readBlocks(dxf: any): Raw[] {
  const out: Raw[] = [];
  for (const name of Object.keys(dxf.blocks ?? {})) {
    if (name.startsWith("*")) continue;
    const ents = dxf.blocks[name].entities ?? [];
    const pts = blockPolyline(ents);
    if (!pts) continue;
    const turns = ents
      .filter((e: any) => e.type === "POINT" && String(e.layer) === TURN_LAYER)
      .map((e: any) => [e.position.x / MM, e.position.y / MM] as Pt);
    out.push({ name, pts, turns });
  }
  return out;
}

/** Turn points -> indices into the boundary polyline, deduplicated and ordered. */
function edgeStarts(pts: Pt[], turns: Pt[]): number[] {
  const idx = new Set<number>();
  for (const t of turns) {
    let best = 0, bd = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const d = (pts[i][0] - t[0]) ** 2 + (pts[i][1] - t[1]) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    idx.add(best);
  }
  return [...idx].sort((a, b) => a - b);
}

/** One edge's sampled points -> (k_t, absolute params).  Port of fit_edge(). */
export function fitEdge(seg: Pt[]): [number, number[]] {
  const p0 = seg[0], p1 = seg[seg.length - 1];
  const cx = p1[0] - p0[0], cy = p1[1] - p0[1];
  const L = Math.hypot(cx, cy);
  if (seg.length <= 2 || L < 1e-12) return [KT.STRAIGHT, []];

  const nx = -cy / L, ny = cx / L;
  const t = seg.map((q) => ((q[0] - p0[0]) * cx + (q[1] - p0[1]) * cy) / (L * L));
  const off = seg.map((q) => (q[0] - p0[0]) * nx + (q[1] - p0[1]) * ny);
  if (Math.max(...off.map(Math.abs)) < 1e-6 * Math.max(L, 1.0)) return [KT.STRAIGHT, []];

  // least squares for one free control point (quadratic) and two (cubic)
  const solve = (basis: number[][], rhs: Pt[]): number[][] => {
    // normal equations, k x k with k = basis[0].length (1 or 2)
    const k = basis[0].length;
    const A = Array.from({ length: k }, () => new Array(k).fill(0));
    const b = Array.from({ length: k }, () => [0, 0]);
    for (let i = 0; i < basis.length; i++) {
      for (let r = 0; r < k; r++) {
        for (let c = 0; c < k; c++) A[r][c] += basis[i][r] * basis[i][c];
        b[r][0] += basis[i][r] * rhs[i][0];
        b[r][1] += basis[i][r] * rhs[i][1];
      }
    }
    if (k === 1) return [[b[0][0] / (A[0][0] || 1e-12), b[0][1] / (A[0][0] || 1e-12)]];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0] || 1e-12;
    return [
      [(b[0][0] * A[1][1] - A[0][1] * b[1][0]) / det, (b[0][1] * A[1][1] - A[0][1] * b[1][1]) / det],
      [(A[0][0] * b[1][0] - b[0][0] * A[1][0]) / det, (A[0][0] * b[1][1] - b[0][1] * A[1][0]) / det],
    ];
  };

  const b2 = t.map((u) => [2 * (1 - u) * u]);
  const r2 = seg.map((q, i) => [q[0] - (1 - t[i]) ** 2 * p0[0] - t[i] ** 2 * p1[0],
                                q[1] - (1 - t[i]) ** 2 * p0[1] - t[i] ** 2 * p1[1]] as Pt);
  const q2 = solve(b2, r2)[0];
  let resQ = 0;
  for (let i = 0; i < seg.length; i++) {
    const u = t[i], w0 = (1 - u) ** 2, w1 = 2 * (1 - u) * u, w2 = u * u;
    resQ = Math.max(resQ, Math.abs(w0 * p0[0] + w1 * q2[0] + w2 * p1[0] - seg[i][0]),
                          Math.abs(w0 * p0[1] + w1 * q2[1] + w2 * p1[1] - seg[i][1]));
  }

  const b3 = t.map((u) => [3 * (1 - u) ** 2 * u, 3 * (1 - u) * u * u]);
  const r3 = seg.map((q, i) => [q[0] - (1 - t[i]) ** 3 * p0[0] - t[i] ** 3 * p1[0],
                                q[1] - (1 - t[i]) ** 3 * p0[1] - t[i] ** 3 * p1[1]] as Pt);
  const c3 = solve(b3, r3);
  let resC = 0;
  for (let i = 0; i < seg.length; i++) {
    const u = t[i];
    const w0 = (1 - u) ** 3, w1 = 3 * (1 - u) ** 2 * u, w2 = 3 * (1 - u) * u * u, w3 = u ** 3;
    resC = Math.max(resC, Math.abs(w0 * p0[0] + w1 * c3[0][0] + w2 * c3[1][0] + w3 * p1[0] - seg[i][0]),
                          Math.abs(w0 * p0[1] + w1 * c3[0][1] + w2 * c3[1][1] + w3 * p1[1] - seg[i][1]));
  }

  // circle: centre equidistant from every sample, solved as 2*cx*x + 2*cy*y + k = x^2+y^2
  let Sxx = 0, Sxy = 0, Syy = 0, Sx = 0, Sy = 0, Sn = seg.length;
  let Sxz = 0, Syz = 0, Sz = 0;
  for (const [x, y] of seg) {
    const z = x * x + y * y;
    Sxx += x * x; Sxy += x * y; Syy += y * y; Sx += x; Sy += y;
    Sxz += x * z; Syz += y * z; Sz += z;
  }
  // 3x3 solve for [2cx, 2cy, k]
  const M3 = [[Sxx, Sxy, Sx], [Sxy, Syy, Sy], [Sx, Sy, Sn]];
  const v3 = [Sxz, Syz, Sz];
  const det3 =
    M3[0][0] * (M3[1][1] * M3[2][2] - M3[1][2] * M3[2][1]) -
    M3[0][1] * (M3[1][0] * M3[2][2] - M3[1][2] * M3[2][0]) +
    M3[0][2] * (M3[1][0] * M3[2][1] - M3[1][1] * M3[2][0]);
  let resA = Infinity, r = 0, ctr: Pt = [0, 0];
  if (Math.abs(det3) > 1e-18) {
    const inv = (i: number) => {
      const m = M3.map((row, ri) => row.map((val, ci) => (ci === i ? v3[ri] : val)));
      return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det3;
    };
    ctr = [inv(0) / 2, inv(1) / 2];
    r = Math.sqrt(Math.max(inv(2) + ctr[0] * ctr[0] + ctr[1] * ctr[1], 0));
    resA = 0;
    for (const [x, y] of seg)
      resA = Math.max(resA, Math.abs(Math.hypot(x - ctr[0], y - ctr[1]) - r));
  }

  const tol = 1e-4 * Math.max(L, 1.0);
  if (resA <= tol && resA <= resQ) {
    const m = seg[Math.floor(seg.length / 2)];
    const cross = cx * (m[1] - p0[1]) - cy * (m[0] - p0[0]);
    const ang = 2 * Math.asin(Math.min(L / (2 * Math.max(r, L / 2)), 1.0));
    return [KT.CIRCLE, [r, ang > Math.PI ? 1 : 0, cross > 0 ? 1 : 0]];
  }
  if (resQ <= tol) return [KT.QUADRATIC, [q2[0], q2[1]]];
  return [KT.CUBIC, [c3[0][0], c3[0][1], c3[1][0], c3[1][1]]];
}

export function parseDxf(text: string, name: string): Pattern {
  const dxf = new DxfParser().parseSync(text);
  const raws = readBlocks(dxf);
  if (!raws.length) throw new Error("no panel blocks with a boundary polyline (layer 8 or 1)");

  const panels: Panel[] = [];
  raws.forEach((raw, pi) => {
    const starts = edgeStarts(raw.pts, raw.turns);
    if (starts.length < 2) return;
    const n = starts.length;
    let segs: Pt[][] = [];
    for (let k = 0; k < n; k++) {
      const a = starts[k], b = starts[(k + 1) % n];
      segs.push(b > a ? raw.pts.slice(a, b + 1)
                      : [...raw.pts.slice(a), ...raw.pts.slice(0, b + 1)]);
    }
    // ACW canonicalisation on the chord polygon, exactly as the parser does
    const chords = segs.map((s) => s[0]);
    let area = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += chords[i][0] * chords[j][1] - chords[j][0] * chords[i][1];
    }
    if (area < 0) segs = segs.map((s) => [...s].reverse()).reverse();

    // panel-local frame: bbox lower-left to the origin (features use this)
    const all = segs.flat();
    const mx = Math.min(...all.map((q) => q[0])), my = Math.min(...all.map((q) => q[1]));
    const local = segs.map((s) => s.map(([x, y]) => [x - mx, y - my] as Pt));

    const edges: Edge[] = local.map((s, j) => {
      const [kt, kp] = fitEdge(s);
      return { panel: raw.name, idxInPanel: j, start: s[0], end: s[s.length - 1], kt, kparams: kp };
    });

    panels.push({
      name: raw.name, orderIdx: pi, edges, nEdgesRaw: n, reversed: area < 0,
      // rendering frame: the DXF coordinates already carry the 2D layout, so the
      // panel needs no transform of its own -- and the sampled polyline is carried
      // on the edge so the drawing does not have to re-derive the curve
      rawVerts: segs.map((s) => s[0]),
      rawEdges: segs.map((s, j) => ({ endpoints: [j, (j + 1) % n], curvature: null, poly: s })),
      translation: [0, 0, 0], rotation: [0, 0, 0],
    });
  });

  if (!panels.length) throw new Error("no panel had at least 2 turn points on layer 2");
  return { name, panels, stitches: [] };
}
