/* Port of autosew/features.py.  24 dims per edge, cycle graph within each panel.

   Two curvature encodings, matching the Python.  "tagged" is the paper's: dim 7 is the
   curve type and dims 8..17 are ten slots whose meaning that type selects.  "sagitta"
   replaces dims 7..17 with K signed sagitta values (see curves.ts) and is what the
   DXF/FBX industrial track uses, because a DXF has no exact curve type to give.

   With ARC_FEATURES two dims are appended: the edge's ARC length and arc/chord.  Dim 4
   is the chord, but what a seam matches is the arc -- the length of fabric you sew
   along -- and the two differ by up to a third on a real collar. */
import { edgePolyline, sagittaProfile } from "./curves";
import { KT, type Pattern, type Pt } from "./parseSpec";

export const F_DIM = 24;
export const N_CURV = 11;          // dims 7..17: the curvature block, whatever encodes it
export const N_ARC = 2;            // appended when ARC_FEATURES: arc/100, arc/chord

export type CurvatureEncoding = "tagged" | "sagitta";
/** Must match the checkpoint the ONNX model was exported from. */
export const ENCODING: CurvatureEncoding = "sagitta";
export const SAGITTA_SAMPLES = 11;
/** Must match the checkpoint too: the shipped model is trained with them. */
export const ARC_FEATURES = true;
const SCALE = 100.0;          // cfg.scale_div
const MAX_PANELS = 37.0;      // cfg.max_panels_norm
const EDGE_MIN = 2.0, EDGE_MAX = 40.0;   // cfg.edge_count_minmax

/** Interior angle at the start vertex of each edge, from chord directions (ACW cycle). */
function interiorAngles(edges: { start: Pt; end: Pt }[]): number[] {
  const dirs = edges.map((e) => {
    const dx = e.end[0] - e.start[0], dy = e.end[1] - e.start[1];
    const L = Math.hypot(dx, dy) || 1.0;
    return [dx / L, dy / L] as Pt;
  });
  return dirs.map((b, i) => {
    const a = dirs[(i - 1 + dirs.length) % dirs.length];
    const turn = Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
    return Math.PI - turn;
  });
}

/** Width of the curvature block, and of a whole feature row, for the active encoding. */
export const NK = ENCODING === "sagitta" ? SAGITTA_SAMPLES : N_CURV;
export const featureDim = F_DIM - N_CURV + NK + (ARC_FEATURES ? N_ARC : 0);

export interface Tensors {
  x: Float32Array;            // (M, featureDim) row-major
  nbr: BigInt64Array;         // (M, 2)
  M: number;
  keys: Array<[string, number]>;   // node -> (panel, original edge idx)
}

export function patternToTensors(p: Pattern): Tensors {
  const keys: Array<[string, number]> = [];
  for (const panel of p.panels) for (const e of panel.edges) keys.push([panel.name, e.idxInPanel]);
  const M = keys.length;
  const x = new Float32Array(M * featureDim);
  const nbr = new BigInt64Array(M * 2);

  let node = 0;
  for (const panel of p.panels) {
    const n = panel.edges.length;
    const base = node;
    const ang = interiorAngles(panel.edges);
    for (let j = 0; j < n; j++) {
      const e = panel.edges[j];
      const o = node * featureDim;
      const dx = e.end[0] - e.start[0], dy = e.end[1] - e.start[1];
      const L = Math.hypot(dx, dy);
      const [ox, oy] = L > 0 ? [dx / L, dy / L] : [0, 0];
      x[o + 0] = e.start[0] / SCALE; x[o + 1] = e.start[1] / SCALE;
      x[o + 2] = e.end[0] / SCALE;   x[o + 3] = e.end[1] / SCALE;
      x[o + 4] = L / SCALE;
      x[o + 5] = ox; x[o + 6] = oy;
      const poly = ENCODING === "sagitta" || ARC_FEATURES ? edgePolyline(e) : null;
      if (ENCODING === "sagitta") {
        sagittaProfile(poly!, SAGITTA_SAMPLES, x, o + 7);
      } else {
        x[o + 7] = e.kt;                                   // raw 0..5
        for (let q = 0; q < Math.min(10, e.kparams.length); q++) {
          // circle params are [radius, flag, flag]: only the radius is a length
          const v = e.kt === KT.CIRCLE ? (q === 0 ? e.kparams[q] / SCALE : e.kparams[q])
                                       : e.kparams[q] / SCALE;
          x[o + 8 + q] = v;
        }
      }
      const t = o + 7 + NK;                                // first index after curvature
      const aL = ang[j], aR = ang[(j + 1) % n];
      x[t + 0] = Math.sin(aL); x[t + 1] = Math.cos(aL);
      x[t + 2] = Math.sin(aR); x[t + 3] = Math.cos(aR);
      x[t + 4] = Math.min(Math.max((n - EDGE_MIN) / Math.max(EDGE_MAX - EDGE_MIN, 1e-6), 0), 1);
      x[t + 5] = panel.orderIdx / MAX_PANELS;
      if (ARC_FEATURES) {
        let arc = 0;
        for (let q = 1; q < poly!.length; q++)
          arc += Math.hypot(poly![q][0] - poly![q - 1][0], poly![q][1] - poly![q - 1][1]);
        x[t + 6] = arc / SCALE;
        x[t + 7] = L > 1e-9 ? arc / L : 1.0;
      }
      nbr[node * 2 + 0] = BigInt(base + ((j - 1 + n) % n));
      nbr[node * 2 + 1] = BigInt(base + ((j + 1) % n));
      node++;
    }
  }
  return { x, nbr, M, keys };
}

/** GT pairs as node-index pairs (for scoring the prediction in the UI). */
export function gtPairs(p: Pattern, keys: Array<[string, number]>): Set<string> {
  const key2node = new Map(keys.map(([pn, ei], i) => [`${pn}#${ei}`, i]));
  const out = new Set<string>();
  for (const sides of p.stitches) {
    const nodes = sides.map(([pn, ei]) => key2node.get(`${pn}#${ei}`))
                       .filter((v): v is number => v !== undefined).sort((a, b) => a - b);
    for (let a = 0; a < nodes.length; a++)
      for (let b = a + 1; b < nodes.length; b++) out.add(`${nodes[a]}-${nodes[b]}`);
  }
  return out;
}
