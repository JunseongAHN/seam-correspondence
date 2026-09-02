/* Port of autosew/features.py.  24 dims per edge, cycle graph within each panel. */
import { KT, type Pattern, type Pt } from "./parseSpec";

export const F_DIM = 24;
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

export interface Tensors {
  x: Float32Array;            // (M, 24) row-major
  nbr: BigInt64Array;         // (M, 2)
  M: number;
  keys: Array<[string, number]>;   // node -> (panel, original edge idx)
}

export function patternToTensors(p: Pattern): Tensors {
  const keys: Array<[string, number]> = [];
  for (const panel of p.panels) for (const e of panel.edges) keys.push([panel.name, e.idxInPanel]);
  const M = keys.length;
  const x = new Float32Array(M * F_DIM);
  const nbr = new BigInt64Array(M * 2);

  let node = 0;
  for (const panel of p.panels) {
    const n = panel.edges.length;
    const base = node;
    const ang = interiorAngles(panel.edges);
    for (let j = 0; j < n; j++) {
      const e = panel.edges[j];
      const o = node * F_DIM;
      const dx = e.end[0] - e.start[0], dy = e.end[1] - e.start[1];
      const L = Math.hypot(dx, dy);
      const [ox, oy] = L > 0 ? [dx / L, dy / L] : [0, 0];
      x[o + 0] = e.start[0] / SCALE; x[o + 1] = e.start[1] / SCALE;
      x[o + 2] = e.end[0] / SCALE;   x[o + 3] = e.end[1] / SCALE;
      x[o + 4] = L / SCALE;
      x[o + 5] = ox; x[o + 6] = oy;
      x[o + 7] = e.kt;                                   // raw 0..5
      for (let q = 0; q < Math.min(10, e.kparams.length); q++) {
        // circle params are [radius, flag, flag]: only the radius is a length
        const v = e.kt === KT.CIRCLE ? (q === 0 ? e.kparams[q] / SCALE : e.kparams[q])
                                     : e.kparams[q] / SCALE;
        x[o + 8 + q] = v;
      }
      const aL = ang[j], aR = ang[(j + 1) % n];
      x[o + 18] = Math.sin(aL); x[o + 19] = Math.cos(aL);
      x[o + 20] = Math.sin(aR); x[o + 21] = Math.cos(aR);
      x[o + 22] = Math.min(Math.max((n - EDGE_MIN) / Math.max(EDGE_MAX - EDGE_MIN, 1e-6), 0), 1);
      x[o + 23] = panel.orderIdx / MAX_PANELS;
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
