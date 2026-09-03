/* Port of autosew/gcd_parser.py.  Keep in step with it: the ONNX model was
   trained on exactly these numbers, so any divergence silently degrades the
   prediction rather than raising. */

export const KT = { STRAIGHT: 0, CIRCLE: 1, QUADRATIC: 2, CUBIC: 3, BSPLINE: 4, UNKNOWN: 5 };

export type Pt = [number, number];
export interface Edge {
  panel: string; idxInPanel: number;      // ORIGINAL json index; stitches refer to this
  start: Pt; end: Pt; kt: number; kparams: number[];
  poly?: Pt[];          // DXF input: the sampled boundary, used instead of refitting
}
export interface Panel {
  name: string; orderIdx: number; edges: Edge[]; nEdgesRaw: number; reversed: boolean;
  rawVerts: Pt[]; rawEdges: any[];        // kept for drawing in the original frame
  translation: number[]; rotation: number[];
}
export interface Pattern {
  name: string; panels: Panel[];
  stitches: Array<Array<[string, number]>>;
}

const perp = (v: Pt): Pt => [-v[1], v[0]];

function relToAbs(s: Pt, e: Pt, c: Pt): Pt {
  const ev: Pt = [e[0] - s[0], e[1] - s[1]];
  const pv = perp(ev);
  return [s[0] + c[0] * ev[0] + c[1] * pv[0], s[1] + c[0] * ev[1] + c[1] * pv[1]];
}

function parseCurvature(curv: any, s: Pt, e: Pt, rel: boolean): [number, number[], number[]] {
  const conv = (c: Pt): [Pt, Pt] => (rel ? [relToAbs(s, e, c), [c[0], c[1]]] : [[c[0], c[1]], [c[0], c[1]]]);
  if (curv === null || curv === undefined) return [KT.STRAIGHT, [], []];
  if (Array.isArray(curv) && curv.length === 2 && curv.every((t) => typeof t === "number")) {
    const [a, r] = conv(curv as Pt);
    return [KT.QUADRATIC, [a[0], a[1]], [r[0], r[1]]];
  }
  if (typeof curv === "object") {
    const t = String(curv.type ?? "").toLowerCase();
    const params = curv.params ?? [];
    if (t.includes("quadratic")) {
      const [a, r] = conv(params[0]);
      return [KT.QUADRATIC, [a[0], a[1]], [r[0], r[1]]];
    }
    if (t.includes("cubic")) {
      const A: number[] = [], R: number[] = [];
      for (const c of params.slice(0, 2)) { const [a, r] = conv(c); A.push(a[0], a[1]); R.push(r[0], r[1]); }
      return [KT.CUBIC, A, R];
    }
    if (t.includes("circ") || t.includes("arc") || t.includes("spline")) {
      const flat: number[] = [];
      for (const p of params) {
        if (Array.isArray(p)) flat.push(...p.map(Number));
        else if (typeof p === "boolean") flat.push(p ? 1 : 0);
        else flat.push(Number(p));
      }
      const kt = t.includes("spline") ? KT.BSPLINE : KT.CIRCLE;
      return [kt, flat.slice(0, 10), flat.slice(0, 10)];
    }
    return [KT.UNKNOWN, [], []];
  }
  return [KT.UNKNOWN, [], []];
}

function reverseRel(kt: number, rel: number[]): number[] {
  if (kt === KT.QUADRATIC && rel.length === 2) return [1 - rel[0], -rel[1]];
  if (kt === KT.CUBIC && rel.length === 4) return [1 - rel[2], -rel[3], 1 - rel[0], -rel[1]];
  return [...rel];
}

export function parseSpec(spec: any, name = "pattern"): Pattern {
  const pat = spec.pattern ?? spec;
  const props = spec.properties ?? {};
  const unit = Number(pat.units_in_meter ?? props.units_in_meter ?? 100);
  const toCm = 100 / unit;
  const relCoords = String(props.curvature_coords ?? "relative").toLowerCase() !== "absolute";

  const keys = Object.keys(pat.panels);
  const order: string[] = pat.panel_order;
  const names = order && order.length === keys.length && order.every((k) => k in pat.panels) ? order : keys;

  const panels: Panel[] = names.map((pname, orderIdx) => {
    const pd = pat.panels[pname];
    const verts: Pt[] = pd.vertices.map((v: number[]) => [v[0] * toCm, v[1] * toCm] as Pt);
    const rawEdges = pd.edges;
    const n = rawEdges.length;

    // signed area of the chord polygon, in edge-traversal order
    let area = 0;
    for (let i = 0; i < n; i++) {
      const a = verts[rawEdges[i].endpoints[0]];
      const b = verts[rawEdges[(i + 1) % n].endpoints[0]];
      area += a[0] * b[1] - b[0] * a[1];
    }
    const rev = area < 0;
    const seq = rev ? [...Array(n).keys()].reverse() : [...Array(n).keys()];

    const edges: Edge[] = seq.map((j) => {
      const e = rawEdges[j];
      let [a, b] = [e.endpoints[0], e.endpoints[1]];
      if (rev) [a, b] = [b, a];
      const start = verts[a], end = verts[b];
      // curvature is read in the ORIGINAL orientation, then transformed if reversed
      let [kt, kabs, krel] = parseCurvature(e.curvature ?? null,
        verts[e.endpoints[0]], verts[e.endpoints[1]], relCoords);
      if (rev) {
        if (relCoords) {
          krel = reverseRel(kt, krel);
          if (kt === KT.QUADRATIC && krel.length === 2) {
            const aa = relToAbs(start, end, [krel[0], krel[1]]); kabs = [aa[0], aa[1]];
          } else if (kt === KT.CUBIC && krel.length === 4) {
            const a1 = relToAbs(start, end, [krel[0], krel[1]]);
            const a2 = relToAbs(start, end, [krel[2], krel[3]]);
            kabs = [a1[0], a1[1], a2[0], a2[1]];
          }
        } else if (kt === KT.CUBIC && kabs.length === 4) {
          kabs = [kabs[2], kabs[3], kabs[0], kabs[1]];
        }
        if (kt === KT.CIRCLE && kabs.length > 2) {
          // circle params are [radius, large_arc, right]; traversing the arc backwards
          // flips its sweep, while radius and the large-arc flag are orientation-free.
          // Matches the same fix in gcd_parser.py -- keep the two in step.
          kabs = kabs.slice();
          kabs[2] = 1 - Number(kabs[2]);
        }
      }
      return { panel: pname, idxInPanel: j, start, end, kt, kparams: kabs };
    });

    // panel bbox lower-left -> origin (chord bbox over the traversed vertices)
    const xs = edges.flatMap((e) => [e.start[0], e.end[0]]);
    const ys = edges.flatMap((e) => [e.start[1], e.end[1]]);
    const mx = Math.min(...xs), my = Math.min(...ys);
    for (const e of edges) {
      e.start = [e.start[0] - mx, e.start[1] - my];
      e.end = [e.end[0] - mx, e.end[1] - my];
      if (e.kt === KT.QUADRATIC || e.kt === KT.CUBIC)
        e.kparams = e.kparams.map((t, i) => t - (i % 2 === 0 ? mx : my));
    }
    return { name: pname, orderIdx, edges, nEdgesRaw: n, reversed: rev,
             rawVerts: verts, rawEdges,
             translation: pd.translation ?? [0, 0, 0], rotation: pd.rotation ?? [0, 0, 0] };
  });

  const stitches: Array<Array<[string, number]>> = [];
  for (const st of pat.stitches ?? []) {
    const sides = st.filter((s: any) => s && typeof s === "object" && "panel" in s && "edge" in s)
                    .map((s: any) => [s.panel, Number(s.edge)] as [string, number]);
    if (sides.length >= 2) stitches.push(sides);
  }
  return { name, panels, stitches };
}
