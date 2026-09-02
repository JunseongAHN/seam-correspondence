/* 2D pattern layout + drawing.  Mirrors autosew/viz/visualize.py: panels are placed
   by their spec transform, back panels (tz<0) shifted right, stitched edge pairs
   share a colour, and the prediction is overlaid green / red / black-dashed. */
import { type Panel, type Pattern, type Pt } from "./parseSpec";

export interface Placed { name: string; verts: Pt[]; edges: any[]; back: boolean }

function sampleEdge(p0: Pt, p1: Pt, curv: any, n = 32): Pt[] {
  const rel2abs = (c: number[]): Pt => {
    const e: Pt = [p1[0] - p0[0], p1[1] - p0[1]];
    return [p0[0] + c[0] * e[0] + c[1] * -e[1], p0[1] + c[0] * e[1] + c[1] * e[0]];
  };
  if (curv == null) return [p0, p1];
  const bez = (pts: Pt[]): Pt[] => {
    const out: Pt[] = [];
    for (let k = 0; k < n; k++) {
      const t = k / (n - 1);
      if (pts.length === 3) {
        const [a, q, b] = pts;
        out.push([(1 - t) ** 2 * a[0] + 2 * (1 - t) * t * q[0] + t * t * b[0],
                  (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * q[1] + t * t * b[1]]);
      } else {
        const [a, q1, q2, b] = pts;
        out.push([(1 - t) ** 3 * a[0] + 3 * (1 - t) ** 2 * t * q1[0] + 3 * (1 - t) * t * t * q2[0] + t ** 3 * b[0],
                  (1 - t) ** 3 * a[1] + 3 * (1 - t) ** 2 * t * q1[1] + 3 * (1 - t) * t * t * q2[1] + t ** 3 * b[1]]);
      }
    }
    return out;
  };
  if (Array.isArray(curv)) return bez([p0, rel2abs(curv), p1]);
  const type = String(curv.type ?? "").toLowerCase();
  const params = curv.params ?? [];
  if (type.includes("quadratic")) return bez([p0, rel2abs(params[0]), p1]);
  if (type.includes("cubic")) return bez([p0, rel2abs(params[0]), rel2abs(params[1]), p1]);
  if (type.includes("circ") || type.includes("arc")) {
    const chord: Pt = [p1[0] - p0[0], p1[1] - p0[1]];
    const d = Math.hypot(chord[0], chord[1]);
    if (d < 1e-9) return [p0, p1];
    const r = Math.max(Number(params[0]), d / 2 + 1e-9);
    const large = !!params[1], sweep = !!params[2];
    const mid: Pt = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
    const h = Math.sqrt(Math.max(r * r - (d * d) / 4, 0));
    const pv: Pt = [-chord[1] / d, chord[0] / d];
    const sgn = large !== sweep ? 1 : -1;
    const c: Pt = [mid[0] + sgn * h * pv[0], mid[1] + sgn * h * pv[1]];
    let a0 = Math.atan2(p0[1] - c[1], p0[0] - c[0]);
    let a1 = Math.atan2(p1[1] - c[1], p1[0] - c[0]);
    if (sweep) { while (a1 < a0) a1 += 2 * Math.PI; } else { while (a1 > a0) a1 -= 2 * Math.PI; }
    return Array.from({ length: n }, (_, k) => {
      const t = a0 + ((a1 - a0) * k) / (n - 1);
      return [c[0] + r * Math.cos(t), c[1] + r * Math.sin(t)] as Pt;
    });
  }
  return [p0, p1];
}

export function placePanels(p: Pattern, gap = 25): Placed[] {
  const placed: Placed[] = p.panels.map((pan: Panel) => {
    const rz = (pan.rotation[2] * Math.PI) / 180;
    const [c, s] = [Math.cos(rz), Math.sin(rz)];
    const verts = pan.rawVerts.map(([x, y]) =>
      [c * x - s * y + pan.translation[0], s * x + c * y + pan.translation[1]] as Pt);
    return { name: pan.name, verts, edges: pan.rawEdges, back: pan.translation[2] < 0 };
  });
  const front = placed.filter((q) => !q.back), back = placed.filter((q) => q.back);
  if (front.length && back.length) {
    const fmax = Math.max(...front.flatMap((q) => q.verts.map((v) => v[0])));
    const bmin = Math.min(...back.flatMap((q) => q.verts.map((v) => v[0])));
    const dx = fmax + gap - bmin;
    for (const q of back) q.verts = q.verts.map(([x, y]) => [x + dx, y] as Pt);
  }
  return placed;
}

/** Same sampling, but in the panel's own 2D frame (no layout transform) --
    this is what the 3D placement needs, since it applies its own rotation. */
export function edgePolylineLocal(panel: Panel, i: number): Pt[] {
  const e = panel.rawEdges[i];
  return sampleEdge(panel.rawVerts[e.endpoints[0]], panel.rawVerts[e.endpoints[1]],
                    e.curvature ?? null);
}

export function edgePolyline(q: Placed, i: number): Pt[] {
  const e = q.edges[i];
  if (e.poly) return e.poly;    // DXF input: the sampled boundary is what we have
  return sampleEdge(q.verts[e.endpoints[0]], q.verts[e.endpoints[1]], e.curvature ?? null);
}

/** True midpoint by arclength: a straight edge samples to only two points, so
    taking the middle element would land on the end vertex. */
export function edgeMid(q: Placed, i: number): Pt {
  const pts = edgePolyline(q, i);
  const cum = [0];
  for (let k = 1; k < pts.length; k++)
    cum.push(cum[k - 1] + Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]));
  const half = cum[cum.length - 1] / 2;
  if (half <= 0) return pts[0];
  let k = 1; while (k < cum.length - 1 && cum[k] < half) k++;
  const t = (half - cum[k - 1]) / Math.max(cum[k] - cum[k - 1], 1e-12);
  return [pts[k - 1][0] + t * (pts[k][0] - pts[k - 1][0]),
          pts[k - 1][1] + t * (pts[k][1] - pts[k - 1][1])];
}

export function stitchColor(i: number): string {
  const phi = 0.61803398875;
  return `hsl(${((0.05 + i * phi) % 1) * 360}, 85%, 45%)`;
}
