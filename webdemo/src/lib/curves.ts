/* Port of autosew/autosew/curves.py -- edge polyline reconstruction and the sagitta
   shape descriptor.  Keep the two in step: the model was trained on exactly these
   numbers, so any divergence silently degrades the prediction rather than raising.

   The tagged curvature encoding (k_t plus ten slots whose meaning k_t selects) is fine
   when k_t comes from a specification file and is exact.  It is not usable when the
   input is a DXF polyline, where k_t has to be estimated: a circular arc and a weakly
   curved quadratic Bezier fit sampled points about equally well, so k_t flips and
   eleven feature dimensions change meaning at once.

   The sagitta profile removes the type decision.  Both a specification curve and a DXF
   polyline reduce to the same thing -- signed perpendicular deviation from the chord, at
   K uniform arclength positions, divided by the chord length. */
import { KT, type Edge, type Pt } from "./parseSpec";

export const SAMPLES_PER_EDGE = 32;

/** SVG-style circular arc.  GCD circle params are [radius, large_arc, right]. */
export function arcPolyline(p0: Pt, p1: Pt, r: number, large: number, sweep: number,
                            n = SAMPLES_PER_EDGE): Pt[] {
  const cx = p1[0] - p0[0], cy = p1[1] - p0[1];
  const d = Math.hypot(cx, cy);
  if (d < 1e-9) return [p0, p1];
  const rr = Math.max(r, d / 2 + 1e-9);
  const h = Math.sqrt(Math.max(rr * rr - (d * d) / 4, 0));
  const px = -cy / d, py = cx / d;
  const s = (!!large) !== (!!sweep) ? 1 : -1;
  const c: Pt = [(p0[0] + p1[0]) / 2 + s * h * px, (p0[1] + p1[1]) / 2 + s * h * py];
  let a0 = Math.atan2(p0[1] - c[1], p0[0] - c[0]);
  let a1 = Math.atan2(p1[1] - c[1], p1[0] - c[0]);
  if (sweep) { while (a1 < a0) a1 += 2 * Math.PI; } else { while (a1 > a0) a1 -= 2 * Math.PI; }
  return Array.from({ length: n }, (_, k) => {
    const t = a0 + ((a1 - a0) * k) / (n - 1);
    return [c[0] + rr * Math.cos(t), c[1] + rr * Math.sin(t)] as Pt;
  });
}

/** Parsed Edge -> polyline in the same panel-local frame as edge.start / edge.end.
    Bezier control points in kparams are already absolute and already translated with
    the panel bbox, so they need no further conversion.  B-splines fall back to the
    chord (their parameter layout is not pinned down and they do not occur in the
    GCD.v2 specs seen so far). */
export function edgePolyline(e: Edge, n = SAMPLES_PER_EDGE): Pt[] {
  if (e.poly) return e.poly;      // DXF input: the sampled boundary IS the curve
  const p0 = e.start, p1 = e.end, kp = e.kparams;
  if (e.kt === KT.QUADRATIC && kp.length >= 2) {
    const q: Pt = [kp[0], kp[1]];
    return Array.from({ length: n }, (_, k) => {
      const t = k / (n - 1), u = 1 - t;
      return [u * u * p0[0] + 2 * u * t * q[0] + t * t * p1[0],
              u * u * p0[1] + 2 * u * t * q[1] + t * t * p1[1]] as Pt;
    });
  }
  if (e.kt === KT.CUBIC && kp.length >= 4) {
    const q1: Pt = [kp[0], kp[1]], q2: Pt = [kp[2], kp[3]];
    return Array.from({ length: n }, (_, k) => {
      const t = k / (n - 1), u = 1 - t;
      return [u * u * u * p0[0] + 3 * u * u * t * q1[0] + 3 * u * t * t * q2[0] + t * t * t * p1[0],
              u * u * u * p0[1] + 3 * u * u * t * q1[1] + 3 * u * t * t * q2[1] + t * t * t * p1[1]] as Pt;
    });
  }
  if (e.kt === KT.CIRCLE && kp.length >= 1) {
    return arcPolyline(p0, p1, kp[0], kp.length > 1 ? kp[1] : 0, kp.length > 2 ? kp[2] : 0, n);
  }
  return [p0, p1];
}

/** Polyline -> K signed perpendicular deviations from the chord, over the chord length,
    sampled at uniform arclength fractions k/(K+1).  Endpoints are excluded: their
    deviation is identically zero and carries nothing. */
export function sagittaProfile(pts: Pt[], K: number, out?: Float32Array, off = 0): Float32Array {
  const o = out ?? new Float32Array(K);
  if (!out) o.fill(0);
  else for (let k = 0; k < K; k++) o[off + k] = 0;
  if (pts.length < 3) return o;

  const p0 = pts[0], p1 = pts[pts.length - 1];
  const cx = p1[0] - p0[0], cy = p1[1] - p0[1];
  const L = Math.hypot(cx, cy);
  if (L < 1e-12) return o;
  const nx = -cy / L, ny = cx / L;                 // left normal, ACW frame

  const s = new Float64Array(pts.length);
  for (let i = 1; i < pts.length; i++)
    s[i] = s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  const total = s[pts.length - 1];
  if (total < 1e-12) return o;

  let j = 1;
  for (let k = 0; k < K; k++) {
    const t = ((k + 1) / (K + 1)) * total;
    while (j < pts.length - 1 && s[j] < t) j++;
    const span = s[j] - s[j - 1];
    const w = span > 1e-15 ? (t - s[j - 1]) / span : 0;
    const qx = pts[j - 1][0] + w * (pts[j][0] - pts[j - 1][0]);
    const qy = pts[j - 1][1] + w * (pts[j][1] - pts[j - 1][1]);
    o[off + k] = ((qx - p0[0]) * nx + (qy - p0[1]) * ny) / L;
  }
  return o;
}
