/* The 2D pattern with its stitching drawn on it.

   Correspondence is shown by colour: the two edges of a predicted pair share a colour,
   and a thin line joins their midpoints so a pair is still readable when two colours sit
   far apart.  Clicking an edge focuses its correspondence -- that one is drawn thick and
   every other pair drops back to grey -- which is the only way to read a crowded pattern.
   Clicking empty space, or the same edge again, clears the focus.

   When the input carries ground truth (a specification.json does, a DXF does not) the
   overlay switches to the scored view: correct / false positive / false negative. */
import { useMemo, useState } from "react";
import { edgePolyline, edgeMid, stitchColor, type Placed } from "./lib/render";
import type { Pattern } from "./lib/parseSpec";

export type SvgProps = {
  pattern: Pattern;
  placed: Placed[];
  keys: Array<[string, number]>;
  pred: Set<string>;
  gt: Set<string>;
  show?: { correct: boolean; fp: boolean; fn: boolean; gt: boolean };
  height?: string;
};

export default function PatternSvg({ pattern, placed, keys, pred, gt, show, height }: SvgProps) {
  const [focus, setFocus] = useState<number | null>(null);   // a node index
  const scored = gt.size > 0;
  const byName = useMemo(() => new Map(placed.map((p) => [p.name, p])), [placed]);

  /* node -> correspondence group, so both sides of a pair share a colour */
  const group = useMemo(() => {
    const m = new Map<number, number>();
    if (scored || pred.size === 0) return m;
    const parent = new Map<number, number>();
    const find = (a: number): number => {
      while (parent.get(a) !== a) { parent.set(a, parent.get(parent.get(a)!)!); a = parent.get(a)!; }
      return a;
    };
    for (const k of pred) for (const t of k.split("-").map(Number)) if (!parent.has(t)) parent.set(t, t);
    for (const k of pred) {
      const [a, b] = k.split("-").map(Number);
      const ra = find(a), rb = find(b);
      if (ra !== rb) parent.set(ra, rb);
    }
    const gi = new Map<number, number>();
    for (const node of [...parent.keys()].sort((a, b) => a - b)) {
      const r = find(node);
      if (!gi.has(r)) gi.set(r, gi.size);
      m.set(node, gi.get(r)!);
    }
    return m;
  }, [pred, scored]);

  const node = useMemo(() => {
    const m = new Map<string, number>();
    keys.forEach(([pn, ei], i) => m.set(`${pn}#${ei}`, i));
    return m;
  }, [keys]);

  const pts = placed.flatMap((p) => p.verts);
  const xs = pts.map((v) => v[0]), ys = pts.map((v) => v[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pad = 12;
  const W = x1 - x0 + 2 * pad, H = y1 - y0 + 2 * pad;
  const tx = (p: [number, number]) => `${p[0] - x0 + pad},${y1 - p[1] + pad}`;
  const S = Math.max(W, H) / 220;                 // keep strokes even across pattern sizes

  const focusGroup = focus === null ? null : group.get(focus) ?? null;

  const stitchOf = new Map<string, number>();
  pattern.stitches.forEach((sides, si) =>
    sides.forEach(([pn, ei]) => stitchOf.set(`${pn}#${ei}`, si)));

  const line = (a: number, b: number, cls: string, key: string, w?: number) => {
    const ka = keys[a], kb = keys[b];
    const pa = byName.get(ka?.[0]), pb = byName.get(kb?.[0]);
    if (!pa || !pb || !pa.edges[ka[1]] || !pb.edges[kb[1]]) return null;
    const m1 = edgeMid(pa, ka[1]), m2 = edgeMid(pb, kb[1]);
    return <line key={key} className={cls} x1={m1[0] - x0 + pad} y1={y1 - m1[1] + pad}
                 x2={m2[0] - x0 + pad} y2={y1 - m2[1] + pad}
                 strokeWidth={w === undefined ? undefined : w * S} />;
  };

  return (
    <svg className="canvas" style={height ? { height } : undefined}
         viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
         onClick={() => setFocus(null)}>
      {placed.map((p) => (
        <g key={p.name}>
          {p.edges.map((_: unknown, i: number) => {
            const key = `${p.name}#${i}`;
            const ni = node.get(key);
            const si = stitchOf.get(key) ?? (ni === undefined ? undefined : group.get(ni));
            const inFocus = focusGroup !== null && ni !== undefined && group.get(ni) === focusGroup;
            const dim = focusGroup !== null && !inFocus;
            const col = si === undefined || dim ? "#c9c9c9" : stitchColor(si);
            const w = si === undefined ? 0.5 : inFocus ? 2.6 : scored ? 1.1 : 0.9;
            const d = edgePolyline(p, i).map(tx).join(" L ");
            return (
              <path key={i} d={`M ${d}`} stroke={col} strokeWidth={w * S} fill="none"
                    strokeLinecap="round"
                    style={{ cursor: si === undefined ? "default" : "pointer" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (ni === undefined || si === undefined) return;
                      setFocus((f) => (f !== null && group.get(f) === group.get(ni) ? null : ni));
                    }} />
            );
          })}
          <text x={p.verts.reduce((s, v) => s + v[0], 0) / p.verts.length - x0 + pad}
                y={y1 - p.verts.reduce((s, v) => s + v[1], 0) / p.verts.length + pad}
                className="plabel" style={{ fontSize: 4 * S }}>{p.name}</text>
        </g>
      ))}

      {!scored && [...pred].map((k) => {
        const [a, b] = k.split("-").map(Number);
        const inFocus = focusGroup !== null && group.get(a) === focusGroup;
        if (focusGroup !== null && !inFocus) return null;
        return line(a, b, "predline", `x${k}`, inFocus ? 1.4 : 0.45);
      })}

      {scored && show && (
        <>
          {show.gt && [...gt].map((k) => {
            const [a, b] = k.split("-").map(Number); return line(a, b, "gtline", `g${k}`);
          })}
          {show.correct && [...pred].filter((k) => gt.has(k)).map((k) => {
            const [a, b] = k.split("-").map(Number); return line(a, b, "ok", `c${k}`);
          })}
          {show.fp && [...pred].filter((k) => !gt.has(k)).map((k) => {
            const [a, b] = k.split("-").map(Number); return line(a, b, "fp", `p${k}`);
          })}
          {show.fn && [...gt].filter((k) => !pred.has(k)).map((k) => {
            const [a, b] = k.split("-").map(Number); return line(a, b, "fn", `n${k}`);
          })}
        </>
      )}
    </svg>
  );
}
