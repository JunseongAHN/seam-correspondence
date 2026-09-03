/* The 2D pattern with its stitching drawn on it.

   Three things share this view.

   Prediction, coloured: the two edges of a predicted pair share a colour, and a thin line
   joins their midpoints so a pair stays readable when the two colours sit far apart.
   Clicking an edge focuses its correspondence -- that one is drawn thick and every other
   pair drops to grey -- which is the only way to read a crowded pattern.

   Scoring, when the input carries ground truth (a specification.json does, a DXF does
   not): correct / false positive / false negative.

   Annotation, when `annotate` is on: click one edge then another to sew them together by
   hand.  A DXF has no stitch list, so this is how ground truth gets made for a real
   garment -- and once it exists the prediction can actually be scored against it. */
import { useEffect, useMemo, useRef, useState } from "react";
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
  /** hand-annotation mode */
  annotate?: boolean;
  /** show "index · arc length" on every edge */
  labels?: boolean;
  userPairs?: Set<string>;          // "a-b" with a < b, node indices
  selected?: number | null;
  onEdgeClick?: (node: number) => void;
};

export default function PatternSvg({
  pattern, placed, keys, pred, gt, show, height,
  annotate = false, labels = true, userPairs, selected = null, onEdgeClick,
}: SvgProps) {
  const [focus, setFocus] = useState<number | null>(null);
  const scored = gt.size > 0 && !annotate;

  /* Focus is a node INDEX, and the same index means a different edge in the next
     garment.  Loading one while a focus is held would otherwise open it with most of
     the pattern greyed out and the first click merely clearing that -- which reads
     exactly like clicking an edge does nothing. */
  useEffect(() => { setFocus(null); }, [keys]);

  /** node -> correspondence group, so both sides of a pair share a colour */
  const group = useMemo(() => {
    const src = annotate ? (userPairs ?? new Set<string>()) : pred;
    const m = new Map<number, number>();
    if ((!annotate && scored) || src.size === 0) return m;
    const parent = new Map<number, number>();
    const find = (a: number): number => {
      while (parent.get(a) !== a) { parent.set(a, parent.get(parent.get(a)!)!); a = parent.get(a)!; }
      return a;
    };
    for (const k of src) for (const t of k.split("-").map(Number)) if (!parent.has(t)) parent.set(t, t);
    for (const k of src) {
      const [a, b] = k.split("-").map(Number);
      const ra = find(a), rb = find(b);
      if (ra !== rb) parent.set(ra, rb);
    }
    /* Number the groups in the order the pairs were MADE, not by node index: otherwise
       adding one pair renumbers the rest and every colour on screen shifts, which makes
       hand-annotation impossible to follow. */
    const gi = new Map<number, number>();
    for (const k of src) {
      for (const t of k.split("-").map(Number)) {
        const r = find(t);
        if (!gi.has(r)) gi.set(r, gi.size);
      }
    }
    for (const t of parent.keys()) m.set(t, gi.get(find(t))!);
    return m;
  }, [pred, userPairs, annotate, scored]);

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

  /* Zoom and pan.  Some pieces are 1.6 cm strips whose two long edges sit closer together
     than a comfortable click target, so without zoom they cannot be picked apart. */
  const [view, setView] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const v = view ?? { x: 0, y: 0, w: W, h: H };
  const svgRef = useRef<SVGSVGElement>(null);
  /* Panning stores the pointer position in CLIENT pixels and the scale as it was when the
     drag began.  Converting through the live viewBox instead would feed each update back
     into the next frame's conversion, which reads as stutter. */
  const drag = useRef<{ cx: number; cy: number; vx: number; vy: number;
                        s: number; moved: boolean } | null>(null);
  const S = Math.max(v.w, v.h) / 220;             // strokes stay even as you zoom in

  /** user units per client pixel, under preserveAspectRatio="xMidYMid meet" */
  const pxScale = () => {
    const r = svgRef.current!.getBoundingClientRect();
    return Math.max(v.w / r.width, v.h / r.height);
  };

  const toUser = (e: { clientX: number; clientY: number }) => {
    const r = svgRef.current!.getBoundingClientRect();
    const s = pxScale();
    return { x: v.x + (e.clientX - r.left - (r.width - v.w / s) / 2) * s,
             y: v.y + (e.clientY - r.top - (r.height - v.h / s) / 2) * s };
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const p = toUser(e);
    const k = Math.exp(e.deltaY * 0.0015);
    const w = Math.min(Math.max(v.w * k, W / 60), W * 1.4);
    const h = w * (v.h / v.w);
    setView({ x: p.x - ((p.x - v.x) * w) / v.w, y: p.y - ((p.y - v.y) * h) / v.h, w, h });
  };

  /* Path strings, midpoints and arc lengths depend on the pattern, not on the viewBox, so
     they are built once.  Recomputing 48 polylines on every pan frame is what made
     dragging stutter even after the coordinate feedback was fixed. */
  const geom = useMemo(() => placed.map((p) => {
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
  }), [placed, x0, y1]);

  const mid = useMemo(() => {
    const m = new Map<number, [number, number]>();
    geom.forEach((g, pi) => g.edges.forEach((e, i) => {
      const n = node.get(`${placed[pi].name}#${i}`);
      if (n !== undefined) m.set(n, [e.mx, e.my]);
    }));
    return m;
  }, [geom, node, placed]);

  /* Which nodes a click keeps lit.  In the scored view `group` is deliberately empty --
     colour comes from the ground-truth stitch index there, not from the prediction -- so
     focus has to be read off the pairings this view actually draws, or clicking an edge in
     a scored pattern dims nothing at all. */
  const partners = useMemo(() => {
    const m = new Map<number, Set<number>>();
    const src = annotate ? [...(userPairs ?? [])] : scored ? [...pred, ...gt] : [...pred];
    for (const k of src) {
      const [a, b] = k.split("-").map(Number);
      if (!m.has(a)) m.set(a, new Set());
      if (!m.has(b)) m.set(b, new Set());
      m.get(a)!.add(b); m.get(b)!.add(a);
    }
    return m;
  }, [pred, gt, userPairs, annotate, scored]);

  const focusSet = useMemo(() => {
    if (focus === null) return null;
    const g = group.get(focus);
    if (g !== undefined) {
      const s = new Set<number>();
      for (const [n, gg] of group) if (gg === g) s.add(n);
      return s;
    }
    return new Set<number>([focus, ...(partners.get(focus) ?? [])]);
  }, [focus, group, partners]);

  /** a stitch line survives the focus if either of its ends is lit */
  const lit = (a: number, b: number) =>
    focusSet === null || focusSet.has(a) || focusSet.has(b);

  /* Which ground-truth stitch each edge belongs to, so both sides share a colour.  A
     specification carries that list; a DXF does not, and its ground truth arrives as node
     pairs instead -- without this fallback a scored DXF would render entirely grey. */
  const stitchOf = new Map<string, number>();
  if (!annotate) {
    if (pattern.stitches.length)
      pattern.stitches.forEach((sides, si) =>
        sides.forEach(([pn, ei]) => stitchOf.set(`${pn}#${ei}`, si)));
    else
      [...gt].forEach((k, si) => k.split("-").map(Number).forEach((n) => {
        const key = keys[n];
        if (key) stitchOf.set(`${key[0]}#${key[1]}`, si);
      }));
  }

  const line = (a: number, b: number, cls: string, key: string, w?: number) => {
    const m1 = mid.get(a), m2 = mid.get(b);
    if (!m1 || !m2) return null;
    return <line key={key} className={cls} x1={m1[0]} y1={m1[1]} x2={m2[0]} y2={m2[1]}
                 strokeWidth={w === undefined ? undefined : w * S} />;
  };

  return (
    <svg ref={svgRef} className={`canvas${annotate ? " annotating" : ""}`}
         style={height ? { height } : undefined}
         viewBox={`${v.x} ${v.y} ${v.w} ${v.h}`} preserveAspectRatio="xMidYMid meet"
         onWheel={onWheel}
         onPointerDown={(e) => {
           if (e.button !== 0) return;
           (e.target as Element).setPointerCapture?.(e.pointerId);
           drag.current = { cx: e.clientX, cy: e.clientY, vx: v.x, vy: v.y,
                            s: pxScale(), moved: false };
         }}
         onPointerMove={(e) => {
           const d = drag.current;
           if (!d) return;
           const dx = (e.clientX - d.cx) * d.s, dy = (e.clientY - d.cy) * d.s;
           if (!d.moved && Math.hypot(e.clientX - d.cx, e.clientY - d.cy) < 4) return;
           d.moved = true;
           setView({ x: d.vx - dx, y: d.vy - dy, w: v.w, h: v.h });
         }}
         onPointerUp={(e) => {
           (e.target as Element).releasePointerCapture?.(e.pointerId);
           setTimeout(() => { drag.current = null; }, 0);
         }}
         onPointerCancel={() => { drag.current = null; }}
         onDoubleClick={() => setView(null)}
         onClick={() => { if (!annotate && !drag.current?.moved) setFocus(null); }}>
      {geom.map((g) => (
        <g key={g.name}>
          {g.edges.map((e, i) => {
            const key = `${g.name}#${i}`;
            const ni = node.get(key);
            const si = stitchOf.get(key) ?? (ni === undefined ? undefined : group.get(ni));
            const isSel = annotate && ni !== undefined && ni === selected;
            const inFocus = focusSet !== null && ni !== undefined && focusSet.has(ni);
            const dim = focusSet !== null && !inFocus;
            const col = isSel ? "#f97316"
                      : si === undefined || dim ? "#c9c9c9"
                      : stitchColor(si);
            const w = isSel ? 3.2
                    : si === undefined ? (annotate ? 0.9 : 0.5)
                    : inFocus ? 2.6 : scored ? 1.1 : 1.4;
            return (
              <g key={i}>
                <path d={e.d} stroke={col} strokeWidth={w * S} fill="none"
                      strokeLinecap="round" />
                {labels && (
                  <text x={e.mx} y={e.my + 1.6 * S} className="elabel"
                        style={{ fontSize: 4.6 * S }}>
                    {i} · {e.arc.toFixed(1)}
                  </text>
                )}
                {/* a fat invisible copy, so a thin edge is still easy to hit */}
                <path d={e.d} stroke="transparent" strokeWidth={2.2 * S} fill="none"
                      strokeLinecap="round"
                      style={{ cursor: annotate || si !== undefined
                                       || (ni !== undefined && partners.get(ni)?.size)
                                       ? "pointer" : "default" }}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        if (ni === undefined || drag.current?.moved) return;   // that was a pan
                        if (annotate) { onEdgeClick?.(ni); return; }
                        if (si === undefined && !partners.get(ni)?.size) return;
                        setFocus((f) => (f !== null && focusSet?.has(ni) ? null : ni));
                      }} />
              </g>
            );
          })}
          <text x={g.cx} y={g.cy} className="plabel" style={{ fontSize: 4 * S }}>{g.name}</text>
          {g.blowUp && (
            <text x={g.cx} y={g.cy + 4.6 * S} className="plabel scalenote"
                  style={{ fontSize: 3.1 * S }}>
              width ×{g.blowUp.toFixed(1)}, not to scale
            </text>
          )}
        </g>
      ))}

      {annotate && [...(userPairs ?? [])].map((k) => {
        const [a, b] = k.split("-").map(Number);
        return line(a, b, "userline", `u${k}`, 1.2);
      })}

      {!annotate && !scored && [...pred].map((k) => {
        const [a, b] = k.split("-").map(Number);
        if (!lit(a, b)) return null;
        return line(a, b, "predline", `x${k}`, focusSet !== null ? 1.4 : 0.45);
      })}

      {scored && show && (() => {
        const w = focusSet === null ? undefined : 1.4;
        const draw = (k: string, cls: string, pfx: string) => {
          const [a, b] = k.split("-").map(Number);
          return lit(a, b) ? line(a, b, cls, `${pfx}${k}`, w) : null;
        };
        return (
          <>
            {show.gt && [...gt].map((k) => draw(k, "gtline", "g"))}
            {show.correct && [...pred].filter((k) => gt.has(k)).map((k) => draw(k, "ok", "c"))}
            {show.fp && [...pred].filter((k) => !gt.has(k)).map((k) => draw(k, "fp", "p"))}
            {show.fn && [...gt].filter((k) => !pred.has(k)).map((k) => draw(k, "fn", "n"))}
          </>
        );
      })()}
    </svg>
  );
}
