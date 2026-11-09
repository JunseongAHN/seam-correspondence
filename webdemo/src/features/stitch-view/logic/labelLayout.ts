/* Where each edge label goes, so labels neither pile on each other nor sit on their edge.

   A label is tried beside its edge, along the normal: first on the side away from the
   panel's centre, then on the side towards it, fontSize × 0.8 from the midpoint.  It takes
   the first spot whose box is clear of every label already placed; if neither is, it is
   hidden -- zooming in shrinks the labels against the pattern, so it comes back.  Labels of
   the focused correspondence are placed first and always shown, even over another.

   Boxes are estimated, not measured -- 0.6 × fontSize per character, fontSize tall,
   centred on the point, matching text-anchor="middle" dominant-baseline="middle" -- so
   nothing waits on layout and the result is the same for the same input. */

export type LabelEdge = {
  id: string;
  mid: [number, number];
  /** the edge's two ends, for its direction */
  a: [number, number];
  b: [number, number];
  /** the centre of the edge's panel: the label goes to the side away from it first */
  center: [number, number];
  text: string;
  /** ordering after focus: longer edges are placed first */
  length: number;
  focused: boolean;
};

/** A label drawn regardless, that edge labels must keep clear of (the panel names). */
export type LabelObstacle = {
  x: number; y: number; text: string; fontSize: number;
  /** where y sits in the text: its middle, or the alphabetic baseline under it */
  baseline: "middle" | "alphabetic";
};

export type PlacedLabel = { id: string; x: number; y: number; visible: boolean };

type Box = { x0: number; y0: number; x1: number; y1: number };

const box = (x: number, y: number, text: string, fontSize: number,
             baseline: LabelObstacle["baseline"] = "middle"): Box => {
  const w = text.length * fontSize * 0.6, h = fontSize;
  const top = baseline === "middle" ? y - h / 2 : y - h;
  return { x0: x - w / 2, y0: top, x1: x + w / 2, y1: top + h };
};

const overlaps = (p: Box, q: Box) => p.x0 < q.x1 && q.x0 < p.x1 && p.y0 < q.y1 && q.y0 < p.y1;

/** The unit normal of a→b that points away from `center`. */
function outward(a: [number, number], b: [number, number], mid: [number, number],
                 center: [number, number]): [number, number] {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L = Math.hypot(dx, dy) || 1;
  const n: [number, number] = [-dy / L, dx / L];
  const side = (mid[0] - center[0]) * n[0] + (mid[1] - center[1]) * n[1];
  return side < 0 ? [-n[0], -n[1]] : n;
}

export function layoutLabels(edges: readonly LabelEdge[],
                             opts: { fontSize: number; obstacles?: readonly LabelObstacle[] }): PlacedLabel[] {
  const { fontSize } = opts;
  const off = fontSize * 0.8;
  const taken: Box[] = (opts.obstacles ?? []).map((o) => box(o.x, o.y, o.text, o.fontSize, o.baseline));

  // focused first, then longest, then input order -- a total order, so the layout is stable
  const order = edges.map((e, i) => ({ e, i })).sort((p, q) =>
    Number(q.e.focused) - Number(p.e.focused) || q.e.length - p.e.length || p.i - q.i);

  const placed = new Map<string, PlacedLabel>();
  for (const { e } of order) {
    const n = outward(e.a, e.b, e.mid, e.center);
    const spots: [number, number][] = [
      [e.mid[0] + n[0] * off, e.mid[1] + n[1] * off],
      [e.mid[0] - n[0] * off, e.mid[1] - n[1] * off],
    ];
    const free = spots.find(([x, y]) => !taken.some((t) => overlaps(t, box(x, y, e.text, fontSize))));
    const at = free ?? (e.focused ? spots[0] : null);
    if (at) taken.push(box(at[0], at[1], e.text, fontSize));
    placed.set(e.id, { id: e.id, x: at?.[0] ?? e.mid[0], y: at?.[1] ?? e.mid[1], visible: at !== null });
  }
  return edges.map((e) => placed.get(e.id)!);   // every id was placed in the loop above
}
