/* Which edges belong together, and so which share a colour and which a click keeps lit.
   A pair is the string "a-b" of two node indices, a < b. */
import type { Pattern } from "@/core-algorithm/pattern";
import { stitchColor } from "./render";

/** "panel#edge" -> node index */
export function nodeIndex(keys: Array<[string, number]>) {
  const m = new Map<string, number>();
  keys.forEach(([pn, ei], i) => m.set(`${pn}#${ei}`, i));
  return m;
}

/** node -> correspondence group, so both sides of a pair share a colour */
export function pairGroups(src: Iterable<string>) {
  const parent = new Map<number, number>();
  const up = (a: number) => parent.get(a) ?? a;
  const find = (a: number): number => {
    while (up(a) !== a) { parent.set(a, up(up(a))); a = up(a); }
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
  const m = new Map<number, number>();
  for (const k of src) {
    for (const t of k.split("-").map(Number)) {
      const r = find(t);
      let g = gi.get(r);
      if (g === undefined) { g = gi.size; gi.set(r, g); }
      m.set(t, g);
    }
  }
  return m;
}

/** node -> the nodes it is paired with */
export function pairPartners(src: Iterable<string>) {
  const m = new Map<number, Set<number>>();
  const link = (a: number, b: number) => {
    let s = m.get(a);
    if (!s) { s = new Set(); m.set(a, s); }
    s.add(b);
  };
  for (const k of src) {
    const [a, b] = k.split("-").map(Number);
    link(a, b); link(b, a);
  }
  return m;
}

/** The nodes a click on `focus` keeps lit: its whole group if it has one, otherwise
    itself and its partners.  null when nothing is focused. */
export function focusNodes(focus: number | null, group: Map<number, number>,
                           partners: Map<number, Set<number>>) {
  if (focus === null) return null;
  const g = group.get(focus);
  if (g !== undefined) {
    const s = new Set<number>();
    for (const [n, gg] of group) if (gg === g) s.add(n);
    return s;
  }
  return new Set<number>([focus, ...(partners.get(focus) ?? [])]);
}

/** a stitch line survives the focus if either of its ends is lit */
export const isLit = (focusSet: Set<number> | null, a: number, b: number) =>
  focusSet === null || focusSet.has(a) || focusSet.has(b);

/* Which ground-truth stitch each edge belongs to, so both sides share a colour.  A
   specification carries that list; a DXF does not, and its ground truth arrives as node
   pairs instead -- without this fallback a scored DXF would render entirely grey. */
export function stitchIndexByEdge(pattern: Pattern, gt: Set<string>, keys: Array<[string, number]>) {
  const stitchOf = new Map<string, number>();
  if (pattern.stitches.length)
    pattern.stitches.forEach((sides, si) =>
      sides.forEach(([pn, ei]) => stitchOf.set(`${pn}#${ei}`, si)));
  else
    [...gt].forEach((k, si) => k.split("-").map(Number).forEach((n) => {
      const key = keys[n];
      if (key) stitchOf.set(`${key[0]}#${key[1]}`, si);
    }));
  return stitchOf;
}

/** Colour and width (before stroke scaling) of one edge. */
export function edgeStyle({ selected, stitch, inFocus, dim, annotate, scored }: {
  selected: boolean; stitch: number | undefined; inFocus: boolean; dim: boolean;
  annotate: boolean; scored: boolean;
}) {
  const color = selected ? "#f97316"
              : stitch === undefined || dim ? "#c9c9c9"
              : stitchColor(stitch);
  const width = selected ? 3.2
              : stitch === undefined ? (annotate ? 0.9 : 0.5)
              : inFocus ? 2.6 : scored ? 1.1 : 1.4;
  return { color, width };
}
