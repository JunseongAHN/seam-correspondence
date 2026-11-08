/* A hand-drawn ground truth as a file.  It names panel/edge, not node indices, so it
   survives a reparse. */

export type GtSource = { pattern: { panels: unknown[] }; M: number; keys: Array<[string, number]> };

export function gtToDoc(res: GtSource, pairs: Set<string>, name: string) {
  return {
    source: name,
    panels: res.pattern.panels.length,
    edges: res.M,
    stitches: [...pairs].map((k) => k.split("-").map(Number).map((n) => res.keys[n])),
  };
}

export function gtFromDoc(doc: any, keys: Array<[string, number]>): Set<string> {
  const idx = new Map(keys.map(([p, e], i) => [`${p}#${e}`, i]));
  const out = new Set<string>();
  let miss = 0;
  for (const st of doc.stitches ?? []) {
    const ns = (st as Array<[string, number]>).map(([p, e]) => idx.get(`${p}#${e}`));
    if (ns.length !== 2 || ns.some((n) => n === undefined)) { miss++; continue; }
    const [a, b] = ns as number[];
    out.add(`${Math.min(a, b)}-${Math.max(a, b)}`);
  }
  if (miss) throw new Error(`${miss} stitch(es) in the file name edges this pattern does not have`);
  return out;
}

export function gtFromText(text: string, keys: Array<[string, number]>): Set<string> {
  return gtFromDoc(JSON.parse(text), keys);
}
