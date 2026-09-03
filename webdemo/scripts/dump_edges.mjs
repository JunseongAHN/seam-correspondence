// What the BROWSER thinks each edge is: panel, index, arc length, endpoints.
// Run the same dump on the Python side and diff, to prove the two agree -- or find where
// they do not, which would mean a hand-drawn ground truth names different edges than the
// scorer reads.
import { readFileSync, writeFileSync } from "node:fs";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
const { parseDxf } = await server.ssrLoadModule("/src/lib/parseDxf.ts");
const { edgePolyline } = await server.ssrLoadModule("/src/lib/curves.ts");

const path = process.argv[2] ?? "C:/repos/seam-correspondence/clo_example/panel_seperated.dxf";
const pat = parseDxf(readFileSync(path, "utf8"), "clo");

const rows = [];
for (const panel of pat.panels) {
  for (const e of panel.edges) {
    const poly = edgePolyline(e);
    let arc = 0;
    for (let i = 1; i < poly.length; i++)
      arc += Math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]);
    rows.push({
      panel: panel.name, idx: e.idxInPanel,
      arc: +arc.toFixed(4),
      chord: +Math.hypot(e.end[0] - e.start[0], e.end[1] - e.start[1]).toFixed(4),
      start: e.start.map((x) => +x.toFixed(3)),
      end: e.end.map((x) => +x.toFixed(3)),
    });
  }
}
writeFileSync(process.argv[3] ?? "ts_edges.json", JSON.stringify(rows, null, 1));
console.log(`${rows.length} edges from the browser parse`);
for (const r of rows.filter((r) => r.panel === "11_M" || r.panel === "3_M"))
  console.log(`  ${r.panel}#${r.idx}  arc ${r.arc.toFixed(2)}  chord ${r.chord.toFixed(2)}`);
await server.close();
