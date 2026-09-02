// What does dxf-parser give us for the CLO file?
import { readFileSync } from "node:fs";
import DxfParser from "dxf-parser";

const txt = readFileSync("C:/repos/seam-correspondence/clo_example/panel_seperated.dxf", "utf8");
const t0 = performance.now();
const d = new DxfParser().parseSync(txt);
console.log(`parseSync: ${(performance.now() - t0).toFixed(0)} ms`);
console.log("top-level keys:", Object.keys(d));

const names = Object.keys(d.blocks ?? {});
console.log(`blocks: ${names.length}`, names.slice(0, 12));

const b = d.blocks[names.find((n) => !n.startsWith("*"))];
console.log("\nblock keys:", Object.keys(b));
const ents = b.entities ?? [];
const byType = {};
for (const e of ents) byType[e.type] = (byType[e.type] ?? 0) + 1;
console.log("entity types:", byType);

const layers = {};
for (const e of ents) {
  const k = `${e.layer}/${e.type}`;
  layers[k] = (layers[k] ?? 0) + 1;
}
console.log("layer/type:", layers);

const poly = ents.find((e) => e.type === "POLYLINE" && String(e.layer) === "8");
console.log("\nPOLYLINE(layer 8) keys:", poly && Object.keys(poly));
if (poly) {
  console.log("  vertices:", poly.vertices?.length, "first:", poly.vertices?.[0]);
  console.log("  shape/closed flags:", poly.shape, poly.closed, poly.flag);
}
const pt = ents.find((e) => e.type === "POINT" && String(e.layer) === "2");
console.log("POINT(layer 2) keys:", pt && Object.keys(pt), pt?.position);
const tx = ents.find((e) => e.type === "TEXT");
console.log("TEXT keys:", tx && Object.keys(tx), tx?.text, tx?.startPoint);

console.log("\nmodelspace entities:", (d.entities ?? []).length,
            [...new Set((d.entities ?? []).map((e) => e.type))]);
const ins = (d.entities ?? []).find((e) => e.type === "INSERT");
console.log("INSERT keys:", ins && Object.keys(ins), ins?.name, ins?.position);
