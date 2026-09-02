// Run the browser DXF path outside the browser: Vite resolves the TS exactly as it
// does for the app, so this exercises the real parseDxf/features code.
import { readFileSync } from "node:fs";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
const { parseDxf } = await server.ssrLoadModule("/src/lib/parseDxf.ts");
const { patternToTensors } = await server.ssrLoadModule("/src/lib/features.ts");

const file = process.argv[2] ?? "C:/repos/seam-correspondence/clo_example/panel_seperated.dxf";
const text = readFileSync(file, "utf8");

const t0 = performance.now();
const pat = parseDxf(text, "panel_seperated");
const t1 = performance.now();
const t = patternToTensors(pat);
const t2 = performance.now();

console.log(`parseDxf        ${(t1 - t0).toFixed(0)} ms`);
console.log(`patternToTensors ${(t2 - t1).toFixed(0)} ms`);
console.log(`panels ${pat.panels.length}   edges(M) ${t.M}   x ${t.x.length / t.M} dims`);

const KTN = ["straight", "circle", "quad", "cubic", "bspline", "unknown"];
const hist = {};
for (const p of pat.panels) for (const e of p.edges) hist[KTN[e.kt]] = (hist[KTN[e.kt]] ?? 0) + 1;
console.log("refitted curve types:", hist);

console.log("\nper panel:");
for (const p of pat.panels) {
  const xs = p.edges.flatMap((e) => [e.start[0], e.end[0]]);
  const ys = p.edges.flatMap((e) => [e.start[1], e.end[1]]);
  console.log(`  ${p.name.padEnd(20)} edges ${String(p.edges.length).padStart(3)}`
    + `  local bbox ${(Math.max(...xs) - Math.min(...xs)).toFixed(1)}`
    + ` x ${(Math.max(...ys) - Math.min(...ys)).toFixed(1)} cm`);
}

// sanity: features must be finite and the panel-local frame must start at the origin
let bad = 0;
for (let i = 0; i < t.x.length; i++) if (!Number.isFinite(t.x[i])) bad++;
console.log(`\nnon-finite feature values: ${bad}`);

await server.close();
