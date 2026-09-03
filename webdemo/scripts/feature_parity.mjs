// Does the TypeScript feature extractor produce the same numbers as the Python one?
// The model was trained on the Python features, so any divergence silently degrades the
// prediction rather than raising.  Dumps the TS side; compare.py-style check on the
// Python side reads the same JSON.
import { readFileSync, writeFileSync } from "node:fs";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
const { parseSpec } = await server.ssrLoadModule("/src/lib/parseSpec.ts");
const { parseDxf } = await server.ssrLoadModule("/src/lib/parseDxf.ts");
const { patternToTensors, featureDim, ENCODING } = await server.ssrLoadModule("/src/lib/features.ts");

const out = { encoding: ENCODING, featureDim, cases: {} };
for (const [name, path, kind] of [
  ["spec", "C:/repos/seam-correspondence/data/rand_00YONAPXZE/rand_00YONAPXZE_specification.json", "spec"],
  ["clo", "C:/repos/seam-correspondence/clo_example/panel_seperated.dxf", "dxf"],
]) {
  const text = readFileSync(path, "utf8");
  const pat = kind === "dxf" ? parseDxf(text, name) : parseSpec(JSON.parse(text), name);
  const t = patternToTensors(pat);
  out.cases[name] = { M: t.M, keys: t.keys, x: Array.from(t.x) };
  console.log(`${name}: M=${t.M} dim=${featureDim}`);
}
writeFileSync(process.argv[2], JSON.stringify(out));
console.log(`wrote ${process.argv[2]}  encoding=${ENCODING} dim=${featureDim}`);
await server.close();
