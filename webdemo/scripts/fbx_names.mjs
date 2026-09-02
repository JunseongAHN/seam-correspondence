// Do the FBX files name the individual panels?  If they do, the DXF piece <-> 3D panel
// matching is exact and the mirror ambiguity that shape invariants cannot break goes away.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { parseBinary } from "fbx-parser";

const files = process.argv.slice(2);
for (const f of files) {
  const buf = readFileSync(f);
  console.log(`\n=== ${f}  ${buf.length} bytes  sha256 ${createHash("sha256").update(buf).digest("hex").slice(0, 16)}`);
  let nodes;
  try { nodes = parseBinary(new Uint8Array(buf)); }
  catch (e) { console.log("  parseBinary failed:", e.message); continue; }

  const objects = nodes.find((n) => n.name === "Objects");
  if (!objects) { console.log("  no Objects node"); continue; }
  const kinds = {};
  for (const n of objects.nodes ?? []) kinds[n.name] = (kinds[n.name] ?? 0) + 1;
  console.log("  Objects children:", kinds);

  for (const kind of ["Model", "Geometry"]) {
    const list = (objects.nodes ?? []).filter((n) => n.name === kind);
    console.log(`  ${kind}: ${list.length}`);
    for (const n of list) {
      const nm = String(n.props?.[1] ?? "").split("::").pop();
      const sub = String(n.props?.[2] ?? "");
      let nv = "";
      const v = (n.nodes ?? []).find((c) => c.name === "Vertices");
      if (v) nv = `  verts ${(v.props?.[0]?.length ?? 0) / 3}`;
      const pv = (n.nodes ?? []).find((c) => c.name === "PolygonVertexIndex");
      const npoly = pv ? (pv.props?.[0] ?? []).filter((x) => x < 0).length : 0;
      console.log(`    ${nm.padEnd(34)} [${sub}]${nv}${npoly ? `  polys ${npoly}` : ""}`);
    }
  }
}
