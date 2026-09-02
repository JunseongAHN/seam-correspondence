// Verify that the `fbx-parser` npm package can parse a real binary FBX and
// reproduce the numbers from dxfcheck/fbx_read.py.
//
//   node scripts/fbx_probe.mjs [path/to/file.fbx]
//
// Only `fs` is used to get bytes off disk (in the browser that becomes
// `new Uint8Array(await (await fetch(url)).arrayBuffer())`); everything else is
// library API and runs unchanged in a browser bundle.

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import pkg from 'fbx-parser'

const { parseBinary, parseText, FBXReader } = pkg

const here = path.dirname(fileURLToPath(import.meta.url))
const file = process.argv[2] ?? path.resolve(here, '../../clo_example/draped_seperated.fbx')

// ---------------------------------------------------------------- parse -----
const buf = readFileSync(file)
// Prove the parser only needs a plain Uint8Array (no Buffer-only methods).
const bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)

let fbx
try {
  fbx = parseBinary(bytes)
} catch (e) {
  console.error('parseBinary threw:', e)
  process.exit(1)
}

console.log(`file            : ${file}`)
console.log(`bytes           : ${bytes.length}`)
console.log(`parseBinary()   : ${Array.isArray(fbx) ? `Array(${fbx.length})` : typeof fbx}`)
console.log(`top-level nodes : ${fbx.map((n) => n.name).join(', ')}`)

// parseBinary consumes the header and does not return the version; it is also
// carried in FBXHeaderExtension/FBXVersion.
const root = new FBXReader(fbx)
const ver = root.node('FBXHeaderExtension')?.node('FBXVersion')?.prop(0, 'number')
console.log(`FBXVersion      : ${ver}`)

// ------------------------------------------------------- collect meshes -----
// Same walk as dxfcheck/fbx_read.py: Objects > Geometry / Model, joined through
// the Connections "OO" records (child geometry id -> parent model id).
const objects = fbx.filter((n) => n.name === 'Objects')

// fbx-parser rewrites the FBX "name\x00\x01Class" strings as "Class::name".
const bareName = (s) =>
  typeof s === 'string' ? (s.includes('::') ? s.split('::').pop() : s.split('\x00')[0]) : ''

const geo = new Map()
const mdl = new Map()
for (const o of objects) {
  for (const g of o.nodes.filter((n) => n.name === 'Geometry')) {
    const gid = Number(g.props[0])
    const verts = g.nodes.find((n) => n.name === 'Vertices')?.props[0]
    const idx = g.nodes.find((n) => n.name === 'PolygonVertexIndex')?.props[0]
    if (!verts) continue
    geo.set(gid, { verts, idx, name: bareName(g.props[1]) })
  }
  for (const m of o.nodes.filter((n) => n.name === 'Model')) {
    mdl.set(Number(m.props[0]), { name: bareName(m.props[1]) })
  }
}

const link = new Map() // child id -> parent id
for (const c of fbx.filter((n) => n.name === 'Connections')) {
  for (const cc of c.nodes.filter((n) => n.name === 'C')) {
    if (cc.props.length >= 3 && cc.props[0] === 'OO') link.set(Number(cc.props[1]), Number(cc.props[2]))
  }
}

const meshes = []
for (const [gid, g] of geo) {
  const m = mdl.get(link.get(gid))
  // Same polygon split as the Python reference: a negative index terminates a
  // polygon and its real value is (-i - 1).
  const polys = []
  let cur = []
  for (const i of g.idx ?? []) {
    if (i < 0) {
      cur.push(-i - 1)
      polys.push(cur)
      cur = []
    } else cur.push(i)
  }
  meshes.push({ name: m ? m.name : g.name, verts: g.verts, nVerts: g.verts.length / 3, polys })
}

console.log(`\nmeshes: ${meshes.length}`)
for (const m of meshes) {
  console.log(
    `  ${m.name.slice(0, 34).padEnd(36)} verts ${String(m.nVerts).padStart(6)}  polys ${String(
      m.polys.length
    ).padStart(6)}`
  )
}

// --------------------------------------------------------- garment mesh -----
const garment =
  meshes.find((m) => m.name.includes('draped_seperated')) ??
  meshes.reduce((a, b) => (b.polys.length > a.polys.length ? b : a))

console.log(`\ngarment mesh   : ${garment.name}`)
console.log(`  vertices     : ${garment.nVerts}`)
console.log(`  polygons     : ${garment.polys.length}`)

// ---------------------------------------------- connected components --------
// Union-find over vertex indices; every polygon edge (including the wrap-around
// edge) unions its two endpoints.
const parent = new Int32Array(garment.nVerts)
for (let i = 0; i < parent.length; ++i) parent[i] = i
const find = (a) => {
  while (parent[a] !== a) a = parent[a] = parent[parent[a]]
  return a
}
const union = (a, b) => {
  const ra = find(a)
  const rb = find(b)
  if (ra !== rb) parent[rb] = ra
}
for (const p of garment.polys) for (let k = 0; k < p.length; ++k) union(p[k], p[(k + 1) % p.length])

const comp = new Int32Array(garment.nVerts)
const roots = new Map()
for (let i = 0; i < garment.nVerts; ++i) {
  const r = find(i)
  if (!roots.has(r)) roots.set(r, roots.size)
  comp[i] = roots.get(r)
}
const compSizes = new Array(roots.size).fill(0)
for (let i = 0; i < garment.nVerts; ++i) compSizes[comp[i]]++
console.log(`  components   : ${roots.size}  sizes ${JSON.stringify(compSizes)}`)

// -------------------------------------- cross-component welded pairs --------
// Vertices sharing a 3D position but belonging to different components are the
// seam welds.  Two counting conventions are reported because they disagree on
// clusters holding more than two coincident vertices:
//   spanning  : (distinct components in the cluster - 1) per cluster  <- ref 240
//   all-pairs : every unordered cross-component pair inside the cluster
const V = garment.verts
function welds(tol) {
  const q = tol > 0 ? 1 / tol : 0
  const key = (i) =>
    tol > 0
      ? Math.round(V[3 * i] * q) + ',' + Math.round(V[3 * i + 1] * q) + ',' + Math.round(V[3 * i + 2] * q)
      : V[3 * i] + ',' + V[3 * i + 1] + ',' + V[3 * i + 2]
  const buckets = new Map()
  for (let i = 0; i < garment.nVerts; ++i) {
    const k = key(i)
    let b = buckets.get(k)
    if (!b) buckets.set(k, (b = []))
    b.push(i)
  }
  let allPairs = 0
  let spanning = 0
  let nClusters = 0
  const hist = {}
  for (const b of buckets.values()) {
    if (b.length < 2) continue
    const comps = new Set(b.map((i) => comp[i]))
    if (comps.size < 2) continue
    nClusters++
    spanning += comps.size - 1
    const k = `${b.length}v/${comps.size}c`
    hist[k] = (hist[k] || 0) + 1
    for (let a = 0; a < b.length; ++a)
      for (let c = a + 1; c < b.length; ++c) if (comp[b[a]] !== comp[b[c]]) allPairs++
  }
  return { allPairs, spanning, nClusters, hist }
}

console.log('\n  cross-component welds (coincident vertices in different components):')
for (const tol of [0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2]) {
  const w = welds(tol)
  console.log(
    `    tol ${(tol === 0 ? 'exact' : tol.toExponential(0)).padEnd(6)} clusters ${String(
      w.nClusters
    ).padStart(4)}   spanning(ncomp-1) ${String(w.spanning).padStart(4)}   all-pairs ${String(
      w.allPairs
    ).padStart(4)}`
  )
}
console.log(`  cluster shapes (exact): ${JSON.stringify(welds(0).hist)}`)

// ----------------------------------------------------- shape dump -----------
const gnode = objects[0].nodes.find((n) => n.name === 'Geometry')
console.log('\nshape of a parsed node (Geometry):')
console.log(`  keys        : ${Object.keys(gnode).join(', ')}`)
console.log(`  name        : ${JSON.stringify(gnode.name)}`)
console.log(
  `  props       : [${gnode.props
    .map((p) =>
      Array.isArray(p) ? `Array(${p.length}) of ${typeof p[0]}` : `${typeof p} ${JSON.stringify(p).slice(0, 32)}`
    )
    .join(', ')}]`
)
console.log(`  nodes       : ${gnode.nodes.map((n) => n.name).join(', ')}`)
const vnode = gnode.nodes.find((n) => n.name === 'Vertices')
console.log(
  `  Vertices.props[0] -> ${vnode.props[0].constructor.name}, length ${vnode.props[0].length}, first 3 = ${vnode.props[0].slice(
    0,
    3
  )}`
)
const inode = gnode.nodes.find((n) => n.name === 'PolygonVertexIndex')
console.log(
  `  PolygonVertexIndex.props[0] -> ${inode.props[0].constructor.name}, length ${
    inode.props[0].length
  }, first 8 = ${inode.props[0].slice(0, 8)}`
)
const cnode = fbx.find((n) => n.name === 'Connections').nodes[0]
console.log(
  `  Connections C props -> ${JSON.stringify(cnode.props.map((p) => typeof p))} = ${JSON.stringify(cnode.props)}`
)
console.log(`  parseText is a separate export: ${typeof parseText}`)
