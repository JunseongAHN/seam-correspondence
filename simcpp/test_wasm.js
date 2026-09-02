// Drive the WASM module the way a browser would: read the dump, copy the arrays
// into the wasm heap, call the C API, read the positions back out.
//
//   node test_wasm.js data/trousers.bin out/wasm_api_trousers.npy
//
// Nothing here is Node-specific except reading/writing the two files; the
// simBuild / simSolve calls are exactly what a browser page would make.

const fs = require('fs');
const createSim = require('./build-wasm/simcpp.js');

function readDump(path) {
  const buf = fs.readFileSync(path);
  if (buf.toString('latin1', 0, 8) !== 'SIMCPP01') throw new Error('bad magic');
  let o = 8;
  const h = [];
  for (let i = 0; i < 8; i++) { h.push(buf.readInt32LE(o)); o += 4; }
  const [n, M, K, NC, NS, hasMu, hasNu, recenter] = h;
  const i32 = (c) => { const a = new Int32Array(c); for (let i = 0; i < c; i++) { a[i] = buf.readInt32LE(o); o += 4; } return a; };
  const f64 = (c) => { const a = new Float64Array(c); for (let i = 0; i < c; i++) { a[i] = buf.readDoubleLE(o); o += 8; } return a; };
  const d = { n, M, K, NC, NS, recenter };
  d.faces = i32(M * 3);
  d.wid = i32(n);
  d.panelOfFace = i32(M);
  d.rest = f64(n * 2);
  d.P0 = f64(n * 3);
  d.pairs = i32(K * 2);
  d.mu = hasMu ? f64(n) : null;
  if (hasNu) { d.nu = f64(n); d.anchor = f64(n * 3); }
  d.cyl = f64(NC * 7);
  d.sph = f64(NS * 4);
  return d;
}

function writeNpy(path, P, n) {
  let dict = `{'descr': '<f8', 'fortran_order': False, 'shape': (${n}, 3), }`;
  const pad = (64 - (10 + dict.length + 1) % 64) % 64;
  dict += ' '.repeat(pad) + '\n';
  const head = Buffer.alloc(10 + dict.length);
  Buffer.from('\x93NUMPY\x01\x00', 'latin1').copy(head, 0);
  head.writeUInt16LE(dict.length, 8);
  Buffer.from(dict, 'latin1').copy(head, 10);
  fs.writeFileSync(path, Buffer.concat([head, Buffer.from(P.buffer, P.byteOffset, n * 3 * 8)]));
}

(async () => {
  const dumpPath = process.argv[2] || 'data/trousers.bin';
  const outPath = process.argv[3] || 'out/wasm_api.npy';
  const d = readDump(dumpPath);
  const Mod = await createSim();

  // copy a typed array into the wasm heap, return the byte offset
  const put = (arr) => {
    const p = Mod._malloc(arr.byteLength);
    if (arr instanceof Float64Array) Mod.HEAPF64.set(arr, p >> 3);
    else Mod.HEAP32.set(arr, p >> 2);
    return p;
  };

  const pFaces = put(d.faces), pWid = put(d.wid), pPof = put(d.panelOfFace);
  const pRest = put(d.rest), pPairs = put(d.pairs);
  const pMu = d.mu ? put(d.mu) : 0;
  const pCyl = d.NC ? put(d.cyl) : 0, pSph = d.NS ? put(d.sph) : 0;

  const nHinges = Mod._sim_build(d.n, d.M, pFaces, pWid, pPof, pRest, d.K, pPairs,
                                 pMu, d.NC, pCyl, d.NS, pSph);
  console.log(`built: ${d.n} verts, ${d.M} faces, ${nHinges} hinges, ${d.K} seam pairs`);

  const ladder = new Float64Array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]);
  const pLadder = put(ladder);
  const pP0 = put(d.P0);
  const pOut = Mod._malloc(d.n * 3 * 8);

  const t0 = Date.now();
  const iters = Mod._sim_solve(pP0, ladder.length, pLadder,
                               1e-2,   // w0
                               1e4,    // w1
                               2.0,    // factor
                               10,     // iters_per_stage
                               30,     // per_lambda
                               20000,  // max_iter
                               1e-10,  // tol
                               d.recenter, 0 /* verbose */, pOut);
  const wall = (Date.now() - t0) / 1000;

  const P = new Float64Array(d.n * 3);
  P.set(Mod.HEAPF64.subarray(pOut >> 3, (pOut >> 3) + d.n * 3));
  writeNpy(outPath, P, d.n);

  console.log(`${iters} iters in ${wall.toFixed(2)} s wall ` +
              `(${Mod._sim_seconds().toFixed(2)} s in the solver), ` +
              `mono violations ${Mod._sim_mono_violations()}`);
  console.log(`E_arap=${Mod._sim_energy(0)} E_bend=${Mod._sim_energy(1)} ` +
              `E_stitch=${Mod._sim_energy(2)} E_obst=${Mod._sim_energy(3)}`);

  // the embind surface reaches the same entry points
  console.log(`embind simNHinges() -> ${Mod.simNHinges()}`);
  console.log(`wrote ${outPath}`);
  Mod._sim_free();
})();
