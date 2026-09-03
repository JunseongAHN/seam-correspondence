/* Browser-side driver for the Emscripten build of simcpp (simcpp/build-wasm-simd).
   Same call sequence as simcpp/test_wasm.js: read the SIMCPP01 dump, copy the
   arrays into the wasm heap, _sim_build then _sim_solve, read positions back.

   The emitted simcpp.js is MODULARIZE'd but has no ESM export, so it must not go
   through Vite's module graph: it is loaded as a classic <script> from public/sim/
   and picked up off `window.createSim`, with locateFile pointing at the served
   sibling .wasm. */

export type SimModule = {
  _malloc(n: number): number;
  _free(p: number): void;
  _sim_build(n: number, M: number, faces: number, wid: number, pof: number, rest: number,
             K: number, pairs: number, mu: number, NC: number, cyl: number,
             NS: number, sph: number): number;
  _sim_solve(p0: number, nLadder: number, ladder: number, w0: number, w1: number,
             factor: number, itersPerStage: number, perLambda: number, maxIter: number,
             tol: number, recenter: number, verbose: number, out: number): number;
  _sim_mono_violations(): number;
  _sim_seconds(): number;
  _sim_energy(which: number): number;
  _sim_free(): void;
  HEAPF64: Float64Array;
  HEAP32: Int32Array;
};

declare global {
  interface Window { createSim?: (opts?: Record<string, unknown>) => Promise<SimModule> }
}

export type Dump = {
  n: number; M: number; K: number; NC: number; NS: number; recenter: number;
  faces: Int32Array; wid: Int32Array; panelOfFace: Int32Array;
  rest: Float64Array; P0: Float64Array; pairs: Int32Array;
  mu: Float64Array | null; cyl: Float64Array; sph: Float64Array;
};

export type SimResult = {
  nHinges: number; iters: number; wallMs: number; solverSeconds: number;
  monoViolations: number; energies: [number, number, number, number];
  positions: Float64Array;
};

const base = () => `${import.meta.env.BASE_URL}sim/`;

let modPromise: Promise<SimModule> | null = null;

/** Inject simcpp.js as a classic script, then instantiate it with locateFile. */
export function loadSim(): Promise<SimModule> {
  if (modPromise) return modPromise;
  const dir = base();
  modPromise = new Promise<void>((resolve, reject) => {
    if (window.createSim) return resolve();
    const s = document.createElement("script");
    s.src = `${dir}simcpp.js`;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`failed to load ${s.src}`));
    document.head.appendChild(s);
  }).then(() => {
    if (!window.createSim) throw new Error("simcpp.js loaded but window.createSim is undefined");
    return window.createSim({ locateFile: (p: string) => dir + p });
  });
  return modPromise;
}

/** Parse the SIMCPP01 binary dump (simcpp/dump_garment.py). */
export function parseDump(buf: ArrayBuffer): Dump {
  const v = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 8));
  if (magic !== "SIMCPP01") throw new Error(`bad magic ${JSON.stringify(magic)}`);
  let o = 8;
  const h: number[] = [];
  for (let i = 0; i < 8; i++) { h.push(v.getInt32(o, true)); o += 4; }
  const [n, M, K, NC, NS, hasMu, hasNu, recenter] = h;
  const i32 = (c: number) => { const a = new Int32Array(c); for (let i = 0; i < c; i++) { a[i] = v.getInt32(o, true); o += 4; } return a; };
  const f64 = (c: number) => { const a = new Float64Array(c); for (let i = 0; i < c; i++) { a[i] = v.getFloat64(o, true); o += 8; } return a; };
  const faces = i32(M * 3), wid = i32(n), panelOfFace = i32(M);
  const rest = f64(n * 2), P0 = f64(n * 3), pairs = i32(K * 2);
  const mu = hasMu ? f64(n) : null;
  if (hasNu) { f64(n); f64(n * 3); }          // nu / anchor: read past, unused by this path
  const cyl = f64(NC * 7), sph = f64(NS * 4);
  return { n, M, K, NC, NS, recenter, faces, wid, panelOfFace, rest, P0, pairs, mu, cyl, sph };
}

/** _sim_build + _sim_solve with the same parameters test_wasm.js uses. */
export function runSim(Mod: SimModule, d: Dump): SimResult {
  const put = (arr: Float64Array | Int32Array) => {
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

  const ladder = new Float64Array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]);
  const pLadder = put(ladder);
  const pP0 = put(d.P0);
  const pOut = Mod._malloc(d.n * 3 * 8);

  const t0 = performance.now();
  const iters = Mod._sim_solve(pP0, ladder.length, pLadder,
                               1e-2, 1e4, 2.0, 10, 30, 20000, 1e-10,
                               d.recenter, 0, pOut);
  const wallMs = performance.now() - t0;

  const positions = new Float64Array(d.n * 3);
  positions.set(Mod.HEAPF64.subarray(pOut >> 3, (pOut >> 3) + d.n * 3));

  const res: SimResult = {
    nHinges, iters, wallMs, solverSeconds: Mod._sim_seconds(),
    monoViolations: Mod._sim_mono_violations(),
    energies: [Mod._sim_energy(0), Mod._sim_energy(1), Mod._sim_energy(2), Mod._sim_energy(3)],
    positions,
  };

  Mod._sim_free();
  for (const p of [pFaces, pWid, pPof, pRest, pPairs, pMu, pCyl, pSph, pLadder, pP0, pOut])
    if (p) Mod._free(p);
  return res;
}

/** Fetch and parse a dump WITHOUT solving it.  Parsing is milliseconds; solving blocks
    the thread for ~10 s, so a viewer can show the input geometry straight away and only
    pay for the solve when it is asked for. */
export async function loadDump(name = "trousers.bin"): Promise<Dump> {
  const r = await fetch(`${base()}${name}`);
  if (!r.ok) throw new Error(`${r.status} fetching ${name}`);
  return parseDump(await r.arrayBuffer());
}

/** Solve a dump that is already parsed. */
export async function solveParsed(d: Dump): Promise<SimResult> {
  return runSim(await loadSim(), d);
}

/** Fetch a dump from public/sim/ and solve it. */
export async function solveDump(name = "trousers.bin"): Promise<SimResult> {
  return (await solveDumpFull(name)).result;
}

/** Same, but hand back the parsed dump too: the flat rest panels, the initial
    placement and the body primitives are what a viewer needs alongside the result. */
export async function solveDumpFull(
  name = "trousers.bin",
): Promise<{ dump: Dump; result: SimResult }> {
  const [Mod, buf] = await Promise.all([
    loadSim(),
    fetch(`${base()}${name}`).then((r) => {
      if (!r.ok) throw new Error(`${r.status} fetching ${name}`);
      return r.arrayBuffer();
    }),
  ]);
  const dump = parseDump(buf);
  return { dump, result: runSim(Mod, dump) };
}

/** Does this browser's WebAssembly accept a v128 (fixed-width SIMD) module? */
export function simdSupported(): boolean {
  // (module (func (result v128) v128.const i32x4 0 0 0 0))
  return WebAssembly.validate(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
    10, 22, 1, 20, 0, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
  ]));
}

/** Solve a parsed dump in a Web Worker, so the page stays alive while it runs.

    The dump's buffers are re-serialised rather than transferred: the caller is showing
    the flat panels and the initial placement from that same object while the solve
    runs, and a transferred ArrayBuffer would be detached out from under it. */
export function solveInWorker(d: Dump, raw: ArrayBuffer): Promise<SimResult> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL("./sim.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e) => {
      w.terminate();
      if (e.data.ok) resolve(e.data.result as SimResult);
      else reject(new Error(e.data.error));
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "worker failed")); };
    w.postMessage({ dir: base(), buf: raw.slice(0) }, [] );
    void d;
  });
}

/** Fetch a dump and keep the raw bytes: the worker needs them again to parse its own
    copy, and the main thread needs the parsed form to draw the input straight away. */
export async function loadDumpRaw(name: string): Promise<{ dump: Dump; raw: ArrayBuffer }> {
  const r = await fetch(`${base()}${name}`);
  if (!r.ok) throw new Error(`${r.status} fetching ${name}`);
  const raw = await r.arrayBuffer();
  return { dump: parseDump(raw), raw };
}
