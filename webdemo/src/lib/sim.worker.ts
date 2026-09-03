/* The assembly solve, off the main thread.

   It is a long call into wasm with no yield points -- 26 s on a 13k-vertex garment and
   128 s on a 33k-vertex one, measured -- so running it inline does not make the page
   slow, it makes the page dead: no scrolling, no cancel, and eventually a browser
   "page unresponsive" dialog.  A worker costs one message each way and gives all of
   that back.

   Emscripten's simcpp.js is a MODULARIZE'd classic script with no ESM export, so a
   module worker cannot import it.  Fetching the text and evaluating it is the way in;
   the build already declares `worker` in ENVIRONMENT, so the module itself is happy. */
import { parseDump, runSim, type SimModule } from "./sim";

type Req = { dir: string; buf: ArrayBuffer };

let mod: Promise<SimModule> | null = null;

function loadSim(dir: string): Promise<SimModule> {
  if (!mod) {
    mod = fetch(`${dir}simcpp.js`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} fetching simcpp.js`);
        return r.text();
      })
      .then((src) => {
        const factory = new Function(`${src}; return createSim;`)() as
          (o?: Record<string, unknown>) => Promise<SimModule>;
        return factory({ locateFile: (p: string) => dir + p });
      });
  }
  return mod;
}

self.onmessage = async (e: MessageEvent<Req>) => {
  try {
    const Mod = await loadSim(e.data.dir);
    const dump = parseDump(e.data.buf);
    const r = runSim(Mod, dump);
    (self as unknown as Worker).postMessage({ ok: true, result: r }, [r.positions.buffer]);
  } catch (err: any) {
    (self as unknown as Worker).postMessage({ ok: false, error: String(err?.message ?? err) });
  }
};
