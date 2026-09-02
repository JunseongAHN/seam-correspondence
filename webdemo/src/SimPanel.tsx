/* Smallest thing that proves the wasm solver runs in the browser: load the
   module from public/sim/, solve data/trousers.bin, show timing + energies. */
import { useCallback, useEffect, useState } from "react";
import { simdSupported, solveDump, type SimResult } from "./lib/sim";

export default function SimPanel() {
  const [res, setRes] = useState<SimResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = useCallback(async () => {
    setBusy(true); setErr(null);
    try { setRes(await solveDump("trousers.bin")); }
    catch (e: any) { setErr(String(e?.stack ?? e?.message ?? e)); setRes(null); }
    finally { setBusy(false); }
  }, []);

  // headless hook: the CDP smoke test calls window.__runSim() and reads the JSON
  useEffect(() => {
    (window as any).__runSim = async () => {
      const r = await solveDump("trousers.bin");
      return { ...r, positions: undefined, p0: r.positions[0], simd: simdSupported() };
    };
  }, []);

  return (
    <div className="sim" style={{ margin: "1rem 0", fontFamily: "monospace", fontSize: 13 }}>
      <button onClick={run} disabled={busy}>
        {busy ? "solving…" : "run wasm solver on trousers.bin"}
      </button>
      {err && <pre style={{ color: "#c33", whiteSpace: "pre-wrap" }}>{err}</pre>}
      {res && (
        <pre style={{ whiteSpace: "pre-wrap" }}>
{`simd supported: ${simdSupported()}
hinges: ${res.nHinges}   iters: ${res.iters}
wall: ${(res.wallMs / 1000).toFixed(2)} s (solver ${res.solverSeconds.toFixed(2)} s)
mono violations: ${res.monoViolations}
E_arap  = ${res.energies[0]}
E_bend  = ${res.energies[1]}
E_stitch= ${res.energies[2]}
E_obst  = ${res.energies[3]}`}
        </pre>
      )}
    </div>
  );
}
