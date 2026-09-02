import { useCallback, useRef, useState } from "react";
import { parseSpec, type Pattern } from "./lib/parseSpec";
import { parseDxf } from "./lib/parseDxf";
import { patternToTensors, gtPairs } from "./lib/features";
import { predict } from "./lib/infer";
import { placePanels, type Placed } from "./lib/render";
import PatternSvg from "./PatternSvg";
import WeldGT from "./WeldGT";         // the CLO drape with its weld-derived seams
import SimViewer from "./SimViewer";   // the wasm assembly solve, rendered
import "./App.css";

// BASE_URL, not an absolute path: the GitHub Pages build is served from /<repo>/.
const B = import.meta.env.BASE_URL;
const EXAMPLES = {
  clo: {
    label: "1. CLO tutorial example",
    file: "panel_seperated.dxf",
    url: `${B}example/panel_seperated.dxf`,
    right: "drape" as const,
    note: "10 pieces exported from CLO. A DXF carries no stitch list, so the prediction "
        + "is shown on its own. The right pane is CLO's own drape, not a simulation of "
        + "this prediction.",
  },
  gcd: {
    label: "2. GarmentCode test data",
    file: "rand_1328ERLDIC_specification.json",
    url: `${B}example/rand_1328ERLDIC_specification.json`,
    right: "sim" as const,
    note: "A held-out GarmentCodeData garment, so the prediction can be scored. The "
        + "right pane assembles this same garment with the wasm solver.",
  },
};
type ExampleKey = keyof typeof EXAMPLES;

type Result = {
  pattern: Pattern; placed: Placed[]; keys: Array<[string, number]>;
  pred: Set<string>; gt: Set<string>; ms: number; M: number;
  source: "spec" | "dxf";
};

export default function App() {
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [show, setShow] = useState({ correct: true, fp: true, fn: true, gt: false });
  const [name, setName] = useState("");
  const [example, setExample] = useState<ExampleKey | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  /** One pipeline for both inputs: text -> Pattern -> features -> ONNX -> pairs.
      A DXF carries no stitches, so there is nothing to score it against. */
  const runText = useCallback(async (text: string, fileName: string, ex: ExampleKey | null) => {
    setBusy(true); setErr(null);
    try {
      const isDxf = /\.dxf$/i.test(fileName);
      const pattern = isDxf
        ? parseDxf(text, fileName.replace(/\.dxf$/i, ""))
        : parseSpec(JSON.parse(text), fileName.replace("_specification.json", ""));
      const t = patternToTensors(pattern);
      const { pairs, ms } = await predict(t);
      setRes({ pattern, placed: placePanels(pattern), keys: t.keys, pred: pairs,
               gt: isDxf ? new Set<string>() : gtPairs(pattern, t.keys),
               ms, M: t.M, source: isDxf ? "dxf" : "spec" });
      setName(fileName);
      setExample(ex);
    } catch (e: any) { setErr(String(e?.message ?? e)); setRes(null); setExample(null); }
    finally { setBusy(false); }
  }, []);

  const runFile = useCallback(async (file: File) => {
    await runText(await file.text(), file.name, null);
  }, [runText]);

  const runExample = useCallback(async (key: ExampleKey) => {
    const e = EXAMPLES[key];
    setBusy(true); setErr(null);
    try {
      const r = await fetch(e.url);
      if (!r.ok) throw new Error(`${e.url}: HTTP ${r.status}`);
      await runText(await r.text(), e.file, key);
    } catch (ex: any) { setErr(String(ex?.message ?? ex)); setBusy(false); }
  }, [runText]);

  const scored = !!res && res.gt.size > 0;
  const stats = (() => {
    if (!res) return null;
    const ok = [...res.pred].filter((k) => res.gt.has(k)).length;
    const P = res.pred.size ? ok / res.pred.size : 0;
    const R = res.gt.size ? ok / res.gt.size : 0;
    return { ok, fp: res.pred.size - ok, fn: res.gt.size - ok,
             f1: P + R ? (2 * P * R) / (P + R) : 0 };
  })();

  return (
    <div className="app">
      <header>
        <h1>AutoSew — sewing pattern → stitch prediction</h1>
        <p className="sub">
          A 2D sewing pattern in, the stitching between panel edges out, predicted by an
          ONNX model running in your browser.
        </p>
      </header>

      <div className="drop"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) runFile(f); }}
        onClick={() => fileRef.current?.click()}>
        <input ref={fileRef} type="file" accept=".dxf,.json" hidden
               onChange={(e) => { const f = e.target.files?.[0]; if (f) runFile(f); }} />
        {busy ? "running…" : "import a DXF — drop it here, or click to choose"}
      </div>

      <div className="tryit">
        or try{" "}
        {(Object.keys(EXAMPLES) as ExampleKey[]).map((k, i) => (
          <span key={k}>
            {i > 0 && <span className="sep">·</span>}
            <button className={`link ${example === k ? "on" : ""}`}
                    disabled={busy} onClick={() => runExample(k)}>
              {EXAMPLES[k].label}
            </button>
          </span>
        ))}
      </div>

      {err && <div className="err">{err}</div>}

      {res && (
        <>
          <div className="bar">
            <span className="pill name">{name}</span>
            <span className="pill">{res.pattern.panels.length} panels</span>
            <span className="pill">{res.M} edges</span>
            {scored && stats ? (
              <>
                <span className="pill ok">correct {stats.ok}</span>
                <span className="pill fp">FP {stats.fp}</span>
                <span className="pill fn">FN {stats.fn}</span>
                <span className="pill">F1 {stats.f1.toFixed(3)}</span>
              </>
            ) : (
              <>
                <span className="pill ok">{res.pred.size} predicted stitches</span>
                <span className="pill">no ground truth in a DXF</span>
              </>
            )}
            <span className="pill t">{res.ms.toFixed(0)} ms</span>
          </div>

          {example && <p className="note wide">{EXAMPLES[example].note}</p>}

          {scored && (
            <div className="toggles">
              {(["correct", "fp", "fn", "gt"] as const).map((k) => (
                <label key={k}>
                  <input type="checkbox" checked={show[k]}
                         onChange={(e) => setShow({ ...show, [k]: e.target.checked })} />
                  {k === "gt" ? "ground-truth stitches" : k}
                </label>
              ))}
            </div>
          )}

          {example ? (
            <div className="split">
              <section>
                <h2>predicted stitching · 2D</h2>
                <PatternSvg {...res} show={show} height="62vh" />
              </section>
              <section>
                <h2>{EXAMPLES[example].right === "sim"
                      ? "assembled by the wasm solver · 3D"
                      : "CLO's own drape · 3D"}</h2>
                {EXAMPLES[example].right === "sim" ? <SimViewer /> : <WeldGT />}
              </section>
            </div>
          ) : (
            <>
              {!scored && (
                <p className="note wide">
                  Edges of a predicted pair share a colour. Click an edge to focus its
                  correspondence — it is drawn thick and every other pair drops to grey.
                  Click again, or click the background, to clear.
                </p>
              )}
              <PatternSvg {...res} show={show} />
            </>
          )}
        </>
      )}
    </div>
  );
}
