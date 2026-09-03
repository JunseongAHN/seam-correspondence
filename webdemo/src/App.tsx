import { useCallback, useRef, useState } from "react";
import { parseSpec, type Pattern } from "./lib/parseSpec";
import { parseDxf } from "./lib/parseDxf";
import { patternToTensors, gtPairs } from "./lib/features";
import { predict } from "./lib/infer";
import { placePanels, type Placed } from "./lib/render";
import PatternSvg from "./PatternSvg";
import WeldGT from "./WeldGT";         // the CLO drape with its weld-derived seams
import SimViewer from "./SimViewer";   // the wasm assembly solve, rendered
import GtDrape from "./GtDrape";       // the ground-truth drape, solved on the host
import "./App.css";

// BASE_URL, not an absolute path: the GitHub Pages build is served from /<repo>/.
const B = import.meta.env.BASE_URL;
/** Hand-drawn ground truth, saved as panel/edge names so it survives a reparse. */
function downloadGt(res: Result, pairs: Set<string>, name: string) {
  const doc = {
    source: name,
    panels: res.pattern.panels.length,
    edges: res.M,
    stitches: [...pairs].map((k) => k.split("-").map(Number).map((n) => res.keys[n])),
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 2)],
                                        { type: "application/json" }));
  a.download = `${name.replace(/\.[^.]+$/, "")}_gt.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function loadGt(text: string, res: Result): Set<string> {
  const doc = JSON.parse(text);
  const idx = new Map(res.keys.map(([p, e], i) => [`${p}#${e}`, i]));
  const out = new Set<string>();
  let miss = 0;
  for (const st of doc.stitches ?? []) {
    const ns = (st as Array<[string, number]>).map(([p, e]) => idx.get(`${p}#${e}`));
    if (ns.length !== 2 || ns.some((n) => n === undefined)) { miss++; continue; }
    const [a, b] = ns as number[];
    out.add(`${Math.min(a, b)}-${Math.max(a, b)}`);
  }
  if (miss) throw new Error(`${miss} stitch(es) in the file name edges this pattern does not have`);
  return out;
}

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

/* More held-out garments, chosen by scoring a sample of the test split with the model
   this demo ships and taking a spread rather than a highlight reel.  The first one it
   gets completely wrong; two it gets exactly right. */
const MORE: Array<{ id: string; f1: string; what: string }> = [
  { id: "rand_JYO1DHSFGH", f1: "0.00", what: "4 panels, 4 stitches — none found" },
  { id: "rand_I88DFY2AKV", f1: "0.76", what: "8 panels, 24 stitches" },
  { id: "rand_E3IN9RH60H", f1: "0.94", what: "14 panels, 44 stitches" },
  { id: "rand_3C7X2I6WQ7", f1: "1.00", what: "6 panels, 24 stitches — exact" },
  { id: "rand_GE9NBC1HFY", f1: "1.00", what: "14 panels, 48 stitches — exact" },
];

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
  const [more, setMore] = useState<string | null>(null);   // which held-out garment
  const [annotate, setAnnotate] = useState(false);
  const [labels, setLabels] = useState(true);
  const [userPairs, setUserPairs] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const gtRef = useRef<HTMLInputElement>(null);

  /** Click one edge then another to sew them together; click a paired edge to undo it.

      Two things to keep right here.  StrictMode invokes a state updater twice to check it
      is pure, so the updater must not have side effects -- setUserPairs cannot live
      inside setSelected -- and it must be idempotent, which a has/delete/add toggle is
      not: run twice it adds and then removes, and the pair silently vanishes.  So the
      add-or-remove decision is made out here and the updater just applies it. */
  const onEdgeClick = useCallback((n: number) => {
    if (selected === n) { setSelected(null); return; }        // same edge: deselect
    if (selected === null) {
      // already sewn?  clicking it again undoes that stitch
      const owner = [...userPairs].find((k) => k.split("-").map(Number).includes(n));
      if (owner) { setUserPairs((p) => { const q = new Set(p); q.delete(owner); return q; }); return; }
      setSelected(n);
      return;
    }
    const k = `${Math.min(selected, n)}-${Math.max(selected, n)}`;
    const remove = userPairs.has(k);
    setUserPairs((p) => {
      const q = new Set(p);
      if (remove) q.delete(k); else q.add(k);
      return q;
    });
    setSelected(null);
  }, [selected, userPairs]);

  /** One pipeline for both inputs: text -> Pattern -> features -> ONNX -> pairs.
      A DXF carries no stitches, so there is nothing to score it against. */
  const runText = useCallback(async (text: string, fileName: string, ex: ExampleKey | null,
                                     moreId: string | null = null) => {
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
      setMore(moreId);
    } catch (e: any) {
      setErr(String(e?.message ?? e)); setRes(null); setExample(null); setMore(null);
    }
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

  // hand-drawn ground truth outranks the file's, so a DXF becomes scorable
  const truth = userPairs.size ? userPairs : res?.gt ?? new Set<string>();
  const scored = !!res && truth.size > 0;
  const stats = (() => {
    if (!res) return null;
    const ok = [...res.pred].filter((k) => truth.has(k)).length;
    const P = res.pred.size ? ok / res.pred.size : 0;
    const R = truth.size ? ok / truth.size : 0;
    return { ok, fp: res.pred.size - ok, fn: truth.size - ok,
             f1: P + R ? (2 * P * R) / (P + R) : 0, P, R,
             mine: userPairs.size > 0 };
  })();
  const exact = !!stats && stats.fp === 0 && stats.fn === 0;

  /* What sits beside the pattern.  A held-out garment gets its ground-truth drape --
     red, and labelled as ground truth, because it is the only pane on the page that is
     not the model's output and a 3D shape is persuasive enough to be mistaken for one. */
  const right = example
    ? (EXAMPLES[example].right === "sim"
        ? { title: "assembled by the wasm solver · 3D", gt: false,
            node: <SimViewer exact={exact} fp={stats?.fp ?? 0} fn={stats?.fn ?? 0} /> }
        : { title: "CLO's own drape · 3D", gt: false, node: <WeldGT /> })
    : more
    ? { title: exact ? "ground-truth drape · 3D — the prediction is identical"
                     : "ground-truth drape · 3D", gt: true,
        node: <GtDrape id={more} exact={exact}
                       fp={stats?.fp ?? 0} fn={stats?.fn ?? 0} /> }
    : null;

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

      <div className="tryit">
        more held-out garments, from worst to best —{" "}
        {MORE.map((g, i) => (
          <span key={g.id}>
            {i > 0 && <span className="sep">·</span>}
            <button className={`link ${name.startsWith(g.id) ? "on" : ""}`}
                    disabled={busy} title={g.what}
                    onClick={async () => {
                      setBusy(true); setErr(null);
                      try {
                        const u = `${B}example/${g.id}_specification.json`;
                        const r = await fetch(u);
                        if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
                        await runText(await r.text(), `${g.id}_specification.json`, null, g.id);
                      } catch (e: any) { setErr(String(e?.message ?? e)); setBusy(false); }
                    }}>
              F1 {g.f1}
            </button>
          </span>
        ))}
        <span className="note" style={{ marginLeft: 10 }}>
          picked by scoring a sample of the test split with the model this page runs, then
          taking a spread — not a highlight reel. Each opens beside{" "}
          <strong className="gt">its ground-truth drape, in red</strong>.
        </span>
      </div>

      {err && <div className="err">{err}</div>}

      {res && (
        <div className="examples">
          <button onClick={() => { setAnnotate((v) => !v); setSelected(null); }}>
            {annotate ? "leave" : "draw"} the ground truth by hand
          </button>
          <label className="note" style={{ userSelect: "none" }}>
            <input type="checkbox" checked={labels}
                   onChange={(e) => setLabels(e.target.checked)} />{" "}
            edge index and length
          </label>
          {annotate && (
            <>
              <button onClick={() => { setUserPairs(new Set()); setSelected(null); }}
                      disabled={!userPairs.size}>clear ({userPairs.size})</button>
              <button disabled={!userPairs.size}
                      onClick={() => downloadGt(res, userPairs, name)}>save as JSON</button>
              <button onClick={() => gtRef.current?.click()}>load JSON</button>
              <input ref={gtRef} type="file" accept=".json" hidden
                     onChange={async (e) => {
                       const f = e.target.files?.[0];
                       if (!f) return;
                       try { setUserPairs(loadGt(await f.text(), res)); setErr(null); }
                       catch (ex: any) { setErr(String(ex?.message ?? ex)); }
                     }} />
              <span className="note">
                click one edge, then the edge it is sewn to. Click a pair again to undo it.
              </span>
            </>
          )}
        </div>
      )}

      {res && (
        <>
          <div className="bar">
            <span className="pill name">{name}</span>
            <span className="pill">{res.pattern.panels.length} panels</span>
            <span className="pill">{res.M} edges</span>
            {scored && stats ? (
              <>
                {stats.mine && <span className="pill">{userPairs.size} hand-drawn stitches</span>}
                <span className="pill ok">correct {stats.ok}</span>
                <span className="pill fp">FP {stats.fp}</span>
                <span className="pill fn">FN {stats.fn}</span>
                <span className="pill">P {stats.P.toFixed(3)}</span>
                <span className="pill">R {stats.R.toFixed(3)}</span>
                <span className="pill">F1 {stats.f1.toFixed(3)}</span>
                {stats.mine && <span className="pill">scored against your ground truth</span>}
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

          {right ? (
            <div className="split">
              <section>
                <h2>predicted stitching · 2D</h2>
                <PatternSvg {...res} gt={truth} show={show} height="62vh"
                              annotate={annotate} labels={labels} userPairs={userPairs}
                              selected={selected} onEdgeClick={onEdgeClick} />
              </section>
              <section>
                <h2 className={right.gt ? "gt" : undefined}>{right.title}</h2>
                {right.node}
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
              <PatternSvg {...res} gt={truth} show={show}
                    annotate={annotate} labels={labels} userPairs={userPairs}
                    selected={selected} onEdgeClick={onEdgeClick} />
            </>
          )}
        </>
      )}
    </div>
  );
}
