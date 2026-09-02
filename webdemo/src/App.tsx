import { useCallback, useMemo, useRef, useState } from "react";
import { parseSpec, type Pattern } from "./lib/parseSpec";
import { patternToTensors, gtPairs } from "./lib/features";
import { predict } from "./lib/infer";
import { placePanels, edgePolyline, edgeMid, stitchColor, type Placed } from "./lib/render";
import "./App.css";

type Result = {
  pattern: Pattern; placed: Placed[]; keys: Array<[string, number]>;
  pred: Set<string>; gt: Set<string>; ms: number; M: number;
};

export default function App() {
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [show, setShow] = useState({ correct: true, fp: true, fn: true, gt: false });
  const [name, setName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (file: File) => {
    setBusy(true); setErr(null);
    try {
      const spec = JSON.parse(await file.text());
      const pattern = parseSpec(spec, file.name.replace("_specification.json", ""));
      const t = patternToTensors(pattern);
      const { pairs, ms } = await predict(t);
      setRes({ pattern, placed: placePanels(pattern), keys: t.keys,
               pred: pairs, gt: gtPairs(pattern, t.keys), ms, M: t.M });
      setName(file.name);
    } catch (e: any) { setErr(String(e?.message ?? e)); setRes(null); }
    finally { setBusy(false); }
  }, []);

  const stats = useMemo(() => {
    if (!res) return null;
    const ok = [...res.pred].filter((k) => res.gt.has(k)).length;
    const fp = res.pred.size - ok, fn = res.gt.size - ok;
    const P = res.pred.size ? ok / res.pred.size : 0;
    const R = res.gt.size ? ok / res.gt.size : 0;
    return { ok, fp, fn, P, R, f1: P + R ? (2 * P * R) / (P + R) : 0 };
  }, [res]);

  return (
    <div className="app">
      <header>
        <h1>AutoSew — sewing pattern → stitch prediction</h1>
        <p className="sub">
          Drop a GarmentCodeData <code>*_specification.json</code>. The panel geometry is turned into
          24 features per edge and run through the exported ONNX model in your browser.
        </p>
      </header>

      <div className="drop"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) run(f); }}
        onClick={() => fileRef.current?.click()}>
        <input ref={fileRef} type="file" accept=".json" hidden
               onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); }} />
        {busy ? "running…" : "drop a specification.json here, or click to choose"}
      </div>

      {err && <div className="err">{err}</div>}

      {res && stats && (
        <>
          <div className="bar">
            <span className="pill name">{name}</span>
            <span className="pill">{res.pattern.panels.length} panels</span>
            <span className="pill">{res.M} edges</span>
            <span className="pill ok">correct {stats.ok}</span>
            <span className="pill fp">FP {stats.fp}</span>
            <span className="pill fn">FN {stats.fn}</span>
            <span className="pill">F1 {stats.f1.toFixed(3)}</span>
            <span className="pill t">{res.ms.toFixed(0)} ms</span>
          </div>
          <div className="toggles">
            {(["correct", "fp", "fn", "gt"] as const).map((k) => (
              <label key={k}>
                <input type="checkbox" checked={show[k]}
                       onChange={(e) => setShow({ ...show, [k]: e.target.checked })} />
                {k === "gt" ? "ground-truth stitches" : k}
              </label>
            ))}
          </div>
          <Svg res={res} show={show} />
        </>
      )}
    </div>
  );
}

function Svg({ res, show }: { res: Result; show: Record<string, boolean> }) {
  const { placed, keys, pred, gt, pattern } = res;
  const byName = useMemo(() => new Map(placed.map((p) => [p.name, p])), [placed]);
  const node2key = keys;

  const pts = placed.flatMap((p) => p.verts);
  const xs = pts.map((v) => v[0]), ys = pts.map((v) => v[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pad = 12;
  const W = x1 - x0 + 2 * pad, H = y1 - y0 + 2 * pad;
  const tx = (p: [number, number]) => `${p[0] - x0 + pad},${y1 - p[1] + pad}`;   // flip y

  const line = (a: number, b: number, cls: string, key: string) => {
    const ka = node2key[a], kb = node2key[b];
    const pa = byName.get(ka[0]), pb = byName.get(kb[0]);
    if (!pa || !pb) return null;
    if (!pa.edges[ka[1]] || !pb.edges[kb[1]]) return null;
    const m1 = edgeMid(pa, ka[1]), m2 = edgeMid(pb, kb[1]);
    return <line key={key} className={cls} x1={m1[0] - x0 + pad} y1={y1 - m1[1] + pad}
                 x2={m2[0] - x0 + pad} y2={y1 - m2[1] + pad} />;
  };

  const stitchOf = new Map<string, number>();
  pattern.stitches.forEach((sides, si) => sides.forEach(([pn, ei]) => stitchOf.set(`${pn}#${ei}`, si)));

  return (
    <svg className="canvas" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
      {placed.map((p) => (
        <g key={p.name}>
          {p.edges.map((_: any, i: number) => {
            const si = stitchOf.get(`${p.name}#${i}`);
            const col = si === undefined ? "#c9c9c9" : stitchColor(si);
            const d = edgePolyline(p, i).map(tx).join(" L ");
            return <path key={i} d={`M ${d}`} stroke={col}
                         strokeWidth={si === undefined ? 0.5 : 1.1} fill="none" strokeLinecap="round" />;
          })}
          <text x={p.verts.reduce((s, v) => s + v[0], 0) / p.verts.length - x0 + pad}
                y={y1 - p.verts.reduce((s, v) => s + v[1], 0) / p.verts.length + pad}
                className="plabel">{p.name}</text>
        </g>
      ))}
      {show.gt && [...gt].map((k) => {
        const [a, b] = k.split("-").map(Number); return line(a, b, "gtline", `g${k}`);
      })}
      {show.correct && [...pred].filter((k) => gt.has(k)).map((k) => {
        const [a, b] = k.split("-").map(Number); return line(a, b, "ok", `c${k}`);
      })}
      {show.fp && [...pred].filter((k) => !gt.has(k)).map((k) => {
        const [a, b] = k.split("-").map(Number); return line(a, b, "fp", `p${k}`);
      })}
      {show.fn && [...gt].filter((k) => !pred.has(k)).map((k) => {
        const [a, b] = k.split("-").map(Number); return line(a, b, "fn", `n${k}`);
      })}
    </svg>
  );
}
