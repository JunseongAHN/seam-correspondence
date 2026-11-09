/* Everything the page shares between its features: the loaded pattern and what the model
   made of it, the ground truth it is scored against, and the view switches.  Features
   get what they need as props; none of them holds this state itself. */
import { useCallback, useState } from "react";
import type { Pattern } from "@/core-algorithm/pattern";
import { gtPairs, modelLoaded, patternToTensors, predict, type Scores } from "@/core-algorithm/inference";
import { annotationClick, gtFromDoc } from "@/features/annotation";
import { scoreStats } from "@/features/evaluation";
import { EXAMPLES, exampleGtUrl, MORE, moreSpecFile, moreSpecUrl, parseInput,
         type ExampleKey } from "@/features/input";
import { pickRightPane } from "@/features/preview3d";
import { placePanels, type Placed } from "@/features/stitch-view";

export type Result = {
  pattern: Pattern; placed: Placed[]; keys: Array<[string, number]>;
  pred: Set<string>; gt: Set<string>; ms: number; M: number;
  source: "spec" | "dxf";
  x: Float32Array;              // the features as given to the model, for the eval export
  scores: Scores;               // and what it made of them, for the same reason
};

export function useWorkspace() {
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  // the first prediction also fetches the model, which takes far longer than running it
  const [loadingModel, setLoadingModel] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [example, setExample] = useState<ExampleKey | null>(null);
  const [more, setMore] = useState<string | null>(null);   // which held-out garment
  const [showReport, setShowReport] = useState(false);
  const [annotate, setAnnotate] = useState(false);
  const [userPairs, setUserPairs] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<number | null>(null);

  /** Click one edge then another to sew them together; click a paired edge to undo it.

      Two things to keep right here.  StrictMode invokes a state updater twice to check it
      is pure, so the updater must not have side effects -- setUserPairs cannot live
      inside setSelected -- and it must be idempotent, which a has/delete/add toggle is
      not: run twice it adds and then removes, and the pair silently vanishes.  So the
      add-or-remove decision is made out here and the updater just applies it. */
  const onEdgeClick = useCallback((n: number) => {
    const { selected: next, change } = annotationClick(selected, userPairs, n);
    if (change) {
      setUserPairs((p) => {
        const q = new Set(p);
        if (change.remove) q.delete(change.key); else q.add(change.key);
        return q;
      });
    }
    setSelected(next);
  }, [selected, userPairs]);

  /** One pipeline for both inputs: text -> Pattern -> features -> ONNX -> pairs.
      A DXF carries no stitches, so there is nothing to score it against. */
  const runText = useCallback(async (text: string, fileName: string, ex: ExampleKey | null,
                                     moreId: string | null = null,
                                     gtUrl: string | null = null) => {
    setBusy(true); setErr(null);
    try {
      const { pattern, isDxf } = parseInput(text, fileName);
      const t = patternToTensors(pattern);
      setLoadingModel(!modelLoaded());
      const { pairs, ms, scores } = await predict(t);
      setLoadingModel(false);
      let gt = isDxf ? new Set<string>() : gtPairs(pattern, t.keys);
      if (gtUrl) {
        const g = await fetch(gtUrl);
        if (!g.ok) throw new Error(`${gtUrl}: HTTP ${g.status}`);
        gt = gtFromDoc(await g.json(), t.keys);
      }
      setRes({ pattern, placed: placePanels(pattern), keys: t.keys, pred: pairs,
               gt, ms, M: t.M, source: isDxf ? "dxf" : "spec", x: t.x, scores });
      setName(fileName);
      setExample(ex);
      setMore(moreId);
    } catch (e: any) {
      setErr(String(e?.message ?? e)); setRes(null); setExample(null); setMore(null);
    }
    finally { setBusy(false); setLoadingModel(false); }
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
      await runText(await r.text(), e.file, key, null, exampleGtUrl(key));
    } catch (ex: any) { setErr(String(ex?.message ?? ex)); setBusy(false); }
  }, [runText]);

  const runMore = useCallback(async (id: string) => {
    setBusy(true); setErr(null);
    try {
      const u = moreSpecUrl(id);
      const r = await fetch(u);
      if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
      await runText(await r.text(), moreSpecFile(id), null, id);
    } catch (e: any) { setErr(String(e?.message ?? e)); setBusy(false); }
  }, [runText]);

  // hand-drawn ground truth outranks the file's, so a DXF becomes scorable
  const truth = userPairs.size ? userPairs : res?.gt ?? new Set<string>();
  const scored = !!res && truth.size > 0;
  const stats = res ? scoreStats(res.pred, truth) : null;
  const exact = !!stats && stats.fp === 0 && stats.fn === 0;
  /* The input brought its own stitch list -- a specification, or the example whose
     hand-drawn truth is checked in. Nothing to draw. */
  const hasOwnGt = !!res && res.gt.size > 0;

  const right = pickRightPane({
    example: example ? EXAMPLES[example].right : null,
    more,
    moreSim: MORE.find((g) => g.id === more)?.sim,
    exact,
    stats,
  });

  return {
    res, busy, loadingModel, err, name, example, more, truth, scored, stats, exact, hasOwnGt, right,
    showReport, annotate, userPairs, selected,
    setErr, setUserPairs,
    toggleReport: () => setShowReport((v) => !v),
    toggleAnnotate: () => { setAnnotate((v) => !v); setSelected(null); },
    clearPairs: () => { setUserPairs(new Set()); setSelected(null); },
    loadPairs: (p: Set<string>) => { setUserPairs(p); setErr(null); },
    onEdgeClick, runFile, runExample, runMore,
  };
}
