/* Everything the page shares between its features: the loaded pattern and what the model
   made of it, the ground truth it is scored against, and the view switches.  Features
   get what they need as props; none of them holds this state itself.

   The load itself now lives in the store (app/workspaceSlice); what is left here moves
   there feature by feature. */
import { useCallback, useMemo } from "react";
import type { Pattern } from "@/core-algorithm/pattern";
import type { Scores } from "@/core-algorithm/inference";
import { toPairList, toPairSet } from "@/adapters/pairs";
import { getResult } from "@/adapters/resultCache";
import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { annotateToggled, errorSet, openPattern, pairsCleared, pairsLoaded,
         selectScored, selectStats, selectTruth, selectUserPairs } from "@/app/workspaceSlice";
import { EXAMPLES, exampleGtUrl, MORE, moreSpecFile, moreSpecUrl,
         type ExampleKey } from "@/features/input";
import { pickRightPane } from "@/features/preview3d";
import { placePanels, type Placed } from "@/features/stitch-view";

export type Result = {
  pattern: Pattern; placed: Placed[]; keys: Array<[string, number]>;
  pred: Set<string>; gt: Set<string>; ms: number; M: number;
  x: Float32Array;              // the features as given to the model, for the eval export
  scores: Scores;               // and what it made of them, for the same reason
};

export function useWorkspace() {
  const dispatch = useAppDispatch();
  const { result: meta, busy, modelStatus, error: err, name, example, more, pane, annotate } =
    useAppSelector((s) => s.workspace);
  const userPairs = useAppSelector(selectUserPairs);
  const truth = useAppSelector(selectTruth);
  const scored = useAppSelector(selectScored);
  const stats = useAppSelector(selectStats);

  /* The result as the components still take it: the store's small part, the cache's big
     part, and the layout drawn from the pattern.  Each piece is memoised on what it is
     made from, so a component sees a new object only when a new result arrives. */
  const objects = getResult(meta?.id);
  const placed = useMemo(() => (objects ? placePanels(objects.pattern) : null), [objects]);
  const pred = useMemo(() => toPairSet(meta?.pred ?? []), [meta?.pred]);
  const gt = useMemo(() => toPairSet(meta?.gt ?? []), [meta?.gt]);
  const res: Result | null = useMemo(() => (meta && objects && placed
    ? { pattern: objects.pattern, placed, keys: objects.keys, pred, gt, ms: meta.ms,
        M: meta.edges, x: objects.x, scores: objects.scores }
    : null), [meta, objects, placed, pred, gt]);


  const runFile = useCallback((file: File) => {
    dispatch(openPattern({ source: "file", file }));
  }, [dispatch]);

  const runExample = useCallback((key: ExampleKey) => {
    const e = EXAMPLES[key];
    dispatch(openPattern({ source: "url", url: e.url, fileName: e.file, gtUrl: exampleGtUrl(key),
                           example: key, more: null, exampleRight: e.right, moreSim: null }));
  }, [dispatch]);

  const runMore = useCallback((id: string) => {
    dispatch(openPattern({ source: "url", url: moreSpecUrl(id), fileName: moreSpecFile(id),
                           gtUrl: null, example: null, more: id, exampleRight: null,
                           moreSim: MORE.find((g) => g.id === id)?.sim ?? null }));
  }, [dispatch]);

  const setErr = useCallback((m: string | null) => { dispatch(errorSet(m)); }, [dispatch]);

  const exact = !!stats && stats.fp === 0 && stats.fn === 0;
  /* The input brought its own stitch list -- a specification, or the example whose
     hand-drawn truth is checked in. Nothing to draw. */
  const hasOwnGt = !!res && res.gt.size > 0;

  const right = pickRightPane({
    example: example ? pane.exampleRight : null,
    more,
    moreSim: more ? pane.moreSim ?? undefined : undefined,
    exact,
    stats,
  });

  return {
    res, busy, loadingModel: modelStatus === "loading", err, name, example, more, truth, scored,
    stats, hasOwnGt, right,
    annotate, userPairs,
    setErr,
    toggleAnnotate: () => { dispatch(annotateToggled()); },
    clearPairs: () => { dispatch(pairsCleared()); },
    loadPairs: (p: Set<string>) => { dispatch(pairsLoaded(toPairList(p))); },
    runFile, runExample, runMore,
  };
}
