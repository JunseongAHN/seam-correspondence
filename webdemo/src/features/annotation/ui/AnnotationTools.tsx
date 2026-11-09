import { useRef } from "react";
import { toPairList } from "@/adapters/pairs";
import { getResult } from "@/adapters/resultCache";
import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { errorSet, pairsCleared, pairsLoaded, selectUserPairs } from "@/app/workspaceSlice";
import { gtFromText, gtToDoc, type GtSource } from "../logic";
import shared from "@/styles/shared.module.css";

/** Hand-drawn ground truth, saved as panel/edge names so it survives a reparse. */
function downloadGt(res: GtSource, pairs: Set<string>, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(gtToDoc(res, pairs, name), null, 2)],
                                        { type: "application/json" }));
  a.download = `${name.replace(/\.[^.]+$/, "")}_gt.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Clear, save and load the hand-drawn stitches, shown while drawing. */
export default function AnnotationTools() {
  const gtRef = useRef<HTMLInputElement>(null);
  const annotate = useAppSelector((s) => s.workspace.annotate);
  const meta = useAppSelector((s) => s.workspace.result);
  const name = useAppSelector((s) => s.workspace.name);
  const userPairs = useAppSelector(selectUserPairs);
  const dispatch = useAppDispatch();
  // the pattern and its node keys are in the result cache, not the store
  const objects = getResult(meta?.id);
  if (!annotate || !meta || !objects) return null;
  const res: GtSource = { pattern: objects.pattern, M: meta.edges, keys: objects.keys };
  return (
            <>
              <button onClick={() => dispatch(pairsCleared())}
                      disabled={!userPairs.size}>clear ({userPairs.size})</button>
              <button disabled={!userPairs.size}
                      onClick={() => downloadGt(res, userPairs, name)}>save as JSON</button>
              <button onClick={() => gtRef.current?.click()}>load JSON</button>
              <input ref={gtRef} type="file" accept=".json" hidden
                     onChange={async (e) => {
                       const f = e.target.files?.[0];
                       if (!f) return;
                       try { dispatch(pairsLoaded(toPairList(gtFromText(await f.text(), res.keys)))); }
                       catch (ex: any) { dispatch(errorSet(String(ex?.message ?? ex))); }
                     }} />
              <span className={shared.note}>
                click one edge, then the edge it is sewn to. Click a pair again to undo it.
              </span>
            </>
  );
}
