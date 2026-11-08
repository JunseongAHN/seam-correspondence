import { useRef } from "react";
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
export default function AnnotationTools({ res, name, userPairs, onClear, onLoad, onError }: {
  res: GtSource;
  name: string;
  userPairs: Set<string>;
  onClear: () => void;
  onLoad: (pairs: Set<string>) => void;
  onError: (message: string) => void;
}) {
  const gtRef = useRef<HTMLInputElement>(null);
  return (
            <>
              <button onClick={onClear}
                      disabled={!userPairs.size}>clear ({userPairs.size})</button>
              <button disabled={!userPairs.size}
                      onClick={() => downloadGt(res, userPairs, name)}>save as JSON</button>
              <button onClick={() => gtRef.current?.click()}>load JSON</button>
              <input ref={gtRef} type="file" accept=".json" hidden
                     onChange={async (e) => {
                       const f = e.target.files?.[0];
                       if (!f) return;
                       try { onLoad(gtFromText(await f.text(), res.keys)); }
                       catch (ex: any) { onError(String(ex?.message ?? ex)); }
                     }} />
              <span className={shared.note}>
                click one edge, then the edge it is sewn to. Click a pair again to undo it.
              </span>
            </>
  );
}
