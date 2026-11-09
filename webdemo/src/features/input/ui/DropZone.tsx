import { useRef } from "react";
import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { openPattern } from "@/app/workspaceSlice";
import { EXAMPLES, exampleOpenArg, type ExampleKey } from "../logic";
import shared from "@/styles/shared.module.css";
import s from "./DropZone.module.css";

/** Where a pattern comes in: drop or choose a file, or pick one of the examples. */
export default function DropZone() {
  const fileRef = useRef<HTMLInputElement>(null);
  const busy = useAppSelector((st) => st.workspace.busy);
  // the model is still being fetched, the first time a pattern is run
  const loadingModel = useAppSelector((st) => st.workspace.modelStatus === "loading");
  const example = useAppSelector((st) => st.workspace.example);
  const dispatch = useAppDispatch();
  const onFile = (file: File) => { dispatch(openPattern({ source: "file", file })); };
  const onExample = (key: ExampleKey) => { dispatch(openPattern(exampleOpenArg(key))); };
  return (
      <div className={`${shared.drop} ${s.zone}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) onFile(f); }}
        onClick={() => fileRef.current?.click()}>
        <input ref={fileRef} type="file" accept=".dxf,.json" hidden
               onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
        <div className={s.dropline}>
          {busy ? (loadingModel ? "loading the model…" : "running…") : "import a DXF — drop it here, or click to choose"}
        </div>
        {/* The examples live inside the drop zone, so stop their clicks reaching it --
            otherwise picking one also opens the file dialog behind it. */}
        <div className={s.tries} onClick={(e) => e.stopPropagation()}>
          or try{" "}
          {(Object.keys(EXAMPLES) as ExampleKey[]).map((k, i) => (
            <span key={k}>
              {i > 0 && <span className="sep">·</span>}
              <button className={`link ${example === k ? "on" : ""}`}
                      disabled={busy} onClick={() => onExample(k)}>
                {EXAMPLES[k].label}
              </button>
            </span>
          ))}
        </div>
      </div>
  );
}
