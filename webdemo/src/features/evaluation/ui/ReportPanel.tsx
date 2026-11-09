import { getResult } from "@/adapters/resultCache";
import { useAppSelector } from "@/app/hooks";
import { selectPred, selectScored, selectTruth } from "@/app/workspaceSlice";
import { buildEval, type EvalSource } from "../logic";
import shared from "@/styles/shared.module.css";
import LlmReport from "./LlmReport";

function downloadEval(res: EvalSource, truth: Set<string>, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(buildEval(res, truth, name), null, 2)],
                                        { type: "application/json" }));
  a.download = `${name.replace(/\.[^.]+$/, "")}_eval.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** The LLM failure report, and the evaluation file it is written from -- once the report is
    open and there is a ground truth to measure the result against. */
export default function ReportPanel() {
  const meta = useAppSelector((s) => s.workspace.result);
  const name = useAppSelector((s) => s.workspace.name);
  const scored = useAppSelector(selectScored);
  const showReport = useAppSelector((s) => s.evaluation.showReport);
  const pred = useAppSelector(selectPred);
  const truth = useAppSelector(selectTruth);
  // the feature tensor and the model's scores are in the result cache, not the store
  const objects = getResult(meta?.id);
  if (!meta || !objects || !scored || !showReport) return null;
  const res: EvalSource = { keys: objects.keys, x: objects.x, pred, M: meta.edges, scores: objects.scores };
  return (
        <>
          <LlmReport name={name} doc={buildEval(res, truth, name)} />
          <div className={shared.examples}>
            <button onClick={() => downloadEval(res, truth, name)}>
              save the evaluation as JSON
            </button>
            <span className={shared.note}>the input the report is written from</span>
          </div>
        </>
  );
}
