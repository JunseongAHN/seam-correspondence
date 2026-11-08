import { buildEval, type EvalSource } from "../logic";
import LlmReport from "./LlmReport";

function downloadEval(res: EvalSource, truth: Set<string>, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(buildEval(res, truth, name), null, 2)],
                                        { type: "application/json" }));
  a.download = `${name.replace(/\.[^.]+$/, "")}_eval.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** The LLM failure report, and the evaluation file it is written from. */
export default function ReportPanel({ res, truth, name }: {
  res: EvalSource;
  truth: Set<string>;
  name: string;
}) {
  return (
        <>
          <LlmReport name={name} doc={buildEval(res, truth, name)} />
          <div className="examples">
            <button onClick={() => downloadEval(res, truth, name)}>
              save the evaluation as JSON
            </button>
            <span className="note">the input the report is written from</span>
          </div>
        </>
  );
}
