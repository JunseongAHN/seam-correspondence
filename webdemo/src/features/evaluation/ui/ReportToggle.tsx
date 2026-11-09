import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { selectScored } from "@/app/workspaceSlice";
import shared from "@/styles/shared.module.css";
import { reportToggled } from "../state";

/* Shown even with nothing to score, because its absence would read as the
   feature not existing rather than as a step not yet taken. The reason rides
   in the label: a disabled button beside an explanation is two things to
   read, and the explanation is short enough to be the label. */
export default function ReportToggle() {
  const scored = useAppSelector(selectScored);
  const showReport = useAppSelector((s) => s.evaluation.showReport);
  const dispatch = useAppDispatch();
  return (
          <button className={showReport ? shared.on : ""} disabled={!scored}
                  title={scored ? undefined
                                : "A report is written from misses and false positives, "
                                  + "which need a ground truth to be measured against."}
                  onClick={() => dispatch(reportToggled())}>
            {scored
              ? `${showReport ? "hide" : "show"} LLM evaluation report`
              : "LLM evaluation report — ⚠️ needs a ground truth"}
          </button>
  );
}
