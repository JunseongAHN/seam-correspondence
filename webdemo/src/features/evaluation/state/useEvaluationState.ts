/* Whether the LLM report is open.  The toggle sits in the toolbar and the report below it,
   so App calls this and passes it down.  It stays open across loading another pattern, as
   it always has; the report itself only shows when there is something to score. */
import { useState } from "react";

export function useEvaluationState() {
  const [showReport, setShowReport] = useState(false);
  return { showReport, toggleReport: () => setShowReport((v) => !v) };
}
