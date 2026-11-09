/* Whether the LLM report is open.  Nothing resets it: it stays open across loading another
   pattern, and shows again as soon as the new one can be scored. */
import { createSlice } from "@reduxjs/toolkit";

export type EvaluationState = { showReport: boolean };

const initialState: EvaluationState = { showReport: false };

const evaluationSlice = createSlice({
  name: "evaluation",
  initialState,
  reducers: {
    reportToggled(state) { state.showReport = !state.showReport; },
  },
});

export const { reportToggled } = evaluationSlice.actions;
export default evaluationSlice.reducer;
