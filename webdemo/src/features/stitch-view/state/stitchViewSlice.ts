/* What the 2D view shows: edge labels, and which kinds of stitch line.  Nothing resets
   these: they survive loading another pattern, as these switches always have. */
import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

export type Show = { correct: boolean; fp: boolean; fn: boolean; gt: boolean };
export type StitchViewState = { labels: boolean; show: Show };

const initialState: StitchViewState = {
  labels: true,
  show: { correct: true, fp: true, fn: true, gt: false },
};

const stitchViewSlice = createSlice({
  name: "stitchView",
  initialState,
  reducers: {
    labelsSet(state, action: PayloadAction<boolean>) { state.labels = action.payload; },
    showSet(state, action: PayloadAction<Show>) { state.show = action.payload; },
  },
});

export const { labelsSet, showSet } = stitchViewSlice.actions;
export default stitchViewSlice.reducer;
