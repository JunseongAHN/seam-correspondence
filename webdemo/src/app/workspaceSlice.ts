/* State more than one feature reads or changes.  Filled in step by step; empty for now. */
import { createSlice } from "@reduxjs/toolkit";

export type WorkspaceState = Record<string, never>;

const initialState: WorkspaceState = {};

const workspaceSlice = createSlice({
  name: "workspace",
  initialState,
  reducers: {},
});

export default workspaceSlice.reducer;
