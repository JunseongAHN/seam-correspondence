import { configureStore } from "@reduxjs/toolkit";
import workspace from "./workspaceSlice";

export const store = configureStore({
  reducer: { workspace },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
