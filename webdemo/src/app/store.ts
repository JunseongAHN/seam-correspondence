import { configureStore } from "@reduxjs/toolkit";
import workspace from "./workspaceSlice";

/** A fresh store; tests make their own so no state leaks between them. */
export const makeStore = () => configureStore({
  reducer: { workspace },
});

export const store = makeStore();

export type AppStore = ReturnType<typeof makeStore>;
export type RootState = ReturnType<AppStore["getState"]>;
export type AppDispatch = AppStore["dispatch"];
