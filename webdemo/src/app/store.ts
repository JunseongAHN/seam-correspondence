import { configureStore } from "@reduxjs/toolkit";
import { evaluationReducer } from "@/features/evaluation/state";
import { stitchViewReducer } from "@/features/stitch-view/state";
import workspace from "./workspaceSlice";

/** A fresh store; tests make their own so no state leaks between them. */
export const makeStore = () => configureStore({
  reducer: { workspace, stitchView: stitchViewReducer, evaluation: evaluationReducer },
});

export const store = makeStore();

export type AppStore = ReturnType<typeof makeStore>;
export type RootState = ReturnType<AppStore["getState"]>;
export type AppDispatch = AppStore["dispatch"];
