/* Typed useSelector/useDispatch.  Types only from store.ts, so a component importing these
   does not pull the store module in and close an import cycle through the slices. */
import { useDispatch, useSelector } from "react-redux";
import type { AppDispatch, RootState } from "./store";

export const useAppSelector = useSelector.withTypes<RootState>();
export const useAppDispatch = useDispatch.withTypes<AppDispatch>();
