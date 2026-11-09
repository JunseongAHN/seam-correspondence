import { useAppSelector } from "@/app/hooks";
import { selectScored } from "@/app/workspaceSlice";
import shared from "@/styles/shared.module.css";

/** How to read an unscored pattern; a scored one explains itself with its line toggles. */
export default function FocusHint() {
  const loaded = useAppSelector((s) => s.workspace.result !== null);
  const scored = useAppSelector(selectScored);
  if (!loaded || scored) return null;
  return (
                <p className={`${shared.note} ${shared.wide}`}>
                  Edges of a predicted pair share a colour. Click an edge to focus its
                  correspondence — it is drawn thick and every other pair drops to grey.
                  Click again, or click the background, to clear.
                </p>
  );
}
