import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { selectScored } from "@/app/workspaceSlice";
import shared from "@/styles/shared.module.css";
import { showSet } from "../state";

/** Which kinds of stitch line the scored view draws; nothing to choose when unscored. */
export default function LineToggles() {
  const scored = useAppSelector(selectScored);
  const show = useAppSelector((s) => s.stitchView.show);
  const dispatch = useAppDispatch();
  if (!scored) return null;
  return (
            <div className={shared.toggles}>
              {(["correct", "fp", "fn", "gt"] as const).map((k) => (
                <label key={k}>
                  <input type="checkbox" checked={show[k]}
                         onChange={(e) => dispatch(showSet({ ...show, [k]: e.target.checked }))} />
                  {k === "gt" ? "ground-truth stitches" : k === "correct" ? k : k.toUpperCase()}
                </label>
              ))}
            </div>
  );
}
