import { useAppDispatch, useAppSelector } from "@/app/hooks";
import shared from "@/styles/shared.module.css";
import { labelsSet } from "../state";

/** Show "index · arc length" on every edge. */
export default function LabelsToggle() {
  const labels = useAppSelector((s) => s.stitchView.labels);
  const dispatch = useAppDispatch();
  return (
          <label className={shared.note} style={{ userSelect: "none" }}>
            <input type="checkbox" checked={labels}
                   onChange={(e) => dispatch(labelsSet(e.target.checked))} />{" "}
            edge index and length
          </label>
  );
}
