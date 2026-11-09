import { useAppSelector } from "@/app/hooks";
import shared from "@/styles/shared.module.css";
import { selectRightPane } from "../state";
import GtDrape from "./GtDrape";
import SimViewer from "./SimViewer";
import WeldGT from "./WeldGT";

/** The 3D pane beside the pattern, with its heading; nothing for an input that has none. */
export default function RightPane() {
  const pane = useAppSelector(selectRightPane);
  if (!pane) return null;
  return (
    <>
      <h2 className={`${shared.paneTitle}${pane.gt ? ` ${shared.gt}` : ""}`}>{pane.title}</h2>
      {pane.kind === "sim" ? <SimViewer {...pane.props} />
       : pane.kind === "weld" ? <WeldGT />
       : <GtDrape {...pane.props} />}
    </>
  );
}
