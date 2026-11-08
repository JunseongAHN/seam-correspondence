import type { RightPane as Pane } from "../logic";
import shared from "@/styles/shared.module.css";
import GtDrape from "./GtDrape";
import SimViewer from "./SimViewer";
import WeldGT from "./WeldGT";

/** The 3D pane beside the pattern, with its heading. */
export default function RightPane({ pane }: { pane: Pane }) {
  return (
    <>
      <h2 className={`${shared.paneTitle}${pane.gt ? ` ${shared.gt}` : ""}`}>{pane.title}</h2>
      {pane.kind === "sim" ? <SimViewer {...pane.props} />
       : pane.kind === "weld" ? <WeldGT />
       : <GtDrape {...pane.props} />}
    </>
  );
}
