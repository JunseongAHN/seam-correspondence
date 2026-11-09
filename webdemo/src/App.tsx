import { useAppSelector } from "@/app/hooks";
import { AnnotateToggle, AnnotationTools } from "@/features/annotation";
import { ReportPanel, ReportToggle, ResultBar } from "@/features/evaluation";
import { ExampleNote, InputSection } from "@/features/input";
import { RightPane, selectRightPane } from "@/features/preview3d";
import { FocusHint, LabelsToggle, LineToggles, PatternSvg } from "@/features/stitch-view";
import shared from "@/styles/shared.module.css";
import ErrorLine from "./ErrorLine";
import Header from "./Header";
import s from "./App.module.css";

/** Where everything goes.  Each component reads what it shows, and when, from the store;
    App decides only the two things that are layout: whether the toolbar row exists, and
    whether the pattern stands alone or beside a 3D pane. */
export default function App() {
  // an empty toolbar row would still take its margin, so the row itself waits for a result
  const loaded = useAppSelector((st) => st.workspace.result !== null);
  const twoPanes = useAppSelector(selectRightPane) !== null;

  return (
    <div className={s.app}>
      <Header />

      <InputSection />

      <ErrorLine />

      {loaded && (
        <div className={shared.examples}>
          <AnnotateToggle />
          <ReportToggle />
          <LabelsToggle />
          <AnnotationTools />
        </div>
      )}

      <ReportPanel />

      <ResultBar />

      <ExampleNote />

      <LineToggles />

      {twoPanes ? (
        <div className={s.split}>
          <section>
            <h2 className={shared.paneTitle}>predicted stitching · 2D</h2>
            {/* the only prop left: beside a 3D pane the pattern matches its 62vh height */}
            <PatternSvg height="62vh" />
          </section>
          <section>
            <RightPane />
          </section>
        </div>
      ) : (
        <>
          <FocusHint />
          <PatternSvg />
        </>
      )}
    </div>
  );
}
