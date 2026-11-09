import { AnnotateToggle, AnnotationTools } from "@/features/annotation";
import { ReportPanel, ReportToggle, ResultBar } from "@/features/evaluation";
import { DropZone, ExampleNote, MoreGarments } from "@/features/input";
import { RightPane } from "@/features/preview3d";
import { FocusHint, LabelsToggle, LineToggles, PatternSvg } from "@/features/stitch-view";
import { useWorkspace } from "./useWorkspace";
import shared from "@/styles/shared.module.css";
import s from "./App.module.css";

export default function App() {
  const w = useWorkspace();
  const { res } = w;

  return (
    <div className={s.app}>
      <header>
        <h1>AutoSew — sewing pattern → stitch prediction</h1>
        <p className={s.sub}>
          A 2D sewing pattern in, the stitching between panel edges out, from an ONNX
          model in your browser. A DXF has no stitch list — draw the ground truth to
          score the prediction, and to get a failure report from{" "}
          <strong>claude-sonnet-5</strong>.
        </p>
        {/* The one limit a reader has to know before judging anything on this page. */}
        <p className={s.limit}>
          Currently only one-to-one correspondence is available — the training data
          contains no multi-edge stitches, so gathers cannot be expressed.
        </p>
      </header>

      <DropZone busy={w.busy} loadingModel={w.loadingModel} example={w.example} onFile={w.runFile} onExample={w.runExample} />

      {(w.example === "gcd" || w.more !== null) && (
        <MoreGarments busy={w.busy} name={w.name} onPick={w.runMore} />
      )}

      {w.err && <div className={shared.err}>{w.err}</div>}

      {res && (
        <div className={shared.examples}>
          <AnnotateToggle hasOwnGt={w.hasOwnGt} annotate={w.annotate} onToggle={w.toggleAnnotate} />
          <ReportToggle />
          <LabelsToggle />
          {w.annotate && (
            <AnnotationTools res={res} name={w.name} userPairs={w.userPairs}
                             onClear={w.clearPairs} onLoad={w.loadPairs} onError={w.setErr} />
          )}
        </div>
      )}

      <ReportPanel />

      {res && (
        <>
          <ResultBar />

          {w.example && <ExampleNote example={w.example} />}

          <LineToggles />

          {w.right ? (
            <div className={s.split}>
              <section>
                <h2 className={shared.paneTitle}>predicted stitching · 2D</h2>
                <PatternSvg height="62vh" />
              </section>
              <section>
                <RightPane pane={w.right} />
              </section>
            </div>
          ) : (
            <>
              <FocusHint />
              <PatternSvg />
            </>
          )}
        </>
      )}
    </div>
  );
}
