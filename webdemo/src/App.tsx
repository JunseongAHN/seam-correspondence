import { AnnotateToggle, AnnotationTools } from "@/features/annotation";
import { ReportPanel, ReportToggle, ResultBar, useEvaluationState } from "@/features/evaluation";
import { DropZone, ExampleNote, MoreGarments } from "@/features/input";
import { RightPane } from "@/features/preview3d";
import { FocusHint, LabelsToggle, LineToggles, PatternSvg,
         useStitchViewState } from "@/features/stitch-view";
import { useWorkspace } from "./useWorkspace";
import shared from "@/styles/shared.module.css";
import s from "./App.module.css";

export default function App() {
  const w = useWorkspace();
  const { res } = w;
  const view = useStitchViewState();
  const evaluation = useEvaluationState();

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
          <ReportToggle scored={w.scored} showReport={evaluation.showReport} onToggle={evaluation.toggleReport} />
          <LabelsToggle labels={view.labels} onChange={view.setLabels} />
          {w.annotate && (
            <AnnotationTools res={res} name={w.name} userPairs={w.userPairs}
                             onClear={w.clearPairs} onLoad={w.loadPairs} onError={w.setErr} />
          )}
        </div>
      )}

      {res && w.scored && evaluation.showReport && <ReportPanel res={res} truth={w.truth} name={w.name} />}

      {res && (
        <>
          <ResultBar name={w.name} panels={res.pattern.panels.length} edges={res.M}
                     predicted={res.pred.size} ms={res.ms}
                     stats={w.scored ? w.stats : null} handDrawn={w.userPairs.size} />

          {w.example && <ExampleNote example={w.example} />}

          {w.scored && <LineToggles show={view.show} onChange={view.setShow} />}

          {w.right ? (
            <div className={s.split}>
              <section>
                <h2 className={shared.paneTitle}>predicted stitching · 2D</h2>
                <PatternSvg {...res} gt={w.truth} show={view.show} height="62vh"
                              annotate={w.annotate} labels={view.labels} userPairs={w.userPairs}
                              selected={w.selected} onEdgeClick={w.onEdgeClick} />
              </section>
              <section>
                <RightPane pane={w.right} />
              </section>
            </div>
          ) : (
            <>
              {!w.scored && <FocusHint />}
              <PatternSvg {...res} gt={w.truth} show={view.show}
                    annotate={w.annotate} labels={view.labels} userPairs={w.userPairs}
                    selected={w.selected} onEdgeClick={w.onEdgeClick} />
            </>
          )}
        </>
      )}
    </div>
  );
}
