import { AnnotateToggle, AnnotationTools } from "@/features/annotation";
import { ReportPanel, ReportToggle, ResultBar } from "@/features/evaluation";
import { DropZone, ExampleNote, MoreGarments } from "@/features/input";
import { RightPane } from "@/features/preview3d";
import { FocusHint, LabelsToggle, LineToggles, PatternSvg } from "@/features/stitch-view";
import { useWorkspace } from "./useWorkspace";
import "./App.css";

export default function App() {
  const w = useWorkspace();
  const { res } = w;

  return (
    <div className="app">
      <header>
        <h1>AutoSew — sewing pattern → stitch prediction</h1>
        <p className="sub">
          A 2D sewing pattern in, the stitching between panel edges out, from an ONNX
          model in your browser. A DXF has no stitch list — draw the ground truth to
          score the prediction, and to get a failure report from{" "}
          <strong>claude-sonnet-5</strong>.
        </p>
        {/* The one limit a reader has to know before judging anything on this page. */}
        <p className="limit">
          Currently only one-to-one correspondence is available — the training data
          contains no multi-edge stitches, so gathers cannot be expressed.
        </p>
      </header>

      <DropZone busy={w.busy} example={w.example} onFile={w.runFile} onExample={w.runExample} />

      {(w.example === "gcd" || w.more !== null) && (
        <MoreGarments busy={w.busy} name={w.name} onPick={w.runMore} />
      )}

      {w.err && <div className="err">{w.err}</div>}

      {res && (
        <div className="examples">
          <AnnotateToggle hasOwnGt={w.hasOwnGt} annotate={w.annotate} onToggle={w.toggleAnnotate} />
          <ReportToggle scored={w.scored} showReport={w.showReport} onToggle={w.toggleReport} />
          <LabelsToggle labels={w.labels} onChange={w.setLabels} />
          {w.annotate && (
            <AnnotationTools res={res} name={w.name} userPairs={w.userPairs}
                             onClear={w.clearPairs} onLoad={w.loadPairs} onError={w.setErr} />
          )}
        </div>
      )}

      {res && w.scored && w.showReport && <ReportPanel res={res} truth={w.truth} name={w.name} />}

      {res && (
        <>
          <ResultBar name={w.name} panels={res.pattern.panels.length} edges={res.M}
                     predicted={res.pred.size} ms={res.ms}
                     stats={w.scored ? w.stats : null} handDrawn={w.userPairs.size} />

          {w.example && <ExampleNote example={w.example} />}

          {w.scored && <LineToggles show={w.show} onChange={w.setShow} />}

          {w.right ? (
            <div className="split">
              <section>
                <h2>predicted stitching · 2D</h2>
                <PatternSvg {...res} gt={w.truth} show={w.show} height="62vh"
                              annotate={w.annotate} labels={w.labels} userPairs={w.userPairs}
                              selected={w.selected} onEdgeClick={w.onEdgeClick} />
              </section>
              <section>
                <RightPane pane={w.right} />
              </section>
            </div>
          ) : (
            <>
              {!w.scored && <FocusHint />}
              <PatternSvg {...res} gt={w.truth} show={w.show}
                    annotate={w.annotate} labels={w.labels} userPairs={w.userPairs}
                    selected={w.selected} onEdgeClick={w.onEdgeClick} />
            </>
          )}
        </>
      )}
    </div>
  );
}
