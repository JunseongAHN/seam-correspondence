import type { Stats } from "../logic";

/** One line of what came out: the input's size, the score, and how long it took. */
export default function ResultBar({ name, panels, edges, predicted, ms, stats, handDrawn }: {
  name: string;
  panels: number;
  edges: number;
  predicted: number;
  ms: number;
  /** null when there is no ground truth to score against */
  stats: Stats | null;
  /** how many stitches were drawn by hand; 0 when scoring against the file's own */
  handDrawn: number;
}) {
  return (
          <div className="bar">
            <span className="pill name">{name}</span>
            <span className="pill">{panels} panels</span>
            <span className="pill">{edges} edges</span>
            {stats ? (
              <>
                {handDrawn > 0 && <span className="pill">{handDrawn} hand-drawn stitches</span>}
                <span className="pill ok">correct {stats.ok}</span>
                <span className="pill fp">FP {stats.fp}</span>
                <span className="pill fn">FN {stats.fn}</span>
                <span className="pill">P {stats.P.toFixed(3)}</span>
                <span className="pill">R {stats.R.toFixed(3)}</span>
                <span className="pill">F1 {stats.f1.toFixed(3)}</span>
                {handDrawn > 0 && <span className="pill">scored against your ground truth</span>}
              </>
            ) : (
              <>
                <span className="pill ok">{predicted} predicted stitches</span>
                <span className="pill">no ground truth in a DXF</span>
              </>
            )}
            <span className="pill t">{ms.toFixed(0)} ms</span>
          </div>
  );
}
