import type { Stats } from "../logic";
import shared from "@/styles/shared.module.css";

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
          <div className={shared.bar}>
            <span className={`${shared.pill} ${shared.name}`}>{name}</span>
            <span className={shared.pill}>{panels} panels</span>
            <span className={shared.pill}>{edges} edges</span>
            {stats ? (
              <>
                {handDrawn > 0 && <span className={shared.pill}>{handDrawn} hand-drawn stitches</span>}
                <span className={`${shared.pill} ${shared.ok}`}>correct {stats.ok}</span>
                <span className={`${shared.pill} ${shared.fp}`}>FP {stats.fp}</span>
                <span className={`${shared.pill} ${shared.fn}`}>FN {stats.fn}</span>
                <span className={shared.pill}>P {stats.P.toFixed(3)}</span>
                <span className={shared.pill}>R {stats.R.toFixed(3)}</span>
                <span className={shared.pill}>F1 {stats.f1.toFixed(3)}</span>
                {handDrawn > 0 && <span className={shared.pill}>scored against your ground truth</span>}
              </>
            ) : (
              <>
                <span className={`${shared.pill} ${shared.ok}`}>{predicted} predicted stitches</span>
                <span className={shared.pill}>no ground truth in a DXF</span>
              </>
            )}
            <span className={`${shared.pill} ${shared.t}`}>{ms.toFixed(0)} ms</span>
          </div>
  );
}
