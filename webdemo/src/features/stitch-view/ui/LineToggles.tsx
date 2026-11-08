import shared from "@/styles/shared.module.css";

export type Show = { correct: boolean; fp: boolean; fn: boolean; gt: boolean };

/** Which kinds of stitch line the scored view draws. */
export default function LineToggles({ show, onChange }: {
  show: Show;
  onChange: (show: Show) => void;
}) {
  return (
            <div className={shared.toggles}>
              {(["correct", "fp", "fn", "gt"] as const).map((k) => (
                <label key={k}>
                  <input type="checkbox" checked={show[k]}
                         onChange={(e) => onChange({ ...show, [k]: e.target.checked })} />
                  {k === "gt" ? "ground-truth stitches" : k === "correct" ? k : k.toUpperCase()}
                </label>
              ))}
            </div>
  );
}
