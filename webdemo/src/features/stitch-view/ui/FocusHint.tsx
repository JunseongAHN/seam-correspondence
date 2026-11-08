import shared from "@/styles/shared.module.css";

/** How to read an unscored pattern. */
export default function FocusHint() {
  return (
                <p className={`${shared.note} ${shared.wide}`}>
                  Edges of a predicted pair share a colour. Click an edge to focus its
                  correspondence — it is drawn thick and every other pair drops to grey.
                  Click again, or click the background, to clear.
                </p>
  );
}
