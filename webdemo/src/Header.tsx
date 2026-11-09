import s from "./Header.module.css";

/** What the page is, and the one limit a reader has to know before judging anything on it. */
export default function Header() {
  return (
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
  );
}
