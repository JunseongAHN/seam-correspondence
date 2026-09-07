import { featureDim, NK } from "./features";
import type { Scores } from "./infer";

/* The evaluation, as a file tools/seam-report.ts and the report Worker both read.

   The geometry alone -- two lengths and two curvatures per pair -- cannot say why a
   stitch was missed, so a report built on it can only guess, and will. What separates
   the failure modes is carried here instead:

     shape_mismatch   how badly the two edges fail to interlock, the metric behind the
                      0-of-32 finding: the best of the four ways one sagitta profile can
                      be laid against the other (as-is, negated, reversed, both),
                      as a fraction of the larger profile's amplitude. 0 means the two
                      edges fit together; above 0.5 they are unrelated shapes.
     a_chose/b_chose  what each edge's argmax partner actually was, with its probability.
                      For a MISS this is the whole story: which edge took it, and whether
                      the model was confident or barely preferred it.
     p_pair           the model's probability for the pair in question.

   Lengths are ARC lengths in cm (dim 24), not the chord in dim 4. Everything is read
   from the tensor the model was given, so the report analyses the model's own view. */
export type EvalSource = {
  keys: Array<[string, number]>;
  x: Float32Array;
  pred: Set<string>;
  M: number;
  scores: Scores;
};

export type EvalDoc = ReturnType<typeof buildEval>;

export function buildEval(res: EvalSource, truth: Set<string>, name: string) {
  const D = featureDim;
  const sag = (n: number) =>
    Array.from({ length: NK }, (_, k) => res.x[n * D + 7 + k]);
  const arc = (n: number) => res.x[n * D + 7 + NK + 6] * 100;      // dim 24, cm
  const curv = (n: number) => {
    let best = 0;
    for (const v of sag(n)) if (Math.abs(v) > Math.abs(best)) best = v;
    return best;
  };

  /* Port of dxfcheck/shape_vs_hit.py:88-94, kept identical so the browser and the
     analysis scripts describe the same garment the same way. */
  const shape = (i: number, j: number) => {
    const s1 = sag(i), s2 = sag(j);
    const amp = Math.max(...s1.map(Math.abs), ...s2.map(Math.abs), 1e-9);
    const cands: [string, number[]][] = [
      ["+s", s1], ["-s", s1.map((v) => -v)],
      ["+rev", [...s1].reverse()], ["-rev", [...s1].reverse().map((v) => -v)],
    ];
    let bw = "+s", bv = Infinity;
    for (const [w, v] of cands) {
      const d = Math.max(...s2.map((q, k) => Math.abs(q - v[k])));
      if (d < bv) { bv = d; bw = w; }
    }
    return { mismatch: bv / amp, relation: bw };
  };

  const label = (n: number) => `${res.keys[n][0]}#${res.keys[n][1]}`;
  const chose = (n: number) => {
    const b = res.scores.best[n];
    return b < 0 || b >= res.M
      ? { partner: "dustbin (nothing)", p: res.scores.prob(n, res.M) }
      : { partner: label(b), p: res.scores.prob(n, b) };
  };
  const ok = [...res.pred].filter((k) => truth.has(k)).length;
  const P = res.pred.size ? ok / res.pred.size : 0;
  const R = truth.size ? ok / truth.size : 0;
  return {
    garment: name,
    f1: P + R ? (2 * P * R) / (P + R) : 0,
    pairs: [...new Set([...res.pred, ...truth])].map((k) => {
      const [a, b] = k.split("-").map(Number);
      const sh = shape(a, b);
      const ca = chose(a), cb = chose(b);
      return { edge_a: label(a), edge_b: label(b),
               pred: res.pred.has(k), gt: truth.has(k),
               len_a: arc(a), len_b: arc(b), curv_a: curv(a), curv_b: curv(b),
               shape_mismatch: sh.mismatch, shape_relation: sh.relation,
               p_pair: res.scores.prob(a, b),
               a_chose: ca.partner, p_a_chose: ca.p,
               b_chose: cb.partner, p_b_chose: cb.p };
    }),
  };
}

