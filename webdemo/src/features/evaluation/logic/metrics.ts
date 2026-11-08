/** How a prediction scores against a ground truth: counts, precision, recall, F1. */
export function scoreStats(pred: Set<string>, truth: Set<string>) {
  const ok = [...pred].filter((k) => truth.has(k)).length;
  const P = pred.size ? ok / pred.size : 0;
  const R = truth.size ? ok / truth.size : 0;
  return { ok, fp: pred.size - ok, fn: truth.size - ok,
           f1: P + R ? (2 * P * R) / (P + R) : 0, P, R };
}
export type Stats = ReturnType<typeof scoreStats>;
