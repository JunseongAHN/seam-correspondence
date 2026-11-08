import { describe, expect, it } from "vitest";
import { scoreStats } from "./metrics";

const s = (...k: string[]) => new Set(k);

describe("scoreStats", () => {
  it("counts correct, false positives and false negatives", () => {
    const r = scoreStats(s("0-1", "2-3", "4-5"), s("0-1", "2-3", "6-7", "8-9"));
    expect([r.ok, r.fp, r.fn]).toEqual([2, 1, 2]);
  });
  it("computes P = ok/pred, R = ok/truth, F1 = 2PR/(P+R)", () => {
    const r = scoreStats(s("0-1", "2-3", "4-5"), s("0-1", "2-3", "6-7", "8-9"));
    expect(r.P).toBeCloseTo(2 / 3, 12);
    expect(r.R).toBeCloseTo(1 / 2, 12);
    expect(r.f1).toBeCloseTo((2 * (2 / 3) * (1 / 2)) / (2 / 3 + 1 / 2), 12);
  });
  it("is 1/1/1 on an exact prediction", () => {
    const r = scoreStats(s("0-1", "2-3"), s("2-3", "0-1"));
    expect([r.P, r.R, r.f1, r.fp, r.fn]).toEqual([1, 1, 1, 0, 0]);
  });
  it("is 0, not NaN, with nothing predicted", () => {
    const r = scoreStats(s(), s("0-1"));
    expect([r.P, r.R, r.f1, r.fn]).toEqual([0, 0, 0, 1]);
  });
  it("is 0, not NaN, with no ground truth", () => {
    const r = scoreStats(s("0-1"), s());
    expect([r.P, r.R, r.f1, r.fp]).toEqual([0, 0, 0, 1]);
  });
  it("agrees with buildEval's F1 on the same pairs", async () => {
    const { buildEval } = await import("./evalDoc");
    const { featureDim } = await import("@/core-algorithm/inference");
    const M = 10, pred = s("0-1", "2-3", "4-5"), truth = s("0-1", "6-7");
    const doc = buildEval({ keys: Array.from({ length: M }, (_, i): [string, number] => ["p", i]),
                            x: new Float32Array(M * featureDim), pred, M,
                            scores: { best: new Int32Array(M).fill(-1), prob: () => 0 } }, truth, "g");
    expect(scoreStats(pred, truth).f1).toBe(doc.f1);
  });
});
