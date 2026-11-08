/* The F1 the evaluation document reports.  P and R are not exposed here; they are
   computed separately in App.tsx and get their own test when moved to logic/. */
import { describe, expect, it } from "vitest";
import { buildEval } from "./evalDoc";
import { featureDim } from "@/core-algorithm/inference";

const source = (M: number, pred: string[]) => ({
  keys: Array.from({ length: M }, (_, i): [string, number] => ["p", i]),
  x: new Float32Array(M * featureDim),
  pred: new Set(pred),
  M,
  scores: { best: new Int32Array(M).fill(-1), prob: () => 0 },
});

describe("buildEval f1", () => {
  it("is 2PR/(P+R)", () => {
    // 1 of 2 predictions correct (P = .5), 1 of 1 truth found (R = 1)
    expect(buildEval(source(4, ["0-1", "2-3"]), new Set(["0-1"]), "g").f1).toBeCloseTo(2 / 3, 12);
  });
  it("is 1 when prediction equals truth", () => {
    expect(buildEval(source(4, ["0-1", "2-3"]), new Set(["0-1", "2-3"]), "g").f1).toBe(1);
  });
  it("is 0 with no correct pair", () => {
    expect(buildEval(source(4, ["0-1"]), new Set(["2-3"]), "g").f1).toBe(0);
  });
  it("is 0 with nothing predicted", () => {
    expect(buildEval(source(4, []), new Set(["0-1"]), "g").f1).toBe(0);
  });
  it("is 0 with no truth", () => {
    expect(buildEval(source(4, ["0-1"]), new Set(), "g").f1).toBe(0);
  });
  it("lists the union of predicted and true pairs, flagged", () => {
    const d = buildEval(source(4, ["0-1", "2-3"]), new Set(["0-1", "1-2"]), "g");
    expect(d.pairs.map((q) => [q.edge_a, q.edge_b, q.pred, q.gt])).toEqual([
      ["p#0", "p#1", true, true], ["p#2", "p#3", true, false], ["p#1", "p#2", false, true],
    ]);
  });
});
