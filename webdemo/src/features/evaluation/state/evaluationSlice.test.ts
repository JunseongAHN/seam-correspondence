import { describe, expect, it } from "vitest";
import { resultLoaded } from "@/app/workspaceSlice";
import evaluation, { reportToggled } from "./evaluationSlice";

const start = evaluation(undefined, { type: "init" });

describe("evaluationSlice", () => {
  it("starts with the report closed", () => {
    expect(start.showReport).toBe(false);
  });
  it("opens and closes the report", () => {
    const open = evaluation(start, reportToggled());
    expect(open.showReport).toBe(true);
    expect(evaluation(open, reportToggled()).showReport).toBe(false);
  });
  /* Nothing resets it: an open report stays open when another pattern is loaded. */
  it("stays open when a new result is loaded", () => {
    const open = evaluation(start, reportToggled());
    const loaded = resultLoaded(
      { result: { id: "r", name: "g.json", panels: 1, edges: 4, ms: 1, pred: [], gt: [] },
        example: null, more: null, pane: { exampleRight: null, moreSim: null } },
      "req", { source: "file", file: new File([], "g.json") });
    expect(evaluation(open, loaded)).toBe(open);
  });
});
