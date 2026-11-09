import { describe, expect, it } from "vitest";
import { resultLoaded } from "@/app/workspaceSlice";
import stitchView, { labelsSet, showSet } from "./stitchViewSlice";

const start = stitchView(undefined, { type: "init" });

describe("stitchViewSlice", () => {
  it("starts with labels on, and every stitch line but the ground truth's", () => {
    expect(start).toEqual({ labels: true, show: { correct: true, fp: true, fn: true, gt: false } });
  });
  it("turns labels off and on", () => {
    const off = stitchView(start, labelsSet(false));
    expect(off.labels).toBe(false);
    expect(stitchView(off, labelsSet(true)).labels).toBe(true);
  });
  it("switches one kind of line without touching the others", () => {
    expect(stitchView(start, showSet({ ...start.show, fp: false })).show)
      .toEqual({ correct: true, fp: false, fn: true, gt: false });
  });
  /* Nothing resets these: a new result leaves the switches as the reader set them. */
  it("keeps its values when a new result is loaded", () => {
    const set = stitchView(stitchView(start, labelsSet(false)),
                           showSet({ correct: false, fp: true, fn: false, gt: true }));
    const loaded = resultLoaded(
      { result: { id: "r", name: "g.json", panels: 1, edges: 4, ms: 1, pred: [], gt: [] },
        example: null, more: null, pane: { exampleRight: null, moreSim: null } },
      "req", { source: "file", file: new File([], "g.json") });
    expect(stitchView(set, loaded)).toBe(set);
  });
});
