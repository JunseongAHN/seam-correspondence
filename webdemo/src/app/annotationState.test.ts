/* Drawing a ground truth by hand, and the derived values that read it, as reducers and
   selectors -- no load needed, a result is put in place directly. */
import { describe, expect, it } from "vitest";
import workspace, {
  annotateToggled, edgeClicked, errorSet, pairsCleared, pairsLoaded, openPattern,
  selectPred, selectScored, selectStats, selectTruth, selectUserPairs, type WorkspaceState,
} from "./workspaceSlice";

const start = workspace(undefined, { type: "init" });
const arg = { source: "file" as const, file: new File([], "g.json") };
const withResult = (pred: string[], gt: string[]): WorkspaceState => workspace(
  workspace(start, openPattern.pending("req", arg)),
  openPattern.fulfilled({ result: { id: "r", name: "g.json", panels: 2, edges: 8, ms: 1, pred, gt },
                         example: null, more: null, pane: { exampleRight: null, moreSim: null } }, "req", arg));
const run = (state: WorkspaceState, ...actions: Parameters<typeof workspace>[1][]) =>
  actions.reduce((s, a) => workspace(s, a), state);
const root = (w: WorkspaceState) => ({ workspace: w });

describe("drawing a ground truth", () => {
  it("starts off, with nothing drawn", () => {
    expect([start.annotate, start.userPairs, start.selected]).toEqual([false, [], null]);
  });
  it("selects the first edge, sews the second to it smaller index first", () => {
    const s = run(start, edgeClicked(5));
    expect(s.selected).toBe(5);
    expect(run(s, edgeClicked(2))).toMatchObject({ selected: null, userPairs: ["2-5"] });
  });
  it("deselects an edge clicked twice", () => {
    expect(run(start, edgeClicked(5), edgeClicked(5))).toMatchObject({ selected: null, userPairs: [] });
  });
  it("keeps pairs in the order they were sewn, which picks their colours", () => {
    const s = run(start, edgeClicked(8), edgeClicked(9), edgeClicked(0), edgeClicked(1));
    expect(s.userPairs).toEqual(["8-9", "0-1"]);
  });
  it("undoes a stitch when one of its edges is clicked with nothing selected", () => {
    const s = run(start, edgeClicked(8), edgeClicked(9), edgeClicked(0), edgeClicked(1), edgeClicked(9));
    expect(s.userPairs).toEqual(["0-1"]);
  });
  it("forgets a half-made pair when drawing is switched on or off", () => {
    const s = run(start, edgeClicked(3), annotateToggled());
    expect([s.annotate, s.selected]).toEqual([true, null]);
    expect(run(s, annotateToggled()).annotate).toBe(false);
  });
  it("clears every pair and the selection", () => {
    expect(run(start, edgeClicked(1), edgeClicked(2), edgeClicked(4), pairsCleared()))
      .toMatchObject({ userPairs: [], selected: null });
  });
  it("replaces the pairs with a loaded file's and clears the page error", () => {
    expect(run(start, errorSet("bad file"), edgeClicked(1), edgeClicked(2), pairsLoaded(["3-4", "0-6"])))
      .toMatchObject({ userPairs: ["3-4", "0-6"], error: null });
  });
});

describe("shared selectors", () => {
  it("score nothing without a result", () => {
    expect([selectScored(root(start)), selectStats(root(start))]).toEqual([false, null]);
  });
  it("score against the file's ground truth", () => {
    const w = withResult(["0-1", "2-3"], ["0-1", "4-5"]);
    expect([...selectTruth(root(w))]).toEqual(["0-1", "4-5"]);
    expect(selectScored(root(w))).toBe(true);
    expect(selectStats(root(w))).toMatchObject({ ok: 1, fp: 1, fn: 1 });
  });
  it("let a hand-drawn ground truth outrank the file's", () => {
    const w = run(withResult(["0-1", "2-3"], ["0-1"]), edgeClicked(2), edgeClicked(3));
    expect([...selectTruth(root(w))]).toEqual(["2-3"]);
    expect(selectStats(root(w))).toMatchObject({ ok: 1, fp: 1, fn: 0 });
  });
  it("make a result with no ground truth at all unscored", () => {
    expect(selectScored(root(withResult(["0-1"], [])))).toBe(false);
  });
  it("hand back the same Set until its list changes, so components do not redraw for nothing", () => {
    const w = withResult(["0-1"], ["0-1"]);
    const again = run(w, errorSet("unrelated"));
    expect(selectPred(root(again))).toBe(selectPred(root(w)));
    expect(selectTruth(root(again))).toBe(selectTruth(root(w)));
    expect(selectUserPairs(root(again))).toBe(selectUserPairs(root(w)));
  });
});
