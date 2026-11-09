import { describe, expect, it } from "vitest";
import { edgeStyle, focusNodes, isLit, nodeIndex, pairGroups, pairPartners,
         stitchIndexByEdge, touchesFocus } from "./pairs";
import type { Pattern } from "@/core-algorithm/pattern";

describe("pairGroups", () => {
  it("puts both sides of a pair, and a chain of pairs, in one group", () => {
    const g = pairGroups(new Set(["0-1", "1-2", "5-6"]));
    expect(g.get(0)).toBe(g.get(1));
    expect(g.get(1)).toBe(g.get(2));
    expect(g.get(5)).toBe(g.get(6));
    expect(g.get(5)).not.toBe(g.get(0));
    expect(g.has(3)).toBe(false);
  });

  it("numbers groups in the order their pairs were made", () => {
    const g = pairGroups(new Set(["8-9", "0-1"]));
    expect([g.get(8), g.get(0)]).toEqual([0, 1]);
  });

  it("does not renumber existing groups when a pair is added", () => {
    const before = pairGroups(new Set(["4-7", "2-3", "10-11"]));
    // a lower node index than any existing one, made last
    const after = pairGroups(new Set(["4-7", "2-3", "10-11", "0-1"]));
    for (const [n, g] of before) expect(after.get(n)).toBe(g);
    expect(after.get(0)).toBe(3);
  });

  it("is empty with no pairs", () => {
    expect(pairGroups(new Set()).size).toBe(0);
  });
});

describe("pairPartners", () => {
  it("links both directions", () => {
    const p = pairPartners(["0-1", "0-2"]);
    expect([...(p.get(0) ?? [])]).toEqual([1, 2]);
    expect([...(p.get(2) ?? [])]).toEqual([0]);
  });
});

describe("focusNodes", () => {
  const group = pairGroups(new Set(["0-1", "1-2", "5-6"]));
  const partners = pairPartners(["0-1", "1-2", "5-6", "8-9"]);

  it("is null with nothing focused", () => {
    expect(focusNodes(null, group, partners)).toBeNull();
  });
  it("lights the whole group of a grouped node", () => {
    expect([...(focusNodes(2, group, partners) ?? [])].sort()).toEqual([0, 1, 2]);
  });
  it("falls back to the node and its partners when it has no group", () => {
    expect([...(focusNodes(8, group, partners) ?? [])].sort()).toEqual([8, 9]);
  });
  it("is just the node when it has neither", () => {
    expect([...(focusNodes(20, group, partners) ?? [])]).toEqual([20]);
  });
  it("keeps a stitch line lit if either end is lit", () => {
    const f = focusNodes(8, group, partners);
    expect([isLit(f, 8, 30), isLit(f, 30, 9), isLit(f, 30, 31), isLit(null, 30, 31)])
      .toEqual([true, true, false, true]);
  });
});

describe("stitchIndexByEdge", () => {
  const keys: Array<[string, number]> = [["a", 0], ["a", 1], ["b", 0], ["b", 1]];
  it("reads a specification's stitch list", () => {
    const p = { name: "x", panels: [], stitches: [[["a", 1], ["b", 0]]] } as Pattern;
    expect([...stitchIndexByEdge(p, new Set(["0-3"]), keys)]).toEqual([["a#1", 0], ["b#0", 0]]);
  });
  it("falls back to ground-truth node pairs when there is no list (a DXF)", () => {
    const p = { name: "x", panels: [], stitches: [] } as Pattern;
    expect([...stitchIndexByEdge(p, new Set(["0-3", "1-2"]), keys)])
      .toEqual([["a#0", 0], ["b#1", 0], ["a#1", 1], ["b#0", 1]]);
  });
});

describe("nodeIndex", () => {
  it("maps panel#edge to the node index", () => {
    expect(nodeIndex([["a", 0], ["a", 2]]).get("a#2")).toBe(1);
  });
});

describe("edgeStyle", () => {
  const base = { selected: false, stitch: undefined, inFocus: false, dim: false,
                 annotate: false, scored: false };
  it("draws an unpaired edge thin and grey", () => {
    expect(edgeStyle(base)).toEqual({ color: "#c9c9c9", width: 0.5 });
  });
  it("draws the selected edge orange and thick, over everything else", () => {
    expect(edgeStyle({ ...base, selected: true, stitch: 2, dim: true }))
      .toEqual({ color: "#f97316", width: 3.2 });
  });
  it("greys a dimmed stitch but keeps its width", () => {
    expect(edgeStyle({ ...base, stitch: 2, dim: true }).color).toBe("#c9c9c9");
  });
});

describe("touchesFocus", () => {
  it("draws every line when nothing is focused", () => {
    expect(touchesFocus(null, 4, 9)).toBe(true);
  });
  it("draws the clicked edge's own lines, from either end", () => {
    expect([touchesFocus(3, 3, 8), touchesFocus(3, 1, 3)]).toEqual([true, true]);
  });
  /* 3 is predicted with 8, but 8's ground truth is 5: that line belongs to 8, not to 3 */
  it("does not draw a line between a partner and some third edge", () => {
    expect(touchesFocus(3, 8, 5)).toBe(false);
  });
});
