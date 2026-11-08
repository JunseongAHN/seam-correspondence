import { describe, expect, it } from "vitest";
import { annotationClick } from "./annotationClick";
import { gtFromDoc, gtFromText, gtToDoc } from "./gtDoc";

const keys: Array<[string, number]> = [["front", 0], ["front", 1], ["back", 0], ["back", 1]];

describe("annotationClick", () => {
  const none = new Set<string>();
  it("selects the first edge clicked", () => {
    expect(annotationClick(null, none, 2)).toEqual({ selected: 2, change: null });
  });
  it("deselects when the same edge is clicked again", () => {
    expect(annotationClick(2, none, 2)).toEqual({ selected: null, change: null });
  });
  it("sews the second edge to the first, smaller index first", () => {
    expect(annotationClick(3, none, 1)).toEqual({ selected: null, change: { key: "1-3", remove: false } });
  });
  it("unsews a pair picked again in either order", () => {
    expect(annotationClick(3, new Set(["1-3"]), 1))
      .toEqual({ selected: null, change: { key: "1-3", remove: true } });
  });
  it("undoes a stitch when an already sewn edge is clicked with nothing selected", () => {
    expect(annotationClick(null, new Set(["0-2", "1-3"]), 3))
      .toEqual({ selected: null, change: { key: "1-3", remove: true } });
  });
});

describe("ground-truth file", () => {
  const res = { pattern: { panels: [{}, {}] }, M: 4, keys };

  it("names panel/edge, not node indices", () => {
    expect(gtToDoc(res, new Set(["0-3"]), "shirt.dxf")).toEqual({
      source: "shirt.dxf", panels: 2, edges: 4, stitches: [[["front", 0], ["back", 1]]],
    });
  });
  it("round-trips through save and load", () => {
    const pairs = new Set(["0-3", "1-2"]);
    expect(gtFromText(JSON.stringify(gtToDoc(res, pairs, "x")), keys)).toEqual(pairs);
  });
  it("orders a loaded pair smaller index first", () => {
    expect(gtFromDoc({ stitches: [[["back", 1], ["front", 0]]] }, keys)).toEqual(new Set(["0-3"]));
  });
  it("refuses a file that names edges this pattern does not have", () => {
    expect(() => gtFromDoc({ stitches: [[["front", 0], ["sleeve", 4]]] }, keys))
      .toThrow("1 stitch(es) in the file name edges this pattern does not have");
  });
  it("refuses a stitch that is not a pair", () => {
    expect(() => gtFromDoc({ stitches: [[["front", 0], ["back", 0], ["back", 1]]] }, keys)).toThrow();
  });
  it("reads a file with no stitches as an empty truth", () => {
    expect(gtFromDoc({}, keys).size).toBe(0);
  });
});
