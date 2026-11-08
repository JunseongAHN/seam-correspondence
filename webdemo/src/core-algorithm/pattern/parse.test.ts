/* Characterization: what the parsers produce today for the shipped examples, so a
   refactor that changes a panel or edge count fails here instead of in the browser. */
import { describe, expect, it } from "vitest";
import { parseSpec } from "./parseSpec";
import { parseDxf } from "./parseDxf";
import clo from "/public/example/panel_seperated.dxf?raw";
import s1328 from "/public/example/rand_1328ERLDIC_specification.json?raw";
import s3C7X from "/public/example/rand_3C7X2I6WQ7_specification.json?raw";
import sE3IN from "/public/example/rand_E3IN9RH60H_specification.json?raw";
import sGE9N from "/public/example/rand_GE9NBC1HFY_specification.json?raw";
import sI88D from "/public/example/rand_I88DFY2AKV_specification.json?raw";
import sJYO1 from "/public/example/rand_JYO1DHSFGH_specification.json?raw";
import type { Pattern } from "./types";

const summary = (p: Pattern) => ({
  panels: p.panels.length,
  edges: p.panels.reduce((s, q) => s + q.edges.length, 0),
  stitches: p.stitches.length,
});

describe("parseSpec", () => {
  it.each([
    ["rand_1328ERLDIC", s1328], ["rand_3C7X2I6WQ7", s3C7X], ["rand_E3IN9RH60H", sE3IN],
    ["rand_GE9NBC1HFY", sGE9N], ["rand_I88DFY2AKV", sI88D], ["rand_JYO1DHSFGH", sJYO1],
  ])("%s", (name, text) => {
    expect(summary(parseSpec(JSON.parse(text), name))).toMatchSnapshot();
  });
});

describe("parseDxf", () => {
  it("panel_seperated.dxf", () => {
    expect(summary(parseDxf(clo, "panel_seperated"))).toMatchSnapshot();
  });
});
