import { describe, expect, it } from "vitest";
import { EXAMPLES, exampleGtUrl, MORE, moreSpecFile } from "./examples";
import { parseInput } from "./parseInput";
import clo from "/public/example/panel_seperated.dxf?raw";
import spec from "/public/example/rand_JYO1DHSFGH_specification.json?raw";

describe("parseInput", () => {
  it("reads a .dxf as DXF and names it without the extension", () => {
    const r = parseInput(clo, "panel_seperated.dxf");
    expect([r.isDxf, r.pattern.name, r.pattern.panels.length]).toEqual([true, "panel_seperated", 10]);
  });
  it("matches the extension case-insensitively", () => {
    expect(parseInput(clo, "PANEL.DXF").isDxf).toBe(true);
  });
  it("reads anything else as a specification", () => {
    const r = parseInput(spec, "rand_JYO1DHSFGH_specification.json");
    expect([r.isDxf, r.pattern.name, r.pattern.panels.length]).toEqual([false, "rand_JYO1DHSFGH", 4]);
  });
  it("names the file and what it was read as when JSON does not parse", () => {
    expect(() => parseInput("not a pattern", "x.json"))
      .toThrow(/^could not read x\.json as a pattern specification \(JSON\): /);
  });
  it("says the same of JSON that is not a specification", () => {
    expect(() => parseInput("{}", "x.json"))
      .toThrow(/^could not read x\.json as a pattern specification \(JSON\): /);
  });
  it("names the file when a DXF does not parse", () => {
    expect(() => parseInput("hello\nworld\n", "x.dxf")).toThrow(/^could not read x\.dxf as a DXF: /);
    expect(() => parseInput("", "x.dxf")).toThrow(/^could not read x\.dxf as a DXF: /);
  });
  it("passes on the DXF reader's own reason when it finds no panels", () => {
    expect(() => parseInput("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n", "x.dxf"))
      .toThrow("could not read x.dxf as a DXF: no panel blocks with a boundary polyline (layer 8 or 1)");
  });
  it("refuses a specification that parses but has no panels", () => {
    expect(() => parseInput('{"panels": {}}', "x.json")).toThrow("no panels found in x.json");
  });
});

describe("examples", () => {
  it("gives a checked-in ground truth only to the example that has one", () => {
    expect(exampleGtUrl("clo")).toMatch(/example\/panel_seperated_gt\.json$/);
    expect(exampleGtUrl("gcd")).toBeNull();
  });
  it("ships solver input only for the garments the model gets exactly right", () => {
    expect(MORE.filter((g) => g.sim).every((g) => g.f1 === "1.00")).toBe(true);
  });
  it("names held-out specs the way the example files are named", () => {
    expect(moreSpecFile("rand_X")).toBe("rand_X_specification.json");
    expect(Object.keys(EXAMPLES)).toEqual(["clo", "gcd"]);
  });
});
