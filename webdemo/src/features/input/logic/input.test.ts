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
  it("throws on text that is neither", () => {
    expect(() => parseInput("not json", "x.json")).toThrow();
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
