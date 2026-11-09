import { describe, expect, it } from "vitest";
import { EXAMPLES, MORE } from "./examples";
import { exampleOpenArg, moreOpenArg } from "./openArgs";

describe("openPattern arguments", () => {
  it("give the CLO example its checked-in ground truth and CLO's own drape", () => {
    expect(exampleOpenArg("clo")).toEqual({
      source: "url", url: EXAMPLES.clo.url, fileName: "panel_seperated.dxf",
      gtUrl: EXAMPLES.clo.gtUrl, example: "clo", more: null, exampleRight: "drape", moreSim: null });
  });
  it("give the GarmentCode example no gt file, since a specification carries its own", () => {
    expect(exampleOpenArg("gcd")).toMatchObject({ gtUrl: null, example: "gcd", exampleRight: "sim" });
  });
  it("pass a held-out garment's solver input only when it ships one", () => {
    const exact = MORE.find((g) => g.sim)!;      // the list has two
    const missed = MORE.find((g) => !g.sim)!;
    expect(moreOpenArg(exact.id)).toMatchObject({ more: exact.id, example: null, moreSim: exact.sim,
                                                  fileName: `${exact.id}_specification.json` });
    expect(moreOpenArg(missed.id).moreSim).toBeNull();
  });
});
