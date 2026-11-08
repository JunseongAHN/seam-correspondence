/* Characterization of the feature tensor the model is fed.  The Python comparison
   (scripts/feature_parity.mjs + dxfcheck/feature_parity.py) needs the training checkpoint
   and data; this pins the TS output so a refactor cannot move it unnoticed. */
import { describe, expect, it } from "vitest";
import { parseSpec } from "./parseSpec";
import { parseDxf } from "./parseDxf";
import { featureDim, gtPairs, patternToTensors } from "./features";
import clo from "../../public/example/panel_seperated.dxf?raw";
import s3C7X from "../../public/example/rand_3C7X2I6WQ7_specification.json?raw";
import sJYO1 from "../../public/example/rand_JYO1DHSFGH_specification.json?raw";

describe("patternToTensors", () => {
  it("has the width the shipped model expects", () => {
    expect(featureDim).toBe(26);
  });

  it.each([
    ["rand_3C7X2I6WQ7", () => parseSpec(JSON.parse(s3C7X), "rand_3C7X2I6WQ7")],
    ["rand_JYO1DHSFGH", () => parseSpec(JSON.parse(sJYO1), "rand_JYO1DHSFGH")],
    ["panel_seperated.dxf", () => parseDxf(clo, "panel_seperated")],
  ])("%s", async (name, load) => {
    const p = load();
    const t = patternToTensors(p);
    expect(t.x.length).toBe(t.M * featureDim);
    expect(t.nbr.length).toBe(t.M * 2);
    await expect(JSON.stringify({
      M: t.M, keys: t.keys, x: Array.from(t.x), nbr: Array.from(t.nbr, Number),
      gt: [...gtPairs(p, t.keys)],
    })).toMatchFileSnapshot(`./__snapshots__/features.${name}.json`);
  });
});
