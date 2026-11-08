import { describe, expect, it } from "vitest";
import { pickRightPane } from "./pickRightPane";

const stats = { ok: 20, fp: 2, fn: 3 };
const sim = { file: "g.bin", expectS: 26 };
const base = { example: null, more: null, moreSim: undefined, exact: false, stats };

describe("pickRightPane", () => {
  it("shows nothing for a user's own file", () => {
    expect(pickRightPane(base)).toBeNull();
  });
  it("shows CLO's drape for the CLO example", () => {
    expect(pickRightPane({ ...base, example: "drape" })?.kind).toBe("weld");
  });
  it("shows the solver for the GarmentCode example, with the score it can or cannot assemble", () => {
    expect(pickRightPane({ ...base, example: "sim" })).toEqual({
      kind: "sim", title: "assembled by the wasm solver · 3D", gt: false,
      props: { exact: false, fp: 2, fn: 3 } });
  });
  it("lets an example win over a held-out garment id", () => {
    expect(pickRightPane({ ...base, example: "drape", more: "g", moreSim: sim, exact: true })?.kind)
      .toBe("weld");
  });
  it("solves a held-out garment the model got exactly right", () => {
    expect(pickRightPane({ ...base, more: "g", moreSim: sim, exact: true })).toEqual({
      kind: "sim", title: "assembled by the wasm solver · 3D — the prediction, solved", gt: false,
      props: { exact: true, fp: 0, fn: 0, ok: 20, file: "g.bin", expectS: 26 } });
  });
  it("shows the red ground-truth drape when the prediction is not exact", () => {
    const p = pickRightPane({ ...base, more: "g", moreSim: sim, exact: false });
    expect(p).toEqual({ kind: "gtdrape", title: "ground-truth drape · 3D", gt: true,
                        props: { id: "g", exact: false, fp: 2, fn: 3 } });
  });
  it("shows the ground-truth drape for an exact garment that has no solver input", () => {
    expect(pickRightPane({ ...base, more: "g", exact: true })?.kind).toBe("gtdrape");
  });
});
