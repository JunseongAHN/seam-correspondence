import { afterEach, describe, expect, it } from "vitest";
import { getResult, putResult, retainOnly, type ResultObjects } from "./resultCache";

const fake = (name: string): ResultObjects => ({
  pattern: { name, panels: [], stitches: [] },
  keys: [["p", 0]],
  x: new Float32Array(4),
  scores: { best: new Int32Array(1), prob: () => 0 },
});

describe("resultCache", () => {
  afterEach(() => retainOnly(null));

  it("hands back the same objects for the id it gave", () => {
    const r = fake("a");
    const id = putResult(r);
    expect(getResult(id)).toBe(r);
  });
  it("gives every result its own id", () => {
    expect(putResult(fake("a"))).not.toBe(putResult(fake("a")));
  });
  it("has nothing for no id or an unknown one", () => {
    expect(getResult(null)).toBeUndefined();
    expect(getResult(undefined)).toBeUndefined();
    expect(getResult("result-unknown")).toBeUndefined();
  });
  it("keeps only the result the store points at", () => {
    const a = putResult(fake("a")), b = putResult(fake("b")), c = putResult(fake("c"));
    retainOnly(b);
    expect([getResult(a), getResult(b)?.pattern.name, getResult(c)]).toEqual([undefined, "b", undefined]);
  });
  it("forgets everything when the store points at none", () => {
    const a = putResult(fake("a"));
    retainOnly(null);
    expect(getResult(a)).toBeUndefined();
  });
});
