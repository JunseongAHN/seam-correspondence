import { describe, expect, it } from "vitest";
import { selectHasOwnGt } from "./selectors";

const at = (result: { gt: string[] } | null) => ({ workspace: { result } });

describe("selectHasOwnGt", () => {
  it("is false with nothing loaded", () => {
    expect(selectHasOwnGt(at(null))).toBe(false);
  });
  it("is false for an input with no stitch list, which can be drawn by hand", () => {
    expect(selectHasOwnGt(at({ gt: [] }))).toBe(false);
  });
  it("is true for an input that brought its own ground truth", () => {
    expect(selectHasOwnGt(at({ gt: ["0-1"] }))).toBe(true);
  });
});
