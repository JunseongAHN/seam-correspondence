import { describe, expect, it } from "vitest";
import { addPair, removePair, toPairList, toPairSet } from "./pairs";

describe("pair lists", () => {
  it("keep the order the pairs were made, not sorted order", () => {
    expect(toPairList(new Set(["8-9", "0-1", "3-4"]))).toEqual(["8-9", "0-1", "3-4"]);
  });
  it("drop repeats, keeping the first", () => {
    expect(toPairList(["2-3", "0-1", "2-3"])).toEqual(["2-3", "0-1"]);
  });
  it("round-trip through a Set unchanged", () => {
    const list = ["5-9", "1-2", "0-7"];
    expect(toPairList(toPairSet(list))).toEqual(list);
  });
  it("are serializable: they survive JSON", () => {
    const list = toPairList(new Set(["5-9", "1-2"]));
    expect(JSON.parse(JSON.stringify(list))).toEqual(list);
  });
  it("do not change the list they are given", () => {
    const list = ["0-1"];
    addPair(list, "2-3"); removePair(list, "0-1");
    expect(list).toEqual(["0-1"]);
  });

  /* The store's list must behave exactly as the Set it replaces, or the order -- and the
     colours -- drift apart after a run of clicks. */
  it("track a Set through any sequence of adds and deletes", () => {
    let seed = 11;
    const rnd = (n: number) => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; };
    for (let run = 0; run < 500; run++) {
      const set = new Set<string>();
      let list: string[] = [];
      for (let step = 0; step < 40; step++) {
        const a = rnd(12), b = rnd(12);
        const key = `${Math.min(a, b)}-${Math.max(a, b)}`;
        if (rnd(3) === 0) { set.delete(key); list = removePair(list, key); }
        else { set.add(key); list = addPair(list, key); }
        expect(list).toEqual([...set]);
      }
    }
  });
});
