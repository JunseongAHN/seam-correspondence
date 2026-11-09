import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import spec from "/public/example/rand_JYO1DHSFGH_specification.json?raw";
import { getResult, retainOnly } from "@/adapters/resultCache";
import { makeStore } from "./store";
import { errorSet, openPattern, type OpenArg } from "./workspaceSlice";

// Everything real but the model: parsing, features and ground truth run as in the page.
/* `held`, when set, collects each prediction unfinished so a test decides when -- and in
   which order -- they end. */
const model = vi.hoisted(() => ({
  loaded: false, fail: null as Error | null,
  held: null as null | Array<{ finish: () => void; fail: (e: Error) => void }>,
}));
vi.mock("@/core-algorithm/inference", async (orig) => {
  const real = await orig<typeof import("@/core-algorithm/inference")>();
  return {
    ...real,
    modelLoaded: () => model.loaded,
    predict: vi.fn(async () => {
      if (model.held) {
        const held = model.held;
        await new Promise<void>((finish, fail) => held.push({ finish, fail }));
      }
      if (model.fail) throw model.fail;
      model.loaded = true;
      return { pairs: new Set(["2-3", "0-1"]), ms: 7, scores: { best: new Int32Array(16), prob: () => 0 } };
    }),
  };
});

const file = (text: string, name: string): OpenArg => ({ source: "file", file: new File([text], name) });
const url = (over: Partial<Extract<OpenArg, { source: "url" }>> = {}): OpenArg => ({
  source: "url", url: "/example/g_specification.json", fileName: "g_specification.json",
  gtUrl: null, example: "gcd", more: null, exampleRight: "sim", moreSim: null, ...over });
const respond = (routes: Record<string, () => Response>) =>
  vi.stubGlobal("fetch", vi.fn(async (u: string) => (routes[u] ?? (() => new Response("", { status: 404 })))()));

describe("openPattern", () => {
  beforeEach(() => { model.loaded = false; model.fail = null; model.held = null; });
  afterEach(() => { vi.unstubAllGlobals(); retainOnly(null); });

  it("puts the small part of a result in the store and the rest in the cache", async () => {
    const store = makeStore();
    await store.dispatch(openPattern(file(spec, "rand_JYO1DHSFGH_specification.json")));
    const w = store.getState().workspace;
    expect(w.result).toMatchObject({ name: "rand_JYO1DHSFGH_specification.json", panels: 4, edges: 16,
                                     ms: 7, pred: ["2-3", "0-1"] });
    expect(w.result?.gt).toHaveLength(4);
    expect([w.name, w.example, w.more, w.busy, w.error]).toEqual(
      ["rand_JYO1DHSFGH_specification.json", null, null, false, null]);
    expect(getResult(w.result?.id)?.pattern.name).toBe("rand_JYO1DHSFGH");
  });

  it("keeps the pairs in the order the model gave them", async () => {
    const store = makeStore();
    await store.dispatch(openPattern(file(spec, "x_specification.json")));
    expect(store.getState().workspace.result?.pred).toEqual(["2-3", "0-1"]);
  });

  it("is busy while it runs, and clears the last error when it starts", async () => {
    const store = makeStore();
    store.dispatch(errorSet("old"));
    const running = store.dispatch(openPattern(file(spec, "x_specification.json")));
    expect([store.getState().workspace.busy, store.getState().workspace.error]).toEqual([true, null]);
    await running;
    expect(store.getState().workspace.busy).toBe(false);
  });

  it("records what the result was opened as, for the 3D pane", async () => {
    respond({ "/example/g_specification.json": () => new Response(spec) });
    const store = makeStore();
    await store.dispatch(openPattern(url({ example: null, more: "g", exampleRight: null,
                                           moreSim: { file: "g.bin", expectS: 26 } })));
    const w = store.getState().workspace;
    expect([w.example, w.more, w.pane]).toEqual([null, "g", { exampleRight: null, moreSim: { file: "g.bin", expectS: 26 } }]);
  });

  it("takes the ground truth from the gt file when there is one", async () => {
    respond({
      "/example/g_specification.json": () => new Response(spec),
      "/example/g_gt.json": () => Response.json({ stitches: [[["skirt_panel_0", 1], ["skirt_panel_1", 3]]] }),
    });
    const store = makeStore();
    await store.dispatch(openPattern(url({ gtUrl: "/example/g_gt.json" })));
    expect(store.getState().workspace.result?.gt).toHaveLength(1);
  });

  it("leaves the previous result on screen when a download fails", async () => {
    const store = makeStore();
    await store.dispatch(openPattern(file(spec, "x_specification.json")));
    const before = store.getState().workspace;
    respond({});
    await store.dispatch(openPattern(url({ example: "clo" })));
    const w = store.getState().workspace;
    expect(w.error).toBe("/example/g_specification.json: HTTP 404");
    expect([w.result, w.example, w.more, w.name]).toEqual([before.result, before.example, before.more, before.name]);
    expect(getResult(w.result?.id)).toBeDefined();
  });

  it("clears the result when the pattern cannot be read, but keeps the file name", async () => {
    respond({ "/example/g_specification.json": () => new Response(spec) });
    const store = makeStore();
    await store.dispatch(openPattern(url()));
    const old = store.getState().workspace.result?.id;
    await store.dispatch(openPattern(file("not a pattern", "bad.json")));
    const w = store.getState().workspace;
    expect(w.error).toMatch(/^could not read bad\.json as a pattern specification \(JSON\): /);
    expect([w.result, w.example, w.more, w.name]).toEqual([null, null, null, "g_specification.json"]);
    expect(getResult(old)).toBeUndefined();
  });

  it("says the model is loading only while the first prediction fetches it", async () => {
    const store = makeStore();
    const seen: string[] = [];
    const stop = store.subscribe(() => seen.push(store.getState().workspace.modelStatus));
    await store.dispatch(openPattern(file(spec, "x_specification.json")));
    await store.dispatch(openPattern(file(spec, "x_specification.json")));
    stop();
    expect([...new Set(seen)]).toEqual(["idle", "loading", "ready"]);
  });

  it("marks the model failed, and tries it again on the next load", async () => {
    const store = makeStore();
    model.fail = new Error("could not load the model (/model/autosew.onnx): offline");
    await store.dispatch(openPattern(file(spec, "x_specification.json")));
    expect(store.getState().workspace.modelStatus).toBe("error");
    expect(store.getState().workspace.error).toBe("could not load the model (/model/autosew.onnx): offline");
    model.fail = null;
    await store.dispatch(openPattern(file(spec, "x_specification.json")));
    expect(store.getState().workspace.modelStatus).toBe("ready");
  });

  it("does not touch the model status when the pattern fails before the model is needed", async () => {
    const store = makeStore();
    await store.dispatch(openPattern(file("not a pattern", "bad.json")));
    expect(store.getState().workspace.modelStatus).toBe("idle");
  });

  describe("when loads overlap", () => {
    const flush = () => new Promise((r) => setTimeout(r, 0));
    /** Start two loads; both predictions wait until the test lets them finish. */
    async function twoLoads() {
      model.held = [];
      const store = makeStore();
      const first = store.dispatch(openPattern(file(spec, "first_specification.json")));
      await flush();
      const second = store.dispatch(openPattern(file(spec, "second_specification.json")));
      await flush();
      const [a, b] = model.held;
      return { store, first, second, a, b };
    }

    it("does not let an older load that finishes late overwrite the newer result", async () => {
      const { store, first, second, a, b } = await twoLoads();
      b.finish(); await second;
      const shown = store.getState().workspace.result;
      a.finish(); await first;
      const w = store.getState().workspace;
      expect(w.result?.name).toBe("second_specification.json");
      expect(w.result).toBe(shown);
      expect([w.name, w.busy, w.error]).toEqual(["second_specification.json", false, null]);
      expect(getResult(w.result?.id)?.pattern.name).toBe("second");
    });

    it("drops the late load's objects from the cache and keeps the shown one", async () => {
      const { store, first, second, a, b } = await twoLoads();
      b.finish(); await second;
      const shownId = store.getState().workspace.result?.id;
      a.finish(); const late = await first;
      const lateId = (late.payload as { result: { id: string } }).result.id;
      expect(getResult(lateId)).toBeUndefined();
      expect(getResult(shownId)).toBeDefined();
    });

    it("stays busy while the newer load runs, though the older one has ended", async () => {
      const { store, first, second, a, b } = await twoLoads();
      a.finish(); await first;
      expect(store.getState().workspace.busy).toBe(true);
      expect(store.getState().workspace.result).toBeNull();
      b.finish(); await second;
      expect(store.getState().workspace.result?.name).toBe("second_specification.json");
    });

    it("does not let an older load that fails late clear the newer result or set an error", async () => {
      const { store, first, second, a, b } = await twoLoads();
      b.finish(); await second;
      model.fail = new Error("boom");
      a.finish(); await first;
      const w = store.getState().workspace;
      expect([w.result?.name, w.error]).toEqual(["second_specification.json", null]);
      expect(getResult(w.result?.id)).toBeDefined();
    });

    it("still learns from a stale load that the model is now in memory", async () => {
      model.held = [];
      const store = makeStore();
      const first = store.dispatch(openPattern(file(spec, "first_specification.json")));
      await flush();
      await store.dispatch(openPattern(file("not a pattern", "bad.json")));   // fails before the model
      expect(store.getState().workspace.modelStatus).toBe("loading");
      model.held[0].finish(); await first;
      expect(store.getState().workspace.modelStatus).toBe("ready");
    });
  });
});
