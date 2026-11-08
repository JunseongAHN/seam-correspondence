/* The solve runs in a worker, so a solver that cannot start or crashes must come back as a
   rejected promise -- a message in the pane -- rather than take the page down.  Node has no
   Worker; a stand-in plays each way it can end. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { solveInWorker, type Dump } from "./sim";

type Ending = (w: FakeWorker) => void;
class FakeWorker {
  static ending: Ending = () => {};
  static terminated = 0;
  onmessage: ((e: { data: unknown }) => void) | null = null;
  onerror: ((e: { message: string }) => void) | null = null;
  postMessage() { queueMicrotask(() => FakeWorker.ending(this)); }
  terminate() { FakeWorker.terminated++; }
}

const dump = {} as Dump;
const raw = new ArrayBuffer(8);

describe("solveInWorker", () => {
  afterEach(() => { vi.unstubAllGlobals(); FakeWorker.terminated = 0; });
  const run = (ending: Ending) => {
    FakeWorker.ending = ending;
    vi.stubGlobal("Worker", FakeWorker);
    return solveInWorker(dump, raw);
  };

  it("resolves with the worker's result", async () => {
    await expect(run((w) => w.onmessage?.({ data: { ok: true, result: { wallMs: 1 } } })))
      .resolves.toEqual({ wallMs: 1 });
  });
  it("rejects with the solver's own message when the solve throws", async () => {
    await expect(run((w) => w.onmessage?.({ data: { ok: false, error: "404 fetching simcpp.js" } })))
      .rejects.toThrow("404 fetching simcpp.js");
    expect(FakeWorker.terminated).toBe(1);
  });
  it("rejects when the worker itself dies", async () => {
    await expect(run((w) => w.onerror?.({ message: "out of memory" }))).rejects.toThrow("out of memory");
    expect(FakeWorker.terminated).toBe(1);
  });
  it("still rejects when the worker dies without a message", async () => {
    await expect(run((w) => w.onerror?.({ message: "" }))).rejects.toThrow("worker failed");
  });
  it("rejects when no worker can be started", async () => {
    vi.stubGlobal("Worker", class { constructor() { throw new Error("blocked"); } });
    await expect(solveInWorker(dump, raw)).rejects.toThrow("blocked");
  });
});
