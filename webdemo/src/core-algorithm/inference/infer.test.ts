import { beforeEach, describe, expect, it, vi } from "vitest";

const create = vi.fn();
vi.mock("onnxruntime-web", () => ({ InferenceSession: { create } }));

describe("loadModel", () => {
  beforeEach(() => { vi.resetModules(); create.mockReset(); });

  it("says it is the model that failed, and where it was looked for", async () => {
    create.mockRejectedValue(new Error("failed to load external data file: /model/autosew.onnx"));
    const { loadModel, modelLoaded } = await import("./infer");
    await expect(loadModel("/model/autosew.onnx")).rejects.toThrow(
      "could not load the model (/model/autosew.onnx): failed to load external data file: /model/autosew.onnx");
    expect(modelLoaded()).toBe(false);
  });

  it("tries again on the next call after a failure", async () => {
    create.mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce({ fake: true });
    const { loadModel, modelLoaded } = await import("./infer");
    await expect(loadModel("m.onnx")).rejects.toThrow();
    await expect(loadModel("m.onnx")).resolves.toEqual({ fake: true });
    expect(modelLoaded()).toBe(true);
  });

  it("loads once", async () => {
    create.mockResolvedValue({ fake: true });
    const { loadModel } = await import("./infer");
    await loadModel("m.onnx"); await loadModel("m.onnx");
    expect(create).toHaveBeenCalledTimes(1);
  });
});
