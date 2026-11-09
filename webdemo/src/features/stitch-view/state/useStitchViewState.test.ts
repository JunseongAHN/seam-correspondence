// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useStitchViewState } from "./useStitchViewState";

describe("useStitchViewState", () => {
  it("starts with labels on, and every stitch line but the ground truth's", () => {
    const { result } = renderHook(() => useStitchViewState());
    expect(result.current.labels).toBe(true);
    expect(result.current.show).toEqual({ correct: true, fp: true, fn: true, gt: false });
  });

  it("turns labels off and on", () => {
    const { result } = renderHook(() => useStitchViewState());
    act(() => result.current.setLabels(false));
    expect(result.current.labels).toBe(false);
    act(() => result.current.setLabels(true));
    expect(result.current.labels).toBe(true);
  });

  it("switches one kind of line without touching the others", () => {
    const { result } = renderHook(() => useStitchViewState());
    act(() => result.current.setShow({ ...result.current.show, fp: false }));
    expect(result.current.show).toEqual({ correct: true, fp: false, fn: true, gt: false });
  });

  /* Nothing resets these: loading another pattern only re-renders App, which calls this
     hook again with no new input, so the switches keep what the reader chose. */
  it("keeps its values across re-renders, as when another pattern is loaded", () => {
    const { result, rerender } = renderHook(() => useStitchViewState());
    act(() => { result.current.setLabels(false); result.current.setShow({ correct: false, fp: true, fn: false, gt: true }); });
    rerender();
    expect(result.current.labels).toBe(false);
    expect(result.current.show).toEqual({ correct: false, fp: true, fn: false, gt: true });
  });
});
