// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useEvaluationState } from "./useEvaluationState";

describe("useEvaluationState", () => {
  it("starts with the report closed", () => {
    const { result } = renderHook(() => useEvaluationState());
    expect(result.current.showReport).toBe(false);
  });

  it("opens and closes the report", () => {
    const { result } = renderHook(() => useEvaluationState());
    act(() => result.current.toggleReport());
    expect(result.current.showReport).toBe(true);
    act(() => result.current.toggleReport());
    expect(result.current.showReport).toBe(false);
  });

  it("applies two quick toggles in turn rather than one", () => {
    const { result } = renderHook(() => useEvaluationState());
    act(() => { result.current.toggleReport(); result.current.toggleReport(); });
    expect(result.current.showReport).toBe(false);
  });

  /* Nothing resets it: loading another pattern only re-renders App, so an open report
     stays open and shows again as soon as the new pattern can be scored. */
  it("stays open across re-renders, as when another pattern is loaded", () => {
    const { result, rerender } = renderHook(() => useEvaluationState());
    act(() => result.current.toggleReport());
    rerender();
    expect(result.current.showReport).toBe(true);
  });
});
