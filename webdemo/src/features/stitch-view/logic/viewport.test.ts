import { describe, expect, it } from "vitest";
import { clientToUser, dragView, strokeScale, unitsPerPixel, zoomAt, type View } from "./viewport";

const W = 600;
const full: View = { x: 0, y: 0, w: W, h: 300 };

describe("zoomAt", () => {
  it("never zooms in past W/60", () => {
    let v = full;
    for (let i = 0; i < 50; i++) v = zoomAt(v, { x: 100, y: 100 }, -1000, W);
    expect(v.w).toBeCloseTo(W / 60, 9);
  });
  it("never zooms out past 1.4 W", () => {
    let v = full;
    for (let i = 0; i < 50; i++) v = zoomAt(v, { x: 100, y: 100 }, 1000, W);
    expect(v.w).toBeCloseTo(W * 1.4, 9);
  });
  it("keeps the aspect ratio", () => {
    const v = zoomAt(full, { x: 100, y: 100 }, -200, W);
    expect(v.h / v.w).toBeCloseTo(full.h / full.w, 12);
  });
  it("keeps the point under the cursor fixed", () => {
    const p = { x: 150, y: 90 };
    const v = zoomAt(full, p, -200, W);
    // p sits at the same fraction of the view before and after
    expect((p.x - v.x) / v.w).toBeCloseTo((p.x - full.x) / full.w, 12);
    expect((p.y - v.y) / v.h).toBeCloseTo((p.y - full.y) / full.h, 12);
  });
  it("does not move with no wheel delta", () => {
    expect(zoomAt(full, { x: 10, y: 10 }, 0, W)).toEqual(full);
  });
});

describe("clientToUser", () => {
  // a 1200x300 box showing a 600x300 view: height-limited, letterboxed 300 px each side
  const r = { left: 0, top: 0, width: 1200, height: 300 };
  it("uses the larger of the two scales (meet)", () => {
    expect(unitsPerPixel(full, r)).toBe(1);
  });
  it("accounts for the letterbox", () => {
    expect(clientToUser(full, r, { clientX: 300, clientY: 0 })).toEqual({ x: 0, y: 0 });
    expect(clientToUser(full, r, { clientX: 900, clientY: 300 })).toEqual({ x: 600, y: 300 });
  });
});

describe("dragView", () => {
  const d = { cx: 100, cy: 100, vx: 0, vy: 0, s: 2, moved: false };
  it("ignores movement under 4 px so a click stays a click", () => {
    expect(dragView(d, 103, 100, full)).toBeNull();
  });
  it("pans opposite to the pointer, in user units", () => {
    expect(dragView(d, 110, 95, full)).toEqual({ x: -20, y: 10, w: W, h: 300 });
  });
  it("keeps following small moves once a drag has started", () => {
    expect(dragView({ ...d, moved: true }, 101, 100, full)).toEqual({ x: -2, y: 0, w: W, h: 300 });
  });
});

describe("strokeScale", () => {
  it("scales with the larger side of the view", () => {
    expect(strokeScale({ x: 0, y: 0, w: 440, h: 100 })).toBe(2);
  });
});
