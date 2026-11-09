import { describe, expect, it } from "vitest";
import clo from "/public/example/panel_seperated.dxf?raw";
import gcd from "/public/example/rand_1328ERLDIC_specification.json?raw";
import { parseDxf, parseSpec, type Pattern } from "@/core-algorithm/pattern";
import { patternGeometry, svgFrame } from "./geometry";
import { layoutLabels, type LabelEdge, type LabelObstacle, type PlacedLabel } from "./labelLayout";
import { placePanels } from "./render";
import { strokeScale } from "./viewport";

/** Labels for a whole pattern, built the way PatternSvg builds them at the default zoom. */
function patternLabels(p: Pattern, focusedIds: Set<string> = new Set()) {
  const placed = placePanels(p);
  const frame = svgFrame(placed);
  const geom = patternGeometry(placed, frame);
  const S = strokeScale({ x: 0, y: 0, w: frame.W, h: frame.H });
  const edges: LabelEdge[] = geom.flatMap((g) => g.edges.map((e, i) => ({
    id: `${g.name}#${i}`, mid: [e.mx, e.my], a: [e.ax, e.ay], b: [e.bx, e.by], center: [g.cx, g.cy],
    text: `${i} · ${e.arc.toFixed(1)}`, length: e.arc, focused: focusedIds.has(`${g.name}#${i}`),
  })));
  const obstacles: LabelObstacle[] = geom.map((g) =>
    ({ x: g.cx, y: g.cy, text: g.name, fontSize: 4 * S, baseline: "alphabetic" as const }));
  return { edges, obstacles, fontSize: 4.6 * S };
}

const boxOf = (l: PlacedLabel, text: string, fs: number) => {
  const w = text.length * fs * 0.6;
  return { x0: l.x - w / 2, x1: l.x + w / 2, y0: l.y - fs / 2, y1: l.y + fs / 2 };
};
const overlap = (p: ReturnType<typeof boxOf>, q: ReturnType<typeof boxOf>) =>
  p.x0 < q.x1 && q.x0 < p.x1 && p.y0 < q.y1 && q.y0 < p.y1;

const examples: Array<[string, () => Pattern]> = [
  ["example 1 (CLO DXF)", () => parseDxf(clo, "panel_seperated")],
  ["example 2 (GarmentCode)", () => parseSpec(JSON.parse(gcd), "rand_1328ERLDIC")],
];

describe("layoutLabels on the shipped examples", () => {
  it.each(examples)("%s: no two shown labels overlap", (_, load) => {
    const { edges, obstacles, fontSize } = patternLabels(load());
    const out = layoutLabels(edges, { fontSize, obstacles });
    const text = new Map(edges.map((e) => [e.id, e.text]));
    const shown = out.filter((l) => l.visible).map((l) => boxOf(l, text.get(l.id)!, fontSize));
    expect(shown.length).toBeGreaterThan(0);
    for (let i = 0; i < shown.length; i++)
      for (let j = i + 1; j < shown.length; j++) expect(overlap(shown[i], shown[j])).toBe(false);
  });

  it.each(examples)("%s: gives back one label per edge, in the input's order", (_, load) => {
    const { edges, obstacles, fontSize } = patternLabels(load());
    expect(layoutLabels(edges, { fontSize, obstacles }).map((l) => l.id)).toEqual(edges.map((e) => e.id));
  });

  it.each(examples)("%s: always shows the focused pair, even where labels crowd", (_, load) => {
    const all = patternLabels(load());
    const out = layoutLabels(all.edges, { fontSize: all.fontSize, obstacles: all.obstacles });
    // focus two labels the unfocused layout had to hide, if there are any, else any two
    const hidden = out.filter((l) => !l.visible).map((l) => l.id);
    const pick = new Set((hidden.length >= 2 ? hidden : out.map((l) => l.id)).slice(0, 2));
    const f = patternLabels(load(), pick);
    const focused = layoutLabels(f.edges, { fontSize: f.fontSize, obstacles: f.obstacles });
    for (const id of pick) expect(focused.find((l) => l.id === id)?.visible).toBe(true);
  });

  it.each(examples)("%s: is the same every time for the same input", (_, load) => {
    const { edges, obstacles, fontSize } = patternLabels(load());
    expect(layoutLabels(edges, { fontSize, obstacles })).toEqual(layoutLabels(edges, { fontSize, obstacles }));
  });
});

/* One edge on its own: where does its label go?  SVG y points down. */
const one = (a: [number, number], b: [number, number], center: [number, number],
             extra: Partial<LabelEdge> = {}): LabelEdge => ({
  id: "e", a, b, mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], center, text: "0 · 10.0",
  length: 10, focused: false, ...extra });

describe("layoutLabels on one edge", () => {
  const fontSize = 1;
  const at = (e: LabelEdge, obstacles: LabelObstacle[] = []) =>
    layoutLabels([e], { fontSize, obstacles })[0];

  it("puts a horizontal edge's label on the side away from the panel", () => {
    expect(at(one([0, 0], [10, 0], [5, -5]))).toEqual({ id: "e", x: 5, y: 0.8, visible: true });
    expect(at(one([0, 0], [10, 0], [5, 5]))).toEqual({ id: "e", x: 5, y: -0.8, visible: true });
  });
  it("puts a vertical edge's label on the side away from the panel", () => {
    expect(at(one([0, 0], [0, 10], [-5, 5]))).toMatchObject({ x: 0.8, y: 5 });
    expect(at(one([0, 0], [0, 10], [5, 5]))).toMatchObject({ x: -0.8, y: 5 });
  });
  it("does not care which way round the edge runs", () => {
    expect(at(one([10, 0], [0, 0], [5, -5]))).toMatchObject({ x: 5, y: 0.8 });
  });
  it("still picks the side away from a centre lying almost on the edge", () => {
    expect(at(one([0, 0], [10, 0], [5, -0.001]))).toMatchObject({ y: 0.8 });
    expect(at(one([0, 0], [10, 0], [5, 0.001]))).toMatchObject({ y: -0.8 });
  });
  it("falls back to the panel side when the outer side is taken", () => {
    const block: LabelObstacle = { x: 5, y: 0.8, text: "X", fontSize: 1, baseline: "middle" };
    expect(at(one([0, 0], [10, 0], [5, -5]), [block])).toMatchObject({ y: -0.8, visible: true });
  });
  it("hides a label with no free side, unless it is focused", () => {
    const blocks: LabelObstacle[] = [0.8, -0.8].map((y) => ({ x: 5, y, text: "X", fontSize: 1, baseline: "middle" }));
    expect(at(one([0, 0], [10, 0], [5, -5]), blocks).visible).toBe(false);
    expect(at(one([0, 0], [10, 0], [5, -5], { focused: true }), blocks)).toMatchObject({ y: 0.8, visible: true });
  });
  it("places focused labels first, then longer edges, then by order", () => {
    // three labels competing for the same two spots
    const same = (id: string, length: number, focused = false) =>
      one([0, 0], [10, 0], [5, -5], { id, length, focused });
    const out = layoutLabels([same("short", 3), same("long", 9), same("pick", 1, true)], { fontSize });
    expect(out.map((l) => [l.id, l.visible, l.y])).toEqual([
      ["short", false, 0], ["long", true, -0.8], ["pick", true, 0.8]]);
  });
});
