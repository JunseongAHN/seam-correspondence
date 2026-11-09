/* The 2D pattern with its stitching drawn on it.

   Three things share this view.

   Prediction, coloured: the two edges of a predicted pair share a colour, and a thin line
   joins their midpoints so a pair stays readable when the two colours sit far apart.
   Clicking an edge focuses its correspondence -- that one is drawn thick and every other
   pair drops to grey -- which is the only way to read a crowded pattern.

   Scoring, when the input carries ground truth (a specification.json does, a DXF does
   not): correct / false positive / false negative.

   Annotation, when `annotate` is on: click one edge then another to sew them together by
   hand.  A DXF has no stitch list, so this is how ground truth gets made for a real
   garment -- and once it exists the prediction can actually be scored against it. */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getResult } from "@/adapters/resultCache";
import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { edgeClicked, selectPred, selectTruth, selectUserPairs } from "@/app/workspaceSlice";
import {
  clientToUser, dragView, edgeStyle, focusNodes, isLit, layoutLabels, midpointsByNode, nodeIndex,
  pairGroups, pairPartners, patternGeometry, placePanels, stitchIndexByEdge, strokeScale,
  svgFrame, unitsPerPixel, zoomAt, type Drag, type LabelEdge, type LabelObstacle, type Placed,
  type View,
} from "../logic";
import type { Pattern } from "@/core-algorithm/pattern";
import s from "./PatternSvg.module.css";

/** The loaded result's pattern, or nothing before one is loaded.  Only `height` comes from
    outside: the same view is taller on its own than beside the 3D pane. */
export default function PatternSvg({ height }: { height?: string }) {
  const meta = useAppSelector((st) => st.workspace.result);
  const annotate = useAppSelector((st) => st.workspace.annotate);
  const selected = useAppSelector((st) => st.workspace.selected);
  const userPairs = useAppSelector(selectUserPairs);
  const pred = useAppSelector(selectPred);
  const truth = useAppSelector(selectTruth);
  const show = useAppSelector((st) => st.stitchView.show);
  const labels = useAppSelector((st) => st.stitchView.labels);
  const dispatch = useAppDispatch();
  const onEdgeClick = useCallback((n: number) => { dispatch(edgeClicked(n)); }, [dispatch]);
  // the pattern and its node keys are in the result cache; the layout is drawn from them once
  const objects = getResult(meta?.id);
  const placed = useMemo(() => (objects ? placePanels(objects.pattern) : null), [objects]);
  if (!objects || !placed) return null;
  return <PatternView pattern={objects.pattern} placed={placed} keys={objects.keys} pred={pred}
                      gt={truth} show={show} height={height} annotate={annotate} labels={labels}
                      userPairs={userPairs} selected={selected} onEdgeClick={onEdgeClick} />;
}

type SvgProps = {
  pattern: Pattern;
  placed: Placed[];
  keys: Array<[string, number]>;
  pred: Set<string>;
  gt: Set<string>;
  show?: { correct: boolean; fp: boolean; fn: boolean; gt: boolean };
  height?: string;
  /** hand-annotation mode */
  annotate?: boolean;
  /** show "index · arc length" on every edge */
  labels?: boolean;
  userPairs?: Set<string>;          // "a-b" with a < b, node indices
  selected?: number | null;
  onEdgeClick?: (node: number) => void;
};

function PatternView({
  pattern, placed, keys, pred, gt, show, height,
  annotate = false, labels = true, userPairs, selected = null, onEdgeClick,
}: SvgProps) {
  const [focus, setFocus] = useState<number | null>(null);
  const scored = gt.size > 0 && !annotate;

  /* Focus is a node INDEX, and the same index means a different edge in the next
     garment.  Loading one while a focus is held would otherwise open it with most of
     the pattern greyed out and the first click merely clearing that -- which reads
     exactly like clicking an edge does nothing. */
  useEffect(() => { setFocus(null); }, [keys]);

  /* In the scored view colour comes from the ground-truth stitch index, not from the
     prediction, so there are no prediction groups to draw. */
  const group = useMemo(
    () => (!annotate && scored ? new Map<number, number>()
                               : pairGroups(annotate ? (userPairs ?? new Set<string>()) : pred)),
    [pred, userPairs, annotate, scored]);

  const node = useMemo(() => nodeIndex(keys), [keys]);

  const frame = useMemo(() => svgFrame(placed), [placed]);
  const { W, H } = frame;

  const [view, setView] = useState<View | null>(null);
  const v = view ?? { x: 0, y: 0, w: W, h: H };
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const S = strokeScale(v);

  /* React registers wheel listeners as passive, so preventDefault inside onWheel is
     ignored and the page scrolls along with the zoom.  Block it with a native listener. */
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const stop = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", stop, { passive: false });
    return () => el.removeEventListener("wheel", stop);
  }, []);

  const onWheel = (e: React.WheelEvent) => {
    const el = svgRef.current;
    if (!el) return;
    setView(zoomAt(v, clientToUser(v, el.getBoundingClientRect(), e), e.deltaY, W));
  };

  const geom = useMemo(() => patternGeometry(placed, frame), [placed, frame]);
  const mid = useMemo(() => midpointsByNode(geom, node), [geom, node]);

  /* Which nodes a click keeps lit.  In the scored view `group` is deliberately empty --
     colour comes from the ground-truth stitch index there, not from the prediction -- so
     focus has to be read off the pairings this view actually draws, or clicking an edge in
     a scored pattern dims nothing at all. */
  const partners = useMemo(
    () => pairPartners(annotate ? (userPairs ?? []) : scored ? [...pred, ...gt] : pred),
    [pred, gt, userPairs, annotate, scored]);

  const focusSet = useMemo(() => focusNodes(focus, group, partners), [focus, group, partners]);
  const lit = (a: number, b: number) => isLit(focusSet, a, b);

  /* Edge labels placed so they keep clear of each other and of the panel names.  Their
     size in user units follows S, so a zoom lays them out again -- and a label hidden for
     want of room comes back once there is room. */
  const placedLabels = useMemo(() => {
    if (!labels) return new Map<string, { x: number; y: number; visible: boolean }>();
    const edges: LabelEdge[] = geom.flatMap((g) => g.edges.map((e, i) => {
      const ni = node.get(`${g.name}#${i}`);
      return { id: `${g.name}#${i}`, mid: [e.mx, e.my], a: [e.ax, e.ay], b: [e.bx, e.by],
               center: [g.cx, g.cy], text: `${i} · ${e.arc.toFixed(1)}`, length: e.arc,
               focused: focusSet !== null && ni !== undefined && focusSet.has(ni) };
    }));
    const obstacles: LabelObstacle[] = geom.flatMap((g) => [
      { x: g.cx, y: g.cy, text: g.name, fontSize: 4 * S, baseline: "alphabetic" as const },
      ...(g.blowUp ? [{ x: g.cx, y: g.cy + 4.6 * S, text: `width ×${g.blowUp.toFixed(1)}, not to scale`,
                        fontSize: 3.1 * S, baseline: "alphabetic" as const }] : []),
    ]);
    return new Map(layoutLabels(edges, { fontSize: 4.6 * S, obstacles }).map((l) => [l.id, l]));
  }, [labels, geom, node, S, focusSet]);

  const stitchOf = useMemo(
    () => (annotate ? new Map<string, number>() : stitchIndexByEdge(pattern, gt, keys)),
    [annotate, pattern, gt, keys]);

  const line = (a: number, b: number, cls: string, key: string, w?: number) => {
    const m1 = mid.get(a), m2 = mid.get(b);
    if (!m1 || !m2) return null;
    return <line key={key} className={cls} x1={m1[0]} y1={m1[1]} x2={m2[0]} y2={m2[1]}
                 strokeWidth={w === undefined ? undefined : w * S} />;
  };

  return (
    <svg ref={svgRef} className={`${s.canvas}${annotate ? ` ${s.annotating}` : ""}`}
         style={height ? { height } : undefined}
         viewBox={`${v.x} ${v.y} ${v.w} ${v.h}`} preserveAspectRatio="xMidYMid meet"
         onWheel={onWheel}
         onPointerDown={(e) => {
           if (e.button !== 0) return;
           (e.target as Element).setPointerCapture?.(e.pointerId);
           drag.current = { cx: e.clientX, cy: e.clientY, vx: v.x, vy: v.y,
                            s: unitsPerPixel(v, e.currentTarget.getBoundingClientRect()),
                            moved: false };
         }}
         onPointerMove={(e) => {
           const d = drag.current;
           if (!d) return;
           const next = dragView(d, e.clientX, e.clientY, v);
           if (!next) return;
           d.moved = true;
           setView(next);
         }}
         onPointerUp={(e) => {
           (e.target as Element).releasePointerCapture?.(e.pointerId);
           setTimeout(() => { drag.current = null; }, 0);
         }}
         onPointerCancel={() => { drag.current = null; }}
         onDoubleClick={() => setView(null)}
         onClick={() => { if (!annotate && !drag.current?.moved) setFocus(null); }}>
      {geom.map((g) => (
        <g key={g.name}>
          {g.edges.map((e, i) => {
            const key = `${g.name}#${i}`;
            const ni = node.get(key);
            const si = stitchOf.get(key) ?? (ni === undefined ? undefined : group.get(ni));
            const isSel = annotate && ni !== undefined && ni === selected;
            const inFocus = focusSet !== null && ni !== undefined && focusSet.has(ni);
            const dim = focusSet !== null && !inFocus;
            const { color, width } = edgeStyle({ selected: isSel, stitch: si, inFocus, dim,
                                                 annotate, scored });
            return (
              <g key={i}>
                <path d={e.d} stroke={color} strokeWidth={width * S} fill="none"
                      strokeLinecap="round" />
                {/* a fat invisible copy, so a thin edge is still easy to hit */}
                <path d={e.d} stroke="transparent" strokeWidth={2.2 * S} fill="none"
                      strokeLinecap="round"
                      style={{ cursor: annotate || si !== undefined
                                       || (ni !== undefined && partners.get(ni)?.size)
                                       ? "pointer" : "default" }}
                      onClick={(ev) => {
                        ev.stopPropagation();
                        if (ni === undefined || drag.current?.moved) return;   // that was a pan
                        if (annotate) { onEdgeClick?.(ni); return; }
                        if (si === undefined && !partners.get(ni)?.size) return;
                        setFocus((f) => (f !== null && focusSet?.has(ni) ? null : ni));
                      }} />
              </g>
            );
          })}
        </g>
      ))}

      {annotate && [...(userPairs ?? [])].map((k) => {
        const [a, b] = k.split("-").map(Number);
        return line(a, b, s.userline, `u${k}`, 1.2);
      })}

      {!annotate && !scored && [...pred].map((k) => {
        const [a, b] = k.split("-").map(Number);
        if (!lit(a, b)) return null;
        return line(a, b, s.predline, `x${k}`, focusSet !== null ? 1.4 : 0.45);
      })}

      {scored && show && (() => {
        const w = focusSet === null ? undefined : 1.4;
        const draw = (k: string, cls: string, pfx: string) => {
          const [a, b] = k.split("-").map(Number);
          return lit(a, b) ? line(a, b, cls, `${pfx}${k}`, w) : null;
        };
        return (
          <>
            {show.gt && [...gt].map((k) => draw(k, s.gtline, "g"))}
            {show.correct && [...pred].filter((k) => gt.has(k)).map((k) => draw(k, s.ok, "c"))}
            {show.fp && [...pred].filter((k) => !gt.has(k)).map((k) => draw(k, s.fp, "p"))}
            {show.fn && [...gt].filter((k) => !pred.has(k)).map((k) => draw(k, s.fn, "n"))}
          </>
        );
      })()}

      {/* labels last, so no edge or stitch line drawn after them can cover them */}
      <g className="labels">
        {geom.map((g) => (
          <g key={g.name}>
            {labels && g.edges.map((e, i) => {
              const at = placedLabels.get(`${g.name}#${i}`);
              if (!at?.visible) return null;    // no room for it at this zoom
              return (
                <text key={i} x={at.x} y={at.y} dominantBaseline="middle" className={s.elabel}
                      style={{ fontSize: 4.6 * S }}>
                  {i} · {e.arc.toFixed(1)}
                </text>
              );
            })}
            <text x={g.cx} y={g.cy} className={s.plabel} style={{ fontSize: 4 * S }}>{g.name}</text>
            {g.blowUp && (
              <text x={g.cx} y={g.cy + 4.6 * S} className={`${s.plabel} ${s.scalenote}`}
                    style={{ fontSize: 3.1 * S }}>
                width ×{g.blowUp.toFixed(1)}, not to scale
              </text>
            )}
          </g>
        ))}
      </g>
    </svg>
  );
}
