/* Zoom and pan.  Some pieces are 1.6 cm strips whose two long edges sit closer together
   than a comfortable click target, so without zoom they cannot be picked apart. */

/** the SVG viewBox */
export type View = { x: number; y: number; w: number; h: number };
/** the SVG element's box on screen, in client pixels */
export type Rect = { left: number; top: number; width: number; height: number };

/** stroke and label scale for a view, so strokes stay even as you zoom in */
export const strokeScale = (v: View) => Math.max(v.w, v.h) / 220;

/** user units per client pixel, under preserveAspectRatio="xMidYMid meet" */
export const unitsPerPixel = (v: View, r: Rect) => Math.max(v.w / r.width, v.h / r.height);

export function clientToUser(v: View, r: Rect, e: { clientX: number; clientY: number }) {
  const s = unitsPerPixel(v, r);
  return { x: v.x + (e.clientX - r.left - (r.width - v.w / s) / 2) * s,
           y: v.y + (e.clientY - r.top - (r.height - v.h / s) / 2) * s };
}

/** Zoom about the user-space point p, keeping it under the cursor.  The width is held
    between W/60 (deep enough to split a 1.6 cm strip) and 1.4 W (the whole pattern). */
export function zoomAt(v: View, p: { x: number; y: number }, deltaY: number, W: number): View {
  const k = Math.exp(deltaY * 0.0015);
  const w = Math.min(Math.max(v.w * k, W / 60), W * 1.4);
  const h = w * (v.h / v.w);
  return { x: p.x - ((p.x - v.x) * w) / v.w, y: p.y - ((p.y - v.y) * h) / v.h, w, h };
}

/* Panning stores the pointer position in CLIENT pixels and the scale as it was when the
   drag began.  Converting through the live viewBox instead would feed each update back
   into the next frame's conversion, which reads as stutter. */
export type Drag = { cx: number; cy: number; vx: number; vy: number; s: number; moved: boolean };

/** The view after the pointer moved to (clientX, clientY), or null while the press is
    still under 4 px from where it began and so may yet be a click. */
export function dragView(d: Drag, clientX: number, clientY: number, v: View): View | null {
  if (!d.moved && Math.hypot(clientX - d.cx, clientY - d.cy) < 4) return null;
  return { x: d.vx - (clientX - d.cx) * d.s, y: d.vy - (clientY - d.cy) * d.s, w: v.w, h: v.h };
}
