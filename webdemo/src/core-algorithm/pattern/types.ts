/* The pattern shape every reader produces and everything downstream consumes. */

export const KT = { STRAIGHT: 0, CIRCLE: 1, QUADRATIC: 2, CUBIC: 3, BSPLINE: 4, UNKNOWN: 5 };

export type Pt = [number, number];
export interface Edge {
  panel: string; idxInPanel: number;      // ORIGINAL json index; stitches refer to this
  start: Pt; end: Pt; kt: number; kparams: number[];
  poly?: Pt[];          // DXF input: the sampled boundary, used instead of refitting
}
export interface Panel {
  name: string; orderIdx: number; edges: Edge[]; nEdgesRaw: number; reversed: boolean;
  rawVerts: Pt[]; rawEdges: any[];        // kept for drawing in the original frame
  translation: number[]; rotation: number[];
}
export interface Pattern {
  name: string; panels: Panel[];
  stitches: Array<Array<[string, number]>>;
}
