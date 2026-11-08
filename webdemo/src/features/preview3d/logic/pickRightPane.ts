/* What sits beside the pattern.  A held-out garment gets its ground-truth drape --
   red, and labelled as ground truth, because it is the only pane on the page that is
   not the model's output and a 3D shape is persuasive enough to be mistaken for one. */

export type RightPane =
  | { kind: "sim"; title: string; gt: false;
      props: { exact: boolean; fp: number; fn: number; ok?: number; file?: string; expectS?: number } }
  | { kind: "weld"; title: string; gt: false }
  | { kind: "gtdrape"; title: string; gt: true;
      props: { id: string; exact: boolean; fp: number; fn: number } };

export function pickRightPane({ example, more, moreSim, exact, stats }: {
  /** the loaded example's own pane, if an example is loaded */
  example: "sim" | "drape" | null;
  /** the loaded held-out garment, if one is */
  more: string | null;
  /** that garment's solver input, shipped only when the model gets it exactly right */
  moreSim: { file: string; expectS: number } | undefined;
  exact: boolean;
  stats: { ok: number; fp: number; fn: number } | null;
}): RightPane | null {
  if (example)
    return example === "sim"
      ? { kind: "sim", title: "assembled by the wasm solver · 3D", gt: false,
          props: { exact, fp: stats?.fp ?? 0, fn: stats?.fn ?? 0 } }
      : { kind: "weld", title: "CLO's own drape · 3D", gt: false };
  if (!more) return null;
  /* Exact prediction: it can actually be assembled, so offer the solve rather
     than a picture of the answer.  Anything else gets the ground truth's shape,
     in red, because its own stitching would tear the mesh. */
  return moreSim && exact
    ? { kind: "sim", title: "assembled by the wasm solver · 3D — the prediction, solved",
        gt: false,
        props: { exact: true, fp: 0, fn: 0, ok: stats?.ok ?? 0,
                 file: moreSim.file, expectS: moreSim.expectS } }
    : { kind: "gtdrape", title: "ground-truth drape · 3D", gt: true,
        props: { id: more, exact, fp: stats?.fp ?? 0, fn: stats?.fn ?? 0 } };
}
