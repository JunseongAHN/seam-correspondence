/* onnxruntime-web session + the hard assignment that stays outside the graph
   (autosew/metrics.py hard_assign_single: P' = (P + P^T)/2, row argmax + tau). */
import * as ort from "onnxruntime-web";
import { featureDim, type Tensors } from "./features";

const TAU = 0.4;             // cfg.tau_multi
/** Keep only stitches both edges agree on, so no edge lands in two of them.
    Matches cfg.hard_mode = "mutual"; set false for the paper's "union" rule. */
const ONE_TO_ONE = true;
let session: ort.InferenceSession | null = null;

export async function loadModel(url = `${import.meta.env.BASE_URL}model/autosew.onnx`) {
  if (!session) session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  return session;
}

export async function predict(t: Tensors): Promise<{ pairs: Set<string>; ms: number }> {
  const s = await loadModel();
  const M = t.M;
  const feeds: Record<string, ort.Tensor> = {
    // featureDim, never a literal: the shipped model has been 24 dims and is now 26,
    // and a stale width here is not a wrong answer, it is a thrown tensor-size error.
    x: new ort.Tensor("float32", t.x, [1, M, featureDim]),
    nbr: new ort.Tensor("int64", t.nbr, [1, M, 2]),
    mask: new ort.Tensor("bool", new Uint8Array(M).fill(1), [1, M]),
  };
  const t0 = performance.now();
  const out = await s.run(feeds);
  const ms = performance.now() - t0;

  const logP = out.logP.data as Float32Array;      // (1, M+1, M+1)
  const N = M + 1;
  // P' = 1/2 (P + P^T) on the exponentiated assignment
  const Ps = new Float64Array(N * N);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      Ps[i * N + j] = 0.5 * (Math.exp(logP[i * N + j]) + Math.exp(logP[j * N + i]));

  // each edge's best partner, or the dustbin (index M) if it is sewn to nothing
  const best = new Int32Array(M);
  for (let i = 0; i < M; i++) {
    let b = -1, bv = -Infinity;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const v = Ps[i * N + j];
      if (v > bv) { bv = v; b = j; }
    }
    best[i] = b;
  }

  const pairs = new Set<string>();
  if (ONE_TO_ONE) {
    /* Keep a pair only when both edges name each other.  Each row has exactly one
       argmax, so this is a matching: no edge can end up in two stitches.

       The alternative -- the paper's rule, and what this used to do -- also emits every
       partner scoring above tau, which is how a multi-edge stitch gets expressed.  But
       the training data contains no multi-edge stitches at all, so the model never
       learned when to use it; on a real garment the extra pairs are the model hedging,
       and every one of them is a false positive.  Measured on the CLO example, switching
       to mutual never lowered F1 and raised it for two of the five checkpoints. */
    for (let i = 0; i < M; i++) {
      const j = best[i];
      if (j < 0 || j >= M) continue;                // dustbin: this edge is unstitched
      if (best[j] === i) pairs.add(`${Math.min(i, j)}-${Math.max(i, j)}`);
    }
  } else {
    for (let i = 0; i < M; i++) {
      if (best[i] === M) continue;
      const add = (j: number) => pairs.add(`${Math.min(i, j)}-${Math.max(i, j)}`);
      add(best[i]);
      for (let j = 0; j < M; j++)
        if (j !== i && j !== best[i] && Ps[i * N + j] >= TAU) add(j);
    }
  }
  return { pairs, ms };
}
