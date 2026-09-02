/* onnxruntime-web session + the hard assignment that stays outside the graph
   (autosew/metrics.py hard_assign_single: P' = (P + P^T)/2, row argmax + tau). */
import * as ort from "onnxruntime-web";
import type { Tensors } from "./features";

const TAU = 0.4;             // cfg.tau_multi
let session: ort.InferenceSession | null = null;

export async function loadModel(url = `${import.meta.env.BASE_URL}model/autosew.onnx`) {
  if (!session) session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  return session;
}

export async function predict(t: Tensors): Promise<{ pairs: Set<string>; ms: number }> {
  const s = await loadModel();
  const M = t.M;
  const feeds: Record<string, ort.Tensor> = {
    x: new ort.Tensor("float32", t.x, [1, M, 24]),
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

  const pairs = new Set<string>();
  for (let i = 0; i < M; i++) {
    let best = -1, bestV = -Infinity;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const v = Ps[i * N + j];
      if (v > bestV) { bestV = v; best = j; }
    }
    if (best === M) continue;                       // dustbin wins -> edge is unstitched
    const add = (j: number) => pairs.add(`${Math.min(i, j)}-${Math.max(i, j)}`);
    add(best);
    for (let j = 0; j < M; j++)
      if (j !== i && j !== best && Ps[i * N + j] >= TAU) add(j);
  }
  return { pairs, ms };
}
