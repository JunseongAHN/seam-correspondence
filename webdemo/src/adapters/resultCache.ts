/* The parts of a prediction the Redux store cannot hold.  The features as given to the
   model are a Float32Array and the model's scores carry a function, neither of which is
   serializable; the parsed pattern is plain data but 15-90 KB, and the store would copy
   and diff it on every action.  They live here, and the store keeps only the id. */
import type { Scores } from "@/core-algorithm/inference";
import type { Pattern } from "@/core-algorithm/pattern";

export type ResultObjects = {
  pattern: Pattern;
  keys: Array<[string, number]>;   // node -> (panel, original edge idx)
  x: Float32Array;                 // the features as given to the model, for the eval export
  scores: Scores;                  // and what it made of them, for the same reason
};

const results = new Map<string, ResultObjects>();
let issued = 0;

/** Keep a result; returns the id the store refers to it by. */
export function putResult(r: ResultObjects): string {
  const id = `result-${++issued}`;
  results.set(id, r);
  return id;
}

export function getResult(id: string | null | undefined): ResultObjects | undefined {
  return id ? results.get(id) : undefined;
}

/** Forget every result but `id`.  The page shows one result at a time, so once the store
    points at a new one the others are garbage -- including any a stale load put here. */
export function retainOnly(id: string | null): void {
  for (const k of [...results.keys()]) if (k !== id) results.delete(k);
}
