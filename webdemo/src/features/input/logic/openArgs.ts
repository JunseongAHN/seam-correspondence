/* What a load needs to know about an example, read off input's catalogue so the store
   never has to look it up: where to fetch it, its ground truth if one is checked in, and
   what the 3D pane should show beside it.  The shape is openPattern's argument. */
import { EXAMPLES, exampleGtUrl, MORE, moreSpecFile, moreSpecUrl, type ExampleKey } from "./examples";

export function exampleOpenArg(key: ExampleKey) {
  const e = EXAMPLES[key];
  return { source: "url" as const, url: e.url, fileName: e.file, gtUrl: exampleGtUrl(key),
           example: key, more: null, exampleRight: e.right, moreSim: null };
}

export function moreOpenArg(id: string) {
  return { source: "url" as const, url: moreSpecUrl(id), fileName: moreSpecFile(id),
           gtUrl: null, example: null, more: id, exampleRight: null,
           moreSim: MORE.find((g) => g.id === id)?.sim ?? null };
}
