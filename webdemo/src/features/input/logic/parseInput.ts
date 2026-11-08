import { parseDxf, parseSpec } from "@/core-algorithm/pattern";

/** An imported file's text -> Pattern, by its extension.  A DXF carries no stitches, so
    there is nothing to score it against. */
export function parseInput(text: string, fileName: string) {
  const isDxf = /\.dxf$/i.test(fileName);
  const pattern = isDxf
    ? parseDxf(text, fileName.replace(/\.dxf$/i, ""))
    : parseSpec(JSON.parse(text), fileName.replace("_specification.json", ""));
  return { pattern, isDxf };
}
