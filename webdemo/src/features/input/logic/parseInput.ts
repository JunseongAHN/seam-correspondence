import { parseDxf, parseSpec } from "@/core-algorithm/pattern";

/** An imported file's text -> Pattern, by its extension.  A DXF carries no stitches, so
    there is nothing to score it against.

    A parser's own error ("Cannot convert undefined or null to object", "EOF group not
    read") says nothing about which file or what it was expected to be, so failures are
    rethrown naming both. */
export function parseInput(text: string, fileName: string) {
  const isDxf = /\.dxf$/i.test(fileName);
  const kind = isDxf ? "a DXF" : "a pattern specification (JSON)";
  let pattern;
  try {
    pattern = isDxf
      ? parseDxf(text, fileName.replace(/\.dxf$/i, ""))
      : parseSpec(JSON.parse(text), fileName.replace("_specification.json", ""));
  } catch (e: any) {
    throw new Error(`could not read ${fileName} as ${kind}: ${e?.message ?? e}`);
  }
  if (!pattern.panels.length) throw new Error(`no panels found in ${fileName}`);
  return { pattern, isDxf };
}
