// BASE_URL, not an absolute path: the GitHub Pages build is served from /<repo>/.
const B = import.meta.env.BASE_URL;

export const EXAMPLES = {
  clo: {
    label: "1. CLO tutorial example",
    file: "panel_seperated.dxf",
    url: `${B}example/panel_seperated.dxf`,
    right: "drape" as const,
    // A DXF carries no stitch list.  This one is scorable only because the ground truth
    // was drawn by hand in this very demo and exported -- 20 stitches over 40 of the 48
    // edges -- which is why the CLO numbers in the report exist at all.
    gtUrl: `${B}example/panel_seperated_gt.json`,
    note: "10 pieces exported from CLO. A DXF carries no stitch list, so the ground truth "
        + "here was drawn by hand in this demo — 20 stitches over 40 of the 48 edges — and "
        + "the prediction is scored against it. The right pane is CLO's own drape, not a "
        + "simulation of this prediction.",
  },
  gcd: {
    label: "2. GarmentCode test data",
    file: "rand_1328ERLDIC_specification.json",
    url: `${B}example/rand_1328ERLDIC_specification.json`,
    right: "sim" as const,
    note: "A held-out GarmentCodeData garment, so the prediction can be scored.",
  },
};
export type ExampleKey = keyof typeof EXAMPLES;

/** The checked-in ground truth for an example that has one, else null. */
export function exampleGtUrl(key: ExampleKey): string | null {
  const e = EXAMPLES[key];
  return "gtUrl" in e ? e.gtUrl : null;
}

/* More held-out garments, chosen by scoring a sample of the test split with the model
   this demo ships and taking a spread rather than a highlight reel.  The first one it
   gets completely wrong; two it gets exactly right. */
/* `sim` is the solver input, shipped only for the garments the model gets exactly
   right: the dump's constraints are vertex pairs built from the ground truth and it
   carries no edge->vertex mapping, so a wrong stitching cannot honestly be assembled.
   expectS is measured in a real browser, not guessed: 25.8 s and 109.4 s, both with
   mono violations 0 and energies matching the native solve to the printed digits. */
export const MORE: Array<{ id: string; f1: string; what: string;
                           sim?: { file: string; expectS: number } }> = [
  { id: "rand_JYO1DHSFGH", f1: "0.00", what: "4 panels, 4 stitches — none found" },
  { id: "rand_I88DFY2AKV", f1: "0.76", what: "8 panels, 24 stitches" },
  { id: "rand_E3IN9RH60H", f1: "0.94", what: "14 panels, 44 stitches" },
  { id: "rand_3C7X2I6WQ7", f1: "1.00", what: "6 panels, 24 stitches — exact",
    sim: { file: "rand_3C7X2I6WQ7.bin", expectS: 26 } },
  { id: "rand_GE9NBC1HFY", f1: "1.00", what: "14 panels, 48 stitches — exact",
    sim: { file: "rand_GE9NBC1HFY.bin", expectS: 110 } },
];

export const moreSpecFile = (id: string) => `${id}_specification.json`;
export const moreSpecUrl = (id: string) => `${B}example/${moreSpecFile(id)}`;
