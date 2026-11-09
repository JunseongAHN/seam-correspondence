/* What the 2D view shows: edge labels, and which kinds of stitch line.  Its toggles and the
   pattern sit in different places on the page, so App calls this and passes it down.  It
   survives loading another pattern, as these switches always have. */
import { useState } from "react";
import type { Show } from "../ui/LineToggles";

export function useStitchViewState() {
  const [labels, setLabels] = useState(true);
  const [show, setShow] = useState<Show>({ correct: true, fp: true, fn: true, gt: false });
  return { labels, setLabels, show, setShow };
}
