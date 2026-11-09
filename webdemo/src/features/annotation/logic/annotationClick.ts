/** What a click on edge n does while drawing: the next selection, and at most one stitch
    to add or remove.  Click one edge then another to sew them together; click a paired
    edge to undo it.  Pure, so the edgeClicked reducer in app/workspaceSlice only applies it. */
export function annotationClick(selected: number | null, userPairs: Set<string>, n: number):
    { selected: number | null; change: { key: string; remove: boolean } | null } {
  if (selected === n) return { selected: null, change: null };        // same edge: deselect
  if (selected === null) {
    // already sewn?  clicking it again undoes that stitch
    const owner = [...userPairs].find((k) => k.split("-").map(Number).includes(n));
    if (owner) return { selected: null, change: { key: owner, remove: true } };
    return { selected: n, change: null };
  }
  const key = `${Math.min(selected, n)}-${Math.max(selected, n)}`;
  return { selected: null, change: { key, remove: userPairs.has(key) } };
}
