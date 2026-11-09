/* Derived values only annotation reads.  logic/ may not import the store, so the state is
   typed by the little of it a selector looks at; the store's own state satisfies it. */

type WithResultGt = { workspace: { result: { gt: readonly string[] } | null } };

/** The input brought its own stitch list -- a specification, or the example whose
    hand-drawn truth is checked in -- so there is nothing to draw. */
export const selectHasOwnGt = (s: WithResultGt): boolean => (s.workspace.result?.gt.length ?? 0) > 0;
