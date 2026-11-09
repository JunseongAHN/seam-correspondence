/* Which 3D pane sits beside the pattern.  Here rather than in logic/ because it builds on
   the shared score (selectStats), which lives with the store; the decision itself is the
   pure pickRightPane. */
import { createSelector } from "@reduxjs/toolkit";
import { selectStats, type WorkspaceState } from "@/app/workspaceSlice";
import { pickRightPane } from "../logic";

type WithWorkspace = { workspace: WorkspaceState };

export const selectRightPane = createSelector(
  [(s: WithWorkspace) => s.workspace.example, (s: WithWorkspace) => s.workspace.more,
   (s: WithWorkspace) => s.workspace.pane, selectStats],
  (example, more, pane, stats) => pickRightPane({
    example: example ? pane.exampleRight : null,
    more,
    moreSim: more ? pane.moreSim ?? undefined : undefined,
    exact: !!stats && stats.fp === 0 && stats.fn === 0,
    stats,
  }));
