/* What App still wires by hand while the features move into the store: the error line
   and the 3D pane's choice.  Everything else is read from the store by the component
   that uses it. */
import { useAppSelector } from "@/app/hooks";
import { selectStats } from "@/app/workspaceSlice";
import { pickRightPane } from "@/features/preview3d";

export function useWorkspace() {
  const { result: meta, error: err, example, more, pane } = useAppSelector((s) => s.workspace);
  const stats = useAppSelector(selectStats);

  const exact = !!stats && stats.fp === 0 && stats.fn === 0;
  const right = pickRightPane({
    example: example ? pane.exampleRight : null,
    more,
    moreSim: more ? pane.moreSim ?? undefined : undefined,
    exact,
    stats,
  });

  return { loaded: meta !== null, err, right };
}
