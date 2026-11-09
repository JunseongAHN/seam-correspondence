/* What App still wires by hand while the features move into the store one by one: the
   input row's loads and the 3D pane's choice.  Everything else is read from the store by
   the component that uses it. */
import { useCallback } from "react";
import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { openPattern, selectStats } from "@/app/workspaceSlice";
import { EXAMPLES, exampleGtUrl, MORE, moreSpecFile, moreSpecUrl,
         type ExampleKey } from "@/features/input";
import { pickRightPane } from "@/features/preview3d";

export function useWorkspace() {
  const dispatch = useAppDispatch();
  const { result: meta, busy, modelStatus, error: err, name, example, more, pane } =
    useAppSelector((s) => s.workspace);
  const stats = useAppSelector(selectStats);

  const runFile = useCallback((file: File) => {
    dispatch(openPattern({ source: "file", file }));
  }, [dispatch]);

  const runExample = useCallback((key: ExampleKey) => {
    const e = EXAMPLES[key];
    dispatch(openPattern({ source: "url", url: e.url, fileName: e.file, gtUrl: exampleGtUrl(key),
                           example: key, more: null, exampleRight: e.right, moreSim: null }));
  }, [dispatch]);

  const runMore = useCallback((id: string) => {
    dispatch(openPattern({ source: "url", url: moreSpecUrl(id), fileName: moreSpecFile(id),
                           gtUrl: null, example: null, more: id, exampleRight: null,
                           moreSim: MORE.find((g) => g.id === id)?.sim ?? null }));
  }, [dispatch]);

  const exact = !!stats && stats.fp === 0 && stats.fn === 0;
  const right = pickRightPane({
    example: example ? pane.exampleRight : null,
    more,
    moreSim: more ? pane.moreSim ?? undefined : undefined,
    exact,
    stats,
  });

  return {
    loaded: meta !== null, busy, loadingModel: modelStatus === "loading", err, name, example, more,
    right, runFile, runExample, runMore,
  };
}
