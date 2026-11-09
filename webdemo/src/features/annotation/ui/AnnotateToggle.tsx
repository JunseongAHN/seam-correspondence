import { useAppDispatch, useAppSelector } from "@/app/hooks";
import { annotateToggled } from "@/app/workspaceSlice";
import { selectHasOwnGt } from "../logic";

/* Only an input with no stitch list needs one drawn. Offering it on an example
   that already has one invites a second, conflicting truth for the same
   garment -- so the button says why it is off rather than disappearing. */
export default function AnnotateToggle() {
  const hasOwnGt = useAppSelector(selectHasOwnGt);
  const annotate = useAppSelector((s) => s.workspace.annotate);
  const dispatch = useAppDispatch();
  return (
          <button disabled={hasOwnGt}
                  onClick={() => dispatch(annotateToggled())}>
            {hasOwnGt
              ? "ground truth — this example already has one"
              : `${annotate ? "leave" : "draw"} the ground truth by hand`}
          </button>
  );
}
