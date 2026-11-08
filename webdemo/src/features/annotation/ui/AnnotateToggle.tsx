/* Only an input with no stitch list needs one drawn. Offering it on an example
   that already has one invites a second, conflicting truth for the same
   garment -- so the button says why it is off rather than disappearing. */
export default function AnnotateToggle({ hasOwnGt, annotate, onToggle }: {
  hasOwnGt: boolean;
  annotate: boolean;
  onToggle: () => void;
}) {
  return (
          <button disabled={hasOwnGt}
                  onClick={onToggle}>
            {hasOwnGt
              ? "ground truth — this example already has one"
              : `${annotate ? "leave" : "draw"} the ground truth by hand`}
          </button>
  );
}
