import DropZone from "./DropZone";
import MoreGarments from "./MoreGarments";

/** Getting a pattern in: the drop zone with its examples, and the held-out garments. */
export default function InputSection() {
  return (
    <>
      <DropZone />

      <MoreGarments />
    </>
  );
}
