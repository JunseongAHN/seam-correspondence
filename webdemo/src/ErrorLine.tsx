import { useAppSelector } from "@/app/hooks";
import shared from "@/styles/shared.module.css";

/** The page's one error line.  It is not any one feature's: a file that cannot be read, a
    model that cannot load and a saved ground truth that does not fit all end up here. */
export default function ErrorLine() {
  const error = useAppSelector((s) => s.workspace.error);
  if (!error) return null;
  return <div className={shared.err}>{error}</div>;
}
