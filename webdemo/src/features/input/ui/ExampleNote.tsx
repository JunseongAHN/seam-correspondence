import { useAppSelector } from "@/app/hooks";
import { EXAMPLES } from "../logic";
import shared from "@/styles/shared.module.css";

/** What the loaded example is, and what its right pane is not; nothing for other inputs. */
export default function ExampleNote() {
  const example = useAppSelector((s) => s.workspace.example);
  if (!example) return null;
  return <p className={`${shared.note} ${shared.wide}`}>{EXAMPLES[example].note}</p>;
}
