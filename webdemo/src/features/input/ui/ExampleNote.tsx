import { EXAMPLES, type ExampleKey } from "../logic";
import shared from "@/styles/shared.module.css";

/** What the loaded example is, and what its right pane is not. */
export default function ExampleNote({ example }: { example: ExampleKey }) {
  return <p className={`${shared.note} ${shared.wide}`}>{EXAMPLES[example].note}</p>;
}
