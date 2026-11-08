import { EXAMPLES, type ExampleKey } from "../logic";

/** What the loaded example is, and what its right pane is not. */
export default function ExampleNote({ example }: { example: ExampleKey }) {
  return <p className="note wide">{EXAMPLES[example].note}</p>;
}
