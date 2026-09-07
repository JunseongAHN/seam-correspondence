/* Everything the failure report is, minus how it is delivered.
 *
 * The CLI (tools/seam-report.ts) and the Worker (worker/index.ts) both produce the same
 * report from the same evaluation, so the model, the schema, the system prompt, the
 * condensing and the gate live here and neither owns them. Copying them into two places
 * is how the deployed page and the local run start quietly disagreeing.
 *
 * Nothing here touches the filesystem, the network or the environment, so it runs
 * unchanged in node and on Workers.
 */

export const MODEL = "claude-sonnet-5"; // current Sonnet id per docs.claude.com/en/docs/about-claude/models
export const DEBUG_MODEL = "claude-haiku-4-5"; // --model claude-haiku-4-5 while iterating: ~1/5 the price

/* The three causes behind the failures in this eval file, measured — see REPORT.md:
   shape mismatch is the open one (0 of 32 seams whose two edges do not interlock),
   arc is dim 4 being the chord when a seam matches the arc, and mirror is CLO splitting
   one seam into two edges across a panel's mirror axis. */
export const KNOWN_CAUSES = ["shape mismatch", "arc", "mirror"];

export type Pair = {
  edge_a: string; edge_b: string; pred: boolean; gt: boolean;
  len_a: number; len_b: number; curv_a: number; curv_b: number;
  shape_mismatch: number; shape_relation: string; p_pair: number;
  a_chose: string; p_a_chose: number; b_chose: string; p_b_chose: number;
};
export type EvalFile = { garment: string; f1: number; pairs: Pair[] };
export type Report = {
  summary: string;
  failure_categories: { category: string; count: number; example_pairs: string[]; likely_cause: string }[];
  next_actions: string[];
};
export type Usage = { input_tokens: number; output_tokens: number };

export const REPORT_TOOL = {
  name: "write_failure_report",
  description: "Structured failure analysis of seam-matching results.",
  input_schema: {
    type: "object",
    properties: {
      summary: {
        type: "string",
        description: "REQUIRED, write this field first: exactly 3 sentences a "
                   + "non-ML reader can follow, covering what fails and why.",
      },
      failure_categories: {
        type: "array",
        description:
          "Every MISS and FALSE_POSITIVE in the input must belong to exactly one category, "
          + "so the counts must add up to the total number of failures.",
        items: {
          type: "object",
          properties: {
            category: { type: "string" },
            count: { type: "integer" },
            example_pairs: { type: "array", items: { type: "string" }, maxItems: 3,
                             description: "At most 3; only pairs present in the input." },
            likely_cause: { type: "string" },
          },
          required: ["category", "count", "example_pairs", "likely_cause"],
        },
      },
      next_actions: { type: "array", items: { type: "string" }, minItems: 3, maxItems: 3,
                      description: "Exactly 3, ranked by expected F1 gain per hour." },
    },
    required: ["summary", "failure_categories", "next_actions"],
  },
} as const;

export const SYSTEM =
  "You are reviewing a seam-matching model for 2D garment patterns (edge-pair correspondence). " +
  "Group failures by geometric pattern, name a likely cause for each group, and propose next actions. " +
  "Fill in every field of the tool, including the summary -- omitting one is a failed response. " +
  "Be concrete; never invent pairs that are not in the input. " +
  "Every MISS and FALSE_POSITIVE listed must fall into exactly one category, and the counts must " +
  "sum to the stated failure count. " +
  "Ground every claim in the numbers given. The evidence that separates the cases is " +
  "shape_mismatch and what each edge chose instead: a MISS where an edge chose the dustbin at " +
  "high probability is the model refusing to sew it at all, which is a different failure from " +
  "one where a rival edge took it at a probability just above the true pair's. " +
  "Three causes are established for this pipeline and you must address all three BY NAME: " +
  "'shape mismatch' (edges that do not interlock -- read shape_mismatch directly), " +
  "'arc' length versus chord length (a seam matches the fabric sewn along; the lengths here " +
  "are already arc lengths), and 'mirror' axis artifacts (a CAD export splitting one seam into " +
  "two edges where a panel crosses its own mirror line). " +
  "If the data does not support one of them as a cause of THESE failures, say so plainly -- " +
  "'no evidence of X here, because ...' is the correct answer and inventing a role for it is not.";

/* Lengths and curvatures alone do not distinguish the failure modes, so a report built
   on them can only speculate. What decides each case is what the model chose instead,
   how sure it was, and whether the two edges interlock at all -- so those go in too, and
   the legend tells the reader how to read them. */
const LEGEND = [
  "Each failure is two lines. Field guide:",
  "  len            arc length in cm -- the fabric sewn along, not the chord.",
  "  curvature      signed sagitta of largest magnitude, as a fraction of the chord.",
  "                 0.00 means a straight edge; sign says which way it bows.",
  "  shape_mismatch how badly the two edges fail to interlock, over the best of the four",
  "                 ways one sagitta profile can lie against the other. 0.00 = they fit",
  "                 together; >0.50 = unrelated shapes. The bracket says which relation",
  "                 fitted best (+s as-is, -s negated, +rev reversed, -rev both).",
  "  p_pair         the model's probability for THIS pair.",
  "  chose          each edge's actual argmax partner and its probability. 'dustbin'",
  "                 means the model decided that edge is sewn to nothing at all.",
].join("\n");

export function failures(data: EvalFile): Pair[] {
  return data.pairs.filter((p) => p.pred !== p.gt);
}

export function condense(data: EvalFile): string {
  const fails = failures(data);
  const lines = fails.flatMap((p) => {
    const kind = p.gt && !p.pred ? "MISS" : "FALSE_POSITIVE";
    return [
      `${kind}: ${p.edge_a} <-> ${p.edge_b} | len ${p.len_a.toFixed(1)}/${p.len_b.toFixed(1)}`
      + ` | curvature ${p.curv_a.toFixed(2)}/${p.curv_b.toFixed(2)}`
      + ` | shape_mismatch ${p.shape_mismatch.toFixed(2)} (${p.shape_relation})`
      + ` | p_pair ${p.p_pair.toFixed(3)}`,
      `    ${p.edge_a} chose ${p.a_chose} p=${p.p_a_chose.toFixed(2)}`
      + ` ; ${p.edge_b} chose ${p.b_chose} p=${p.p_b_chose.toFixed(2)}`,
    ];
  });
  return `${LEGEND}\n\ngarment=${data.garment} f1=${data.f1.toFixed(3)}`
       + ` failures=${fails.length}\n${lines.join("\n")}`;
}

export function render(r: Report, usage: Usage, ms: number, model: string): string {
  const out = ["# Seam-matching failure report", "", r.summary, "", "## Failure categories", ""];
  for (const c of r.failure_categories) {
    out.push(`- **${c.category}** (${c.count}) — ${c.likely_cause}`);
    for (const ex of c.example_pairs.slice(0, 3)) out.push(`  - ${ex}`);
  }
  out.push("", "## Next actions (ranked)", "", ...r.next_actions.map((a, i) => `${i + 1}. ${a}`));
  out.push("", `_model=${model} · in=${usage.input_tokens} out=${usage.output_tokens} tokens · ${(ms / 1000).toFixed(1)}s_`);
  return out.join("\n");
}

export function gate(md: string, r: Report, nFailures: number): string[] {
  const problems: string[] = [];
  // DoD 1 -- the header claims the gate enforces this, so it has to actually check it.
  const covered = r.failure_categories.reduce((s, c) => s + c.count, 0);
  if (covered !== nFailures) problems.push(`categories cover ${covered} of ${nFailures} failures`);
  if (r.next_actions.length !== 3) problems.push("next_actions must be exactly 3");
  const missing = KNOWN_CAUSES.filter((c) => !md.toLowerCase().includes(c.toLowerCase()));
  if (missing.length) problems.push(`known causes not mentioned: ${missing.join(", ")}`);
  return problems;
}

/** The one Messages request this tool makes, as a plain body both callers can POST. */
export function requestBody(data: EvalFile, model: string) {
  return {
    model,
    max_tokens: 4000,
    system: SYSTEM,
    tools: [REPORT_TOOL],
    tool_choice: { type: "tool", name: REPORT_TOOL.name },
    messages: [{ role: "user", content: condense(data) }],
  };
}

/** Pull the report out of a Messages response, saying which field is missing if one is. */
export function reportFrom(resp: any): Report {
  if (resp.stop_reason === "max_tokens")
    throw new Error(`response hit max_tokens (${resp.usage?.output_tokens}); the tool call is `
                    + "truncated and its later fields are missing -- raise max_tokens");
  const block = (resp.content ?? []).find((b: any) => b.type === "tool_use");
  if (!block) throw new Error("no tool_use block in response");
  const report = block.input as Report;
  for (const k of ["summary", "failure_categories", "next_actions"] as const)
    if (report[k] === undefined) throw new Error(`the model omitted "${k}" from the tool call`);
  return report;
}
