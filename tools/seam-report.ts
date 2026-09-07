/**
 * tools/seam-report.ts — raw seam-matching eval JSON → reviewer-readable failure report (Markdown).
 *
 * Run locally only (the API key must never reach the browser or git):
 *   export ANTHROPIC_API_KEY=sk-ant-...      # or put it in api.env (git-ignored)
 *   npm i -D @anthropic-ai/sdk tsx
 *   npm run report -- eval/real_dxf_01.json --out reports/real_dxf_01.md
 *
 * --model overrides the model for a run.  Iterate on the cheapest one and keep the
 * default for anything whose output is being kept.
 *
 * Definition of done (the gate at the bottom enforces it):
 *   1. every MISS / FALSE_POSITIVE is assigned to a category
 *   2. the three root causes we already found are mentioned
 *   3. exactly 3 next actions, ranked
 */
import fs from "node:fs";
import Anthropic from "@anthropic-ai/sdk";

const MODEL = "claude-sonnet-5";      // current Sonnet id per docs.claude.com/en/docs/about-claude/models
const DEBUG_MODEL = "claude-haiku-4-5"; // --model claude-haiku-4-5 while iterating: ~1/5 the price
/* The three causes behind the failures in this eval file, measured — see REPORT.md:
   shape mismatch is the open one (0 of 32 seams whose two edges do not interlock),
   arc is dim 4 being the chord when a seam matches the arc, and mirror is CLO splitting
   one seam into two edges across a panel's mirror axis. */
const KNOWN_CAUSES = ["shape mismatch", "arc", "mirror"];

type Pair = { edge_a: string; edge_b: string; pred: boolean; gt: boolean; len_a: number; len_b: number; curv_a: number; curv_b: number };
type EvalFile = { garment: string; f1: number; pairs: Pair[] };
type Report = {
  summary: string;
  failure_categories: { category: string; count: number; example_pairs: string[]; likely_cause: string }[];
  next_actions: string[];
};

const reportTool: Anthropic.Tool = {
  name: "write_failure_report",
  description: "Structured failure analysis of seam-matching results.",
  input_schema: {
    type: "object",
    properties: {
      summary: { type: "string",
                 description: "REQUIRED, write this field first: exactly 3 sentences a "
                            + "non-ML reader can follow, covering what fails and why." },
      failure_categories: {
        type: "array",
        description:
          "Every MISS and FALSE_POSITIVE in the input must belong to exactly one category, " +
          "so the counts must add up to the total number of failures.",
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
};

function condense(data: EvalFile): string {
  const fails = data.pairs.filter((p) => p.pred !== p.gt);
  const lines = fails.map((p) => {
    const kind = p.gt && !p.pred ? "MISS" : "FALSE_POSITIVE";
    return `${kind}: ${p.edge_a} <-> ${p.edge_b} | len ${p.len_a.toFixed(1)}/${p.len_b.toFixed(1)} | curvature ${p.curv_a.toFixed(2)}/${p.curv_b.toFixed(2)}`;
  });
  return `garment=${data.garment} f1=${data.f1.toFixed(3)} failures=${fails.length}\n${lines.join("\n")}`;
}

function render(r: Report, usage: Anthropic.Usage, ms: number, model: string): string {
  const out = ["# Seam-matching failure report", "", r.summary, "", "## Failure categories", ""];
  for (const c of r.failure_categories) {
    out.push(`- **${c.category}** (${c.count}) — ${c.likely_cause}`);
    for (const ex of c.example_pairs.slice(0, 3)) out.push(`  - ${ex}`);
  }
  out.push("", "## Next actions (ranked)", "", ...r.next_actions.map((a, i) => `${i + 1}. ${a}`));
  out.push("", `_model=${model} · in=${usage.input_tokens} out=${usage.output_tokens} tokens · ${(ms / 1000).toFixed(1)}s_`);
  return out.join("\n");
}

function gate(md: string, r: Report, nFailures: number): string[] {
  const problems: string[] = [];
  // DoD 1 -- the header claims the gate enforces this, so it has to actually check it.
  const covered = r.failure_categories.reduce((s, c) => s + c.count, 0);
  if (covered !== nFailures) problems.push(`categories cover ${covered} of ${nFailures} failures`);
  if (r.next_actions.length !== 3) problems.push("next_actions must be exactly 3");
  const missing = KNOWN_CAUSES.filter((c) => !md.toLowerCase().includes(c.toLowerCase()));
  if (missing.length) problems.push(`known causes not mentioned: ${missing.join(", ")}`);
  return problems;
}

async function main() {
  const [input, ...rest] = process.argv.slice(2);
  const out = rest.includes("--out") ? rest[rest.indexOf("--out") + 1] : "report.md";
  const model = rest.includes("--model") ? rest[rest.indexOf("--model") + 1] : MODEL;
  void DEBUG_MODEL;
  // api.env is the documented place for the key and it is git-ignored; node reads it
  // natively, so this needs no dependency. A shell export works just as well.
  for (const f of ["api.env", ".env"]) {
    try { process.loadEnvFile(f); } catch { /* absent is fine */ }
  }
  if (!process.env.ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY is not set");

  const data: EvalFile = JSON.parse(fs.readFileSync(input, "utf8"));
  const client = new Anthropic(); // reads ANTHROPIC_API_KEY from env

  const t0 = Date.now();
  const resp = await client.messages.create({
    model,
    max_tokens: 4000,
    system:
      "You are reviewing a seam-matching model for 2D garment patterns (edge-pair correspondence). " +
      "Group failures by geometric pattern, name a likely cause for each group, and propose next actions. " +
      "Fill in every field of the tool, including the summary -- omitting one is a failed response. " +
      "Be concrete; never invent pairs that are not in the input. " +
      "Every MISS and FALSE_POSITIVE listed must fall into exactly one category, and the counts must " +
      "sum to the stated failure count. " +
      "Three causes are already established for this pipeline and your analysis must engage with them " +
      "by name where the evidence supports it: 'shape mismatch' (a concave edge eased onto a convex one, " +
      "so the two sagitta values disagree in sign or magnitude even when the lengths match), " +
      "'arc' length versus chord length (a seam matches the fabric sewn along, not the straight line " +
      "between corners), and 'mirror' axis artifacts (a CAD export splitting one seam into two edges " +
      "where a panel crosses its own mirror line).",
    tools: [reportTool],
    tool_choice: { type: "tool", name: "write_failure_report" },
    messages: [{ role: "user", content: condense(data) }],
  });
  if (resp.stop_reason === "max_tokens")
    throw new Error(`response hit max_tokens (${resp.usage.output_tokens}); the tool call is `
                    + "truncated and its later fields are missing -- raise max_tokens");
  const block = resp.content.find((b) => b.type === "tool_use");
  if (!block || block.type !== "tool_use") throw new Error("no tool_use block in response");
  const report = block.input as Report;
  for (const k of ["summary", "failure_categories", "next_actions"] as const)
    if (report[k] === undefined) throw new Error(`the model omitted "${k}" from the tool call`);

  const md = render(report, resp.usage, Date.now() - t0, model);
  fs.mkdirSync(out.substring(0, out.lastIndexOf("/")) || ".", { recursive: true });
  fs.writeFileSync(out, md);
  console.log(md);

  const problems = gate(md, report, data.pairs.filter((p) => p.pred !== p.gt).length);
  if (problems.length) {
    console.error("\nGATE FAILED:\n- " + problems.join("\n- "));
    process.exit(1);
  }
  console.log("\nGATE PASSED");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
