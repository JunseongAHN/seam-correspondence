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
 * The report itself -- model, schema, system prompt, condensing, gate -- lives in
 * report-core.ts, because api/report.ts serves the same report to the deployed page and
 * the two must not drift.
 *
 * Definition of done (the gate at the bottom enforces it):
 *   1. every MISS / FALSE_POSITIVE is assigned to a category
 *   2. the three root causes we already found are mentioned
 *   3. exactly 3 next actions, ranked
 */
import fs from "node:fs";
import Anthropic from "@anthropic-ai/sdk";
import {
  MODEL, DEBUG_MODEL, condense, failures, gate, render, reportFrom, REPORT_TOOL, SYSTEM,
  type EvalFile,
} from "./report-core.js";

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
    system: SYSTEM,
    tools: [REPORT_TOOL as unknown as Anthropic.Tool],
    tool_choice: { type: "tool", name: REPORT_TOOL.name },
    messages: [{ role: "user", content: condense(data) }],
  });
  const report = reportFrom(resp);

  const md = render(report, resp.usage, Date.now() - t0, model);
  fs.mkdirSync(out.substring(0, out.lastIndexOf("/")) || ".", { recursive: true });
  fs.writeFileSync(out, md);
  console.log(md);

  const problems = gate(md, report, failures(data).length);
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
