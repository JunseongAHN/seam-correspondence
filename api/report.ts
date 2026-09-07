/* The Anthropic call, from a network Anthropic accepts.
 *
 * Cloudflare cannot make this call: the same key and body that answer 200 from a laptop
 * answer 403 "Request not allowed" from a Worker, and from AI Gateway too, while every
 * genuine key problem answers 401. So the Worker keeps the parts it is good at -- CORS
 * and a KV-backed 30 s window per IP -- and this function does the part it cannot.
 *
 * It is not public: the Worker is the only caller, and it proves that with a shared
 * secret. Without it this endpoint would be an open, unmetered Anthropic key.
 *
 *   npx vercel --prod
 *   npx vercel env add ANTHROPIC_API_KEY production
 *   npx vercel env add PROXY_SECRET production
 */
import { failures, gate, render, reportFrom, requestBody, MODEL, type EvalFile }
  from "../tools/report-core.js";

export const config = { runtime: "nodejs" };

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST an eval JSON" });

  const secret = process.env.PROXY_SECRET;
  if (!secret) return res.status(500).json({ error: "PROXY_SECRET is not configured" });
  if (req.headers["x-proxy-secret"] !== secret)
    return res.status(401).json({ error: "not authorised" });
  /* Trim and check the shape. A key pasted with a stray character is not a 401 later,
     it is a TypeError while the header object is being built -- "Cannot convert argument
     to a ByteString" -- which surfaces as an opaque FUNCTION_INVOCATION_FAILED with no
     hint that the credential is what is wrong. */
  const key = (process.env.ANTHROPIC_API_KEY ?? "").trim();
  if (!key) return res.status(500).json({ error: "ANTHROPIC_API_KEY is not configured" });
  const bad = [...key].findIndex((c) => c.charCodeAt(0) > 255);
  if (bad >= 0 || !/^sk-ant-/.test(key))
    return res.status(500).json({
      error: bad >= 0
        ? `ANTHROPIC_API_KEY has a non-Latin-1 character at index ${bad}; it cannot go in `
          + "a header. Re-add it with nothing but the key: vercel env rm/add"
        : "ANTHROPIC_API_KEY does not start with sk-ant- ; re-add it with nothing but the key",
    });

  let data: EvalFile;
  try {
    data = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    if (!Array.isArray(data?.pairs)) throw new Error("no pairs[] in the body");
  } catch (e: any) {
    return res.status(400).json({ error: `bad eval JSON: ${e?.message ?? e}` });
  }
  const nFail = failures(data).length;
  if (!nFail) return res.status(400).json({ error: "nothing failed in this evaluation" });

  const t0 = Date.now();
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(requestBody(data, MODEL)),
  });
  if (!r.ok)
    return res.status(502).json({ error: `anthropic ${r.status}: ${(await r.text()).slice(0, 300)}` });

  try {
    const resp: any = await r.json();
    const report = reportFrom(resp);
    const markdown = render(report, resp.usage, Date.now() - t0, MODEL);
    return res.status(200).json({ markdown, usage: resp.usage, gate: gate(markdown, report, nFail) });
  } catch (e: any) {
    return res.status(502).json({ error: String(e?.message ?? e) });
  }
}
