/* The front door for the failure report: CORS, and one report per IP per 30 seconds.
 *
 * It does not call Anthropic, and not for want of trying. Measured from the edge against
 * an echo service, a Worker's fetch sends exactly `content-type`, `x-api-key`,
 * `anthropic-version` and `accept-encoding` -- nothing unusual -- and that request
 * answers 403 "Request not allowed", while the same key, body and headers answer 200
 * from a laptop and every genuine key problem answers 401. Routing through AI Gateway
 * does not help, with the key sent alongside or stored in the gateway as BYOK: all three
 * give the same 403, because the gateway egresses from Cloudflare too. The API refuses
 * this network, not this credential.
 *
 * So the Anthropic call is made by a Vercel function (api/report.ts) and this forwards to
 * it with a shared secret. The Anthropic key lives there and nowhere else.
 *
 * The limit is enforced here rather than in the page, because a limit the client owns is
 * a limit anyone can skip. KV holds the last-call timestamp with a 60 s TTL, which is
 * KV's floor -- the 30 s window is the timestamp comparison, not the expiry.
 *
 *   npx wrangler kv namespace create RATE -c worker/wrangler.toml
 *   npx wrangler secret put PROXY_SECRET  -c worker/wrangler.toml
 *   npm run worker:deploy
 */

export interface Env {
  RATE: KVNamespace;
  UPSTREAM: string;          // the Vercel function's URL
  PROXY_SECRET: string;      // shared with it, so that endpoint is not open to the world
  ALLOWED_ORIGIN?: string;   // comma-separated list; unset or "*" = any
}

const WINDOW_MS = 30_000;
const MAX_BODY = 256 * 1024;   // an eval file is ~6 KB; this is slack, not a target

/* An allow-list has to echo back the caller's own origin -- a browser compares the
   header against the page it is on, so returning some other allowed origin blocks every
   caller including the permitted one. Note that CORS keeps other PAGES out, not other
   clients: curl ignores it. What actually limits abuse here is the 30 s window. */
function cors(env: Env, origin: string | null) {
  const list = (env.ALLOWED_ORIGIN ?? "*").split(",").map((s) => s.trim()).filter(Boolean);
  const any = list.includes("*");
  const allow = any ? "*" : (origin && list.includes(origin) ? origin : list[0] ?? "null");
  return {
    "access-control-allow-origin": allow,
    "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
    vary: "origin",
  };
}

const json = (env: Env, origin: string | null, body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...cors(env, origin) },
  });

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const origin = req.headers.get("origin");
    if (req.method === "OPTIONS") return new Response(null, { headers: cors(env, origin) });
    if (req.method !== "POST") return json(env, origin, { error: "POST an eval JSON" }, 405);
    if (!env.UPSTREAM || !env.PROXY_SECRET)
      return json(env, origin, { error: "UPSTREAM / PROXY_SECRET are not configured" }, 500);

    const text = await req.text();
    if (text.length > MAX_BODY) return json(env, origin, { error: "body too large" }, 413);
    try {
      if (!Array.isArray(JSON.parse(text)?.pairs)) throw new Error("no pairs[] in the body");
    } catch (e: any) {
      return json(env, origin, { error: `bad eval JSON: ${e?.message ?? e}` }, 400);
    }

    const ip = req.headers.get("cf-connecting-ip") ?? "unknown";
    const now = Date.now();
    const prev = await env.RATE.get(ip);
    if (prev) {
      const wait = WINDOW_MS - (now - Number(prev));
      if (wait > 0) return json(env, origin, { error: "rate limited", retry_after_ms: wait }, 429);
    }
    // Charge the window before the call, so a slow or failed call cannot be retried in a loop.
    await env.RATE.put(ip, String(now), { expirationTtl: 60 });

    const r = await fetch(env.UPSTREAM, {
      method: "POST",
      headers: { "content-type": "application/json", "x-proxy-secret": env.PROXY_SECRET },
      body: text,
    });
    return new Response(await r.text(), {
      status: r.status,
      headers: { "content-type": "application/json", ...cors(env, origin) },
    });
  },
};
