/* The failure report, generated on demand from the evaluation this page just produced.
 *
 * The button calls a Cloudflare Worker, not Anthropic: the page is static, so a key
 * shipped to it would be readable by everyone who loads it. The Worker holds the key and
 * allows one report per IP per 30 seconds. Without VITE_REPORT_API configured there is no
 * endpoint to call, so only the checked-in report is shown and the command that made it.
 */
import { useEffect, useRef, useState } from "react";
import type { EvalDoc } from "./lib/evalDoc";

const B = import.meta.env.BASE_URL;
const API = import.meta.env.VITE_REPORT_API as string | undefined;
const COOLDOWN_MS = 30_000;

/** Garments a report is checked in for, by the file name the demo loaded. */
const REPORTS: Record<string, string> = { "panel_seperated.dxf": "real_dxf_01.md" };

/** Just enough Markdown for what the tool emits: headings, bullets, ordered items, bold. */
function inline(s: string) {
  return s.split(/(\*\*[^*]+\*\*|_[^_]+_|`[^`]+`)/g).map((p, i) =>
    p.startsWith("**") ? <strong key={i}>{p.slice(2, -2)}</strong>
    : p.startsWith("`") ? <code key={i}>{p.slice(1, -1)}</code>
    : p.startsWith("_") && p.endsWith("_") && p.length > 2 ? <em key={i}>{p.slice(1, -1)}</em>
    : <span key={i}>{p}</span>);
}

function Markdown({ md }: { md: string }) {
  return (
    <div className="md">
      {md.split("\n").map((ln, i) => {
        if (!ln.trim()) return null;
        if (ln.startsWith("## ")) return <h3 key={i}>{ln.slice(3)}</h3>;
        if (ln.startsWith("# ")) return <h2 key={i}>{ln.slice(2)}</h2>;
        const bullet = ln.match(/^(\s*)- (.*)$/);
        if (bullet) return <li key={i} className={bullet[1] ? "sub" : ""}>{inline(bullet[2])}</li>;
        const num = ln.match(/^(\d+)\. (.*)$/);
        if (num) return <li key={i} className="num"><b>{num[1]}.</b> {inline(num[2])}</li>;
        return <p key={i}>{inline(ln)}</p>;
      })}
    </div>
  );
}

export default function LlmReport({ name, doc }: { name: string; doc: EvalDoc }) {
  const file = REPORTS[name];
  const [md, setMd] = useState<string | null>(null);
  const [live, setLive] = useState(false);          // is what's shown freshly generated?
  const [gateProblems, setGate] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [cool, setCool] = useState(0);              // seconds left before the next run
  const until = useRef(0);

  useEffect(() => {
    setMd(null); setErr(null); setLive(false); setGate(null);
    if (!file) return;
    let ok = true;
    fetch(`${B}report/${file}`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((t) => { if (ok) setMd(t); })
      .catch((e) => { if (ok) setErr(String(e?.message ?? e)); });
    return () => { ok = false; };
  }, [file]);

  useEffect(() => {
    const t = setInterval(() => setCool(Math.max(0, Math.ceil((until.current - Date.now()) / 1000))), 250);
    return () => clearInterval(t);
  }, []);

  const generate = async () => {
    if (!API) return;
    setBusy(true); setErr(null);
    try {
      const r = await fetch(API, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(doc),
      });
      const j = await r.json();
      if (!r.ok) {
        if (r.status === 429) until.current = Date.now() + (j.retry_after_ms ?? COOLDOWN_MS);
        throw new Error(j.error ?? `HTTP ${r.status}`);
      }
      until.current = Date.now() + COOLDOWN_MS;
      setMd(j.markdown); setLive(true); setGate(j.gate ?? []);
    } catch (e: any) { setErr(String(e?.message ?? e)); }
    finally { setBusy(false); }
  };

  return (
    <div className="llmreport">
      <div className="examples">
        {API ? (
          <button onClick={generate} disabled={busy || cool > 0}>
            {busy ? "writing…" : cool > 0 ? `generate again in ${cool}s` : "generate a new report"}
          </button>
        ) : null}
        <span className="note">
          <strong>claude-sonnet-5</strong>, from this page's evaluation.
          {" "}The gate checks the report's <em>shape</em>, not whether it is right.
          {API ? " One per 30s." : ""}
          {md && !live && file ? " Showing the checked-in run." : ""}
        </span>
      </div>
      {!file && !API && (
        <p className="note">
          No report for this garment. Save the evaluation, then{" "}
          <code>npm run report -- &lt;file&gt;.json --out reports/&lt;name&gt;.md</code>.
        </p>
      )}
      {gateProblems && (
        <p className={`note ${gateProblems.length ? "gatefail" : "gateok"}`}>
          {gateProblems.length ? `GATE FAILED — ${gateProblems.join("; ")}` : "GATE PASSED"}
        </p>
      )}
      {err && <div className="err">{err}</div>}
      {busy && !md && <div className="drop">writing the report…</div>}
      {file && !md && !err && !busy && <div className="drop">loading…</div>}
      {md && <Markdown md={md} />}
    </div>
  );
}
