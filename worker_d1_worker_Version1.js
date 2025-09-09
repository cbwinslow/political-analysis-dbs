// =================================================================================================
// Name: Cloudflare D1 Worker (Read-Only Bill Summaries)
// Date: 2025-09-09
// Script Name: d1_worker.js
// Version: 0.5.0
// Log Summary:
//   - Simple Cloudflare Worker exposing read-only bill list & detail from a D1 table.
// Description:
//   Deploy via Wrangler. Expects a D1 binding named DB and a table `bills_snapshot` with schema:
//     CREATE TABLE bills_snapshot (bill_id TEXT PRIMARY KEY, title TEXT, jurisdiction TEXT, excerpt TEXT);
//   Populate table by ingesting data/exports/d1_snapshot.json externally.
// Change Summary: Initial stub.
// Inputs: D1 binding
// Outputs: JSON responses
// =================================================================================================
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health") {
      return new Response(JSON.stringify({status:"ok",source:"cloudflare-d1"}), {headers: {"Content-Type":"application/json"}});
    }
    if (url.pathname === "/bills") {
      const { results } = await env.DB.prepare("SELECT bill_id,title,jurisdiction FROM bills_snapshot ORDER BY ROWID DESC LIMIT 100").all();
      return new Response(JSON.stringify(results), {headers: {"Content-Type":"application/json"}});
    }
    if (url.pathname.startsWith("/bill/")) {
      const bid = url.pathname.split("/").pop();
      const { results } = await env.DB.prepare("SELECT * FROM bills_snapshot WHERE bill_id=?").bind(bid).all();
      if (!results.length) {
        return new Response(JSON.stringify({error:"Not found"}), {status:404, headers: {"Content-Type":"application/json"}});
      }
      return new Response(JSON.stringify(results[0]), {headers: {"Content-Type":"application/json"}});
    }
    return new Response(JSON.stringify({error:"Not found"}), {status:404, headers: {"Content-Type":"application/json"}});
  }
}