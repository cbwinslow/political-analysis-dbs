# Civic Legislative Unified Hub (v0.5.0)

A single-script orchestration platform (`civic_legis_unified.py`) for ingesting multi-jurisdiction legislation, building embeddings (pgvector), profiling politicians & votes, generating plain-language summaries, optional graph export (Neo4j / FalkorDB), and integrating with Cloudflare (Vectorize, D1 snapshot), RAGFlow triggers, and memory stubs.

## Quick Start (Local)

```bash
python -m venv .venv
source .venv/bin/activate
export POSTGRES_PASSWORD=postgres
docker run -d --name civic_pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 ankane/pgvector
pip install requests tqdm psycopg2-binary pydantic fastapi uvicorn python-dotenv sentence-transformers numpy scikit-learn beautifulsoup4 lxml PyPDF2
python civic_legis_unified.py --init-db
python civic_legis_unified.py --sync-govinfo --govinfo-collections BILLSTATUS --govinfo-days 5
python civic_legis_unified.py --embed
python civic_legis_unified.py --serve --port 8100
```

Query:
```bash
curl -X POST http://localhost:8100/query -H "Content-Type: application/json" -d '{"query":"Explain BILLSTATUS package HR-1234 privacy","k":4}'
```

## Docker / Compose

```bash
cp .env.example .env
docker compose up -d --build
```

## CLI Flags (Selected)

| Flag | Purpose |
|------|---------|
| --init-db | Initialize schema |
| --sync-govinfo --govinfo-collections X | Ingest federal packages |
| --sync-openstates --openstates-states "California,New York" | State ingestion |
| --propublica-sync --congress 118 --propublica-chambers house,senate | Vote ingestion |
| --local-ingest --local-patterns "data/local/**/*.txt" | Local docs |
| --embed | Create embeddings & (optional) mirror to Cloudflare Vectorize |
| --mirror-cloudflare-vectorize | Upsert embeddings to Vectorize |
| --export-d1-snapshot | Export JSON snapshot for D1 worker ingestion |
| --export-bloom | Generate Neo4j Bloom perspective (if ENABLE_NEO4J=1) |
| --populate-falkordb | Insert doc2graph triples into FalkorDB |
| --build-profiles | Recompute politician statistics |
| --one-shot-query "text" --plain | Single query |
| --serve --port 8100 | Start API |
| --run-self-tests | Internal sanity tests |
| --generate-review-report | Outputs a JSON code review guidance report |

## Cloudflare Strategy

- Heavy ingestion & embedding runs on a VM (e.g., OCI free tier).
- Cloudflare:
  - DNS + caching of read endpoints.
  - Vectorize mirror for semantic retrieval (optional).
  - D1 snapshot (read-only small dataset).
  - R2 (not implemented directly here) can store large raw legislative archives (scripts can be added quickly).
- Worker (see worker/d1_worker.js) can serve preprocessed summaries with ultra-low latency.

## Terraform

In `terraform/`:
- Provisions OCI compute instance (shape, keys), security lists.
- Configures basic Cloudflare DNS A record and optional zero-trust placeholder.
- Extend by adding R2 bucket & Worker deployments.

## Politician Profiling

After ingesting votes:
```bash
python civic_legis_unified.py --build-profiles
curl -X POST http://localhost:8100/politician -H "Content-Type: application/json" -d '{"politician_id":"SOME_MEMBER_ID"}'
```

## Cloudflare D1 Snapshot

```bash
python civic_legis_unified.py --export-d1-snapshot
# Output: data/exports/d1_snapshot.json -> import via wrangler or Worker init script
```

## Code Review Guidelines

See `.github/` workflows:
- CI (lint + self-tests)
- CodeQL (security scanning)
- Docker build
- Dependabot automatic updates
Use PR template and CODEOWNERS for mandatory reviews.

## Free Deployment Recommendation

1. Oracle Cloud Free Tier (ARM instance) for API + Postgres (or external managed PG).
2. Cloudflare fronting DNS + caching static JSON snapshot.
3. Cloudflare D1 only if you want a mini read-only dataset deployed via Worker.
4. Vectorize if you want global edge retrieval (keep canonical in Postgres).

## Extending

- Add advanced NER + entity linking for richer triples.
- Integrate real Graphiti or memory frameworks (replace stubs).
- Add re-ranking step (e.g. cross-encoder) for retrieval improvements.
- Add streaming WebSocket for live legislative updates.

## Disclaimer
Plain-language summaries are not legal advice. Validate significant outputs with legal professionals.

## License
(Choose a license: MIT / Apache-2.0 / etc.)
