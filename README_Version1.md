# Civic Legislative Knowledge Graph & RAG Platform

A single-script platform to ingest legislative data (including govinfo.gov bulk/API), build a Neo4j knowledge graph, construct a RAG (Retrieval Augmented Generation) index, provide plain-language translations, and serve an agentic API.

## Features

- govinfo.gov ingestion (API + ZIP bulk) for collections (e.g., BILLSTATUS, PLAW).
- Incremental package tracking (state stored at `data/govinfo/govinfo_state.json`).
- Neo4j graph:
  - Nodes: Bill, Section, Politician
  - Relationships: HAS_SECTION, SPONSORS, VOTED_ON
- Vector index (FAISS if available) for semantic retrieval.
- Plain-language summarization via LocalAI or OpenAI-compatible endpoint.
- Agent answering combining retrieval + graph context.
- FastAPI service endpoints: `/health`, `/bill/summary`, `/query`.
- Bloom perspective generator (`bloom_perspective.json`).
- Optional Docker stack (Neo4j, LocalAI, Flowise, n8n, API).

## Quick Start (Local, No Docker)

```bash
python -m venv .venv
source .venv/bin/activate
python civic_legis_platform.py --init-all --sample
export NEO4J_PASSWORD=yourpass
# Start Neo4j separately or via Docker, then:
python civic_legis_platform.py --govinfo-sync --govinfo-collections BILLSTATUS,PLAW --govinfo-days 7 --build-rag
python civic_legis_platform.py --serve
```

Query:
```bash
curl -X POST http://localhost:8088/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain BILLSTATUS for HR-1234 consumer rights"}'
```

Get Bill Summary (with plain-language):
```bash
curl -X POST http://localhost:8088/bill/summary \
  -H "Content-Type: application/json" \
  -d '{"bill_id":"HR-1234","plain":true}'
```

## Docker Deployment

1. Create `.env` (see `.env.example`).
2. Build/run:
   ```bash
   docker compose up -d --build
   ```
3. API: http://localhost:8088/health  
   Neo4j Browser: http://localhost:7474 (user: neo4j / pass: your .env password)

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| NEO4J_PASSWORD | Auth for Neo4j | Yes |
| GOVINFO_API_KEY | govinfo API key (higher limits) | Recommended |
| LOCALAI_ENDPOINT | LocalAI base URL (OpenAI-style) | Optional |
| OPENAI_API_KEY | Remote OpenAI key | Optional |
| EMBED_MODEL | SentenceTransformer embedding model | Optional |
| MODEL_NAME | Chat model for summarization | Optional |

## govinfo Collections

Common: `BILLSTATUS`, `PLAW`, `BILLS`, `STATUTE`, `USCODE`.  
Pass via `--govinfo-collections BIllSTATUS,PLAW` (comma-separated).  
Use `--govinfo-days N` to set the lookback window.

## Bloom Perspective

Generate perspective JSON:
```bash
python civic_legis_platform.py --emit-bloom
```
Import `bloom_perspective.json` into Neo4j Bloom (File → Open Perspective).

## RAG Pipeline Notes

- Embeddings stored in `data/vector_store`.
- Rebuild after new ingest:
  ```bash
  python civic_legis_platform.py --build-rag
  ```

## Single Script Guarantee

All logic resides in `civic_legis_platform.py`. Additional files (Dockerfile, compose, perspective) are generated automatically or provided for convenience.

## Extending

- Add real sponsor & voting data (e.g., ProPublica Congress API).
- Enhance parsing for legislative XML schema (map actions, committees).
- Add GraphRAG enhancements: path exploration queries for context expansion.
- Cache LLM outputs centrally for reproducibility.

## Safety / Legal

Plain-language summaries are heuristic; verify with qualified legal professionals.  
Respect data source rate limits and licensing (govinfo content is public domain but derivative analysis may impose additional responsibilities).

## Commands Reference

```bash
# Full cycle with govinfo sync + RAG + API
python civic_legis_platform.py --govinfo-sync --govinfo-collections BILLSTATUS,PLAW --govinfo-days 14 --build-rag --serve

# Emit Docker artifacts
python civic_legis_platform.py --emit-docker-compose

# Interactive agent shell
python civic_legis_platform.py --interactive-agent
```

## Troubleshooting

- Missing FAISS: script falls back to numpy similarities.
- No documents after govinfo sync: increase `--govinfo-days` or add more collections.
- Neo4j auth error: ensure `NEO4J_PASSWORD` matches container setting (NEO4J_AUTH).

## License

You may adapt freely. Verify upstream repository or data license requirements for any added sources.
