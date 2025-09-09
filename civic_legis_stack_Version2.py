#!/usr/bin/env python3
# =================================================================================================
# Name: Civic Legislative Data Stack
# Date: 2025-09-09
# Script Name: civic_legis_stack.py
# Version: 0.3.0
# Log Summary:
#   - Adds PostgreSQL + pgvector primary store (documents, sections, embeddings, entities, votes).
#   - Integrates govinfo ingestion, OpenStates API ingestion (state legislatures), generic local file & repo ingestion.
#   - Adds pluggable ingestion adapters (federal, state, local placeholders).
#   - Adds RAG over pgvector; optional Neo4j graph mode with Bloom perspective export.
#   - Plain-language summarization via LocalAI/OpenAI-compatible endpoint.
#   - Unified CLI orchestration: init-db, ingest, embed, serve, export-bloom, sync-govinfo, sync-openstates.
#   - Modular design while remaining a single-file executable (imports optional).
# Description:
#   End-to-end system for compiling a multi-jurisdiction legislative corpus (federal, state, local),
#   storing structured entities and text in PostgreSQL (with pgvector for semantic retrieval),
#   optionally mirroring structural relationships into Neo4j, and exposing an agentic API that
#   blends embedding retrieval + relational/graph lookups + plain-language summarization.
# Change Summary:
#   *0.1.x -> 0.2.x (previous iteration)*: Neo4j-centric, FAISS index, govinfo ingestion scaffold.
#   *0.2.x -> 0.3.0*: Introduces PostgreSQL + pgvector core, full SQL schema, embedding pipeline,
#                      OpenStates integration, improved ingestion normalizer, unified doc registry,
#                      multi-source provenance, flexible query API, Docker/Supabase alignment.
# Inputs:
#   - Environment Variables (see README / .env.example):
#       POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
#       PGVECTOR_DIM (embedding dimension; default 384 for MiniLM)
#       EMBED_MODEL (SentenceTransformer name)
#       LOCALAI_ENDPOINT or OPENAI_API_KEY + MODEL_NAME
#       GOVINFO_API_KEY (optional)
#       OPENSTATES_API_KEY (optional)
#       ENABLE_NEO4J (optional; "1" to activate)
#       NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (if Neo4j active)
# Outputs:
#   - PostgreSQL tables (documents, sections, embeddings, bills, politicians, votes, sources).
#   - pgvector-based similarity search results.
#   - Optional Neo4j nodes/relationships & Bloom perspective JSON.
#   - Plain-language summary cache: data/cache/plain_summaries.json
#   - HTTP API (FastAPI) at configurable port (default 8090).
# =================================================================================================

import os
import sys
import json
import time
import math
import glob
import uuid
import argparse
import zipfile
import datetime
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# -------------------------------------------
# Dependency Management (Minimal On-the-Fly)
# -------------------------------------------
REQUIRED = [
    "requests", "tqdm", "psycopg2-binary", "pydantic", "fastapi", "uvicorn",
    "python-dotenv", "sentence-transformers", "numpy", "scikit-learn",
    "beautifulsoup4", "lxml", "PyPDF2"
]
if os.environ.get("SKIP_AUTO_INSTALL") != "1":
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg.split("==")[0])
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SETUP] Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# Imports post install
import requests
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import psycopg2.extras

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Optional Neo4j
NEO4J_ENABLED = os.environ.get("ENABLE_NEO4J") == "1"
if NEO4J_ENABLED:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[WARN] Neo4j enabled but driver missing; disable or install neo4j.")
        NEO4J_ENABLED = False

# -------------------------------------------
# Utility
# -------------------------------------------
def log(msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    print(f"[{ts}] {msg}")

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -------------------------------------------
# Data Classes
# -------------------------------------------
@dataclass
class IngestedDocument:
    ext_id: str
    title: str
    jurisdiction: str
    full_text: str
    sections: List[Dict[str, Any]]
    source_type: str
    provenance: Dict[str, Any]
    bill_id: Optional[str] = None

# -------------------------------------------
# PostgreSQL / pgvector Layer
# -------------------------------------------
class PGStore:
    def __init__(self):
        self.host = os.environ.get("POSTGRES_HOST", "localhost")
        self.port = int(os.environ.get("POSTGRES_PORT", "5432"))
        self.db = os.environ.get("POSTGRES_DB", "civic_kg")
        self.user = os.environ.get("POSTGRES_USER", "postgres")
        self.password = os.environ.get("POSTGRES_PASSWORD", "postgres")
        self.dim = int(os.environ.get("PGVECTOR_DIM", "384"))
        self.conn = None

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.host, port=self.port, dbname=self.db,
            user=self.user, password=self.password
        )
        self.conn.autocommit = True

    def close(self):
        if self.conn:
            self.conn.close()

    def init_schema(self):
        cur = self.conn.cursor()
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Core tables
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            source_type TEXT,
            meta JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY,
            ext_id TEXT,
            bill_id TEXT,
            title TEXT,
            jurisdiction TEXT,
            source_type TEXT,
            provenance JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            id UUID PRIMARY KEY,
            document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            section_no TEXT,
            heading TEXT,
            text TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            section_id UUID REFERENCES sections(id) ON DELETE CASCADE,
            embedding vector({self.dim}),
            model TEXT,
            PRIMARY KEY (section_id)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS bills (
            bill_id TEXT PRIMARY KEY,
            title TEXT,
            jurisdiction TEXT,
            raw_text TEXT,
            source_type TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS politicians (
            politician_id TEXT PRIMARY KEY,
            name TEXT,
            party TEXT,
            region TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            vote_id TEXT PRIMARY KEY,
            bill_id TEXT REFERENCES bills(bill_id),
            vote_date DATE,
            meta JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS vote_choices (
            vote_id TEXT REFERENCES votes(vote_id) ON DELETE CASCADE,
            politician_id TEXT REFERENCES politicians(politician_id),
            choice TEXT,
            PRIMARY KEY (vote_id, politician_id)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS documents_ext_id_idx ON documents(ext_id);")
        cur.close()
        log("PostgreSQL schema initialized.")

    def insert_document(self, doc: IngestedDocument):
        cur = self.conn.cursor()
        doc_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO documents (id, ext_id, bill_id, title, jurisdiction, source_type, provenance)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (doc_id, doc.ext_id, doc.bill_id, doc.title, doc.jurisdiction, doc.source_type, json.dumps(doc.provenance)))
        # Bill registry (idempotent)
        if doc.bill_id:
            cur.execute("""
            INSERT INTO bills (bill_id, title, jurisdiction, raw_text, source_type)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (bill_id) DO UPDATE SET title=EXCLUDED.title
            """, (doc.bill_id, doc.title, doc.jurisdiction, doc.full_text[:200000], doc.source_type))
        # Sections
        for idx, sec in enumerate(doc.sections):
            sec_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO sections (id, document_id, section_no, heading, text)
                VALUES (%s, %s, %s, %s, %s)
            """, (sec_id, doc_id, sec.get("number") or str(idx+1),
                  sec.get("heading"), sec.get("text")[:50000]))
        cur.close()

    def fetch_sections_for_embedding(self, limit=5000, model="all-MiniLM-L6-v2"):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(f"""
        SELECT s.id, s.text FROM sections s
        LEFT JOIN embeddings e ON e.section_id = s.id AND e.model=%s
        WHERE e.section_id IS NULL
        ORDER BY s.created_at ASC
        LIMIT %s
        """, (model, limit))
        rows = cur.fetchall()
        cur.close()
        return rows

    def insert_embeddings(self, embeddings: List[Tuple[str, np.ndarray]], model: str):
        cur = self.conn.cursor()
        for sid, vec in embeddings:
            cur.execute("""
                INSERT INTO embeddings (section_id, embedding, model)
                VALUES (%s, %s, %s)
                ON CONFLICT (section_id) DO NOTHING
            """, (sid, list(vec), model))
        cur.close()

    def semantic_search(self, query_vec: np.ndarray, k: int = 8, model: str = "all-MiniLM-L6-v2"):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(f"""
        SELECT s.id, s.text, d.bill_id, d.title, 1 - (embedding <=> %s::vector) AS score
        FROM embeddings e
        JOIN sections s ON s.id = e.section_id
        JOIN documents d ON d.id = s.document_id
        WHERE e.model=%s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """, (list(query_vec), model, list(query_vec), k))
        rows = cur.fetchall()
        cur.close()
        return rows

    def get_bill(self, bill_id: str):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT b.bill_id, b.title, b.jurisdiction, b.raw_text
        FROM bills b WHERE b.bill_id=%s
        """, (bill_id,))
        bill = cur.fetchone()
        if not bill:
            cur.close()
            return None
        cur.execute("""
        SELECT s.id, s.section_no, s.heading, s.text
        FROM sections s
        JOIN documents d ON d.id = s.document_id
        WHERE d.bill_id=%s
        ORDER BY s.section_no::int NULLS LAST, s.created_at ASC
        LIMIT 400
        """, (bill_id,))
        sections = cur.fetchall()
        cur.close()
        return {"bill": dict(bill), "sections": [dict(r) for r in sections]}

# -------------------------------------------
# Embedding Engine
# -------------------------------------------
class Embedder:
    def __init__(self):
        model_name = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer not installed.")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True), dtype="float32")

# -------------------------------------------
# Plain Language Summarizer
# -------------------------------------------
class Summarizer:
    def __init__(self):
        self.endpoint = os.environ.get("LOCALAI_ENDPOINT") or "https://api.openai.com/v1"
        self.api_key = os.environ.get("OPENAI_API_KEY", "DUMMY_KEY")
        self.model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        ensure_dirs("data/cache")
        self.cache_path = "data/cache/plain_summaries.json"
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def summarize(self, bill_id: str, section_id: str, text: str) -> str:
        key = f"{bill_id}:{section_id}"
        if key in self.cache:
            return self.cache[key]
        prompt = f"Rewrite this legislative section plainly for the general public:\n\n{text}\n\nPlain language:"
        try:
            r = requests.post(
                f"{self.endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You convert legislative text to clear non-legal English."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                },
                timeout=90
            )
            if r.status_code >= 400:
                raise RuntimeError(r.text[:200])
            content = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            content = f"[SUMMARY_ERROR] {e}"
        self.cache[key] = content
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
        return content

# -------------------------------------------
# Ingestion Adapters
# -------------------------------------------
class GovInfoAdapter:
    API_BASE = "https://api.govinfo.gov"

    def __init__(self, days: int, collections: List[str]):
        self.days = days
        self.collections = collections
        self.api_key = os.environ.get("GOVINFO_API_KEY")
        ensure_dirs("data/govinfo")
        self.state_file = "data/govinfo/state.json"
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"processed": {}}

    def _save_state(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _headers(self):
        h = {"User-Agent": "CivicStack/0.3"}
        if self.api_key:
            h["X-Api-Key"] = self.api_key
        return h

    def _list_packages(self, collection: str, start: str, end: str):
        url = f"{self.API_BASE}/collections/{collection}/{start}/{end}"
        params = {"pageSize": 500}
        out = []
        offset = 0
        while True:
            params["offset"] = offset
            r = requests.get(url, headers=self._headers(), params=params, timeout=60)
            if r.status_code != 200:
                break
            data = r.json()
            pkgs = data.get("packages", [])
            if not pkgs:
                break
            out.extend(pkgs)
            if len(pkgs) < params["pageSize"]:
                break
            offset += params["pageSize"]
        return out

    def _download_zip(self, package_id: str, target_dir: str):
        url = f"{self.API_BASE}/packages/{package_id}/zip"
        r = requests.get(url, headers=self._headers(), timeout=120)
        if r.status_code != 200:
            return False
        ensure_dirs(target_dir)
        zpath = os.path.join(target_dir, f"{package_id}.zip")
        with open(zpath, "wb") as f:
            f.write(r.content)
        try:
            with zipfile.ZipFile(zpath, 'r') as z:
                z.extractall(target_dir)
        except Exception:
            return False
        return True

    def _parse_package(self, collection: str, package_dir: str, package_id: str) -> Optional[IngestedDocument]:
        xmls = []
        for root, _, files in os.walk(package_dir):
            for fn in files:
                if fn.lower().endswith(".xml"):
                    xmls.append(os.path.join(root, fn))
        if not xmls:
            return None
        xmls.sort(key=lambda p: os.path.getsize(p), reverse=True)
        main = xmls[0]
        try:
            with open(main, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml-xml")
        except Exception:
            return None
        title = None
        for t in ["title", "official-title", "dc:title", "docTitle"]:
            el = soup.find(t)
            if el and el.text.strip():
                title = el.text.strip()
                break
        if not title:
            title = package_id
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["section", "p", "Paragraph"])
                      if p.get_text(strip=True)]
        dedup = []
        seen = set()
        for p in paragraphs:
            if p not in seen:
                seen.add(p)
                dedup.append(p)
        sections = []
        for idx, chunk in enumerate(dedup[:60]):
            sections.append({"number": str(idx+1), "heading": None, "text": chunk[:10000]})
        return IngestedDocument(
            ext_id=package_id,
            title=title,
            jurisdiction="US-Federal",
            full_text="\n\n".join(dedup),
            sections=sections if sections else [{"number": "1", "heading": None, "text": "\n\n".join(dedup)[:10000]}],
            source_type=f"govinfo:{collection}",
            provenance={"collection": collection, "package_id": package_id},
            bill_id=package_id  # heuristic
        )

    def ingest(self) -> List[IngestedDocument]:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=self.days)
        start_s, end_s = start.isoformat(), end.isoformat()
        new_docs = []
        for collection in self.collections:
            pkgs = self._list_packages(collection, start_s, end_s)
            log(f"[govinfo] {collection} packages in window: {len(pkgs)}")
            for pkg in tqdm(pkgs, desc=f"govinfo {collection}"):
                pkg_id = pkg.get("packageId")
                if not pkg_id or pkg_id in self.state["processed"]:
                    continue
                target_dir = os.path.join("data", "govinfo", collection, pkg_id)
                if not self._download_zip(pkg_id, target_dir):
                    continue
                doc = self._parse_package(collection, target_dir, pkg_id)
                if doc:
                    new_docs.append(doc)
                self.state["processed"][pkg_id] = {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "collection": collection
                }
                if len(self.state["processed"]) % 50 == 0:
                    self._save_state()
            self._save_state()
        log(f"[govinfo] New docs parsed: {len(new_docs)}")
        return new_docs

class OpenStatesAdapter:
    API = "https://v3.openstates.org/bills"

    def __init__(self, states: List[str], pages: int = 1):
        self.api_key = os.environ.get("OPENSTATES_API_KEY")
        self.states = states
        self.pages = pages

    def _headers(self):
        return {"X-API-KEY": self.api_key} if self.api_key else {}

    def ingest(self) -> List[IngestedDocument]:
        docs = []
        params_base = {
            "sort": "updated_desc",
            "include": "sponsors",
            "per_page": 25
        }
        for st in self.states:
            for page in range(1, self.pages + 1):
                params = params_base.copy()
                params["jurisdiction"] = st
                params["page"] = page
                r = requests.get(self.API, headers=self._headers(), params=params, timeout=60)
                if r.status_code != 200:
                    break
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break
                for b in results:
                    bid = b.get("identifier") or b.get("id") or str(uuid.uuid4())
                    title = b.get("title") or bid
                    full_text = (b.get("title") or "") + "\n\n" + (b.get("summary") or "")
                    sections = [{"number": "1", "heading": None, "text": full_text[:12000]}]
                    docs.append(IngestedDocument(
                        ext_id=b.get("id") or bid,
                        title=title,
                        jurisdiction=st,
                        full_text=full_text,
                        sections=sections,
                        source_type="openstates",
                        provenance={"raw": b},
                        bill_id=bid
                    ))
        log(f"[openstates] Ingested: {len(docs)}")
        return docs

class LocalFilesAdapter:
    def __init__(self, patterns: List[str], jurisdiction_label="Local/Generic"):
        self.patterns = patterns
        self.jurisdiction_label = jurisdiction_label

    def ingest(self) -> List[IngestedDocument]:
        paths = []
        for p in self.patterns:
            paths.extend(glob.glob(p, recursive=True))
        docs = []
        for path in tqdm(paths, desc="local-files"):
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".pdf":
                    with open(path, "rb") as f:
                        reader = PdfReader(f)
                        pages = [pg.extract_text() or "" for pg in reader.pages]
                    text = "\n".join(pages)
                elif ext in (".xml", ".html", ".htm"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        soup = BeautifulSoup(f.read(), "lxml")
                    text = soup.get_text("\n")
                else:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                if not text.strip():
                    continue
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                title = lines[0][:200] if lines else os.path.basename(path)
                sections = _simple_section_chunk(text)
                docs.append(IngestedDocument(
                    ext_id=path,
                    title=title,
                    jurisdiction=self.jurisdiction_label,
                    full_text=text,
                    sections=sections,
                    source_type="local_file",
                    provenance={"path": path},
                    bill_id=None
                ))
            except Exception:
                continue
        log(f"[local-files] Ingested: {len(docs)}")
        return docs

def _simple_section_chunk(text: str, max_len=1800):
    words = text.split()
    cur = []
    sections = []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) >= max_len:
            sections.append({"number": str(len(sections)+1), "heading": None, "text": " ".join(cur)})
            cur = []
    if cur:
        sections.append({"number": str(len(sections)+1), "heading": None, "text": " ".join(cur)})
    return sections

# -------------------------------------------
# Optional Neo4j Bridge
# -------------------------------------------
class Neo4jBridge:
    def __init__(self):
        if not NEO4J_ENABLED:
            raise RuntimeError("Neo4j not enabled.")
        self.uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.environ.get("NEO4J_USER", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT bill_id IF NOT EXISTS FOR (b:Bill) REQUIRE b.bill_id IS UNIQUE")

    def upsert_bill(self, bill_id: str, title: str, jurisdiction: str):
        with self.driver.session() as s:
            s.run("""
            MERGE (b:Bill {bill_id:$bill_id})
            SET b.title=$title, b.jurisdiction=$jurisdiction
            """, bill_id=bill_id, title=title, jurisdiction=jurisdiction)

    def export_bloom(self, path="bloom_perspective.json"):
        perspective = {
            "name": "CivicPostgresMirror",
            "version": "1.0",
            "lastUpdated": datetime.datetime.utcnow().isoformat(),
            "categories": [
                {"name": "Bills", "label": "Bill", "cypher": "MATCH (b:Bill) RETURN b",
                 "style": {"color": "#1f77b4", "size": 50}}
            ],
            "relationships": []
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(perspective, f, indent=2)
        log(f"Bloom perspective exported: {path}")

# -------------------------------------------
# Agent
# -------------------------------------------
class CivicAgent:
    def __init__(self, pg: PGStore, embedder: Embedder, summarizer: Summarizer):
        self.pg = pg
        self.embedder = embedder
        self.summarizer = summarizer
        self.model_name = embedder.model_name

    def answer(self, query: str, k: int = 6, plain: bool = False):
        q_vec = self.embedder.encode([query])[0]
        rows = self.pg.semantic_search(q_vec, k=k, model=self.model_name)
        bill_id = None
        tokens = query.split()
        for t in tokens:
            if "-" in t and any(c.isdigit() for c in t):
                bill_id = t.upper().strip(",.")
                break
        bill_data = self.pg.get_bill(bill_id) if bill_id else None
        response = {
            "query": query,
            "bill_id_detected": bill_id,
            "hits": [],
            "bill": None
        }
        for r in rows:
            response["hits"].append({
                "section_id": r["id"],
                "bill_id": r["bill_id"],
                "title": r["title"],
                "score": float(r["score"]),
                "snippet": r["text"][:600]
            })
        if bill_data:
            response["bill"] = {
                "bill_id": bill_data["bill"]["bill_id"],
                "title": bill_data["bill"]["title"],
                "sections": []
            }
            for sec in bill_data["sections"][:20]:
                sec_obj = {
                    "section_no": sec["section_no"],
                    "heading": sec["heading"],
                    "text": sec["text"][:1200]
                }
                if plain:
                    sec_obj["plain_language"] = self.summarizer.summarize(
                        bill_data["bill"]["bill_id"], sec["id"], sec["text"][:4000])
                response["bill"]["sections"].append(sec_obj)
        return response

# -------------------------------------------
# FastAPI Setup
# -------------------------------------------
app = FastAPI(title="Civic Legislative Stack API", version="0.3.0")

class QueryReq(BaseModel):
    query: str
    k: int = 6
    plain: bool = False

class BillReq(BaseModel):
    bill_id: str
    plain: bool = True

RUNTIME = {"pg": None, "agent": None, "embedder": None, "summarizer": None}

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.3.0"}

@app.post("/query")
def query_endpoint(req: QueryReq):
    if not RUNTIME["agent"]:
        raise HTTPException(status_code=500, detail="Agent not initialized.")
    return RUNTIME["agent"].answer(req.query, k=req.k, plain=req.plain)

@app.post("/bill")
def bill_endpoint(req: BillReq):
    pg: PGStore = RUNTIME["pg"]
    summ: Summarizer = RUNTIME["summarizer"]
    if not pg:
        raise HTTPException(status_code=500, detail="PG not ready.")
    data = pg.get_bill(req.bill_id)
    if not data:
        raise HTTPException(status_code=404, detail="Bill not found")
    if req.plain:
        for sec in data["sections"]:
            sec["plain_language"] = summ.summarize(data["bill"]["bill_id"], sec["id"], sec["text"][:4000])
    return data

# -------------------------------------------
# CLI Orchestration
# -------------------------------------------
def run():
    parser = argparse.ArgumentParser(description="Civic Legislative Stack (PostgreSQL + pgvector)")
    parser.add_argument("--init-db", action="store_true", help="Initialize PostgreSQL schema.")
    parser.add_argument("--sync-govinfo", action="store_true", help="Ingest govinfo collections.")
    parser.add_argument("--govinfo-collections", type=str, default="BILLSTATUS", help="Comma list.")
    parser.add_argument("--govinfo-days", type=int, default=7)
    parser.add_argument("--sync-openstates", action="store_true", help="Ingest OpenStates API.")
    parser.add_argument("--openstates-states", type=str, default="California,New York")
    parser.add_argument("--openstates-pages", type=int, default=1)
    parser.add_argument("--ingest-local", action="store_true", help="Ingest local file patterns.")
    parser.add_argument("--local-patterns", type=str, default="data/local/**/*.txt")
    parser.add_argument("--embed", action="store_true", help="Compute embeddings for new sections.")
    parser.add_argument("--serve", action="store_true", help="Run API server.")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--export-bloom", action="store_true", help="Export Bloom perspective (Neo4j).")
    parser.add_argument("--one-shot-query", type=str, help="Run a single query via agent.")
    parser.add_argument("--plain", action="store_true", help="Plain-language expansions if supported.")
    args = parser.parse_args()

    ensure_dirs("data/cache", "data/local")

    pg = PGStore()
    pg.connect()
    if args.init_db:
        pg.init_schema()

    # Ingestion Phase
    ingest_docs: List[IngestedDocument] = []

    if args.sync_govinfo:
        colls = [c.strip().upper() for c in args.govinfo_collections.split(",") if c.strip()]
        gov_adp = GovInfoAdapter(days=args.govinfo_days, collections=colls)
        ingest_docs.extend(gov_adp.ingest())

    if args.sync_openstates:
        states = [s.strip() for s in args.openstates_states.split(",") if s.strip()]
        os_adp = OpenStatesAdapter(states=states, pages=args.openstates_pages)
        ingest_docs.extend(os_adp.ingest())

    if args.ingest_local:
        patterns = [p.strip() for p in args.local_patterns.split(",") if p.strip()]
        lf_adp = LocalFilesAdapter(patterns)
        ingest_docs.extend(lf_adp.ingest())

    if ingest_docs:
        log(f"Inserting {len(ingest_docs)} documents into PG...")
        for doc in ingest_docs:
            try:
                pg.insert_document(doc)
            except Exception as e:
                log(f"Insert failed for {doc.ext_id}: {e}")

    embedder = None
    summarizer = None

    if args.embed or args.one_shot_query or args.serve:
        summarizer = Summarizer()
        try:
            embedder = Embedder()
        except Exception as e:
            log(f"Embedding model load failed: {e}")
            embedder = None

    if args.embed and embedder:
        rows = pg.fetch_sections_for_embedding(limit=5000, model=embedder.model_name)
        if rows:
            log(f"Embedding {len(rows)} new sections...")
            texts = [r["text"] for r in rows]
            vectors = embedder.encode(texts)
            emb_payload = [(rows[i]["id"], vectors[i]) for i in range(len(rows))]
            pg.insert_embeddings(emb_payload, embedder.model_name)
            log("Embeddings inserted.")

    # Neo4j optional export minimal
    if args.export_bloom and NEO4J_ENABLED:
        neo = Neo4jBridge()
        neo.ensure_constraints()
        # Mirror just bills inserted this run (optional). For simplicity not re-querying all docs.
        for d in ingest_docs:
            if d.bill_id:
                neo.upsert_bill(d.bill_id, d.title, d.jurisdiction)
        neo.export_bloom()
        neo.close()

    # Agent init
    agent = None
    if embedder and summarizer:
        agent = CivicAgent(pg, embedder, summarizer)

    if args.one_shot_query and agent:
        ans = agent.answer(args.one_shot_query, plain=args.plain)
        print(json.dumps(ans, indent=2))

    if args.serve:
        RUNTIME["pg"] = pg
        RUNTIME["agent"] = agent
        RUNTIME["embedder"] = embedder
        RUNTIME["summarizer"] = summarizer
        import uvicorn
        log(f"Starting API on 0.0.0.0:{args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        pg.close()

if __name__ == "__main__":
    run()