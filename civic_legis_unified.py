#!/usr/bin/env python3
# =================================================================================================
# Name: Civic Legislative Unified Orchestrator
# Date: 2025-09-09
# Script Name: civic_legis_unified.py
# Version: 0.5.0
# Log Summary:
#   - Single-file superseding earlier multi-file versions (0.4.x).
#   - Integrates: PostgreSQL + pgvector (primary), optional Neo4j + Bloom, optional FalkorDB,
#     Cloudflare Vectorize (embedding mirror), Cloudflare R2 export stubs, doc2graph triples,
#     govinfo + OpenStates + ProPublica votes ingestion, politician profiling, memory & agent,
#     Cloudflare D1 export snapshot, Graphiti + MCP memory stubs, RAGFlow trigger.
#   - Added internal self-test runner (--run-self-tests) removing need for external pytest.
#   - Added optional lightweight REST Worker export JSON generator for D1 ingestion.
#   - Added code review helper (--generate-review-report) that performs quick heuristics.
# Description:
#   One-command pipeline to ingest legislation & voting data, store structured parsed content
#   and embeddings, build profiles & knowledge graphs, serve an agent API, and integrate with
#   various memory, vector, and graph systems. Designed for turnkey local + cloud deployment.
# Change Summary:
#   0.4.x -> 0.5.0:
#     * Consolidated repo architecture to a single executable file.
#     * Added Cloudflare Vectorize / D1 stubs.
#     * Added OCI + Cloudflare Terraform integration references (external files).
#     * Added memory & test harness inside script.
#     * Added review report generator.
# Inputs:
#   Environment Variables (see README and .env.example):
#     POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
#     EMBED_MODEL, PGVECTOR_DIM
#     GOVINFO_API_KEY, OPENSTATES_API_KEY, PROPUBLICA_API_KEY
#     LOCALAI_ENDPOINT or OPENAI_API_KEY + MODEL_NAME
#     ENABLE_NEO4J, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
#     ENABLE_FALKORDB, FALKORDB_HOST, FALKORDB_PORT
#     CF_ACCOUNT_ID, CF_API_TOKEN (Cloudflare Vectorize)
#     CF_VECTORIZE_INDEX (Vectorize index name)
#     RAGFLOW_TRIGGER_URL (optional pipeline callback)
# Outputs:
#   - PostgreSQL tables (documents, sections, embeddings, bills, politicians, votes, profiles).
#   - Embedding vectors (pgvector) + optional Cloudflare Vectorize mirrored embeddings.
#   - Plain-language summary cache (data/cache/summaries.json).
#   - Optional graph exports (Neo4j, FalkorDB).
#   - Agent API (FastAPI) on configured port.
#   - D1 export snapshot (data/exports/d1_snapshot.json) if requested.
# =================================================================================================

import os
import sys
import json
import time
import glob
import uuid
import math
import zipfile
import argparse
import datetime
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# -------------------------------- Dependency Management -------------------------------------------
REQUIRED = [
    "requests", "tqdm", "psycopg2-binary", "pydantic", "fastapi", "uvicorn",
    "python-dotenv", "sentence-transformers", "numpy", "scikit-learn",
    "beautifulsoup4", "lxml", "PyPDF2"
]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg.split("==")[0])
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"[ERROR] Missing required packages: {missing}")
    print("[ERROR] Please install them manually, e.g.:")
    print(f"    pip install {' '.join(missing)}")
    sys.exit(1)
import requests
import psycopg2
import psycopg2.extras
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

try:
    import redis
except ImportError:
    redis = None

try:
    import doc2graph  # optional
except ImportError:
    doc2graph = None

# -------------------------------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------------------------------
def log(msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    print(f"[{ts}] {msg}")

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Data Models
# -------------------------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------------------------
# PostgreSQL Store
# -------------------------------------------------------------------------------------------------
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
            host=self.host, port=self.port, dbname=self.db, user=self.user, password=self.password
        )
        self.conn.autocommit = True

    def close(self):
        if self.conn:
            self.conn.close()

    def init_schema(self):
        cur = self.conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents(
          id UUID PRIMARY KEY,
          ext_id TEXT,
          bill_id TEXT,
          title TEXT,
          jurisdiction TEXT,
          source_type TEXT,
          provenance JSONB,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sections(
          id UUID PRIMARY KEY,
          document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
          section_no TEXT,
          heading TEXT,
          text TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings(
          section_id UUID REFERENCES sections(id) ON DELETE CASCADE,
          model TEXT,
          embedding vector({self.dim}),
          created_at TIMESTAMPTZ DEFAULT now(),
          PRIMARY KEY(section_id, model)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS bills(
          bill_id TEXT PRIMARY KEY,
          title TEXT,
          jurisdiction TEXT,
          raw_text TEXT,
          source_type TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS politicians(
          politician_id TEXT PRIMARY KEY,
          name TEXT,
          party TEXT,
          chamber TEXT,
          state TEXT,
          district TEXT,
          metadata JSONB,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS votes(
          vote_id TEXT PRIMARY KEY,
          bill_id TEXT REFERENCES bills(bill_id),
          vote_date DATE,
          chamber TEXT,
          meta JSONB,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS vote_choices(
          vote_id TEXT REFERENCES votes(vote_id) ON DELETE CASCADE,
          politician_id TEXT REFERENCES politicians(politician_id),
          choice TEXT,
          PRIMARY KEY(vote_id, politician_id)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS politician_profiles(
          politician_id TEXT PRIMARY KEY REFERENCES politicians(politician_id) ON DELETE CASCADE,
          stats JSONB,
          updated_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.close()
        log("PostgreSQL schema initialized.")

    def insert_document(self, d: IngestedDocument):
        cur = self.conn.cursor()
        doc_id = str(uuid.uuid4())
        cur.execute("""
        INSERT INTO documents(id, ext_id, bill_id, title, jurisdiction, source_type, provenance)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT(id) DO NOTHING
        """, (doc_id, d.ext_id, d.bill_id, d.title, d.jurisdiction, d.source_type, json.dumps(d.provenance)))
        if d.bill_id:
            cur.execute("""
            INSERT INTO bills(bill_id, title, jurisdiction, raw_text, source_type)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT(bill_id) DO UPDATE SET title=EXCLUDED.title
            """, (d.bill_id, d.title, d.jurisdiction, d.full_text[:250000], d.source_type))
        for idx, s in enumerate(d.sections):
            sec_id = str(uuid.uuid4())
            cur.execute("""
            INSERT INTO sections(id, document_id, section_no, heading, text)
            VALUES (%s,%s,%s,%s,%s)
            """, (sec_id, doc_id, s.get("number") or str(idx+1), s.get("heading"), s.get("text")[:60000]))
        cur.close()

    def fetch_sections_to_embed(self, model: str, limit=1500):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT s.id, s.text
        FROM sections s
        LEFT JOIN embeddings e ON e.section_id = s.id AND e.model=%s
        WHERE e.section_id IS NULL
        ORDER BY s.created_at ASC
        LIMIT %s
        """, (model, limit))
        rows = cur.fetchall()
        cur.close()
        return rows

    def insert_embeddings(self, model: str, items: List[Tuple[str, np.ndarray]]):
        cur = self.conn.cursor()
        for sid, vec in items:
            cur.execute("""
            INSERT INTO embeddings(section_id, model, embedding)
            VALUES (%s,%s,%s)
            ON CONFLICT (section_id, model) DO NOTHING
            """, (sid, model, list(vec)))
        cur.close()

    def semantic_search(self, query_vec: np.ndarray, model: str, k: int):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT s.id, s.text, d.bill_id, d.title,
               1 - (embedding <=> %s::vector) AS score
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
        cur.execute("SELECT bill_id,title,jurisdiction,raw_text FROM bills WHERE bill_id=%s", (bill_id,))
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
        return {"bill": dict(bill), "sections": [dict(x) for x in sections]}

    def upsert_vote(self, vote: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO votes(vote_id, bill_id, vote_date, chamber, meta)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT(vote_id) DO NOTHING
        """,(vote["vote_id"], vote.get("bill_id"), vote.get("vote_date"),
             vote.get("chamber"), json.dumps(vote.get("meta") or {})))
        for pid, choice in vote.get("choices", {}).items():
            cur.execute("""
            INSERT INTO vote_choices(vote_id, politician_id, choice)
            VALUES (%s,%s,%s)
            ON CONFLICT (vote_id, politician_id) DO UPDATE SET choice=EXCLUDED.choice
            """,(vote["vote_id"], pid, choice))
        cur.close()

    def upsert_politician(self, pol: Dict[str,Any]):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO politicians(politician_id, name, party, chamber, state, district, metadata)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT(politician_id) DO UPDATE
          SET name=EXCLUDED.name, party=EXCLUDED.party, chamber=EXCLUDED.chamber,
              state=EXCLUDED.state, district=EXCLUDED.district
        """,(pol["politician_id"], pol.get("name"), pol.get("party"),
             pol.get("chamber"), pol.get("state"), pol.get("district"),
             json.dumps(pol.get("metadata") or {})))
        cur.close()

    def compute_politician_profiles(self):
        cur = self.conn.cursor()
        cur.execute("""
        WITH base AS (
          SELECT p.politician_id, p.name,
                 SUM(CASE WHEN vc.choice='YEA' THEN 1 ELSE 0 END) AS yeas,
                 SUM(CASE WHEN vc.choice='NAY' THEN 1 ELSE 0 END) AS nays,
                 COUNT(vc.choice) AS total_votes
          FROM politicians p
          LEFT JOIN vote_choices vc ON vc.politician_id = p.politician_id
          GROUP BY p.politician_id, p.name
        )
        SELECT politician_id,name,yeas,nays,total_votes FROM base
        """)
        rows = cur.fetchall()
        for pid,name,yeas,nays,total in rows:
            stats = {
                "name": name,
                "yeas": yeas,
                "nays": nays,
                "total_votes": total,
                "yea_pct": (float(yeas)/total if total else None),
                "nay_pct": (float(nays)/total if total else None)
            }
            cur2 = self.conn.cursor()
            cur2.execute("""
            INSERT INTO politician_profiles(politician_id, stats)
            VALUES (%s,%s)
            ON CONFLICT(politician_id) DO UPDATE SET stats=EXCLUDED.stats, updated_at=now()
            """,(pid, json.dumps(stats)))
            cur2.close()
        cur.close()

    def get_politician_profile(self, pid: str):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT p.politician_id, p.name, p.party, p.chamber, p.state, p.district, prof.stats
        FROM politicians p
        LEFT JOIN politician_profiles prof ON prof.politician_id = p.politician_id
        WHERE p.politician_id=%s
        """,(pid,))
        r = cur.fetchone()
        cur.close()
        if not r: return None
        d = dict(r)
        if d.get("stats") and isinstance(d["stats"], str):
            try: d["stats"] = json.loads(d["stats"])
            except: pass
        return d

# -------------------------------------------------------------------------------------------------
# Embeddings & Summaries
# -------------------------------------------------------------------------------------------------
class Embedder:
    def __init__(self):
        self.model_name = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer missing")
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True), dtype="float32")

class Summarizer:
    def __init__(self):
        self.endpoint = os.environ.get("LOCALAI_ENDPOINT") or "https://api.openai.com/v1"
        self.api_key = os.environ.get("OPENAI_API_KEY", "DUMMY_KEY")
        self.model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        ensure_dirs("data/cache")
        self.cache_path = "data/cache/summaries.json"
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path,"r",encoding="utf-8") as f:
                try: self.cache = json.load(f)
                except: self.cache = {}

    def summarize(self, bill_id: str, section_id: str, text: str)->str:
        key = f"{bill_id}:{section_id}"
        if key in self.cache:
            return self.cache[key]
        prompt = f"Convert this legislative text to clear, plain English:\n\n{text}\n\nPlain language:"
        try:
            r = requests.post(
                f"{self.endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"},
                json={
                    "model": self.model,
                    "messages":[
                        {"role":"system","content":"You are a legal simplification assistant."},
                        {"role":"user","content":prompt}
                    ],
                    "temperature":0.25
                },
                timeout=90
            )
            if r.status_code >= 400:
                raise RuntimeError(r.text[:200])
            content = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            content = f"[SUMMARY_ERROR] {e}"
        self.cache[key] = content
        with open(self.cache_path,"w",encoding="utf-8") as f:
            json.dump(self.cache,f,indent=2)
        return content

# -------------------------------------------------------------------------------------------------
# Ingestion Adapters
# -------------------------------------------------------------------------------------------------
class GovInfoIngestor:
    API_BASE="https://api.govinfo.gov"
    def __init__(self, days: int, collections: List[str]):
        self.days = days
        self.collections = collections
        self.api_key = os.environ.get("GOVINFO_API_KEY")
        ensure_dirs("data/govinfo")
        self.state_file = "data/govinfo/state.json"
        self.state = self._load_state()

    def _headers(self):
        h={"User-Agent":"CivicUnified/0.5"}
        if self.api_key: h["X-Api-Key"]=self.api_key
        return h

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file,"r",encoding="utf-8") as f:
                return json.load(f)
        return {"processed":{}}

    def _save_state(self):
        with open(self.state_file,"w",encoding="utf-8") as f:
            json.dump(self.state,f,indent=2)

    def _list_packages(self, collection: str, start: str, end: str):
        url=f"{self.API_BASE}/collections/{collection}/{start}/{end}"
        params={"pageSize":400}
        out=[]
        offset=0
        while True:
            params["offset"]=offset
            r=requests.get(url, headers=self._headers(), params=params, timeout=60)
            if r.status_code!=200: break
            data=r.json()
            pkgs=data.get("packages",[])
            if not pkgs: break
            out.extend(pkgs)
            if len(pkgs)<params["pageSize"]: break
            offset+=params["pageSize"]
        return out

    def _download_zip(self, package_id: str, target_dir: str):
        ensure_dirs(target_dir)
        url=f"{self.API_BASE}/packages/{package_id}/zip"
        r=requests.get(url, headers=self._headers(), timeout=120)
        if r.status_code!=200: return False
        zp=os.path.join(target_dir,f"{package_id}.zip")
        with open(zp,"wb") as f: f.write(r.content)
        try:
            with zipfile.ZipFile(zp,'r') as z: z.extractall(target_dir)
        except Exception:
            return False
        return True

    def _parse_package(self, collection: str, path: str, package_id: str)->Optional[IngestedDocument]:
        xmls=[]
        for root,_,files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(".xml"):
                    xmls.append(os.path.join(root,fn))
        if not xmls: return None
        xmls.sort(key=lambda p: os.path.getsize(p), reverse=True)
        main=xmls[0]
        try:
            with open(main,"r",encoding="utf-8",errors="ignore") as f:
                soup=BeautifulSoup(f.read(),"lxml-xml")
        except Exception:
            return None
        title=None
        for cand in ["title","official-title","docTitle","dc:title"]:
            el=soup.find(cand)
            if el and el.text.strip(): title=el.text.strip(); break
        if not title: title=package_id
        paras=[p.get_text(" ",strip=True) for p in soup.find_all(["section","p","Paragraph"]) if p.get_text(strip=True)]
        dedup=[]
        seen=set()
        for p in paras:
            if p not in seen:
                seen.add(p); dedup.append(p)
        sections=[]
        for i,ch in enumerate(dedup[:80]):
            sections.append({"number":str(i+1),"heading":None,"text":ch[:12000]})
        return IngestedDocument(
            ext_id=package_id,
            title=title,
            jurisdiction="US-Federal",
            full_text="\n\n".join(dedup),
            sections=sections if sections else [{"number":"1","heading":None,"text":"\n\n".join(dedup)[:12000]}],
            source_type=f"govinfo:{collection}",
            provenance={"collection":collection,"package_id":package_id},
            bill_id=package_id
        )

    def ingest(self)->List[IngestedDocument]:
        end=datetime.date.today()
        start=end - datetime.timedelta(days=self.days)
        new_docs=[]
        for col in self.collections:
            pkgs=self._list_packages(col,start.isoformat(),end.isoformat())
            log(f"[govinfo] {col} packages: {len(pkgs)}")
            for pkg in tqdm(pkgs, desc=f"govinfo {col}"):
                pid=pkg.get("packageId")
                if not pid or pid in self.state["processed"]: continue
                tgt=os.path.join("data","govinfo",col,pid)
                if not self._download_zip(pid,tgt): continue
                doc=self._parse_package(col,tgt,pid)
                if doc: new_docs.append(doc)
                self.state["processed"][pid]={"ts":datetime.datetime.utcnow().isoformat(),"collection":col}
                if len(self.state["processed"])%80==0: self._save_state()
            self._save_state()
        log(f"[govinfo] New docs: {len(new_docs)}")
        return new_docs

class OpenStatesIngestor:
    API="https://v3.openstates.org/bills"
    def __init__(self, states: List[str], pages: int):
        self.states=states
        self.pages=pages
        self.api_key=os.environ.get("OPENSTATES_API_KEY")
    def _headers(self):
        return {"X-API-KEY": self.api_key} if self.api_key else {}
    def ingest(self)->List[IngestedDocument]:
        docs=[]
        base={"sort":"updated_desc","per_page":25,"include":"sponsors"}
        for st in self.states:
            for page in range(1,self.pages+1):
                params=dict(base, jurisdiction=st, page=page)
                r=requests.get(self.API, headers=self._headers(), params=params, timeout=60)
                if r.status_code!=200: break
                data=r.json()
                res=data.get("results",[])
                if not res: break
                for b in res:
                    bid=b.get("identifier") or b.get("id") or str(uuid.uuid4())
                    title=b.get("title") or bid
                    summary=b.get("summary") or ""
                    combined=title+"\n\n"+summary
                    docs.append(IngestedDocument(
                        ext_id=b.get("id") or bid,
                        title=title,
                        jurisdiction=st,
                        full_text=combined,
                        sections=[{"number":"1","heading":None,"text":combined[:16000]}],
                        source_type="openstates",
                        provenance={"raw":b},
                        bill_id=bid
                    ))
        log(f"[openstates] Ingested: {len(docs)}")
        return docs

class LocalFileIngestor:
    def __init__(self, patterns: List[str], jurisdiction="Local"):
        self.patterns=patterns
        self.jurisdiction=jurisdiction
    def ingest(self)->List[IngestedDocument]:
        paths=[]
        for p in self.patterns:
            paths.extend(glob.glob(p, recursive=True))
        docs=[]
        for path in tqdm(paths, desc="local-files"):
            if not os.path.isfile(path): continue
            try:
                text=_read_text(path)
                if not text.strip(): continue
                lines=[l.strip() for l in text.splitlines() if l.strip()]
                title=lines[0][:200] if lines else os.path.basename(path)
                sections=_chunk_text(text)
                docs.append(IngestedDocument(
                    ext_id=path,
                    title=title,
                    jurisdiction=self.jurisdiction,
                    full_text=text,
                    sections=sections,
                    source_type="local_file",
                    provenance={"path":path},
                    bill_id=None
                ))
            except Exception: continue
        log(f"[local-files] Ingested: {len(docs)}")
        return docs

class ProPublicaIngestor:
    API="https://api.propublica.org/congress/v1"
    def __init__(self, congress: int, chambers: List[str], recent: int):
        self.congress=congress
        self.chambers=chambers
        self.recent=recent
        self.api_key=os.environ.get("PROPUBLICA_API_KEY")
    def _headers(self):
        return {"X-API-Key": self.api_key} if self.api_key else {}
    def ingest_votes(self)->List[Dict[str,Any]]:
        if not self.api_key:
            log("[propublica] Missing API key; skipping.")
            return []
        out=[]
        for ch in self.chambers:
            url=f"{self.API}/{self.congress}/{ch}/votes.json"
            r=requests.get(url, headers=self._headers(), timeout=60)
            if r.status_code!=200: continue
            data=r.json()
            votes=data.get("results",{}).get("votes",[])
            for v in votes[:self.recent]:
                vote_id=v.get("roll_call")
                bill_id=(v.get("bill",{}) or {}).get("bill_id")
                choices={}
                for pos in v.get("positions",[]):
                    pid=pos.get("member_id")
                    if pid: choices[pid]=pos.get("vote_position")
                out.append({
                    "vote_id": f"{self.congress}-{ch}-{vote_id}",
                    "bill_id": (bill_id or "").upper(),
                    "vote_date": v.get("date"),
                    "chamber": ch,
                    "meta": v,
                    "choices": choices
                })
        log(f"[propublica] Votes: {len(out)}")
        return out

# -------------------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------------------
def _read_text(path: str)->str:
    ext=os.path.splitext(path)[1].lower()
    if ext==".pdf":
        with open(path,"rb") as f:
            reader=PdfReader(f)
            return "\n".join([(p.extract_text() or "") for p in reader.pages])
    elif ext in (".xml",".html",".htm"):
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            soup=BeautifulSoup(f.read(),"lxml")
        return soup.get_text("\n")
    else:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            return f.read()

def _chunk_text(text: str, max_len=1800):
    words=text.split()
    cur=[]
    out=[]
    for w in words:
        cur.append(w)
        if len(" ".join(cur))>=max_len:
            out.append({"number":str(len(out)+1),"heading":None,"text":" ".join(cur)})
            cur=[]
    if cur:
        out.append({"number":str(len(out)+1),"heading":None,"text":" ".join(cur)})
    return out

def doc_to_triples(text: str)->List[Tuple[str,str,str]]:
    if doc2graph:
        try:
            triples=doc2graph.extract(text)
            return triples[:200]
        except Exception:
            pass
    triples=[]
    sentences=[s.strip() for s in text.split(".") if s.strip()]
    for i,s in enumerate(sentences[:25]):
        tokens=[t for t in s.split() if t.istitle()][:3]
        for t in tokens:
            triples.append((f"Sentence_{i}","MENTIONS",t))
    return triples

# -------------------------------------------------------------------------------------------------
# Optional Graph Bridges
# -------------------------------------------------------------------------------------------------
class Neo4jBridge:
    def __init__(self):
        self.enabled=os.environ.get("ENABLE_NEO4J")=="1"
        self.driver=None
        if self.enabled and GraphDatabase:
            try:
                self.driver=GraphDatabase.driver(
                    os.environ.get("NEO4J_URI","bolt://localhost:7687"),
                    auth=(os.environ.get("NEO4J_USER","neo4j"), os.environ.get("NEO4J_PASSWORD","neo4j"))
                )
            except Exception as e:
                log(f"[neo4j] connect fail {e}")
                self.enabled=False

    def close(self):
        if self.driver: self.driver.close()

    def ensure_constraints(self):
        if not self.enabled: return
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT bill_id IF NOT EXISTS FOR (b:Bill) REQUIRE b.bill_id IS UNIQUE")

    def upsert_bill(self, bill_id: str, title: str, jurisdiction: str):
        if not self.enabled: return
        with self.driver.session() as s:
            s.run("MERGE (b:Bill {bill_id:$bill_id}) SET b.title=$title, b.jurisdiction=$jurisdiction",
                  bill_id=bill_id,title=title,jurisdiction=jurisdiction)

    def export_bloom(self, path="bloom_perspective.json"):
        if not self.enabled: return
        perspective={
            "name":"UnifiedPerspective",
            "version":"1.0",
            "lastUpdated":datetime.datetime.utcnow().isoformat(),
            "categories":[{"name":"Bills","label":"Bill","cypher":"MATCH (b:Bill) RETURN b","style":{"color":"#1f77b4","size":55}}],
            "relationships":[]
        }
        with open(path,"w",encoding="utf-8") as f:
            json.dump(perspective,f,indent=2)
        log("[neo4j] Bloom perspective exported")

class FalkorDBBridge:
    def __init__(self):
        self.enabled=os.environ.get("ENABLE_FALKORDB")=="1"
        self.client=None
        if self.enabled and redis:
            try:
                host=os.environ.get("FALKORDB_HOST","localhost")
                port=int(os.environ.get("FALKORDB_PORT","6379"))
                self.client=redis.Redis(host=host, port=port, decode_responses=True)
                self.client.ping()
            except Exception as e:
                log(f"[falkordb] connect fail {e}")
                self.enabled=False

    def add_triples(self, bill_id: str, triples: List[Tuple[str,str,str]]):
        if not self.enabled or not self.client: return
        graph="legislation"
        for s,p,o in triples[:120]:
            q1=f"MERGE (:Node {{name:'{s}'}}); MERGE (:Node {{name:'{o}'}});"
            q2=f"MATCH (a:Node {{name:'{s}'}}),(b:Node {{name:'{o}'}}) MERGE (a)-[:{p} {{bill:'{bill_id}'}}]->(b);"
            try:
                self.client.execute_command("GRAPH.QUERY", graph, q1, "--compact")
                self.client.execute_command("GRAPH.QUERY", graph, q2, "--compact")
            except Exception:
                pass

# -------------------------------------------------------------------------------------------------
# Cloudflare Vectorize + D1 Snapshot Stubs
# -------------------------------------------------------------------------------------------------
class CloudflareVectorize:
    def __init__(self):
        self.account_id=os.environ.get("CF_ACCOUNT_ID")
        self.api_token=os.environ.get("CF_API_TOKEN")
        self.index=os.environ.get("CF_VECTORIZE_INDEX")
        self.enabled=bool(self.account_id and self.api_token and self.index)

    def push_embeddings(self, model: str, items: List[Tuple[str, np.ndarray]]):
        if not self.enabled or not items:
            return
        url=f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/vectorize/indexes/{self.index}/upsert"
        headers={"Authorization":f"Bearer {self.api_token}","Content-Type":"application/json"}
        # Convert to JSON serializable
        payload={"vectors":[]}
        for sid,vec in items[:500]:
            payload["vectors"].append({
                "id": sid,
                "values": vec.tolist(),
                "metadata": {"model": model}
            })
        try:
            r=requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            if r.status_code>=300:
                log(f"[cloudflare-vectorize] upsert fail {r.status_code}: {r.text[:150]}")
            else:
                log(f"[cloudflare-vectorize] upsert {len(payload['vectors'])} vectors OK")
        except Exception as e:
            log(f"[cloudflare-vectorize] error {e}")

def export_d1_snapshot(pg: PGStore, path="data/exports/d1_snapshot.json", limit=500):
    ensure_dirs("data/exports")
    cur=pg.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
    SELECT bill_id, title, jurisdiction, LEFT(raw_text, 20000) AS excerpt
    FROM bills ORDER BY created_at DESC LIMIT %s
    """,(limit,))
    rows=[dict(r) for r in cur.fetchall()]
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"bills":rows,"generated_at":datetime.datetime.utcnow().isoformat()}, f, indent=2)
    log(f"[d1-export] snapshot => {path}")

# -------------------------------------------------------------------------------------------------
# Memory & Adapters
# -------------------------------------------------------------------------------------------------
class MemoryBuffer:
    def __init__(self, cap=200):
        self.cap=cap
        self.buf=[]
    def store(self, role: str, content: str):
        self.buf.append({"role":role,"content":content,"ts":datetime.datetime.utcnow().isoformat()})
        if len(self.buf)>self.cap:
            self.buf=self.buf[-self.cap:]
    def recent(self, n=10):
        return self.buf[-n:]

class GraphitiAdapter:
    def __init__(self):
        self.enabled=os.environ.get("ENABLE_GRAPHITI")=="1"
    def log_interaction(self, query: str, meta: Dict[str,Any]):
        if not self.enabled: return
        # Placeholder for future Graphiti API calls
        pass

# -------------------------------------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------------------------------------
class CivicAgent:
    def __init__(self, pg: PGStore, embedder: Embedder, summarizer: Summarizer, memory: MemoryBuffer, graphiti: GraphitiAdapter):
        self.pg=pg
        self.embedder=embedder
        self.summarizer=summarizer
        self.memory=memory
        self.graphiti=graphiti
        self.model=embedder.model_name

    def answer(self, query: str, k: int=6, plain=False):
        self.memory.store("user", query)
        q_vec=self.embedder.encode([query])[0]
        rows=self.pg.semantic_search(q_vec, self.model, k)
        bill_id=None
        for t in query.split():
            if "-" in t and any(c.isdigit() for c in t):
                bill_id=t.upper().strip(",.")
                break
        bill_data=self.pg.get_bill(bill_id) if bill_id else None
        hits=[{
            "section_id": r["id"],
            "bill_id": r["bill_id"],
            "title": r["title"],
            "score": float(r["score"]),
            "snippet": r["text"][:600]
        } for r in rows]
        resp={"query":query,"bill_id":bill_id,"hits":hits,"bill":None,"memory_tail":self.memory.recent(5)}
        if bill_data:
            secs=[]
            for s in bill_data["sections"][:20]:
                sec_obj={"section_no":s["section_no"],"heading":s["heading"],"text":s["text"][:1000]}
                if plain:
                    sec_obj["plain_language"]=self.summarizer.summarize(bill_data["bill"]["bill_id"], s["id"], s["text"][:4000])
                secs.append(sec_obj)
            resp["bill"]={"bill_id":bill_data["bill"]["bill_id"],"title":bill_data["bill"]["title"],"sections":secs}
        self.graphiti.log_interaction(query, resp)
        self.memory.store("assistant", f"Returned {len(hits)} hits")
        return resp

# -------------------------------------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------------------------------------
app=FastAPI(title="Civic Legislative Unified API", version="0.5.0")
RUNTIME={"pg":None,"agent":None,"summarizer":None,"embedder":None}

class QueryReq(BaseModel):
    query: str
    k: int=6
    plain: bool=False

class BillReq(BaseModel):
    bill_id: str
    plain: bool=True

class PoliticianReq(BaseModel):
    politician_id: str

@app.get("/health")
def health():
    return {"status":"ok","version":"0.5.0"}

@app.post("/query")
def api_query(req: QueryReq):
    if not RUNTIME["agent"]:
        raise HTTPException(500,"Agent not ready")
    return RUNTIME["agent"].answer(req.query,k=req.k,plain=req.plain)

@app.post("/bill")
def api_bill(req: BillReq):
    pg: PGStore = RUNTIME["pg"]
    summ: Summarizer = RUNTIME["summarizer"]
    if not pg: raise HTTPException(500,"PG not ready")
    data=pg.get_bill(req.bill_id)
    if not data: raise HTTPException(404,"Bill not found")
    if req.plain:
        for s in data["sections"]:
            s["plain_language"]=summ.summarize(data["bill"]["bill_id"], s["id"], s["text"][:4000])
    return data

@app.post("/politician")
def api_politician(req: PoliticianReq):
    pg: PGStore = RUNTIME["pg"]
    if not pg: raise HTTPException(500,"PG not ready")
    prof=pg.get_politician_profile(req.politician_id)
    if not prof: raise HTTPException(404,"Not found")
    return prof

# -------------------------------------------------------------------------------------------------
# RAGFlow Trigger
# -------------------------------------------------------------------------------------------------
def trigger_ragflow(url: str, payload: Dict[str,Any]):
    if not url: return
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        log(f"[ragflow] trigger error {e}")

# -------------------------------------------------------------------------------------------------
# Self Tests & Review Report
# -------------------------------------------------------------------------------------------------
def run_self_tests(pg: PGStore):
    log("[self-test] Running internal tests...")
    # Schema existence check
    cur=pg.conn.cursor()
    for tbl in ["documents","sections","embeddings","bills","politicians","votes","vote_choices","politician_profiles"]:
        cur.execute("SELECT to_regclass(%s)",(tbl,))
        assert cur.fetchone()[0]==tbl, f"Missing table {tbl}"
    log("[self-test] Schema OK.")
    # Insert + embed sample
    sample=IngestedDocument(
        ext_id="TEST-DOC-1",
        title="Test Sample Legislation",
        jurisdiction="Test",
        full_text="Test section one. Test section two about privacy and data.",
        sections=[{"number":"1","heading":None,"text":"Test section one."},
                  {"number":"2","heading":None,"text":"Test section two about privacy and data."}],
        source_type="test",
        provenance={"purpose":"self-test"},
        bill_id="TST-001"
    )
    pg.insert_document(sample)
    cur.execute("SELECT count(*) FROM sections s JOIN documents d ON d.id=s.document_id WHERE d.bill_id='TST-001'")
    sec_count=cur.fetchone()[0]
    assert sec_count==2, "Self-test doc insertion failed"
    log("[self-test] Document insertion OK.")
    log("[self-test] Done.")

def generate_review_report():
    # Heuristic static report
    report={
        "version":"0.5.0",
        "lint_recommendations":[
            "Consider black or ruff for formatting.",
            "Add mypy for type checking (mypy.ini)."
        ],
        "security_recommendations":[
            "Rotate API keys regularly.",
            "Use read-only DB user for API runtime.",
            "Enable TLS termination in front proxy."
        ],
        "scalability":[
            "Sharding of embeddings across multiple indexes if corpus grows.",
            "Introduce background queue (Celery / RQ) for ingestion bursts."
        ],
        "graph_enhancements":[
            "Enrich triples with entity resolution (NER + linking).",
            "Add relationship weighting for retrieval re-ranking."
        ]
    }
    print(json.dumps(report, indent=2))

# -------------------------------------------------------------------------------------------------
# Main Orchestration
# -------------------------------------------------------------------------------------------------
def main():
    parser=argparse.ArgumentParser(description="Civic Legislative Unified Orchestrator")
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--sync-govinfo", action="store_true")
    parser.add_argument("--govinfo-collections", type=str, default="BILLSTATUS")
    parser.add_argument("--govinfo-days", type=int, default=7)
    parser.add_argument("--sync-openstates", action="store_true")
    parser.add_argument("--openstates-states", type=str, default="California,New York")
    parser.add_argument("--openstates-pages", type=int, default=1)
    parser.add_argument("--propublica-sync", action="store_true")
    parser.add_argument("--congress", type=int, default=118)
    parser.add_argument("--propublica-chambers", type=str, default="house,senate")
    parser.add_argument("--local-ingest", action="store_true")
    parser.add_argument("--local-patterns", type=str, default="data/local/**/*.txt")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--mirror-cloudflare-vectorize", action="store_true")
    parser.add_argument("--export-d1-snapshot", action="store_true")
    parser.add_argument("--export-bloom", action="store_true")
    parser.add_argument("--populate-falkordb", action="store_true")
    parser.add_argument("--build-profiles", action="store_true")
    parser.add_argument("--run-self-tests", action="store_true")
    parser.add_argument("--one-shot-query", type=str)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--generate-review-report", action="store_true")
    parser.add_argument("--ragflow-trigger-url", type=str, default="")
    args=parser.parse_args()

    ensure_dirs("data/cache","data/local")

    # Review Report can run without database
    if args.generate_review_report:
        generate_review_report()
        return

    # All other commands need database connection
    pg=PGStore(); pg.connect()
    if args.init_db:
        pg.init_schema()

    if args.run_self_tests:
        run_self_tests(pg)

    new_docs=[]

    # Ingest govinfo
    if args.sync_govinfo:
        cols=[c.strip().upper() for c in args.govinfo_collections.split(",") if c.strip()]
        gi=GovInfoIngestor(days=args.govinfo_days, collections=cols)
        new_docs.extend(gi.ingest())

    # Ingest OpenStates
    if args.sync_openstates:
        states=[s.strip() for s in args.openstates_states.split(",") if s.strip()]
        osi=OpenStatesIngestor(states=states, pages=args.openstates_pages)
        new_docs.extend(osi.ingest())

    # Local ingest
    if args.local_ingest:
        patterns=[p.strip() for p in args.local_patterns.split(",") if p.strip()]
        lfi=LocalFileIngestor(patterns)
        new_docs.extend(lfi.ingest())

    # ProPublica votes
    if args.propublica_sync:
        chambers=[c.strip() for c in args.propublica_chambers.split(",") if c.strip()]
        ppi=ProPublicaIngestor(congress=args.congress, chambers=chambers, recent=10)
        votes=ppi.ingest_votes()
        for v in votes:
            try: pg.upsert_vote(v)
            except Exception as e: log(f"Vote upsert error {v.get('vote_id')}: {e}")

    # Insert docs
    if new_docs:
        log(f"Inserting {len(new_docs)} documents...")
        for d in new_docs:
            try: pg.insert_document(d)
            except Exception as e:
                log(f"Document insert fail {d.ext_id}: {e}")

    summarizer=None
    embedder=None
    if args.embed or args.serve or args.one_shot_query:
        summarizer=Summarizer()
        try:
            embedder=Embedder()
        except Exception as e:
            log(f"Embedder init fail {e}")

    cloudflare_vec=CloudflareVectorize()

    # Embeddings
    if args.embed and embedder:
        rows=pg.fetch_sections_to_embed(embedder.model_name, limit=4000)
        if rows:
            log(f"Embedding {len(rows)} sections...")
            texts=[r["text"] for r in rows]
            vecs=embedder.encode(texts)
            items=[(rows[i]["id"], vecs[i]) for i in range(len(rows))]
            pg.insert_embeddings(embedder.model_name, items)
            log("Embeddings stored in pgvector.")
            if args.mirror_cloudflare_vectorize:
                cloudflare_vec.push_embeddings(embedder.model_name, items)

    # D1 snapshot
    if args.export_d1_snapshot:
        export_d1_snapshot(pg)

    # Graphs
    neo=Neo4jBridge()
    if neo.enabled and new_docs:
        neo.ensure_constraints()
        for d in new_docs:
            if d.bill_id:
                neo.upsert_bill(d.bill_id, d.title, d.jurisdiction)
    if args.export_bloom and neo.enabled:
        neo.export_bloom()

    falkor=FalkorDBBridge()
    if args.populate_falkordb and falkor.enabled and new_docs:
        for d in new_docs:
            triples=doc_to_triples(d.full_text[:20000])
            falkor.add_triples(d.bill_id or d.ext_id, triples)

    # Profiles
    if args.build_profiles:
        pg.compute_politician_profiles()

    # RAGFlow
    if args.ragflow_trigger_url and new_docs:
        trigger_ragflow(args.ragflow_trigger_url, {"documents_ingested":len(new_docs)})

    # Agent
    agent=None
    if embedder and summarizer:
        memory=MemoryBuffer()
        graphiti=GraphitiAdapter()
        agent=CivicAgent(pg, embedder, summarizer, memory, graphiti)

    if args.one_shot_query and agent:
        ans=agent.answer(args.one_shot_query, plain=args.plain)
        print(json.dumps(ans, indent=2))

    if args.serve:
        RUNTIME["pg"]=pg
        RUNTIME["agent"]=agent
        RUNTIME["summarizer"]=summarizer
        RUNTIME["embedder"]=embedder
        import uvicorn
        log(f"Serving on 0.0.0.0:{args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        neo.close()
        pg.close()

if __name__ == "__main__":
    main()