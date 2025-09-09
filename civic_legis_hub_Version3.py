#!/usr/bin/env python3
# =================================================================================================
# Name: Civic Legislative Hub Orchestrator
# Date: 2025-09-09
# Script Name: civic_legis_hub.py
# Version: 0.4.0
# Log Summary:
#   - Unified entry point orchestrating ingestion, embeddings, RAG, memory, graph exports.
#   - Integrations: PostgreSQL + pgvector (primary), FalkorDB (optional), Neo4j (optional),
#     Graphiti, doc2graph, ProPublica votes, OpenStates, govinfo, RAGFlow pipeline trigger.
#   - Adds politician profiling & vote roll-ups, doc2graph triple extraction, memory abstraction.
#   - Provides agent API (FastAPI) + CLI commands + plain-language summarization cache.
# Description:
#   This script can run the full pipeline or modular steps for building a legislative intelligence
#   platform with semantic retrieval, knowledge graph enrichment, and agentic Q&A across multiple
#   jurisdictions and sources.
# Change Summary:
#   0.1.x → 0.2.x: Neo4j + govinfo + RAG (previous).
#   0.2.x → 0.3.x: PostgreSQL pgvector core, OpenStates integration.
#   0.3.x → 0.4.0: FalkorDB + Graphiti adapters, ProPublica votes, doc2graph parsing, memory modules,
#                  politician profiling, test harness alignment, RAGFlow stub, improved CLI.
# Inputs:
#   Environment variables (see README / .env.example) controlling DB, APIs, models, graph toggles.
# Outputs:
#   - PostgreSQL tables populated (documents, sections, embeddings, bills, politicians, votes...).
#   - Optional FalkorDB (graph nodes/edges) and Neo4j + Bloom export.
#   - REST API (FastAPI) on configured port.
#   - Summaries cache, embeddings, profiling JSON, graph export artifacts.
# =================================================================================================

import os
import sys
import json
import time
import glob
import uuid
import math
import zipfile
import socket
import argparse
import datetime
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# -------------------------- Dependency Setup ------------------------------------------------------
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

# Optional imports try/except (Graphiti, doc2graph, etc.)
try:
    import doc2graph  # hypothetical public interface
except ImportError:
    doc2graph = None

try:
    import redis  # for FalkorDB connectivity (Redis protocol)
except ImportError:
    redis = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

# -------------------------- Utility ---------------------------------------------------------------
def log(msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    print(f"[{ts}] {msg}")

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -------------------------- Data Models -----------------------------------------------------------
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

# -------------------------- PostgreSQL Store ------------------------------------------------------
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
        self.conn = psycopg2.connect(host=self.host, port=self.port,
                                     dbname=self.db, user=self.user, password=self.password)
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
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sections(
          id UUID PRIMARY KEY,
          document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
          section_no TEXT,
          heading TEXT,
          text TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings(
          section_id UUID REFERENCES sections(id) ON DELETE CASCADE,
          embedding vector({self.dim}),
          model TEXT,
          created_at TIMESTAMPTZ DEFAULT now(),
          PRIMARY KEY(section_id, model)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS bills(
          bill_id TEXT PRIMARY KEY,
          title TEXT,
          jurisdiction TEXT,
          raw_text TEXT,
          source_type TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
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
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS votes(
          vote_id TEXT PRIMARY KEY,
          bill_id TEXT REFERENCES bills(bill_id),
          vote_date DATE,
          chamber TEXT,
          meta JSONB,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS vote_choices(
          vote_id TEXT REFERENCES votes(vote_id) ON DELETE CASCADE,
          politician_id TEXT REFERENCES politicians(politician_id),
          choice TEXT,
          PRIMARY KEY(vote_id, politician_id)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS politician_profiles(
          politician_id TEXT PRIMARY KEY REFERENCES politicians(politician_id) ON DELETE CASCADE,
          stats JSONB,
          updated_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS documents_ext_id_idx ON documents(ext_id);")
        cur.close()
        log("PostgreSQL schema initialized.")

    def insert_document(self, doc: IngestedDocument):
        cur = self.conn.cursor()
        doc_id = str(uuid.uuid4())
        cur.execute("""
        INSERT INTO documents(id, ext_id, bill_id, title, jurisdiction, source_type, provenance)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (id) DO NOTHING
        """, (doc_id, doc.ext_id, doc.bill_id, doc.title, doc.jurisdiction, doc.source_type, json.dumps(doc.provenance)))
        if doc.bill_id:
            cur.execute("""
            INSERT INTO bills(bill_id, title, jurisdiction, raw_text, source_type)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT(bill_id) DO UPDATE SET title=EXCLUDED.title
            """, (doc.bill_id, doc.title, doc.jurisdiction, doc.full_text[:250000], doc.source_type))
        for idx, s in enumerate(doc.sections):
            sec_id = str(uuid.uuid4())
            cur.execute("""
            INSERT INTO sections(id, document_id, section_no, heading, text)
            VALUES (%s,%s,%s,%s,%s)
            """, (sec_id, doc_id, s.get("number") or str(idx+1), s.get("heading"), s.get("text")[:60000]))
        cur.close()

    def fetch_sections_to_embed(self, model: str, limit=1000):
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
            INSERT INTO embeddings(section_id, embedding, model)
            VALUES (%s,%s,%s)
            ON CONFLICT (section_id, model) DO NOTHING
            """, (sid, list(vec), model))
        cur.close()

    def semantic_search(self, query_vec: np.ndarray, model: str, k: int = 8):
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
        cur.execute("""
        SELECT bill_id, title, jurisdiction, raw_text FROM bills WHERE bill_id=%s
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
        return {"bill": dict(bill), "sections": [dict(x) for x in sections]}

    def upsert_politician(self, pol: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO politicians(politician_id, name, party, chamber, state, district, metadata)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT(politician_id) DO UPDATE
          SET name=EXCLUDED.name, party=EXCLUDED.party, chamber=EXCLUDED.chamber,
              state=EXCLUDED.state, district=EXCLUDED.district
        """,(pol["politician_id"], pol.get("name"), pol.get("party"), pol.get("chamber"),
             pol.get("state"), pol.get("district"), json.dumps(pol.get("metadata") or {})))
        cur.close()

    def upsert_vote(self, vote: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO votes(vote_id, bill_id, vote_date, chamber, meta)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT(vote_id) DO NOTHING
        """,(vote["vote_id"], vote.get("bill_id"), vote.get("vote_date"), vote.get("chamber"),
             json.dumps(vote.get("meta") or {})))
        for pid, choice in vote.get("choices", {}).items():
            cur.execute("""
            INSERT INTO vote_choices(vote_id, politician_id, choice)
            VALUES (%s,%s,%s)
            ON CONFLICT (vote_id, politician_id) DO UPDATE SET choice=EXCLUDED.choice
            """,(vote["vote_id"], pid, choice))
        cur.close()

    def compute_politician_profiles(self):
        # Simple aggregate: counts of YEA/NAY by party, top jurisdictions
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
        SELECT * FROM base
        """)
        rows = cur.fetchall()
        for r in rows:
            pid, name, yeas, nays, total = r
            stats = {
                "name": name,
                "total_votes": total,
                "yea_pct": float(yeas)/total if total else None,
                "nay_pct": float(nays)/total if total else None,
                "yeas": yeas,
                "nays": nays
            }
            cur2 = self.conn.cursor()
            cur2.execute("""
            INSERT INTO politician_profiles(politician_id, stats)
            VALUES (%s,%s)
            ON CONFLICT(politician_id) DO UPDATE SET stats=EXCLUDED.stats, updated_at=now()
            """,(pid, json.dumps(stats)))
            cur2.close()
        cur.close()

    def get_politician_profile(self, politician_id: str):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT p.politician_id, p.name, p.party, p.chamber, p.state, p.district, prof.stats
        FROM politicians p
        LEFT JOIN politician_profiles prof ON prof.politician_id = p.politician_id
        WHERE p.politician_id=%s
        """,(politician_id,))
        r = cur.fetchone()
        cur.close()
        if not r: return None
        d = dict(r)
        if d.get("stats") and isinstance(d["stats"], str):
            try:
                d["stats"] = json.loads(d["stats"])
            except:
                pass
        return d

# -------------------------- Embeddings / Summarization --------------------------------------------
class Embedder:
    def __init__(self):
        self.model_name = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer missing.")
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True), dtype="float32")

class Summarizer:
    def __init__(self):
        self.endpoint = os.environ.get("LOCALAI_ENDPOINT") or "https://api.openai.com/v1"
        self.api_key = os.environ.get("OPENAI_API_KEY", "DUMMY_KEY")
        self.model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        ensure_dirs("data/cache")
        self.cache_file = "data/cache/summaries.json"
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def summarize(self, bill_id: str, section_id: str, text: str) -> str:
        key = f"{bill_id}:{section_id}"
        if key in self.cache:
            return self.cache[key]
        prompt = f"Rewrite this legislative text in clear, accessible language:\n\n{text}\n\nPlain-language version:"
        try:
            r = requests.post(
                f"{self.endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"},
                json={
                    "model": self.model,
                    "messages":[
                        {"role":"system","content":"You convert legislation into plain clear non-legal English."},
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
        with open(self.cache_file,"w",encoding="utf-8") as f:
            json.dump(self.cache,f,indent=2)
        return content

# -------------------------- Ingestion Adapters ----------------------------------------------------
class GovInfoIngestor:
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
            with open(self.state_file,"r",encoding="utf-8") as f:
                return json.load(f)
        return {"processed":{}}

    def _save_state(self):
        with open(self.state_file,"w",encoding="utf-8") as f:
            json.dump(self.state,f,indent=2)

    def _headers(self):
        h = {"User-Agent":"CivicHub/0.4"}
        if self.api_key: h["X-Api-Key"] = self.api_key
        return h

    def _list_packages(self, collection: str, start: str, end: str):
        url = f"{self.API_BASE}/collections/{collection}/{start}/{end}"
        params = {"pageSize":400}
        out=[]
        off=0
        while True:
            params["offset"]=off
            r=requests.get(url, headers=self._headers(), params=params, timeout=60)
            if r.status_code!=200: break
            data=r.json(); pkgs=data.get("packages",[])
            if not pkgs: break
            out.extend(pkgs)
            if len(pkgs)<params["pageSize"]: break
            off+=params["pageSize"]
        return out

    def _download_zip(self, package_id: str, target_dir: str):
        url = f"{self.API_BASE}/packages/{package_id}/zip"
        r=requests.get(url, headers=self._headers(), timeout=120)
        if r.status_code!=200: return False
        ensure_dirs(target_dir)
        zp=os.path.join(target_dir,f"{package_id}.zip")
        with open(zp,"wb") as f: f.write(r.content)
        try:
            with zipfile.ZipFile(zp,'r') as z: z.extractall(target_dir)
        except Exception: return False
        return True

    def _parse_package(self, collection: str, package_dir: str, package_id: str)->Optional[IngestedDocument]:
        xmls=[]
        for root,_,files in os.walk(package_dir):
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
        for cand in ["title","official-title","dc:title","docTitle"]:
            el=soup.find(cand)
            if el and el.text.strip():
                title=el.text.strip(); break
        if not title: title=package_id
        paragraphs=[p.get_text(" ",strip=True) for p in soup.find_all(["section","p","Paragraph"]) if p.get_text(strip=True)]
        dedup=[]
        seen=set()
        for p in paragraphs:
            if p not in seen:
                seen.add(p); dedup.append(p)
        secs=[]
        for idx, chunk in enumerate(dedup[:80]):
            secs.append({"number":str(idx+1),"heading":None,"text":chunk[:12000]})
        return IngestedDocument(
            ext_id=package_id,
            title=title,
            jurisdiction="US-Federal",
            full_text="\n\n".join(dedup),
            sections=secs if secs else [{"number":"1","heading":None,"text":"\n\n".join(dedup)[:12000]}],
            source_type=f"govinfo:{collection}",
            provenance={"collection":collection,"package_id":package_id},
            bill_id=package_id
        )

    def ingest(self)->List[IngestedDocument]:
        end=datetime.date.today()
        start=end - datetime.timedelta(days=self.days)
        new_docs=[]
        for col in self.collections:
            packages=self._list_packages(col,start.isoformat(),end.isoformat())
            log(f"[govinfo] {col} packages: {len(packages)}")
            for pkg in tqdm(packages, desc=f"govinfo {col}"):
                package_id=pkg.get("packageId")
                if not package_id or package_id in self.state["processed"]:
                    continue
                tgt=os.path.join("data","govinfo",col,package_id)
                if not self._download_zip(package_id,tgt):
                    continue
                doc=self._parse_package(col,tgt,package_id)
                if doc: new_docs.append(doc)
                self.state["processed"][package_id]={"ts":datetime.datetime.utcnow().isoformat(),"collection":col}
                if len(self.state["processed"])%60==0: self._save_state()
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
        params_base={"sort":"updated_desc","per_page":25,"include":"sponsors"}
        for st in self.states:
            for page in range(1,self.pages+1):
                params=params_base.copy()
                params["jurisdiction"]=st
                params["page"]=page
                r=requests.get(self.API, headers=self._headers(), params=params, timeout=60)
                if r.status_code!=200: break
                data=r.json()
                results=data.get("results",[])
                if not results: break
                for b in results:
                    bid=b.get("identifier") or b.get("id") or str(uuid.uuid4())
                    title=b.get("title") or bid
                    summary=b.get("summary") or ""
                    full=title+"\n\n"+summary
                    docs.append(IngestedDocument(
                        ext_id=b.get("id") or bid,
                        title=title,
                        jurisdiction=st,
                        full_text=full,
                        sections=[{"number":"1","heading":None,"text":full[:15000]}],
                        source_type="openstates",
                        provenance={"raw":b},
                        bill_id=bid
                    ))
        log(f"[openstates] Ingested: {len(docs)}")
        return docs

class LocalFileIngestor:
    def __init__(self, patterns: List[str], jurisdiction="Local/Generic"):
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
                text=_read_text_file(path)
                if not text.strip(): continue
                lines=[l.strip() for l in text.splitlines() if l.strip()]
                title=lines[0][:200] if lines else os.path.basename(path)
                secs=_chunk_text(text)
                docs.append(IngestedDocument(
                    ext_id=path,
                    title=title,
                    jurisdiction=self.jurisdiction,
                    full_text=text,
                    sections=secs,
                    source_type="local_file",
                    provenance={"path":path},
                    bill_id=None
                ))
            except Exception:
                continue
        log(f"[local-files] Ingested: {len(docs)}")
        return docs

class ProPublicaVotesIngestor:
    """
    Fetches recent House/Senate votes and enriches politician + bill alignment.
    Requires PROPUBLICA_API_KEY.
    """
    API_BASE="https://api.propublica.org/congress/v1"
    def __init__(self, congress: int, chambers: List[str], recent: int = 5):
        self.congress=congress
        self.chambers=chambers
        self.recent=recent
        self.api_key=os.environ.get("PROPUBLICA_API_KEY")
    def _headers(self):
        return {"X-API-Key": self.api_key} if self.api_key else {}
    def ingest(self)->List[Dict[str,Any]]:
        votes=[]
        if not self.api_key:
            log("[propublica] API key missing; skipping.")
            return votes
        for chamber in self.chambers:
            url=f"{self.API_BASE}/{self.congress}/{chamber}/votes.json"
            r=requests.get(url, headers=self._headers(), timeout=60)
            if r.status_code!=200: continue
            data=r.json()
            vlist=data.get("results",{}).get("votes",[])
            for v in vlist[:self.recent]:
                vote_id=v.get("roll_call")
                bill_id=v.get("bill",{}).get("bill_id")
                vote_date=v.get("date")
                choices_map={}
                for mem in v.get("positions",[]):
                    pid=mem.get("member_id")
                    if pid: choices_map[pid]=mem.get("vote_position")
                votes.append({
                    "vote_id": f"{self.congress}-{chamber}-{vote_id}",
                    "bill_id": (bill_id or "").upper(),
                    "vote_date": vote_date,
                    "chamber": chamber,
                    "meta": v,
                    "choices": choices_map
                })
        log(f"[propublica] Votes ingested: {len(votes)}")
        return votes

# -------------------------- Helper Functions ------------------------------------------------------
def _read_text_file(path: str)->str:
    ext=os.path.splitext(path)[1].lower()
    if ext==".pdf":
        with open(path,"rb") as f:
            reader=PdfReader(f)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
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
    secs=[]
    for w in words:
        cur.append(w)
        if len(" ".join(cur))>=max_len:
            secs.append({"number":str(len(secs)+1),"heading":None,"text":" ".join(cur)})
            cur=[]
    if cur:
        secs.append({"number":str(len(secs)+1),"heading":None,"text":" ".join(cur)})
    return secs

# -------------------------- doc2graph Integration -------------------------------------------------
def doc_to_graph_triples(text: str)->List[Tuple[str,str,str]]:
    """
    Use doc2graph if available; else naive extraction of (Sentence_i, 'MENTIONS', Token_j)
    """
    if doc2graph:
        try:
            # Hypothetical call
            triples=doc2graph.extract(text)
            return triples[:200]
        except Exception:
            pass
    triples=[]
    sentences=[s.strip() for s in text.split(".") if s.strip()]
    for i,sent in enumerate(sentences[:20]):
        tokens=[t for t in sent.split() if t.istitle()][:3]
        for t in tokens:
            triples.append((f"Sentence_{i}", "MENTIONS", t))
    return triples

# -------------------------- FalkorDB Bridge -------------------------------------------------------
class FalkorDBBridge:
    """
    Minimal Redis protocol client expecting FalkorDB/RedisGraph style GRAPH.QUERY support.
    """
    def __init__(self):
        self.host=os.environ.get("FALKORDB_HOST","falkordb")
        self.port=int(os.environ.get("FALKORDB_PORT","6379"))
        self.enabled=os.environ.get("ENABLE_FALKORDB")=="1"
        self.client=None
        if self.enabled and redis:
            try:
                self.client=redis.Redis(host=self.host, port=self.port, decode_responses=True)
                self.client.ping()
            except Exception as e:
                log(f"[falkordb] connection failed: {e}")
                self.enabled=False

    def add_bill_triples(self, bill_id: str, triples: List[Tuple[str,str,str]]):
        if not self.enabled or not self.client: return
        # Build simple Cypher for FalkorDB / RedisGraph style
        graph_name="legislation"
        for s,p,o in triples[:100]:
            q=f"MERGE (:Node {{name:'{s}'}}); MERGE (:Node {{name:'{o}'}});"
            rel=f"MATCH (a:Node {{name:'{s}'}}), (b:Node {{name:'{o}'}}) MERGE (a)-[:{p} {{bill:'{bill_id}'}}]->(b);"
            try:
                self.client.execute_command("GRAPH.QUERY", graph_name, q, "--compact")
                self.client.execute_command("GRAPH.QUERY", graph_name, rel, "--compact")
            except Exception:
                pass

# -------------------------- Neo4j Bridge (Optional) -----------------------------------------------
class Neo4jBridge:
    def __init__(self):
        self.enabled = os.environ.get("ENABLE_NEO4J")=="1"
        self.driver=None
        if self.enabled and GraphDatabase:
            try:
                self.driver=GraphDatabase.driver(
                    os.environ.get("NEO4J_URI","bolt://localhost:7687"),
                    auth=(os.environ.get("NEO4J_USER","neo4j"), os.environ.get("NEO4J_PASSWORD","neo4j"))
                )
            except Exception as e:
                log(f"[neo4j] connect fail: {e}")
                self.enabled=False

    def close(self):
        if self.driver:
            self.driver.close()

    def ensure_constraints(self):
        if not self.enabled: return
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT bill_id IF NOT EXISTS FOR (b:Bill) REQUIRE b.bill_id IS UNIQUE")

    def upsert_bill(self, bill_id: str, title: str, jurisdiction: str):
        if not self.enabled: return
        with self.driver.session() as s:
            s.run("""
            MERGE (b:Bill {bill_id:$bill_id})
            SET b.title=$title, b.jurisdiction=$jurisdiction
            """, bill_id=bill_id, title=title, jurisdiction=jurisdiction)

    def export_bloom(self, path="bloom_perspective.json"):
        if not self.enabled: return
        perspective={
            "name":"CivicHubPerspective",
            "version":"1.0",
            "lastUpdated":datetime.datetime.utcnow().isoformat(),
            "categories":[{
                "name":"Bills","label":"Bill","cypher":"MATCH (b:Bill) RETURN b",
                "style":{"color":"#1f77b4","size":55}
            }],
            "relationships":[]
        }
        with open(path,"w",encoding="utf-8") as f:
            json.dump(perspective,f,indent=2)
        log(f"[neo4j] Bloom perspective exported: {path}")

# -------------------------- Graphiti / Memory Abstractions (Stubs) -------------------------------
class GraphitiAdapter:
    def __init__(self):
        self.enabled = os.environ.get("ENABLE_GRAPHITI")=="1"
    def log_interaction(self, query: str, context: Dict[str,Any]):
        if not self.enabled:
            return
        # Placeholder: integrate with Graphiti APIs if library installed.
        pass

class MCPMemoryAdapter:
    """
    Placeholder for mcp-memory-libsql / mcp-neo4j-agent-memory / memonto / aius
    Provides uniform store() / fetch() interfaces for conversation or agent state.
    """
    def __init__(self):
        self.enabled=True
        self.buffer=[]
        self.max_items=200
    def store(self, role: str, content: str):
        self.buffer.append({"role":role,"content":content,"ts":datetime.datetime.utcnow().isoformat()})
        if len(self.buffer)>self.max_items:
            self.buffer=self.buffer[-self.max_items:]
    def recent(self, n=10):
        return self.buffer[-n:]

# -------------------------- Agent ----------------------------------------------------------------
class CivicAgent:
    def __init__(self, pg: PGStore, embedder: Embedder, summarizer: Summarizer,
                 graphiti: GraphitiAdapter, memory: MCPMemoryAdapter):
        self.pg=pg
        self.embedder=embedder
        self.summarizer=summarizer
        self.graphiti=graphiti
        self.memory=memory
        self.model_name=embedder.model_name

    def answer(self, query: str, k: int=6, plain=False):
        self.memory.store("user", query)
        q_vec=self.embedder.encode([query])[0]
        rows=self.pg.semantic_search(q_vec,model=self.model_name,k=k)
        bill_id=None
        for token in query.split():
            if "-" in token and any(c.isdigit() for c in token):
                bill_id=token.upper().strip(",.")
                break
        bill_data=self.pg.get_bill(bill_id) if bill_id else None
        hits=[]
        for r in rows:
            hits.append({
                "section_id": r["id"],
                "bill_id": r["bill_id"],
                "title": r["title"],
                "score": float(r["score"]),
                "snippet": r["text"][:600]
            })
        answer={
            "query": query,
            "bill_id": bill_id,
            "hits": hits,
            "bill": None,
            "memory_tail": self.memory.recent(5)
        }
        if bill_data:
            summary_sections=[]
            for sec in bill_data["sections"][:15]:
                sec_obj={
                    "section_no": sec["section_no"],
                    "heading": sec["heading"],
                    "text": sec["text"][:1000]
                }
                if plain:
                    sec_obj["plain_language"]=self.summarizer.summarize(
                        bill_data["bill"]["bill_id"], sec["id"], sec["text"][:4000])
                summary_sections.append(sec_obj)
            answer["bill"]={
                "bill_id": bill_data["bill"]["bill_id"],
                "title": bill_data["bill"]["title"],
                "sections": summary_sections
            }
        self.graphiti.log_interaction(query, answer)
        self.memory.store("assistant", f"Returned {len(hits)} hits")
        return answer

# -------------------------- FastAPI ---------------------------------------------------------------
app=FastAPI(title="Civic Legislative Hub", version="0.4.0")
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
    return {"status":"ok","version":"0.4.0"}

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
        for sec in data["sections"]:
            sec["plain_language"]=summ.summarize(data["bill"]["bill_id"], sec["id"], sec["text"][:4000])
    return data

@app.post("/politician")
def api_politician(req: PoliticianReq):
    pg: PGStore = RUNTIME["pg"]
    if not pg: raise HTTPException(500,"PG not ready")
    prof=pg.get_politician_profile(req.politician_id)
    if not prof: raise HTTPException(404,"Not found")
    return prof

# -------------------------- RAGFlow Trigger Stub --------------------------------------------------
def trigger_ragflow_pipeline(pipeline_url: str, payload: Dict[str,Any]):
    if not pipeline_url: return
    try:
        requests.post(pipeline_url, json=payload, timeout=15)
    except Exception as e:
        log(f"[ragflow] trigger failed: {e}")

# -------------------------- Main Orchestration ----------------------------------------------------
def main():
    parser=argparse.ArgumentParser(description="Civic Legislative Hub Orchestrator")
    parser.add_argument("--init-db", action="store_true", help="Initialize PostgreSQL schema.")
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
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--export-bloom", action="store_true")
    parser.add_argument("--build-profiles", action="store_true")
    parser.add_argument("--one-shot-query", type=str)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--populate-falkordb", action="store_true")
    parser.add_argument("--ragflow-trigger-url", type=str, default="")
    args=parser.parse_args()

    ensure_dirs("data/cache","data/local")

    pg=PGStore(); pg.connect()
    if args.init_db:
        pg.init_schema()

    new_docs: List[IngestedDocument] = []

    if args.sync_govinfo:
        cols=[c.strip().upper() for c in args.govinfo_collections.split(",") if c.strip()]
        gi=GovInfoIngestor(days=args.govinfo_days, collections=cols)
        new_docs.extend(gi.ingest())

    if args.sync_openstates:
        sts=[s.strip() for s in args.openstates_states.split(",") if s.strip()]
        osi=OpenStatesIngestor(states=sts, pages=args.openstates_pages)
        new_docs.extend(osi.ingest())

    if args.local_ingest:
        patterns=[p.strip() for p in args.local_patterns.split(",") if p.strip()]
        lfi=LocalFileIngestor(patterns)
        new_docs.extend(lfi.ingest())

    # Insert docs
    if new_docs:
        log(f"Inserting {len(new_docs)} documents...")
        for d in new_docs:
            try: pg.insert_document(d)
            except Exception as e: log(f"Insert fail {d.ext_id}: {e}")

    # Votes & Politicians via ProPublica
    if args.propublica_sync:
        chambers=[c.strip() for c in args.propublica_chambers.split(",") if c.strip()]
        ppi=ProPublicaVotesIngestor(congress=args.congress,chambers=chambers)
        votes=ppi.ingest()
        for v in votes:
            try:
                pg.upsert_vote(v)
            except Exception as e:
                log(f"Vote upsert error {v.get('vote_id')}: {e}")

    # Embeddings
    summarizer=None
    embedder=None
    if args.embed or args.serve or args.one_shot_query:
        summarizer=Summarizer()
        try:
            embedder=Embedder()
        except Exception as e:
            log(f"Embedder init failed: {e}")
    if args.embed and embedder:
        rows=pg.fetch_sections_to_embed(model=embedder.model_name, limit=4000)
        if rows:
            log(f"Embedding {len(rows)} sections...")
            texts=[r["text"] for r in rows]
            vecs=embedder.encode(texts)
            pg.insert_embeddings(embedder.model_name, [(rows[i]["id"],vecs[i]) for i in range(len(rows))])
            log("Embeddings stored.")

    # FalkorDB population from new docs (doc2graph)
    falkor=FalkorDBBridge()
    if args.populate_falkordb and falkor.enabled and new_docs:
        for d in new_docs:
            triples=doc_to_graph_triples(d.full_text[:20000])
            falkor.add_bill_triples(d.bill_id or d.ext_id, triples)

    # Neo4j optional export
    neo=Neo4jBridge()
    if neo.enabled and new_docs:
        neo.ensure_constraints()
        for d in new_docs:
            if d.bill_id:
                neo.upsert_bill(d.bill_id, d.title, d.jurisdiction)
    if args.export_bloom and neo.enabled:
        neo.export_bloom()

    # Politician profiles
    if args.build_profiles:
        pg.compute_politician_profiles()

    # Agent
    graphiti=GraphitiAdapter()
    memory=MCPMemoryAdapter()
    agent=None
    if embedder and summarizer:
        agent=CivicAgent(pg, embedder, summarizer, graphiti, memory)

    # RAGFlow trigger
    if args.ragflow_trigger_url and new_docs:
        trigger_ragflow_pipeline(args.ragflow_trigger_url, {"documents_ingested": len(new_docs)})

    # One shot query
    if args.one_shot_query and agent:
        ans=agent.answer(args.one_shot_query, plain=args.plain)
        print(json.dumps(ans, indent=2))

    if args.serve:
        RUNTIME["pg"]=pg
        RUNTIME["agent"]=agent
        RUNTIME["embedder"]=embedder
        RUNTIME["summarizer"]=summarizer
        import uvicorn
        log(f"Serving API on 0.0.0.0:{args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        pg.close()
        neo.close()

if __name__ == "__main__":
    main()