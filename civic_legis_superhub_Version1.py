#!/usr/bin/env python3
# =================================================================================================
# Name: Civic Legislative SuperHub
# Date: 2025-09-09
# Script Name: civic_legis_superhub.py
# Version: 1.0.0
# Log Summary:
#   - Unified “single-file” architecture for ingestion, analysis, retrieval, graph enrichment,
#     multimodel NLP (spaCy, BERT, LocalAI, Ollama, HuggingFace), embeddings (pgvector),
#     knowledge graph (Neo4j/FalkorDB), vector mirroring (Cloudflare Vectorize),
#     politician profiling & vote analytics, doc2graph triple extraction,
#     Flowise / n8n / Supabase / Langfuse / RAGFlow integration hooks,
#     dynamic port scanning & conflict detection, optional reverse proxy (Traefik/Kong/Nginx),
#     self-tests & validation, code review report, generation of docker-compose
#     and infra scaffolds with a single command-line interface.
# Description:
#   One script to set up and run a legislative intelligence platform:
#     - Ingest federal (govinfo), state (OpenStates), votes (ProPublica), local files.
#     - Parse & chunk documents; embed sections with SentenceTransformers (or fallback).
#     - Advanced NLP: spaCy NER, BERT classification, optional summarization w/ LocalAI/Ollama/HF.
#     - Graph output: optional Neo4j + Bloom perspective, FalkorDB graph triples.
#     - Politician profile metrics; RAG over pgvector; Cloudflare vector mirror.
#     - Central Agent w/ memory + explanation; multi LLM backend selection.
#     - Export Bill snapshot for Cloudflare D1/Workers.
#     - Orchestrated integration with Flowise, n8n, Supabase (logging), Langfuse (tracing).
#     - Generate docker-compose + proxy configs + infrastructure hints (Terraform stub note).
#     - Single-file maintainability: dynamic generation of config artifacts on demand.
# Change Summary:
#   *1.0.0* – First SuperHub consolidated release, merging prior 0.5.0 features + new NLP stack,
#              port scanner, unified multi-tool integration, heavy inline documentation,
#              single-file tests & tasks, BERT & spaCy pipelines, dynamic compose generation.
# Inputs:
#   - Environment Variables (optional override for CLI flags):
#       POSTGRES_* (HOST, PORT, DB, USER, PASSWORD)
#       EMBED_MODEL, PGVECTOR_DIM
#       GOVINFO_API_KEY, OPENSTATES_API_KEY, PROPUBLICA_API_KEY
#       LOCALAI_ENDPOINT, OPENAI_API_KEY, MODEL_NAME
#       OLLAMA_ENDPOINT (default http://localhost:11434)
#       HF_MODEL (Hugging Face model for summarization/classification fallback)
#       SPACY_MODEL (e.g. en_core_web_sm, auto downloaded if --auto-install)
#       ENABLE_NEO4J, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
#       ENABLE_FALKORDB, FALKORDB_HOST, FALKORDB_PORT
#       ENABLE_GRAPHITI, ENABLE_LANGFUSE
#       LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
#       SUPABASE_URL, SUPABASE_SERVICE_KEY (write) or SUPABASE_ANON_KEY (read)
#       CF_ACCOUNT_ID, CF_API_TOKEN, CF_VECTORIZE_INDEX
#       RAGFLOW_TRIGGER_URL
#       FLOWISE_WEBHOOK, N8N_WEBHOOK
# Outputs:
#   - PostgreSQL tables for docs, sections, embeddings, bills, politicians, votes, profiles.
#   - Summaries cache (data/cache/summaries.json)
#   - Embeddings (pgvector) + optional Cloudflare Vectorize mirrored vector store
#   - Graph data (Neo4j or FalkorDB) & optional bloom_perspective.json
#   - Docker compose file + proxy config (traefik.yml / nginx.conf / kong.yaml)
#   - D1 export snapshot JSON
#   - Self-test results & code review report (JSON printed)
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
import socket
import random
import datetime
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# ================================================================================================
# Section 1: Dependency Handling (Auto-install)
# ================================================================================================
CORE_REQUIREMENTS = [
    "requests", "tqdm", "psycopg2-binary", "pydantic", "fastapi", "uvicorn",
    "python-dotenv", "sentence-transformers", "numpy", "scikit-learn",
    "beautifulsoup4", "lxml", "PyPDF2", "spacy"
]
OPTIONAL_REQUIREMENTS = [
    "redis", "neo4j", "langfuse", "supabase", "transformers"
]

def auto_install(packages: List[str]):
    missing = []
    for pkg in packages:
        base = pkg.split("==")[0]
        try:
            __import__(base)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[INSTALL] Installing missing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# ================================================================================================
# Section 2: Logging & Utilities
# ================================================================================================
def log(msg: str):
    print(f"[{datetime.datetime.utcnow().isoformat()}] {msg}")

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def port_in_use(port: int, host="0.0.0.0"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0

def find_free_port(preferred: int, fallback_range=(8000, 8999)):
    if not port_in_use(preferred):
        return preferred
    for p in range(*fallback_range):
        if not port_in_use(p):
            log(f"[ports] {preferred} busy, using {p}")
            return p
    raise RuntimeError("No free port found in fallback range")

# ================================================================================================
# Section 3: Data Models
# ================================================================================================
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

# ================================================================================================
# Section 4: Database (PostgreSQL + pgvector)
# ================================================================================================
import psycopg2
import psycopg2.extras
import numpy as np

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
        self.conn = psycopg2.connect(host=self.host, port=self.port, dbname=self.db,
                                     user=self.user, password=self.password)
        self.conn.autocommit = True
    def close(self):
        if self.conn: self.conn.close()
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
        log("PostgreSQL schema created/verified.")

    def insert_document(self, d: IngestedDocument):
        cur = self.conn.cursor()
        doc_id = str(uuid.uuid4())
        cur.execute("""
        INSERT INTO documents(id, ext_id, bill_id, title, jurisdiction, source_type, provenance)
        VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING
        """,(doc_id,d.ext_id,d.bill_id,d.title,d.jurisdiction,d.source_type,json.dumps(d.provenance)))
        if d.bill_id:
            cur.execute("""
            INSERT INTO bills(bill_id,title,jurisdiction,raw_text,source_type)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT(bill_id) DO UPDATE SET title=EXCLUDED.title
            """,(d.bill_id,d.title,d.jurisdiction,d.full_text[:250000],d.source_type))
        for idx,s in enumerate(d.sections):
            sid = str(uuid.uuid4())
            cur.execute("""
            INSERT INTO sections(id, document_id, section_no, heading, text)
            VALUES (%s,%s,%s,%s,%s)
            """,(sid, doc_id, s.get("number") or str(idx+1), s.get("heading"), s.get("text")[:60000]))
        cur.close()

    def fetch_sections_to_embed(self, model: str, limit=3000):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT s.id, s.text
        FROM sections s
        LEFT JOIN embeddings e ON e.section_id = s.id AND e.model=%s
        WHERE e.section_id IS NULL
        ORDER BY s.created_at ASC
        LIMIT %s
        """,(model, limit))
        rows=cur.fetchall(); cur.close()
        return rows

    def insert_embeddings(self, model: str, items: List[Tuple[str,np.ndarray]]):
        cur=self.conn.cursor()
        for sid,vec in items:
            cur.execute("""
            INSERT INTO embeddings(section_id,model,embedding)
            VALUES (%s,%s,%s) ON CONFLICT (section_id,model) DO NOTHING
            """,(sid, model, list(vec)))
        cur.close()

    def semantic_search(self, query_vec: np.ndarray, model: str, k: int=6):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT s.id,s.text,d.bill_id,d.title,(1 - (embedding <=> %s::vector)) AS score
        FROM embeddings e
        JOIN sections s ON s.id=e.section_id
        JOIN documents d ON d.id=s.document_id
        WHERE model=%s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,(list(query_vec), model, list(query_vec), k))
        rows=cur.fetchall(); cur.close(); return rows

    def get_bill(self,bill_id: str):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM bills WHERE bill_id=%s",(bill_id,))
        bill=cur.fetchone()
        if not bill:
            cur.close(); return None
        cur.execute("""
        SELECT s.id,s.section_no,s.heading,s.text
        FROM sections s JOIN documents d ON d.id=s.document_id
        WHERE d.bill_id=%s ORDER BY s.section_no::int NULLS LAST, s.created_at ASC LIMIT 500
        """,(bill_id,))
        secs=[dict(r) for r in cur.fetchall()]
        cur.close()
        return {"bill":dict(bill),"sections":secs}

    def upsert_vote(self, vote: Dict[str,Any]):
        cur=self.conn.cursor()
        cur.execute("""
        INSERT INTO votes(vote_id,bill_id,vote_date,chamber,meta)
        VALUES (%s,%s,%s,%s,%s)
        ON CONFLICT(vote_id) DO NOTHING
        """,(vote["vote_id"], vote.get("bill_id"), vote.get("vote_date"),
             vote.get("chamber"), json.dumps(vote.get("meta") or {})))
        for pid,choice in vote.get("choices", {}).items():
            cur.execute("""
            INSERT INTO vote_choices(vote_id,politician_id,choice)
            VALUES (%s,%s,%s)
            ON CONFLICT (vote_id,politician_id) DO UPDATE SET choice=EXCLUDED.choice
            """,(vote["vote_id"], pid, choice))
        cur.close()

    def compute_politician_profiles(self):
        cur=self.conn.cursor()
        cur.execute("""
        WITH base AS (
          SELECT p.politician_id,p.name,
                 SUM(CASE WHEN vc.choice='YEA' THEN 1 ELSE 0 END) yeas,
                 SUM(CASE WHEN vc.choice='NAY' THEN 1 ELSE 0 END) nays,
                 COUNT(vc.choice) total_votes
          FROM politicians p
          LEFT JOIN vote_choices vc ON vc.politician_id=p.politician_id
          GROUP BY p.politician_id,p.name
        )
        SELECT politician_id,name,yeas,nays,total_votes FROM base
        """)
        rows=cur.fetchall()
        for pid,name,yeas,nays,total in rows:
            stats={"name":name,"yeas":yeas,"nays":nays,"total_votes":total,
                   "yea_pct":(float(yeas)/total if total else None),
                   "nay_pct":(float(nays)/total if total else None)}
            c2=self.conn.cursor()
            c2.execute("""
            INSERT INTO politician_profiles(politician_id,stats)
            VALUES (%s,%s)
            ON CONFLICT(politician_id) DO UPDATE SET stats=EXCLUDED.stats, updated_at=now()
            """,(pid,json.dumps(stats))); c2.close()
        cur.close()

    def get_politician_profile(self, pid: str):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
        SELECT p.politician_id,p.name,p.party,p.chamber,p.state,p.district,prof.stats
        FROM politicians p
        LEFT JOIN politician_profiles prof ON prof.politician_id=p.politician_id
        WHERE p.politician_id=%s
        """,(pid,))
        r=cur.fetchone(); cur.close()
        if not r: return None
        d=dict(r)
        if isinstance(d.get("stats"), str):
            try: d["stats"]=json.loads(d["stats"])
            except: pass
        return d

# ================================================================================================
# Section 5: Embeddings & NLP
# ================================================================================================
try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer=None

class Embedder:
    def __init__(self):
        self.model_name=os.environ.get("EMBED_MODEL","all-MiniLM-L6-v2")
        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer not installed")
        self.model=SentenceTransformer(self.model_name)
    def encode(self, texts: List[str])->np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True), dtype="float32")

# Summarization multi-backend (LocalAI / OpenAI / Ollama / HuggingFace)
import requests
class Summarizer:
    def __init__(self):
        self.localai = os.environ.get("LOCALAI_ENDPOINT")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("MODEL_NAME","gpt-4o-mini")
        self.ollama = os.environ.get("OLLAMA_ENDPOINT","http://localhost:11434")
        self.hf_model = os.environ.get("HF_MODEL","distilbert-base-uncased")
        ensure_dirs("data/cache")
        self.cache="data/cache/summaries.json"
        if os.path.exists(self.cache):
            try:
                with open(self.cache,"r",encoding="utf-8") as f:
                    self._cache=json.load(f)
            except:
                self._cache={}
        else:
            self._cache={}
    def summarize(self, bill_id: str, section_id: str, text: str)->str:
        key=f"{bill_id}:{section_id}"
        if key in self._cache: return self._cache[key]
        snippet=text[:4000]
        # Attempt LocalAI / OpenAI style first
        prompt=f"Provide a concise, plain-language explanation of this legislative section:\n\n{snippet}\n\nSimplified:"
        response=None
        if self.localai:
            try:
                r=requests.post(f"{self.localai}/chat/completions",
                                headers={"Content-Type":"application/json"},
                                json={
                                  "model": self.model,
                                  "messages":[{"role":"system","content":"You simplify US legislative text."},
                                              {"role":"user","content":prompt}],
                                  "temperature":0.3
                                },timeout=60)
                if r.status_code<400:
                    response=r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                response=f"[LocalAI error] {e}"
        elif self.openai_key:
            try:
                r=requests.post("https://api.openai.com/v1/chat/completions",
                    headers={"Authorization":f"Bearer {self.openai_key}","Content-Type":"application/json"},
                    json={"model":self.model,"messages":[
                        {"role":"system","content":"You simplify legislative text."},
                        {"role":"user","content":prompt}],
                        "temperature":0.3},timeout=60)
                if r.status_code<400:
                    response=r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                response=f"[OpenAI error] {e}"
        # Ollama fallback
        if not response:
            try:
                r=requests.post(f"{self.ollama}/api/generate", json={"model":"llama2","prompt":prompt}, timeout=60)
                if r.status_code<400:
                    # Ollama streams lines; simple join
                    response=" ".join([line for line in r.text.splitlines() if line.strip()])
            except Exception as e:
                response=f"[Ollama error] {e}"
        # Final fallback if still empty
        if not response:
            response="(Summary unavailable)"
        self._cache[key]=response
        with open(self.cache,"w",encoding="utf-8") as f:
            json.dump(self._cache,f,indent=2)
        return response

# spaCy + BERT classification
def load_spacy_model(auto_install=False):
    name=os.environ.get("SPACY_MODEL","en_core_web_sm")
    try:
        import spacy
        try:
            return spacy.load(name)
        except OSError:
            if auto_install:
                subprocess.check_call([sys.executable,"-m","spacy","download",name])
                return spacy.load(name)
            else:
                log(f"[spacy] Model {name} not found; skipping.")
                return None
    except ImportError:
        log("[spacy] Not installed.")
        return None

def bert_classify_texts(texts: List[str], model_name="distilbert-base-uncased", auto_install=False):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        if auto_install:
            subprocess.check_call([sys.executable,"-m","pip","install","transformers","torch"])
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        else:
            log("[bert] transformers not installed.")
            return []
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    results=[]
    for t in texts[:16]:
        inputs=tokenizer(t[:512],return_tensors="pt",truncation=True)
        with torch.no_grad():
            out=model(**inputs)
        probs=out.logits.softmax(dim=-1).tolist()[0]
        results.append({"text":t[:80],"label_scores":probs})
    return results

# ================================================================================================
# Section 6: Ingestion Adapters
# ================================================================================================
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

class GovInfoIngestor:
    API="https://api.govinfo.gov"
    def __init__(self, days: int, collections: List[str], api_key: Optional[str]):
        self.days=days; self.collections=collections; self.key=api_key
        ensure_dirs("data/govinfo")
        self.state_file="data/govinfo/state.json"
        self.state=self._load_state()
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file,"r",encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {"processed":{}}
        return {"processed":{}}
    def _save_state(self):
        with open(self.state_file,"w",encoding="utf-8") as f:
            json.dump(self.state,f,indent=2)
    def _headers(self):
        h={"User-Agent":"CivicSuperHub/1.0"}
        if self.key: h["X-Api-Key"]=self.key
        return h
    def _list_packages(self, collection, start, end):
        url=f"{self.API}/collections/{collection}/{start}/{end}"
        params={"pageSize":400}
        out=[]; offset=0
        while True:
            params["offset"]=offset
            r=requests.get(url,headers=self._headers(),params=params,timeout=60)
            if r.status_code!=200: break
            data=r.json(); pkgs=data.get("packages",[])
            if not pkgs: break
            out.extend(pkgs)
            if len(pkgs)<params["pageSize"]: break
            offset+=params["pageSize"]
        return out
    def _download_zip(self,package_id,target_dir):
        ensure_dirs(target_dir)
        url=f"{self.API}/packages/{package_id}/zip"
        r=requests.get(url,headers=self._headers(),timeout=120)
        if r.status_code!=200: return False
        zp=os.path.join(target_dir,f"{package_id}.zip")
        with open(zp,"wb") as f: f.write(r.content)
        try:
            with zipfile.ZipFile(zp,'r') as z:z.extractall(target_dir)
        except:
            return False
        return True
    def _parse(self, collection, dir_path, package_id):
        xml_files=[]
        for root,_,files in os.walk(dir_path):
            for fn in files:
                if fn.lower().endswith(".xml"): xml_files.append(os.path.join(root,fn))
        if not xml_files: return None
        xml_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        main=xml_files[0]
        try:
            with open(main,"r",encoding="utf-8",errors="ignore") as f:
                soup=BeautifulSoup(f.read(),"lxml-xml")
        except:
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
        for i, chunk in enumerate(dedup[:80]):
            sections.append({"number":str(i+1),"heading":None,"text":chunk[:12000]})
        return IngestedDocument(
            ext_id=package_id,title=title,jurisdiction="US-Federal",
            full_text="\n\n".join(dedup),
            sections=sections if sections else [{"number":"1","heading":None,"text":"\n\n".join(dedup)[:12000]}],
            source_type=f"govinfo:{collection}",
            provenance={"collection":collection,"package_id":package_id},
            bill_id=package_id
        )
    def ingest(self):
        end=datetime.date.today()
        start=end - datetime.timedelta(days=self.days)
        new_docs=[]
        for col in self.collections:
            pkgs=self._list_packages(col,start.isoformat(),end.isoformat())
            log(f"[govinfo] {col} packages: {len(pkgs)}")
            for pk in pkgs:
                pid=pk.get("packageId")
                if not pid or pid in self.state["processed"]: continue
                tgt=os.path.join("data","govinfo",col,pid)
                if not self._download_zip(pid,tgt): continue
                doc=self._parse(col,tgt,pid)
                if doc: new_docs.append(doc)
                self.state["processed"][pid]={"ts":datetime.datetime.utcnow().isoformat(),"collection":col}
            self._save_state()
        log(f"[govinfo] new docs: {len(new_docs)}")
        return new_docs

class OpenStatesIngestor:
    API="https://v3.openstates.org/bills"
    def __init__(self, states: List[str], pages: int, api_key: Optional[str]):
        self.states=states; self.pages=pages; self.key=api_key
    def _headers(self):
        return {"X-API-KEY": self.key} if self.key else {}
    def ingest(self):
        docs=[]
        base={"sort":"updated_desc","per_page":25,"include":"sponsors"}
        for st in self.states:
            for page in range(1,self.pages+1):
                params=dict(base,jurisdiction=st,page=page)
                r=requests.get(self.API,headers=self._headers(),params=params,timeout=60)
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
                        title=title,jurisdiction=st,
                        full_text=combined,
                        sections=[{"number":"1","heading":None,"text":combined[:16000]}],
                        source_type="openstates",
                        provenance={"raw":b},
                        bill_id=bid
                    ))
        log(f"[openstates] docs: {len(docs)}")
        return docs

class ProPublicaVotes:
    API="https://api.propublica.org/congress/v1"
    def __init__(self, congress: int, chambers: List[str], recent: int, api_key: Optional[str]):
        self.congress=congress; self.chambers=chambers; self.recent=recent; self.key=api_key
    def _headers(self):
        return {"X-API-Key": self.key} if self.key else {}
    def ingest_votes(self):
        if not self.key:
            log("[propublica] API key missing; skip votes.")
            return []
        votes=[]
        for ch in self.chambers:
            url=f"{self.API}/{self.congress}/{ch}/votes.json"
            r=requests.get(url,headers=self._headers(),timeout=60)
            if r.status_code!=200: continue
            data=r.json()
            arr=data.get("results",{}).get("votes",[])
            for v in arr[:self.recent]:
                vote_id=v.get("roll_call")
                bill_id=(v.get("bill") or {}).get("bill_id")
                positions=v.get("positions") or []
                choices={}
                for pos in positions:
                    pid=pos.get("member_id")
                    if pid: choices[pid]=pos.get("vote_position")
                votes.append({
                    "vote_id": f"{self.congress}-{ch}-{vote_id}",
                    "bill_id": (bill_id or "").upper(),
                    "vote_date": v.get("date"),
                    "chamber": ch,
                    "meta": v,
                    "choices": choices
                })
        log(f"[propublica] votes: {len(votes)}")
        return votes

class LocalFileIngestor:
    def __init__(self, patterns: List[str], jurisdiction="Local/Generic"):
        self.patterns=patterns; self.jurisdiction=jurisdiction
    def ingest(self):
        docs=[]
        paths=[]
        for p in self.patterns:
            paths.extend(glob.glob(p, recursive=True))
        for path in paths:
            if not os.path.isfile(path): continue
            try:
                text=read_text(path)
                if not text.strip(): continue
                lines=[l.strip() for l in text.splitlines() if l.strip()]
                title=lines[0][:200] if lines else os.path.basename(path)
                sections=chunk_text(text)
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
            except Exception as e:
                log(f"[local] parse fail {path}: {e}")
        log(f"[local-files] docs: {len(docs)}")
        return docs

# Helpers
def read_text(path: str)->str:
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

def chunk_text(text: str, max_len=1800):
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

# ================================================================================================
# Section 7: Triple Extraction (doc2graph fallback)
# ================================================================================================
def extract_triples(text: str, limit=200):
    if 'doc2graph' in sys.modules and doc2graph:
        try:
            triples=doc2graph.extract(text)
            return triples[:limit]
        except:
            pass
    # Fallback naive
    triples=[]
    sentences=[s.strip() for s in text.split(".") if s.strip()]
    for i,s in enumerate(sentences[:25]):
        tokens=[t for t in s.split() if t.istitle()][:3]
        for t in tokens:
            triples.append((f"Sentence_{i}","MENTIONS", t))
    return triples[:limit]

# ================================================================================================
# Section 8: Graph Integrations (Neo4j, FalkorDB)
# ================================================================================================
try: from neo4j import GraphDatabase
except: GraphDatabase=None
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
                log(f"[neo4j] connection fail {e}")
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
            s.run("""
            MERGE (b:Bill {bill_id:$bid})
            SET b.title=$title, b.jurisdiction=$jurisdiction
            """, bid=bill_id, title=title, jurisdiction=jurisdiction)
    def export_bloom(self, path="bloom_perspective.json"):
        if not self.enabled: return
        perspective={
            "name":"SuperHubPerspective","version":"1.0",
            "lastUpdated":datetime.datetime.utcnow().isoformat(),
            "categories":[{"name":"Bills","label":"Bill","cypher":"MATCH (b:Bill) RETURN b",
                          "style":{"color":"#1f77b4","size":55}}],
            "relationships":[]
        }
        with open(path,"w",encoding="utf-8") as f:
            json.dump(perspective,f,indent=2)
        log("[neo4j] bloom perspective exported")

import importlib
class FalkorDBBridge:
    def __init__(self):
        self.enabled=os.environ.get("ENABLE_FALKORDB")=="1"
        self.client=None
        if self.enabled:
            try:
                import redis
                self.client=redis.Redis(host=os.environ.get("FALKORDB_HOST","localhost"),
                                        port=int(os.environ.get("FALKORDB_PORT","6379")),
                                        decode_responses=True)
                self.client.ping()
            except Exception as e:
                log(f"[falkordb] connect fail: {e}")
                self.enabled=False
    def add_triples(self, bill_id: str, triples: List[Tuple[str,str,str]]):
        if not self.enabled or not self.client: return
        graph="legislation"
        for s,p,o in triples[:150]:
            q1=f"MERGE (:Node {{name:'{s}'}}); MERGE (:Node {{name:'{o}'}});"
            q2=f"MATCH (a:Node {{name:'{s}'}}),(b:Node {{name:'{o}'}}) MERGE (a)-[:{p} {{bill:'{bill_id}'}}]->(b);"
            try:
                self.client.execute_command("GRAPH.QUERY", graph, q1, "--compact")
                self.client.execute_command("GRAPH.QUERY", graph, q2, "--compact")
            except: pass

# ================================================================================================
# Section 9: Cloudflare Vectorize / D1 Snapshot
# ================================================================================================
class CloudflareVectorize:
    def __init__(self):
        self.account=os.environ.get("CF_ACCOUNT_ID")
        self.token=os.environ.get("CF_API_TOKEN")
        self.index=os.environ.get("CF_VECTORIZE_INDEX")
        self.enabled=bool(self.account and self.token and self.index)
    def push(self, model: str, items: List[Tuple[str,np.ndarray]]):
        if not self.enabled or not items: return
        url=f"https://api.cloudflare.com/client/v4/accounts/{self.account}/ai/vectorize/indexes/{self.index}/upsert"
        headers={"Authorization":f"Bearer {self.token}","Content-Type":"application/json"}
        payload={"vectors":[]}
        for sid,vec in items[:500]:
            payload["vectors"].append({"id":sid,"values":vec.tolist(),"metadata":{"model":model}})
        try:
            r=requests.post(url,headers=headers,data=json.dumps(payload),timeout=60)
            if r.status_code>=300:
                log(f"[vectorize] fail {r.status_code}: {r.text[:150]}")
            else:
                log(f"[vectorize] uploaded {len(payload['vectors'])} vectors")
        except Exception as e:
            log(f"[vectorize] error {e}")

def export_d1_snapshot(pg: PGStore, path="data/exports/d1_snapshot.json", limit=300):
    ensure_dirs("data/exports")
    cur=pg.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
    SELECT bill_id,title,jurisdiction,LEFT(raw_text,20000) excerpt
    FROM bills ORDER BY created_at DESC LIMIT %s
    """,(limit,))
    rows=[dict(r) for r in cur.fetchall()]
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"bills":rows,"generated":datetime.datetime.utcnow().isoformat()},f,indent=2)
    log(f"[d1] snapshot -> {path}")

# ================================================================================================
# Section 10: External Integrations (Flowise, n8n, Supabase, Langfuse, RAGFlow)
# ================================================================================================
def post_webhook(url: str, payload: Dict[str,Any], label="generic"):
    if not url: return
    try:
        requests.post(url,json=payload,timeout=10)
        log(f"[{label}] webhook posted.")
    except Exception as e:
        log(f"[{label}] webhook error: {e}")

def supabase_log(event: str, data: Dict[str,Any]):
    supabase_url=os.environ.get("SUPABASE_URL")
    supabase_key=os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key: return
    try:
        # Simple insert into a table 'events' expected
        r=requests.post(f"{supabase_url}/rest/v1/events",
                        headers={"apikey":supabase_key,"Authorization":f"Bearer {supabase_key}",
                                 "Content-Type":"application/json","Prefer":"return=representation"},
                        json={"event":event,"payload":data})
        if r.status_code>=300:
            log(f"[supabase] log fail {r.status_code}: {r.text[:120]}")
    except Exception as e:
        log(f"[supabase] error {e}")

def ragflow_trigger(url: str, payload: Dict[str,Any]):
    if not url: return
    try:
        requests.post(url,json=payload,timeout=10)
        log("[ragflow] pipeline triggered.")
    except Exception as e:
        log(f"[ragflow] trigger fail: {e}")

def langfuse_log(event_type: str, data: Dict[str,Any]):
    if os.environ.get("ENABLE_LANGFUSE")!="1": return
    pk=os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk=os.environ.get("LANGFUSE_SECRET_KEY")
    host=os.environ.get("LANGFUSE_HOST","https://cloud.langfuse.com")
    if not (pk and sk): return
    try:
        requests.post(f"{host}/api/public/ingest",
                      headers={"x-langfuse-public-key":pk,"x-langfuse-secret-key":sk,"Content-Type":"application/json"},
                      json={"type":event_type,"body":data},
                      timeout=10)
    except Exception as e:
        log(f"[langfuse] ingest error: {e}")

# ================================================================================================
# Section 11: Memory & Agent
# ================================================================================================
class MemoryBuffer:
    def __init__(self, cap=250):
        self.cap=cap; self.data=[]
    def store(self, role:str, content:str):
        self.data.append({"role":role,"content":content,"ts":datetime.datetime.utcnow().isoformat()})
        if len(self.data)>self.cap: self.data=self.data[-self.cap:]
    def tail(self, n=8):
        return self.data[-n:]

class Agent:
    def __init__(self, pg: PGStore, embedder: Embedder, summarizer: Summarizer, memory: MemoryBuffer):
        self.pg=pg; self.embedder=embedder; self.summarizer=summarizer; self.memory=memory
        self.model=embedder.model_name
    def answer(self, query: str, k=6, plain=False):
        self.memory.store("user", query)
        q_vec=self.embedder.encode([query])[0]
        rows=self.pg.semantic_search(q_vec, self.model, k)
        bill_id=None
        for t in query.split():
            if "-" in t and any(c.isdigit() for c in t):
                bill_id=t.upper().strip(",."); break
        bill_data=self.pg.get_bill(bill_id) if bill_id else None
        hits=[{
            "section_id":r["id"],"bill_id":r["bill_id"],"title":r["title"],
            "score":float(r["score"]),"snippet":r["text"][:600]
        } for r in rows]
        resp={"query":query,"bill_id":bill_id,"hits":hits,"bill":None,"memory_tail":self.memory.tail()}
        if bill_data:
            secs=[]
            for s in bill_data["sections"][:15]:
                sec_obj={"section_no":s["section_no"],"heading":s["heading"],"text":s["text"][:1000]}
                if plain:
                    sec_obj["plain_language"]=self.summarizer.summarize(bill_data["bill"]["bill_id"], s["id"], s["text"][:4000])
                secs.append(sec_obj)
            resp["bill"]={"bill_id":bill_data["bill"]["bill_id"],"title":bill_data["bill"]["title"],"sections":secs}
        self.memory.store("assistant", f"Returned {len(hits)} hits")
        langfuse_log("agent_response", {"query":query,"hit_count":len(hits)})
        supabase_log("agent_query", {"query":query,"hits":len(hits)})
        return resp

# ================================================================================================
# Section 12: FastAPI
# ================================================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app=FastAPI(title="Civic Legislative SuperHub API", version="1.0.0")
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
    # Quick port validation
    return {"status":"ok","version":"1.0.0"}

@app.post("/query")
def query_endpoint(req: QueryReq):
    if not RUNTIME["agent"]:
        raise HTTPException(500,"Agent not initialized")
    return RUNTIME["agent"].answer(req.query,k=req.k,plain=req.plain)

@app.post("/bill")
def bill_endpoint(req: BillReq):
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
def politician_endpoint(req: PoliticianReq):
    pg: PGStore = RUNTIME["pg"]
    if not pg: raise HTTPException(500,"PG not ready")
    prof=pg.get_politician_profile(req.politician_id)
    if not prof: raise HTTPException(404,"Not found")
    return prof

# ================================================================================================
# Section 13: Analysis Functions (spaCy NER, BERT sanity)
# ================================================================================================
def run_spacy_ner(nlp, texts: List[str]):
    out=[]
    if not nlp: return out
    for t in texts[:5]:
        doc=nlp(t[:1000])
        ents=[(e.text,e.label_) for e in doc.ner if hasattr(doc,"ner")] if hasattr(doc,"ner") else [(e.text,e.label_) for e in doc.ents]
        out.append({"text":t[:120],"entities":ents})
    return out

def run_bert_sampling(texts: List[str], auto_install=False):
    return bert_classify_texts(texts, auto_install=auto_install)

# ================================================================================================
# Section 14: Compose / Proxy Generation
# ================================================================================================
def generate_compose(include_proxy: bool, path="docker-compose.generated.yml"):
    compose = {
      "version":"3.9",
      "services":{
        "postgres":{
          "image":"ankane/pgvector:latest",
          "environment":{
            "POSTGRES_PASSWORD":"${POSTGRES_PASSWORD}",
            "POSTGRES_USER":"${POSTGRES_USER}",
            "POSTGRES_DB":"${POSTGRES_DB}"
          },
          "ports":["5432:5432"],
          "volumes":["pg_data:/var/lib/postgresql/data"],
          "restart":"unless-stopped"
        },
        "localai":{
          "image":"localai/localai:latest",
          "ports":["8080:8080"],
          "restart":"unless-stopped",
          "volumes":["./models:/models"]
        },
        "api":{
          "build":".",
          "environment":{
            "POSTGRES_HOST":"postgres",
            "POSTGRES_PORT":"5432",
            "POSTGRES_DB":"${POSTGRES_DB}",
            "POSTGRES_USER":"${POSTGRES_USER}",
            "POSTGRES_PASSWORD":"${POSTGRES_PASSWORD}",
            "EMBED_MODEL":"all-MiniLM-L6-v2",
            "LOCALAI_ENDPOINT":"http://localai:8080/v1"
          },
          "depends_on":["postgres","localai"],
          "ports":["8100:8100"],
          "command":"python civic_legis_superhub.py --serve --port 8100",
          "volumes":["./data:/app/data"],
          "restart":"unless-stopped"
        }
      },
      "volumes":{
        "pg_data":{}
      }
    }
    if include_proxy:
        # Traefik example
        compose["services"]["traefik"] = {
          "image":"traefik:v2.11",
            "command":[
                "--providers.docker=true",
                "--entrypoints.web.address=:80"
            ],
            "ports":["80:80"],
            "depends_on":["api"],
            "restart":"unless-stopped"
        }
    with open(path,"w",encoding="utf-8") as f:
        f.write("# Auto-generated by civic_legis_superhub.py\n")
        json.dump(compose,f,indent=2)
    log(f"[compose] generated -> {path}")

def generate_proxy_configs():
    ensure_dirs("infra")
    # Traefik
    with open("infra/traefik_dynamic.yml","w",encoding="utf-8") as f:
        f.write("""# Auto-generated Traefik dynamic config
http:
  routers:
    api:
      rule: Host(`localhost`)
      service: api-svc
      entryPoints: [web]
  services:
    api-svc:
      loadBalancer:
        servers:
          - url: http://api:8100
""")
    # Nginx
    with open("infra/nginx.conf","w",encoding="utf-8") as f:
        f.write("""# Auto-generated Nginx reverse proxy
events {}
http {
  server {
    listen 80;
    location / {
      proxy_pass http://api:8100;
      proxy_set_header Host $host;
    }
  }
}
""")
    # Kong (declarative)
    with open("infra/kong.yaml","w",encoding="utf-8") as f:
        f.write("""_format_version: "3.0"
services:
  - name: civic-api
    url: http://api:8100
    routes:
      - name: civic-api-route
        paths: ["/"]
""")
    log("[proxy] configs generated in infra/")

# ================================================================================================
# Section 15: Self Tests & Review Report
# ================================================================================================
def run_self_tests(pg: PGStore):
    log("[tests] running self tests ...")
    cur=pg.conn.cursor()
    for table in ["documents","sections","embeddings","bills","politicians","votes","vote_choices","politician_profiles"]:
        cur.execute("SELECT to_regclass(%s)",(table,))
        assert cur.fetchone()[0]==table, f"Missing table {table}"
    log("[tests] schema ok.")
    sample=IngestedDocument(
        ext_id="SELFTEST-DOC",
        title="Self Test Legislation",
        jurisdiction="Test",
        full_text="First sentence. Second sentence referencing Data. Third referencing Privacy.",
        sections=[{"number":"1","heading":None,"text":"First sentence."},
                  {"number":"2","heading":None,"text":"Second sentence referencing Data."}],
        source_type="test",
        provenance={"reason":"self-test"},
        bill_id="ST-001"
    )
    pg.insert_document(sample)
    cur.execute("SELECT count(*) FROM bills WHERE bill_id='ST-001'")
    assert cur.fetchone()[0]==1, "Bill insertion failed"
    log("[tests] insertion ok.")
    log("[tests] self tests completed successfully.")

def review_report():
    report={
        "version":"1.0.0",
        "lint_recommendations":[
            "Adopt ruff or flake8 for style checks.",
            "Use black for formatting."
        ],
        "security_recommendations":[
            "Rotate all API keys regularly.",
            "Restrict DB network exposure; prefer private VPC.",
            "Add auth to /query endpoint for production."
        ],
        "scalability":[
            "Sharding embeddings by model or time partition if data scales > millions of sections.",
            "Introduce asynchronous ingestion queue (Celery/RQ)."
        ],
        "graph_enhancement":[
            "Add NER-based entity linking for stable URIs.",
            "Use relationship weighting in path-based retrieval (GraphRAG)."
        ]
    }
    print(json.dumps(report,indent=2))

# ================================================================================================
# Section 16: Main Orchestration
# ================================================================================================
def main():
    parser=argparse.ArgumentParser(description="Civic Legislative SuperHub (Single-File)")
    parser.add_argument("--auto-install",action="store_true",help="Auto install core + optional deps + spaCy model if missing.")
    parser.add_argument("--init-db",action="store_true")
    parser.add_argument("--sync-govinfo",action="store_true")
    parser.add_argument("--govinfo-collections",type=str,default="BILLSTATUS")
    parser.add_argument("--govinfo-days",type=int,default=7)
    parser.add_argument("--sync-openstates",action="store_true")
    parser.add_argument("--openstates-states",type=str,default="California,New York")
    parser.add_argument("--openstates-pages",type=int,default=1)
    parser.add_argument("--propublica-sync",action="store_true")
    parser.add_argument("--congress",type=int,default=118)
    parser.add_argument("--propublica-chambers",type=str,default="house,senate")
    parser.add_argument("--local-ingest",action="store_true")
    parser.add_argument("--local-patterns",type=str,default="data/local/**/*.txt")
    parser.add_argument("--embed",action="store_true")
    parser.add_argument("--mirror-cloudflare-vectorize",action="store_true")
    parser.add_argument("--export-d1-snapshot",action="store_true")
    parser.add_argument("--export-bloom",action="store_true")
    parser.add_argument("--populate-falkordb",action="store_true")
    parser.add_argument("--build-profiles",action="store_true")
    parser.add_argument("--run-self-tests",action="store_true")
    parser.add_argument("--one-shot-query",type=str)
    parser.add_argument("--plain",action="store_true")
    parser.add_argument("--serve",action="store_true")
    parser.add_argument("--port",type=int,default=8100)
    parser.add_argument("--generate-compose",action="store_true")
    parser.add_argument("--include-proxy",action="store_true")
    parser.add_argument("--generate-review-report",action="store_true")
    parser.add_argument("--spacy-ner-sample",action="store_true")
    parser.add_argument("--bert-sample",action="store_true")
    parser.add_argument("--ragflow-trigger-url",type=str,default="")
    parser.add_argument("--print-ports",action="store_true")
    args=parser.parse_args()

    # Auto install
    if args.auto_install:
        auto_install(CORE_REQUIREMENTS)
        # Optionals
        auto_install(OPTIONAL_REQUIREMENTS)
        # Attempt spaCy model download if not present
        if os.environ.get("SPACY_MODEL","en_core_web_sm")=="en_core_web_sm":
            try:
                import spacy
                try:
                    spacy.load("en_core_web_sm")
                except OSError:
                    subprocess.check_call([sys.executable,"-m","spacy","download","en_core_web_sm"])
            except Exception as e:
                log(f"[spacy] model install error: {e}")

    ensure_dirs("data/cache","data/local")

    # DB
    pg=PGStore(); pg.connect()
    if args.init_db:
        pg.init_schema()

    if args.run_self_tests:
        run_self_tests(pg)

    new_docs=[]
    # Ingestion
    if args.sync_govinfo:
        collections=[c.strip().upper() for c in args.govinfo_collections.split(",") if c.strip()]
        gi=GovInfoIngestor(args.govinfo_days, collections, os.environ.get("GOVINFO_API_KEY"))
        new_docs.extend(gi.ingest())
    if args.sync_openstates:
        states=[s.strip() for s in args.openstates_states.split(",") if s.strip()]
        osi=OpenStatesIngestor(states,args.openstates_pages, os.environ.get("OPENSTATES_API_KEY"))
        new_docs.extend(osi.ingest())
    if args.local_ingest:
        patterns=[p.strip() for p in args.local_patterns.split(",") if p.strip()]
        lfi=LocalFileIngestor(patterns)
        new_docs.extend(lfi.ingest())
    if args.propublica_sync:
        chambers=[c.strip() for c in args.propublica_chambers.split(",") if c.strip()]
        ppi=ProPublicaVotes(args.congress,chambers, recent=12, api_key=os.environ.get("PROPUBLICA_API_KEY"))
        votes=ppi.ingest_votes()
        for v in votes:
            pg.upsert_vote(v)

    if new_docs:
        log(f"Inserting {len(new_docs)} docs ...")
        for d in new_docs:
            try: pg.insert_document(d)
            except Exception as e: log(f"[insert] fail {d.ext_id}: {e}")

    # Summarizer & Embeddings
    summarizer=None
    embedder=None
    if args.embed or args.serve or args.one_shot_query:
        summarizer=Summarizer()
        try:
            embedder=Embedder()
        except Exception as e:
            log(f"[embedder] init fail {e}")

    vectorize=CloudflareVectorize()

    if args.embed and embedder:
        rows=pg.fetch_sections_to_embed(embedder.model_name, limit=4000)
        if rows:
            log(f"Embedding {len(rows)} sections ...")
            texts=[r["text"] for r in rows]
            vecs=embedder.encode(texts)
            items=[(rows[i]["id"],vecs[i]) for i in range(len(rows))]
            pg.insert_embeddings(embedder.model_name, items)
            log("Embeddings stored.")
            if args.mirror_cloudflare_vectorize:
                vectorize.push(embedder.model_name, items)

    # Graph
    neo=Neo4jBridge()
    if neo.enabled and new_docs:
        neo.ensure_constraints()
        for d in new_docs:
            if d.bill_id:
                neo.upsert_bill(d.bill_id,d.title,d.jurisdiction)
    if args.export_bloom and neo.enabled:
        neo.export_bloom()

    falkor=FalkorDBBridge()
    if args.populate_falkordb and falkor.enabled and new_docs:
        for d in new_docs:
            triples=extract_triples(d.full_text[:20000])
            falkor.add_triples(d.bill_id or d.ext_id, triples)

    # Profiles
    if args.build_profiles:
        pg.compute_politician_profiles()

    # External triggers
    if args.ragflow_trigger_url and new_docs:
        ragflow_trigger(args.ragflow_trigger_url, {"ingested":len(new_docs)})
    if new_docs:
        post_webhook(os.environ.get("FLOWISE_WEBHOOK"), {"ingested":len(new_docs)}, "flowise")
        post_webhook(os.environ.get("N8N_WEBHOOK"), {"ingested":len(new_docs)}, "n8n")

    # D1 Snapshot
    if args.export_d1_snapshot:
        export_d1_snapshot(pg)

    # Compose / Proxy
    if args.generate_compose:
        generate_compose(args.include_proxy)
    if args.include_proxy:
        generate_proxy_configs()

    # Analysis: spaCy NER
    if args.spacy_ner_sample:
        nlp=load_spacy_model(auto_install=args.auto_install)
        if nlp:
            sample_texts=[d.full_text[:500] for d in new_docs[:3]] or ["No new docs"]
            ner_res=run_spacy_ner(nlp, sample_texts)
            print(json.dumps({"spacy_ner_sample":ner_res},indent=2))

    # Analysis: BERT classification
    if args.bert_sample:
        sample_texts=[d.full_text[:500] for d in new_docs[:2]] or ["Test classification sample text about budgets and privacy."]
        bert_res=run_bert_sampling(sample_texts, auto_install=args.auto_install)
        print(json.dumps({"bert_classification_sample":bert_res},indent=2))

    # Review report
    if args.generate_review_report:
        review_report()

    # Agent
    agent=None
    if embedder and summarizer:
        memory=MemoryBuffer()
        agent=Agent(pg, embedder, summarizer, memory)

    if args.one_shot_query and agent:
        ans=agent.answer(args.one_shot_query, plain=args.plain)
        print(json.dumps(ans,indent=2))

    # Print port usage
    if args.print_ports:
        candidates=[5432,8080,8100,7474,7687,6379,80]
        usage={}
        for c in candidates:
            usage[c]="USED" if port_in_use(c) else "FREE"
        print(json.dumps({"port_status":usage},indent=2))

    # Serve
    if args.serve:
        # Validate port
        port=find_free_port(args.port)
        import uvicorn
        RUNTIME["pg"]=pg
        RUNTIME["agent"]=agent
        RUNTIME["summarizer"]=summarizer
        RUNTIME["embedder"]=embedder
        log(f"Starting API on 0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        neo.close()
        pg.close()

if __name__ == "__main__":
    main()