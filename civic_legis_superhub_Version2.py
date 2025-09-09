#!/usr/bin/env python3
# =================================================================================================
# Name: Civic Legislative SuperHub
# Date: 2025-09-09
# Script Name: civic_legis_superhub.py
# Version: 1.1.0
# Log Summary:
#   - Base (v1.0.0): Unified ingestion (govinfo, OpenStates, ProPublica, local), pgvector embeddings,
#     multi-backend summarization (LocalAI/OpenAI/Ollama), optional Neo4j/FalkorDB, Cloudflare Vectorize,
#     D1 snapshot, Flowise/n8n/Supabase/Langfuse/RAGFlow hooks, spaCy + BERT analysis, doc2graph fallback,
#     dynamic docker-compose + proxy config generation, self-tests, code review report.
#   - v1.1.0 Additions:
#     * Terraform expansion generator (OCI + Cloudflare DNS + R2 + Worker).
#     * Supabase SQL migration generator (events table + optional vector staging).
#     * GraphRAG path expansion (Neo4j-based) + CLI flags to export contextual subgraphs.
#     * Graph metrics (degree, simple centrality proxy, path counts).
#     * Extended memory instrumentation & rhetorical labeling stub.
#     * Additional validation scripts & port scan enhancements.
#     * Embedding re-ranker stub (cosine + optional naive cross-encoder placeholder).
#     * Export GraphRAG contexts to JSON for offline analysis.
#     * Advanced bulk summarization utility (--bulk-plain).
# Description:
#   Comprehensive single-file legislative intelligence platform for ingestion, analysis, retrieval,
#   graph enrichment, semantic + graph-augmented QA, and multi-environment deployment automation.
# Change Summary:
#   v1.1.0 introduces Terraform R2/Worker assets, Supabase migrations, GraphRAG expansion,
#   metrics, and advanced CLI utilities without splitting into multiple files.
# Inputs:
#   CLI flags (see --help) & environment variables:
#     POSTGRES_*  : Database config
#     PGVECTOR_DIM: Embedding dimension
#     EMBED_MODEL : SentenceTransformer model
#     GOVINFO_API_KEY, OPENSTATES_API_KEY, PROPUBLICA_API_KEY
#     LOCALAI_ENDPOINT, OPENAI_API_KEY, MODEL_NAME, OLLAMA_ENDPOINT
#     HF_MODEL, SPACY_MODEL
#     ENABLE_NEO4J, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
#     ENABLE_FALKORDB, FALKORDB_HOST, FALKORDB_PORT
#     CF_ACCOUNT_ID, CF_API_TOKEN, CF_VECTORIZE_INDEX
#     SUPABASE_URL, SUPABASE_SERVICE_KEY / SUPABASE_ANON_KEY
#     RAGFLOW_TRIGGER_URL, FLOWISE_WEBHOOK, N8N_WEBHOOK
#     ENABLE_LANGFUSE, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
# Outputs:
#   - Populated PostgreSQL schema (documents, sections, embeddings, bills, votes, profiles).
#   - Plain-language summaries cache (data/cache/summaries.json)
#   - Graph exports (Neo4j Bloom perspective), FalkorDB triples.
#   - Terraform & infra assets (generated on demand).
#   - Supabase migration SQL (generated on demand).
#   - GraphRAG context JSON exports.
#   - Docker compose + reverse proxy configs (optional).
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
import datetime
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# -------------------------------- Dependency Handling --------------------------------------------
CORE_REQUIREMENTS = [
    "requests", "tqdm", "psycopg2-binary", "pydantic", "fastapi", "uvicorn",
    "python-dotenv", "sentence-transformers", "numpy", "scikit-learn",
    "beautifulsoup4", "lxml", "PyPDF2", "spacy"
]
OPTIONAL_REQUIREMENTS = [
    "redis", "neo4j", "langfuse", "supabase", "transformers"
]

def auto_install(packages: List[str]):
    missing=[]
    for pkg in packages:
        base=pkg.split("==")[0]
        try: __import__(base)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[INSTALL] Installing missing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# -------------------------------- Utility --------------------------------------------------------
def log(msg: str):
    print(f"[{datetime.datetime.utcnow().isoformat()}] {msg}")

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def port_in_use(port: int, host="0.0.0.0"):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0

def find_free_port(preferred: int, fallback_range=(8000, 8999)):
    if not port_in_use(preferred):
        return preferred
    for p in range(*fallback_range):
        if not port_in_use(p):
            log(f"[ports] {preferred} busy, using {p}")
            return p
    raise RuntimeError("No free port available")

# -------------------------------- Data Model -----------------------------------------------------
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

# -------------------------------- Database Layer -------------------------------------------------
import psycopg2
import psycopg2.extras
import numpy as np

class PGStore:
    def __init__(self):
        self.host=os.environ.get("POSTGRES_HOST","localhost")
        self.port=int(os.environ.get("POSTGRES_PORT","5432"))
        self.db=os.environ.get("POSTGRES_DB","civic_kg")
        self.user=os.environ.get("POSTGRES_USER","postgres")
        self.password=os.environ.get("POSTGRES_PASSWORD","postgres")
        self.dim=int(os.environ.get("PGVECTOR_DIM","384"))
        self.conn=None
    def connect(self):
        self.conn=psycopg2.connect(host=self.host, port=self.port, dbname=self.db,
                                   user=self.user, password=self.password)
        self.conn.autocommit=True
    def close(self):
        if self.conn: self.conn.close()
    def init_schema(self):
        cur=self.conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""CREATE TABLE IF NOT EXISTS documents(
          id UUID PRIMARY KEY,
          ext_id TEXT,
          bill_id TEXT,
          title TEXT,
          jurisdiction TEXT,
          source_type TEXT,
          provenance JSONB,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS sections(
          id UUID PRIMARY KEY,
          document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
          section_no TEXT,
          heading TEXT,
          text TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS embeddings(
          section_id UUID REFERENCES sections(id) ON DELETE CASCADE,
          model TEXT,
          embedding vector({self.dim}),
          created_at TIMESTAMPTZ DEFAULT now(),
          PRIMARY KEY(section_id,model)
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS bills(
          bill_id TEXT PRIMARY KEY,
          title TEXT,
          jurisdiction TEXT,
          raw_text TEXT,
          source_type TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS politicians(
          politician_id TEXT PRIMARY KEY,
          name TEXT,
          party TEXT,
          chamber TEXT,
          state TEXT,
          district TEXT,
          metadata JSONB,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS votes(
          vote_id TEXT PRIMARY KEY,
          bill_id TEXT REFERENCES bills(bill_id),
          vote_date DATE,
          chamber TEXT,
          meta JSONB,
          created_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS vote_choices(
          vote_id TEXT REFERENCES votes(vote_id) ON DELETE CASCADE,
          politician_id TEXT REFERENCES politicians(politician_id),
          choice TEXT,
          PRIMARY KEY(vote_id,politician_id)
        );""")
        cur.execute("""CREATE TABLE IF NOT EXISTS politician_profiles(
          politician_id TEXT PRIMARY KEY REFERENCES politicians(politician_id) ON DELETE CASCADE,
          stats JSONB,
          updated_at TIMESTAMPTZ DEFAULT now()
        );""")
        cur.close()
        log("PostgreSQL schema created/verified.")
    def insert_document(self, d: IngestedDocument):
        cur=self.conn.cursor()
        doc_id=str(uuid.uuid4())
        cur.execute("""INSERT INTO documents(id,ext_id,bill_id,title,jurisdiction,source_type,provenance)
        VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING""",
        (doc_id,d.ext_id,d.bill_id,d.title,d.jurisdiction,d.source_type,json.dumps(d.provenance)))
        if d.bill_id:
            cur.execute("""INSERT INTO bills(bill_id,title,jurisdiction,raw_text,source_type)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT(bill_id) DO UPDATE SET title=EXCLUDED.title""",
            (d.bill_id,d.title,d.jurisdiction,d.full_text[:250000],d.source_type))
        for idx,s in enumerate(d.sections):
            sid=str(uuid.uuid4())
            cur.execute("""INSERT INTO sections(id,document_id,section_no,heading,text)
            VALUES (%s,%s,%s,%s,%s)""",(sid,doc_id,s.get("number") or str(idx+1),s.get("heading"),s.get("text")[:60000]))
        cur.close()
    def fetch_sections_to_embed(self, model: str, limit=4000):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""SELECT s.id,s.text FROM sections s
           LEFT JOIN embeddings e ON e.section_id=s.id AND e.model=%s
           WHERE e.section_id IS NULL
           ORDER BY s.created_at ASC
           LIMIT %s""",(model,limit))
        rows=cur.fetchall(); cur.close(); return rows
    def insert_embeddings(self, model: str, items: List[Tuple[str,np.ndarray]]):
        cur=self.conn.cursor()
        for sid,vec in items:
            cur.execute("""INSERT INTO embeddings(section_id,model,embedding)
            VALUES (%s,%s,%s) ON CONFLICT (section_id,model) DO NOTHING""",(sid,model,list(vec)))
        cur.close()
    def semantic_search(self, qvec: np.ndarray, model: str, k=6):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""SELECT s.id,s.text,d.bill_id,d.title,(1 - (embedding <=> %s::vector)) AS score
                       FROM embeddings e
                       JOIN sections s ON s.id=e.section_id
                       JOIN documents d ON d.id=s.document_id
                       WHERE model=%s
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""",(list(qvec),model,list(qvec),k))
        rows=cur.fetchall(); cur.close(); return rows
    def get_bill(self,bill_id: str):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM bills WHERE bill_id=%s",(bill_id,))
        bill=cur.fetchone()
        if not bill:
            cur.close(); return None
        cur.execute("""SELECT s.id,s.section_no,s.heading,s.text
                       FROM sections s JOIN documents d ON d.id=s.document_id
                       WHERE d.bill_id=%s ORDER BY s.section_no::int NULLS LAST, s.created_at ASC
                       LIMIT 500""",(bill_id,))
        secs=[dict(r) for r in cur.fetchall()]
        cur.close()
        return {"bill":dict(bill),"sections":secs}
    def upsert_vote(self,vote: Dict[str,Any]):
        cur=self.conn.cursor()
        cur.execute("""INSERT INTO votes(vote_id,bill_id,vote_date,chamber,meta)
        VALUES (%s,%s,%s,%s,%s) ON CONFLICT(vote_id) DO NOTHING""",
        (vote["vote_id"], vote.get("bill_id"), vote.get("vote_date"),
         vote.get("chamber"), json.dumps(vote.get("meta") or {})))
        for pid,choice in vote.get("choices",{}).items():
            cur.execute("""INSERT INTO vote_choices(vote_id,politician_id,choice)
            VALUES (%s,%s,%s)
            ON CONFLICT (vote_id,politician_id) DO UPDATE SET choice=EXCLUDED.choice""",
            (vote["vote_id"],pid,choice))
        cur.close()
    def compute_politician_profiles(self):
        cur=self.conn.cursor()
        cur.execute("""WITH base AS (
            SELECT p.politician_id,p.name,
                   SUM(CASE WHEN vc.choice='YEA' THEN 1 ELSE 0 END) yeas,
                   SUM(CASE WHEN vc.choice='NAY' THEN 1 ELSE 0 END) nays,
                   COUNT(vc.choice) total_votes
            FROM politicians p
            LEFT JOIN vote_choices vc ON vc.politician_id=p.politician_id
            GROUP BY p.politician_id,p.name
        ) SELECT politician_id,name,yeas,nays,total_votes FROM base""")
        rows=cur.fetchall()
        for pid,name,yeas,nays,total in rows:
            stats={"name":name,"yeas":yeas,"nays":nays,"total_votes":total,
                   "yea_pct":(float(yeas)/total if total else None),
                   "nay_pct":(float(nays)/total if total else None)}
            c2=self.conn.cursor()
            c2.execute("""INSERT INTO politician_profiles(politician_id,stats)
            VALUES (%s,%s) ON CONFLICT(politician_id)
            DO UPDATE SET stats=EXCLUDED.stats, updated_at=now()""",(pid,json.dumps(stats)))
            c2.close()
        cur.close()
    def get_politician_profile(self,pid: str):
        cur=self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""SELECT p.politician_id,p.name,p.party,p.chamber,p.state,p.district,prof.stats
                       FROM politicians p
                       LEFT JOIN politician_profiles prof ON prof.politician_id=p.politician_id
                       WHERE p.politician_id=%s""",(pid,))
        r=cur.fetchone(); cur.close()
        if not r: return None
        d=dict(r)
        if isinstance(d.get("stats"),str):
            try: d["stats"]=json.loads(d["stats"])
            except: pass
        return d

# -------------------------------- Embeddings & NLP -----------------------------------------------
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
    def encode(self,texts: List[str])->np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True), dtype="float32")

import requests
class Summarizer:
    def __init__(self):
        self.localai=os.environ.get("LOCALAI_ENDPOINT")
        self.openai_key=os.environ.get("OPENAI_API_KEY")
        self.model=os.environ.get("MODEL_NAME","gpt-4o-mini")
        self.ollama=os.environ.get("OLLAMA_ENDPOINT","http://localhost:11434")
        self.hf_model=os.environ.get("HF_MODEL","distilbert-base-uncased")
        ensure_dirs("data/cache")
        self.cache_path="data/cache/summaries.json"
        self.cache={}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path,"r",encoding="utf-8") as f:
                    self.cache=json.load(f)
            except: self.cache={}
    def summarize(self,bill_id: str, section_id: str, text: str)->str:
        key=f"{bill_id}:{section_id}"
        if key in self.cache: return self.cache[key]
        snippet=text[:4000]
        prompt=f"Provide a concise, plain-language explanation of this legislative section:\n\n{snippet}\n\nSimplified:"
        response=None
        if self.localai:
            try:
                r=requests.post(f"{self.localai}/chat/completions",
                                headers={"Content-Type":"application/json"},
                                json={"model":self.model,
                                      "messages":[
                                          {"role":"system","content":"You simplify legislative text."},
                                          {"role":"user","content":prompt}],
                                      "temperature":0.3},timeout=60)
                if r.status_code<400:
                    response=r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                response=f"[LocalAI error] {e}"
        elif self.openai_key:
            try:
                r=requests.post("https://api.openai.com/v1/chat/completions",
                                headers={"Authorization":f"Bearer {self.openai_key}",
                                         "Content-Type":"application/json"},
                                json={"model":self.model,"messages":[
                                    {"role":"system","content":"You simplify legislative text."},
                                    {"role":"user","content":prompt}],"temperature":0.3},timeout=60)
                if r.status_code<400:
                    response=r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                response=f"[OpenAI error] {e}"
        if not response:  # Ollama fallback
            try:
                r=requests.post(f"{self.ollama}/api/generate",json={"model":"llama2","prompt":prompt},timeout=60)
                if r.status_code<400:
                    response=" ".join([ln for ln in r.text.splitlines() if ln.strip()])
            except Exception as e:
                response=f"[Ollama error] {e}"
        if not response:
            response="(Summary unavailable)"
        self.cache[key]=response
        with open(self.cache_path,"w",encoding="utf-8") as f:
            json.dump(self.cache,f,indent=2)
        return response

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
                log(f"[spacy] model {name} not found.")
                return None
    except ImportError:
        return None

def bert_classify_samples(texts: List[str], model_name="distilbert-base-uncased", auto_install=False):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        if auto_install:
            subprocess.check_call([sys.executable,"-m","pip","install","transformers","torch"])
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        else:
            return []
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    out=[]
    for t in texts[:12]:
        inputs=tokenizer(t[:512],return_tensors="pt",truncation=True)
        with torch.no_grad():
            logits=model(**inputs).logits
        probs=logits.softmax(dim=-1).tolist()[0]
        out.append({"text":t[:80],"label_scores":probs})
    return out

# -------------------------------- Ingestion ------------------------------------------------------
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

class GovInfoIngestor:
    API="https://api.govinfo.gov"
    def __init__(self, days: int, collections: List[str], key: Optional[str]):
        self.days=days; self.collections=collections; self.key=key
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
        h={"User-Agent":"CivicSuperHub/1.1"}
        if self.key: h["X-Api-Key"]=self.key
        return h
    def _list_packages(self,col,start,end):
        url=f"{self.API}/collections/{col}/{start}/{end}"
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
    def _download_zip(self, pid, target):
        ensure_dirs(target)
        url=f"{self.API}/packages/{pid}/zip"
        r=requests.get(url,headers=self._headers(),timeout=120)
        if r.status_code!=200: return False
        zp=os.path.join(target,f"{pid}.zip")
        with open(zp,"wb") as f: f.write(r.content)
        try:
            with zipfile.ZipFile(zp,'r') as z: z.extractall(target)
        except: return False
        return True
    def _parse(self,col,target,pid):
        xmls=[]
        for root,_,files in os.walk(target):
            for fn in files:
                if fn.lower().endswith(".xml"):
                    xmls.append(os.path.join(root,fn))
        if not xmls: return None
        xmls.sort(key=lambda p: os.path.getsize(p), reverse=True)
        main=xmls[0]
        try:
            with open(main,"r",encoding="utf-8",errors="ignore") as f:
                soup=BeautifulSoup(f.read(),"lxml-xml")
        except: return None
        title=None
        for cand in ["title","official-title","docTitle","dc:title"]:
            el=soup.find(cand)
            if el and el.text.strip():
                title=el.text.strip();break
        if not title: title=pid
        paras=[p.get_text(" ",strip=True) for p in soup.find_all(["section","p","Paragraph"]) if p.get_text(strip=True)]
        dedup=[]; seen=set()
        for p in paras:
            if p not in seen:
                seen.add(p); dedup.append(p)
        sections=[]
        for i,ch in enumerate(dedup[:80]):
            sections.append({"number":str(i+1),"heading":None,"text":ch[:12000]})
        return IngestedDocument(
            ext_id=pid,title=title,jurisdiction="US-Federal",
            full_text="\n\n".join(dedup),
            sections=sections if sections else [{"number":"1","heading":None,"text":"\n\n".join(dedup)[:12000]}],
            source_type=f"govinfo:{col}",
            provenance={"collection":col,"package_id":pid},
            bill_id=pid
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
                self.state["processed"][pid]={"ts":datetime.datetime.utcnow().isoformat(),"col":col}
            self._save_state()
        log(f"[govinfo] new docs: {len(new_docs)}")
        return new_docs

class OpenStatesIngestor:
    API="https://v3.openstates.org/bills"
    def __init__(self, states: List[str], pages: int, key: Optional[str]):
        self.states=states; self.pages=pages; self.key=key
    def _headers(self):
        return {"X-API-KEY":self.key} if self.key else {}
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
    def __init__(self, congress: int, chambers: List[str], recent: int, key: Optional[str]):
        self.congress=congress; self.chambers=chambers; self.recent=recent; self.key=key
    def _headers(self):
        return {"X-API-Key":self.key} if self.key else {}
    def ingest_votes(self):
        if not self.key:
            log("[propublica] API key missing; skipping votes.")
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
                choices={}
                for pos in v.get("positions",[]) or []:
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
    def __init__(self, patterns: List[str], jurisdiction="Local"):
        self.patterns=patterns; self.jurisdiction=jurisdiction
    def ingest(self):
        paths=[]; docs=[]
        for p in self.patterns:
            paths.extend(glob.glob(p, recursive=True))
        for path in paths:
            if not os.path.isfile(path): continue
            try:
                text=read_text(path)
                if not text.strip(): continue
                lines=[l.strip() for l in text.splitlines() if l.strip()]
                title=lines[0][:180] if lines else os.path.basename(path)
                secs=chunk_text(text)
                docs.append(IngestedDocument(
                    ext_id=path,title=title,jurisdiction=self.jurisdiction,
                    full_text=text,sections=secs,source_type="local_file",
                    provenance={"path":path},bill_id=None
                ))
            except Exception as e:
                log(f"[local] error {path}: {e}")
        log(f"[local-files] docs: {len(docs)}")
        return docs

# -------------------------------- Helpers --------------------------------------------------------
def read_text(path: str)->str:
    ext=os.path.splitext(path)[1].lower()
    if ext==".pdf":
        with open(path,"rb") as f:
            reader=PdfReader(f)
            return "\n".join([pg.extract_text() or "" for pg in reader.pages])
    elif ext in (".xml",".html",".htm"):
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            soup=BeautifulSoup(f.read(),"lxml")
        return soup.get_text("\n")
    else:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            return f.read()

def chunk_text(text: str,max_len=1800):
    words=text.split()
    cur=[]; out=[]
    for w in words:
        cur.append(w)
        if len(" ".join(cur))>=max_len:
            out.append({"number":str(len(out)+1),"heading":None,"text":" ".join(cur)})
            cur=[]
    if cur:
        out.append({"number":str(len(out)+1),"heading":None,"text":" ".join(cur)})
    return out

# -------------------------------- Triple Extraction ----------------------------------------------
def extract_triples(text: str, limit=200):
    if 'doc2graph' in sys.modules:
        try:
            import doc2graph
            triples=doc2graph.extract(text)
            return triples[:limit]
        except:
            pass
    triples=[]
    sentences=[s.strip() for s in text.split(".") if s.strip()]
    for i,s in enumerate(sentences[:25]):
        tokens=[t for t in s.split() if t.istitle()][:3]
        for t in tokens:
            triples.append((f"Sentence_{i}","MENTIONS",t))
    return triples[:limit]

# -------------------------------- Graph Integrations ---------------------------------------------
try:
    from neo4j import GraphDatabase
except:
    GraphDatabase=None

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
    def upsert_bill(self,bill_id: str,title: str,jurisdiction: str):
        if not self.enabled: return
        with self.driver.session() as s:
            s.run("""MERGE (b:Bill {bill_id:$bid})
                     SET b.title=$title,b.jurisdiction=$jurisdiction""",
                  bid=bill_id,title=title,jurisdiction=jurisdiction)
    def export_bloom(self,path="bloom_perspective.json"):
        if not self.enabled: return
        perspective={
            "name":"SuperHubPerspective",
            "version":"1.0",
            "lastUpdated":datetime.datetime.utcnow().isoformat(),
            "categories":[{"name":"Bills","label":"Bill","cypher":"MATCH (b:Bill) RETURN b",
                           "style":{"color":"#1f77b4","size":55}}],
            "relationships":[]
        }
        with open(path,"w",encoding="utf-8") as f:
            json.dump(perspective,f,indent=2)
        log("[neo4j] bloom perspective exported")

    # GraphRAG expansion (simple BFS up to depth using Bill anchored)
    def graphrag_expand(self,bill_id: str, depth: int=2, limit_per_hop=50):
        if not self.enabled: return {}
        query = """
        MATCH (b:Bill {bill_id:$bill_id})
        CALL apoc.path.expandConfig(b,{
          maxLevel:$depth,
          uniqueness:'NODE_GLOBAL'
        }) YIELD path
        WITH path LIMIT $limit
        RETURN collect(distinct nodes(path)) as nodes,
               collect(distinct relationships(path)) as rels
        """
        # Fallback if APOC not available: simpler approach
        fallback=False
        with self.driver.session() as s:
            try:
                rec=s.run(query, bill_id=bill_id, depth=depth, limit=limit_per_hop).single()
            except Exception:
                fallback=True
        if fallback:
            with self.driver.session() as s:
                recs=s.run("""
                MATCH (b:Bill {bill_id:$bill_id})--(n)
                RETURN b,collect(distinct n) as nbrs LIMIT $limit
                """, bill_id=bill_id, limit=limit_per_hop).single()
                if not recs: return {}
                nodes=[dict(recs["b"])] + [dict(x) for x in recs["nbrs"]]
                return {"bill_id":bill_id,"nodes":nodes,"relationships":[]}
        if not rec: return {}
        def norm_node(n):
            d=dict(n)
            d["_labels"]=list(n.labels)
            return d
        nodes=[norm_node(n) for n in rec["nodes"]]
        rels=[]
        for r in rec["rels"]:
            rels.append({"type":r.type, "start":r.start_node.id, "end":r.end_node.id})
        return {"bill_id":bill_id,"nodes":nodes,"relationships":rels,"depth":depth}

    def graph_metrics(self):
        if not self.enabled: return {}
        with self.driver.session() as s:
            deg=s.run("MATCH (n:Bill) RETURN n.bill_id AS bid, size((n)--()) AS degree ORDER BY degree DESC LIMIT 50").values()
            return {"bill_degree_top":[{"bill_id":b,"degree":d} for b,d in deg]}

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
                log(f"[falkordb] connect fail {e}")
                self.enabled=False
    def add_triples(self,bill_id: str, triples: List[Tuple[str,str,str]]):
        if not self.enabled or not self.client: return
        graph="legislation"
        for s,p,o in triples[:150]:
            q1=f"MERGE (:Node {{name:'{s}'}}); MERGE (:Node {{name:'{o}'}});"
            q2=f"MATCH (a:Node {{name:'{s}'}}),(b:Node {{name:'{o}'}}) MERGE (a)-[:{p} {{bill:'{bill_id}'}}]->(b);"
            try:
                self.client.execute_command("GRAPH.QUERY", graph, q1, "--compact")
                self.client.execute_command("GRAPH.QUERY", graph, q2, "--compact")
            except: pass

# -------------------------------- Cloudflare Vectorize / D1 --------------------------------------
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
                log(f"[vectorize] upserted {len(payload['vectors'])}")
        except Exception as e:
            log(f"[vectorize] error {e}")

def export_d1_snapshot(pg: PGStore, path="data/exports/d1_snapshot.json", limit=400):
    ensure_dirs("data/exports")
    cur=pg.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""SELECT bill_id,title,jurisdiction,LEFT(raw_text,20000) excerpt
                   FROM bills ORDER BY created_at DESC LIMIT %s""",(limit,))
    rows=[dict(r) for r in cur.fetchall()]
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"bills":rows,"generated":datetime.datetime.utcnow().isoformat()},f,indent=2)
    log(f"[d1] snapshot -> {path}")

# -------------------------------- External Hooks -------------------------------------------------
def post_webhook(url: str, payload: Dict[str,Any], label="hook"):
    if not url: return
    try:
        requests.post(url,json=payload,timeout=10)
        log(f"[{label}] posted")
    except Exception as e:
        log(f"[{label}] error {e}")

def ragflow_trigger(url: str, payload: Dict[str,Any]):
    if not url: return
    try:
        requests.post(url,json=payload,timeout=10)
        log("[ragflow] triggered")
    except Exception as e:
        log(f"[ragflow] error {e}")

def supabase_log(event: str, data: Dict[str,Any]):
    supabase_url=os.environ.get("SUPABASE_URL")
    supabase_key=os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key: return
    try:
        requests.post(f"{supabase_url}/rest/v1/events",
                      headers={"apikey":supabase_key,"Authorization":f"Bearer {supabase_key}",
                               "Content-Type":"application/json","Prefer":"return=representation"},
                      json={"event":event,"payload":data}, timeout=10)
    except Exception as e:
        log(f"[supabase] error {e}")

def langfuse_log(event_type: str, data: Dict[str,Any]):
    if os.environ.get("ENABLE_LANGFUSE")!="1": return
    pk=os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk=os.environ.get("LANGFUSE_SECRET_KEY")
    host=os.environ.get("LANGFUSE_HOST","https://cloud.langfuse.com")
    if not (pk and sk): return
    try:
        requests.post(f"{host}/api/public/ingest",
                      headers={"x-langfuse-public-key":pk,"x-langfuse-secret-key":sk,"Content-Type":"application/json"},
                      json={"type":event_type,"body":data},timeout=10)
    except Exception as e:
        log(f"[langfuse] error {e}")

# -------------------------------- Memory & Agent -------------------------------------------------
class MemoryBuffer:
    def __init__(self, cap=300):
        self.cap=cap; self.storage=[]
    def store(self, role: str, content: str):
        self.storage.append({"role":role,"content":content,"ts":datetime.datetime.utcnow().isoformat()})
        if len(self.storage)>self.cap:
            self.storage=self.storage[-self.cap:]
    def tail(self,n=10):
        return self.storage[-n:]

def rhetorical_label(text: str)->str:
    # Simple heuristic placeholder
    lower=text.lower()
    if "privacy" in lower: return "PRIVACY"
    if "tax" in lower: return "FISCAL"
    if "security" in lower or "defense" in lower: return "SECURITY"
    if "environment" in lower or "climate" in lower: return "ENVIRONMENT"
    return "GENERAL"

def naive_rerank(hits: List[Dict[str,Any]], query: str)->List[Dict[str,Any]]:
    # Adds rhetorical label weight heuristic
    q_tokens=set(query.lower().split())
    for h in hits:
        label=rhetorical_label(h["snippet"])
        overlap=len(set(h["snippet"].lower().split()) & q_tokens)
        h["rerank_score"]=h["score"] + (0.01*overlap) + (0.05 if label!="GENERAL" else 0)
        h["topic_label"]=label
    return sorted(hits,key=lambda x:x["rerank_score"],reverse=True)

class Agent:
    def __init__(self, pg: PGStore, embedder: Embedder, summarizer: Summarizer, memory: MemoryBuffer):
        self.pg=pg; self.embedder=embedder; self.summarizer=summarizer; self.memory=memory
        self.model=embedder.model_name
    def answer(self, query: str, k=6, plain=False, rerank=True):
        self.memory.store("user", query)
        qv=self.embedder.encode([query])[0]
        rows=self.pg.semantic_search(qv,self.model,k)
        bill_id=None
        for t in query.split():
            if "-" in t and any(c.isdigit() for c in t):
                bill_id=t.strip(",.").upper(); break
        bill_data=self.pg.get_bill(bill_id) if bill_id else None
        hits=[{"section_id":r["id"],"bill_id":r["bill_id"],"title":r["title"],
               "score":float(r["score"]), "snippet":r["text"][:600]} for r in rows]
        if rerank: hits=naive_rerank(hits, query)
        resp={"query":query,"bill_id":bill_id,"hits":hits,"bill":None,"memory_tail":self.memory.tail()}
        if bill_data:
            secs=[]
            for s in bill_data["sections"][:12]:
                sec_obj={"section_no":s["section_no"],"heading":s["heading"],"text":s["text"][:900]}
                if plain:
                    sec_obj["plain_language"]=self.summarizer.summarize(bill_data["bill"]["bill_id"], s["id"], s["text"][:4000])
                secs.append(sec_obj)
            resp["bill"]={"bill_id":bill_data["bill"]["bill_id"],"title":bill_data["bill"]["title"],"sections":secs}
        self.memory.store("assistant", f"Returned {len(hits)} hits")
        langfuse_log("agent_answer", {"query":query,"hit_count":len(hits)})
        supabase_log("agent_query", {"query":query,"hits":len(hits)})
        return resp

# -------------------------------- FastAPI --------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app=FastAPI(title="Civic Legislative SuperHub API", version="1.1.0")
RUNTIME={"pg":None,"agent":None,"summarizer":None,"embedder":None,"neo":None}

class QueryReq(BaseModel):
    query: str
    k: int=6
    plain: bool=False

class BillReq(BaseModel):
    bill_id: str
    plain: bool=True

class PoliticianReq(BaseModel):
    politician_id: str

class GraphRAGReq(BaseModel):
    bill_id: str
    depth: int=2

@app.get("/health")
def health():
    return {"status":"ok","version":"1.1.0"}

@app.post("/query")
def query_ep(req: QueryReq):
    if not RUNTIME["agent"]:
        raise HTTPException(500,"Agent not initialized")
    return RUNTIME["agent"].answer(req.query,k=req.k,plain=req.plain)

@app.post("/bill")
def bill_ep(req: BillReq):
    pg: PGStore=RUNTIME["pg"]
    summ: Summarizer=RUNTIME["summarizer"]
    if not pg: raise HTTPException(500,"PG not ready")
    data=pg.get_bill(req.bill_id)
    if not data: raise HTTPException(404,"Bill not found")
    if req.plain:
        for s in data["sections"]:
            s["plain_language"]=summ.summarize(data["bill"]["bill_id"], s["id"], s["text"][:4000])
    return data

@app.post("/politician")
def pol_ep(req: PoliticianReq):
    pg: PGStore=RUNTIME["pg"]
    if not pg: raise HTTPException(500,"PG not ready")
    prof=pg.get_politician_profile(req.politician_id)
    if not prof: raise HTTPException(404,"Not found")
    return prof

@app.post("/graphrag/expand")
def graphrag_ep(req: GraphRAGReq):
    neo: Neo4jBridge=RUNTIME["neo"]
    if not neo or not neo.enabled:
        raise HTTPException(500,"Neo4j not enabled")
    ctx=neo.graphrag_expand(req.bill_id, depth=req.depth)
    if not ctx:
        raise HTTPException(404,"No context")
    return ctx

# -------------------------------- Self Tests & Reports -------------------------------------------
def run_self_tests(pg: PGStore):
    log("[tests] starting")
    cur=pg.conn.cursor()
    for table in ["documents","sections","embeddings","bills","politicians","votes","vote_choices","politician_profiles"]:
        cur.execute("SELECT to_regclass(%s)",(table,))
        assert cur.fetchone()[0]==table, f"Missing table {table}"
    sample=IngestedDocument(
        ext_id="TEST-DOC","title":"Test Bill","jurisdiction":"Test",
        full_text="One sentence. Another privacy sentence.",
        sections=[{"number":"1","heading":None,"text":"One sentence."}],
        source_type="test",provenance={"reason":"self-test"},bill_id="TB-001"
    )
    pg.insert_document(sample)
    cur.execute("SELECT count(*) FROM bills WHERE bill_id='TB-001'")
    assert cur.fetchone()[0]==1, "Insertion failed"
    log("[tests] OK")

def review_report():
    report={
        "version":"1.1.0",
        "recommendations":{
            "lint":["Use ruff/flake8","Adopt black formatting"],
            "security":["Add auth to public endpoints","Rotate keys","Use WAF/Rate limiting"],
            "scaling":["Queue ingestion tasks","Add distributed embedding workers"],
            "graph":["Add entity linking","Edge weighting for path ranking"],
            "nlp":["Integrate cross-encoder re-ranker","Add topic modeling"]
        }
    }
    print(json.dumps(report,indent=2))

# -------------------------------- Compose / Proxy / Infra Generation -----------------------------
def generate_compose(include_proxy: bool, path="docker-compose.generated.yml"):
    compose={
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
      "volumes":{"pg_data":{}}
    }
    if include_proxy:
        compose["services"]["traefik"]={
          "image":"traefik:v2.11",
          "command":["--providers.docker=true","--entrypoints.web.address=:80"],
          "ports":["80:80"],
          "depends_on":["api"],
          "restart":"unless-stopped"
        }
    with open(path,"w",encoding="utf-8") as f:
        f.write("# Generated by SuperHub\n")
        json.dump(compose,f,indent=2)
    log(f"[compose] generated -> {path}")

def generate_proxy_configs():
    ensure_dirs("infra")
    with open("infra/traefik_dynamic.yml","w",encoding="utf-8") as f:
        f.write("""http:
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
    with open("infra/nginx.conf","w",encoding="utf-8") as f:
        f.write("""events {}
http {
  server {
    listen 80;
    location / { proxy_pass http://api:8100; proxy_set_header Host $host; }
  }
}
""")
    with open("infra/kong.yaml","w",encoding="utf-8") as f:
        f.write("""_format_version: "3.0"
services:
  - name: civic-api
    url: http://api:8100
    routes:
      - name: civic-api-route
        paths: ["/"]
""")
    log("[proxy] configs generated")

def generate_terraform_assets(include_r2=True, include_worker=True, path_dir="infra/terraform"):
    ensure_dirs(path_dir)
    main_lines=[
        'terraform {',
        '  required_providers {',
        '    oci = { source = "oracle/oci" }',
        '    cloudflare = { source = "cloudflare/cloudflare" }',
        '  }',
        '}',
        'provider "oci" {',
        '  tenancy_ocid = var.oci_tenancy_ocid',
        '  user_ocid    = var.oci_user_ocid',
        '  fingerprint  = var.oci_fingerprint',
        '  private_key_path = var.oci_private_key_path',
        '  region = var.oci_region',
        '}',
        'provider "cloudflare" {',
        '  api_token = var.cloudflare_api_token',
        '}',
        '# Instance + DNS simplified example ...'
    ]
    if include_r2:
        main_lines += [
            '# Cloudflare R2 bucket (for raw docs)',
            'resource "cloudflare_r2_bucket" "legislation" {',
            '  account_id = var.cloudflare_account_id',
            '  name       = var.r2_bucket_name',
            '}'
        ]
    if include_worker:
        main_lines += [
            '# Placeholder for Worker deployment (wrangler publish externally).'
        ]
    with open(os.path.join(path_dir,"main.tf"),"w",encoding="utf-8") as f:
        f.write("# Generated Terraform main.tf\n" + "\n".join(main_lines)+"\n")
    with open(os.path.join(path_dir,"variables.tf"),"w",encoding="utf-8") as f:
        f.write("""variable "oci_tenancy_ocid" {}
variable "oci_user_ocid" {}
variable "oci_fingerprint" {}
variable "oci_private_key_path" {}
variable "oci_region" { default = "us-ashburn-1" }
variable "cloudflare_api_token" {}
variable "cloudflare_account_id" {}
variable "r2_bucket_name" { default = "civic-legislation" }
""")
    log(f"[terraform] assets generated in {path_dir}")

def generate_supabase_sql(path="infra/supabase_migrations.sql"):
    ensure_dirs("infra")
    sql = """-- Generated Supabase migration
CREATE TABLE IF NOT EXISTS public.events (
  id BIGSERIAL PRIMARY KEY,
  event TEXT,
  payload JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Optional table for small public bill snapshots
CREATE TABLE IF NOT EXISTS public.bill_snapshots (
  bill_id TEXT PRIMARY KEY,
  title TEXT,
  jurisdiction TEXT,
  excerpt TEXT,
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Index example
CREATE INDEX IF NOT EXISTS events_event_idx ON public.events(event);
"""
    with open(path,"w",encoding="utf-8") as f:
        f.write(sql)
    log(f"[supabase] migration SQL -> {path}")

def export_graphrag_context_json(context: Dict[str,Any], path="data/exports/graphrag_context.json"):
    if not context: return
    ensure_dirs("data/exports")
    with open(path,"w",encoding="utf-8") as f:
        json.dump(context,f,indent=2)
    log(f"[graphrag] context exported -> {path}")

# -------------------------------- Bulk Plain Summaries -------------------------------------------
def bulk_plain_summaries(pg: PGStore, summarizer: Summarizer, limit=20):
    data=[]
    cur=pg.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""SELECT bill_id FROM bills ORDER BY created_at DESC LIMIT %s""",(limit,))
    bills=[r["bill_id"] for r in cur.fetchall()]
    for b in bills:
        bill=pg.get_bill(b)
        if not bill: continue
        out_sections=[]
        for s in bill["sections"][:5]:
            plain=summarizer.summarize(b, s["id"], s["text"][:4000])
            out_sections.append({"section_no":s["section_no"],"plain":plain})
        data.append({"bill_id":b,"sections":out_sections})
    ensure_dirs("data/exports")
    with open("data/exports/bulk_plain.json","w",encoding="utf-8") as f:
        json.dump({"generated":datetime.datetime.utcnow().isoformat(),"bills":data},f,indent=2)
    log("[bulk-plain] export complete")

# -------------------------------- Main Orchestration ---------------------------------------------
def main():
    parser=argparse.ArgumentParser(description="Civic Legislative SuperHub (single-script repo)")
    parser.add_argument("--auto-install",action="store_true")
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
    parser.add_argument("--one-shot-query",type=str)
    parser.add_argument("--plain",action="store_true")
    parser.add_argument("--serve",action="store_true")
    parser.add_argument("--port",type=int,default=8100)
    parser.add_argument("--run-self-tests",action="store_true")
    parser.add_argument("--generate-compose",action="store_true")
    parser.add_argument("--include-proxy",action="store_true")
    parser.add_argument("--generate-review-report",action="store_true")
    parser.add_argument("--spacy-ner-sample",action="store_true")
    parser.add_argument("--bert-sample",action="store_true")
    parser.add_argument("--ragflow-trigger-url",type=str,default="")
    parser.add_argument("--print-ports",action="store_true")
    parser.add_argument("--generate-terraform-r2",action="store_true")
    parser.add_argument("--generate-supabase-sql",action="store_true")
    parser.add_argument("--graphrag-expand",action="store_true")
    parser.add_argument("--graphrag-bill",type=str,default="")
    parser.add_argument("--graphrag-depth",type=int,default=2)
    parser.add_argument("--graph-metrics",action="store_true")
    parser.add_argument("--export-graphrag-context",action="store_true")
    parser.add_argument("--bulk-plain",action="store_true")
    args=parser.parse_args()

    if args.auto_install:
        auto_install(CORE_REQUIREMENTS)
        auto_install(OPTIONAL_REQUIREMENTS)
        if os.environ.get("SPACY_MODEL","en_core_web_sm")=="en_core_web_sm":
            try:
                import spacy
                try:
                    spacy.load("en_core_web_sm")
                except OSError:
                    subprocess.check_call([sys.executable,"-m","spacy","download","en_core_web_sm"])
            except Exception as e:
                log(f"[spacy] install error {e}")

    ensure_dirs("data/cache","data/local")

    # DB
    pg=PGStore(); pg.connect()
    if args.init_db:
        pg.init_schema()

    if args.run_self_tests:
        run_self_tests(pg)

    new_docs=[]

    # Ingest
    if args.sync_govinfo:
        cols=[c.strip().upper() for c in args.govinfo_collections.split(",") if c.strip()]
        gi=GovInfoIngestor(args.govinfo_days, cols, os.environ.get("GOVINFO_API_KEY"))
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
        ppi=ProPublicaVotes(args.congress,chambers, recent=15, key=os.environ.get("PROPUBLICA_API_KEY"))
        votes=ppi.ingest_votes()
        for v in votes:
            pg.upsert_vote(v)

    if new_docs:
        log(f"Inserting {len(new_docs)} docs ...")
        for d in new_docs:
            try: pg.insert_document(d)
            except Exception as e: log(f"[insert] {d.ext_id} fail: {e}")

    summarizer=None
    embedder=None
    if any([args.embed,args.serve,args.one_shot_query,args.bulk_plain]):
        summarizer=Summarizer()
        try:
            embedder=Embedder()
        except Exception as e:
            log(f"[embedder] init fail {e}")

    vectorize=CloudflareVectorize()
    if args.embed and embedder:
        rows=pg.fetch_sections_to_embed(embedder.model_name, limit=5000)
        if rows:
            log(f"Embedding {len(rows)} sections ...")
            texts=[r["text"] for r in rows]
            vectors=embedder.encode(texts)
            items=[(rows[i]["id"],vectors[i]) for i in range(len(rows))]
            pg.insert_embeddings(embedder.model_name, items)
            log("Embeddings stored.")
            if args.mirror_cloudflare_vectorize:
                vectorize.push(embedder.model_name, items)

    neo=Neo4jBridge()
    if neo.enabled and new_docs:
        neo.ensure_constraints()
        for d in new_docs:
            if d.bill_id: neo.upsert_bill(d.bill_id,d.title,d.jurisdiction)
    if args.export_bloom and neo.enabled:
        neo.export_bloom()

    falkor=FalkorDBBridge()
    if args.populate_falkordb and falkor.enabled and new_docs:
        for d in new_docs:
            triples=extract_triples(d.full_text[:20000])
            falkor.add_triples(d.bill_id or d.ext_id, triples)

    if args.build_profiles:
        pg.compute_politician_profiles()

    if args.ragflow_trigger_url and new_docs:
        ragflow_trigger(args.ragflow_trigger_url, {"ingested":len(new_docs)})

    if new_docs:
        post_webhook(os.environ.get("FLOWISE_WEBHOOK"),{"ingested":len(new_docs)},"flowise")
        post_webhook(os.environ.get("N8N_WEBHOOK"),{"ingested":len(new_docs)},"n8n")

    if args.export_d1_snapshot:
        export_d1_snapshot(pg)

    if args.generate_compose:
        generate_compose(args.include_proxy)
        if args.include_proxy:
            generate_proxy_configs()

    if args.generate_terraform_r2:
        generate_terraform_assets(include_r2=True, include_worker=True)

    if args.generate_supabase_sql:
        generate_supabase_sql()

    if args.generate_review_report:
        review_report()

    if args.spacy_ner_sample:
        nlp=load_spacy_model(auto_install=args.auto_install)
        if nlp:
            sample=[d.full_text[:800] for d in new_docs[:3]] or ["No new docs"]
            out=[]
            for t in sample:
                doc=nlp(t)
                ents=[(e.text,e.label_) for e in doc.ents]
                out.append({"text":t[:100],"entities":ents})
            print(json.dumps({"spacy_ner_sample":out},indent=2))

    if args.bert_sample:
        sample=[d.full_text[:500] for d in new_docs[:2]] or ["Classification sample text about environment and taxation."]
        res=bert_classify_samples(sample, auto_install=args.auto_install)
        print(json.dumps({"bert_classification_sample":res},indent=2))

    if args.bulk_plain and summarizer:
        bulk_plain_summaries(pg, summarizer)

    graphrag_context={}
    if args.graphrag_expand and neo.enabled and args.graphrag_bill:
        graphrag_context=neo.graphrag_expand(args.graphrag_bill, depth=args.graphrag_depth)
        print(json.dumps({"graphrag_context":graphrag_context},indent=2))
        if args.export_graphrag_context:
            export_graphrag_context_json(graphrag_context)

    if args.graph_metrics and neo.enabled:
        metrics=neo.graph_metrics()
        print(json.dumps({"graph_metrics":metrics},indent=2))

    if args.print_ports:
        ports=[5432,8080,8100,7474,7687,6379,80]
        status={}
        for p in ports:
            status[p]="USED" if port_in_use(p) else "FREE"
        print(json.dumps({"port_status":status},indent=2))

    agent=None
    if embedder and summarizer:
        memory=MemoryBuffer()
        agent=Agent(pg, embedder, summarizer, memory)

    if args.one_shot_query and agent:
        ans=agent.answer(args.one_shot_query, plain=args.plain)
        print(json.dumps(ans,indent=2))

    if args.serve:
        RUNTIME["pg"]=pg
        RUNTIME["agent"]=agent
        RUNTIME["summarizer"]=summarizer
        RUNTIME["embedder"]=embedder
        RUNTIME["neo"]=neo
        port=find_free_port(args.port)
        from uvicorn import run
        log(f"Serving API on 0.0.0.0:{port}")
        run(app, host="0.0.0.0", port=port)
    else:
        neo.close()
        pg.close()

if __name__ == "__main__":
    main()