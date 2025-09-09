#!/usr/bin/env python3
# =================================================================================================
# Name: Civic Legislative Knowledge Graph & RAG Pipeline
# Date: 2025-09-09
# Script Name: legal_civic_graph.py
# Version: 0.1.0
# Log Summary:
#   - Initial scaffold for multi-source legal ingestion, RAG, Neo4j knowledge graph, agent interface,
#     vote tracking, and plain-language translation of legislation with a single-file architecture.
# Description:
#   This script provides an end-to-end framework to:
#     1. Fetch or sync external repositories containing legal texts or support utilities.
#     2. Ingest legal documents (PDF, HTML, plain text) into a normalized internal model.
#     3. Extract structure (Bills, Sections, Clauses) and named entities (Politicians, Committees).
#     4. Build embeddings and a vector store for Retrieval Augmented Generation (RAG).
#     5. Construct and maintain a Neo4j knowledge graph linking legislation to sponsors and votes.
#     6. Track politician voting behavior; associate votes with bills and compute accountability metrics.
#     7. Provide an agentic reasoning layer (planning + retrieval + graph traversal).
#     8. Offer plain-language translations of complex legal sections using a local or remote LLM.
#     9. Expose a small FastAPI service for programmatic access (bill summaries, entity queries, QA).
#     10. Provide hooks/placeholders for Flowise, n8n automation, LocalAI, and GraphRAG-like workflows.
# Change Summary:
#   - 0.1.0: Initial creation with modular classes, runtime dependency installer, sample data bootstrap,
#            graph schema creation, RAG indexing, agent loop, and REST interface.
# Inputs:
#   - Environment Variables:
#       NEO4J_URI            (e.g. bolt://localhost:7687)
#       NEO4J_USER           (default: neo4j)
#       NEO4J_PASSWORD       (required for graph operations)
#       LOCALAI_ENDPOINT     (e.g. http://localhost:8080/v1) for OpenAI-compatible LLM
#       OPENAI_API_KEY       (optional if using remote OpenAI)
#       MODEL_NAME           (model identifier; default: gpt-4o-mini or local model)
#   - Command Line Arguments (see argparse section):
#       --init-all, --fetch, --ingest, --build-graph, --build-rag, --update-votes, --agent, --serve etc.
#   - Embedded CONFIG (JSON) controlling repositories, parsing rules, etc.
# Outputs:
#   - Local data directory: ./data/{raw,processed,embeddings}
#   - Neo4j nodes & relationships
#   - Vector store (FAISS or Chroma) persisted under ./data/vector_store
#   - Plain-language summaries cached in ./data/cache/summaries.json
#   - API service (FastAPI) when --serve is used
#   - Console logs and optional structured JSON logs
# =================================================================================================

import os
import sys
import json
import time
import math
import glob
import uuid
import queue
import shutil
import signal
import random
import argparse
import textwrap
import datetime
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Iterable

# -------------------------------------------------------------------------------------------------
# Runtime dependency management (lightweight). You can disable by setting SKIP_AUTO_INSTALL=1.
# -------------------------------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "requests", "tqdm", "pydantic", "fastapi", "uvicorn",
    "python-dotenv", "neo4j", "beautifulsoup4", "lxml",
    "PyPDF2", "sentence-transformers", "faiss-cpu", "transformers", "numpy", "scikit-learn"
]

if os.environ.get("SKIP_AUTO_INSTALL") != "1":
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.split("==")[0])
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SETUP] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

# Now import after potential install
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------------------------------------------------------
# Embedded configuration (editable)
# -------------------------------------------------------------------------------------------------
CONFIG = {
    "external_repos": [
        # Placeholder examples (you can add actual legal corpora repos)
        {"name": "us-congress-bills-sample", "git": "https://github.com/unitedstates/congress.git", "path": "data/external/congress"},
        {"name": "eu-parliament-proposals", "git": "https://github.com/euparl/sample.git", "path": "data/external/eu"},
        # Add more as needed
    ],
    "document_globs": [
        "data/external/congress/**/*.xml",
        "data/external/congress/**/*.txt",
        "data/external/eu/**/*.html",
        "data/sample/**/*.txt",
        "data/sample/**/*.pdf"
    ],
    "vector_store_path": "data/vector_store",
    "model_name_default": "all-MiniLM-L6-v2",
    "llm_plain_model_fallback": "gpt-4o-mini",
    "graph": {
        "constraints": [
            "CREATE CONSTRAINT bill_id IF NOT EXISTS FOR (b:Bill) REQUIRE b.bill_id IS UNIQUE",
            "CREATE CONSTRAINT politician_id IF NOT EXISTS FOR (p:Politician) REQUIRE p.politician_id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE"
        ]
    },
    "sample_bills": [
        {
            "bill_id": "HR-1234",
            "title": "Data Privacy Enhancement Act",
            "jurisdiction": "US",
            "text": "A bill to enhance consumer data privacy protections and impose requirements on data brokers...",
            "sections": [
                {"number": "1", "heading": "Short Title", "text": "This Act may be cited as the Data Privacy Enhancement Act."},
                {"number": "2", "heading": "Definitions", "text": "Definitions for personal data, broker, consent, and processing."},
                {"number": "3", "heading": "Consumer Rights", "text": "Establishes rights to access, deletion, portability, and opt-out."}
            ],
            "sponsors": ["P-001", "P-002"]
        }
    ],
    "sample_politicians": [
        {"politician_id": "P-001", "name": "Alex Johnson", "party": "Independent", "region": "State A"},
        {"politician_id": "P-002", "name": "Maria Lopez", "party": "Reform", "region": "State B"}
    ],
    "sample_votes": [
        {
            "vote_id": "V-0001",
            "bill_id": "HR-1234",
            "date": "2025-09-01",
            "results": {
                "P-001": "YEA",
                "P-002": "YEA"
            }
        }
    ]
}

# -------------------------------------------------------------------------------------------------
# Utility & Logging
# -------------------------------------------------------------------------------------------------
def log(msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    print(f"[{ts}] {msg}")

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Repository Fetcher
# -------------------------------------------------------------------------------------------------
class RepoFetcher:
    def __init__(self, repos: List[Dict[str, str]]):
        self.repos = repos

    def sync_all(self):
        for repo in self.repos:
            path = repo["path"]
            git_url = repo["git"]
            if os.path.isdir(os.path.join(path, ".git")):
                log(f"Updating repo {repo['name']} at {path}")
                subprocess.call(["git", "-C", path, "pull", "--ff-only"])
            else:
                log(f"Cloning repo {repo['name']} into {path}")
                ensure_dirs(os.path.dirname(path))
                subprocess.call(["git", "clone", "--depth", "1", git_url, path])

# -------------------------------------------------------------------------------------------------
# Document Model
# -------------------------------------------------------------------------------------------------
@dataclass
class LegalDocument:
    doc_id: str
    source_path: str
    jurisdiction: Optional[str]
    bill_id: Optional[str]
    title: Optional[str]
    raw_text: str
    sections: List[Dict[str, Any]] = field(default_factory=list)

# -------------------------------------------------------------------------------------------------
# Ingestor / Parser
# -------------------------------------------------------------------------------------------------
class DocumentIngestor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def collect_files(self) -> List[str]:
        files = []
        for pattern in self.config["document_globs"]:
            files.extend(glob.glob(pattern, recursive=True))
        files = [f for f in files if os.path.isfile(f)]
        return sorted(list(set(files)))

    def parse_file(self, path: str) -> Optional[LegalDocument]:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                with open(path, "rb") as f:
                    reader = PdfReader(f)
                    pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
            elif ext in (".html", ".htm", ".xml"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "lxml")
                text = soup.get_text("\n")
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if not text.strip():
                return None
            # Heuristic extraction (extend with real bill/section parsing rules)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            title = lines[0][:200] if lines else "Untitled"
            doc = LegalDocument(
                doc_id=str(uuid.uuid4()),
                source_path=path,
                jurisdiction=None,
                bill_id=None,
                title=title,
                raw_text=text,
                sections=self._chunk_sections(text)
            )
            return doc
        except Exception as e:
            log(f"Failed to parse {path}: {e}")
            return None

    def _chunk_sections(self, text: str, max_len: int = 1200) -> List[Dict[str, Any]]:
        words = text.split()
        sections = []
        cur = []
        for w in words:
            cur.append(w)
            if len(" ".join(cur)) >= max_len:
                sections.append({"text": " ".join(cur)})
                cur = []
        if cur:
            sections.append({"text": " ".join(cur)})
        return sections

    def ingest(self) -> List[LegalDocument]:
        files = self.collect_files()
        docs = []
        for fpath in tqdm(files, desc="Ingesting files"):
            doc = self.parse_file(fpath)
            if doc:
                docs.append(doc)
        log(f"Ingested {len(docs)} documents.")
        return docs

# -------------------------------------------------------------------------------------------------
# RAG Vector Store
# -------------------------------------------------------------------------------------------------
class VectorStore:
    def __init__(self, path: str, model_name: str):
        self.path = path
        ensure_dirs(path)
        self.model_name = model_name
        log(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        if faiss is None:
            log("FAISS not available; storing embeddings in memory list (fallback).")
            self.index = embeddings
        else:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        self.meta = chunks
        self._persist()

    def _persist(self):
        meta_path = os.path.join(self.path, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        if faiss and isinstance(self.index, faiss.Index):
            faiss.write_index(self.index, os.path.join(self.path, "index.faiss"))
        else:
            np.save(os.path.join(self.path, "index.npy"), self.index)
        log(f"Vector store persisted at {self.path}")

    def load(self):
        meta_path = os.path.join(self.path, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Vector store not built yet.")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if faiss and os.path.exists(os.path.join(self.path, "index.faiss")):
            self.index = faiss.read_index(os.path.join(self.path, "index.faiss"))
        elif os.path.exists(os.path.join(self.path, "index.npy")):
            self.index = np.load(os.path.join(self.path, "index.npy"))
        else:
            raise FileNotFoundError("No index file found for vector store.")
        log("Vector store loaded.")

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query_text]).astype("float32")
        if faiss and isinstance(self.index, faiss.Index):
            faiss.normalize_L2(q_emb)
            distances, idxs = self.index.search(q_emb, k)
            results = []
            for rank, (i, dist) in enumerate(zip(idxs[0], distances[0])):
                meta = self.meta[i].copy()
                meta["score"] = float(dist)
                meta["rank"] = rank
                results.append(meta)
            return results
        else:
            # Fallback manual similarity
            embeddings = self.index  # np.ndarray
            q_norm = q_emb / np.linalg.norm(q_emb)
            emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            sims = emb_norm @ q_norm.T
            sims = sims.flatten()
            top_idx = sims.argsort()[-k:][::-1]
            results = []
            for rank, i in enumerate(top_idx):
                meta = self.meta[i].copy()
                meta["score"] = float(sims[i])
                meta["rank"] = rank
                results.append(meta)
            return results

# -------------------------------------------------------------------------------------------------
# Neo4j Graph Manager
# -------------------------------------------------------------------------------------------------
class GraphManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        with self.driver.session() as session:
            for c in CONFIG["graph"]["constraints"]:
                session.run(c)
        log("Graph constraints ensured.")

    def upsert_bill(self, bill: Dict[str, Any]):
        q = """
        MERGE (b:Bill {bill_id:$bill_id})
        SET b.title=$title, b.jurisdiction=$jurisdiction, b.text=$text
        """
        with self.driver.session() as s:
            s.run(q, **{
                "bill_id": bill["bill_id"],
                "title": bill.get("title"),
                "jurisdiction": bill.get("jurisdiction"),
                "text": bill.get("text")
            })
        # Sections
        if "sections" in bill:
            for sec in bill["sections"]:
                self.upsert_section(bill["bill_id"], sec)

    def upsert_section(self, bill_id: str, section: Dict[str, Any]):
        q = """
        MERGE (s:Section {section_id:$sid})
        SET s.number=$number, s.heading=$heading, s.text=$text
        WITH s
        MATCH (b:Bill {bill_id:$bill_id})
        MERGE (b)-[:HAS_SECTION]->(s)
        """
        sid = f"{bill_id}-SEC-{section.get('number','X')}"
        with self.driver.session() as s:
            s.run(q, sid=sid,
                  number=section.get("number"),
                  heading=section.get("heading"),
                  text=section.get("text"),
                  bill_id=bill_id)

    def upsert_politician(self, pol: Dict[str, Any]):
        q = """
        MERGE (p:Politician {politician_id:$pid})
        SET p.name=$name, p.party=$party, p.region=$region
        """
        with self.driver.session() as s:
            s.run(q, pid=pol["politician_id"], name=pol.get("name"),
                  party=pol.get("party"), region=pol.get("region"))

    def create_sponsorship(self, bill_id: str, politician_id: str):
        q = """
        MATCH (b:Bill {bill_id:$bill_id}), (p:Politician {politician_id:$pid})
        MERGE (p)-[:SPONSORS]->(b)
        """
        with self.driver.session() as s:
            s.run(q, bill_id=bill_id, pid=politician_id)

    def register_vote(self, vote: Dict[str, Any]):
        # vote: {vote_id, bill_id, date, results: {politician_id: "YEA"/"NAY"/"ABS"}}
        with self.driver.session() as session:
            for pid, v in vote["results"].items():
                q = """
                MATCH (b:Bill {bill_id:$bill_id})
                MATCH (p:Politician {politician_id:$pid})
                MERGE (p)-[r:VOTED_ON {vote_id:$vote_id}]->(b)
                SET r.choice=$choice, r.date=$date
                """
                session.run(q, bill_id=vote["bill_id"], pid=pid,
                            vote_id=vote["vote_id"], choice=v, date=vote.get("date"))

    def bill_summary(self, bill_id: str) -> Optional[Dict[str, Any]]:
        q = """
        MATCH (b:Bill {bill_id:$bill_id})
        OPTIONAL MATCH (b)-[:HAS_SECTION]->(s:Section)
        OPTIONAL MATCH (p:Politician)-[sp:SPONSORS]->(b)
        OPTIONAL MATCH (p2:Politician)-[v:VOTED_ON]->(b)
        RETURN b,
               collect(distinct s) as sections,
               collect(distinct p) as sponsors,
               collect(distinct {politician:p2, vote:v}) as votes
        """
        with self.driver.session() as s:
            rec = s.run(q, bill_id=bill_id).single()
            if not rec:
                return None
            b = rec["b"]
            sections = [dict(r) for r in rec["sections"] if r]
            sponsors = [dict(r) for r in rec["sponsors"] if r]
            votes = []
            for v in rec["votes"]:
                pol = dict(v["politician"]) if v["politician"] else None
                vote_rel = dict(v["vote"]) if v["vote"] else None
                if pol and vote_rel:
                    votes.append({"politician": pol, "vote": vote_rel})
            return {"bill": dict(b), "sections": sections, "sponsors": sponsors, "votes": votes}

    def search_politician_votes(self, politician_name: str) -> List[Dict[str, Any]]:
        q = """
        MATCH (p:Politician)-[v:VOTED_ON]->(b:Bill)
        WHERE toLower(p.name) CONTAINS toLower($name)
        RETURN p, b, v
        LIMIT 100
        """
        with self.driver.session() as s:
            recs = s.run(q, name=politician_name)
            out = []
            for r in recs:
                out.append({
                    "politician": dict(r["p"]),
                    "bill": dict(r["b"]),
                    "vote": dict(r["v"])
                })
            return out

# -------------------------------------------------------------------------------------------------
# Plain Language Summarizer (LLM Interface)
# -------------------------------------------------------------------------------------------------
class PlainLanguageExplainer:
    def __init__(self):
        self.endpoint = os.environ.get("LOCALAI_ENDPOINT") or "https://api.openai.com/v1"
        self.api_key = os.environ.get("OPENAI_API_KEY", "DUMMY_KEY")
        self.model = os.environ.get("MODEL_NAME", CONFIG["llm_plain_model_fallback"])
        ensure_dirs("data/cache")
        self.cache_file = "data/cache/summaries.json"
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def _persist(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def summarize_section(self, bill_id: str, section_id: str, text: str) -> str:
        cache_key = f"{bill_id}:{section_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        prompt = f"Rewrite the following legislative text in clear, plain language for an average citizen:\n\n{text}\n\nPlain language:"
        # Minimal generic OpenAI-compatible request
        try:
            resp = requests.post(
                f"{self.endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a legal plain-language assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                },
                timeout=60
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:200]}")
            j = resp.json()
            content = j["choices"][0]["message"]["content"].strip()
        except Exception as e:
            content = f"[LLM_ERROR_FALLBACK] Could not summarize: {e}"
        self.cache[cache_key] = content
        self._persist()
        return content

# -------------------------------------------------------------------------------------------------
# Agent (Basic Planner + Retrieval + Graph Lookups)
# -------------------------------------------------------------------------------------------------
class CivicAgent:
    def __init__(self, vector_store: VectorStore, graph: GraphManager, explainer: PlainLanguageExplainer):
        self.vs = vector_store
        self.graph = graph
        self.explainer = explainer

    def answer(self, query: str, k: int = 4) -> Dict[str, Any]:
        # Simple heuristic intent detection
        q_lower = query.lower()
        retrieval_results = self.vs.query(query, k=k)
        graph_context = ""
        bill_id = None
        if "bill" in q_lower or "hr-" in q_lower:
            # naive extraction attempt
            tokens = query.replace(",", " ").split()
            candidates = [t for t in tokens if "-" in t and any(c.isdigit() for c in t)]
            if candidates:
                bill_id = candidates[0].upper()
                bs = self.graph.bill_summary(bill_id)
                if bs:
                    graph_context = f"BILL {bill_id} TITLE: {bs['bill'].get('title')} SPONSORS: {[p.get('name') for p in bs['sponsors']]}"
        # Compose answer using retrieval + minimal summarization
        context_snippets = "\n---\n".join([r["text"][:800] for r in retrieval_results])
        final_answer = f"Query: {query}\nRelevant Text Chunks:\n{context_snippets}\n"
        if graph_context:
            final_answer += f"\nGraph Context:\n{graph_context}\n"
        # Could add LLM-based synthesis step
        return {
            "query": query,
            "bill_id_detected": bill_id,
            "retrieval_count": len(retrieval_results),
            "answer": final_answer
        }

# -------------------------------------------------------------------------------------------------
# FastAPI Service
# -------------------------------------------------------------------------------------------------
app = FastAPI(title="Civic Legislative KG API", version="0.1.0")
GLOBAL_OBJECTS = {
    "graph": None,
    "vector_store": None,
    "explainer": None,
    "agent": None
}

class BillSummaryRequest(BaseModel):
    bill_id: str
    plain: bool = True

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/bill/summary")
def bill_summary(req: BillSummaryRequest):
    gm: GraphManager = GLOBAL_OBJECTS["graph"]
    expl: PlainLanguageExplainer = GLOBAL_OBJECTS["explainer"]
    if not gm:
        raise HTTPException(status_code=500, detail="Graph not initialized.")
    data = gm.bill_summary(req.bill_id)
    if not data:
        raise HTTPException(status_code=404, detail="Bill not found")
    if req.plain:
        for sec in data["sections"]:
            sid = sec.get("section_id") or "X"
            sec["plain_language"] = expl.summarize_section(req.bill_id, sid, sec.get("text", "")[:4000])
    return data

@app.post("/query")
def agent_query(req: QueryRequest):
    agent: CivicAgent = GLOBAL_OBJECTS["agent"]
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not ready.")
    return agent.answer(req.query)

# -------------------------------------------------------------------------------------------------
# Sample Data Loader
# -------------------------------------------------------------------------------------------------
def load_sample_into_graph(graph: GraphManager):
    log("Loading sample data into graph...")
    for pol in CONFIG["sample_politicians"]:
        graph.upsert_politician(pol)
    for bill in CONFIG["sample_bills"]:
        graph.upsert_bill(bill)
        for sponsor in bill.get("sponsors", []):
            graph.create_sponsorship(bill["bill_id"], sponsor)
    for vote in CONFIG["sample_votes"]:
        graph.register_vote(vote)
    log("Sample data loaded.")

# -------------------------------------------------------------------------------------------------
# Orchestration Functions
# -------------------------------------------------------------------------------------------------
def build_rag_pipeline(docs: List[LegalDocument], graph: GraphManager) -> VectorStore:
    log("Building RAG index...")
    chunks = []
    for d in docs:
        for i, sec in enumerate(d.sections):
            chunks.append({
                "doc_id": d.doc_id,
                "bill_id": d.bill_id,
                "section_index": i,
                "text": sec["text"][:2000],
                "source": d.source_path
            })
    model_name = os.environ.get("EMBED_MODEL", CONFIG["model_name_default"])
    vs = VectorStore(CONFIG["vector_store_path"], model_name)
    vs.build(chunks)
    return vs

def init_graph() -> GraphManager:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required.")
    gm = GraphManager(uri, user, password)
    gm.setup_constraints()
    return gm

def run_agent_loop(agent: CivicAgent):
    log("Agent interactive mode. Type 'exit' to quit.")
    while True:
        try:
            q = input("Query> ").strip()
            if q.lower() in ("exit", "quit"):
                break
            ans = agent.answer(q)
            print("\n--- ANSWER ---")
            print(ans["answer"])
            print("--------------\n")
        except KeyboardInterrupt:
            break

# -------------------------------------------------------------------------------------------------
# Flowise / n8n / Graphite Hooks (placeholders)
# -------------------------------------------------------------------------------------------------
def notify_flowise(event: str, payload: Dict[str, Any]):
    # Placeholder: send to Flowise webhook
    if os.environ.get("FLOWISE_WEBHOOK"):
        try:
            requests.post(os.environ["FLOWISE_WEBHOOK"], json={"event": event, "payload": payload}, timeout=5)
        except Exception as e:
            log(f"Flowise hook error: {e}")

def trigger_n8n(event: str, payload: Dict[str, Any]):
    if os.environ.get("N8N_WEBHOOK"):
        try:
            requests.post(os.environ["N8N_WEBHOOK"], json={"event": event, "payload": payload}, timeout=5)
        except Exception as e:
            log(f"n8n hook error: {e}")

def graphite_note(message: str):
    # If referencing Graphite.dev workflow, you might call its CLI externally.
    # Placeholder only.
    log(f"[GraphiteNote] {message}")

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Civic Legislative Knowledge Graph & RAG Pipeline (Single-File Scaffold)"
    )
    parser.add_argument("--fetch", action="store_true", help="Fetch/sync external repositories.")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from external repos.")
    parser.add_argument("--build-graph", action="store_true", help="Build or update graph with sample data.")
    parser.add_argument("--build-rag", action="store_true", help="Build vector store index.")
    parser.add_argument("--update-votes", action="store_true", help="Update vote records (placeholder).")
    parser.add_argument("--agent", type=str, default=None, help="Run a single agent query.")
    parser.add_argument("--interactive-agent", action="store_true", help="Start interactive agent loop.")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server.")
    parser.add_argument("--init-all", action="store_true", help="Do fetch, ingest, build graph, build rag.")
    parser.add_argument("--sample", action="store_true", help="Load sample synthetic bill/politician/vote data.")
    args = parser.parse_args()

    docs = []
    vector_store = None
    graph = None
    explainer = PlainLanguageExplainer()

    if args.fetch or args.init_all:
        rf = RepoFetcher(CONFIG["external_repos"])
        rf.sync_all()
        graphite_note("Repositories synchronized.")

    if args.ingest or args.init_all:
        ingestor = DocumentIngestor(CONFIG)
        docs = ingestor.ingest()
        ensure_dirs("data/processed")
        with open("data/processed/ingested_docs.json", "w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in docs], f, indent=2)
        notify_flowise("ingest_complete", {"count": len(docs)})
        trigger_n8n("ingest_complete", {"count": len(docs)})

    if args.build_graph or args.init_all or args.sample:
        graph = init_graph()
        if args.sample or args.init_all:
            load_sample_into_graph(graph)

    if args.build_rag or args.init_all:
        if not docs and os.path.exists("data/processed/ingested_docs.json"):
            with open("data/processed/ingested_docs.json", "r", encoding="utf-8") as f:
                docs_json = json.load(f)
            # Rehydrate minimal objects
            docs = [LegalDocument(**d) for d in docs_json]
        if not docs and args.sample:
            # Build vector store from sample bill text
            sample_docs = []
            for b in CONFIG["sample_bills"]:
                sample_docs.append(LegalDocument(
                    doc_id=b["bill_id"],
                    source_path="(sample)",
                    jurisdiction=b.get("jurisdiction"),
                    bill_id=b["bill_id"],
                    title=b["title"],
                    raw_text=b["text"],
                    sections=[{"text": s["text"]} for s in b["sections"]]
                ))
            docs = sample_docs
        if not docs:
            log("No documents available to build RAG index.")
        else:
            vector_store = build_rag_pipeline(docs, graph)
            notify_flowise("rag_built", {"chunks": "ok"})

    if graph is None:
        # Attempt late init only if needed
        try:
            graph = init_graph()
        except Exception:
            graph = None

    # Build or load vector store if needed for agent operations
    if vector_store is None and (args.agent or args.interactive_agent or args.serve):
        try:
            vector_store = VectorStore(CONFIG["vector_store_path"], os.environ.get("EMBED_MODEL", CONFIG["model_name_default"]))
            vector_store.load()
        except Exception as e:
            log(f"Could not load vector store: {e}")

    agent = None
    if vector_store and graph:
        agent = CivicAgent(vector_store, graph, explainer)

    # Single query
    if args.agent and agent:
        ans = agent.answer(args.agent)
        print(json.dumps(ans, indent=2))

    # Interactive loop
    if args.interactive_agent and agent:
        run_agent_loop(agent)

    # Serve API
    if args.serve:
        GLOBAL_OBJECTS["graph"] = graph
        GLOBAL_OBJECTS["vector_store"] = vector_store
        GLOBAL_OBJECTS["explainer"] = explainer
        GLOBAL_OBJECTS["agent"] = agent
        log("Starting API server at http://0.0.0.0:8088")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8088)

    # Cleanup / close
    if graph:
        graph.close()

if __name__ == "__main__":
    main()