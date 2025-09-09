#!/usr/bin/env python3
# =================================================================================================
# Name: Civic Legislative Knowledge Graph & RAG Platform
# Date: 2025-09-09
# Script Name: civic_legis_platform.py
# Version: 0.2.0
# Log Summary:
#   - Added govinfo.gov Bulk Data + API ingestion (collections: BILLSTATUS, PLAW, etc.)
#   - Incremental package tracking & ZIP extraction.
#   - Bloom perspective export generator for Neo4j Bloom visualization.
#   - Docker Compose emission option.
#   - Extended CLI: govinfo sync flags, perspective generation, docker-compose emitter.
#   - Inherited core features: multi-source ingestion, RAG index, Neo4j KG, agentic QA, plain-language
#     explanation via LLM (LocalAI/OpenAI compatible), sample data seeding.
# Description:
#   Unified, single-file scaffold to:
#     1. Fetch & sync external Git sources (optional).
#     2. Ingest documents from:
#         - Local sample or synced repositories
#         - govinfo.gov (via API & bulk ZIP) for legislative collections
#     3. Parse and normalize legislative artifacts into internal LegalDocument objects.
#     4. Build / update Neo4j knowledge graph (Bills, Sections, Politicians, Votes).
#     5. Build vector (FAISS or in-memory) retrieval store for RAG.
#     6. Provide plain-language translation of legislative sections.
#     7. Offer agentic query interface (retrieval + graph context).
#     8. Expose FastAPI service for programmatic access.
#     9. Generate Bloom perspective JSON for quick KG visualization.
#    10. Optionally emit docker-compose.yml for full stack deployment.
# Change Summary:
#   - 0.1.0: Initial scaffold (previous version).
#   - 0.2.0: govinfo integration, incremental tracking, perspective generation, Compose emitter.
# Inputs:
#   Environment Variables (core):
#       NEO4J_URI (default: bolt://neo4j:7687 or bolt://localhost:7687)
#       NEO4J_USER (default: neo4j)
#       NEO4J_PASSWORD (required)
#       LOCALAI_ENDPOINT (e.g. http://localai:8080/v1)
#       OPENAI_API_KEY (if using remote OpenAI)
#       MODEL_NAME (LLM model; fallback in config)
#       EMBED_MODEL (embedding model; default all-MiniLM-L6-v2)
#       GOVINFO_API_KEY (govinfo API key - improves rate & completeness)
#   Command Line (see argparse in code):
#       --govinfo-sync, --govinfo-collections, --govinfo-days
#       --emit-bloom, --emit-docker-compose, etc.
# Outputs:
#   - data/govinfo/* (downloaded ZIPs + extracted)
#   - data/processed/ingested_docs.json
#   - data/vector_store/{index.faiss|index.npy, meta.json}
#   - bloom_perspective.json (or generated)
#   - docker-compose.generated.yml (when emitted)
#   - FastAPI service on chosen port (default 8088)
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
import zipfile
import argparse
import textwrap
import datetime
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# -------------------------------------------------------------------------------------------------
# Runtime dependencies
# -------------------------------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "requests", "tqdm", "pydantic", "fastapi", "uvicorn",
    "python-dotenv", "neo4j", "beautifulsoup4", "lxml",
    "PyPDF2", "sentence-transformers", "faiss-cpu", "transformers",
    "numpy", "scikit-learn"
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

# Post-install imports
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
# Global CONFIG
# -------------------------------------------------------------------------------------------------
CONFIG = {
    "external_repos": [
        # Extend with actual repos if desired
        {"name": "us-congress-bills-sample", "git": "https://github.com/unitedstates/congress.git", "path": "data/external/congress"}
    ],
    "document_globs": [
        "data/external/congress/**/*.xml",
        "data/external/congress/**/*.txt",
        "data/govinfo/**/*.xml",
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
            "results": {"P-001": "YEA", "P-002": "YEA"}
        }
    ],
    "bloom_perspective_path": "bloom_perspective.json"
}

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
# Repo Fetcher
# -------------------------------------------------------------------------------------------------
class RepoFetcher:
    def __init__(self, repos: List[Dict[str, str]]):
        self.repos = repos

    def sync_all(self):
        for repo in self.repos:
            path = repo["path"]
            url = repo["git"]
            if os.path.isdir(os.path.join(path, ".git")):
                log(f"Updating repo {repo['name']}")
                subprocess.call(["git", "-C", path, "pull", "--ff-only"])
            else:
                log(f"Cloning repo {repo['name']} → {path}")
                ensure_dirs(os.path.dirname(path))
                subprocess.call(["git", "clone", "--depth", "1", url, path])

# -------------------------------------------------------------------------------------------------
# Data Models
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
    meta: Dict[str, Any] = field(default_factory=dict)

# -------------------------------------------------------------------------------------------------
# Generic File Ingestor
# -------------------------------------------------------------------------------------------------
class DocumentIngestor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def collect_files(self) -> List[str]:
        files = []
        for pattern in self.config["document_globs"]:
            files.extend(glob.glob(pattern, recursive=True))
        files = [f for f in files if os.path.isfile(f)]
        return sorted(set(files))

    def parse_file(self, path: str) -> Optional[LegalDocument]:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                with open(path, "rb") as f:
                    reader = PdfReader(f)
                    pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
            elif ext in (".xml", ".html", ".htm"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "lxml")
                text = soup.get_text("\n")
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if not text.strip():
                return None
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            title = lines[0][:240] if lines else "Untitled"
            return LegalDocument(
                doc_id=str(uuid.uuid4()),
                source_path=path,
                jurisdiction="US",
                bill_id=None,
                title=title,
                raw_text=text,
                sections=self._chunk_sections(text),
                meta={"source_type": "generic_file"}
            )
        except Exception as e:
            log(f"Parse fail {path}: {e}")
            return None

    def _chunk_sections(self, text: str, max_len: int = 1500) -> List[Dict[str, Any]]:
        words = text.split()
        current = []
        sections = []
        for w in words:
            current.append(w)
            if len(" ".join(current)) >= max_len:
                sections.append({"text": " ".join(current)})
                current = []
        if current:
            sections.append({"text": " ".join(current)})
        return sections

    def ingest(self) -> List[LegalDocument]:
        collected = self.collect_files()
        docs = []
        for f in tqdm(collected, desc="File ingestion"):
            doc = self.parse_file(f)
            if doc:
                docs.append(doc)
        log(f"Ingested {len(docs)} local documents.")
        return docs

# -------------------------------------------------------------------------------------------------
# govinfo.gov Ingestor
# -------------------------------------------------------------------------------------------------
class GovInfoIngestor:
    """
    Ingests legislative packages from govinfo.gov via its API and Bulk Data ZIP archives.

    Strategy:
      1. For each collection (e.g., BILLSTATUS, PLAW) determine date window (N days).
      2. Call /collections/{collection}/{start}/{end}? to list packages.
      3. For each packageId not yet processed (tracked in state JSON), fetch summary & zip:
          /packages/{packageId}/summary
          /packages/{packageId}/zip
      4. Extract ZIP → XML/JSON files into data/govinfo/{collection}/{packageId}/
      5. Parse primary XML to derive:
          - title, bill number (if available), congress, sponsors (basic heuristics)
      6. Map to LegalDocument objects.
    """
    API_BASE = "https://api.govinfo.gov"
    BULK_BASE = "https://www.govinfo.gov/bulkdata"
    STATE_FILE = "data/govinfo/govinfo_state.json"

    def __init__(self, collections: List[str], days: int, api_key: Optional[str] = None):
        self.collections = collections
        self.days = days
        self.api_key = api_key or os.environ.get("GOVINFO_API_KEY")
        ensure_dirs("data/govinfo")
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.STATE_FILE):
            with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"processed_packages": {}}

    def _persist_state(self):
        with open(self.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _headers(self):
        h = {"User-Agent": "CivicLegisBot/0.2"}
        if self.api_key:
            h["X-Api-Key"] = self.api_key
        return h

    def _list_packages(self, collection: str, start: str, end: str) -> List[Dict[str, Any]]:
        url = f"{self.API_BASE}/collections/{collection}/{start}/{end}"
        params = {"pageSize": 1000}
        packages = []
        offset = 0
        while True:
            params["offset"] = offset
            r = requests.get(url, headers=self._headers(), params=params, timeout=60)
            if r.status_code != 200:
                log(f"[govinfo] Collection list error {r.status_code} {r.text[:150]}")
                break
            data = r.json()
            pkgs = data.get("packages", [])
            if not pkgs:
                break
            packages.extend(pkgs)
            if len(pkgs) < params["pageSize"]:
                break
            offset += params["pageSize"]
        return packages

    def _download_zip(self, package_id: str, target_dir: str):
        ensure_dirs(target_dir)
        url = f"{self.API_BASE}/packages/{package_id}/zip"
        r = requests.get(url, headers=self._headers(), timeout=120)
        if r.status_code != 200:
            log(f"[govinfo] ZIP fetch fail {package_id}: {r.status_code}")
            return False
        zip_path = os.path.join(target_dir, f"{package_id}.zip")
        with open(zip_path, "wb") as f:
            f.write(r.content)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(target_dir)
        except Exception as e:
            log(f"[govinfo] ZIP extract error {package_id}: {e}")
            return False
        return True

    def _parse_package(self, collection: str, package_dir: str, package_id: str) -> Optional[LegalDocument]:
        # Heuristic: find largest XML file
        xml_files = [os.path.join(dp, f) for dp, _, files in os.walk(package_dir)
                     for f in files if f.lower().endswith(".xml")]
        if not xml_files:
            return None
        xml_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        main_xml = xml_files[0]
        try:
            with open(main_xml, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml-xml")
        except Exception:
            with open(main_xml, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "lxml")
        # Attempt extraction
        title = None
        for candidate in ["title", "dc:title", "official-title", "docTitle"]:
            el = soup.find(candidate)
            if el and el.text.strip():
                title = el.text.strip()
                break
        if not title:
            title = package_id
        # Bill number heuristics
        bill_num = None
        for tag in ["billNumber", "bill-number", "billNum"]:
            el = soup.find(tag)
            if el and el.text.strip():
                bill_num = el.text.strip()
                break
        # Sections (very naive: split paragraphs)
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["section", "p", "Paragraph"]) if p.get_text(strip=True)]
        unique_paras = []
        seen = set()
        for para in paragraphs:
            if para not in seen:
                seen.add(para)
                unique_paras.append(para)
        sections = []
        for idx, chunk in enumerate(unique_paras[:40]):  # limit to prevent explosion
            sections.append({"number": str(idx + 1), "heading": None, "text": chunk[:8000]})
        text_full = "\n\n".join(unique_paras)
        doc = LegalDocument(
            doc_id=package_id,
            source_path=main_xml,
            jurisdiction="US",
            bill_id=bill_num or package_id,
            title=title,
            raw_text=text_full,
            sections=sections if sections else [{"text": text_full[:8000]}],
            meta={
                "collection": collection,
                "package_id": package_id
            }
        )
        return doc

    def sync(self) -> List[LegalDocument]:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=self.days)
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        new_docs = []
        for collection in self.collections:
            log(f"[govinfo] Sync collection {collection} ({start_str} → {end_str})")
            packages = self._list_packages(collection, start_str, end_str)
            log(f"[govinfo] Found {len(packages)} package entries in window.")
            for pkg in tqdm(packages, desc=f"Collection {collection}"):
                package_id = pkg.get("packageId")
                if not package_id:
                    continue
                if package_id in self.state["processed_packages"]:
                    continue
                target_dir = os.path.join("data", "govinfo", collection, package_id)
                if not self._download_zip(package_id, target_dir):
                    continue
                doc = self._parse_package(collection, target_dir, package_id)
                if doc:
                    new_docs.append(doc)
                self.state["processed_packages"][package_id] = {
                    "collection": collection,
                    "ts": datetime.datetime.utcnow().isoformat()
                }
                # Persist periodically
                if len(self.state["processed_packages"]) % 25 == 0:
                    self._persist_state()
            self._persist_state()
        log(f"[govinfo] New documents parsed: {len(new_docs)}")
        return new_docs

# -------------------------------------------------------------------------------------------------
# Vector Store
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
        if faiss:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            log("FAISS not available; fallback to numpy index.")
            self.index = embeddings
        self.meta = chunks
        self._persist()

    def _persist(self):
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)
        if faiss and isinstance(self.index, faiss.Index):
            faiss.write_index(self.index, os.path.join(self.path, "index.faiss"))
        else:
            np.save(os.path.join(self.path, "index.npy"), self.index)
        log("Vector store persisted.")

    def load(self):
        with open(os.path.join(self.path, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        if faiss and os.path.exists(os.path.join(self.path, "index.faiss")):
            self.index = faiss.read_index(os.path.join(self.path, "index.faiss"))
        else:
            self.index = np.load(os.path.join(self.path, "index.npy"))
        log("Vector store loaded.")

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query_text]).astype("float32")
        if faiss and isinstance(self.index, faiss.Index):
            faiss.normalize_L2(q_emb)
            distances, idxs = self.index.search(q_emb, k)
            out = []
            for rank, (i, dist) in enumerate(zip(idxs[0], distances[0])):
                meta = self.meta[i].copy()
                meta["score"] = float(dist)
                meta["rank"] = rank
                out.append(meta)
            return out
        else:
            embeddings = self.index
            qn = q_emb / np.linalg.norm(q_emb)
            emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            sims = emb_norm @ qn.T
            sims = sims.flatten()
            top = sims.argsort()[-k:][::-1]
            out = []
            for rank, i in enumerate(top):
                m = self.meta[i].copy()
                m["score"] = float(sims[i])
                m["rank"] = rank
                out.append(m)
            return out

# -------------------------------------------------------------------------------------------------
# Graph Manager
# -------------------------------------------------------------------------------------------------
class GraphManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        with self.driver.session() as s:
            for c in CONFIG["graph"]["constraints"]:
                s.run(c)
        log("Graph constraints ensured.")

    def upsert_bill(self, bill: Dict[str, Any]):
        q = """
        MERGE (b:Bill {bill_id:$bill_id})
        SET b.title=$title, b.jurisdiction=$jurisdiction, b.text=$text, b.collection=$collection
        """
        with self.driver.session() as s:
            s.run(q, bill_id=bill["bill_id"],
                  title=b.get("title"), jurisdiction=b.get("jurisdiction"),
                  text=b.get("text"), collection=b.get("collection"))

        for sec in bill.get("sections", []):
            self.upsert_section(bill["bill_id"], sec)

    def upsert_section(self, bill_id: str, section: Dict[str, Any]):
        sid = f"{bill_id}-SEC-{section.get('number','X')}"
        q = """
        MERGE (s:Section {section_id:$sid})
        SET s.number=$number, s.heading=$heading, s.text=$text
        WITH s
        MATCH (b:Bill {bill_id:$bill_id})
        MERGE (b)-[:HAS_SECTION]->(s)
        """
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
        OPTIONAL MATCH (p:Politician)-[:SPONSORS]->(b)
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

# -------------------------------------------------------------------------------------------------
# Plain Language Explainer
# -------------------------------------------------------------------------------------------------
class PlainLanguageExplainer:
    def __init__(self):
        self.endpoint = os.environ.get("LOCALAI_ENDPOINT") or "https://api.openai.com/v1"
        self.api_key = os.environ.get("OPENAI_API_KEY", "DUMMY_KEY")
        self.model = os.environ.get("MODEL_NAME", CONFIG["llm_plain_model_fallback"])
        ensure_dirs("data/cache")
        self.cache_file = "data/cache/summaries.json"
        self.cache = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _persist(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def summarize_section(self, bill_id: str, section_id: str, text: str) -> str:
        key = f"{bill_id}:{section_id}"
        if key in self.cache:
            return self.cache[key]
        prompt = f"Rewrite the following legislative text in clear, plain language for the average citizen:\n\n{text}\n\nPlain language:"
        try:
            resp = requests.post(
                f"{self.endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a legal plain language assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                },
                timeout=90
            )
            if resp.status_code >= 400:
                raise RuntimeError(resp.text[:200])
            content = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            content = f"[LLM_FALLBACK] Could not summarize: {e}"
        self.cache[key] = content
        self._persist()
        return content

# -------------------------------------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------------------------------------
class CivicAgent:
    def __init__(self, vs: VectorStore, graph: GraphManager, explainer: PlainLanguageExplainer):
        self.vs = vs
        self.graph = graph
        self.explainer = explainer

    def answer(self, query: str, k: int = 6) -> Dict[str, Any]:
        retrieval = self.vs.query(query, k=k) if self.vs else []
        bill_id = None
        graph_context = ""
        tokens = query.split()
        for t in tokens:
            if "-" in t and any(c.isdigit() for c in t):
                bill_id = t.upper().strip(",.")
                break
        if bill_id:
            bs = self.graph.bill_summary(bill_id) if self.graph else None
            if bs:
                graph_context = f"BILL {bill_id} Title: {bs['bill'].get('title')} Sponsors: {[p.get('name') for p in bs['sponsors']]}"
        context_snippets = "\n---\n".join([r["text"][:700] for r in retrieval])
        answer_text = f"Query: {query}\n\nRelevant Chunks:\n{context_snippets}\n"
        if graph_context:
            answer_text += f"\nGraph Context:\n{graph_context}\n"
        return {
            "query": query,
            "bill_id": bill_id,
            "chunks_used": len(retrieval),
            "answer": answer_text
        }

# -------------------------------------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------------------------------------
app = FastAPI(title="Civic Legislative KG API", version="0.2.0")
GLOBAL = {"graph": None, "vector_store": None, "explainer": None, "agent": None}

class BillSummaryRequest(BaseModel):
    bill_id: str
    plain: bool = True

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}

@app.post("/bill/summary")
def bill_summary(req: BillSummaryRequest):
    gm: GraphManager = GLOBAL["graph"]
    expl: PlainLanguageExplainer = GLOBAL["explainer"]
    if not gm:
        raise HTTPException(status_code=500, detail="Graph not ready.")
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
    agent: CivicAgent = GLOBAL["agent"]
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not ready.")
    return agent.answer(req.query)

# -------------------------------------------------------------------------------------------------
# Sample Data Loader
# -------------------------------------------------------------------------------------------------
def load_sample(graph: GraphManager):
    log("Loading sample dataset...")
    for p in CONFIG["sample_politicians"]:
        graph.upsert_politician(p)
    for b in CONFIG["sample_bills"]:
        graph.upsert_bill(b)
        for s in b.get("sponsors", []):
            graph.create_sponsorship(b["bill_id"], s)
    for v in CONFIG["sample_votes"]:
        graph.register_vote(v)
    log("Sample data loaded.")

# -------------------------------------------------------------------------------------------------
# RAG Builder
# -------------------------------------------------------------------------------------------------
def build_rag(docs: List[LegalDocument], graph: Optional[GraphManager]) -> VectorStore:
    log("Building RAG index...")
    chunks = []
    for d in docs:
        for i, s in enumerate(d.sections):
            chunks.append({
                "doc_id": d.doc_id,
                "bill_id": d.bill_id,
                "section_index": i,
                "text": s["text"][:3000],
                "source": d.source_path,
                "collection": d.meta.get("collection") if d.meta else None
            })
        # Optionally ingest each doc as Bill
        if graph and d.bill_id:
            graph.upsert_bill({
                "bill_id": d.bill_id,
                "title": d.title,
                "jurisdiction": d.jurisdiction,
                "text": d.raw_text[:15000],
                "sections": [{"number": str(idx+1), "heading": sec.get("heading"),
                              "text": sec.get("text")} for idx, sec in enumerate(d.sections[:50])],
                "collection": d.meta.get("collection") if d.meta else None
            })
    model_name = os.environ.get("EMBED_MODEL", CONFIG["model_name_default"])
    vs = VectorStore(CONFIG["vector_store_path"], model_name)
    vs.build(chunks)
    return vs

# -------------------------------------------------------------------------------------------------
# Bloom Perspective Generator
# -------------------------------------------------------------------------------------------------
def emit_bloom_perspective(path: str):
    perspective = {
        "name": "CivicLegislationPerspective",
        "version": "1.0",
        "lastUpdated": datetime.datetime.utcnow().isoformat(),
        "categories": [
            {
                "name": "Bills",
                "label": "Bill",
                "cypher": "MATCH (b:Bill) RETURN b",
                "style": {"color": "#1f77b4", "size": 55}
            },
            {
                "name": "Sections",
                "label": "Section",
                "cypher": "MATCH (s:Section) RETURN s",
                "style": {"color": "#ff7f0e", "size": 30}
            },
            {
                "name": "Politicians",
                "label": "Politician",
                "cypher": "MATCH (p:Politician) RETURN p",
                "style": {"color": "#2ca02c", "size": 50}
            }
        ],
        "relationships": [
            {
                "name": "SPONSORS",
                "cypher": "MATCH (p:Politician)-[r:SPONSORS]->(b:Bill) RETURN p,r,b",
                "style": {"color": "#9467bd"}
            },
            {
                "name": "HAS_SECTION",
                "cypher": "MATCH (b:Bill)-[r:HAS_SECTION]->(s:Section) RETURN b,r,s",
                "style": {"color": "#8c564b"}
            },
            {
                "name": "VOTED_ON",
                "cypher": "MATCH (p:Politician)-[r:VOTED_ON]->(b:Bill) RETURN p,r,b",
                "style": {"color": "#d62728"}
            }
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(perspective, f, indent=2)
    log(f"Bloom perspective written: {path}")

# -------------------------------------------------------------------------------------------------
# Docker Compose Emitter
# -------------------------------------------------------------------------------------------------
def emit_docker_compose(path: str):
    compose = textwrap.dedent(f"""
    # =========================================================================================
    # Name: docker-compose.generated.yml
    # Date: {datetime.date.today().isoformat()}
    # Version: 0.2.0
    # Description: Deploys:
    #   - Neo4j (graph)
    #   - LocalAI (optional local inference)
    #   - Civic API (this script)
    #   - Flowise (optional orchestration)
    #   - n8n automation
    # Ports:
    #   Neo4j HTTP: 7474, Bolt: 7687
    #   API: 8088
    #   LocalAI: 8080
    #   Flowise: 3000
    #   n8n: 5678
    # =========================================================================================
    version: "3.9"
    services:
      neo4j:
        image: neo4j:5.20
        container_name: neo4j
        restart: unless-stopped
        environment:
          NEO4J_AUTH: "neo4j/${{NEO4J_PASSWORD}}"
          NEO4J_dbms_memory_heap_initial__size: 512m
          NEO4J_dbms_memory_heap_max__size: 1024m
        ports:
          - "7474:7474"
          - "7687:7687"
        volumes:
          - neo4j_data:/data
          - neo4j_logs:/logs

      localai:
        image: localai/localai:latest
        container_name: localai
        restart: unless-stopped
        environment:
          - DEBUG=false
        volumes:
          - ./models:/models
        ports:
          - "8080:8080"

      api:
        build:
          context: .
          dockerfile: Dockerfile.civic
        container_name: civic_api
        restart: unless-stopped
        environment:
          - NEO4J_URI=bolt://neo4j:7687
          - NEO4J_USER=neo4j
          - NEO4J_PASSWORD=${{NEO4J_PASSWORD}}
          - LOCALAI_ENDPOINT=http://localai:8080/v1
          - EMBED_MODEL=all-MiniLM-L6-v2
          - GOVINFO_API_KEY=${{GOVINFO_API_KEY}}
        depends_on:
          - neo4j
        volumes:
          - ./data:/app/data
        command: >
          sh -c "python civic_legis_platform.py --govinfo-sync --govinfo-collections BILLSTATUS,PLAW
                 --govinfo-days 7 --build-rag --serve"
        ports:
          - "8088:8088"

      flowise:
        image: flowiseai/flowise:latest
        container_name: flowise
        restart: unless-stopped
        environment:
          - PORT=3000
        ports:
          - "3000:3000"

      n8n:
        image: n8nio/n8n:latest
        container_name: n8n
        restart: unless-stopped
        environment:
          - N8N_BASIC_AUTH_ACTIVE=false
        ports:
          - "5678:5678"
        volumes:
          - n8n_data:/home/node/.n8n

    volumes:
      neo4j_data:
      neo4j_logs:
      n8n_data:
    """).strip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(compose)
    log(f"Docker Compose emitted: {path}")

    # Also emit a Dockerfile for the api service
    dockerfile = textwrap.dedent(f"""
    # ======================================================================
    # Name: Dockerfile.civic
    # Date: {datetime.date.today().isoformat()}
    # Version: 0.2.0
    # Description: Builds container for civic_legis_platform.py API service
    # ======================================================================
    FROM python:3.11-slim
    WORKDIR /app
    COPY civic_legis_platform.py /app/
    RUN pip install --no-cache-dir requests tqdm pydantic fastapi uvicorn python-dotenv neo4j \\
        beautifulsoup4 lxml PyPDF2 sentence-transformers faiss-cpu transformers numpy scikit-learn
    EXPOSE 8088
    CMD ["python", "civic_legis_platform.py", "--serve"]
    """).strip() + "\n"
    with open("Dockerfile.civic", "w", encoding="utf-8") as f:
        f.write(dockerfile)
    log("Dockerfile.civic emitted.")

# -------------------------------------------------------------------------------------------------
# Initialization Helpers
# -------------------------------------------------------------------------------------------------
def init_graph() -> GraphManager:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required.")
    gm = GraphManager(uri, user, password)
    gm.setup_constraints()
    return gm

def run_interactive(agent: CivicAgent):
    log("Interactive agent shell. Type 'exit' to quit.")
    while True:
        try:
            q = input("Query> ").strip()
            if q.lower() in ("exit", "quit"):
                break
            ans = agent.answer(q)
            print("\n=== ANSWER ===")
            print(ans["answer"])
            print("==============\n")
        except KeyboardInterrupt:
            break

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Civic Legislative KG & RAG Platform")
    parser.add_argument("--fetch", action="store_true", help="Sync external Git repos.")
    parser.add_argument("--ingest", action="store_true", help="Ingest local repo/sample documents.")
    parser.add_argument("--govinfo-sync", action="store_true", help="Ingest from govinfo API/ZIP.")
    parser.add_argument("--govinfo-collections", type=str, default="BILLSTATUS",
                        help="Comma list (e.g. BILLSTATUS,PLAW).")
    parser.add_argument("--govinfo-days", type=int, default=7, help="Days back for govinfo.")
    parser.add_argument("--build-graph", action="store_true", help="Build/seed graph.")
    parser.add_argument("--build-rag", action="store_true", help="Build vector store.")
    parser.add_argument("--agent", type=str, help="Run single agent query.")
    parser.add_argument("--interactive-agent", action="store_true", help="Interactive agent shell.")
    parser.add_argument("--serve", action="store_true", help="Run API server.")
    parser.add_argument("--init-all", action="store_true", help="Fetch, ingest local, build graph, build RAG.")
    parser.add_argument("--sample", action="store_true", help="Load sample synthetic data.")
    parser.add_argument("--emit-bloom", action="store_true", help="Generate Bloom perspective JSON.")
    parser.add_argument("--emit-docker-compose", action="store_true", help="Emit docker-compose + Dockerfile.")
    args = parser.parse_args()

    ensure_dirs("data", "data/processed", "data/cache")

    docs: List[LegalDocument] = []
    graph = None
    explainer = PlainLanguageExplainer()
    vector_store = None

    if args.fetch or args.init_all:
        RepoFetcher(CONFIG["external_repos"]).sync_all()

    if args.ingest or args.init_all:
        di = DocumentIngestor(CONFIG)
        docs_local = di.ingest()
        docs.extend(docs_local)

    if args.govinfo_sync:
        collections = [c.strip().upper() for c in args.govinfo_collections.split(",") if c.strip()]
        gi = GovInfoIngestor(collections, args.govinfo_days, api_key=os.environ.get("GOVINFO_API_KEY"))
        docs_gov = gi.sync()
        docs.extend(docs_gov)

    if docs:
        with open("data/processed/ingested_docs.json", "w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in docs], f, indent=2)
        log("Persisted ingested_docs.json")

    if args.build_graph or args.init_all or args.sample or args.build_rag:
        try:
            graph = init_graph()
        except Exception as e:
            log(f"Graph init failed (continuing without graph): {e}")
            graph = None

    if args.sample or args.init_all:
        if graph:
            load_sample(graph)

    # If building RAG and no docs loaded yet, attempt reload
    if (args.build_rag or args.init_all) and not docs and os.path.exists("data/processed/ingested_docs.json"):
        with open("data/processed/ingested_docs.json", "r", encoding="utf-8") as f:
            docs_json = json.load(f)
        docs = [LegalDocument(**d) for d in docs_json]

    if args.build_rag or args.init_all:
        if docs:
            vector_store = build_rag(docs, graph)
        else:
            log("No documents to build RAG index.")

    # Load existing vector store if needed for queries / API
    if not vector_store and (args.agent or args.interactive_agent or args.serve):
        try:
            vector_store = VectorStore(CONFIG["vector_store_path"],
                                       os.environ.get("EMBED_MODEL", CONFIG["model_name_default"]))
            vector_store.load()
        except Exception as e:
            log(f"Vector store load failed: {e}")
            vector_store = None

    agent = None
    if vector_store and graph:
        agent = CivicAgent(vector_store, graph, explainer)

    if args.emit_bloom:
        emit_bloom_perspective(CONFIG["bloom_perspective_path"])

    if args.emit_docker_compose:
        emit_docker_compose("docker-compose.generated.yml")

    if args.agent and agent:
        ans = agent.answer(args.agent)
        print(json.dumps(ans, indent=2))

    if args.interactive_agent and agent:
        run_interactive(agent)

    if args.serve:
        GLOBAL["graph"] = graph
        GLOBAL["vector_store"] = vector_store
        GLOBAL["explainer"] = explainer
        GLOBAL["agent"] = agent
        log("Starting API server at http://0.0.0.0:8088")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8088)

    if graph:
        graph.close()

if __name__ == "__main__":
    main()