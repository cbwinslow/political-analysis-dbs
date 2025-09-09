from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn

from app.models.database import get_db
from app.models.schemas import (
    LegislatorCreate, LegislatorResponse,
    BillCreate, BillResponse,
    VoteCreate, VoteResponse,
    SimilaritySearch
)
from app.services.legislator_service import LegislatorService
from app.services.bill_service import BillService
from app.services.vote_service import VoteService
from app.services.embedding_service import EmbeddingService
from app.services.graph_service import GraphService

app = FastAPI(
    title="Political Analysis Database API",
    description="A comprehensive API for political legislature analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
legislator_service = LegislatorService()
bill_service = BillService()
vote_service = VoteService()
embedding_service = EmbeddingService()
graph_service = GraphService()

@app.get("/")
async def root():
    return {"message": "Political Analysis Database API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Legislator endpoints
@app.post("/api/v1/legislators/", response_model=LegislatorResponse)
async def create_legislator(legislator: LegislatorCreate, db: Session = Depends(get_db)):
    return await legislator_service.create_legislator(db, legislator)

@app.get("/api/v1/legislators/", response_model=List[LegislatorResponse])
async def get_legislators(
    skip: int = 0, 
    limit: int = 100, 
    party: Optional[str] = None,
    state: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return await legislator_service.get_legislators(db, skip, limit, party, state)

@app.get("/api/v1/legislators/{legislator_id}", response_model=LegislatorResponse)
async def get_legislator(legislator_id: str, db: Session = Depends(get_db)):
    legislator = await legislator_service.get_legislator(db, legislator_id)
    if not legislator:
        raise HTTPException(status_code=404, detail="Legislator not found")
    return legislator

@app.post("/api/v1/legislators/search/similar")
async def find_similar_legislators(search: SimilaritySearch, db: Session = Depends(get_db)):
    return await legislator_service.find_similar_legislators(db, search.query, search.threshold, search.limit)

# Bill endpoints
@app.post("/api/v1/bills/", response_model=BillResponse)
async def create_bill(bill: BillCreate, db: Session = Depends(get_db)):
    return await bill_service.create_bill(db, bill)

@app.get("/api/v1/bills/", response_model=List[BillResponse])
async def get_bills(
    skip: int = 0, 
    limit: int = 100, 
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return await bill_service.get_bills(db, skip, limit, status)

@app.get("/api/v1/bills/{bill_id}", response_model=BillResponse)
async def get_bill(bill_id: str, db: Session = Depends(get_db)):
    bill = await bill_service.get_bill(db, bill_id)
    if not bill:
        raise HTTPException(status_code=404, detail="Bill not found")
    return bill

@app.post("/api/v1/bills/search/similar")
async def find_similar_bills(search: SimilaritySearch, db: Session = Depends(get_db)):
    return await bill_service.find_similar_bills(db, search.query, search.threshold, search.limit)

# Vote endpoints
@app.post("/api/v1/votes/", response_model=VoteResponse)
async def create_vote(vote: VoteCreate, db: Session = Depends(get_db)):
    return await vote_service.create_vote(db, vote)

@app.get("/api/v1/votes/")
async def get_votes(
    skip: int = 0, 
    limit: int = 100,
    legislator_id: Optional[str] = None,
    bill_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return await vote_service.get_votes(db, skip, limit, legislator_id, bill_id)

# Analytics endpoints
@app.get("/api/v1/analytics/voting-patterns/{legislator_id}")
async def get_voting_patterns(legislator_id: str, db: Session = Depends(get_db)):
    return await vote_service.get_voting_patterns(db, legislator_id)

@app.get("/api/v1/analytics/bill-support/{bill_id}")
async def get_bill_support(bill_id: str, db: Session = Depends(get_db)):
    return await bill_service.get_bill_support(db, bill_id)

@app.get("/api/v1/analytics/party-alignment")
async def get_party_alignment(db: Session = Depends(get_db)):
    return await vote_service.get_party_alignment(db)

# Graph analytics endpoints
@app.get("/api/v1/graph/relationships/{entity_id}")
async def get_entity_relationships(entity_id: str):
    return await graph_service.get_entity_relationships(entity_id)

@app.get("/api/v1/graph/influence-network")
async def get_influence_network():
    return await graph_service.get_influence_network()

@app.get("/api/v1/graph/community-detection")
async def detect_communities():
    return await graph_service.detect_communities()

# Embedding endpoints
@app.post("/api/v1/embeddings/generate")
async def generate_embedding(text: str):
    return await embedding_service.generate_embedding(text)

@app.post("/api/v1/embeddings/batch")
async def generate_batch_embeddings(texts: List[str]):
    return await embedding_service.generate_batch_embeddings(texts)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)