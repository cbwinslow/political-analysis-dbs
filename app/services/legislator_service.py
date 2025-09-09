from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
import uuid

from app.models.schemas import LegislatorCreate, LegislatorResponse
from app.services.embedding_service import EmbeddingService

class LegislatorService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    async def create_legislator(self, db: Session, legislator: LegislatorCreate) -> LegislatorResponse:
        """Create a new legislator with bio embedding"""
        legislator_id = str(uuid.uuid4())
        
        # Generate embedding for bio if provided
        bio_embedding = None
        if legislator.bio_text:
            bio_embedding = await self.embedding_service.generate_embedding(legislator.bio_text)
        
        query = text("""
            INSERT INTO legislators (id, name, party, state, district, chamber, bio_embedding)
            VALUES (:id, :name, :party, :state, :district, :chamber, :bio_embedding)
            RETURNING *
        """)
        
        result = db.execute(query, {
            "id": legislator_id,
            "name": legislator.name,
            "party": legislator.party,
            "state": legislator.state,
            "district": legislator.district,
            "chamber": legislator.chamber,
            "bio_embedding": bio_embedding
        })
        
        db.commit()
        row = result.fetchone()
        
        return LegislatorResponse(
            id=row.id,
            name=row.name,
            party=row.party,
            state=row.state,
            district=row.district,
            chamber=row.chamber,
            bio_embedding=row.bio_embedding,
            created_at=row.created_at,
            updated_at=row.updated_at
        )

    async def get_legislators(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        party: Optional[str] = None,
        state: Optional[str] = None
    ) -> List[LegislatorResponse]:
        """Get legislators with optional filtering"""
        conditions = []
        params = {"skip": skip, "limit": limit}
        
        if party:
            conditions.append("party = :party")
            params["party"] = party
            
        if state:
            conditions.append("state = :state")
            params["state"] = state
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = text(f"""
            SELECT * FROM legislators 
            {where_clause}
            ORDER BY created_at DESC
            OFFSET :skip LIMIT :limit
        """)
        
        result = db.execute(query, params)
        rows = result.fetchall()
        
        return [
            LegislatorResponse(
                id=row.id,
                name=row.name,
                party=row.party,
                state=row.state,
                district=row.district,
                chamber=row.chamber,
                bio_embedding=row.bio_embedding,
                voting_record_embedding=row.voting_record_embedding,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            for row in rows
        ]

    async def get_legislator(self, db: Session, legislator_id: str) -> Optional[LegislatorResponse]:
        """Get a single legislator by ID"""
        query = text("SELECT * FROM legislators WHERE id = :id")
        result = db.execute(query, {"id": legislator_id})
        row = result.fetchone()
        
        if not row:
            return None
            
        return LegislatorResponse(
            id=row.id,
            name=row.name,
            party=row.party,
            state=row.state,
            district=row.district,
            chamber=row.chamber,
            bio_embedding=row.bio_embedding,
            voting_record_embedding=row.voting_record_embedding,
            created_at=row.created_at,
            updated_at=row.updated_at
        )

    async def find_similar_legislators(
        self, 
        db: Session, 
        query_text: str, 
        threshold: float = 0.8, 
        limit: int = 10
    ) -> List[dict]:
        """Find legislators similar to the query text"""
        query_embedding = await self.embedding_service.generate_embedding(query_text)
        
        if not query_embedding:
            return []
        
        sql_query = text("""
            SELECT l.*, 1 - (l.bio_embedding <=> :query_embedding) AS similarity
            FROM legislators l
            WHERE l.bio_embedding IS NOT NULL
            AND 1 - (l.bio_embedding <=> :query_embedding) > :threshold
            ORDER BY l.bio_embedding <=> :query_embedding
            LIMIT :limit
        """)
        
        result = db.execute(sql_query, {
            "query_embedding": query_embedding,
            "threshold": threshold,
            "limit": limit
        })
        
        rows = result.fetchall()
        
        return [
            {
                "legislator": LegislatorResponse(
                    id=row.id,
                    name=row.name,
                    party=row.party,
                    state=row.state,
                    district=row.district,
                    chamber=row.chamber,
                    bio_embedding=row.bio_embedding,
                    voting_record_embedding=row.voting_record_embedding,
                    created_at=row.created_at,
                    updated_at=row.updated_at
                ),
                "similarity": float(row.similarity)
            }
            for row in rows
        ]

    async def update_voting_record_embedding(self, db: Session, legislator_id: str):
        """Update voting record embedding based on voting history"""
        # Get voting history
        votes_query = text("""
            SELECT v.vote, b.title, b.summary 
            FROM votes v
            JOIN bills b ON v.bill_id = b.id
            WHERE v.legislator_id = :legislator_id
            ORDER BY v.vote_date DESC
        """)
        
        result = db.execute(votes_query, {"legislator_id": legislator_id})
        votes = result.fetchall()
        
        if not votes:
            return
        
        # Create text representation of voting record
        voting_text = []
        for vote in votes:
            voting_text.append(f"Voted {vote.vote} on: {vote.title} - {vote.summary or ''}")
        
        combined_text = " ".join(voting_text)
        voting_embedding = await self.embedding_service.generate_embedding(combined_text)
        
        if voting_embedding:
            update_query = text("""
                UPDATE legislators 
                SET voting_record_embedding = :embedding, updated_at = NOW()
                WHERE id = :legislator_id
            """)
            
            db.execute(update_query, {
                "embedding": voting_embedding,
                "legislator_id": legislator_id
            })
            db.commit()