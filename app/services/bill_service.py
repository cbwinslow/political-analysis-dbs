from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from app.models.schemas import BillCreate, BillResponse, BillSupport
from app.services.embedding_service import EmbeddingService

class BillService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    async def create_bill(self, db: Session, bill: BillCreate) -> BillResponse:
        """Create a new bill with content embedding"""
        bill_id = str(uuid.uuid4())
        
        # Generate embedding for bill content
        content_text = f"{bill.title} {bill.summary or ''} {bill.full_text or ''}".strip()
        content_embedding = None
        if content_text:
            content_embedding = await self.embedding_service.generate_embedding(content_text)
        
        query = text("""
            INSERT INTO bills (id, title, summary, full_text, bill_number, status, introduced_date, content_embedding)
            VALUES (:id, :title, :summary, :full_text, :bill_number, :status, :introduced_date, :content_embedding)
            RETURNING *
        """)
        
        result = db.execute(query, {
            "id": bill_id,
            "title": bill.title,
            "summary": bill.summary,
            "full_text": bill.full_text,
            "bill_number": bill.bill_number,
            "status": bill.status,
            "introduced_date": bill.introduced_date,
            "content_embedding": content_embedding
        })
        
        db.commit()
        row = result.fetchone()
        
        return BillResponse(
            id=row.id,
            title=row.title,
            summary=row.summary,
            full_text=row.full_text,
            bill_number=row.bill_number,
            status=row.status,
            introduced_date=row.introduced_date,
            content_embedding=row.content_embedding,
            created_at=row.created_at,
            updated_at=row.updated_at
        )

    async def get_bills(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[BillResponse]:
        """Get bills with optional status filtering"""
        conditions = []
        params = {"skip": skip, "limit": limit}
        
        if status:
            conditions.append("status = :status")
            params["status"] = status
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = text(f"""
            SELECT * FROM bills 
            {where_clause}
            ORDER BY created_at DESC
            OFFSET :skip LIMIT :limit
        """)
        
        result = db.execute(query, params)
        rows = result.fetchall()
        
        return [
            BillResponse(
                id=row.id,
                title=row.title,
                summary=row.summary,
                full_text=row.full_text,
                bill_number=row.bill_number,
                status=row.status,
                introduced_date=row.introduced_date,
                content_embedding=row.content_embedding,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            for row in rows
        ]

    async def get_bill(self, db: Session, bill_id: str) -> Optional[BillResponse]:
        """Get a single bill by ID"""
        query = text("SELECT * FROM bills WHERE id = :id")
        result = db.execute(query, {"id": bill_id})
        row = result.fetchone()
        
        if not row:
            return None
            
        return BillResponse(
            id=row.id,
            title=row.title,
            summary=row.summary,
            full_text=row.full_text,
            bill_number=row.bill_number,
            status=row.status,
            introduced_date=row.introduced_date,
            content_embedding=row.content_embedding,
            created_at=row.created_at,
            updated_at=row.updated_at
        )

    async def find_similar_bills(
        self, 
        db: Session, 
        query_text: str, 
        threshold: float = 0.8, 
        limit: int = 10
    ) -> List[dict]:
        """Find bills similar to the query text"""
        query_embedding = await self.embedding_service.generate_embedding(query_text)
        
        if not query_embedding:
            return []
        
        sql_query = text("""
            SELECT b.*, 1 - (b.content_embedding <=> :query_embedding) AS similarity
            FROM bills b
            WHERE b.content_embedding IS NOT NULL
            AND 1 - (b.content_embedding <=> :query_embedding) > :threshold
            ORDER BY b.content_embedding <=> :query_embedding
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
                "bill": BillResponse(
                    id=row.id,
                    title=row.title,
                    summary=row.summary,
                    full_text=row.full_text,
                    bill_number=row.bill_number,
                    status=row.status,
                    introduced_date=row.introduced_date,
                    content_embedding=row.content_embedding,
                    created_at=row.created_at,
                    updated_at=row.updated_at
                ),
                "similarity": float(row.similarity)
            }
            for row in rows
        ]

    async def get_bill_support(self, db: Session, bill_id: str) -> BillSupport:
        """Get support statistics for a bill"""
        query = text("""
            SELECT 
                COUNT(*) as total_votes,
                SUM(CASE WHEN vote = 'yes' THEN 1 ELSE 0 END) as yes_votes,
                SUM(CASE WHEN vote = 'no' THEN 1 ELSE 0 END) as no_votes,
                SUM(CASE WHEN vote = 'abstain' THEN 1 ELSE 0 END) as abstain_votes,
                SUM(CASE WHEN vote = 'present' THEN 1 ELSE 0 END) as present_votes
            FROM votes 
            WHERE bill_id = :bill_id
        """)
        
        result = db.execute(query, {"bill_id": bill_id})
        row = result.fetchone()
        
        if not row or row.total_votes == 0:
            return BillSupport(
                total_votes=0,
                yes_votes=0,
                no_votes=0,
                abstain_votes=0,
                present_votes=0,
                support_percentage=0.0,
                party_breakdown={}
            )
        
        # Get party breakdown
        party_query = text("""
            SELECT 
                l.party,
                v.vote,
                COUNT(*) as count
            FROM votes v
            JOIN legislators l ON v.legislator_id = l.id
            WHERE v.bill_id = :bill_id AND l.party IS NOT NULL
            GROUP BY l.party, v.vote
            ORDER BY l.party, v.vote
        """)
        
        party_result = db.execute(party_query, {"bill_id": bill_id})
        party_rows = party_result.fetchall()
        
        party_breakdown = {}
        for party_row in party_rows:
            party = party_row.party
            if party not in party_breakdown:
                party_breakdown[party] = {
                    "yes": 0, "no": 0, "abstain": 0, "present": 0
                }
            party_breakdown[party][party_row.vote] = party_row.count
        
        support_percentage = (row.yes_votes / row.total_votes) * 100 if row.total_votes > 0 else 0
        
        return BillSupport(
            total_votes=row.total_votes,
            yes_votes=row.yes_votes,
            no_votes=row.no_votes,
            abstain_votes=row.abstain_votes,
            present_votes=row.present_votes,
            support_percentage=round(support_percentage, 2),
            party_breakdown=party_breakdown
        )

    async def get_bills_by_topic(self, db: Session, topic: str, limit: int = 20) -> List[dict]:
        """Find bills related to a specific topic using embedding similarity"""
        topic_embedding = await self.embedding_service.generate_embedding(topic)
        
        if not topic_embedding:
            return []
        
        query = text("""
            SELECT b.*, 1 - (b.content_embedding <=> :topic_embedding) AS relevance
            FROM bills b
            WHERE b.content_embedding IS NOT NULL
            ORDER BY b.content_embedding <=> :topic_embedding
            LIMIT :limit
        """)
        
        result = db.execute(query, {
            "topic_embedding": topic_embedding,
            "limit": limit
        })
        
        rows = result.fetchall()
        
        return [
            {
                "bill": BillResponse(
                    id=row.id,
                    title=row.title,
                    summary=row.summary,
                    full_text=row.full_text,
                    bill_number=row.bill_number,
                    status=row.status,
                    introduced_date=row.introduced_date,
                    content_embedding=row.content_embedding,
                    created_at=row.created_at,
                    updated_at=row.updated_at
                ),
                "relevance": float(row.relevance)
            }
            for row in rows
        ]

    async def get_trending_bills(self, db: Session, days: int = 30, limit: int = 10) -> List[dict]:
        """Get bills with most activity in recent days"""
        query = text("""
            SELECT 
                b.*,
                COUNT(v.id) as recent_votes,
                COUNT(DISTINCT v.legislator_id) as unique_voters
            FROM bills b
            LEFT JOIN votes v ON b.id = v.bill_id 
                AND v.vote_date >= NOW() - INTERVAL '%s days'
            GROUP BY b.id
            HAVING COUNT(v.id) > 0
            ORDER BY recent_votes DESC, unique_voters DESC
            LIMIT :limit
        """ % days)
        
        result = db.execute(query, {"limit": limit})
        rows = result.fetchall()
        
        return [
            {
                "bill": BillResponse(
                    id=row.id,
                    title=row.title,
                    summary=row.summary,
                    full_text=row.full_text,
                    bill_number=row.bill_number,
                    status=row.status,
                    introduced_date=row.introduced_date,
                    content_embedding=row.content_embedding,
                    created_at=row.created_at,
                    updated_at=row.updated_at
                ),
                "recent_activity": {
                    "recent_votes": row.recent_votes,
                    "unique_voters": row.unique_voters
                }
            }
            for row in rows
        ]