from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from app.models.schemas import VoteCreate, VoteResponse, VotingPattern, PartyAlignment

class VoteService:
    def __init__(self):
        pass

    async def create_vote(self, db: Session, vote: VoteCreate) -> VoteResponse:
        """Create a new vote record"""
        vote_id = str(uuid.uuid4())
        
        query = text("""
            INSERT INTO votes (id, legislator_id, bill_id, vote, vote_date)
            VALUES (:id, :legislator_id, :bill_id, :vote, :vote_date)
            RETURNING *
        """)
        
        result = db.execute(query, {
            "id": vote_id,
            "legislator_id": vote.legislator_id,
            "bill_id": vote.bill_id,
            "vote": vote.vote,
            "vote_date": vote.vote_date or datetime.utcnow()
        })
        
        db.commit()
        row = result.fetchone()
        
        return VoteResponse(
            id=row.id,
            legislator_id=row.legislator_id,
            bill_id=row.bill_id,
            vote=row.vote,
            vote_date=row.vote_date,
            created_at=row.created_at
        )

    async def get_votes(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        legislator_id: Optional[str] = None,
        bill_id: Optional[str] = None
    ) -> List[VoteResponse]:
        """Get votes with optional filtering"""
        conditions = []
        params = {"skip": skip, "limit": limit}
        
        if legislator_id:
            conditions.append("legislator_id = :legislator_id")
            params["legislator_id"] = legislator_id
            
        if bill_id:
            conditions.append("bill_id = :bill_id")
            params["bill_id"] = bill_id
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = text(f"""
            SELECT * FROM votes 
            {where_clause}
            ORDER BY vote_date DESC
            OFFSET :skip LIMIT :limit
        """)
        
        result = db.execute(query, params)
        rows = result.fetchall()
        
        return [
            VoteResponse(
                id=row.id,
                legislator_id=row.legislator_id,
                bill_id=row.bill_id,
                vote=row.vote,
                vote_date=row.vote_date,
                created_at=row.created_at
            )
            for row in rows
        ]

    async def get_voting_patterns(self, db: Session, legislator_id: str) -> VotingPattern:
        """Analyze voting patterns for a specific legislator"""
        
        # Get basic voting stats
        basic_stats_query = text("""
            SELECT 
                COUNT(*) as total_votes,
                SUM(CASE WHEN vote = 'yes' THEN 1 ELSE 0 END) as yes_votes,
                SUM(CASE WHEN vote = 'no' THEN 1 ELSE 0 END) as no_votes,
                SUM(CASE WHEN vote = 'abstain' THEN 1 ELSE 0 END) as abstain_votes
            FROM votes 
            WHERE legislator_id = :legislator_id
        """)
        
        basic_result = db.execute(basic_stats_query, {"legislator_id": legislator_id})
        basic_row = basic_result.fetchone()
        
        if not basic_row or basic_row.total_votes == 0:
            return VotingPattern(
                party_alignment=0.0,
                issue_categories={},
                voting_frequency=0.0,
                controversial_votes=0
            )
        
        # Get party alignment
        party_alignment_query = text("""
            WITH legislator_party AS (
                SELECT party FROM legislators WHERE id = :legislator_id
            ),
            party_votes AS (
                SELECT 
                    v.bill_id,
                    v.vote as legislator_vote,
                    mode() WITHIN GROUP (ORDER BY v2.vote) as party_majority_vote
                FROM votes v
                JOIN legislators l ON v.legislator_id = l.id
                JOIN votes v2 ON v.bill_id = v2.bill_id
                JOIN legislators l2 ON v2.legislator_id = l2.id
                CROSS JOIN legislator_party lp
                WHERE v.legislator_id = :legislator_id
                AND l2.party = lp.party
                AND l2.id != :legislator_id
                GROUP BY v.bill_id, v.vote
            )
            SELECT 
                COUNT(*) as total_comparable_votes,
                SUM(CASE WHEN legislator_vote = party_majority_vote THEN 1 ELSE 0 END) as aligned_votes
            FROM party_votes
        """)
        
        party_result = db.execute(party_alignment_query, {"legislator_id": legislator_id})
        party_row = party_result.fetchone()
        
        party_alignment = 0.0
        if party_row and party_row.total_comparable_votes > 0:
            party_alignment = party_row.aligned_votes / party_row.total_comparable_votes
        
        # Get controversial votes (close votes)
        controversial_query = text("""
            WITH bill_vote_stats AS (
                SELECT 
                    bill_id,
                    SUM(CASE WHEN vote = 'yes' THEN 1 ELSE 0 END) as yes_count,
                    SUM(CASE WHEN vote = 'no' THEN 1 ELSE 0 END) as no_count,
                    COUNT(*) as total_count
                FROM votes
                GROUP BY bill_id
            ),
            controversial_bills AS (
                SELECT bill_id
                FROM bill_vote_stats
                WHERE ABS(yes_count - no_count) <= (total_count * 0.1)  -- Within 10%
                AND total_count >= 10  -- Minimum vote threshold
            )
            SELECT COUNT(*) as controversial_votes
            FROM votes v
            JOIN controversial_bills cb ON v.bill_id = cb.bill_id
            WHERE v.legislator_id = :legislator_id
        """)
        
        controversial_result = db.execute(controversial_query, {"legislator_id": legislator_id})
        controversial_row = controversial_result.fetchone()
        
        controversial_votes = controversial_row.controversial_votes if controversial_row else 0
        
        # Calculate voting frequency (votes per day over active period)
        frequency_query = text("""
            SELECT 
                EXTRACT(DAYS FROM (MAX(vote_date) - MIN(vote_date))) as active_days,
                COUNT(*) as total_votes
            FROM votes 
            WHERE legislator_id = :legislator_id
            AND vote_date IS NOT NULL
        """)
        
        frequency_result = db.execute(frequency_query, {"legislator_id": legislator_id})
        frequency_row = frequency_result.fetchone()
        
        voting_frequency = 0.0
        if frequency_row and frequency_row.active_days and frequency_row.active_days > 0:
            voting_frequency = frequency_row.total_votes / frequency_row.active_days
        
        # TODO: Implement issue categories analysis based on bill content embeddings
        issue_categories = {
            "healthcare": 0,
            "economy": 0,
            "environment": 0,
            "education": 0,
            "defense": 0
        }
        
        return VotingPattern(
            party_alignment=round(party_alignment, 3),
            issue_categories=issue_categories,
            voting_frequency=round(voting_frequency, 3),
            controversial_votes=controversial_votes
        )

    async def get_party_alignment(self, db: Session) -> PartyAlignment:
        """Analyze overall party alignment in voting"""
        
        # Get party-to-party alignment rates
        alignment_query = text("""
            WITH party_bill_votes AS (
                SELECT 
                    l.party,
                    v.bill_id,
                    mode() WITHIN GROUP (ORDER BY v.vote) as majority_vote,
                    COUNT(*) as party_vote_count
                FROM votes v
                JOIN legislators l ON v.legislator_id = l.id
                WHERE l.party IS NOT NULL
                GROUP BY l.party, v.bill_id
                HAVING COUNT(*) >= 3  -- Minimum votes per party per bill
            ),
            party_alignments AS (
                SELECT 
                    p1.party as party1,
                    p2.party as party2,
                    COUNT(*) as total_bills,
                    SUM(CASE WHEN p1.majority_vote = p2.majority_vote THEN 1 ELSE 0 END) as aligned_bills
                FROM party_bill_votes p1
                JOIN party_bill_votes p2 ON p1.bill_id = p2.bill_id
                WHERE p1.party < p2.party  -- Avoid duplicates
                GROUP BY p1.party, p2.party
                HAVING COUNT(*) >= 10  -- Minimum bills for comparison
            )
            SELECT 
                party1,
                party2,
                total_bills,
                aligned_bills,
                ROUND(CAST(aligned_bills AS FLOAT) / total_bills, 3) as alignment_rate
            FROM party_alignments
            ORDER BY alignment_rate DESC
        """)
        
        alignment_result = db.execute(alignment_query)
        alignment_rows = alignment_result.fetchall()
        
        party_pairs = [
            {
                "party1": row.party1,
                "party2": row.party2,
                "total_bills": row.total_bills,
                "aligned_bills": row.aligned_bills,
                "alignment_rate": row.alignment_rate
            }
            for row in alignment_rows
        ]
        
        # Calculate overall polarization (average distance from 0.5)
        if party_pairs:
            polarization_scores = [abs(pair["alignment_rate"] - 0.5) for pair in party_pairs]
            overall_polarization = sum(polarization_scores) / len(polarization_scores) * 2  # Scale to 0-1
        else:
            overall_polarization = 0.0
        
        # TODO: Implement issue-based alignment analysis
        issue_based_alignment = {
            "healthcare": {},
            "economy": {},
            "environment": {},
            "education": {},
            "defense": {}
        }
        
        return PartyAlignment(
            party_pairs=party_pairs,
            overall_polarization=round(overall_polarization, 3),
            issue_based_alignment=issue_based_alignment
        )

    async def get_vote_similarity_matrix(self, db: Session, legislator_ids: List[str]) -> Dict[str, Any]:
        """Calculate vote similarity matrix between legislators"""
        
        if len(legislator_ids) < 2:
            return {"matrix": {}, "legislators": []}
        
        # Get all vote pairs between legislators
        similarity_query = text("""
            SELECT 
                v1.legislator_id as legislator1,
                v2.legislator_id as legislator2,
                COUNT(*) as total_shared_votes,
                SUM(CASE WHEN v1.vote = v2.vote THEN 1 ELSE 0 END) as matching_votes
            FROM votes v1
            JOIN votes v2 ON v1.bill_id = v2.bill_id
            WHERE v1.legislator_id = ANY(:legislator_ids)
            AND v2.legislator_id = ANY(:legislator_ids)
            AND v1.legislator_id < v2.legislator_id  -- Avoid self-comparison and duplicates
            GROUP BY v1.legislator_id, v2.legislator_id
            HAVING COUNT(*) >= 5  -- Minimum shared votes
        """)
        
        result = db.execute(similarity_query, {"legislator_ids": legislator_ids})
        rows = result.fetchall()
        
        # Build similarity matrix
        matrix = {}
        for legislator_id in legislator_ids:
            matrix[legislator_id] = {}
            for other_id in legislator_ids:
                if legislator_id == other_id:
                    matrix[legislator_id][other_id] = 1.0
                else:
                    matrix[legislator_id][other_id] = 0.0
        
        for row in rows:
            leg1, leg2 = row.legislator1, row.legislator2
            similarity = row.matching_votes / row.total_shared_votes if row.total_shared_votes > 0 else 0.0
            
            matrix[leg1][leg2] = similarity
            matrix[leg2][leg1] = similarity  # Symmetric matrix
        
        # Get legislator names for reference
        names_query = text("""
            SELECT id, name, party 
            FROM legislators 
            WHERE id = ANY(:legislator_ids)
        """)
        
        names_result = db.execute(names_query, {"legislator_ids": legislator_ids})
        names_rows = names_result.fetchall()
        
        legislators = [
            {
                "id": row.id,
                "name": row.name,
                "party": row.party
            }
            for row in names_rows
        ]
        
        return {
            "matrix": matrix,
            "legislators": legislators
        }