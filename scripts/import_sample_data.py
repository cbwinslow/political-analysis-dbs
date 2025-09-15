#!/usr/bin/env python3
"""
Sample data import script for Political Analysis Database
"""

import asyncio
import sys
import os
from datetime import datetime, date, timedelta
import uuid
import random

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models.schemas import LegislatorCreate, BillCreate, VoteCreate
from app.services.legislator_service import LegislatorService
from app.services.bill_service import BillService
from app.services.vote_service import VoteService
from app.services.graph_service import GraphService

# Sample data
SAMPLE_LEGISLATORS = [
    {
        "name": "Alice Johnson",
        "party": "Democratic",
        "state": "CA",
        "district": "12",
        "chamber": "house",
        "bio_text": "Environmental advocate with focus on climate change legislation and renewable energy initiatives."
    },
    {
        "name": "Bob Smith",
        "party": "Republican", 
        "state": "TX",
        "district": "5",
        "chamber": "house",
        "bio_text": "Business leader focused on economic development and tax reform legislation."
    },
    {
        "name": "Carol Williams",
        "party": "Democratic",
        "state": "NY",
        "district": None,
        "chamber": "senate",
        "bio_text": "Healthcare policy expert advocating for universal healthcare and mental health reform."
    },
    {
        "name": "David Brown",
        "party": "Republican",
        "state": "FL",
        "district": None,
        "chamber": "senate", 
        "bio_text": "Former military officer focusing on defense and veterans affairs legislation."
    },
    {
        "name": "Emma Davis",
        "party": "Democratic",
        "state": "WA",
        "district": "7",
        "chamber": "house",
        "bio_text": "Technology policy specialist working on privacy rights and digital infrastructure."
    },
    {
        "name": "Frank Miller",
        "party": "Republican",
        "state": "OH",
        "district": "3",
        "chamber": "house",
        "bio_text": "Agriculture and rural development advocate representing farming communities."
    }
]

SAMPLE_BILLS = [
    {
        "title": "Clean Energy Transition Act",
        "summary": "Legislation to accelerate the transition to renewable energy sources and reduce carbon emissions.",
        "full_text": "A comprehensive bill establishing renewable energy standards, providing incentives for clean energy development, and setting carbon reduction targets for 2030.",
        "bill_number": "HR-2024-001",
        "status": "introduced",
        "introduced_date": date(2024, 1, 15)
    },
    {
        "title": "Healthcare Access Improvement Act",
        "summary": "Bill to expand healthcare access and reduce prescription drug costs.",
        "full_text": "This legislation aims to improve healthcare accessibility by expanding Medicare eligibility and implementing measures to control prescription drug pricing.",
        "bill_number": "S-2024-002",
        "status": "committee",
        "introduced_date": date(2024, 2, 1)
    },
    {
        "title": "Small Business Tax Relief Act",
        "summary": "Tax reform legislation to support small businesses and entrepreneurs.",
        "full_text": "Provides tax credits and deductions for small businesses, reduces regulatory burden, and establishes support programs for entrepreneurship.",
        "bill_number": "HR-2024-003",
        "status": "passed_house",
        "introduced_date": date(2024, 1, 20)
    },
    {
        "title": "Digital Privacy Protection Act",
        "summary": "Comprehensive data protection and privacy rights legislation.",
        "full_text": "Establishes consumer privacy rights, regulates data collection practices, and provides enforcement mechanisms for privacy violations.",
        "bill_number": "S-2024-004",
        "status": "introduced",
        "introduced_date": date(2024, 3, 1)
    },
    {
        "title": "Veterans Mental Health Support Act", 
        "summary": "Expanded mental health services for veterans and active military personnel.",
        "full_text": "Increases funding for veteran mental health programs, improves access to counseling services, and addresses suicide prevention.",
        "bill_number": "HR-2024-005",
        "status": "committee",
        "introduced_date": date(2024, 2, 15)
    }
]

class SampleDataImporter:
    def __init__(self):
        # Database connection
        self.engine = create_engine(
            os.getenv("DATABASE_URL", "postgresql://postgres:political123@localhost:5432/political_analysis")
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Services
        self.legislator_service = LegislatorService()
        self.bill_service = BillService()
        self.vote_service = VoteService()
        self.graph_service = GraphService()
        
        # Track created entities
        self.created_legislators = []
        self.created_bills = []

    async def import_legislators(self):
        """Import sample legislators"""
        print("📋 Importing legislators...")
        
        db = self.SessionLocal()
        try:
            for legislator_data in SAMPLE_LEGISLATORS:
                legislator = LegislatorCreate(**legislator_data)
                created = await self.legislator_service.create_legislator(db, legislator)
                self.created_legislators.append(created)
                
                # Also create in Neo4j
                await self.graph_service.create_legislator_node({
                    "id": str(created.id),
                    "name": created.name,
                    "party": created.party,
                    "state": created.state,
                    "district": created.district,
                    "chamber": created.chamber
                })
                
                print(f"   ✅ Created legislator: {created.name}")
        finally:
            db.close()

    async def import_bills(self):
        """Import sample bills"""
        print("📜 Importing bills...")
        
        db = self.SessionLocal()
        try:
            for bill_data in SAMPLE_BILLS:
                bill = BillCreate(**bill_data)
                created = await self.bill_service.create_bill(db, bill)
                self.created_bills.append(created)
                
                # Also create in Neo4j
                await self.graph_service.create_bill_node({
                    "id": str(created.id),
                    "title": created.title,
                    "bill_number": created.bill_number,
                    "status": created.status,
                    "summary": created.summary,
                    "introduced_date": created.introduced_date.isoformat() if created.introduced_date else None
                })
                
                print(f"   ✅ Created bill: {created.title}")
        finally:
            db.close()

    async def import_votes(self):
        """Import sample votes"""
        print("🗳️  Importing votes...")
        
        if not self.created_legislators or not self.created_bills:
            print("   ❌ No legislators or bills to create votes for")
            return
        
        db = self.SessionLocal()
        try:
            vote_count = 0
            for bill in self.created_bills:
                # Create votes for a random subset of legislators
                voting_legislators = random.sample(
                    self.created_legislators, 
                    k=random.randint(3, len(self.created_legislators))
                )
                
                for legislator in voting_legislators:
                    # Determine vote based on party and bill type (simplified logic)
                    vote_choice = self._determine_vote(legislator, bill)
                    vote_date = bill.created_at + timedelta(days=random.randint(1, 30))
                    
                    vote = VoteCreate(
                        legislator_id=legislator.id,
                        bill_id=bill.id,
                        vote=vote_choice,
                        vote_date=vote_date
                    )
                    
                    created_vote = await self.vote_service.create_vote(db, vote)
                    
                    # Also create in Neo4j
                    await self.graph_service.create_vote_relationship({
                        "legislator_id": str(legislator.id),
                        "bill_id": str(bill.id),
                        "vote": vote_choice,
                        "vote_date": vote_date.isoformat()
                    })
                    
                    vote_count += 1
            
            print(f"   ✅ Created {vote_count} votes")
        finally:
            db.close()

    def _determine_vote(self, legislator, bill):
        """Simplified vote determination based on party and bill content"""
        # Environmental bills
        if "clean energy" in bill.title.lower() or "climate" in bill.summary.lower():
            return "yes" if legislator.party == "Democratic" else random.choice(["no", "no", "abstain"])
        
        # Healthcare bills
        if "healthcare" in bill.title.lower():
            return "yes" if legislator.party == "Democratic" else random.choice(["no", "abstain"])
        
        # Business/tax bills
        if "tax" in bill.title.lower() and "small business" in bill.title.lower():
            return "yes" if legislator.party == "Republican" else random.choice(["yes", "no"])
        
        # Privacy/tech bills
        if "privacy" in bill.title.lower() or "digital" in bill.title.lower():
            return random.choice(["yes", "no", "abstain"])  # Bipartisan
        
        # Veterans bills
        if "veteran" in bill.title.lower():
            return "yes"  # Generally bipartisan support
        
        # Default random vote
        return random.choice(["yes", "no", "abstain"])

    async def create_sample_committees(self):
        """Create sample committees and memberships"""
        print("🏛️  Creating committees...")
        
        committees = [
            {"id": str(uuid.uuid4()), "committee_name": "Energy and Commerce", "chamber": "house"},
            {"id": str(uuid.uuid4()), "committee_name": "Healthcare", "chamber": "senate"},
            {"id": str(uuid.uuid4()), "committee_name": "Small Business", "chamber": "house"},
            {"id": str(uuid.uuid4()), "committee_name": "Veterans Affairs", "chamber": "joint"}
        ]
        
        for committee in committees:
            # Assign random legislators to committees
            committee_members = random.sample(
                self.created_legislators,
                k=random.randint(2, 4)
            )
            
            for legislator in committee_members:
                await self.graph_service.create_committee_relationship(
                    str(legislator.id),
                    {
                        **committee,
                        "role": "chair" if legislator == committee_members[0] else "member",
                        "start_date": "2024-01-01"
                    }
                )
        
        print(f"   ✅ Created {len(committees)} committees")

    async def run_import(self):
        """Run the complete import process"""
        print("🚀 Starting sample data import...")
        
        try:
            await self.import_legislators()
            await self.import_bills()
            await self.import_votes()
            await self.create_sample_committees()
            
            print("\n✅ Sample data import completed successfully!")
            print(f"   📊 Imported:")
            print(f"      • {len(self.created_legislators)} legislators")
            print(f"      • {len(self.created_bills)} bills")
            print(f"      • Various votes and relationships")
            
        except Exception as e:
            print(f"\n❌ Import failed: {e}")
            raise
        finally:
            # Close graph service connection
            self.graph_service.close()

if __name__ == "__main__":
    importer = SampleDataImporter()
    asyncio.run(importer.run_import())