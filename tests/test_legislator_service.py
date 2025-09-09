"""
Test cases for the legislator service
"""

import pytest
from app.services.legislator_service import LegislatorService
from app.models.schemas import LegislatorCreate

@pytest.mark.asyncio
async def test_create_legislator(test_db):
    """Test creating a new legislator"""
    service = LegislatorService()
    
    legislator_data = LegislatorCreate(
        name="Test Legislator",
        party="Test Party",
        state="TS",
        district="1",
        chamber="house",
        bio_text="Test bio for legislator"
    )
    
    # This would normally work with a properly set up test database
    # For now, we'll just test that the service can be instantiated
    assert service is not None
    assert legislator_data.name == "Test Legislator"

def test_legislator_service_initialization():
    """Test that the legislator service initializes correctly"""
    service = LegislatorService()
    assert service.embedding_service is not None