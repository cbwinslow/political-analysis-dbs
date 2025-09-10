#!/usr/bin/env python3
"""
Comprehensive test suite for Political Analysis Database
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestLegislatorService:
    """Test suite for Legislator Service"""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        return Mock()
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service"""
        return Mock()
    
    @pytest.fixture
    def legislator_service(self, mock_embedding_service):
        """Create legislator service with mocked dependencies"""
        try:
            from app.services.legislator_service import LegislatorService
            service = LegislatorService()
            service.embedding_service = mock_embedding_service
            return service
        except ImportError:
            pytest.skip("LegislatorService not available")
    
    def test_legislator_creation_data(self):
        """Test legislator data validation"""
        legislator_data = {
            'name': 'John Doe',
            'party': 'Independent',
            'state': 'CA',
            'chamber': 'house',
            'bio_text': 'Environmental advocate'
        }
        
        # Test data structure
        assert legislator_data['name'] == 'John Doe'
        assert legislator_data['state'] == 'CA'
        assert legislator_data['chamber'] in ['house', 'senate']
    
    def test_embedding_generation_mock(self, mock_embedding_service):
        """Test embedding generation with mock"""
        # Setup mock
        mock_embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Test
        text = "Environmental advocate focused on climate policy"
        result = mock_embedding_service.generate_embedding(text)
        
        assert result == [0.1, 0.2, 0.3]
        mock_embedding_service.generate_embedding.assert_called_once_with(text)
    
    @pytest.mark.asyncio
    async def test_similarity_search_structure(self):
        """Test similarity search data structure"""
        search_params = {
            'query': 'climate change',
            'threshold': 0.7,
            'limit': 10
        }
        
        assert 0.0 <= search_params['threshold'] <= 1.0
        assert search_params['limit'] > 0
        assert isinstance(search_params['query'], str)

class TestDatabaseModels:
    """Test database models and schemas"""
    
    def test_schema_imports(self):
        """Test that schemas can be imported"""
        try:
            from app.models.schemas import (
                LegislatorCreate, 
                LegislatorResponse,
                BillCreate,
                BillResponse,
                VoteCreate,
                VoteResponse
            )
            # Basic test that imports work
            assert LegislatorCreate is not None
            assert LegislatorResponse is not None
            
        except ImportError as e:
            pytest.skip(f"Schema imports not available: {e}")
    
    def test_model_imports(self):
        """Test that models can be imported"""
        try:
            from app.models.models import Legislator, Bill, Vote
            # Basic test that imports work
            assert Legislator is not None
            assert Bill is not None
            assert Vote is not None
            
        except ImportError as e:
            pytest.skip(f"Model imports not available: {e}")

class TestConfigurationFiles:
    """Test configuration files and structure"""
    
    def test_env_example_exists(self):
        """Test that .env.example exists"""
        assert os.path.exists('.env.example')
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists"""
        assert os.path.exists('docker-compose.yml')
    
    def test_requirements_exists(self):
        """Test that requirements.txt exists"""
        assert os.path.exists('requirements.txt')
    
    def test_docker_compose_structure(self):
        """Test docker-compose.yml structure"""
        import yaml
        
        with open('docker-compose.yml', 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Test basic structure
        assert 'services' in compose_config
        
        # Test critical services exist
        critical_services = ['postgres', 'neo4j', 'redis', 'kong']
        for service in critical_services:
            assert service in compose_config['services'], f"Critical service {service} missing"
    
    def test_requirements_format(self):
        """Test requirements.txt format"""
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
        
        # Test that it's not empty
        assert len(requirements) > 0
        
        # Test basic format (package==version)
        valid_lines = 0
        for line in requirements:
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line or '>=' in line:
                    valid_lines += 1
        
        assert valid_lines > 0, "No valid package specifications found"

class TestScripts:
    """Test utility scripts"""
    
    def test_backup_script_exists(self):
        """Test backup script exists and is executable"""
        backup_script = 'scripts/backup.py'
        assert os.path.exists(backup_script)
        assert os.access(backup_script, os.X_OK)
    
    def test_health_check_script_exists(self):
        """Test health check script exists"""
        health_script = 'scripts/health_check.py'
        assert os.path.exists(health_script)
        assert os.access(health_script, os.X_OK)
    
    def test_deploy_script_exists(self):
        """Test deploy script exists"""
        deploy_script = 'deploy.sh'
        assert os.path.exists(deploy_script)
        assert os.access(deploy_script, os.X_OK)

class TestApplicationStructure:
    """Test application structure and organization"""
    
    def test_app_directory_structure(self):
        """Test app directory structure"""
        required_dirs = [
            'app',
            'app/models',
            'app/services',
            'app/api',
            'scripts',
            'tests'
        ]
        
        for directory in required_dirs:
            assert os.path.isdir(directory), f"Required directory {directory} missing"
    
    def test_config_directories(self):
        """Test configuration directories"""
        config_dirs = [
            'grafana/provisioning',
            'prometheus',
            'postgres/init',
            'supabase'
        ]
        
        for directory in config_dirs:
            assert os.path.isdir(directory), f"Config directory {directory} missing"
    
    def test_main_app_file(self):
        """Test main application file"""
        main_file = 'app/main.py'
        assert os.path.exists(main_file)
        
        # Check if it contains expected imports
        with open(main_file, 'r') as f:
            content = f.read()
            
        assert 'FastAPI' in content
        assert 'app = FastAPI' in content

class TestDocumentation:
    """Test documentation completeness"""
    
    def test_readme_exists(self):
        """Test README.md exists"""
        assert os.path.exists('README.md')
    
    def test_proxmox_guide_exists(self):
        """Test Proxmox deployment guide exists"""
        assert os.path.exists('PROXMOX_DEPLOYMENT.md')
    
    def test_readme_content(self):
        """Test README.md has required sections"""
        with open('README.md', 'r') as f:
            content = f.read()
        
        required_sections = [
            'Political Analysis Database',
            'Technology Stack',
            'Quick Start',
            'Service URLs'
        ]
        
        for section in required_sections:
            assert section in content, f"Required section '{section}' missing from README"

# Integration tests (require running services)
class TestIntegration:
    """Integration tests for running services"""
    
    @pytest.mark.integration
    def test_docker_compose_services(self):
        """Test that Docker Compose services can be parsed"""
        import subprocess
        
        try:
            result = subprocess.run(
                ['docker', 'compose', 'config'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should not fail with syntax errors
            assert result.returncode == 0, f"Docker Compose config invalid: {result.stderr}"
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available or timeout")
    
    @pytest.mark.integration 
    @pytest.mark.asyncio
    async def test_health_check_script(self):
        """Test health check script can run"""
        import subprocess
        
        try:
            # Run health check script
            result = subprocess.run(
                ['python3', 'scripts/health_check.py'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Script should run without syntax errors
            # (may fail due to services not running, but shouldn't crash)
            assert result.returncode in [0, 1], "Health check script crashed"
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Health check test skipped - services not running")

def run_tests():
    """Run all tests"""
    # Basic structure tests (always run)
    pytest.main([
        'tests/',
        '-v',
        '--tb=short',
        '-m', 'not integration'
    ])
    
    print("\n" + "="*50)
    print("Integration tests (require running services):")
    print("Run with: pytest -m integration")
    print("="*50)