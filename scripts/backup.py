#!/usr/bin/env python3
"""
Comprehensive backup script for Political Analysis Database
"""

import os
import sys
import subprocess
import json
import tarfile
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupManager:
    def __init__(self, backup_dir: str = "backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_backup_dir = self.backup_dir / self.timestamp
        self.current_backup_dir.mkdir(exist_ok=True)
        
    def backup_postgresql(self) -> bool:
        """Backup PostgreSQL databases"""
        logger.info("Starting PostgreSQL backup...")
        
        try:
            # Backup main database
            main_db_backup = self.current_backup_dir / "postgresql_main.sql"
            result = subprocess.run([
                'docker', 'compose', 'exec', '-T', 'postgres',
                'pg_dump', '-U', 'postgres', '-d', 'political_analysis'
            ], stdout=open(main_db_backup, 'w'), stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                logger.error(f"PostgreSQL main database backup failed: {result.stderr}")
                return False
            
            # Backup Airflow database
            airflow_db_backup = self.current_backup_dir / "postgresql_airflow.sql"
            result = subprocess.run([
                'docker', 'compose', 'exec', '-T', 'postgres',
                'pg_dump', '-U', 'airflow', '-d', 'airflow_db'
            ], stdout=open(airflow_db_backup, 'w'), stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                logger.warning(f"PostgreSQL Airflow database backup failed: {result.stderr}")
            
            # Create a full cluster backup as well
            cluster_backup = self.current_backup_dir / "postgresql_cluster.sql"
            result = subprocess.run([
                'docker', 'compose', 'exec', '-T', 'postgres',
                'pg_dumpall', '-U', 'postgres'
            ], stdout=open(cluster_backup, 'w'), stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                logger.warning(f"PostgreSQL cluster backup failed: {result.stderr}")
            
            logger.info("PostgreSQL backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL backup error: {e}")
            return False
    
    def backup_neo4j(self) -> bool:
        """Backup Neo4j database"""
        logger.info("Starting Neo4j backup...")
        
        try:
            # Stop Neo4j for consistent backup
            subprocess.run(['docker', 'compose', 'stop', 'neo4j'], check=True)
            
            # Create backup using neo4j-admin dump
            neo4j_backup = self.current_backup_dir / f"neo4j_backup_{self.timestamp}.dump"
            result = subprocess.run([
                'docker', 'compose', 'run', '--rm', '--volumes-from', 'political-neo4j',
                'neo4j:5.15',
                'neo4j-admin', 'database', 'dump', 'neo4j', '--to-path=/var/lib/neo4j/backups'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Neo4j backup failed: {result.stderr}")
                # Restart Neo4j even if backup failed
                subprocess.run(['docker', 'compose', 'start', 'neo4j'])
                return False
            
            # Copy backup file from container
            subprocess.run([
                'docker', 'cp', 
                f'political-neo4j:/var/lib/neo4j/backups/neo4j.dump',
                str(neo4j_backup)
            ], check=True)
            
            # Restart Neo4j
            subprocess.run(['docker', 'compose', 'start', 'neo4j'], check=True)
            
            logger.info("Neo4j backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Neo4j backup error: {e}")
            # Ensure Neo4j is restarted
            subprocess.run(['docker', 'compose', 'start', 'neo4j'])
            return False
    
    def backup_redis(self) -> bool:
        """Backup Redis data"""
        logger.info("Starting Redis backup...")
        
        try:
            # Trigger Redis save
            result = subprocess.run([
                'docker', 'compose', 'exec', '-T', 'redis',
                'redis-cli', '-a', 'political123', 'BGSAVE'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Redis BGSAVE failed: {result.stderr}")
                return False
            
            # Wait for save to complete
            import time
            time.sleep(5)
            
            # Copy RDB file
            redis_backup = self.current_backup_dir / "redis_dump.rdb"
            result = subprocess.run([
                'docker', 'cp',
                'political-redis:/data/dump.rdb',
                str(redis_backup)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Redis file copy failed: {result.stderr}")
                return False
            
            logger.info("Redis backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Redis backup error: {e}")
            return False
    
    def backup_qdrant(self) -> bool:
        """Backup Qdrant vector database"""
        logger.info("Starting Qdrant backup...")
        
        try:
            # Create a snapshot via API
            import requests
            response = requests.post('http://localhost:6333/snapshots', timeout=30)
            
            if response.status_code == 200:
                snapshot_info = response.json()
                snapshot_name = snapshot_info.get('result', {}).get('name', 'snapshot')
            else:
                logger.warning("Failed to create Qdrant snapshot via API, using file copy")
                snapshot_name = None
            
            # Copy Qdrant data directory
            qdrant_backup_dir = self.current_backup_dir / "qdrant_data"
            result = subprocess.run([
                'docker', 'cp',
                'political-qdrant:/qdrant/storage',
                str(qdrant_backup_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Qdrant backup failed: {result.stderr}")
                return False
            
            logger.info("Qdrant backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Qdrant backup error: {e}")
            return False
    
    def backup_elasticsearch(self) -> bool:
        """Backup Elasticsearch data"""
        logger.info("Starting Elasticsearch backup...")
        
        try:
            # Create a snapshot repository first
            import requests
            
            # Register snapshot repository
            repo_config = {
                "type": "fs",
                "settings": {
                    "location": "/usr/share/elasticsearch/backups",
                    "compress": True
                }
            }
            
            response = requests.put(
                'http://localhost:9200/_snapshot/backup_repo',
                json=repo_config,
                timeout=30
            )
            
            if response.status_code not in [200, 201]:
                logger.warning("Failed to register Elasticsearch snapshot repository")
            
            # Create snapshot
            snapshot_name = f"snapshot_{self.timestamp}"
            response = requests.put(
                f'http://localhost:9200/_snapshot/backup_repo/{snapshot_name}',
                json={
                    "indices": "*",
                    "ignore_unavailable": True,
                    "include_global_state": True
                },
                timeout=300
            )
            
            if response.status_code == 200:
                logger.info("Elasticsearch snapshot created successfully")
                
                # Copy snapshot files
                es_backup_dir = self.current_backup_dir / "elasticsearch_data"
                result = subprocess.run([
                    'docker', 'cp',
                    'political-elasticsearch:/usr/share/elasticsearch/data',
                    str(es_backup_dir)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Elasticsearch file copy failed: {result.stderr}")
                
                return True
            else:
                logger.error(f"Elasticsearch snapshot failed: {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Elasticsearch backup error: {e}")
            return False
    
    def backup_configurations(self) -> bool:
        """Backup configuration files"""
        logger.info("Starting configuration backup...")
        
        try:
            config_backup = self.current_backup_dir / "configurations.tar.gz"
            
            # List of configuration files and directories to backup
            config_items = [
                '.env',
                'docker-compose.yml',
                'prometheus/',
                'grafana/',
                'localai/',
                'supabase/',
                'postgres/init/',
                'scripts/',
                'app/',
                'README.md',
                'PROXMOX_DEPLOYMENT.md'
            ]
            
            # Create tar archive
            with tarfile.open(config_backup, 'w:gz') as tar:
                for item in config_items:
                    if os.path.exists(item):
                        tar.add(item, arcname=item)
                    else:
                        logger.warning(f"Configuration item not found: {item}")
            
            logger.info("Configuration backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration backup error: {e}")
            return False
    
    def backup_volumes(self) -> bool:
        """Backup Docker volumes"""
        logger.info("Starting Docker volumes backup...")
        
        try:
            volumes_backup_dir = self.current_backup_dir / "volumes"
            volumes_backup_dir.mkdir(exist_ok=True)
            
            # List of important volumes to backup
            volumes = [
                'grafana_data',
                'prometheus_data',
                'minio_data',
                'jupyter_data',
                'n8n_data',
                'flowise_data',
                'open_webui_data'
            ]
            
            for volume in volumes:
                try:
                    volume_backup = volumes_backup_dir / f"{volume}.tar.gz"
                    result = subprocess.run([
                        'docker', 'run', '--rm',
                        '-v', f'{volume}:/source',
                        '-v', f'{os.getcwd()}:/backup',
                        'alpine',
                        'tar', 'czf', f'/backup/{volume_backup}', '-C', '/source', '.'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"Volume {volume} backed up successfully")
                    else:
                        logger.warning(f"Volume {volume} backup failed: {result.stderr}")
                        
                except Exception as e:
                    logger.warning(f"Error backing up volume {volume}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Volumes backup error: {e}")
            return False
    
    def create_manifest(self, backup_results: Dict[str, bool]) -> bool:
        """Create backup manifest file"""
        try:
            manifest = {
                'timestamp': self.timestamp,
                'date': datetime.now().isoformat(),
                'backup_dir': str(self.current_backup_dir),
                'results': backup_results,
                'files': []
            }
            
            # List all backup files
            for file_path in self.current_backup_dir.rglob('*'):
                if file_path.is_file():
                    manifest['files'].append({
                        'path': str(file_path.relative_to(self.current_backup_dir)),
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            
            # Calculate total size
            total_size = sum(f['size'] for f in manifest['files'])
            manifest['total_size'] = total_size
            manifest['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            # Save manifest
            manifest_file = self.current_backup_dir / "backup_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Backup manifest created: {manifest['total_size_mb']}MB total")
            return True
            
        except Exception as e:
            logger.error(f"Manifest creation error: {e}")
            return False
    
    def cleanup_old_backups(self, retention_days: int = 7) -> None:
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            deleted_count = 0
            
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir() and backup_dir.name != self.timestamp:
                    try:
                        # Parse timestamp from directory name
                        backup_date = datetime.strptime(backup_dir.name, "%Y%m%d_%H%M%S")
                        if backup_date < cutoff_date:
                            shutil.rmtree(backup_dir)
                            deleted_count += 1
                            logger.info(f"Deleted old backup: {backup_dir.name}")
                    except ValueError:
                        # Skip directories that don't match timestamp format
                        continue
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backups")
            else:
                logger.info("No old backups to clean up")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def run_full_backup(self, retention_days: int = 7) -> bool:
        """Run complete backup process"""
        logger.info(f"Starting full backup: {self.timestamp}")
        
        backup_results = {}
        
        # Run all backup operations
        backup_results['postgresql'] = self.backup_postgresql()
        backup_results['neo4j'] = self.backup_neo4j()
        backup_results['redis'] = self.backup_redis()
        backup_results['qdrant'] = self.backup_qdrant()
        backup_results['elasticsearch'] = self.backup_elasticsearch()
        backup_results['configurations'] = self.backup_configurations()
        backup_results['volumes'] = self.backup_volumes()
        
        # Create manifest
        backup_results['manifest'] = self.create_manifest(backup_results)
        
        # Summary
        successful_backups = sum(1 for success in backup_results.values() if success)
        total_backups = len(backup_results)
        
        logger.info(f"Backup completed: {successful_backups}/{total_backups} successful")
        
        if successful_backups == total_backups:
            logger.info("✅ Full backup completed successfully!")
        else:
            logger.warning("⚠️ Some backup operations failed")
        
        # Cleanup old backups
        self.cleanup_old_backups(retention_days)
        
        return successful_backups >= (total_backups - 2)  # Allow 2 failures

def main():
    """Main backup function"""
    parser = argparse.ArgumentParser(description='Political Analysis Database Backup Tool')
    parser.add_argument('--backup-dir', default='backup', help='Backup directory')
    parser.add_argument('--retention-days', type=int, default=7, help='Backup retention period in days')
    parser.add_argument('--component', choices=['postgresql', 'neo4j', 'redis', 'qdrant', 'elasticsearch', 'config', 'volumes'], 
                       help='Backup only specific component')
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    backup_manager = BackupManager(args.backup_dir)
    
    if args.component:
        # Backup specific component
        logger.info(f"Starting {args.component} backup...")
        
        if args.component == 'postgresql':
            success = backup_manager.backup_postgresql()
        elif args.component == 'neo4j':
            success = backup_manager.backup_neo4j()
        elif args.component == 'redis':
            success = backup_manager.backup_redis()
        elif args.component == 'qdrant':
            success = backup_manager.backup_qdrant()
        elif args.component == 'elasticsearch':
            success = backup_manager.backup_elasticsearch()
        elif args.component == 'config':
            success = backup_manager.backup_configurations()
        elif args.component == 'volumes':
            success = backup_manager.backup_volumes()
        else:
            logger.error(f"Unknown component: {args.component}")
            sys.exit(1)
        
        sys.exit(0 if success else 1)
    else:
        # Full backup
        success = backup_manager.run_full_backup(args.retention_days)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()