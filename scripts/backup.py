#!/usr/bin/env python3
"""
Backup script for Political Analysis Database
"""

import subprocess
import os
import sys
from datetime import datetime
import json
import shutil
import tarfile

class DatabaseBackup:
    def __init__(self):
        self.backup_dir = os.getenv('BACKUP_DIR', './backups')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_path = os.path.join(self.backup_dir, f'backup_{self.timestamp}')
        
        # Ensure backup directory exists
        os.makedirs(self.backup_path, exist_ok=True)

    def backup_postgresql(self):
        """Backup PostgreSQL database"""
        print("📦 Backing up PostgreSQL database...")
        
        try:
            backup_file = os.path.join(self.backup_path, 'postgresql_backup.sql')
            
            cmd = [
                'docker', 'exec', 'political-postgres',
                'pg_dump', '-U', 'postgres', '-d', 'political_analysis'
            ]
            
            with open(backup_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            
            print(f"   ✅ PostgreSQL backup saved to {backup_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ PostgreSQL backup failed: {e}")
            return False

    def backup_neo4j(self):
        """Backup Neo4j database"""
        print("📦 Backing up Neo4j database...")
        
        try:
            neo4j_backup_dir = os.path.join(self.backup_path, 'neo4j')
            os.makedirs(neo4j_backup_dir, exist_ok=True)
            
            # Export Neo4j data using APOC
            export_file = os.path.join(neo4j_backup_dir, 'neo4j_export.cypher')
            
            cmd = [
                'docker', 'exec', 'political-neo4j',
                'cypher-shell', '-u', 'neo4j', '-p', 'political123',
                'CALL apoc.export.cypher.all("' + export_file + '", {useOptimizations: {type: "UNWIND_BATCH", unwindBatchSize: 20}})'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Neo4j backup saved to {export_file}")
                return True
            else:
                # Fallback: copy data directory
                print("   ⚠️  Cypher export failed, copying data directory...")
                cmd = ['docker', 'cp', 'political-neo4j:/data', neo4j_backup_dir]
                subprocess.run(cmd, check=True)
                print(f"   ✅ Neo4j data directory copied to {neo4j_backup_dir}")
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Neo4j backup failed: {e}")
            return False

    def backup_configurations(self):
        """Backup configuration files"""
        print("📦 Backing up configuration files...")
        
        try:
            config_backup_dir = os.path.join(self.backup_path, 'configs')
            os.makedirs(config_backup_dir, exist_ok=True)
            
            # Files to backup
            config_files = [
                'docker-compose.yml',
                '.env.example',
                'prometheus/prometheus.yml',
                'grafana/provisioning',
                'supabase/kong.yml',
                'localai/config',
                'postgres/init'
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    if os.path.isdir(config_file):
                        shutil.copytree(config_file, os.path.join(config_backup_dir, os.path.basename(config_file)))
                    else:
                        shutil.copy2(config_file, config_backup_dir)
            
            print(f"   ✅ Configuration files backed up to {config_backup_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ Configuration backup failed: {e}")
            return False

    def backup_volumes(self):
        """Backup Docker volumes"""
        print("📦 Backing up Docker volumes...")
        
        try:
            volumes_backup_dir = os.path.join(self.backup_path, 'volumes')
            os.makedirs(volumes_backup_dir, exist_ok=True)
            
            # Important volumes to backup
            volumes = [
                'redis_data',
                'qdrant_data',
                'grafana_data',
                'prometheus_data',
                'minio_data'
            ]
            
            for volume in volumes:
                try:
                    volume_backup = os.path.join(volumes_backup_dir, f'{volume}.tar')
                    cmd = [
                        'docker', 'run', '--rm',
                        '-v', f'political-analysis-dbs_{volume}:/data',
                        '-v', f'{os.path.abspath(volumes_backup_dir)}:/backup',
                        'alpine:latest',
                        'tar', 'czf', f'/backup/{volume}.tar.gz', '-C', '/data', '.'
                    ]
                    subprocess.run(cmd, check=True)
                    print(f"   ✅ Volume {volume} backed up")
                except subprocess.CalledProcessError:
                    print(f"   ⚠️  Volume {volume} backup skipped (may not exist)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Volume backup failed: {e}")
            return False

    def create_backup_manifest(self):
        """Create backup manifest with metadata"""
        print("📦 Creating backup manifest...")
        
        try:
            manifest = {
                'backup_timestamp': self.timestamp,
                'backup_date': datetime.now().isoformat(),
                'backup_version': '1.0',
                'components': {
                    'postgresql': True,
                    'neo4j': True,
                    'configurations': True,
                    'volumes': True
                },
                'restore_instructions': {
                    'postgresql': 'docker exec -i political-postgres psql -U postgres -d political_analysis < postgresql_backup.sql',
                    'neo4j': 'Copy data directory back to Neo4j container',
                    'configurations': 'Copy configuration files back to project directory',
                    'volumes': 'Extract volume archives back to Docker volumes'
                }
            }
            
            manifest_file = os.path.join(self.backup_path, 'backup_manifest.json')
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"   ✅ Backup manifest created: {manifest_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Manifest creation failed: {e}")
            return False

    def compress_backup(self):
        """Compress the backup directory"""
        print("📦 Compressing backup...")
        
        try:
            archive_name = f'political_analysis_backup_{self.timestamp}.tar.gz'
            archive_path = os.path.join(self.backup_dir, archive_name)
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(self.backup_path, arcname=f'backup_{self.timestamp}')
            
            # Remove uncompressed directory
            shutil.rmtree(self.backup_path)
            
            # Get size
            size_mb = os.path.getsize(archive_path) / (1024 * 1024)
            
            print(f"   ✅ Backup compressed to {archive_path}")
            print(f"   📏 Archive size: {size_mb:.1f} MB")
            return archive_path
            
        except Exception as e:
            print(f"   ❌ Compression failed: {e}")
            return None

    def cleanup_old_backups(self, keep_count=5):
        """Remove old backup files"""
        print(f"🧹 Cleaning up old backups (keeping {keep_count})...")
        
        try:
            # Find all backup archives
            backup_files = []
            for file in os.listdir(self.backup_dir):
                if file.startswith('political_analysis_backup_') and file.endswith('.tar.gz'):
                    file_path = os.path.join(self.backup_dir, file)
                    backup_files.append((file_path, os.path.getctime(file_path)))
            
            # Sort by creation time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            removed_count = 0
            for file_path, _ in backup_files[keep_count:]:
                os.remove(file_path)
                print(f"   🗑️  Removed old backup: {os.path.basename(file_path)}")
                removed_count += 1
            
            if removed_count == 0:
                print(f"   ✅ No old backups to remove")
            else:
                print(f"   ✅ Removed {removed_count} old backup(s)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Cleanup failed: {e}")
            return False

    def run_backup(self):
        """Run complete backup process"""
        print("🚀 Starting Political Analysis Database backup...")
        print("=" * 60)
        
        success_count = 0
        total_tasks = 4
        
        # Run backup tasks
        if self.backup_postgresql():
            success_count += 1
        
        if self.backup_neo4j():
            success_count += 1
        
        if self.backup_configurations():
            success_count += 1
        
        if self.backup_volumes():
            success_count += 1
        
        # Create manifest
        self.create_backup_manifest()
        
        # Compress backup
        archive_path = self.compress_backup()
        
        # Cleanup old backups
        self.cleanup_old_backups()
        
        print("\n" + "=" * 60)
        print("📊 BACKUP SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Successful tasks: {success_count}/{total_tasks}")
        
        if archive_path:
            print(f"Backup archive: {archive_path}")
            print("✅ Backup completed successfully!")
            return True
        else:
            print("❌ Backup completed with errors!")
            return False

def main():
    """Main backup function"""
    backup = DatabaseBackup()
    success = backup.run_backup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()