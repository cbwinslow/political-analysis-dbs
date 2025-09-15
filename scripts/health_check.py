#!/usr/bin/env python3
"""
Comprehensive health check script for Political Analysis Database
"""

import asyncio
import aiohttp
import json
import time
import subprocess
from typing import Dict, List, Any, Optional
import os
import sys
import socket
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HealthChecker:
    def __init__(self):
        self.services = {
            'api': {'url': 'http://localhost:8000/health', 'critical': True},
            'grafana': {'url': 'http://localhost:3000/api/health', 'critical': False},
            'neo4j': {'url': 'http://localhost:7474/db/manage/server/core/available', 'critical': True},
            'qdrant': {'url': 'http://localhost:6333/health', 'critical': True},
            'elasticsearch': {'url': 'http://localhost:9200/_cluster/health', 'critical': True},
            'prometheus': {'url': 'http://localhost:9090/-/healthy', 'critical': False},
            'minio': {'url': 'http://localhost:9000/minio/health/live', 'critical': False},
            'flowise': {'url': 'http://localhost:3002/api/v1/ping', 'critical': False},
            'open-webui': {'url': 'http://localhost:3001/health', 'critical': False},
            'postgres': {'port': 5432, 'critical': True},
            'redis': {'port': 6379, 'critical': True}
        }
        
    async def check_http_service(self, service_name: str, url: str) -> Dict[str, Any]:
        """Check health of an HTTP service"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    status = 'healthy' if response.status == 200 else 'unhealthy'
                    response_time = time.time() - start_time
                    
                    # Try to get response content for additional info
                    try:
                        content = await response.text()
                        if len(content) > 200:
                            content = content[:200] + "..."
                    except:
                        content = None
                    
                    return {
                        'service': service_name,
                        'status': status,
                        'response_time': response_time,
                        'status_code': response.status,
                        'url': url,
                        'content': content
                    }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'service': service_name,
                'status': 'error',
                'response_time': response_time,
                'error': str(e),
                'url': url
            }
    
    def check_port_service(self, service_name: str, port: int, host: str = 'localhost') -> Dict[str, Any]:
        """Check if a port-based service is responding"""
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            response_time = time.time() - start_time
            status = 'healthy' if result == 0 else 'unhealthy'
            
            return {
                'service': service_name,
                'status': status,
                'response_time': response_time,
                'port': port,
                'host': host
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'service': service_name,
                'status': 'error',
                'response_time': response_time,
                'error': str(e),
                'port': port,
                'host': host
            }
    
    def check_docker_services(self) -> Dict[str, Any]:
        """Check Docker container status"""
        try:
            result = subprocess.run(['docker', 'compose', 'ps', '--format', 'json'], 
                                   capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            container = json.loads(line)
                            containers.append({
                                'name': container.get('Name', 'unknown'),
                                'state': container.get('State', 'unknown'),
                                'status': container.get('Status', 'unknown'),
                                'health': container.get('Health', 'unknown')
                            })
                        except json.JSONDecodeError:
                            pass
                
                running_count = sum(1 for c in containers if c['state'] == 'running')
                total_count = len(containers)
                
                return {
                    'status': 'healthy' if running_count == total_count else 'unhealthy',
                    'running_containers': running_count,
                    'total_containers': total_count,
                    'containers': containers
                }
            else:
                return {
                    'status': 'error',
                    'error': result.stderr or 'Failed to get docker status'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # Memory usage
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = None
            mem_available = None
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) * 1024  # Convert to bytes
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) * 1024  # Convert to bytes
            
            mem_used = mem_total - mem_available if mem_total and mem_available else None
            mem_usage_percent = (mem_used / mem_total * 100) if mem_total and mem_used else None
            
            # Disk usage
            disk_usage = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
            disk_info = disk_usage.stdout.split('\n')[1].split() if disk_usage.returncode == 0 else []
            
            # Load average
            with open('/proc/loadavg', 'r') as f:
                load_avg = f.read().split()[:3]
            
            return {
                'memory': {
                    'total_gb': round(mem_total / (1024**3), 2) if mem_total else None,
                    'used_gb': round(mem_used / (1024**3), 2) if mem_used else None,
                    'available_gb': round(mem_available / (1024**3), 2) if mem_available else None,
                    'usage_percent': round(mem_usage_percent, 1) if mem_usage_percent else None
                },
                'disk': {
                    'filesystem': disk_info[0] if len(disk_info) > 0 else None,
                    'size': disk_info[1] if len(disk_info) > 1 else None,
                    'used': disk_info[2] if len(disk_info) > 2 else None,
                    'available': disk_info[3] if len(disk_info) > 3 else None,
                    'usage_percent': disk_info[4] if len(disk_info) > 4 else None
                },
                'load_average': {
                    '1min': float(load_avg[0]) if len(load_avg) > 0 else None,
                    '5min': float(load_avg[1]) if len(load_avg) > 1 else None,
                    '15min': float(load_avg[2]) if len(load_avg) > 2 else None
                }
            }
        except Exception as e:
            return {
                'error': str(e)
            }
    
    async def check_all_services(self) -> List[Dict[str, Any]]:
        """Check health of all services"""
        results = []
        
        # Check HTTP services
        http_tasks = []
        for service_name, config in self.services.items():
            if 'url' in config:
                http_tasks.append(self.check_http_service(service_name, config['url']))
        
        if http_tasks:
            http_results = await asyncio.gather(*http_tasks)
            results.extend(http_results)
        
        # Check port-based services
        for service_name, config in self.services.items():
            if 'port' in config:
                port_result = self.check_port_service(service_name, config['port'])
                results.append(port_result)
        
        return results
    
    def print_results(self, results: List[Dict[str, Any]], docker_status: Dict[str, Any], 
                     system_resources: Dict[str, Any]) -> int:
        """Print comprehensive health check results"""
        print(f"\n🏥 Political Analysis Database Health Check")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Service health
        print("\n📊 Service Health:")
        healthy_count = 0
        critical_healthy = 0
        critical_total = 0
        total_count = len(results)
        
        for result in results:
            service_config = self.services.get(result['service'], {})
            is_critical = service_config.get('critical', False)
            
            if is_critical:
                critical_total += 1
            
            status_emoji = "✅" if result['status'] == 'healthy' else "❌" if result['status'] == 'unhealthy' else "⚠️"
            service_name = result['service'].ljust(15)
            status = result['status'].upper().ljust(10)
            response_time = f"{result['response_time']:.2f}s".ljust(8)
            critical_marker = " (CRITICAL)" if is_critical else ""
            
            print(f"{status_emoji} {service_name} {status} {response_time}{critical_marker}")
            
            if result['status'] == 'healthy':
                healthy_count += 1
                if is_critical:
                    critical_healthy += 1
            elif result['status'] == 'error':
                error_msg = result.get('error', 'Unknown error')
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                print(f"   Error: {error_msg}")
        
        # Docker status
        print(f"\n🐳 Docker Status:")
        if docker_status.get('status') == 'healthy':
            print(f"✅ All containers running ({docker_status['running_containers']}/{docker_status['total_containers']})")
        else:
            print(f"❌ Container issues ({docker_status.get('running_containers', 0)}/{docker_status.get('total_containers', 0)} running)")
            if 'error' in docker_status:
                print(f"   Error: {docker_status['error']}")
        
        # System resources
        print(f"\n💾 System Resources:")
        if 'error' not in system_resources:
            mem = system_resources.get('memory', {})
            disk = system_resources.get('disk', {})
            load = system_resources.get('load_average', {})
            
            if mem.get('usage_percent'):
                mem_emoji = "✅" if mem['usage_percent'] < 80 else "⚠️" if mem['usage_percent'] < 90 else "❌"
                print(f"{mem_emoji} Memory: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['usage_percent']:.1f}%)")
            
            if disk.get('usage_percent'):
                disk_percent = float(disk['usage_percent'].rstrip('%'))
                disk_emoji = "✅" if disk_percent < 80 else "⚠️" if disk_percent < 90 else "❌"
                print(f"{disk_emoji} Disk: {disk['used']} / {disk['size']} ({disk['usage_percent']})")
            
            if load.get('1min') is not None:
                load_emoji = "✅" if load['1min'] < 2.0 else "⚠️" if load['1min'] < 4.0 else "❌"
                print(f"{load_emoji} Load: {load['1min']:.2f} {load['5min']:.2f} {load['15min']:.2f}")
        else:
            print(f"❌ Error getting system resources: {system_resources['error']}")
        
        # Overall status
        print("\n" + "=" * 70)
        print(f"📈 Overall Status:")
        print(f"   Services: {healthy_count}/{total_count} healthy")
        print(f"   Critical: {critical_healthy}/{critical_total} healthy")
        print(f"   Docker: {docker_status.get('running_containers', 0)}/{docker_status.get('total_containers', 0)} running")
        
        # Determine exit code
        if critical_healthy == critical_total and docker_status.get('status') == 'healthy':
            print("🎉 System is healthy!")
            return 0
        else:
            print("⚠️  System has issues that need attention")
            return 1

async def main():
    """Main health check function"""
    checker = HealthChecker()
    
    print("Running comprehensive health check...")
    
    # Check services
    service_results = await checker.check_all_services()
    
    # Check Docker
    docker_status = checker.check_docker_services()
    
    # Check system resources
    system_resources = checker.check_system_resources()
    
    # Print results
    exit_code = checker.print_results(service_results, docker_status, system_resources)
    
    # Save detailed results to file for monitoring
    os.makedirs('logs', exist_ok=True)
    health_report = {
        'timestamp': datetime.now().isoformat(),
        'services': service_results,
        'docker': docker_status,
        'system': system_resources,
        'summary': {
            'healthy_services': sum(1 for r in service_results if r['status'] == 'healthy'),
            'total_services': len(service_results),
            'critical_healthy': sum(1 for r in service_results 
                                  if r['status'] == 'healthy' and 
                                  checker.services.get(r['service'], {}).get('critical', False)),
            'critical_total': sum(1 for service, config in checker.services.items() 
                                if config.get('critical', False)),
            'overall_status': 'healthy' if exit_code == 0 else 'unhealthy'
        }
    }
    
    with open('logs/health_check.json', 'w') as f:
        json.dump(health_report, f, indent=2)
    
    # Also save a simple status file for monitoring scripts
    with open('logs/health_status', 'w') as f:
        f.write('healthy' if exit_code == 0 else 'unhealthy')
    
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())