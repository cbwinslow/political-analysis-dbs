#!/usr/bin/env python3
"""
Health check script for Political Analysis Database services
"""

import asyncio
import aiohttp
import asyncpg
from neo4j import GraphDatabase
import redis
import json
import sys
from datetime import datetime

class HealthChecker:
    def __init__(self):
        self.services = {
            'api': 'http://localhost:8000/health',
            'neo4j_http': 'http://localhost:7474',
            'grafana': 'http://localhost:3000/api/health',
            'qdrant': 'http://localhost:6333/health',
            'elasticsearch': 'http://localhost:9200/_cluster/health',
            'kibana': 'http://localhost:5601/api/status',
            'prometheus': 'http://localhost:9090/-/healthy',
            'minio': 'http://localhost:9000/minio/health/live',
            'localai': 'http://localhost:8080/health',
            'open_webui': 'http://localhost:3001',
            'flowise': 'http://localhost:3002',
            'n8n': 'http://localhost:5678/healthz',
            'airflow': 'http://localhost:8081/health',
            'jupyter': 'http://localhost:8888'
        }
        
        self.results = {}

    async def check_http_service(self, name, url):
        """Check HTTP service health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status < 400:
                        return {'status': 'healthy', 'response_time': 'ok'}
                    else:
                        return {'status': 'unhealthy', 'error': f'HTTP {response.status}'}
        except Exception as e:
            return {'status': 'unreachable', 'error': str(e)}

    async def check_postgresql(self):
        """Check PostgreSQL connection"""
        try:
            conn = await asyncpg.connect(
                'postgresql://postgres:political123@localhost:5432/political_analysis'
            )
            await conn.execute('SELECT 1')
            await conn.close()
            return {'status': 'healthy', 'response_time': 'ok'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

    def check_neo4j(self):
        """Check Neo4j connection"""
        try:
            driver = GraphDatabase.driver(
                'bolt://localhost:7687',
                auth=('neo4j', 'political123')
            )
            with driver.session() as session:
                result = session.run('RETURN 1 as test')
                result.single()
            driver.close()
            return {'status': 'healthy', 'response_time': 'ok'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

    def check_redis(self):
        """Check Redis connection"""
        try:
            r = redis.Redis(host='localhost', port=6379, password='political123', socket_timeout=5)
            r.ping()
            return {'status': 'healthy', 'response_time': 'ok'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

    async def run_health_checks(self):
        """Run all health checks"""
        print("🏥 Running health checks...")
        print("=" * 50)
        
        # Check HTTP services
        for name, url in self.services.items():
            print(f"Checking {name}...")
            result = await self.check_http_service(name, url)
            self.results[name] = result
            
            status_emoji = "✅" if result['status'] == 'healthy' else "❌"
            print(f"  {status_emoji} {name}: {result['status']}")
            if result['status'] != 'healthy':
                print(f"     Error: {result.get('error', 'Unknown')}")
        
        # Check database services
        print(f"Checking postgresql...")
        postgres_result = await self.check_postgresql()
        self.results['postgresql'] = postgres_result
        status_emoji = "✅" if postgres_result['status'] == 'healthy' else "❌"
        print(f"  {status_emoji} postgresql: {postgres_result['status']}")
        if postgres_result['status'] != 'healthy':
            print(f"     Error: {postgres_result.get('error', 'Unknown')}")
        
        print(f"Checking neo4j_bolt...")
        neo4j_result = self.check_neo4j()
        self.results['neo4j_bolt'] = neo4j_result
        status_emoji = "✅" if neo4j_result['status'] == 'healthy' else "❌"
        print(f"  {status_emoji} neo4j_bolt: {neo4j_result['status']}")
        if neo4j_result['status'] != 'healthy':
            print(f"     Error: {neo4j_result.get('error', 'Unknown')}")
        
        print(f"Checking redis...")
        redis_result = self.check_redis()
        self.results['redis'] = redis_result
        status_emoji = "✅" if redis_result['status'] == 'healthy' else "❌"
        print(f"  {status_emoji} redis: {redis_result['status']}")
        if redis_result['status'] != 'healthy':
            print(f"     Error: {redis_result.get('error', 'Unknown')}")

    def generate_report(self):
        """Generate health check report"""
        print("\n" + "=" * 50)
        print("📊 HEALTH CHECK SUMMARY")
        print("=" * 50)
        
        healthy_count = sum(1 for result in self.results.values() if result['status'] == 'healthy')
        total_count = len(self.results)
        
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Total Services: {total_count}")
        print(f"Healthy Services: {healthy_count}")
        print(f"Unhealthy Services: {total_count - healthy_count}")
        print(f"Health Percentage: {(healthy_count / total_count * 100):.1f}%")
        
        # Group by status
        healthy_services = [name for name, result in self.results.items() if result['status'] == 'healthy']
        unhealthy_services = [name for name, result in self.results.items() if result['status'] != 'healthy']
        
        if healthy_services:
            print(f"\n✅ Healthy Services ({len(healthy_services)}):")
            for service in sorted(healthy_services):
                print(f"   • {service}")
        
        if unhealthy_services:
            print(f"\n❌ Unhealthy Services ({len(unhealthy_services)}):")
            for service in sorted(unhealthy_services):
                error = self.results[service].get('error', 'Unknown error')
                print(f"   • {service}: {error}")
        
        # Service URLs for reference
        print(f"\n🌐 Service URLs:")
        service_urls = {
            'API Server': 'http://localhost:8000',
            'Neo4j Browser': 'http://localhost:7474',
            'Grafana': 'http://localhost:3000',
            'Qdrant': 'http://localhost:6333',
            'Kibana': 'http://localhost:5601',
            'Prometheus': 'http://localhost:9090',
            'MinIO Console': 'http://localhost:9001',
            'OpenWebUI': 'http://localhost:3001',
            'Flowise': 'http://localhost:3002',
            'n8n': 'http://localhost:5678',
            'Airflow': 'http://localhost:8081',
            'Jupyter': 'http://localhost:8888'
        }
        
        for name, url in service_urls.items():
            print(f"   • {name}: {url}")
        
        return healthy_count == total_count

    def save_report_json(self, filename='health_report.json'):
        """Save health check results to JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_services': len(self.results),
                'healthy_services': sum(1 for r in self.results.values() if r['status'] == 'healthy'),
                'health_percentage': (sum(1 for r in self.results.values() if r['status'] == 'healthy') / len(self.results)) * 100
            },
            'services': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Health report saved to {filename}")

async def main():
    """Main health check function"""
    checker = HealthChecker()
    await checker.run_health_checks()
    
    all_healthy = checker.generate_report()
    checker.save_report_json()
    
    # Exit with appropriate code
    sys.exit(0 if all_healthy else 1)

if __name__ == "__main__":
    asyncio.run(main())