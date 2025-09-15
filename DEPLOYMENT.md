# Deployment Guide - Political Analysis Database System

This guide provides step-by-step instructions for deploying the Political Analysis Database System in different environments.

## 🚀 Quick Start (Recommended)

### Option 1: Automated Deployment Script
```bash
# Clone the repository
git clone <repository-url>
cd political-analysis-dbs

# Run the automated deployment script
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### Option 2: Manual Docker Compose Deployment
```bash
# 1. Ensure Docker and Docker Compose are installed
docker --version
docker compose version

# 2. Create environment file
cp .env.example .env

# 3. Edit .env with your settings (optional for local development)
nano .env

# 4. Start all services
docker compose up -d --build

# 5. Wait for services to be healthy
docker compose ps

# 6. Access the applications
# API: http://localhost:8100
# Supabase Studio: http://localhost:3001
# Neo4j: http://localhost:7474
```

## 🏗️ Architecture Overview

The system consists of several interconnected services:

### Core Database Stack
- **PostgreSQL with pgvector**: Primary database with vector embeddings
- **Supabase**: Complete backend-as-a-service stack
  - Auth service (port 9999)
  - REST API (port 3000) 
  - Realtime (port 4000)
  - Storage (port 5000)
  - Studio/Dashboard (port 3001)

### AI and Processing
- **LocalAI** (port 8080): Local LLM processing
- **Political Analysis API** (port 8100): Main application

### Knowledge Graphs
- **Neo4j** (ports 7474, 7687): Graph database with browser
- **Redis Stack** (ports 6379, 8001): Caching and optional graph storage

### Admin Tools
- **Adminer** (port 8082): Database administration
- **Supabase Meta** (port 8081): Database metadata API

## 🔧 Configuration

### Environment Variables

#### Database Configuration
```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=civic_kg
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password  # Change in production
```

#### API Keys (Optional - for data ingestion)
```bash
GOVINFO_API_KEY=your-govinfo-key
OPENSTATES_API_KEY=your-openstates-key  
PROPUBLICA_API_KEY=your-propublica-key
```

#### AI Configuration
```bash
EMBED_MODEL=all-MiniLM-L6-v2
LOCALAI_ENDPOINT=http://localai:8080/v1
OPENAI_API_KEY=optional-for-external-llm
```

#### Neo4j Configuration
```bash
ENABLE_NEO4J=1
NEO4J_PASSWORD=your-neo4j-password  # Change in production
```

#### Supabase Configuration
```bash
JWT_SECRET=your-super-secret-jwt-token-with-at-least-32-characters-long
```

## 🚦 Service Health Checks

### Check All Services
```bash
docker compose ps
```

### Individual Service Health
```bash
# API Health
curl http://localhost:8100/health

# Supabase REST API
curl http://localhost:3000/

# Neo4j (requires authentication)
curl -u neo4j:your-password http://localhost:7474/

# LocalAI
curl http://localhost:8080/health
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f postgres
docker compose logs -f supabase-auth
```

## 📊 Initial Data Setup

### 1. Initialize Database Schema
```bash
# Via API call
curl -X GET http://localhost:8100/health

# Or via CLI
docker compose exec api python civic_legis_unified.py --init-db
```

### 2. Ingest Sample Data (Requires API Keys)
```bash
# Federal legislation (past 5 days)
docker compose exec api python civic_legis_unified.py \
  --sync-govinfo --govinfo-collections BILLSTATUS --govinfo-days 5

# State legislation  
docker compose exec api python civic_legis_unified.py \
  --sync-openstates --openstates-states "California,New York"

# Congressional votes
docker compose exec api python civic_legis_unified.py \
  --propublica-sync --congress 118
```

### 3. Generate Embeddings
```bash
docker compose exec api python civic_legis_unified.py --embed
```

### 4. Build Politician Profiles
```bash
docker compose exec api python civic_legis_unified.py --build-profiles
```

## 🔄 Development Workflow

### Development with Hot Reload
```bash
# Use development compose file
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Or use the dev helper script
chmod +x scripts/dev.sh
./scripts/dev.sh start
```

### Development Commands
```bash
# View logs
./scripts/dev.sh logs api

# Open shell in container
./scripts/dev.sh shell api

# Run tests
./scripts/dev.sh test

# Query system
./scripts/dev.sh query "healthcare legislation"

# Check health
./scripts/dev.sh health
```

## 📱 Using the APIs

### Political Analysis API

#### Query Documents
```bash
curl -X POST http://localhost:8100/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "healthcare privacy legislation", 
    "k": 5, 
    "plain": true
  }'
```

#### Get Bill Information
```bash
curl -X POST http://localhost:8100/bill \
  -H "Content-Type: application/json" \
  -d '{
    "bill_id": "HR-1234", 
    "plain": true
  }'
```

#### Politician Analysis
```bash
curl -X POST http://localhost:8100/politician \
  -H "Content-Type: application/json" \
  -d '{"politician_id": "MEMBER_ID"}'
```

### Supabase REST API

#### Query Documents Table
```bash
curl "http://localhost:3000/documents?select=*&limit=10" \
  -H "apikey: your-anon-key"
```

#### Query with Filters
```bash
curl "http://localhost:3000/documents?jurisdiction=eq.US-Federal&select=title,jurisdiction" \
  -H "apikey: your-anon-key"
```

## 🛠️ Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker resources
docker system df
docker system prune  # Free up space if needed

# Check individual service logs
docker compose logs postgres
docker compose logs api

# Restart specific service
docker compose restart api
```

#### Database Connection Issues
```bash
# Check PostgreSQL is ready
docker compose exec postgres pg_isready -U postgres

# Check from application
docker compose exec api python -c "
import psycopg2
try:
    conn = psycopg2.connect(host='postgres', user='postgres', password='postgres', database='civic_kg')
    print('Database connection: OK')
    conn.close()
except Exception as e:
    print(f'Database connection error: {e}')
"
```

#### Out of Memory Issues
```bash
# Check resource usage
docker stats

# Reduce resources if needed
docker compose down
docker compose up -d --scale api=1  # Reduce replicas
```

#### Port Conflicts
```bash
# Check what's using ports
netstat -tulpn | grep :8100
netstat -tulpn | grep :3000

# Change ports in docker-compose.yml if needed
```

### Performance Tuning

#### PostgreSQL Optimization
```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U postgres -d civic_kg

# Check database size
SELECT pg_size_pretty(pg_database_size('civic_kg'));

# Check table sizes
SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Optimize tables
VACUUM ANALYZE;
```

#### Application Performance
```bash
# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8100/health

# Check embedding performance
docker compose exec api python civic_legis_unified.py --embed --verbose
```

## 🔐 Security Considerations

### Production Deployment

#### 1. Change Default Passwords
```bash
# Update .env file
POSTGRES_PASSWORD=your-secure-password
NEO4J_PASSWORD=your-secure-neo4j-password
JWT_SECRET=your-unique-jwt-secret-at-least-32-characters
```

#### 2. Enable TLS/SSL
```bash
# Add reverse proxy (nginx/traefik) with SSL termination
# Update SITE_URL to https
SITE_URL=https://your-domain.com
```

#### 3. Network Security
```bash
# Limit exposed ports in production
# Use internal networks for service communication
# Enable firewall rules
```

#### 4. Backup Strategy
```bash
# Database backup
docker compose exec postgres pg_dump -U postgres civic_kg > backup.sql

# Volume backup
docker run --rm -v political-analysis-dbs_postgres_data:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup.tar.gz /data
```

## 📈 Monitoring

### Health Checks
```bash
# Automated health check script
#!/bin/bash
services=("api" "postgres" "supabase-auth" "supabase-rest" "localai" "neo4j" "redis")

for service in "${services[@]}"; do
    if docker compose ps "$service" | grep -q "Up (healthy)"; then
        echo "✅ $service is healthy"
    else
        echo "❌ $service is unhealthy"
    fi
done
```

### Log Monitoring
```bash
# Follow logs with timestamps
docker compose logs -f -t

# Filter specific logs
docker compose logs api | grep ERROR
docker compose logs postgres | grep FATAL
```

### Resource Monitoring
```bash
# Container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# System resources
free -h
df -h
```

## 🚀 Advanced Deployment Options

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml political-analysis
```

### Kubernetes
```bash
# Convert compose to k8s (using kompose)
kompose convert

# Apply to cluster
kubectl apply -f .
```

### Cloud Deployment
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- DigitalOcean App Platform

Each cloud provider offers managed services that can replace individual components (managed PostgreSQL, Redis, etc.)

## 📞 Support

### Getting Help
1. Check the logs: `docker compose logs -f`
2. Review this troubleshooting guide
3. Check GitHub issues
4. Verify environment variables
5. Ensure adequate system resources (8GB+ RAM recommended)

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Docker and Docker Compose versions
- Complete error logs
- Steps to reproduce
- Environment configuration (without secrets)

## 🔄 Updates and Maintenance

### Updating the System
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker compose down
docker compose up -d --build

# Check for issues
docker compose ps
docker compose logs -f
```

### Database Maintenance
```bash
# Regular maintenance
docker compose exec postgres psql -U postgres -d civic_kg -c "VACUUM ANALYZE;"

# Update statistics
docker compose exec postgres psql -U postgres -d civic_kg -c "ANALYZE;"
```

This deployment guide should provide everything needed to successfully deploy and maintain the Political Analysis Database System.