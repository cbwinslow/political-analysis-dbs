# Political Analysis Database

A comprehensive political legislature analysis platform using modern database technologies, AI/ML services, and workflow automation tools.

## 🏛️ Overview

This project provides a complete stack for analyzing political data including legislators, bills, votes, and relationships. It combines multiple database technologies and AI services to provide deep insights into political patterns and relationships.

## 🛠️ Technology Stack

### Core Databases
- **PostgreSQL + Supabase**: Primary relational database with real-time capabilities
- **pgvector**: Vector similarity search for semantic analysis
- **Neo4j**: Graph database for relationship analysis
- **Qdrant**: Vector database for embeddings storage
- **Redis**: Caching and session storage

### AI/ML Services
- **LocalAI**: Local LLM inference
- **Ollama**: LLM model management
- **OpenWebUI**: Web interface for AI models
- **Sentence Transformers**: Text embeddings

### Workflow & Analytics
- **Flowise**: Low-code AI workflow builder
- **n8n**: Workflow automation
- **Apache Airflow**: Data pipeline orchestration
- **Elasticsearch + Kibana**: Search and analytics

### Monitoring & Storage
- **Grafana + Prometheus**: Monitoring and dashboards
- **MinIO**: S3-compatible object storage
- **Jupyter**: Data analysis notebooks

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- 8GB+ RAM recommended
- GPU support recommended for AI models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cbwinslow/political-analysis-dbs.git
   cd political-analysis-dbs
   ```

2. **Run the setup script**
   ```bash
   ./setup.sh
   ```

3. **Start the API server**
   ```bash
   cd app
   python main.py
   ```

4. **Import sample data**
   ```bash
   python scripts/import_sample_data.py
   ```

## 🌐 Service URLs

After setup, access the following services:

| Service | URL | Credentials |
|---------|-----|-------------|
| API Server | http://localhost:8000 | - |
| Neo4j Browser | http://localhost:7474 | neo4j/political123 |
| Grafana | http://localhost:3000 | admin/political123 |
| Supabase | http://localhost:8000 | - |
| OpenWebUI | http://localhost:3001 | - |
| Flowise | http://localhost:3002 | - |
| n8n | http://localhost:5678 | admin/political123 |
| Kibana | http://localhost:5601 | - |
| MinIO | http://localhost:9001 | political/political123 |
| Airflow | http://localhost:8081 | airflow/political123 |
| Jupyter | http://localhost:8888 | token: political123 |

## 📊 Data Model

### Core Entities
- **Legislators**: Representatives and senators with biographical data
- **Bills**: Legislative proposals with full text and metadata
- **Votes**: Individual voting records linking legislators to bills
- **Committees**: Legislative committees and memberships
- **Speeches**: Recorded speeches and statements
- **Lobbying Activities**: Lobbying efforts and expenditures

### Relationships
- Voting patterns and alignments
- Committee memberships and roles
- Bill sponsorship and co-sponsorship
- Political entity relationships
- Influence networks

## 🤖 AI Features

### Semantic Search
- Find similar legislators based on biographical information
- Discover related bills using content similarity
- Analyze speech patterns and sentiment

### Graph Analytics
- Community detection in voting networks
- Influence and relationship mapping
- Party alignment analysis
- Bill support network visualization

### Embeddings
- Text embeddings for semantic similarity
- Vector storage in multiple databases
- Batch embedding processing
- Similarity threshold controls

## 📈 Analytics Capabilities

### Voting Analysis
- Party alignment scoring
- Controversial vote identification
- Voting frequency patterns
- Cross-party agreement analysis

### Bill Tracking
- Support/opposition statistics
- Topic-based bill categorization
- Trending legislation identification
- Amendment tracking

### Network Analysis
- Political influence networks
- Community detection algorithms
- Relationship strength scoring
- Centrality measures

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```env
# Database connections
DATABASE_URL=postgresql://postgres:political123@localhost:5432/political_analysis
NEO4J_URI=bolt://localhost:7687
REDIS_URL=redis://:political123@localhost:6379/0

# AI Services
OPENAI_API_KEY=your_api_key_here
LOCALAI_URL=http://localhost:8080

# Additional service configurations...
```

### Docker Compose
The stack includes comprehensive service definitions with:
- Persistent volumes for data storage
- Network isolation and communication
- Resource limits and health checks
- GPU support for AI workloads

## 🔍 API Usage

### Example Requests

**Create a Legislator**
```bash
curl -X POST "http://localhost:8000/api/v1/legislators/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "party": "Independent",
    "state": "CA",
    "chamber": "house",
    "bio_text": "Environmental advocate focused on climate policy"
  }'
```

**Search Similar Bills**
```bash
curl -X POST "http://localhost:8000/api/v1/bills/search/similar" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change renewable energy",
    "threshold": 0.7,
    "limit": 10
  }'
```

**Get Voting Patterns**
```bash
curl "http://localhost:8000/api/v1/analytics/voting-patterns/{legislator_id}"
```

## 🚨 Workflow Automation

### n8n Workflows
- Automated data ingestion from legislative APIs
- Real-time alerts for new bills and votes
- Social media monitoring for political sentiment
- Automated report generation

### Flowise AI Flows
- Question-answering over legislative data
- Automated bill summarization
- Sentiment analysis of speeches
- Recommendation systems for similar legislation

### Airflow Pipelines
- Daily data synchronization
- Embedding regeneration
- Graph relationship updates
- Analytics report generation

## 📚 Data Sources

The platform can integrate with various political data sources:
- Congress.gov API
- OpenStates API
- ProPublica Congress API
- Legislative branch RSS feeds
- Campaign finance databases
- Lobbying disclosure reports

## 🧪 Development

### Testing
```bash
# Run tests
pytest tests/

# Test specific service
pytest tests/test_legislator_service.py
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

### Adding New Services
1. Add service to `docker-compose.yml`
2. Update configuration files
3. Add service integration to application code
4. Update documentation

## 🔒 Security

- Row Level Security (RLS) policies in PostgreSQL
- JWT-based authentication via Supabase Auth
- API rate limiting and CORS configuration
- Secure secrets management
- Network isolation between services

## 📊 Monitoring

### Grafana Dashboards
- Database performance metrics
- API response times and error rates
- AI model inference statistics
- Resource utilization monitoring

### Prometheus Metrics
- Custom application metrics
- Service health monitoring
- Alert configuration
- Performance tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example notebooks in `/notebooks`

## 🚧 Roadmap

- [ ] Real-time legislative data feeds
- [ ] Advanced NLP for bill analysis
- [ ] Machine learning prediction models
- [ ] Enhanced visualization capabilities
- [ ] Mobile application interface
- [ ] Multi-language support
- [ ] Advanced graph algorithms
- [ ] Automated fact-checking integration