#!/bin/bash

# Political Analysis Database Setup Script

echo "🏛️  Setting up Political Analysis Database Stack..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📄 Creating environment file..."
    cp .env.example .env
    echo "✅ Environment file created. Please edit .env with your API keys."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p {neo4j/plugins,postgres/init,prometheus,grafana/provisioning/{dashboards,datasources},supabase,localai/config,notebooks,logs}

# Download Neo4j plugins
echo "🔌 Downloading Neo4j plugins..."
mkdir -p neo4j/plugins
cd neo4j/plugins

# APOC plugin
if [ ! -f apoc-5.15.0-core.jar ]; then
    wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/5.15.0/apoc-5.15.0-core.jar
fi

# Graph Data Science plugin
if [ ! -f neo4j-graph-data-science-2.5.0.jar ]; then
    wget https://graphdatascience.ninja/neo4j-graph-data-science-2.5.0.jar
fi

cd ../..

# Pull Docker images
echo "🐳 Pulling Docker images..."
docker-compose pull

# Start core services first
echo "🚀 Starting core services..."
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 15

# Start remaining services
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Setup Ollama models
echo "🤖 Setting up Ollama models..."
docker exec political-ollama ollama pull llama2:7b
docker exec political-ollama ollama pull codellama:7b
docker exec political-ollama ollama pull mistral:7b

echo "✅ Setup complete!"
echo ""
echo "🎯 Service URLs:"
echo "   • API Server: http://localhost:8000"
echo "   • Neo4j Browser: http://localhost:7474 (neo4j/political123)"
echo "   • Grafana: http://localhost:3000 (admin/political123)"
echo "   • Supabase: http://localhost:8000"
echo "   • Qdrant: http://localhost:6333"
echo "   • OpenWebUI: http://localhost:3001"
echo "   • Flowise: http://localhost:3002"
echo "   • n8n: http://localhost:5678 (admin/political123)"
echo "   • Elasticsearch: http://localhost:9200"
echo "   • Kibana: http://localhost:5601"
echo "   • Prometheus: http://localhost:9090"
echo "   • MinIO: http://localhost:9001 (political/political123)"
echo "   • Airflow: http://localhost:8081 (airflow/political123)"
echo "   • Jupyter: http://localhost:8888 (token: political123)"
echo ""
echo "🔧 To start the API server:"
echo "   cd app && python main.py"
echo ""
echo "📊 To run sample data import:"
echo "   python scripts/import_sample_data.py"