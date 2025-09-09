#!/bin/bash
# Quick deployment script for Political Analysis Database System

set -euo pipefail

echo "🚀 Starting Political Analysis Database System..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please review and update .env file with your settings before continuing."
    echo "Press Enter to continue or Ctrl+C to stop and edit .env"
    read -r
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/cache data/local data/exports models

# Build and start services
echo "🔨 Building and starting services..."
docker compose up -d --build

echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
services=("postgres" "localai" "neo4j" "redis")
for service in "${services[@]}"; do
    if docker compose ps "$service" | grep -q "Up (healthy)"; then
        echo "✅ $service is healthy"
    else
        echo "⚠️  $service may still be starting..."
    fi
done

# Initialize the application
echo "🔧 Initializing application..."
docker compose exec -T api uv run python civic_legis_unified.py --init-db

echo "🎉 Political Analysis Database System is ready!"
echo ""
echo "📊 Access URLs:"
echo "  - Main API: http://localhost:8100"
echo "  - Supabase Studio: http://localhost:3001"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - Adminer: http://localhost:8082"
echo ""
echo "🔧 Next steps:"
echo "  1. Visit http://localhost:8100/health to verify the API"
echo "  2. Set up API keys in .env for data ingestion"
echo "  3. Run: docker compose exec api uv run python civic_legis_unified.py --sync-govinfo --embed"
echo ""
echo "📚 View logs: docker compose logs -f"
echo "🛑 Stop services: docker compose down"