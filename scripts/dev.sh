#!/bin/bash
# Development helper script

set -euo pipefail

COMMAND=${1:-help}

case $COMMAND in
    "start")
        echo "🚀 Starting development environment..."
        docker compose up -d
        ;;
    "stop")
        echo "🛑 Stopping development environment..."
        docker compose down
        ;;
    "restart")
        echo "🔄 Restarting development environment..."
        docker compose restart
        ;;
    "logs")
        SERVICE=${2:-}
        if [ -n "$SERVICE" ]; then
            docker compose logs -f "$SERVICE"
        else
            docker compose logs -f
        fi
        ;;
    "shell")
        SERVICE=${2:-api}
        echo "🐚 Opening shell in $SERVICE..."
        docker compose exec "$SERVICE" /bin/bash
        ;;
    "test")
        echo "🧪 Running tests..."
        docker compose exec api uv run python civic_legis_unified.py --run-self-tests
        ;;
    "ingest")
        echo "📥 Running data ingestion..."
        docker compose exec api uv run python civic_legis_unified.py --sync-govinfo --govinfo-collections BILLSTATUS --govinfo-days 5
        ;;
    "embed")
        echo "🧠 Generating embeddings..."
        docker compose exec api uv run python civic_legis_unified.py --embed
        ;;
    "profiles")
        echo "👥 Building politician profiles..."
        docker compose exec api uv run python civic_legis_unified.py --build-profiles
        ;;
    "query")
        QUERY=${2:-"healthcare legislation"}
        echo "🔍 Querying: $QUERY"
        curl -X POST http://localhost:8100/query \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"$QUERY\",\"k\":5,\"plain\":true}" | jq
        ;;
    "health")
        echo "🏥 Checking service health..."
        echo "API Health:"
        curl -s http://localhost:8100/health | jq
        echo -e "\nSupabase Health:"
        curl -s http://localhost:3000/ | head -5
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        docker compose down -v
        docker system prune -f
        ;;
    "backup")
        echo "💾 Creating backup..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        docker compose exec postgres pg_dump -U postgres civic_kg > "backup_${TIMESTAMP}.sql"
        echo "Backup created: backup_${TIMESTAMP}.sql"
        ;;
    "help"|*)
        echo "🛠️  Political Analysis Database System - Development Helper"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start          Start development environment"
        echo "  stop           Stop development environment"
        echo "  restart        Restart development environment"
        echo "  logs [service] View logs (all services or specific service)"
        echo "  shell [service] Open shell in service (default: api)"
        echo "  test           Run self-tests"
        echo "  ingest         Run data ingestion"
        echo "  embed          Generate embeddings"
        echo "  profiles       Build politician profiles"
        echo "  query [text]   Query the system"
        echo "  health         Check service health"
        echo "  clean          Clean up containers and volumes"
        echo "  backup         Create database backup"
        echo "  help           Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs api"
        echo "  $0 query 'privacy legislation'"
        echo "  $0 shell postgres"
        ;;
esac