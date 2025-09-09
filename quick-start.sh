#!/bin/bash
# Quick start script for Political Analysis Database System
# This script demonstrates the system without requiring Docker

set -euo pipefail

echo "🚀 Political Analysis Database System - Quick Demo"
echo ""

# Check if UV is available
if command -v uv &> /dev/null; then
    echo "✅ UV package manager found"
    PYTHON_CMD="uv run python"
else
    echo "⚠️  UV not found, using regular Python"
    PYTHON_CMD="python"
fi

# Show help
echo "📖 Application help:"
SKIP_AUTO_INSTALL=1 $PYTHON_CMD civic_legis_unified.py --help

echo ""
echo "📊 Available commands:"
echo "  --init-db                    Initialize database schema"
echo "  --sync-govinfo              Sync federal legislation data"
echo "  --sync-openstates           Sync state legislation data"
echo "  --propublica-sync           Sync voting records"
echo "  --embed                     Generate embeddings"
echo "  --serve                     Start REST API server"
echo "  --generate-review-report    Generate code review report"
echo ""

# Generate a review report as a demo
echo "📋 Generating code review report:"
SKIP_AUTO_INSTALL=1 $PYTHON_CMD civic_legis_unified.py --generate-review-report

echo ""
echo "🐳 To start the complete system with Docker:"
echo "  chmod +x scripts/deploy.sh"
echo "  ./scripts/deploy.sh"
echo ""
echo "🛠️  For development:"
echo "  chmod +x scripts/dev.sh"
echo "  ./scripts/dev.sh help"
echo ""
echo "📚 See README.md for detailed setup instructions"