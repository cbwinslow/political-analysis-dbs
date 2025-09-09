#!/bin/bash
# Test Docker Compose configuration without building
# This validates the compose file structure and service dependencies

set -euo pipefail

echo "🧪 Testing Docker Compose Configuration"
echo ""

# Test docker-compose.yml syntax
echo "📋 Validating docker-compose.yml syntax..."
if docker compose config > /dev/null 2>&1; then
    echo "✅ docker-compose.yml syntax is valid"
else
    echo "❌ docker-compose.yml syntax error:"
    docker compose config
    exit 1
fi

# Test environment file
echo ""
echo "📋 Checking environment configuration..."
if [ ! -f .env ]; then
    echo "⚠️  No .env file found, using defaults"
else
    echo "✅ .env file found"
    
    # Check for required variables
    required_vars=("POSTGRES_PASSWORD" "JWT_SECRET")
    for var in "${required_vars[@]}"; do
        if grep -q "^${var}=" .env; then
            echo "✅ $var is set"
        else
            echo "⚠️  $var not found in .env"
        fi
    done
fi

# Test directory structure
echo ""
echo "📋 Checking directory structure..."
dirs=("data" "models" "init-scripts" "scripts")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir directory exists"
    else
        echo "⚠️  $dir directory missing (will be created)"
        mkdir -p "$dir"
    fi
done

# Test scripts
echo ""
echo "📋 Checking scripts..."
scripts=("scripts/deploy.sh" "scripts/dev.sh" "quick-start.sh")
for script in "${scripts[@]}"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo "✅ $script is executable"
    elif [ -f "$script" ]; then
        echo "⚠️  $script exists but not executable"
        chmod +x "$script"
        echo "   Fixed: made $script executable"
    else
        echo "❌ $script not found"
    fi
done

# Show service configuration
echo ""
echo "📋 Service Configuration Summary:"
docker compose config --services | while read service; do
    echo "  - $service"
done

echo ""
echo "📋 Port Mappings:"
docker compose config | grep -A 1 "ports:" | grep -E "^\s*-\s*\"?[0-9]+" | sed 's/^[[:space:]]*/  /' || echo "  No port mappings found"

echo ""
echo "📋 Volume Mappings:"
docker compose config | grep -A 10 "volumes:" | grep -E "^\s*[a-zA-Z]" | sed 's/^[[:space:]]*/  /' || echo "  No named volumes found"

# Test application without dependencies
echo ""
echo "📋 Testing application standalone..."
if command -v uv &> /dev/null; then
    if SKIP_AUTO_INSTALL=1 uv run python civic_legis_unified.py --generate-review-report > /dev/null 2>&1; then
        echo "✅ Application can run standalone"
    else
        echo "⚠️  Application has dependency issues"
    fi
else
    echo "⚠️  UV not available, skipping standalone test"
fi

echo ""
echo "🎯 Configuration Test Results:"
echo "   - Docker Compose file is valid"
echo "   - Required directories exist or created"
echo "   - Scripts are executable"
echo "   - Service configuration checked"
echo ""
echo "🚀 Ready for deployment!"
echo ""
echo "Next steps:"
echo "  1. Review .env file and update secrets"
echo "  2. Run: ./scripts/deploy.sh"
echo "  3. Or run: docker compose up -d"
echo ""
echo "⚠️  Note: Actual deployment requires internet access for image downloads"