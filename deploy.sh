#!/bin/bash

# Political Analysis Database - Comprehensive Deployment Script for Local and Remote (Proxmox)
# This script handles both local development and production deployment

set -e

# Configuration
PROJECT_NAME="political-analysis-dbs"
DEFAULT_ENV="development"
ENVIRONMENT=${1:-$DEFAULT_ENV}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Proxmox
is_proxmox() {
    if [ -f /etc/pve/local/pve-ssl.pem ] || [ -d /etc/pve ]; then
        return 0
    else
        return 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available memory
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
    if (( $(echo "$AVAILABLE_MEMORY < 4.0" | bc -l) )); then
        log_warning "Available memory is ${AVAILABLE_MEMORY}GB. Recommended minimum is 4GB."
        log_warning "Some services may not start properly with limited memory."
    fi
    
    # Check available disk space
    AVAILABLE_DISK=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if (( AVAILABLE_DISK < 20 )); then
        log_warning "Available disk space is ${AVAILABLE_DISK}GB. Recommended minimum is 20GB."
    fi
    
    log_success "System requirements check completed."
}

# Setup environment
setup_environment() {
    log_info "Setting up environment for: $ENVIRONMENT"
    
    # Create environment file
    if [ ! -f .env ]; then
        log_info "Creating environment file..."
        cp .env.example .env
        
        # Generate secure JWT secret
        JWT_SECRET=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        sed -i "s/your-super-secret-key-for-jwt-tokens/$JWT_SECRET/g" .env
        
        log_success "Environment file created. Please review and update API keys in .env"
    else
        log_info "Environment file already exists."
    fi
    
    # Create necessary directories
    log_info "Creating required directories..."
    mkdir -p {neo4j/plugins,postgres/init,prometheus,grafana/provisioning/{dashboards,datasources,notifiers},supabase,localai/config,notebooks,logs,backup,data}
    
    # Set proper permissions
    if is_proxmox; then
        log_info "Setting Proxmox-specific permissions..."
        chown -R 1000:1000 data/ logs/ backup/
        chmod -R 755 data/ logs/ backup/
    fi
}

# Download required plugins and configurations
download_components() {
    log_info "Downloading required components..."
    
    # Neo4j plugins
    log_info "Downloading Neo4j plugins..."
    cd neo4j/plugins
    
    # APOC plugin
    if [ ! -f apoc-5.15.0-core.jar ]; then
        wget -q https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/5.15.0/apoc-5.15.0-core.jar || log_warning "Failed to download APOC plugin"
    fi
    
    # Graph Data Science plugin  
    if [ ! -f neo4j-graph-data-science-2.5.0.jar ]; then
        wget -q https://graphdatascience.ninja/neo4j-graph-data-science-2.5.0.jar || log_warning "Failed to download GDS plugin"
    fi
    
    cd ../..
    
    log_success "Component download completed."
}

# Setup monitoring configuration
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'political-api'
    static_configs:
      - targets: ['kong:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'neo4j'
    static_configs:
      - targets: ['neo4j:7474']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'elasticsearch'
    static_configs:
      - targets: ['elasticsearch:9200']
    metrics_path: '/_prometheus/metrics'
EOF

    # Grafana datasource configuration
    mkdir -p grafana/provisioning/datasources
    cat > grafana/provisioning/datasources/datasources.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
  
  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: political_analysis
    user: postgres
    secureJsonData:
      password: political123
    jsonData:
      sslmode: disable
      maxOpenConns: 0
      maxIdleConns: 2
      connMaxLifetime: 14400

  - name: Elasticsearch
    type: elasticsearch
    url: http://elasticsearch:9200
    database: "[logs-]YYYY.MM.DD"
    jsonData:
      interval: Daily
      timeField: "@timestamp"
      esVersion: 70
EOF

    log_success "Monitoring configuration completed."
}

# Choose deployment method based on environment
deploy_services() {
    log_info "Deploying services for environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        "development"|"local")
            deploy_local
            ;;
        "production"|"proxmox")
            deploy_production
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Available environments: development, local, production, proxmox"
            exit 1
            ;;
    esac
}

# Local development deployment
deploy_local() {
    log_info "Starting local development deployment..."
    
    # Use development docker-compose file
    export COMPOSE_FILE="docker-compose.yml"
    
    # Pull latest images
    log_info "Pulling Docker images..."
    docker compose pull
    
    # Start core services first
    log_info "Starting core services..."
    docker compose up -d postgres redis
    
    # Wait for PostgreSQL
    log_info "Waiting for PostgreSQL to be ready..."
    timeout=60
    while ! docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; do
        if [ $timeout -le 0 ]; then
            log_error "PostgreSQL failed to start within 60 seconds"
            exit 1
        fi
        sleep 2
        timeout=$((timeout-2))
    done
    
    # Start remaining services
    log_info "Starting all services..."
    docker compose up -d
    
    # Wait for services to be ready
    sleep 30
    
    log_success "Local deployment completed!"
    show_service_urls
}

# Production deployment for Proxmox
deploy_production() {
    log_info "Starting production deployment for Proxmox..."
    
    # Use production docker-compose file
    export COMPOSE_FILE="docker-compose.yml:docker-compose.prod.yml"
    
    # Create production overrides if they don't exist
    if [ ! -f docker-compose.prod.yml ]; then
        create_production_compose
    fi
    
    # Pull latest images
    log_info "Pulling Docker images..."
    docker compose pull
    
    # Start with resource limits
    log_info "Starting services with production configuration..."
    docker compose up -d
    
    # Setup systemd service for auto-start
    setup_systemd_service
    
    # Setup backup cron job
    setup_backup_cron
    
    log_success "Production deployment completed!"
    show_service_urls
}

# Create production docker-compose override
create_production_compose() {
    log_info "Creating production configuration..."
    
    cat > docker-compose.prod.yml << EOF
services:
  # Production overrides
  postgres:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    volumes:
      - /opt/political-analysis/data/postgres:/var/lib/postgresql/data
      
  neo4j:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    volumes:
      - /opt/political-analysis/data/neo4j:/data
      - /opt/political-analysis/logs/neo4j:/logs
      
  redis:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    volumes:
      - /opt/political-analysis/data/redis:/data
      
  elasticsearch:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    volumes:
      - /opt/political-analysis/data/elasticsearch:/usr/share/elasticsearch/data
      
  grafana:
    restart: unless-stopped
    volumes:
      - /opt/political-analysis/data/grafana:/var/lib/grafana
      
  prometheus:
    restart: unless-stopped
    volumes:
      - /opt/political-analysis/data/prometheus:/prometheus
      
volumes:
  # Use external volumes for production
  postgres_data:
    external: true
  neo4j_data:
    external: true
  redis_data:
    external: true
  elasticsearch_data:
    external: true
  grafana_data:
    external: true
  prometheus_data:
    external: true
EOF
}

# Setup systemd service for auto-start
setup_systemd_service() {
    if is_proxmox; then
        log_info "Setting up systemd service..."
        
        cat > /etc/systemd/system/political-analysis.service << EOF
[Unit]
Description=Political Analysis Database Stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/political-analysis
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
        
        systemctl daemon-reload
        systemctl enable political-analysis.service
        
        log_success "Systemd service configured."
    fi
}

# Setup backup cron job
setup_backup_cron() {
    if is_proxmox; then
        log_info "Setting up backup cron job..."
        
        # Create backup script
        cat > /opt/political-analysis/scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/political-analysis/backup"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup PostgreSQL
docker compose exec -T postgres pg_dumpall -U postgres > "$BACKUP_DIR/$DATE/postgres_backup.sql"

# Backup Neo4j
docker compose exec -T neo4j neo4j-admin dump --database=neo4j --to=/backup/neo4j_backup_$DATE.dump

# Backup configurations
tar -czf "$BACKUP_DIR/$DATE/config_backup.tar.gz" .env docker-compose.yml prometheus/ grafana/

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: $DATE"
EOF
        
        chmod +x /opt/political-analysis/scripts/backup.sh
        
        # Add to cron (daily at 2 AM)
        (crontab -l 2>/dev/null; echo "0 2 * * * /opt/political-analysis/scripts/backup.sh >> /opt/political-analysis/logs/backup.log 2>&1") | crontab -
        
        log_success "Backup cron job configured."
    fi
}

# Setup AI models
setup_ai_models() {
    log_info "Setting up AI models..."
    
    # Wait for Ollama to be ready
    timeout=120
    while ! docker compose exec ollama ollama list > /dev/null 2>&1; do
        if [ $timeout -le 0 ]; then
            log_warning "Ollama not ready, skipping model setup"
            return
        fi
        sleep 5
        timeout=$((timeout-5))
    done
    
    # Pull models
    log_info "Pulling Ollama models..."
    docker compose exec ollama ollama pull llama2:7b &
    docker compose exec ollama ollama pull mistral:7b &
    docker compose exec ollama ollama pull codellama:7b &
    
    wait
    
    log_success "AI models setup completed."
}

# Install Python dependencies in app container
setup_python_dependencies() {
    log_info "Setting up Python dependencies..."
    
    # Install in a local python environment for development
    if [ "$ENVIRONMENT" = "development" ] || [ "$ENVIRONMENT" = "local" ]; then
        if command -v python3 &> /dev/null && command -v pip3 &> /dev/null; then
            pip3 install -r requirements.txt --user || log_warning "Failed to install Python dependencies locally"
        fi
    fi
    
    log_success "Python dependencies setup completed."
}

# Initialize database
initialize_database() {
    log_info "Initializing database..."
    
    # Wait for PostgreSQL to be ready
    timeout=60
    while ! docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; do
        if [ $timeout -le 0 ]; then
            log_error "PostgreSQL not ready for database initialization"
            return 1
        fi
        sleep 2
        timeout=$((timeout-2))
    done
    
    # Run Python database initialization if available
    if [ -f app/models/database.py ]; then
        python3 -c "
import sys
sys.path.append('.')
try:
    from app.models.database import init_db
    init_db()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
" 2>/dev/null || log_warning "Python database initialization failed"
    fi
    
    log_success "Database initialization completed."
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Check if services are running
    services=("postgres" "neo4j" "redis" "kong" "grafana")
    failed_services=()
    
    for service in "${services[@]}"; do
        if ! docker compose ps | grep -q "${service}.*Up"; then
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log_success "All core services are running."
    else
        log_warning "Failed services: ${failed_services[*]}"
    fi
    
    # Test database connectivity
    if docker compose exec postgres pg_isready -U postgres > /dev/null 2>&1; then
        log_success "PostgreSQL is responding."
    else
        log_warning "PostgreSQL is not responding."
    fi
    
    # Test API endpoint (if available)
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API endpoint is responding."
    else
        log_warning "API endpoint is not responding."
    fi
}

# Show service URLs
show_service_urls() {
    log_success "Political Analysis Database Stack is running!"
    echo ""
    echo "🎯 Service URLs:"
    echo "   • API Server:      http://localhost:8000"
    echo "   • Neo4j Browser:   http://localhost:7474 (neo4j/political123)"
    echo "   • Grafana:         http://localhost:3000 (admin/political123)"
    echo "   • Qdrant:          http://localhost:6333"
    echo "   • OpenWebUI:       http://localhost:3001"
    echo "   • Flowise:         http://localhost:3002"
    echo "   • n8n:             http://localhost:5678 (admin/political123)"
    echo "   • Elasticsearch:   http://localhost:9200"
    echo "   • Kibana:          http://localhost:5601"
    echo "   • Prometheus:      http://localhost:9090"
    echo "   • MinIO:           http://localhost:9001 (political/political123)"
    echo "   • Airflow:         http://localhost:8081 (airflow/political123)"
    echo "   • Jupyter:         http://localhost:8888 (token: political123)"
    echo ""
    echo "📋 Management Commands:"
    echo "   • View logs:       docker compose logs -f [service]"
    echo "   • Stop services:   docker compose down"
    echo "   • Update services: docker compose pull && docker compose up -d"
    echo "   • Backup data:     ./scripts/backup.sh"
    echo ""
    if is_proxmox; then
        echo "🔧 Proxmox Management:"
        echo "   • Service status:  systemctl status political-analysis"
        echo "   • Service logs:    journalctl -u political-analysis -f"
        echo "   • Data location:   /opt/political-analysis/data/"
        echo "   • Backup location: /opt/political-analysis/backup/"
    fi
}

# Cleanup on error
cleanup_on_error() {
    log_error "Deployment failed. Cleaning up..."
    docker compose down
    exit 1
}

# Trap errors
trap cleanup_on_error ERR

# Main deployment flow
main() {
    echo "🏛️  Political Analysis Database Deployment"
    echo "==========================================="
    echo ""
    
    check_requirements
    setup_environment
    download_components
    setup_monitoring
    deploy_services
    
    if [ "$ENVIRONMENT" != "config-only" ]; then
        setup_ai_models
        setup_python_dependencies
        initialize_database
        health_check
    fi
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"