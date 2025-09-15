# Proxmox Deployment Guide for Political Analysis Database

This guide provides step-by-step instructions for deploying the Political Analysis Database stack on a Proxmox server.

## Prerequisites

### Proxmox Server Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended (16GB+ for full AI capabilities)
- **Storage**: 50GB+ free space
- **Network**: Internet access for initial setup

### Container/VM Specifications
- **OS**: Ubuntu 22.04 LTS or Debian 12
- **CPU**: 4 vCPUs
- **RAM**: 8GB
- **Storage**: 40GB root + 50GB data disk
- **Network**: Bridged network with static IP

## Installation Steps

### 1. Prepare Proxmox Container/VM

#### Option A: LXC Container (Recommended for efficiency)
```bash
# Create privileged LXC container
pct create 200 local:vztmpl/ubuntu-22.04-standard_22.04-1_amd64.tar.zst \
  --cores 4 \
  --memory 8192 \
  --swap 2048 \
  --storage local-lvm \
  --rootfs local-lvm:40 \
  --net0 name=eth0,bridge=vmbr0,ip=192.168.1.100/24,gw=192.168.1.1 \
  --nameserver 8.8.8.8 \
  --hostname political-analysis \
  --password your_password \
  --features keyctl=1,nesting=1

# Add additional storage for data
pct set 200 --mp0 /opt/political-analysis,mp=/opt/political-analysis,size=50G

# Start container
pct start 200
pct enter 200
```

#### Option B: VM (Alternative)
```bash
# Create VM through Proxmox web interface or CLI
qm create 200 \
  --name political-analysis \
  --cores 4 \
  --memory 8192 \
  --scsi0 local-lvm:40 \
  --scsi1 local-lvm:50 \
  --net0 virtio,bridge=vmbr0 \
  --ostype l26 \
  --boot order=scsi0

# Install Ubuntu 22.04 LTS through ISO
```

### 2. Initial System Setup

```bash
# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y curl wget git nano vim htop docker.io docker-compose-plugin python3 python3-pip openssl bc

# Add user to docker group
usermod -aG docker $USER
newgrp docker

# Enable and start Docker
systemctl enable docker
systemctl start docker

# Create application directory
mkdir -p /opt/political-analysis
cd /opt/political-analysis
```

### 3. Deploy Political Analysis Database

```bash
# Clone repository
git clone https://github.com/cbwinslow/political-analysis-dbs.git .

# Run deployment script for production
./deploy.sh production

# Or manually run setup steps:
# ./deploy.sh config-only  # Setup configs only
# docker compose up -d     # Start services
```

### 4. Configure Firewall (Optional but Recommended)

```bash
# Install UFW
apt install -y ufw

# Allow SSH
ufw allow 22

# Allow application ports
ufw allow 8000   # API
ufw allow 3000   # Grafana
ufw allow 7474   # Neo4j
ufw allow 6333   # Qdrant
ufw allow 3001   # OpenWebUI
ufw allow 3002   # Flowise
ufw allow 5678   # n8n
ufw allow 5601   # Kibana
ufw allow 9090   # Prometheus
ufw allow 9001   # MinIO
ufw allow 8081   # Airflow
ufw allow 8888   # Jupyter

# Enable firewall
ufw enable
```

### 5. Setup SSL/TLS (Production)

#### Option A: Using Let's Encrypt with Nginx Proxy
```bash
# Install Nginx
apt install -y nginx certbot python3-certbot-nginx

# Create Nginx configuration
cat > /etc/nginx/sites-available/political-analysis << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /grafana/ {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host \$host;
    }

    location /neo4j/ {
        proxy_pass http://localhost:7474/;
        proxy_set_header Host \$host;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/political-analysis /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# Get SSL certificate
certbot --nginx -d your-domain.com
```

### 6. Configure Monitoring and Alerting

#### Grafana Setup
1. Access Grafana at `http://your-ip:3000`
2. Login with admin/political123
3. Import dashboards from `grafana/dashboards/`
4. Configure alert notifications

#### Prometheus Setup
1. Access Prometheus at `http://your-ip:9090`
2. Verify all targets are UP
3. Create custom alert rules

### 7. Setup Backup Strategy

#### Automated Backups
```bash
# The deployment script automatically sets up:
# - Daily PostgreSQL dumps
# - Neo4j database exports
# - Configuration backups
# - 7-day retention policy

# Manual backup
./scripts/backup.sh

# Restore from backup
./scripts/restore.sh /opt/political-analysis/backup/20240101_120000/
```

#### External Backup Storage
```bash
# Setup Proxmox Backup Server integration
apt install -y proxmox-backup-client

# Configure backup job in Proxmox
# Backup: /opt/political-analysis/data/
```

### 8. Performance Optimization for Proxmox

#### Memory Settings
```bash
# Add to /etc/security/limits.conf
echo "* soft memlock unlimited" >> /etc/security/limits.conf
echo "* hard memlock unlimited" >> /etc/security/limits.conf

# Optimize PostgreSQL for container environment
echo "shared_buffers = 1GB" >> /opt/political-analysis/postgres/postgresql.conf
echo "effective_cache_size = 3GB" >> /opt/political-analysis/postgres/postgresql.conf
```

#### Storage Optimization
```bash
# Use SSD storage for database volumes
# Mount with noatime for better performance
echo "/dev/sdb1 /opt/political-analysis/data ext4 defaults,noatime 0 0" >> /etc/fstab
```

## Service Management

### Systemd Service Control
```bash
# Service status
systemctl status political-analysis

# Start/stop services
systemctl start political-analysis
systemctl stop political-analysis

# View logs
journalctl -u political-analysis -f

# Restart all containers
systemctl restart political-analysis
```

### Docker Management
```bash
# View running containers
docker compose ps

# View logs for specific service
docker compose logs -f postgres

# Scale services
docker compose up -d --scale airflow-worker=3

# Update services
docker compose pull
docker compose up -d
```

## Maintenance Tasks

### Daily Tasks
- [x] Automated backups via cron
- [x] Log rotation
- [x] Health checks

### Weekly Tasks
```bash
# Update system packages
apt update && apt upgrade -y

# Update Docker images
cd /opt/political-analysis
docker compose pull
docker compose up -d

# Clean up unused Docker resources
docker system prune -f
```

### Monthly Tasks
```bash
# Review backup retention
find /opt/political-analysis/backup -type f -mtime +30 -delete

# Review logs
find /opt/political-analysis/logs -type f -mtime +30 -delete

# Update SSL certificates (if using Let's Encrypt)
certbot renew
```

## Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check container logs
docker compose logs [service-name]

# Check system resources
htop
df -h

# Restart specific service
docker compose restart [service-name]
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker compose exec postgres pg_isready -U postgres

# Connect to database
docker compose exec postgres psql -U postgres -d political_analysis

# Check database logs
docker compose logs postgres
```

#### Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Reduce service memory limits in docker-compose.yml
# Restart services with new limits
docker compose up -d
```

### Log Locations
- Application logs: `/opt/political-analysis/logs/`
- Docker logs: `docker compose logs [service]`
- System logs: `/var/log/syslog`
- Backup logs: `/opt/political-analysis/logs/backup.log`

## Security Considerations

### Network Security
- Use firewall to restrict access to necessary ports only
- Consider VPN access for administrative interfaces
- Regular security updates

### Data Security
- Encrypt backups
- Use strong passwords (change defaults!)
- Enable audit logging
- Regular security assessments

### Access Control
- Change default passwords immediately
- Use SSH keys instead of passwords
- Implement proper user management
- Regular access reviews

## Support and Updates

### Getting Help
1. Check logs for error messages
2. Review troubleshooting section
3. Check GitHub issues
4. Contact support team

### Staying Updated
1. Watch GitHub repository for updates
2. Subscribe to security notifications
3. Regular backup testing
4. Performance monitoring

## Performance Tuning

### For Proxmox LXC Containers
```bash
# Increase container limits if needed
pct set 200 --memory 16384 --cores 8

# Add more storage
pct set 200 --mp1 /backup,mp=/opt/political-analysis/backup,size=100G
```

### For Production Workloads
```bash
# Optimize PostgreSQL
# Edit postgresql.conf for your workload
# Tune kernel parameters
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' >> /etc/sysctl.conf
sysctl -p
```

This guide should help you successfully deploy and manage the Political Analysis Database on your Proxmox server. Remember to customize the configuration for your specific environment and requirements.