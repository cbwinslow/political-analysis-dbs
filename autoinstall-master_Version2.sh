#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Name:         autoinstall-master.sh
# Date:         2025-09-09
# Script name:  autoinstall-master.sh
# Version:      0.2.0
# Log summary:  Comprehensive Ubuntu bootstrap for homelab/dev workstation
# Description:
#   Single-file master installer that bootstraps a reproducible environment for
#   development, telemetry, AI, monitoring, and self-hosted services on Ubuntu.
#   Installs (selectively) Node (via nvm), npm, Docker CE + Compose plugin,
#   rootless Docker option, Homebrew (Linux), VS Code (stable + optional Insiders),
#   chezmoi + dotfiles bootstrap, ZeroTier join, IDS (Snort/Suricata best-effort),
#   monitoring stacks (Prometheus, Grafana, Loki, Alertmanager, node_exporter),
#   Supabase self-host skeleton, ClickHouse, InfluxDB, OpenSearch/Graylog skeleton,
#   RabbitMQ, LocalAI/OpenWebUI skeleton, Keycloak (simple OAuth provider),
#   SSH key creation & optional upload to GitHub/GitLab, and a small orchestrator
#   agent that posts host telemetry to Supabase (or prints to stdout).
#
# Change summary:
#   v0.2.0 - Added ClickHouse/InfluxDB, Keycloak OAuth scaffold, GitHub key upload,
#            improved orchestration agent & more complete docker-compose skeletons.
#
# Inputs (environment variables or interactive prompts):
#   DOTFILES_REPO      - git URL to your dotfiles (default: https://github.com/cbwinslow/dotfiles)
#   DOTFILES_BRANCH    - branch to use (default: main)
#   SUPABASE_URL       - URL to your Supabase insert endpoint (optional)
#   SUPABASE_KEY       - API key for Supabase REST (optional)
#   ZEROTIER_NETWORK   - ZeroTier network ID to join (optional)
#   INSTALL_PROFILE    - comma-separated list of components to install (default below)
#   NONINTERACTIVE     - set to 1 to auto-accept prompts
#   GITHUB_TOKEN       - (optional) Personal Access Token to upload SSH key to GitHub
#   GITLAB_TOKEN       - (optional) Personal Access Token to upload SSH key to GitLab
#
# Outputs:
#   - /opt/autoinstall populated with docker-compose stacks and configs
#   - systemd service "autoinstall-orchestrator.service" (enabled)
#   - chezmoi applied dotfiles (if DOTFILES_REPO provided)
#   - optionally uploaded SSH public key to GitHub/GitLab
# -----------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# ----- Configuration defaults -----
DOTFILES_REPO="${DOTFILES_REPO:-https://github.com/cbwinslow/dotfiles}"
DOTFILES_BRANCH="${DOTFILES_BRANCH:-main}"
SUPABASE_URL="${SUPABASE_URL:-}"
SUPABASE_KEY="${SUPABASE_KEY:-}"
ZEROTIER_NETWORK="${ZEROTIER_NETWORK:-}"
INSTALL_PROFILE="${INSTALL_PROFILE:-base,node,docker,devtools,monitoring,ids,chezmoi,ai,datastores,oauth,ssh}"
NONINTERACTIVE="${NONINTERACTIVE:-0}"
WORKDIR="/opt/autoinstall"
AGENT_INTERVAL="${AGENT_INTERVAL:-60}"   # seconds

mkdir -p "$WORKDIR"
info() { printf "\e[34m[INFO]\e[0m %s\n" "$*"; }
warn() { printf "\e[33m[WARN]\e[0m %s\n" "$*"; }
err()  { printf "\e[31m[ERROR]\e[0m %s\n" "$*"; }
confirm() {
  if [ "$NONINTERACTIVE" = "1" ]; then return 0; fi
  read -r -p "$1 [y/N]: " ans
  case "$ans" in [Yy]*) return 0;; *) return 1;; esac
}

# Detect distro
if [ -f /etc/os-release ]; then . /etc/os-release; fi
info "Detected OS: ${ID:-unknown} ${VERSION_ID:-unknown}"
if [ "$EUID" -ne 0 ]; then err "Run this script as root (sudo)"; exit 1; fi

# ----- Helpers -----
run_as_user() {
  local cmd="$1"
  local user="${SUDO_USER:-$USER}"
  su - "$user" -c "$cmd"
}

# ----- Install base packages -----
install_base() {
  info "Installing base apt packages..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl wget git build-essential ca-certificates apt-transport-https gnupg lsb-release \
    software-properties-common jq unzip python3 python3-venv python3-pip apt-transport-https \
    net-tools socat dnsutils
  info "Base packages installed."
}

# ----- Homebrew (Linux) -----
install_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    info "Homebrew already installed"
    return
  fi
  info "Installing Homebrew (Linux)..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || warn "Homebrew installer failed"
  if [ -d /home/linuxbrew/.linuxbrew/bin ]; then eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"; fi
}

# ----- nvm + node -----
install_nvm_node() {
  local NVM_DIR="/home/${SUDO_USER:-$USER}/.nvm"
  if [ -d "$NVM_DIR" ]; then info "nvm appears present"; else
    info "Installing nvm for user ${SUDO_USER:-$USER}..."
    run_as_user "bash -lc 'curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.6/install.sh | bash'"
  fi
  info "Installing Node LTS via nvm for user..."
  run_as_user "bash -lc 'export NVM_DIR=\"\$HOME/.nvm\"; [ -s \"\$NVM_DIR/nvm.sh\" ] && . \"\$NVM_DIR/nvm.sh\"; nvm install --lts; nvm alias default lts/*'"
  info "Node installed (if nvm finished)."
}

# ----- Docker CE & Compose plugin + rootless option -----
install_docker() {
  if command -v docker >/dev/null 2>&1; then
    info "Docker already installed"
  else
    info "Installing Docker CE..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io
    usermod -aG docker "${SUDO_USER:-$USER}"
    info "Docker installed. User '${SUDO_USER:-$USER}' added to docker group. Re-login needed for group to take effect."
  fi

  if docker compose version >/dev/null 2>&1; then
    info "Docker Compose plugin present"
  else
    info "Installing Docker Compose plugin..."
    DOCKER_CLI_PLUGINS_DIR=/usr/libexec/docker/cli-plugins
    mkdir -p "$DOCKER_CLI_PLUGINS_DIR"
    curl -sSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" -o "$DOCKER_CLI_PLUGINS_DIR/docker-compose"
    chmod +x "$DOCKER_CLI_PLUGINS_DIR/docker-compose"
  fi

  if confirm "Enable Docker rootless mode for user ${SUDO_USER:-$USER}? (recommended)"; then
    info "Attempting Docker rootless setup (best-effort)..."
    apt-get install -y uidmap
    run_as_user "bash -lc 'curl -fsSL https://get.docker.com/rootless | bash' || true"
    info "Docker rootless attempted."
  fi
}

# ----- VS Code -----
install_vscode() {
  if ! command -v code >/dev/null 2>&1; then
    info "Installing VS Code (stable)..."
    wget -qO /tmp/vscode.deb "https://update.code.visualstudio.com/latest/linux-deb-x64/stable"
    apt-get install -y /tmp/vscode.deb || apt-get -f install -y
    rm -f /tmp/vscode.deb
  else
    info "VS Code already installed"
  fi
  if confirm "Install VS Code Insiders too?"; then
    if ! command -v code-insiders >/dev/null 2>&1; then
      info "Installing VS Code Insiders..."
      wget -qO /tmp/vscode-insiders.deb "https://update.code.visualstudio.com/latest/linux-deb-x64/insider"
      apt-get install -y /tmp/vscode-insiders.deb || apt-get -f install -y
      rm -f /tmp/vscode-insiders.deb
    else
      info "VS Code Insiders already present"
    fi
  fi
}

# ----- chezmoi + dotfiles bootstrap -----
install_chezmoi_and_dotfiles() {
  if ! command -v chezmoi >/dev/null 2>&1; then
    info "Installing chezmoi..."
    sh -c "$(curl -fsLS get.chezmoi.io)" -- -b /usr/local/bin
  else
    info "chezmoi already installed"
  fi

  if [ -n "$DOTFILES_REPO" ]; then
    info "Bootstrapping chezmoi from $DOTFILES_REPO (branch $DOTFILES_BRANCH)"
    run_as_user "chezmoi init --apply --branch=${DOTFILES_BRANCH} ${DOTFILES_REPO} || true"
    info "chezmoi applied (or attempted). Run 'chezmoi doctor' as the user to verify."
  else
    warn "No DOTFILES_REPO provided; skipping dotfiles bootstrap."
  fi
}

# ----- ZeroTier -----
install_zerotier() {
  if ! command -v zerotier-cli >/dev/null 2>&1; then
    info "Installing ZeroTier..."
    curl -s https://install.zerotier.com | bash || warn "ZeroTier installer failed"
  else
    info "ZeroTier already installed"
  fi
  if [ -n "$ZEROTIER_NETWORK" ]; then
    info "Joining ZeroTier network $ZEROTIER_NETWORK"
    run_as_user "zerotier-cli join $ZEROTIER_NETWORK" || warn "ZeroTier join failed. Authorize node in ZeroTier Central if required."
  fi
}

# ----- IDS (Snort, Suricata) best-effort -----
install_ids() {
  info "Installing IDS packages (best-effort). You will need to tune rules manually."
  DEBIAN_FRONTEND=noninteractive apt-get install -y snort suricata || warn "Some IDS packages failed to install; manual install may be required"
  info "Snort/Suricata installed (if available)."
}

# ----- SSH key generation and optional upload to GitHub/GitLab -----
setup_ssh_keys_and_upload() {
  local user="${SUDO_USER:-$USER}"
  local sshdir="/home/$user/.ssh"
  local keyfile="$sshdir/id_ed25519_autoinstall"
  mkdir -p "$sshdir"
  chown "$user:$user" "$sshdir"
  chmod 700 "$sshdir"

  if [ ! -f "$keyfile" ]; then
    info "Generating new ed25519 SSH key at $keyfile for user $user"
    run_as_user "ssh-keygen -t ed25519 -f $keyfile -N '' -C 'autoinstall@$HOSTNAME'"
  else
    info "SSH key already exists at $keyfile"
  fi

  local pub=$(cat "$keyfile.pub")
  info "Public key:\n$pub"

  if [ -n "${GITHUB_TOKEN:-}" ]; then
    info "Uploading public key to GitHub account (using GITHUB_TOKEN)"
    local ghuser
    ghuser=$(curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | jq -r .login)
    if [ -n "$ghuser" ] && [ "$ghuser" != "null" ]; then
      curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Content-Type: application/json" \
        -d "{\"title\":\"autoinstall-$(hostname)-$(date +%s)\",\"key\":\"$pub\"}" \
        https://api.github.com/user/keys || warn "GitHub key upload may have failed"
      info "Attempted GitHub upload for user $ghuser"
    else
      warn "Could not determine GitHub username from token"
    fi
  fi

  if [ -n "${GITLAB_TOKEN:-}" ]; then
    info "Uploading public key to GitLab (using GITLAB_TOKEN)"
    curl -s -X POST -H "PRIVATE-TOKEN: $GITLAB_TOKEN" -H "Content-Type: application/json" \
      -d "{\"title\":\"autoinstall-$(hostname)-$(date +%s)\",\"key\":\"$pub\"}" \
      https://gitlab.com/api/v4/user/keys || warn "GitLab key upload may have failed"
    info "Attempted GitLab upload"
  fi

  info "You can now use the private key at $keyfile to SSH into other hosts (authorize as needed)."
}

# ----- Docker Compose stacks: monitoring, datastores, oauth, ai, search, logs -----
create_compose_files() {
  info "Creating docker-compose stacks under $WORKDIR"

  # Monitoring stack
  mkdir -p "$WORKDIR/monitoring"
  cat > "$WORKDIR/monitoring/docker-compose.yml" <<'YAML'
version: "3.8"
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/config.yml:ro
    ports:
      - "9093:9093"
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    network_mode: "host"
    restart: unless-stopped

  cadvisor:
    image: gcr.io/google-containers/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8080:8080"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "admin"
    ports:
      - "3000:3000"
    restart: unless-stopped

  loki:
    image: grafana/loki:2.8.2
    ports:
      - "3100:3100"
    restart: unless-stopped
YAML

  cat > "$WORKDIR/monitoring/prometheus.yml" <<'YAML'
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['host.docker.internal:8080','127.0.0.1:8080']
  - job_name: 'node'
    static_configs:
      - targets: ['127.0.0.1:9100']
YAML

  cat > "$WORKDIR/monitoring/alertmanager.yml" <<'YAML'
global: {}
route:
  receiver: 'null'
receivers: []
YAML

  # Datastores stack: Supabase (skeleton), ClickHouse, InfluxDB
  mkdir -p "$WORKDIR/datastores"
  cat > "$WORKDIR/datastores/docker-compose.yml" <<'YAML'
version: "3.8"
services:
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    volumes:
      - clickhouse-data:/var/lib/clickhouse
    ports:
      - "8123:8123"
      - "9000:9000"
    restart: unless-stopped

  influxdb:
    image: influxdb:2.7
    environment:
      - INFLUXDB_HTTP_AUTH_ENABLED=true
      - INFLUXDB_INIT_MODE=setup
      - INFLUXDB_INIT_USERNAME=admin
      - INFLUXDB_INIT_PASSWORD=adminpass
      - INFLUXDB_INIT_ORG=myorg
      - INFLUXDB_INIT_BUCKET=mybucket
    volumes:
      - influxdb-data:/var/lib/influxdb2
    ports:
      - "8086:8086"
    restart: unless-stopped

  # Minimal Postgres for Supabase skeleton (not full supabase platform)
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=example
    volumes:
      - pg-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  clickhouse-data:
  influxdb-data:
  pg-data:
YAML

  # OAuth stack: Keycloak (simple identity provider) for "make our own oauth site"
  mkdir -p "$WORKDIR/oauth"
  cat > "$WORKDIR/oauth/docker-compose.yml" <<'YAML'
version: "3.8"
services:
  keycloak:
    image: quay.io/keycloak/keycloak:21.1.1
    environment:
      KC_DB: "h2"
      KC_HTTP_RELATIVE_PATH: ""
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin
    command: start-dev
    ports:
      - "8081:8080"
    restart: unless-stopped
YAML

  # AI stack: LocalAI + OpenWebUI skeleton
  mkdir -p "$WORKDIR/ai"
  cat > "$WORKDIR/ai/docker-compose.yml" <<'YAML'
version: "3.8"
services:
  localai:
    image: localai/localai:latest
    environment:
      - LLM_MODEL=ggml-alpaca-7b-q4.bin
    ports:
      - "8080:8080"
    restart: unless-stopped

  openwebui:
    image: localai/open-webui:latest
    ports:
      - "3001:3001"
    restart: unless-stopped
YAML

  # Search & logs skeleton (OpenSearch + Graylog)
  mkdir -p "$WORKDIR/search_logs"
  cat > "$WORKDIR/search_logs/docker-compose.yml" <<'YAML'
version: "3.8"
services:
  opensearch:
    image: opensearchproject/opensearch:2.10.0
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    ports:
      - "9200:9200"
    restart: unless-stopped

  graylog:
    image: graylog/graylog:5
    environment:
      - GRAYLOG_PASSWORD_SECRET=somepasswordpepper
      - GRAYLOG_ROOT_PASSWORD_SHA2=2bb80d537b1da3e38bd30361aa855686bde0ba7f (example)
      - GRAYLOG_HTTP_EXTERNAL_URI=http://127.0.0.1:9000/
    depends_on:
      - opensearch
    ports:
      - "9000:9000"
    restart: unless-stopped

volumes:
  opensearch-data:
YAML

  # RabbitMQ skeleton
  mkdir -p "$WORKDIR/rabbitmq"
  cat > "$WORKDIR/rabbitmq/docker-compose.yml" <<'YAML'
version: "3.8"
services:
  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    ports:
      - "5672:5672"
      - "15672:15672"
    restart: unless-stopped
YAML

  info "Docker-compose skeletons created in $WORKDIR. Edit each compose file before starting in production."
}

# ----- Sentry onprem skeleton (placeholder to clone onpremise installer) -----
create_sentry_placeholder() {
  info "Creating placeholder for Sentry onpremise installer under $WORKDIR/sentry"
  mkdir -p "$WORKDIR/sentry"
  cat > "$WORKDIR/sentry/README.md" <<'MD'
# Sentry onpremise bootstrap (placeholder)
This directory is a placeholder. To self-host Sentry, follow official onpremise:
https://github.com/getsentry/onpremise
You can clone that repo here and run ./install.sh inside.
MD
}

# ----- Orchestrator agent (single-file Python) and systemd unit -----
create_orchestrator_agent() {
  info "Creating orchestrator agent at $WORKDIR/agent"
  mkdir -p "$WORKDIR/agent"
  cat > "$WORKDIR/agent/orchestrator.py" <<PY
#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Name:         orchestrator.py
# Date:         2025-09-09
# Script name:  orchestrator.py
# Version:      0.1.0
# Log summary:  Small host telemetry agent for autoinstall stack
# Description:
#   Collects basic host metrics and posts to SUPABASE_URL if provided.
#   Falls back to printing JSON to stdout if no endpoint is configured.
# Inputs:
#   SUPABASE_URL  - full URL to POST telemetry payloads (optional)
#   SUPABASE_KEY  - API key for Supabase (optional)
#   ORCHESTRATOR_INTERVAL - seconds between posts (default 60)
# Outputs:
#   - HTTP POST to SUPABASE_URL or JSON lines printed to stdout
# -----------------------------------------------------------------------------
import os, time, json, socket, subprocess, platform
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
INTERVAL = int(os.environ.get("ORCHESTRATOR_INTERVAL", "60"))

def get_uptime():
    try:
        with open("/proc/uptime") as f:
            return float(f.readline().split()[0])
    except:
        return None

def get_mem():
    d={}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k,v = line.split(":",1)
                d[k.strip()] = v.strip()
    except:
        pass
    return d

def get_docker_info():
    try:
        p = subprocess.run(["docker","info","--format","{{json .}}"], capture_output=True, text=True, check=True)
        return json.loads(p.stdout)
    except:
        return {}

def main():
    hostname = socket.gethostname()
    while True:
        payload = {
            "host": hostname,
            "platform": platform.platform(),
            "uptime_seconds": get_uptime(),
            "mem": {k: v for k,v in get_mem().items() if k in ("MemTotal","MemAvailable")},
            "docker": get_docker_info(),
            "ts": int(time.time())
        }
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                headers = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
                requests.post(SUPABASE_URL, json=payload, headers=headers, timeout=10)
            except Exception:
                pass
        else:
            print(json.dumps(payload))
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
PY

  chmod +x "$WORKDIR/agent/orchestrator.py"
  cat > "$WORKDIR/agent/requirements.txt" <<REQ
requests
REQ

  python3 -m venv "$WORKDIR/agent/venv" || true
  "$WORKDIR/agent/venv/bin/pip" install -r "$WORKDIR/agent/requirements.txt" || true

  cat > /etc/systemd/system/autoinstall-orchestrator.service <<UNIT
[Unit]
Description=Autoinstall Orchestrator Agent
After=network.target

[Service]
Type=simple
User=${SUDO_USER:-$USER}
Environment=SUPABASE_URL=${SUPABASE_URL}
Environment=SUPABASE_KEY=${SUPABASE_KEY}
Environment=ORCHESTRATOR_INTERVAL=${AGENT_INTERVAL}
ExecStart=/usr/bin/env python3 $WORKDIR/agent/orchestrator.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

  systemctl daemon-reload
  systemctl enable --now autoinstall-orchestrator.service || warn "Could not enable orchestrator service"
  info "Orchestrator installed and started (if possible)."
}

# ----- Install Local tools (qwen-coder/codex/opencode etc. best-effort) -----
install_dev_tools() {
  info "Installing common dev tools and language helpers..."
  # Install pipx and python tools
  python3 -m pip install --upgrade pip
  python3 -m pip install --user pipx || true
  run_as_user "python3 -m pipx ensurepath" || true

  # Attempt to install code-server or local wrappers if desired - left as optional
  info "Dev tools installed (minimal). Add qwen-coder/local packages inside docker or via pip/npm as needed."
}

# ----- Apply user-selected profile -----
IFS=',' read -r -a PROFILE_ITEMS <<< "$INSTALL_PROFILE"
for item in "${PROFILE_ITEMS[@]}"; do
  case "$item" in
    base) install_base ;;
    node) install_nvm_node ;;
    docker) install_docker ;;
    devtools) install_vscode; install_homebrew; install_dev_tools ;;
    monitoring) create_compose_files ;;
    ids) install_ids ;;
    chezmoi) install_chezmoi_and_dotfiles ;;
    zerotier) install_zerotier ;;
    ai) create_compose_files ;; # ai compose created in create_compose_files
    datastores) create_compose_files ;;
    oauth) create_compose_files ;;
    ssh) setup_ssh_keys_and_upload ;;
    agent) create_orchestrator_agent ;;
    *) warn "Unknown profile item: $item" ;;
  esac
done

# Ensure orchestrator exists & started
if [ ! -f /etc/systemd/system/autoinstall-orchestrator.service ]; then
  create_orchestrator_agent
fi

# Create additional placeholders
create_sentry_placeholder

# Final summary
cat <<SUMMARY

Autoinstall bootstrap finished.

What I created and configured (locations):
- Working directory: $WORKDIR
  - monitoring/        -> Prometheus, Grafana, Loki, Alertmanager skeleton
  - datastores/        -> ClickHouse, InfluxDB, Postgres skeleton
  - oauth/             -> Keycloak (simple OAuth provider) skeleton
  - ai/                -> LocalAI + OpenWebUI skeleton
  - search_logs/       -> OpenSearch + Graylog skeleton
  - rabbitmq/          -> RabbitMQ management
  - sentry/            -> placeholder for onpremise installer
  - agent/             -> orchestrator agent + venv + systemd service

Next manual steps I recommend (summary):
- Inspect and customize each docker-compose.yml in /opt/autoinstall before running with `docker compose up -d`.
- For Supabase self-host: follow official repo https://github.com/supabase/supabase to deploy full stack.
- If you uploaded SSH keys via GITHUB_TOKEN/GITLAB_TOKEN, verify they appear in your account settings.
- Re-login (or reboot) so docker group membership takes effect for your user.
- Customize Keycloak (http://localhost:8081) to register clients (e.g., Grafana, your internal services).
- Tune Snort/Suricata rules and directories.
- Configure backups for /opt/autoinstall and your dotfiles repo. Consider encrypting secrets with chezmoi + age/gpg.

Useful commands:
- Start monitoring stack: cd /opt/autoinstall/monitoring && docker compose up -d
- Start datastores: cd /opt/autoinstall/datastores && docker compose up -d
- Start oauth (Keycloak): cd /opt/autoinstall/oauth && docker compose up -d
- Start AI stack: cd /opt/autoinstall/ai && docker compose up -d

If you want, I will:
- (A) Expand the Supabase stack to a working self-hosted manifest (automated env file creation + install)
- (B) Create separate versioned repos/gists per service and a Cloudflare Pages / GitHub Pages site with one-click install profiles
- (C) Implement a central OAuth + onboarding web app to manage ZeroTier join + SSH key distribution (requires a small web app + DB)
- (D) Harden IDS/monitoring and create automated rules update and alerting

Pick one of A/B/C/D and I'll implement it next (I can create the repo or a single-file script per your preference). 

SUMMARY