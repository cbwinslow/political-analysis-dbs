#!/usr/bin/env bash
# =================================================================================================
# Name: Ingestion Runner
# Date: 2025-09-09
# Script Name: run_ingestion.sh
# Version: 0.4.0
# Log Summary:
#   - Convenience script to run ingestion and embedding.
# Description:
#   Wraps civic_legis_hub.py for quick CI / cron usage.
# Change Summary:
#   Initial version.
# Inputs:
#   Environment variables (DB, API keys) must be exported.
# Outputs:
#   Updated database state.
# =================================================================================================
set -euo pipefail
python civic_legis_hub.py \
  --sync-govinfo --govinfo-collections BILLSTATUS \
  --sync-openstates --openstates-states "California,New York" \
  --propublica-sync --congress 118 --propublica-chambers house,senate \
  --embed --build-profiles