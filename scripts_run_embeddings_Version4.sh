#!/usr/bin/env bash
# =================================================================================================
# Name: Embedding Runner
# Date: 2025-09-09
# Script Name: run_embeddings.sh
# Version: 0.4.0
# Log Summary:
#   - Standalone embedding refresh script.
# Description:
#   Recomputes embeddings for new sections only.
# Change Summary:
#   Initial version.
# Inputs:
#   EMBED_MODEL env var optional.
# Outputs:
#   New rows in embeddings table.
# =================================================================================================
set -euo pipefail
python civic_legis_hub.py --embed