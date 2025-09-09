# =================================================================================================
# Name: SECURITY
# Date: 2025-09-09
# Script Name: SECURITY.md
# Version: 0.5.0
# Log Summary: Security disclosure & hardening guidelines.
# Description: Provides procedures for vulnerability reporting.
# Change Summary: Initial version.
# Inputs: N/A
# Outputs: Responsible disclosure path.
# =================================================================================================
## Reporting a Vulnerability
Please open a private security advisory or email the maintainer (see repository profile). Do not open a public issue containing exploit details.

## Hardening Recommendations
- Use read-only database role for API runtime.
- Rotate API keys (govinfo, OpenStates, ProPublica, Cloudflare) regularly.
- Enforce HTTPS at edge (Cloudflare proxy).
- Restrict inbound ports to necessary services (80/443/8100, 5432 internal).
- Consider adding rate limiting / auth for write endpoints if exposed publicly.

## Data Sensitivity
Legislative texts are public domain (generally), but aggregated or enriched data might carry analytic value—avoid unintentional exposure of internal notes or pipeline logs.