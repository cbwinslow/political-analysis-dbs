# =================================================================================================
# Name: Pull Request Template
# Date: 2025-09-09
# Script Name: pull_request_template.md
# Version: 0.5.0
# Log Summary: Standard PR checklist for code quality.
# Description: Guides contributors through QA steps.
# Change Summary: Initial version.
# Inputs: PR metadata
# Outputs: Reviewed changes
# =================================================================================================
## Summary
Describe the changes and motivation.

## Changes
- Bullet list of major updates.

## Testing
- [ ] Ran `python civic_legis_unified.py --run-self-tests`
- [ ] Manual ingestion test (if applicable)
- [ ] Verified embeddings (if applicable)

## Security
- Any new secrets or tokens? Explain.

## Deployment
- Steps needed for deployment (if different from README).

## Checklist
- [ ] Code style / formatting
- [ ] No hardcoded secrets
- [ ] Updated README / docs if needed
- [ ] Added integration hooks if new service included