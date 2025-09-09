-- =================================================================================================
-- Name: Sample Analytical Queries
-- Date: 2025-09-09
-- Script Name: sample_queries.sql
-- Version: 0.4.0
-- Log Summary:
--   - Provides example SQL queries for analytics & profiling.
-- Description:
--   Ready-to-run queries for exploring legislative corpus & votes.
-- Change Summary:
--   Initial version.
-- Inputs:
--   PostgreSQL schema from civic_legis_hub.py
-- Outputs:
--   Result sets demonstrating usage.
-- =================================================================================================

-- Top 20 most recently inserted bills
SELECT bill_id, title, jurisdiction, created_at
FROM bills
ORDER BY created_at DESC
LIMIT 20;

-- Politicians with highest number of YEA votes
SELECT p.politician_id, p.name,
       SUM(CASE WHEN vc.choice='YEA' THEN 1 ELSE 0 END) AS yea_votes,
       COUNT(vc.choice) AS total_votes
FROM politicians p
LEFT JOIN vote_choices vc ON vc.politician_id = p.politician_id
GROUP BY p.politician_id, p.name
ORDER BY yea_votes DESC
LIMIT 20;

-- Bill vote breakdown
SELECT v.bill_id,
       SUM(CASE WHEN vc.choice='YEA' THEN 1 END) AS total_yea,
       SUM(CASE WHEN vc.choice='NAY' THEN 1 END) AS total_nay,
       COUNT(vc.choice) AS total_votes
FROM votes v
LEFT JOIN vote_choices vc ON vc.vote_id = v.vote_id
GROUP BY v.bill_id
ORDER BY total_votes DESC
LIMIT 30;