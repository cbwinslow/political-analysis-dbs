# =================================================================================================
# Name: Test Schema Initialization
# Date: 2025-09-09
# Script Name: test_schema.py
# Version: 0.4.0
# Log Summary:
#   - Basic tests for database schema existence.
# Description:
#   Verifies core tables exist after initialization.
# Change Summary:
#   Initial test suite.
# Inputs:
#   Environment variables for DB connection.
# Outputs:
#   Pytest pass/fail.
# =================================================================================================
import os
import psycopg2

def test_tables_exist():
    conn=psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST","localhost"),
        port=int(os.environ.get("POSTGRES_PORT","5432")),
        dbname=os.environ.get("POSTGRES_DB","civic_kg"),
        user=os.environ.get("POSTGRES_USER","postgres"),
        password=os.environ.get("POSTGRES_PASSWORD","postgres")
    )
    cur=conn.cursor()
    needed=["documents","sections","embeddings","bills","politicians","votes","vote_choices","politician_profiles"]
    for table in needed:
        cur.execute("SELECT to_regclass(%s)",(table,))
        assert cur.fetchone()[0]==table, f"Missing table {table}"
    conn.close()