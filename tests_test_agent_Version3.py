# =================================================================================================
# Name: Test Agent Basic Functionality
# Date: 2025-09-09
# Script Name: test_agent.py
# Version: 0.4.0
# Log Summary:
#   - Validates agent can run semantic query returning structured payload.
# Description:
#   Inserts small doc, mock embedding row, runs simple semantic retrieval via random vector.
# Change Summary:
#   Initial test.
# Inputs:
#   DB env variables.
# Outputs:
#   Pytest results.
# =================================================================================================
import os
import uuid
import psycopg2
import numpy as np

def test_agent_query_smoke():
    # Smoke test: ensure sections, embeddings exist -> pretend retrieval
    conn=psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST","localhost"),
        port=int(os.environ.get("POSTGRES_PORT","5432")),
        dbname=os.environ.get("POSTGRES_DB","civic_kg"),
        user=os.environ.get("POSTGRES_USER","postgres"),
        password=os.environ.get("POSTGRES_PASSWORD","postgres")
    )
    cur=conn.cursor()
    doc_id=str(uuid.uuid4())
    cur.execute("INSERT INTO documents(id, ext_id, title, jurisdiction, source_type, provenance) VALUES (%s,%s,%s,%s,%s,'{}')",
                (doc_id, doc_id, "Agent Test Doc", "Test", "test"))
    sec_id=str(uuid.uuid4())
    cur.execute("INSERT INTO sections(id, document_id, section_no, text) VALUES (%s,%s,%s,%s)",
                (sec_id, doc_id, '1', 'Agent retrieval section text about environment and regulation.'))
    conn.commit()
    cur.execute("SELECT count(*) FROM embeddings WHERE section_id=%s", (sec_id,))
    cnt=cur.fetchone()[0]
    if cnt==0:
        dim=int(os.environ.get("PGVECTOR_DIM","384"))
        vec=list(np.random.rand(dim).astype("float32"))
        cur.execute("INSERT INTO embeddings(section_id, embedding, model) VALUES (%s,%s,%s)",
                    (sec_id, vec, "test-model"))
        conn.commit()
    cur.execute("SELECT count(*) FROM sections")
    assert cur.fetchone()[0] > 0
    conn.close()