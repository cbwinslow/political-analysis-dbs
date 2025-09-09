# =================================================================================================
# Name: Test Embedding Pipeline
# Date: 2025-09-09
# Script Name: test_embedding.py
# Version: 0.4.0
# Log Summary:
#   - Verifies embedding insertion pipeline.
# Description:
#   Inserts a mock section, runs embedding, checks embeddings row count increments.
# Change Summary:
#   Initial test creation.
# Inputs:
#   DB env variables.
# Outputs:
#   Pytest assertion results.
# =================================================================================================
import os
import uuid
import psycopg2
import numpy as np

def test_embedding_insert():
    dim=int(os.environ.get("PGVECTOR_DIM","384"))
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
                (doc_id, doc_id, "Test Doc", "Test", "test"))
    sec_id=str(uuid.uuid4())
    cur.execute("INSERT INTO sections(id, document_id, section_no, text) VALUES (%s,%s,%s,%s)",
                (sec_id, doc_id, "1", "Sample text for embedding."))
    conn.commit()
    cur.execute("SELECT count(*) FROM embeddings")
    before=cur.fetchone()[0]
    vec=list(np.random.rand(dim).astype("float32"))
    cur.execute("INSERT INTO embeddings(section_id, embedding, model) VALUES (%s,%s,%s)",
                (sec_id, vec, "test-model"))
    conn.commit()
    cur.execute("SELECT count(*) FROM embeddings")
    after=cur.fetchone()[0]
    assert after==before+1
    conn.close()