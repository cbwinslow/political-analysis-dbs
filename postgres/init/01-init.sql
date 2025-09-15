-- Enable extensions for political analysis
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS storage;
CREATE SCHEMA IF NOT EXISTS realtime;
CREATE SCHEMA IF NOT EXISTS graphql_public;

-- Create roles
CREATE ROLE anon NOLOGIN NOINHERIT;
CREATE ROLE authenticated NOLOGIN NOINHERIT;
CREATE ROLE service_role NOLOGIN NOINHERIT;
CREATE ROLE supabase_auth_admin NOLOGIN NOINHERIT;
CREATE ROLE supabase_storage_admin NOLOGIN NOINHERIT;
CREATE ROLE supabase_admin NOLOGIN NOINHERIT;
CREATE ROLE authenticator NOINHERIT LOGIN PASSWORD 'political123';
CREATE ROLE airflow LOGIN PASSWORD 'political123';

-- Grant permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated, service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated, service_role;

-- Set up authenticator
GRANT anon, authenticated, service_role TO authenticator;

-- Create airflow database
CREATE DATABASE airflow_db OWNER airflow;

-- Core political analysis tables
CREATE TABLE IF NOT EXISTS legislators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    party TEXT,
    state TEXT,
    district TEXT,
    chamber TEXT CHECK (chamber IN ('house', 'senate')),
    bio_embedding vector(1536),
    voting_record_embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    summary TEXT,
    full_text TEXT,
    bill_number TEXT UNIQUE,
    status TEXT,
    introduced_date DATE,
    content_embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS votes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    legislator_id UUID REFERENCES legislators(id),
    bill_id UUID REFERENCES bills(id),
    vote TEXT CHECK (vote IN ('yes', 'no', 'abstain', 'present')),
    vote_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS committees (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    chamber TEXT CHECK (chamber IN ('house', 'senate', 'joint')),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS committee_memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    legislator_id UUID REFERENCES legislators(id),
    committee_id UUID REFERENCES committees(id),
    role TEXT DEFAULT 'member',
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS speeches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    legislator_id UUID REFERENCES legislators(id),
    content TEXT NOT NULL,
    date TIMESTAMP WITH TIME ZONE,
    context TEXT,
    content_embedding vector(1536),
    sentiment_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lobbying_activities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization TEXT NOT NULL,
    amount DECIMAL(15,2),
    description TEXT,
    target_legislators UUID[],
    related_bills UUID[],
    date_reported DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS political_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    entity_type TEXT CHECK (entity_type IN ('person', 'organization', 'location', 'event')),
    description TEXT,
    confidence_score FLOAT,
    source_document_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity1_id UUID REFERENCES political_entities(id),
    entity2_id UUID REFERENCES political_entities(id),
    relationship_type TEXT,
    strength FLOAT,
    source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_legislators_party ON legislators(party);
CREATE INDEX IF NOT EXISTS idx_legislators_state ON legislators(state);
CREATE INDEX IF NOT EXISTS idx_bills_status ON bills(status);
CREATE INDEX IF NOT EXISTS idx_votes_legislator_bill ON votes(legislator_id, bill_id);
CREATE INDEX IF NOT EXISTS idx_speeches_legislator ON speeches(legislator_id);
CREATE INDEX IF NOT EXISTS idx_speeches_date ON speeches(date);

-- Vector similarity search indexes
CREATE INDEX IF NOT EXISTS idx_legislators_bio_embedding ON legislators USING ivfflat (bio_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_bills_content_embedding ON bills USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_speeches_content_embedding ON speeches USING ivfflat (content_embedding vector_cosine_ops);

-- Enable Row Level Security
ALTER TABLE legislators ENABLE ROW LEVEL SECURITY;
ALTER TABLE bills ENABLE ROW LEVEL SECURITY;
ALTER TABLE votes ENABLE ROW LEVEL SECURITY;
ALTER TABLE committees ENABLE ROW LEVEL SECURITY;
ALTER TABLE committee_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE speeches ENABLE ROW LEVEL SECURITY;
ALTER TABLE lobbying_activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE political_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE relationships ENABLE ROW LEVEL SECURITY;

-- Basic RLS policies (allow all for authenticated users)
CREATE POLICY "Allow all for authenticated users" ON legislators FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON bills FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON votes FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON committees FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON committee_memberships FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON speeches FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON lobbying_activities FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON political_entities FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow all for authenticated users" ON relationships FOR ALL TO authenticated USING (true);

-- Functions for similarity search
CREATE OR REPLACE FUNCTION find_similar_legislators(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    legislator legislators,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT 
        legislators.*,
        1 - (legislators.bio_embedding <=> query_embedding) AS similarity
    FROM legislators
    WHERE 1 - (legislators.bio_embedding <=> query_embedding) > match_threshold
    ORDER BY legislators.bio_embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION find_similar_bills(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.8,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    bill bills,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT 
        bills.*,
        1 - (bills.content_embedding <=> query_embedding) AS similarity
    FROM bills
    WHERE 1 - (bills.content_embedding <=> query_embedding) > match_threshold
    ORDER BY bills.content_embedding <=> query_embedding
    LIMIT match_count;
$$;