#!/bin/bash
set -e

# Initialize PostgreSQL with required extensions and configurations
echo "Initializing PostgreSQL database for Political Analysis System..."

# Create pgvector extension
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Create additional roles for Supabase
    CREATE ROLE anon NOINHERIT;
    CREATE ROLE authenticated NOINHERIT;
    CREATE ROLE service_role NOINHERIT BYPASSRLS;
    
    -- Grant permissions
    GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
    GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated, service_role;
    GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated, service_role;
    GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated, service_role;
    
    -- Enable Row Level Security by default
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon, authenticated, service_role;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO anon, authenticated, service_role;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO anon, authenticated, service_role;
    
    -- Create a schema for Supabase Auth
    CREATE SCHEMA IF NOT EXISTS auth;
    GRANT USAGE ON SCHEMA auth TO anon, authenticated, service_role;
    
    -- Create a schema for Supabase Storage
    CREATE SCHEMA IF NOT EXISTS storage;
    GRANT USAGE ON SCHEMA storage TO anon, authenticated, service_role;
    
    -- Create a schema for Supabase Realtime
    CREATE SCHEMA IF NOT EXISTS _realtime;
    GRANT USAGE ON SCHEMA _realtime TO anon, authenticated, service_role;
    
    -- Set up basic RLS policies (will be extended by the application)
    CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid AS \$\$
        SELECT current_setting('request.jwt.claim.sub', true)::uuid;
    \$\$ LANGUAGE sql STABLE;
    
    -- Create basic audit functions
    CREATE OR REPLACE FUNCTION public.handle_new_user()
    RETURNS trigger AS \$\$
    BEGIN
        INSERT INTO public.profiles (id, email)
        VALUES (new.id, new.email);
        RETURN new;
    END;
    \$\$ LANGUAGE plpgsql SECURITY DEFINER;
    
    -- Create profiles table for user management
    CREATE TABLE IF NOT EXISTS public.profiles (
        id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
        email TEXT,
        full_name TEXT,
        avatar_url TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Enable RLS on profiles
    ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
    
    -- Create policy for profiles
    CREATE POLICY "Users can view own profile" ON public.profiles
        FOR SELECT USING (auth.uid() = id);
    
    CREATE POLICY "Users can update own profile" ON public.profiles
        FOR UPDATE USING (auth.uid() = id);
    
    -- Create indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_profiles_id ON public.profiles(id);
    CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);
    
EOSQL

echo "PostgreSQL initialization complete!"