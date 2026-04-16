CREATE TABLE IF NOT EXISTS seed_documents (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    ground_truth_direction TEXT NOT NULL,
    full_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id TEXT PRIMARY KEY,
    seed_id TEXT NOT NULL REFERENCES seed_documents(id),
    topology TEXT NOT NULL,
    condition TEXT NOT NULL,
    trial_id INTEGER NOT NULL,
    rerun_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finalized_at TIMESTAMPTZ,
    final_decision TEXT,
    final_confidence DOUBLE PRECISION,
    correct BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_experiment_runs_seed_id
    ON experiment_runs(seed_id);

CREATE INDEX IF NOT EXISTS idx_experiment_runs_condition
    ON experiment_runs(condition);

CREATE TABLE IF NOT EXISTS agent_messages (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    message_type TEXT NOT NULL,
    content_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_run_id
    ON agent_messages(run_id);

CREATE INDEX IF NOT EXISTS idx_agent_messages_agent_name
    ON agent_messages(agent_name);

CREATE TABLE IF NOT EXISTS evidence_documents (
    id TEXT PRIMARY KEY,
    seed_id TEXT,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    entity TEXT,
    ticker TEXT,
    document_date DATE,
    title TEXT NOT NULL,
    text_content TEXT NOT NULL,
    full_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidence_documents_seed_id
    ON evidence_documents(seed_id);

CREATE INDEX IF NOT EXISTS idx_evidence_documents_source_type
    ON evidence_documents(source_type);

CREATE INDEX IF NOT EXISTS idx_evidence_documents_entity
    ON evidence_documents(entity);

CREATE TABLE IF NOT EXISTS agent_retrievals (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    query TEXT NOT NULL,
    result_ids JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_retrievals_run_id
    ON agent_retrievals(run_id);
