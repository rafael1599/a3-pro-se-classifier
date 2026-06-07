-- A3 — Esquema de caché para CourtListener.
-- Aplicar contra Supabase Postgres una sola vez.

create table if not exists cluster_cache (
    cluster_id   bigint primary key,
    payload      jsonb       not null,
    fetched_at   timestamptz not null default now(),
    last_seen_at timestamptz not null default now()
);

create index if not exists cluster_cache_fetched_idx on cluster_cache (fetched_at desc);

create table if not exists search_cache (
    query_hash text        primary key,
    query      text        not null,
    mode       text        not null,
    results    jsonb       not null,
    fetched_at timestamptz not null default now()
);

create index if not exists search_cache_fetched_idx on search_cache (fetched_at desc);

create table if not exists api_calls_log (
    id         bigserial primary key,
    called_at  timestamptz not null default now(),
    endpoint   text        not null,
    query      text,
    status     int,
    ms         int,
    notes      text
);

create index if not exists api_calls_log_day_idx on api_calls_log (called_at desc);

-- Vista de consumo diario para el endpoint /quota.
create or replace view api_calls_today as
select count(*)::int as calls_today
  from api_calls_log
 where called_at >= date_trunc('day', now() at time zone 'UTC');
