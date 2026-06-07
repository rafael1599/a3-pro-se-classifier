"""Caché Postgres (Supabase) para casos y búsquedas de CourtListener.

Si DATABASE_URL no está seteado, el módulo opera en modo no-op: get_* devuelve
None y put_* es un alias del log a stderr. Esto permite que la app arranque sin
la base configurada (modo demo local).
"""
import json
import os
import sys
import hashlib
import time
from contextlib import contextmanager

try:
    import psycopg
    from psycopg.types.json import Jsonb
except Exception:
    psycopg = None
    Jsonb = None

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
_ENABLED = bool(DATABASE_URL) and psycopg is not None

def enabled() -> bool:
    return _ENABLED

@contextmanager
def _conn():
    if not _ENABLED:
        yield None
        return
    c = psycopg.connect(DATABASE_URL, connect_timeout=5)
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()

def query_hash(q: str, mode: str) -> str:
    h = hashlib.sha1(f"{mode}|{q.strip().lower()}".encode("utf-8")).hexdigest()
    return h[:32]

def get_cluster(cluster_id: int):
    if not _ENABLED: return None
    try:
        with _conn() as c:
            row = c.execute(
                "select payload from cluster_cache where cluster_id = %s",
                (cluster_id,),
            ).fetchone()
            if not row: return None
            c.execute(
                "update cluster_cache set last_seen_at = now() where cluster_id = %s",
                (cluster_id,),
            )
            return row[0]
    except Exception as e:
        print(f"[cache] get_cluster error: {e}", file=sys.stderr, flush=True)
        return None

def put_cluster(cluster_id: int, payload: dict):
    if not _ENABLED: return
    try:
        with _conn() as c:
            c.execute(
                """insert into cluster_cache (cluster_id, payload)
                       values (%s, %s)
                   on conflict (cluster_id) do update
                       set payload = excluded.payload,
                           last_seen_at = now()""",
                (cluster_id, Jsonb(payload)),
            )
    except Exception as e:
        print(f"[cache] put_cluster error: {e}", file=sys.stderr, flush=True)

def get_search(q: str, mode: str):
    if not _ENABLED: return None
    try:
        with _conn() as c:
            row = c.execute(
                "select results from search_cache where query_hash = %s",
                (query_hash(q, mode),),
            ).fetchone()
            return row[0] if row else None
    except Exception as e:
        print(f"[cache] get_search error: {e}", file=sys.stderr, flush=True)
        return None

def put_search(q: str, mode: str, results: list):
    if not _ENABLED: return
    try:
        with _conn() as c:
            c.execute(
                """insert into search_cache (query_hash, query, mode, results)
                       values (%s, %s, %s, %s)
                   on conflict (query_hash) do update
                       set results = excluded.results,
                           fetched_at = now()""",
                (query_hash(q, mode), q, mode, Jsonb(results)),
            )
    except Exception as e:
        print(f"[cache] put_search error: {e}", file=sys.stderr, flush=True)

def log_call(endpoint: str, query: str, status: int, ms: int, notes: str = ""):
    if not _ENABLED: return
    try:
        with _conn() as c:
            c.execute(
                """insert into api_calls_log (endpoint, query, status, ms, notes)
                       values (%s, %s, %s, %s, %s)""",
                (endpoint, query, status, ms, notes),
            )
    except Exception as e:
        print(f"[cache] log_call error: {e}", file=sys.stderr, flush=True)

def calls_today() -> int:
    if not _ENABLED: return 0
    try:
        with _conn() as c:
            row = c.execute("select calls_today from api_calls_today").fetchone()
            return int(row[0]) if row else 0
    except Exception as e:
        print(f"[cache] calls_today error: {e}", file=sys.stderr, flush=True)
        return 0

def ping() -> bool:
    if not _ENABLED: return False
    try:
        with _conn() as c:
            c.execute("select 1").fetchone()
            return True
    except Exception as e:
        print(f"[cache] ping error: {e}", file=sys.stderr, flush=True)
        return False
