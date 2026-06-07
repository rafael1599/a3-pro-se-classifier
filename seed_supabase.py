"""Carga inicial: vuelca los casos etiquetados a cluster_cache en Supabase.

Lee los tres JSON (ifp, counsel, extension), agrega la etiqueta como
metadato dentro del payload (`__label__`) y hace upsert por cluster_id.
Idempotente: correr de nuevo sólo refresca payload + last_seen_at.

Uso:
    DATABASE_URL=postgresql://... python seed_supabase.py
"""
import json
import os
import sys
import time

import psycopg
from psycopg.types.json import Jsonb

FILES = [
    ("full_ifp.json", "ifp"),
    ("full_counsel.json", "counsel"),
    ("full_extension.json", "extension"),
]

UPSERT = """
insert into cluster_cache (cluster_id, payload)
     values (%s, %s)
on conflict (cluster_id) do update
     set payload = excluded.payload,
         last_seen_at = now()
"""

def main() -> int:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        print("DATABASE_URL no seteado.", file=sys.stderr)
        return 2

    rows = []
    seen = set()
    for fname, label in FILES:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            cid = rec.get("cluster_id")
            if cid is None:
                continue
            cid = int(cid)
            payload = dict(rec)
            payload["__label__"] = label
            if cid in seen:
                continue
            seen.add(cid)
            rows.append((cid, Jsonb(payload)))
        print(f"{fname}: {len(data)} registros, etiqueta {label}")

    print(f"Únicos a cargar: {len(rows)}")
    t0 = time.time()
    with psycopg.connect(url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            for i in range(0, len(rows), 200):
                cur.executemany(UPSERT, rows[i:i+200])
                print(f"  insertados {min(i+200, len(rows))}/{len(rows)}")
        conn.commit()
        with conn.cursor() as cur:
            total = cur.execute("select count(*) from cluster_cache").fetchone()[0]
    print(f"OK. {total} filas en cluster_cache. Tiempo: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
