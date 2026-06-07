"""Cliente CourtListener REST v4 con guardia de cuota y caché persistente.

Política:
  1. Toda búsqueda intenta primero el caché (search_cache).
  2. Toda resolución por ID intenta primero el caché (cluster_cache).
  3. La llamada real sólo se hace si la cuota diaria lo permite.
  4. Cada respuesta se persiste para que la próxima sea gratis.
"""
import os
import time
import sys
import requests
import cache

BASE = "https://www.courtlistener.com/api/rest/v4"
TIMEOUT = 12

DAILY_QUOTA = int(os.environ.get("DAILY_QUOTA", "120"))

def _token() -> str:
    t = os.environ.get("COURTLISTENER_TOKEN", "").strip()
    if t: return t
    try:
        with open("/app/.token", "r") as f:
            return f.read().strip()
    except Exception:
        return ""

def quota_status() -> dict:
    used = cache.calls_today()
    remaining = max(0, DAILY_QUOTA - used)
    return {
        "used": used,
        "limit": DAILY_QUOTA,
        "remaining": remaining,
        "soft_block": used >= int(DAILY_QUOTA * 0.9),
        "hard_block": used >= DAILY_QUOTA,
        "cache_enabled": cache.enabled(),
    }

class QuotaExceeded(Exception):
    pass

class NoToken(Exception):
    pass

def _headers() -> dict:
    tok = _token()
    if not tok: raise NoToken("Falta COURTLISTENER_TOKEN.")
    return {"Authorization": f"Token {tok}", "User-Agent": "a3-lopez-perez/1.0"}

def _spend(endpoint: str, query: str, fn):
    if quota_status()["hard_block"]:
        raise QuotaExceeded(f"Cuota diaria agotada ({DAILY_QUOTA}).")
    t0 = time.time()
    status = 0; notes = ""
    try:
        r = fn()
        status = r.status_code
        return r
    except requests.RequestException as e:
        notes = str(e)[:200]
        raise
    finally:
        ms = int((time.time() - t0) * 1000)
        cache.log_call(endpoint, query, status, ms, notes)

def get_cluster(cluster_id: int) -> dict | None:
    """Resuelve un cluster por ID. Caché primero; API si falta."""
    cid = int(cluster_id)
    cached = cache.get_cluster(cid)
    if cached is not None:
        return cached
    def call():
        return requests.get(f"{BASE}/clusters/{cid}/", headers=_headers(), timeout=TIMEOUT)
    r = _spend(f"clusters/{cid}", str(cid), call)
    if r.status_code == 404: return None
    r.raise_for_status()
    payload = r.json()
    cache.put_cluster(cid, payload)
    return payload

def search(q: str, mode: str, limit: int = 5) -> list:
    """Búsqueda en CourtListener. Caché por (modo, query) primero."""
    cached = cache.get_search(q, mode)
    if cached is not None:
        return cached
    params = {"type": "o", "q": q}
    if mode == "judge":     params = {"type": "o", "judge": q}
    elif mode == "court":   params = {"type": "o", "court": q}
    elif mode == "docket":  params = {"type": "o", "docket_number": q}
    elif mode == "case_name": params = {"type": "o", "case_name": q}
    def call():
        return requests.get(f"{BASE}/search/", headers=_headers(), params=params, timeout=TIMEOUT)
    r = _spend("search", q, call)
    r.raise_for_status()
    js = r.json()
    results = (js.get("results") or [])[:limit]
    cache.put_search(q, mode, results)
    for rec in results:
        cid = rec.get("cluster_id")
        if cid is not None:
            cache.put_cluster(int(cid), rec)
    return results
